"""Validate leapfit's BKT against pyBKT and hmm-scalable, the two implementations the literature runs.

**Two references, because one cannot tell BKT from itself.** pyBKT (Badrinath,
Wang & Pardos 2021) is the Python package papers cite; ``hmm-scalable``
(Yudelson) is the C++ tool behind LearnSphere's ``AnalysisBkt`` component, the
one DataShop workflows run. Both fit by Baum-Welch, but they differ in
initialization, in default bounds (``hmm-scalable`` caps guess and slip at 0.3
unless told otherwise) and in their stopping rules, so agreeing with both is
agreement with *BKT* rather than with one implementation's habits.

**Why pyBKT runs in its own process.** pyBKT 1.4 breaks on import against
``scikit-learn >= 1.6`` (its metric probing calls ``log_loss`` with lists and
catches only ``TypeError``), its pure-Python fit was broken on NumPy 2 until a
fix merged upstream in October 2026, and it drags in a compiled extension
leapfit has no business depending on. So it runs in its own environment and
the two halves communicate through files::

    uv venv .venv-pybkt --python 3.11
    uv pip install --python .venv-pybkt/bin/python pyBKT "scikit-learn==1.5.2" "numpy<2"

**Why hmm-scalable is a binary.** It is C++; build it once and point at it::

    git clone https://github.com/myudelson/hmm-scalable
    cd hmm-scalable && make CFLAGS="-O3 -fPIC -w" train predict   # clang: no -fopenmp

Then::

    uv run python scripts/validate_bkt.py examples/student-step.txt \\
        --kc-model Topics --pybkt-python .venv-pybkt/bin/python \\
        --trainhmm ../hmm-scalable/trainhmm

**What is validated, and what is not.** Two checks per reference, in the order
that makes the second interpretable.

1. **Transplantation** — the decisive one. Take the reference's *fitted
   parameters*, load them into leapfit's recursion, and compare predictions
   row by row. If those agree, the two implementations are the same model, and
   any difference in performance is attributable to the fitting alone.
   Comparing two *fits* without this first would confound the recursion with
   the optimizer, which is why the protocol this is adapted from
   (``betabern``'s ``validate_bkt.py``) leads with it.

2. **Recovery** — having established the model is the same, fit leapfit's own
   EM and compare parameters and predictions against the reference's. All
   three fit by EM on the same likelihood, so the parameters themselves should
   agree where the surface has one basin -- and where they do not, the
   difference is evidence about the surface, not about either implementation.
   leapfit also holds every KC to ``G + S < 1`` and neither reference does, so
   a KC the reference fits label-swapped is expected to differ.

**What counts as agreement, and why it is not 1e-15.** pyBKT's one-step
predictions come from the forward messages of its E-step, which accumulate at
single-precision scale; ``hmm-scalable`` prints ten decimals. The two therefore
agree with leapfit as models but not bit-for-bit, and the honest bar is the
scale of each reference's own arithmetic plus the thing a benchmark actually
rests on: that the summary statistics coincide. Only the transplantation
checks gate the exit status.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARAM_NAMES = ("prior", "learn", "guess", "slip", "forget")


# ----------------------------------------------------------------------------------
# Mode 1: runs under the pyBKT interpreter. Imports pyBKT only; must not import leapfit.
# ----------------------------------------------------------------------------------

def emit_pybkt(frame_path: str, out_stem: str, seed: int, num_fits: int) -> None:
    """Fit pyBKT on the prepared frame and write its parameters and per-row predictions."""
    from pyBKT.models import Model

    frame = pd.read_csv(frame_path)
    defaults = {"user_id": "user_id", "skill_name": "skill_name", "correct": "correct"}

    # parallel=False deliberately: pyBKT's multiprocessing path thrashes on a
    # small corpus (minutes on a 480-row export that takes it well under a
    # second serially), and a validation run wants determinism over throughput.
    model = Model(seed=seed, num_fits=num_fits, parallel=False)
    model.fit(data=frame, defaults=defaults)
    predictions = model.predict(data=frame)  # reuses the column mapping given to fit()

    # Read parameters off ``coef_`` rather than ``params()``: the latter is a display
    # helper returning a MultiIndex frame of formatted values, and reshaping it back is
    # needless work.
    rows = []
    for skill, coefficients in model.coef_.items():
        for name, value in coefficients.items():
            rows.append({"skill": skill, "param": name,
                         "value": float(np.ravel(value)[0]) if np.ndim(value) else float(value)})
    pd.DataFrame(rows).to_csv(out_stem + ".params.csv", index=False)
    predictions[["row", "correct_predictions"]].to_csv(out_stem + ".preds.csv", index=False)
    print(f"pyBKT: {len(rows)} parameter rows, {len(predictions)} predictions "
          f"-> {out_stem}.*", file=sys.stderr)


# ----------------------------------------------------------------------------------
# Mode 2: runs under the leapfit interpreter.
# ----------------------------------------------------------------------------------

def prepare_frame(export: Path, kc_model: str, multi_kc: str):
    """The export as sequences, plus the same (row, KC) pairs in the order both
    references read them: each (student, KC) run contiguous, in practice order.

    Every half is keyed by ``row`` -- the position in ``StepData`` -- so the
    predictions can be joined back without anyone re-sorting anything. Under
    ``multi_kc="replicate"`` a row recurs once per KC, which is exactly how
    LearnSphere feeds ``hmm-scalable`` and how pyBKT expects multi-skill data.
    """
    from leapfit import load_student_step
    from leapfit.bkt import extract_sequences

    data = load_student_step(str(export), kc_model=kc_model)
    sequences = extract_sequences(data, multi_kc=multi_kc)

    live = sequences.mask
    seq = np.repeat(np.arange(len(sequences)), live.sum(axis=1))
    order = sequences.rows[live]
    frame = pd.DataFrame({
        "row": order,
        "user_id": [sequences.students[s] for s in seq],
        "skill_name": [sequences.kc_names[sequences.kc_of_seq[s]] for s in seq],
        "correct": sequences.y[live].astype(int),
        "item": [f"{data.items[i]}" for i in order],
    })
    return data, sequences, frame


def pybkt_params_by_kc(params: pd.DataFrame, kc_names: list[str]) -> dict[str, np.ndarray]:
    """Reshape pyBKT's ``coef_`` table into one array per BKT probability, by KC.

    pyBKT names them ``learns``/``forgets``/``guesses``/``slips`` (arrays, one entry per
    class -- the default fit has a single class) plus a scalar ``prior``. A skill whose
    fit produced no ``forgets`` entry has no forgetting, which is 0 rather than missing.
    """
    lookup = {"prior": "prior", "learns": "learn", "forgets": "forget",
              "guesses": "guess", "slips": "slip"}
    table = {name: np.full(len(kc_names), np.nan) for name in PARAM_NAMES}
    index = {kc: i for i, kc in enumerate(kc_names)}
    for _, row in params.iterrows():
        name = lookup.get(str(row["param"]))
        skill = str(row["skill"])
        if name is not None and skill in index:
            table[name][index[skill]] = float(row["value"])
    table["forget"] = np.nan_to_num(table["forget"], nan=0.0)
    return table


def run_trainhmm(trainhmm: str, frame: pd.DataFrame, stem: str, max_iter: int) -> dict:
    """Fit ``hmm-scalable`` on the frame and return its parameters and predictions.

    Its input is four tab-separated columns -- observation (1 correct, 2
    incorrect), student, item, skill -- which is the file LearnSphere's
    component writes. The flags: Baum-Welch by skill (``-s 1.1``) with scaled
    forward/backward variables, a log-likelihood stopping rule, the textbook
    starting point, no cap on guess or slip (its default caps both at 0.3,
    which would hide the label swap the recovery check is there to see), and
    predictions on the training rows with the observation revealed
    (``-U r,t``, its default) -- the same one-step-ahead prediction leapfit's
    filter makes.
    """
    data_path, model_path, pred_path = stem + ".txt", stem + ".model.txt", stem + ".pred.txt"
    pd.DataFrame({"obs": np.where(frame["correct"] == 1, 1, 2), "student": frame["user_id"],
                  "item": frame["item"], "skill": frame["skill_name"]}).to_csv(
        data_path, sep="\t", header=False, index=False)
    command = [trainhmm, "-s", "1.1", "-S", "1", "-e", "0.0000001,l", "-i", str(max_iter),
               "-0", "0.3,1.0,0.1,0.9,0.2",          # L0, 1-F, T, 1-S, G: the textbook start
               "-u", "1,1,1,1,1,1,1,1,1,1",          # no 0.3 cap on slip and guess
               "-m", "1", "-p", "1", data_path, model_path, pred_path]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    summary = next((line for line in result.stdout.splitlines() if "trained model" in line), "")
    print(f"hmm-scalable: {summary.strip()}", file=sys.stderr)

    # The model file: a seven-line header, then four lines per skill. State 0 is
    # "known": PI = (L0, 1 - L0); A = (1-F, F, T, 1-T); B = (1-S, S, G, 1-G).
    lines = Path(model_path).read_text().splitlines()
    params: dict[str, dict[str, float]] = {}
    i = 7
    while i + 3 < len(lines):
        name = lines[i].split("\t", 1)[1]
        pi = [float(v) for v in lines[i + 1].split("\t")[1:]]
        a = [float(v) for v in lines[i + 2].split("\t")[1:]]
        b = [float(v) for v in lines[i + 3].split("\t")[1:]]
        params[name] = {"prior": pi[0], "forget": a[1], "learn": a[2], "slip": b[1], "guess": b[2]}
        i += 4
    predictions = pd.read_csv(pred_path, sep="\t", header=None)[0].to_numpy()
    return {"params": params, "predictions": predictions, "summary": summary}


def hmm_params_by_kc(params: dict[str, dict[str, float]], kc_names: list[str]) -> dict:
    table = {name: np.full(len(kc_names), np.nan) for name in PARAM_NAMES}
    for k, kc in enumerate(kc_names):
        for name in PARAM_NAMES:
            if kc in params:
                table[name][k] = params[kc][name]
    return table


def transplant(sequences, params: dict[str, np.ndarray], kc_names: list[str],
               label: str) -> np.ndarray:
    """Load a reference's parameters into leapfit's BKT and return its per-row P(correct)."""
    from leapfit.bkt import BKTFit

    # The reference's parameters exactly as it printed them, corners included:
    # hmm-scalable fits a guess or a prior of 0.0 outright, and the recursion
    # guards its own divisions, so nothing is clipped on the way in.
    n = len(kc_names)
    fit = BKTFit(
        kc_names=list(kc_names),
        prior=np.clip(params["prior"], 0.0, 1.0),
        learn=np.clip(params["learn"], 0.0, 1.0),
        guess=np.clip(params["guess"], 0.0, 1.0),
        slip=np.clip(params["slip"], 0.0, 1.0),
        forget=np.clip(params["forget"], 0.0, 1.0),
        ll=float("nan"), n_obs=sequences.n_obs, n_rows=sequences.n_rows, n_seq=len(sequences),
        n_seq_per_kc=np.ones(n, dtype=int), n_transitions=np.ones(n, dtype=int),
        converged=np.ones(n, dtype=bool), monotone=np.ones(n, dtype=bool),
        n_iter=np.zeros(n, dtype=int), final_improvement=np.zeros(n),
        restart_lls=np.zeros((n, 1)), bound_active=np.zeros(n, dtype=bool), has_forget=True,
        kc_model=sequences.kc_model, multi_kc=sequences.multi_kc,
        label=f"{label} parameters in leapfit's recursion",
    )
    return fit.predict_proba(sequences)


def per_pair(sequences, per_row: np.ndarray) -> np.ndarray:
    """Per-row predictions expanded to the (row, KC) pairs of the frame, in frame order."""
    return per_row[sequences.rows[sequences.mask]]


def compare(y: np.ndarray, ours: np.ndarray, theirs: np.ndarray) -> dict:
    valid = np.isfinite(ours) & np.isfinite(theirs)
    ours, theirs, y = ours[valid], theirs[valid], y[valid]
    # AUC is a rank statistic, so it moves with every tie: hmm-scalable prints
    # ten decimals, and two rows this recursion separates at the eleventh tie
    # in its output. Both sides are compared at the precision the references
    # actually print, which is where "the same ranking" is a meaningful claim.
    ranked_ours, ranked_theirs = np.round(ours, 8), np.round(theirs, 8)

    def log_loss(p):
        return float(-np.mean(np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-15, 1.0))))

    def auc(p):  # rank-based; no sklearn dependency for one number
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p), dtype=float)
        ranks[order] = np.arange(1, len(p) + 1)
        sorted_p = p[order]
        start = 0
        for end in range(1, len(p) + 1):   # average ranks within ties
            if end == len(p) or sorted_p[end] != sorted_p[start]:
                ranks[order[start:end]] = (start + end + 1) / 2.0
                start = end
        positives, negatives = (y == 1).sum(), (y == 0).sum()
        if positives == 0 or negatives == 0:
            return float("nan")
        return float((ranks[y == 1].sum() - positives * (positives + 1) / 2)
                     / (positives * negatives))

    return {
        "n": int(valid.sum()),
        "max_abs_diff": float(np.max(np.abs(ours - theirs))),
        "mean_abs_diff": float(np.mean(np.abs(ours - theirs))),
        "correlation": float(np.corrcoef(ours, theirs)[0, 1]),
        "auc_ours": auc(ranked_ours), "auc_theirs": auc(ranked_theirs),
        "nll_ours": log_loss(ours), "nll_theirs": log_loss(theirs),
    }


def agrees(check: dict) -> bool:
    return (check["mean_abs_diff"] < 1e-5
            and check["correlation"] > 1 - 1e-9
            and abs(check["auc_ours"] - check["auc_theirs"]) < 1e-5
            and abs(check["nll_ours"] - check["nll_theirs"]) < 1e-5)


def validate_against(label: str, sequences, y_pairs: np.ndarray, params: dict,
                     theirs_pairs: np.ndarray, ours_fit) -> dict:
    """The two checks against one reference, keyed by (row, KC) pair."""
    kc_names = sequences.kc_names
    same_model = compare(y_pairs, per_pair(sequences, transplant(sequences, params, kc_names,
                                                                 label)), theirs_pairs)
    same_fit = compare(y_pairs, per_pair(sequences, ours_fit.predict_proba(sequences)),
                       theirs_pairs)
    parameter_gap = {
        name: float(np.nanmax(np.abs(getattr(ours_fit, name) - params[name])))
        for name in PARAM_NAMES
    }
    swapped = [kc for k, kc in enumerate(kc_names)
               if np.isfinite(params["guess"][k]) and params["guess"][k] + params["slip"][k] >= 1]
    verdict = agrees(same_model)
    print(f"\n{'AGREE' if verdict else 'DISAGREE'} with {label}: transplanting its parameters "
          f"into leapfit's recursion reproduces its predictions — mean |diff| "
          f"{same_model['mean_abs_diff']:.2e}, max {same_model['max_abs_diff']:.2e}, "
          f"correlation {same_model['correlation']:.12f}", file=sys.stderr)
    print(f"       leapfit's own EM: NLL {same_fit['nll_ours']:.6f} vs {label} "
          f"{same_fit['nll_theirs']:.6f}; largest parameter gap "
          f"{max(parameter_gap.values()):.3e} (reported, not asserted — EM is multimodal"
          + (f"; {label} fits {len(swapped)} KC(s) with G + S >= 1, which leapfit excludes"
             if swapped else "") + ")", file=sys.stderr)
    return {"transplantation": same_model, "own_fit": same_fit,
            "max_parameter_gap": parameter_gap, "label_swapped_kcs": swapped,
            "agrees": bool(verdict)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", nargs="?", default=str(ROOT / "examples/student-step.txt"),
                        help="DataShop student-step export")
    parser.add_argument("--kc-model", default="Topics", help="KC model to fit")
    parser.add_argument("--multi-kc", default="refuse", choices=("refuse", "replicate"))
    parser.add_argument("--pybkt-python", default=None,
                        help="interpreter of the isolated pyBKT environment "
                             "(see the module docstring for how to create it)")
    parser.add_argument("--trainhmm", default=None,
                        help="path to hmm-scalable's trainhmm binary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-fits", type=int, default=5,
                        help="restarts, for leapfit and pyBKT (hmm-scalable has none)")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--out", default=None, help="where to write the JSON report")
    parser.add_argument("--emit-pybkt", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frame", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.emit_pybkt:  # running under the pyBKT interpreter
        emit_pybkt(args.frame, args.emit_pybkt, args.seed, args.num_fits)
        return 0
    if not (args.pybkt_python or args.trainhmm):
        parser.error("at least one reference is needed: --pybkt-python and/or --trainhmm")

    work = Path(args.out).parent if args.out else ROOT / "results" / "bkt-validation"
    work.mkdir(parents=True, exist_ok=True)

    _, sequences, frame = prepare_frame(Path(args.export), args.kc_model, args.multi_kc)
    frame_path = work / "bkt_frame.csv"
    frame.to_csv(frame_path, index=False)
    y_pairs = frame["correct"].to_numpy()
    kc_names = sequences.kc_names

    from leapfit.bkt import fit_bkt
    ours = fit_bkt(sequences, forget=False, n_restarts=args.num_fits, seed=args.seed,
                   max_iter=args.max_iter, warn_at_bound=False, warn_not_converged=False)
    print(ours.summary(), file=sys.stderr)

    report = {"export": args.export, "kc_model": args.kc_model, "multi_kc": args.multi_kc,
              "n_kcs": len(kc_names), "n_pairs": len(frame), "n_rows": sequences.n_rows,
              "leapfit": {"ll": ours.ll, "n_params": ours.n_params, "aic": ours.aic,
                          "bic": ours.bic, "at_bound": ours.at_bound}}

    if args.pybkt_python:
        stem = str(work / "pybkt")
        subprocess.run([args.pybkt_python, str(Path(__file__).resolve()),
                        "--emit-pybkt", stem, "--frame", str(frame_path),
                        "--seed", str(args.seed), "--num-fits", str(args.num_fits)],
                       check=True, cwd=ROOT)
        params = pybkt_params_by_kc(pd.read_csv(stem + ".params.csv"), kc_names)
        theirs = (pd.read_csv(stem + ".preds.csv")["correct_predictions"].to_numpy())
        report["pybkt"] = validate_against("pyBKT", sequences, y_pairs, params, theirs, ours)

    if args.trainhmm:
        reference = run_trainhmm(args.trainhmm, frame, str(work / "hmm"), args.max_iter)
        params = hmm_params_by_kc(reference["params"], kc_names)
        report["hmm_scalable"] = validate_against("hmm-scalable", sequences, y_pairs, params,
                                                  reference["predictions"], ours)
        report["hmm_scalable"]["summary"] = reference["summary"]

    verdict = all(report[key]["agrees"] for key in ("pybkt", "hmm_scalable") if key in report)
    report["agrees"] = bool(verdict)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
