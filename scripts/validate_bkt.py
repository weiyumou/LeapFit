"""Validate leapfit's BKT against pyBKT, the implementation the literature cites.

**Why this is a two-process script.** pyBKT 1.4 breaks on import against
``scikit-learn >= 1.6`` (its metric probing calls ``log_loss`` with lists), and
it drags in a compiled extension leapfit has no business depending on. So pyBKT
runs in its own environment and the two halves communicate through files::

    uv venv .venv-pybkt
    uv pip install --python .venv-pybkt/bin/python pyBKT "scikit-learn==1.5.2"
    uv run python scripts/validate_bkt.py examples/student-step.txt \\
        --kc-model Topics --pybkt-python .venv-pybkt/bin/python

**What is validated, and what is not.** Two checks, in the order that makes the
second interpretable.

1. **Transplantation** — the decisive one. Take pyBKT's *fitted parameters*,
   load them into leapfit's recursion, and compare predictions row by row. If
   those agree, the two implementations are the same model, and any difference
   in performance is attributable to the fitting alone. Comparing two *fits*
   without this first would confound the recursion with the optimizer, which is
   why the reference protocol this is adapted from (``betabern``'s
   ``validate_bkt.py``) leads with it.

2. **Recovery** — having established the model is the same, fit leapfit's own
   EM and compare parameters and predictions against pyBKT's. Both fit by EM on
   the same likelihood, so unlike the gradient-descent implementation this is
   adapted from, this comparison is meaningful: the parameters themselves
   should agree, not merely the predictions they imply.

**What counts as agreement, and why it is not 1e-15.** pyBKT does not run the
filtering recursion this does. Its one-step predictions come from the forward
messages of its compiled E-step (``pyBKT/fit/predict_onestep.py`` reuses the EM
machinery), which accumulates at single-precision scale. The two therefore
agree as models but not bit-for-bit, and the honest bar is the scale of pyBKT's
own arithmetic plus the thing a benchmark actually rests on: that the summary
statistics coincide.

EM is also multimodal, and pyBKT's ``num_fits`` restarts and leapfit's
``n_restarts`` do not explore the surface identically. The recovery check is
therefore reported, not asserted — a parameter disagreement there is evidence
about the likelihood surface, not necessarily about either implementation. Only
the transplantation check gates the exit status.
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

    # parallel=False deliberately: pyBKT's OpenMP path thrashes on a small
    # corpus (minutes on a 480-row export that takes it well under a second
    # serially), and a validation run wants determinism over throughput anyway.
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

def prepare_frame(export: Path, kc_model: str):
    """The export as sequences, plus the same rows in the column names pyBKT wants.

    Both halves are keyed by ``row`` -- the position in ``StepData`` -- so the two
    sets of predictions can be joined without either side re-sorting anything.
    """
    from leapfit import load_student_step
    from leapfit.bkt import extract_sequences

    data = load_student_step(str(export), kc_model=kc_model)
    sequences = extract_sequences(data)

    # pyBKT treats each (user, skill) as one sequence in the order rows appear, so the
    # frame must be emitted in leapfit's canonical practice order, not the file's.
    order = sequences.rows[sequences.mask]
    frame = pd.DataFrame({
        "row": order,
        "user_id": [data.students[i] for i in order],
        "skill_name": [data.kcs[i][0] for i in order],
        "correct": data.y[order].astype(int),
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


def transplant(sequences, params: dict[str, np.ndarray], kc_names: list[str]) -> np.ndarray:
    """Load pyBKT's parameters into leapfit's BKT and return its per-row P(correct)."""
    from leapfit.bkt import BKTFit

    n = len(kc_names)
    fit = BKTFit(
        kc_names=list(kc_names),
        prior=np.clip(params["prior"], 1e-6, 1 - 1e-6),
        learn=np.clip(params["learn"], 1e-6, 1 - 1e-6),
        guess=np.clip(params["guess"], 1e-6, 1 - 1e-6),
        slip=np.clip(params["slip"], 1e-6, 1 - 1e-6),
        forget=np.clip(params["forget"], 0.0, 1 - 1e-6),
        ll=float("nan"), n_obs=sequences.n_obs, n_seq=len(sequences),
        n_seq_per_kc=np.ones(n, dtype=int), n_transitions=np.ones(n, dtype=int),
        converged=np.ones(n, dtype=bool), monotone=np.ones(n, dtype=bool),
        n_iter=np.zeros(n, dtype=int), final_improvement=np.zeros(n),
        restart_lls=np.zeros((n, 1)), has_forget=True, bound_degenerate=False,
        kc_model=sequences.kc_model, label="pyBKT parameters in leapfit's recursion",
    )
    return fit.predict_proba(sequences)


def compare(y: np.ndarray, ours: np.ndarray, theirs: np.ndarray) -> dict:
    valid = np.isfinite(ours) & np.isfinite(theirs)
    ours, theirs, y = ours[valid], theirs[valid], y[valid]

    def log_loss(p):
        return float(-np.mean(np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-15, 1.0))))

    def auc(p):  # rank-based; no sklearn dependency for one number
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p), dtype=float)
        ranks[order] = np.arange(1, len(p) + 1)
        # average ranks within ties, so a constant predictor scores 0.5
        sorted_p = p[order]
        start = 0
        for end in range(1, len(p) + 1):
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
        "auc_ours": auc(ours), "auc_pybkt": auc(theirs),
        "nll_ours": log_loss(ours), "nll_pybkt": log_loss(theirs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", nargs="?", default=str(ROOT / "examples/student-step.txt"),
                        help="DataShop student-step export")
    parser.add_argument("--kc-model", default="Topics", help="KC model to fit (single-KC only)")
    parser.add_argument("--pybkt-python", default=None,
                        help="interpreter of the isolated pyBKT environment "
                             "(see the module docstring for how to create it)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-fits", type=int, default=5,
                        help="restarts, for both sides")
    parser.add_argument("--out", default=None, help="where to write the JSON report")
    parser.add_argument("--emit-pybkt", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frame", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.emit_pybkt:  # running under the pyBKT interpreter
        emit_pybkt(args.frame, args.emit_pybkt, args.seed, args.num_fits)
        return 0
    if not args.pybkt_python:
        parser.error("--pybkt-python is required (see the module docstring)")

    work = Path(args.out).parent if args.out else ROOT / "results"
    work.mkdir(parents=True, exist_ok=True)

    data, sequences, frame = prepare_frame(Path(args.export), args.kc_model)
    frame_path = work / "bkt_frame.csv"
    frame.to_csv(frame_path, index=False)

    stem = str(work / "pybkt")
    subprocess.run([args.pybkt_python, str(Path(__file__).resolve()),
                    "--emit-pybkt", stem, "--frame", str(frame_path),
                    "--seed", str(args.seed), "--num-fits", str(args.num_fits)],
                   check=True, cwd=ROOT)

    kc_names = sequences.kc_names
    pybkt_params = pybkt_params_by_kc(pd.read_csv(stem + ".params.csv"), kc_names)
    theirs = (pd.read_csv(stem + ".preds.csv").set_index("row")
              .reindex(np.arange(len(data)))["correct_predictions"].to_numpy())

    # ---- Check 1: the model is the same (decisive) ----
    transplanted = transplant(sequences, pybkt_params, kc_names)
    same_model = compare(data.y, transplanted, theirs)

    # ---- Check 2: the fit recovers the same parameters (reported) ----
    from leapfit.bkt import fit_bkt
    ours = fit_bkt(sequences, forget=True, n_restarts=args.num_fits, seed=args.seed,
                   warn_degenerate=False)
    same_fit = compare(data.y, ours.predict_proba(sequences), theirs)
    parameter_gap = {
        name: float(np.nanmax(np.abs(getattr(ours, name) - pybkt_params[name])))
        for name in PARAM_NAMES
    }

    agrees = (same_model["mean_abs_diff"] < 1e-5
              and same_model["correlation"] > 1 - 1e-9
              and abs(same_model["auc_ours"] - same_model["auc_pybkt"]) < 1e-5
              and abs(same_model["nll_ours"] - same_model["nll_pybkt"]) < 1e-5)

    report = {"export": args.export, "kc_model": args.kc_model, "n_kcs": len(kc_names),
              "transplantation": same_model, "own_fit": same_fit,
              "max_parameter_gap": parameter_gap, "agrees": bool(agrees)}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"\n{'AGREE' if agrees else 'DISAGREE'}: transplanting pyBKT's parameters into "
          f"leapfit's recursion reproduces its predictions — mean |diff| "
          f"{same_model['mean_abs_diff']:.2e}, max {same_model['max_abs_diff']:.2e}, "
          f"correlation {same_model['correlation']:.12f}", file=sys.stderr)
    print(f"       leapfit's own EM: NLL {same_fit['nll_ours']:.6f} vs pyBKT "
          f"{same_fit['nll_pybkt']:.6f}; largest parameter gap "
          f"{max(parameter_gap.values()):.3e} (reported, not asserted — EM is "
          f"multimodal)", file=sys.stderr)
    return 0 if agrees else 1


if __name__ == "__main__":
    raise SystemExit(main())
