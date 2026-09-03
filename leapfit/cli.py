#!/usr/bin/env python3
"""Fit a student model to a DataShop student-step export, one row per KC model.

Produces a model-comparison table (AIC, BIC, held-out RMSE) locally, so KC
models can be compared without a round trip through DataShop or LearnSphere.
One command per model family, identical interfaces:

    leapfit-afm examples/student-step.txt --list-models

    leapfit-afm examples/student-step.txt \\
        --kc-model Topics --kc-model Skills \\
        --cv item_blocked --seeds 0:50 --out afm-results.csv \\
        --predictions annotated-student-step.txt

    leapfit-pfa examples/student-step.txt --cv item_blocked

``leapfit-lfa`` is the odd one out, because it answers a different question —
*which* KC model, rather than how well a given one fits. It searches, then
checks the shortlist out of sample, and can write the labelling it found back
out in a form DataShop can import:

    leapfit-lfa examples/student-step.txt \\
        --factors Topics --factors Skills --heuristic bic \\
        --validate 5 --compare Topics --seeds 0:3 -j -1 \\
        --out frontier.csv --refusals refusals.csv \\
        --qmatrix discovered-kc-model.txt

Its table is one row per *searched state* rather than one per KC model, and it
carries the moves the screens refused beside the states they would have
produced. See :mod:`leapfit.lfa` for what the screens are and why a search
needs them.

Several schemes score the same fits in one pass, and the runs behind the means
can be written out beside them:

    leapfit-afm examples/student-step.txt \\
        --cv student_blocked --cv item_blocked --seeds 0:10 \\
        --out comparison.csv --cv-folds cv-folds.csv \\
        --identification identification.csv

With one ``--cv`` the table's columns are ``cv_rmse``, ``cv_rmse_sd``, ...; with
several they carry a ``_<scheme>`` suffix, because there is then no single
held-out score to name.

``python -m leapfit.cli ...`` is ``leapfit-afm`` from a source checkout.

When several KC models are fitted and cover the same rows — the usual case,
since every model comes from one export — the comparison is **paired**: folds
are drawn once per scheme and seed and every model is scored on those
identical partitions. The ``cv_rmse`` columns are then comparable by
construction, and a contrasts table reports each model's within-fold RMSE
difference against a baseline (the best-scoring model unless ``--baseline``
names one; negative ``mean_diff`` = beats the baseline). That within-fold
difference is the statistically sound comparison: differencing with the fold
held fixed removes the partition from the between-model variance entirely,
where t-testing two independently repeated CV means treats non-independent
resamples as independent. Models that cover *different* rows (steps mapped in
one model but not another) cannot share folds; the run says so and falls back
to independent per-model CV. ``--no-paired`` forces that fallback — which is
also the only route to the deterministic ``LabelKFold`` partition (``per_fold``
convention without ``--seeds``) that reproduces PyAFM literally.

With ``--seeds`` the CV is repeated over each seed and the table reports the
mean and standard deviation. The standard deviation describes how much the
score moves with the fold partition; it is *not* a standard error, and a
t-test across these repeats treats non-independent resamples as independent.
Paired runs without ``--seeds`` use seed 0, since shared folds must be seeded.

That is also where the time goes — a 50-seed 3-fold protocol is 150 fits per
KC model per scheme — so ``-j`` spreads those fits across cores. The fits are
independent and the partitions are drawn before any of them start, so ``-j -1``
produces the same table as ``-j 1``, sooner.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

import numpy as np
import pandas as pd

from leapfit import (
    CONVENTIONS,
    HEURISTICS,
    SCHEMES,
    build_afm_design,
    build_factor_matrix,
    build_pfa_design,
    cross_validate,
    fit_afm,
    fit_pfa,
    from_frame,
    lfa_search,
    list_kc_models,
    load_student_step,
    paired_contrasts,
    paired_cross_validate,
    paired_scores,
    repeated_cross_validate,
    validate_top,
)
from leapfit.lfa import BEAM, MAX_ITERATIONS, MIN_OPPORTUNITIES, PATIENCE, relabel


def parse_seeds(spec: str | None) -> list[int] | None:
    """``"0:50"`` -> range(0, 50); ``"1,2,3"`` -> [1, 2, 3]; ``None`` -> None."""
    if not spec:
        return None
    if ":" in spec:
        lo, hi = spec.split(":", 1)
        return list(range(int(lo), int(hi)))
    return [int(s) for s in spec.split(",") if s.strip()]


def build_parser(family: str = "afm") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"leapfit-{family}", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export", help="DataShop student-step export (tab-delimited)")
    p.add_argument("--kc-model", action="append", dest="kc_models", metavar="NAME",
                   help="KC model to fit; repeat for several. Default: all.")
    p.add_argument("--list-models", action="store_true",
                   help="print the KC models in the export and exit")
    p.add_argument("--cv", action="append", dest="cv_schemes",
                   choices=(*SCHEMES, "none"), metavar="SCHEME",
                   help=f"blocking scheme, one of {', '.join((*SCHEMES, 'none'))}; "
                        "repeat to score several on the same fits (default: "
                        "item_blocked). With one scheme the table's columns are "
                        "'cv_rmse', 'cv_rmse_sd', ...; with several they are "
                        "suffixed by scheme, since there is no longer one score.")
    p.add_argument("--convention", choices=CONVENTIONS, default="per_fold",
                   help="per_fold = PyAFM (mean of fold RMSEs); "
                        "pooled = FastAfmAndCv (RMSE of pooled residuals)")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seeds", metavar="LO:HI|a,b,c",
                   help="repeat CV over these seeds (e.g. '0:50' for a "
                        "50-seed protocol). Paired CV defaults to seed 0, "
                        "since shared folds must be seeded.")
    p.add_argument("--no-paired", action="store_true",
                   help="draw folds independently per KC model instead of "
                        "sharing them across models. Shared folds are the "
                        "default whenever every fitted model covers the same "
                        "rows; independent folds are the pre-0.4 protocol and "
                        "the only route to the deterministic LabelKFold "
                        "partition (per_fold convention without --seeds).")
    p.add_argument("--baseline", metavar="NAME",
                   help="KC model to contrast the others against in paired CV "
                        "(default: the model with the best mean cv_rmse per "
                        "scheme)")
    p.add_argument("--method", default="TNC",
                   help="TNC reproduces LearnSphere; L-BFGS-B converges tighter")
    p.add_argument("--max-fun", type=int, default=None,
                   help="function-evaluation budget; default = solver default, "
                        "which is what published fits effectively used")
    p.add_argument("--jobs", "-j", type=int, default=1, metavar="N",
                   help="worker processes for the cross-validation fits "
                        "(-1 = every core). Folds are drawn in the parent and "
                        "collected in order, so this changes the wall clock and "
                        "not the numbers.")
    if family == "afm":
        p.add_argument("--learnsphere-compat", action="store_true",
                       help="reproduce the published baseline: student ridge 1.0, no "
                            "identification, nPars = students + 2*KCs")
        p.add_argument("--recompute-opportunities", action="store_true",
                       help="derive T from First Transaction Time instead of "
                            "DataShop's Opportunity column (see README)")
        p.add_argument("--student-l2", type=float, default=None)
    else:
        p.add_argument("--pooled-slopes", action="store_true",
                       help="one shared success/failure slope instead of per-KC "
                            "slopes (the 'Simple' PFA variant)")
        p.add_argument("--student-intercepts", action="store_true",
                       help="add fixed student intercepts; canonical PFA has none")
        p.add_argument("--inclusive-counts", action="store_true",
                       help="reproduce AnalysisPfaStepBased's inclusive counts, "
                            "which put each response inside its own predictor. "
                            "For demonstrating that defect only.")
    p.add_argument("--out", help="write the table to this CSV")
    p.add_argument("--contrasts", metavar="FILE",
                   help="write the paired contrasts (per-model RMSE difference "
                        "against the baseline, paired by fold) to this CSV")
    p.add_argument("--cv-folds", metavar="FILE",
                   help="write the cross-validation detail behind the means to "
                        "this CSV — one row per KC model, scheme, seed and fold "
                        "for paired CV (per seed for independent repeated CV; "
                        "per fold for a single run), so a reported mean can be "
                        "traced to the runs behind it")
    p.add_argument("--identification", metavar="FILE",
                   help="write every aliased column and the reason it was dropped "
                        "to this CSV: what the parameter count excludes, and why")
    p.add_argument("--kc-values", metavar="DIR",
                   help="also write per-KC parameters (DataShop layout) into DIR")
    p.add_argument("--predictions", metavar="FILE",
                   help="write the input file back out with one 'Predicted Error "
                        "Rate (<model>)' column per fitted KC model (DataShop's "
                        "convention; rows without a KC stay blank)")
    return p


def _build_and_fit(args, family: str, data, method: str, max_fun):
    """The one family-specific step: assemble the design, fit it."""
    if family == "afm":
        design = build_afm_design(
            data, learnsphere_compat=args.learnsphere_compat,
            student_l2=args.student_l2,
            recompute_opportunities=args.recompute_opportunities)
        fit = fit_afm(design, data.y, method=method, max_fun=max_fun)
    else:
        design = build_pfa_design(
            data,
            slopes="pooled" if args.pooled_slopes else "per_kc",
            student_intercepts=args.student_intercepts,
            counts="inclusive" if args.inclusive_counts else "prior")
        fit = fit_pfa(design, data.y, method=method, max_fun=max_fun)
    return design, fit


def _shared_rows(fitted: list) -> bool:
    """Whether every fitted model covers exactly the same export rows.

    Equal row *counts* are not enough: two KC models can each drop a different
    set of unmapped steps and land on the same total. ``source_rows`` is the
    position of each observation in the export, so equality there is equality
    of coverage (and of ``y``, row for row).
    """
    ref = fitted[0].data.source_rows
    return all(np.array_equal(entry.data.source_rows, ref) for entry in fitted[1:])


class _Fitted:
    """One KC model's artifacts, carried from the fit pass to the CV pass."""

    def __init__(self, name, data, design, fit, row):
        self.name, self.data, self.design, self.fit, self.row = \
            name, data, design, fit, row


def _independent_cv(args, fitted: list, schemes: list[str], seeds, suffix,
                    fold_rows: list) -> None:
    """The pre-paired protocol: each model scored on its own folds."""
    print("\n=== cross-validation: independent folds per model ===", file=sys.stderr)
    for entry in fitted:
        for scheme in schemes:
            cv_kwargs = {"scheme": scheme, "n_folds": args.folds,
                         "convention": args.convention, "method": args.method,
                         "max_fun": args.max_fun, "n_jobs": args.jobs}
            s = suffix(scheme)
            if seeds:
                table = repeated_cross_validate(entry.design, entry.data,
                                                seeds=seeds, **cv_kwargs)
                entry.row |= {
                    f"cv_rmse{s}": table["rmse"].mean(),
                    f"cv_rmse_sd{s}": table["rmse"].std(ddof=1),
                    f"cv_runs{s}": len(table),
                    f"cv_unseen_fraction{s}": table["unseen_column_fraction"].mean(),
                    f"cv_all_converged{s}": bool(table["all_converged"].all()),
                }
                detail = table
                print(f"  {entry.name}: {scheme} / {args.convention} over "
                      f"{len(table)} seeds: RMSE = {entry.row[f'cv_rmse{s}']:.4f} "
                      f"({entry.row[f'cv_rmse_sd{s}']:.4f})", file=sys.stderr)
            else:
                result = cross_validate(entry.design, entry.data, seed=None,
                                        **cv_kwargs)
                entry.row |= {
                    f"cv_rmse{s}": result.rmse,
                    f"cv_rmse_sd{s}": np.nan,
                    f"cv_runs{s}": 1,
                    f"cv_unseen_fraction{s}": float(np.mean(
                        [f.unseen_column_fraction for f in result.folds])),
                    f"cv_all_converged{s}": all(f.converged for f in result.folds),
                }
                detail = result.frame
                print(f"  {entry.name}: {result.summary()}", file=sys.stderr)

            if args.cv_folds:
                detail = detail.copy()
                if "scheme" not in detail:
                    detail.insert(0, "scheme", scheme)
                detail.insert(0, "kc_model", entry.name)
                fold_rows.append(detail)


def _paired_cv(args, fitted: list, schemes: list[str], seeds, suffix,
               fold_rows: list, contrast_rows: list) -> None:
    """Shared folds across all models; scores and contrasts from one set of fits."""
    seeds = seeds or [0]
    models = {entry.name: entry.design for entry in fitted}
    data = fitted[0].data
    print(f"\n=== cross-validation: paired, folds shared by all {len(models)} "
          f"models ({len(seeds)} seed(s); --no-paired for the independent "
          "protocol) ===", file=sys.stderr)

    for scheme in schemes:
        folds = paired_cross_validate(
            models, data, scheme=scheme, n_folds=args.folds, seeds=tuple(seeds),
            convention=args.convention, method=args.method,
            max_fun=args.max_fun, n_jobs=args.jobs)
        scores = paired_scores(folds, args.convention)
        s = suffix(scheme)

        print(f"  {scheme} / {args.convention} / {args.folds} folds:", file=sys.stderr)
        for entry in fitted:
            sc = scores[scores["model"] == entry.name]
            entry.row |= {
                f"cv_rmse{s}": float(sc["rmse"].mean()),
                f"cv_rmse_sd{s}": float(sc["rmse"].std(ddof=1)),
                f"cv_runs{s}": len(sc),
                f"cv_unseen_fraction{s}": float(sc["unseen_column_fraction"].mean()),
                f"cv_all_converged{s}": bool(sc["all_converged"].all()),
            }
            print(f"    {entry.name}: RMSE = {entry.row[f'cv_rmse{s}']:.4f}"
                  f" | {entry.row[f'cv_unseen_fraction{s}']:.1%} of held-out "
                  "rows hit an unseen column", file=sys.stderr)

        baseline = args.baseline or scores.groupby("model")["rmse"].mean().idxmin()
        contrasts = paired_contrasts(folds, baseline=baseline)
        contrasts.insert(0, "scheme", scheme)
        contrast_rows.append(contrasts)
        for c in contrasts.itertuples():
            print(f"    {c.model} vs {baseline}: {c.mean_diff:+.6f} "
                  f"(better in {c.folds_better}/{c.n_folds} folds)", file=sys.stderr)

        if args.cv_folds:
            fold_rows.append(
                folds.rename(columns={"model": "kc_model"}).assign(scheme=scheme)[
                    ["kc_model", "scheme", "seed", "fold", "n_test", "rmse", "sse",
                     "unseen_column_fraction", "converged", "is_optimal"]])


def main(argv: list[str] | None = None, *, family: str = "afm") -> int:
    args = build_parser(family).parse_args(argv)

    available = list_kc_models(args.export)
    if args.list_models:
        print("\n".join(available) or "(no KC models found)")
        return 0
    if not available:
        print(f"No 'KC (...)' columns in {args.export}", file=sys.stderr)
        return 1

    wanted = args.kc_models or available
    if unknown := [m for m in wanted if m not in available]:
        print(f"Unknown KC model(s) {unknown}. Available: {available}", file=sys.stderr)
        return 1
    if args.baseline and args.baseline not in wanted:
        print(f"--baseline {args.baseline!r} is not among the fitted KC models "
              f"{wanted}", file=sys.stderr)
        return 1

    schemes = args.cv_schemes or ["item_blocked"]
    if "none" in schemes and len(schemes) > 1:
        print(f"--cv none cannot be combined with {[s for s in schemes if s != 'none']}",
              file=sys.stderr)
        return 1
    # One scheme keeps the historical column names, so existing invocations and
    # anything parsing their CSV are unaffected; several need the suffix.
    suffix = (lambda s: "") if len(schemes) == 1 else (lambda s: f"_{s}")

    seeds = parse_seeds(args.seeds)
    rows, fold_rows, alias_rows, contrast_rows = [], [], [], []
    annotated = None  # the input table, gaining one prediction column per model

    # ---- fit every model once; CV needs them all before it can share folds ----
    export = pd.read_csv(args.export, sep="\t", dtype=str, keep_default_na=False)
    fitted: list[_Fitted] = []
    for name in wanted:
        data = from_frame(export, kc_model=name)
        print(f"\n=== {name} ===\n{data.summary()}", file=sys.stderr)

        design, fit = _build_and_fit(args, family, data, args.method, args.max_fun)
        print(fit.summary(), file=sys.stderr)

        row = {
            "kc_model": name,
            "n_kcs": len(data.kc_names),
            "n_students": len(data.student_names),
            "n_obs": len(data),
            "n_params": design.n_params,
            "n_aliased": len(design.aliased),
            "n_separated": len(fit.separated),
            "log_likelihood": fit.ll,
            "aic": fit.aic,
            "bic": fit.bic,
            "is_optimal": fit.is_optimal,
        }
        rows.append(row)
        fitted.append(_Fitted(name, data, design, fit, row))

        if args.identification:
            alias_rows += [{"kc_model": name, "column": column, "reason": reason}
                           for column, reason in zip(design.aliased.columns,
                                                     design.aliased.reasons)]

        if args.predictions:
            annotated = fit.annotate(data, into=annotated)

        if args.kc_values:
            import os
            os.makedirs(args.kc_values, exist_ok=True)
            safe = name.replace("/", "_")
            path = os.path.join(args.kc_values, f"{safe}_kc-values.csv")
            fit.kc_values(data).to_csv(path, index=False)
            print(f"  KC parameters -> {path}", file=sys.stderr)

    # ---- cross-validation: paired when the models can share folds ----
    cv_schemes = [s for s in schemes if s != "none"]
    if cv_schemes:
        paired = len(fitted) >= 2 and not args.no_paired
        if paired and not _shared_rows(fitted):
            paired = False
            coverage = {entry.name: len(entry.data) for entry in fitted}
            print("\nKC models cover different rows of the export (a step with "
                  f"no KC label is dropped for that model): {coverage}. Shared "
                  "folds are impossible, so falling back to independent "
                  "per-model CV; no paired contrasts.", file=sys.stderr)
        if paired:
            _paired_cv(args, fitted, cv_schemes, seeds, suffix,
                       fold_rows, contrast_rows)
        else:
            _independent_cv(args, fitted, cv_schemes, seeds, suffix, fold_rows)

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if contrast_rows:
        contrasts = pd.concat(contrast_rows, ignore_index=True)
        print("\npaired contrasts, within-fold (negative mean_diff = model "
              "beats baseline):")
        print(contrasts.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}", file=sys.stderr)
    if args.contrasts:
        if contrast_rows:
            contrasts.to_csv(args.contrasts, index=False)
            print(f"wrote {args.contrasts}", file=sys.stderr)
        else:
            print(f"not writing {args.contrasts}: paired CV did not run, so "
                  "there are no contrasts", file=sys.stderr)
    if args.cv_folds and fold_rows:
        pd.concat(fold_rows, ignore_index=True).to_csv(args.cv_folds, index=False)
        print(f"wrote {args.cv_folds}", file=sys.stderr)
    if args.identification:
        # Written even when empty: "nothing was aliased" is a result, and a
        # missing file would be indistinguishable from a run that never asked.
        pd.DataFrame(alias_rows, columns=["kc_model", "column", "reason"]).to_csv(
            args.identification, index=False)
        print(f"wrote {args.identification}", file=sys.stderr)
    if annotated is not None:
        # Original cells were read as strings so they round-trip verbatim; the
        # only float columns are the ones we added, and NaN writes as blank.
        annotated.to_csv(args.predictions, sep="\t", index=False,
                         float_format="%.6f", lineterminator="\n")
        print(f"wrote {args.predictions}", file=sys.stderr)
    return 0


def main_pfa(argv: list[str] | None = None) -> int:
    """The ``leapfit-pfa`` console script."""
    return main(argv, family="pfa")


# --------------------------------------------------------------------------
# leapfit-lfa: search for the KC model, then check it out of sample
#
# A separate entry point rather than another ``family``, because the output is
# a different shape: one row per *searched state* instead of one per KC model,
# plus a trajectory, the moves the screens refused, and a labelling to write
# back. What it shares with the others is the vocabulary — ``--cv``, ``-j``,
# ``--out`` mean here what they mean there.
# --------------------------------------------------------------------------


def build_lfa_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="leapfit-lfa",
        description="Search for a KC model with Learning Factors Analysis, "
                    "scored by AFM, then validate the shortlist out of sample.")
    p.add_argument("export", help="DataShop student-step export (tab-delimited)")
    p.add_argument("--factors", action="append", dest="factor_models",
                   metavar="NAME",
                   help="KC model to draw difficulty factors from; repeatable. "
                        "Default: every eligible model in the export. A model "
                        "is eligible if it tags one KC per row and covers the "
                        "same rows as the others.")
    p.add_argument("--list-models", action="store_true",
                   help="print the KC models in the export and exit")

    p.add_argument("--heuristic", choices=HEURISTICS, default="bic",
                   help="what the search ranks by (default: bic)")
    p.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                   metavar="N", help=f"expansion budget (default: {MAX_ITERATIONS})")
    p.add_argument("--patience", type=int, default=PATIENCE, metavar="N",
                   help=f"stop after N expansions that do not improve the "
                        f"incumbent; 0 disables (default: {PATIENCE})")
    p.add_argument("--min-opportunities", type=int, default=MIN_OPPORTUNITIES,
                   metavar="N",
                   help="evidence screen: observations at T >= 1 a new KC must "
                        f"have (default: {MIN_OPPORTUNITIES}); 0 disables")
    p.add_argument("--no-screen-separation", action="store_true",
                   help="accept moves whose KC parameters have no finite "
                        "estimate — reproduces the reference, including its "
                        "selection of such models")
    p.add_argument("--no-merge", action="store_true",
                   help="split only; skip lineage-undo merges")
    p.add_argument("--beam", type=int, default=BEAM, metavar="N",
                   help=f"unexpanded states retained (default: {BEAM})")
    p.add_argument("--no-warm-start", action="store_true",
                   help="fit every state from zero instead of from its parent")
    p.add_argument("--learnsphere-compat", action="store_true",
                   help="score with the reference's conventions rather than "
                        "rank(X); reproduction only")
    p.add_argument("--method", default="TNC", help="TNC or L-BFGS-B")
    p.add_argument("--max-fun", type=int, default=None, metavar="N")
    p.add_argument("--jobs", "-j", type=int, default=1, metavar="N",
                   help="processes to score an expansion across; -1 is every "
                        "core. A wall-clock knob only")

    p.add_argument("--validate", type=int, default=5, metavar="N",
                   help="cross-validate the top N states on shared folds "
                        "(default: 5); 0 to skip")
    p.add_argument("--compare", action="append", dest="compare_models",
                   metavar="NAME",
                   help="an authored KC model to score alongside them; "
                        "repeatable")
    p.add_argument("--cv", default="item_blocked", choices=SCHEMES,
                   help="fold scheme for the validation (default: item_blocked)")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seeds", metavar="LO:HI|a,b,c",
                   help="repeat the validation over these seeds")
    p.add_argument("--convention", choices=CONVENTIONS, default="pooled")
    p.add_argument("--baseline", metavar="NAME",
                   help="what the paired contrasts difference against "
                        "(default: root, the model the search began from)")

    p.add_argument("--top", type=int, default=10, metavar="N",
                   help="frontier rows to print (default: 10)")
    p.add_argument("--out", metavar="FILE", help="the ranked frontier, as CSV")
    p.add_argument("--refusals", metavar="FILE",
                   help="the moves the screens refused and why, as CSV")
    p.add_argument("--validation", metavar="FILE",
                   help="the held-out comparison, as CSV")
    p.add_argument("--qmatrix", metavar="FILE",
                   help="the winning labelling, one row per step, ready to "
                        "join for a DataShop KC-model import")
    p.add_argument("--kc-model-name", default="LFA_search", metavar="NAME",
                   help="what to call the discovered model in --qmatrix and "
                        "--predictions (default: LFA_search)")
    p.add_argument("--predictions", metavar="FILE",
                   help="the export plus a Predicted Error Rate column for the "
                        "winning model")
    return p


def _eligible_factor_models(export: str, wanted: list[str]) -> tuple[dict, list]:
    """Load the requested models and drop the ones P cannot be built from.

    The reference aborts on an ineligible model. Reporting which ones and why,
    then proceeding with the rest, is more useful and no less explicit — the
    exclusions are printed, so a run never quietly searches a smaller space
    than was asked for.
    """
    loaded, excluded = {}, []
    for name in wanted:
        data = load_student_step(export, kc_model=name)
        if max((len(row) for row in data.kcs), default=0) > 1:
            excluded.append((name, "tags more than one KC on some rows"))
            continue
        loaded[name] = data
    if not loaded:
        return loaded, excluded

    coverage: dict[int, list[str]] = {}
    for name, data in loaded.items():
        coverage.setdefault(len(data), []).append(name)
    if len(coverage) > 1:
        keep = max(coverage.values(), key=len)
        for size, names in coverage.items():
            if names is keep:
                continue
            for name in names:
                majority = len(loaded[keep[0]])
                reason = (f"covers {size:,} observations, not "
                          f"{majority:,} like the majority")
                excluded.append((name, reason))
                del loaded[name]
    return loaded, excluded


def _qmatrix_frame(data, labels_by_step: dict, kc_name: str) -> pd.DataFrame:
    """One row per step: how it is identified in the export, plus its KC."""
    source = data.source.iloc[data.source_rows]
    columns = [c for c in ("Problem Hierarchy", "Problem Name", "Step Name")
               if c in source.columns]
    frame = source[columns].copy()
    frame[f"KC ({kc_name})"] = [labels_by_step[item] for item in data.items]
    return frame.drop_duplicates().reset_index(drop=True)


def main_lfa(argv: list[str] | None = None) -> int:
    """The ``leapfit-lfa`` console script."""
    args = build_lfa_parser().parse_args(argv)

    available = list_kc_models(args.export)
    if args.list_models:
        print("\n".join(available) or "(no KC models found)")
        return 0
    if not available:
        print(f"No 'KC (...)' columns in {args.export}", file=sys.stderr)
        return 1

    wanted = args.factor_models or available
    if unknown := [m for m in wanted if m not in available]:
        print(f"Unknown KC model(s) {unknown}. Available: {available}",
              file=sys.stderr)
        return 1
    for name in args.compare_models or []:
        if name not in available:
            print(f"--compare {name!r} is not a KC model in the export. "
                  f"Available: {available}", file=sys.stderr)
            return 1

    loaded, excluded = _eligible_factor_models(args.export, wanted)
    for name, reason in excluded:
        print(f"excluding {name!r} from the difficulty factors: {reason}",
              file=sys.stderr)
    if not loaded:
        print("No KC model in the export can supply difficulty factors.",
              file=sys.stderr)
        return 1

    factors = build_factor_matrix(loaded)
    # Any eligible model serves as the observations: they share rows, and the
    # search replaces the labelling anyway.
    data = loaded[next(iter(loaded))]
    print(f"factors from {list(loaded)}: {factors.summary()}", file=sys.stderr)
    for column, reason in zip(factors.dropped, factors.reasons):
        print(f"  dropped {column}: {reason}", file=sys.stderr)

    result = lfa_search(
        data, factors, heuristic=args.heuristic,
        max_iterations=args.max_iterations, patience=args.patience,
        min_opportunities=args.min_opportunities,
        screen_separation=not args.no_screen_separation,
        merge=not args.no_merge, beam=args.beam,
        warm_start=not args.no_warm_start,
        learnsphere_compat=args.learnsphere_compat,
        method=args.method, max_fun=args.max_fun, n_jobs=args.jobs)

    print()
    print(result.summary())
    frame = result.frame()
    print()
    print(frame.head(args.top).to_string(
        index=False, float_format=lambda v: f"{v:.4f}", max_colwidth=52))
    if len(frame) > args.top:
        print(f"({len(frame) - args.top} further state(s); --top N for more, "
              "--out FILE for all)")

    validation = None
    if args.validate:
        extra = {name: load_student_step(args.export, kc_model=name)
                 for name in args.compare_models or []}
        seeds = parse_seeds(args.seeds) or [0]
        print(f"\n=== held-out check of the top {args.validate}, folds shared "
              f"by every candidate ===", file=sys.stderr)
        validation = validate_top(
            result, data, n=args.validate, extra=extra, baseline=args.baseline,
            scheme=args.cv, n_folds=args.folds, seeds=seeds,
            convention=args.convention, method=args.method,
            max_fun=args.max_fun, n_jobs=args.jobs)
        print()
        print(validation.summary())

    if args.out:
        frame.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}", file=sys.stderr)
    if args.refusals:
        # Written even when empty: "the screens refused nothing" is a result,
        # and a missing file cannot be told from a run that never asked.
        pd.DataFrame({"move": result.rejected.moves,
                      "reason": result.rejected.reasons}).to_csv(
            args.refusals, index=False)
        print(f"wrote {args.refusals}", file=sys.stderr)
    if args.validation:
        if validation is None:
            print(f"not writing {args.validation}: --validate 0 skipped it",
                  file=sys.stderr)
        else:
            validation.frame().to_csv(args.validation, index=False)
            print(f"wrote {args.validation}", file=sys.stderr)
    if args.qmatrix:
        _qmatrix_frame(data, result.best.kc_model(factors.steps),
                       args.kc_model_name).to_csv(
            args.qmatrix, sep="\t", index=False, lineterminator="\n")
        print(f"wrote {args.qmatrix}", file=sys.stderr)
    if args.predictions:
        scored = relabel(data, factors.steps, result.best.labels)
        scored = dataclasses.replace(scored, kc_model=args.kc_model_name)
        design = build_afm_design(scored,
                                  learnsphere_compat=args.learnsphere_compat,
                                  recompute_opportunities=False)
        fit = fit_afm(design, scored.y, method=args.method,
                      max_fun=args.max_fun, warn_separated=False)
        fit.annotate(scored).to_csv(args.predictions, sep="\t", index=False,
                                    float_format="%.6f", lineterminator="\n")
        print(f"wrote {args.predictions}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
