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

With ``--seeds`` the CV is repeated over each seed and the table reports the
mean and standard deviation. The standard deviation describes how much the
score moves with the fold partition; it is *not* a standard error, and a
t-test across these repeats treats non-independent resamples as independent.

That is also where the time goes — a 50-seed 3-fold protocol is 150 fits per
KC model per scheme — so ``-j`` spreads those fits across cores. The fits are
independent and the partitions are drawn before any of them start, so ``-j -1``
produces the same table as ``-j 1``, sooner.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from leapfit import (
    CONVENTIONS,
    SCHEMES,
    build_afm_design,
    build_pfa_design,
    cross_validate,
    fit_afm,
    fit_pfa,
    list_kc_models,
    load_student_step,
    repeated_cross_validate,
)


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
                        "50-seed protocol)")
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
    p.add_argument("--cv-folds", metavar="FILE",
                   help="write the per-seed cross-validation detail to this CSV — "
                        "one row per KC model, scheme and seed (per fold when no "
                        "--seeds), so a reported mean can be traced to the runs "
                        "behind it")
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

    schemes = args.cv_schemes or ["item_blocked"]
    if "none" in schemes and len(schemes) > 1:
        print(f"--cv none cannot be combined with {[s for s in schemes if s != 'none']}",
              file=sys.stderr)
        return 1
    # One scheme keeps the historical column names, so existing invocations and
    # anything parsing their CSV are unaffected; several need the suffix.
    suffix = (lambda s: "") if len(schemes) == 1 else (lambda s: f"_{s}")

    seeds = parse_seeds(args.seeds)
    rows, fold_rows, alias_rows = [], [], []
    annotated = None  # the input table, gaining one prediction column per model

    for name in wanted:
        data = load_student_step(args.export, kc_model=name)
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

        for scheme in schemes:
            if scheme == "none":
                continue
            cv_kwargs = {"scheme": scheme, "n_folds": args.folds,
                         "convention": args.convention, "method": args.method,
                         "max_fun": args.max_fun, "n_jobs": args.jobs}
            s = suffix(scheme)
            if seeds:
                table = repeated_cross_validate(design, data, seeds=seeds, **cv_kwargs)
                row |= {
                    f"cv_rmse{s}": table["rmse"].mean(),
                    f"cv_rmse_sd{s}": table["rmse"].std(ddof=1),
                    f"cv_runs{s}": len(table),
                    f"cv_unseen_fraction{s}": table["unseen_column_fraction"].mean(),
                    f"cv_all_converged{s}": bool(table["all_converged"].all()),
                }
                detail = table
                print(f"  {scheme} / {args.convention} over {len(table)} seeds: "
                      f"RMSE = {row[f'cv_rmse{s}']:.4f} "
                      f"({row[f'cv_rmse_sd{s}']:.4f})", file=sys.stderr)
            else:
                result = cross_validate(design, data, seed=None, **cv_kwargs)
                row |= {
                    f"cv_rmse{s}": result.rmse,
                    f"cv_rmse_sd{s}": np.nan,
                    f"cv_runs{s}": 1,
                    f"cv_unseen_fraction{s}": float(np.mean(
                        [f.unseen_column_fraction for f in result.folds])),
                    f"cv_all_converged{s}": all(f.converged for f in result.folds),
                }
                detail = result.frame
                print(f"  {result.summary()}", file=sys.stderr)

            if args.cv_folds:
                detail = detail.copy()
                if "scheme" not in detail:
                    detail.insert(0, "scheme", scheme)
                detail.insert(0, "kc_model", name)
                fold_rows.append(detail)

        if args.identification:
            alias_rows += [{"kc_model": name, "column": column, "reason": reason}
                           for column, reason in zip(design.aliased.columns,
                                                     design.aliased.reasons)]

        rows.append(row)

        if args.predictions:
            annotated = fit.annotate(data, into=annotated)

        if args.kc_values:
            import os
            os.makedirs(args.kc_values, exist_ok=True)
            safe = name.replace("/", "_")
            path = os.path.join(args.kc_values, f"{safe}_kc-values.csv")
            fit.kc_values(data).to_csv(path, index=False)
            print(f"  KC parameters -> {path}", file=sys.stderr)

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}", file=sys.stderr)
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


if __name__ == "__main__":
    raise SystemExit(main())
