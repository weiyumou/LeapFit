#!/usr/bin/env python3
"""Fit AFM to a DataShop student-step export, one row per KC model.

Produces a model-comparison table (AIC, BIC, held-out RMSE) locally, so KC
models can be compared without a round trip through DataShop or LearnSphere.

    leapfit-afm examples/student-step.txt --list-models

    leapfit-afm examples/student-step.txt \\
        --kc-model Topics --kc-model Skills \\
        --cv item_blocked --seeds 0:50 --out afm-results.csv \\
        --predictions annotated-student-step.txt

``python -m leapfit.cli ...`` is the same entry point from a source checkout.

With ``--seeds`` the CV is repeated over each seed and the table reports the
mean and standard deviation. The standard deviation describes how much the
score moves with the fold partition; it is *not* a standard error, and a
t-test across these repeats treats non-independent resamples as independent.
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
    cross_validate,
    fit_afm,
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export", help="DataShop student-step export (tab-delimited)")
    p.add_argument("--kc-model", action="append", dest="kc_models", metavar="NAME",
                   help="KC model to fit; repeat for several. Default: all.")
    p.add_argument("--list-models", action="store_true",
                   help="print the KC models in the export and exit")
    p.add_argument("--cv", choices=(*SCHEMES, "none"), default="item_blocked")
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
    p.add_argument("--learnsphere-compat", action="store_true",
                   help="reproduce the published baseline: student ridge 1.0, no "
                        "identification, nPars = students + 2*KCs")
    p.add_argument("--recompute-opportunities", action="store_true",
                   help="derive T from First Transaction Time instead of DataShop's "
                        "Opportunity column (see README)")
    p.add_argument("--student-l2", type=float, default=None)
    p.add_argument("--out", help="write the table to this CSV")
    p.add_argument("--kc-values", metavar="DIR",
                   help="also write per-KC parameters (DataShop layout) into DIR")
    p.add_argument("--predictions", metavar="FILE",
                   help="write the input file back out with one 'Predicted Error "
                        "Rate (<model>)' column per fitted KC model (DataShop's "
                        "convention; rows without a KC stay blank)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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

    seeds = parse_seeds(args.seeds)
    rows = []
    annotated = None  # the input table, gaining one prediction column per model

    for name in wanted:
        data = load_student_step(args.export, kc_model=name)
        design = build_afm_design(
            data, learnsphere_compat=args.learnsphere_compat,
            student_l2=args.student_l2,
            recompute_opportunities=args.recompute_opportunities)
        print(f"\n=== {name} ===\n{data.summary()}", file=sys.stderr)

        fit = fit_afm(design, data.y, method=args.method, max_fun=args.max_fun)
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

        if args.cv != "none":
            cv_kwargs = {"scheme": args.cv, "n_folds": args.folds,
                         "convention": args.convention, "method": args.method,
                         "max_fun": args.max_fun}
            if seeds:
                table = repeated_cross_validate(design, data, seeds=seeds, **cv_kwargs)
                row |= {
                    "cv_rmse": table["rmse"].mean(),
                    "cv_rmse_sd": table["rmse"].std(ddof=1),
                    "cv_runs": len(table),
                    "cv_unseen_fraction": table["unseen_column_fraction"].mean(),
                    "cv_all_converged": bool(table["all_converged"].all()),
                }
                print(f"  {args.cv} / {args.convention} over {len(table)} seeds: "
                      f"RMSE = {row['cv_rmse']:.4f} ({row['cv_rmse_sd']:.4f})",
                      file=sys.stderr)
            else:
                result = cross_validate(design, data, seed=None, **cv_kwargs)
                row |= {
                    "cv_rmse": result.rmse,
                    "cv_rmse_sd": np.nan,
                    "cv_runs": 1,
                    "cv_unseen_fraction": float(np.mean(
                        [f.unseen_column_fraction for f in result.folds])),
                    "cv_all_converged": all(f.converged for f in result.folds),
                }
                print(f"  {result.summary()}", file=sys.stderr)

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
    if annotated is not None:
        # Original cells were read as strings so they round-trip verbatim; the
        # only float columns are the ones we added, and NaN writes as blank.
        annotated.to_csv(args.predictions, sep="\t", index=False,
                         float_format="%.6f", lineterminator="\n")
        print(f"wrote {args.predictions}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
