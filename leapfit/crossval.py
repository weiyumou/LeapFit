"""Cross-validation for AFM, with LearnSphere's two incompatible conventions.

The two AFM components in WorkflowComponents disagree about what a
"cross-validated RMSE" is, and the difference lands in the third or fourth
decimal — exactly where KC-model comparisons are decided. Both are here and
the choice is explicit, because a table that does not say which one it used
is not reproducible.

``"per_fold"`` — PyAFM (``process_datashop.py``): compute RMSE within each
fold, then average the fold RMSEs. Folds come from ``LabelKFold``, which is
*deterministic*: it sorts labels by frequency and fills the emptiest fold, and
accepts no random seed. Re-running it cannot produce a different partition.

``"pooled"`` — FastAfmAndCv (``learner_performance_prediction.py``): shuffle
the labels, cut them into contiguous blocks, pool every held-out prediction
across folds, and take one RMSE over the pooled vector. The shuffle is seeded,
so repeated runs give different partitions — which is what makes a "50 random
seeds" protocol like EDM 2025's meaningful. Note the last fold absorbs the
remainder, so fold sizes are unequal when the label count is not divisible.

Blocking by item or by student leaves some columns of the design with no
training rows. Their coefficients stay at the ``w0 = 0`` start (zero gradient,
no penalty pulling them), so predictions for those rows fall back to the
student intercept alone. That is the reference's behaviour, and it is the
mechanism by which a fine-grained KC model can post a *worse* item-blocked
RMSE than ``Single-KC`` despite fitting the training data better. We report
the affected fraction per fold as :attr:`FoldResult.unseen_column_fraction`
rather than leaving it invisible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from leapfit.data import StepData
from leapfit.design import Design
from leapfit.fit import DEFAULT_METHOD, fit_logistic

SCHEMES = ("unstratified", "response_stratified", "student_blocked", "item_blocked")
CONVENTIONS = ("per_fold", "pooled")


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    rmse: float
    unseen_column_fraction: float
    converged: bool


@dataclass(frozen=True)
class CVResult:
    scheme: str
    convention: str
    n_folds: int
    seed: int | None
    rmse: float
    folds: tuple[FoldResult, ...]

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.folds])

    def summary(self) -> str:
        unseen = np.mean([f.unseen_column_fraction for f in self.folds])
        bad = sum(not f.converged for f in self.folds)
        return (
            f"{self.scheme} / {self.convention} / {self.n_folds} folds"
            f"{'' if self.seed is None else f' / seed {self.seed}'}: "
            f"RMSE = {self.rmse:.4f}"
            f" | {unseen:.1%} of held-out rows hit an unseen column"
            + (f" | {bad} fold(s) did not converge" if bad else "")
        )


def _label_kfold(labels: list[str], n_folds: int) -> list[np.ndarray]:
    """sklearn 0.17 ``LabelKFold``: deterministic, frequency-balanced."""
    unique, counts = np.unique(np.asarray(labels), return_counts=True)
    order = np.argsort(-counts)
    weight = np.zeros(n_folds, dtype=np.int64)
    assignment: dict[str, int] = {}
    for idx in order:
        target = int(np.argmin(weight))
        weight[target] += counts[idx]
        assignment[unique[idx]] = target
    fold_of = np.array([assignment[v] for v in labels])
    return [np.flatnonzero(fold_of == f) for f in range(n_folds)]


def _shuffle_slice_kfold(labels: list[str], n_folds: int,
                         rng: np.random.Generator) -> list[np.ndarray]:
    """FastAfmAndCv: shuffle unique labels, cut into contiguous blocks."""
    unique = pd.unique(np.asarray(labels))
    unique = unique.copy()
    rng.shuffle(unique)
    span = int(len(unique) / n_folds)
    if span == 0:
        raise ValueError(f"{len(unique)} labels cannot fill {n_folds} folds")
    arr = np.asarray(labels)
    folds = []
    for f in range(n_folds):
        held = unique[f * span:] if f == n_folds - 1 else unique[f * span:(f + 1) * span]
        folds.append(np.flatnonzero(np.isin(arr, held)))
    return folds


def _row_kfold(n: int, n_folds: int, rng: np.random.Generator,
               strata: np.ndarray | None = None) -> list[np.ndarray]:
    """Shuffled row folds, optionally stratified on the response."""
    if strata is None:
        idx = rng.permutation(n)
        return [np.sort(part) for part in np.array_split(idx, n_folds)]
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for value in np.unique(strata):
        member = np.flatnonzero(strata == value)
        member = member[rng.permutation(len(member))]
        for f, part in enumerate(np.array_split(member, n_folds)):
            folds[f].extend(part.tolist())
    return [np.sort(np.asarray(f, dtype=int)) for f in folds]


def make_folds(data: StepData, scheme: str, n_folds: int,
               seed: int | None, convention: str) -> list[np.ndarray]:
    """Held-out row indices per fold, under the requested blocking scheme."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    rng = np.random.default_rng(seed)
    n = len(data)

    match scheme:
        case "unstratified":
            return _row_kfold(n, n_folds, rng)
        case "response_stratified":
            return _row_kfold(n, n_folds, rng, strata=data.y)
        case "student_blocked" | "item_blocked":
            labels = data.students if scheme == "student_blocked" else data.items
            if convention == "per_fold" and seed is None:
                return _label_kfold(labels, n_folds)
            return _shuffle_slice_kfold(labels, n_folds, rng)
    raise AssertionError(scheme)  # unreachable


def cross_validate(design: Design, data: StepData, *, scheme: str = "item_blocked",
                   n_folds: int = 3, seed: int | None = None,
                   convention: str = "per_fold", method: str = DEFAULT_METHOD,
                   max_fun: int | None = None) -> CVResult:
    """Refit the design on each training split and score the held-out rows."""
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")

    y = np.asarray(data.y, dtype=float)
    test_folds = make_folds(data, scheme, n_folds, seed, convention)
    all_rows = np.arange(len(data))

    results, pooled_true, pooled_pred = [], [], []
    for f, test_idx in enumerate(test_folds):
        train_idx = np.setdiff1d(all_rows, test_idx, assume_unique=False)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"Fold {f} is degenerate: {len(train_idx)} train / {len(test_idx)} test")

        train_design, test_design = design.take(train_idx), design.take(test_idx)
        fit = fit_logistic(train_design, y[train_idx], method=method,
                      max_fun=max_fun, warn_not_converged=False,
                      warn_separated=False)
        pred = fit.predict_proba(test_design)

        trained = np.asarray((train_design.matrix != 0).sum(axis=0)).ravel() > 0
        touches_unseen = np.asarray(
            (test_design.matrix[:, ~trained] != 0).sum(axis=1)
        ).ravel() > 0

        results.append(FoldResult(
            fold=f, n_train=len(train_idx), n_test=len(test_idx),
            rmse=float(np.sqrt(np.mean((y[test_idx] - pred) ** 2))),
            unseen_column_fraction=float(touches_unseen.mean()),
            converged=fit.converged,
        ))
        pooled_true.append(y[test_idx])
        pooled_pred.append(pred)

    if convention == "per_fold":
        rmse = float(np.mean([r.rmse for r in results]))
    else:
        resid = np.concatenate(pooled_true) - np.concatenate(pooled_pred)
        rmse = float(np.sqrt(np.mean(resid ** 2)))

    return CVResult(scheme=scheme, convention=convention, n_folds=n_folds,
                    seed=seed, rmse=rmse, folds=tuple(results))


def repeated_cross_validate(design: Design, data: StepData, *, seeds,
                            **kwargs) -> pd.DataFrame:
    """Repeat CV over seeds — the EDM 2025 protocol (item-blocked, 50 seeds).

    Returns one row per seed. Note that averaging these and running a t-test
    over them treats non-independent resamples as independent; the spread is
    a description of partition sensitivity, not a standard error.
    """
    rows = []
    for seed in seeds:
        result = cross_validate(design, data, seed=seed, **kwargs)
        rows.append({
            "seed": seed, "scheme": result.scheme, "convention": result.convention,
            "rmse": result.rmse,
            "unseen_column_fraction": float(np.mean(
                [f.unseen_column_fraction for f in result.folds])),
            "all_converged": all(f.converged for f in result.folds),
        })
    return pd.DataFrame(rows)


def paired_cross_validate(models: dict[str, Design], data: StepData, *,
                          scheme: str = "item_blocked", n_folds: int = 3,
                          seeds=(0,), convention: str = "pooled",
                          method: str = DEFAULT_METHOD,
                          max_fun: int | None = None) -> pd.DataFrame:
    """Score several designs on **identical** folds, for paired comparison.

    Every KC model built from one export covers the same rows, so seed ``s``
    can produce the same split for all of them. Scoring them on shared folds
    turns model comparison into a *paired* contrast — each fold contributes one
    difference per model pair — instead of two independent means.

    That matters because the usual protocol (repeat CV over 50 seeds, average,
    t-test the two averages) treats non-independent resamples as independent
    samples, which has no valid variance estimator. Differences taken within a
    fold sidestep it: the fold is held fixed, so the partition is no longer a
    source of between-model variance at all. It is also far more powerful,
    since fold-to-fold variation is usually much larger than the gap between
    two KC models.

    Returns one row per (seed, fold, model). Pivot on ``model`` and difference
    the columns to get the paired contrasts.
    """
    sizes = {name: d.n_obs for name, d in models.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(
            f"Designs cover different numbers of rows {sizes}; folds cannot be "
            "shared, so the comparison would not be paired."
        )
    if next(iter(sizes.values())) != len(data):
        raise ValueError(f"Designs have {next(iter(sizes.values()))} rows, data has {len(data)}")

    y = np.asarray(data.y, dtype=float)
    all_rows = np.arange(len(data))
    records = []

    for seed in seeds:
        for f, test_idx in enumerate(make_folds(data, scheme, n_folds, seed, convention)):
            train_idx = np.setdiff1d(all_rows, test_idx)
            for name, design in models.items():
                fit = fit_logistic(design.take(train_idx), y[train_idx], method=method,
                              max_fun=max_fun, warn_not_converged=False,
                              warn_separated=False)
                pred = fit.predict_proba(design.take(test_idx))
                records.append({
                    "seed": seed, "fold": f, "model": name,
                    "n_test": len(test_idx),
                    "rmse": float(np.sqrt(np.mean((y[test_idx] - pred) ** 2))),
                    "sse": float(np.sum((y[test_idx] - pred) ** 2)),
                    "is_optimal": fit.is_optimal,
                })
    return pd.DataFrame(records)


def paired_contrasts(folds: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Per-model mean RMSE difference against ``baseline``, paired by fold.

    Negative ``mean_diff`` means the model beats the baseline. The reported
    interval is over folds, which describes how consistently it wins; with a
    handful of folds it is a description, not an inferential claim.
    """
    wide = folds.pivot_table(index=["seed", "fold"], columns="model", values="rmse")
    if baseline not in wide.columns:
        raise KeyError(f"baseline {baseline!r} not among {list(wide.columns)}")
    out = []
    for name in wide.columns:
        if name == baseline:
            continue
        diff = wide[name] - wide[baseline]
        out.append({
            "model": name, "baseline": baseline,
            "mean_diff": diff.mean(), "sd_diff": diff.std(ddof=1),
            "n_folds": int(diff.notna().sum()),
            "folds_better": int((diff < 0).sum()),
        })
    return pd.DataFrame(out).sort_values("mean_diff", ignore_index=True)
