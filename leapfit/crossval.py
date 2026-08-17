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
so repeated runs give different partitions — which is what makes a repeated
"N random seeds" protocol meaningful. Note the last fold absorbs the
remainder, so fold sizes are unequal when the label count is not divisible.

Blocking by item or by student leaves some columns of the design with no
training rows. Their coefficients stay at the ``w0 = 0`` start (zero gradient,
no penalty pulling them), so predictions for those rows fall back to the
student intercept alone. That is the reference's behaviour, and it is the
mechanism by which a fine-grained KC model can post a *worse* item-blocked
RMSE than ``Single-KC`` despite fitting the training data better. We report
the affected fraction per fold as :attr:`FoldResult.unseen_column_fraction`
rather than leaving it invisible.

Every entry point here takes ``n_jobs``. Fold fits are independent — a fold is
a solve over its own row subset and shares nothing with its neighbours — so
they spread over processes with no coordination. What does *not* move is the
randomness: partitions are drawn in the parent from the same seeded generator
and results are collected in submission order, so ``n_jobs=8`` returns bitwise
what ``n_jobs=1`` returns. Parallelism buys wall clock and changes no number,
which is the only form of it worth having in a package whose whole point is
that a reported RMSE can be reproduced.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
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


def _checked_folds(data: StepData, scheme: str, n_folds: int, seed: int | None,
                   convention: str) -> list[np.ndarray]:
    """:func:`make_folds`, refusing a partition that cannot be scored."""
    folds = make_folds(data, scheme, n_folds, seed, convention)
    for f, test_idx in enumerate(folds):
        n_train = len(data) - len(test_idx)
        if n_train == 0 or len(test_idx) == 0:
            raise ValueError(
                f"Fold {f} is degenerate: {n_train} train / {len(test_idx)} test")
    return folds


# --------------------------------------------------------------------------
# Running the fold fits, optionally across processes
#
# Only the designs and the responses cross into a worker, and they cross once
# per worker rather than once per fold: they go through the pool initializer
# and land in ``_WORKER``, and a job is then just ``(model, seed, fold,
# held-out row indices)``. That matters because ``StepData`` can carry a
# 45 MB source table it does not need on the other side, and because a design
# for a 42,000-row course pickles to a couple of megabytes that would
# otherwise be resent for every fold of every seed.
# --------------------------------------------------------------------------

_SOLE = "model"  # the model name used when there is only one design
_WORKER: dict = {}


def _init_worker(models: dict[str, Design], y: np.ndarray, method: str,
                 max_fun: int | None) -> None:
    _WORKER.update(models=models, y=y, method=method, max_fun=max_fun)


def _score_fold(job: tuple) -> dict:
    """Fit one training split and score its held-out rows. Runs in a worker."""
    name, _seed, _fold, test_idx = job
    design, y = _WORKER["models"][name], _WORKER["y"]
    train_idx = np.setdiff1d(np.arange(design.n_obs), test_idx, assume_unique=False)

    train_design, test_design = design.take(train_idx), design.take(test_idx)
    fit = fit_logistic(train_design, y[train_idx], method=_WORKER["method"],
                       max_fun=_WORKER["max_fun"], warn_not_converged=False,
                       warn_separated=False)
    resid = y[test_idx] - fit.predict_proba(test_design)

    trained = np.asarray((train_design.matrix != 0).sum(axis=0)).ravel() > 0
    touches_unseen = np.asarray(
        (test_design.matrix[:, ~trained] != 0).sum(axis=1)
    ).ravel() > 0

    return {
        "n_train": len(train_idx), "n_test": len(test_idx),
        "resid": resid,
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "sse": float(np.sum(resid ** 2)),
        "unseen_column_fraction": float(touches_unseen.mean()),
        "converged": bool(fit.converged),
        "is_optimal": bool(fit.is_optimal),
    }


def _worker_count(n_jobs: int | None, n_tasks: int) -> int:
    """joblib's convention: ``-1`` is every core, ``-2`` all but one."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        n_jobs = (os.cpu_count() or 1) + 1 + n_jobs
    return max(1, min(n_jobs, n_tasks))


def _run_folds(jobs: list[tuple], models: dict[str, Design], y: np.ndarray,
               method: str, max_fun: int | None, n_jobs: int | None) -> list[dict]:
    """Score every job, in submission order whatever the worker count."""
    workers = _worker_count(n_jobs, len(jobs))
    if workers == 1:
        try:
            _init_worker(models, y, method, max_fun)
            return [_score_fold(job) for job in jobs]
        finally:
            _WORKER.clear()  # don't pin the designs in the caller's process
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(models, y, method, max_fun)) as pool:
        return list(pool.map(_score_fold, jobs))


def _combine(scored: list[dict], convention: str) -> float:
    if convention == "per_fold":
        return float(np.mean([s["rmse"] for s in scored]))
    resid = np.concatenate([s["resid"] for s in scored])
    return float(np.sqrt(np.mean(resid ** 2)))


def cross_validate(design: Design, data: StepData, *, scheme: str = "item_blocked",
                   n_folds: int = 3, seed: int | None = None,
                   convention: str = "per_fold", method: str = DEFAULT_METHOD,
                   max_fun: int | None = None, n_jobs: int | None = 1) -> CVResult:
    """Refit the design on each training split and score the held-out rows.

    :param n_jobs: worker processes to fit folds in, ``-1`` for every core.
        The result does not depend on it.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")

    y = np.asarray(data.y, dtype=float)
    test_folds = _checked_folds(data, scheme, n_folds, seed, convention)
    jobs = [(_SOLE, seed, f, idx) for f, idx in enumerate(test_folds)]
    scored = _run_folds(jobs, {_SOLE: design}, y, method, max_fun, n_jobs)

    results = tuple(
        FoldResult(fold=f, n_train=s["n_train"], n_test=s["n_test"],
                   rmse=s["rmse"], unseen_column_fraction=s["unseen_column_fraction"],
                   converged=s["converged"])
        for f, s in enumerate(scored)
    )
    return CVResult(scheme=scheme, convention=convention, n_folds=n_folds,
                    seed=seed, rmse=_combine(scored, convention), folds=results)


def repeated_cross_validate(design: Design, data: StepData, *, seeds,
                            scheme: str = "item_blocked", n_folds: int = 3,
                            convention: str = "per_fold",
                            method: str = DEFAULT_METHOD,
                            max_fun: int | None = None,
                            n_jobs: int | None = 1) -> pd.DataFrame:
    """Repeat CV over seeds, as published KC-model comparisons commonly do.

    Returns one row per seed. Note that averaging these and running a t-test
    over them treats non-independent resamples as independent; the spread is
    a description of partition sensitivity, not a standard error.

    Every (seed, fold) pair is one job in a single pool, rather than one pool
    per seed: a 50-seed 3-fold protocol is 150 independent fits, and cutting it
    at the seed boundary would idle every core past the third.
    """
    seeds = list(seeds)
    y = np.asarray(data.y, dtype=float)

    jobs, widths = [], []
    for seed in seeds:
        folds = _checked_folds(data, scheme, n_folds, seed, convention)
        widths.append(len(folds))
        jobs += [(_SOLE, seed, f, idx) for f, idx in enumerate(folds)]
    scored = _run_folds(jobs, {_SOLE: design}, y, method, max_fun, n_jobs)

    rows, start = [], 0
    for seed, width in zip(seeds, widths):
        chunk, start = scored[start:start + width], start + width
        rows.append({
            "seed": seed, "scheme": scheme, "convention": convention,
            "rmse": _combine(chunk, convention),
            "unseen_column_fraction": float(np.mean(
                [s["unseen_column_fraction"] for s in chunk])),
            "all_converged": all(s["converged"] for s in chunk),
        })
    return pd.DataFrame(rows)


def paired_cross_validate(models: dict[str, Design], data: StepData, *,
                          scheme: str = "item_blocked", n_folds: int = 3,
                          seeds=(0,), convention: str = "pooled",
                          method: str = DEFAULT_METHOD,
                          max_fun: int | None = None,
                          n_jobs: int | None = 1) -> pd.DataFrame:
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

    ``n_jobs`` spreads those (seed, fold, model) fits over processes; every
    design is shipped to each worker once, so adding models to the comparison
    costs fits, not transfers.
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

    jobs = []
    for seed in seeds:
        for f, test_idx in enumerate(
                _checked_folds(data, scheme, n_folds, seed, convention)):
            jobs += [(name, seed, f, test_idx) for name in models]
    scored = _run_folds(jobs, models, y, method, max_fun, n_jobs)

    return pd.DataFrame([
        {"seed": seed, "fold": f, "model": name, "n_test": s["n_test"],
         "rmse": s["rmse"], "sse": s["sse"], "is_optimal": s["is_optimal"]}
        for (name, seed, f, _), s in zip(jobs, scored)
    ])


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
