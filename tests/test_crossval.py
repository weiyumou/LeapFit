"""Cross-validation: the fold schemes, the two scoring conventions, and folds
shared across models so that their scores can be compared pairwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leapfit import (
    build_afm_design,
    cross_validate,
    make_folds,
    paired_contrasts,
    paired_cross_validate,
    paired_scores,
    repeated_cross_validate,
)
from leapfit.crossval import CONVENTIONS

from helpers import synthetic

# --------------------------------------------------------------------------
# Folds and scoring conventions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["unstratified", "response_stratified",
                                    "student_blocked", "item_blocked"])
def test_folds_partition_the_rows(scheme):
    data = synthetic(n_students=9, n_kcs=4, n_items=18, seed=15)
    folds = make_folds(data, scheme, n_folds=3, seed=0, convention="pooled")
    joined = np.concatenate(folds)
    assert len(joined) == len(np.unique(joined)) == len(data)


def test_blocked_folds_keep_a_label_on_one_side():
    data = synthetic(n_students=12, n_kcs=4, n_items=24, seed=16)
    items = np.asarray(data.items)
    for held in make_folds(data, "item_blocked", 3, seed=1, convention="pooled"):
        held_items = set(items[held])
        rest = np.setdiff1d(np.arange(len(data)), held)
        assert not (held_items & set(items[rest])), "an item straddled the split"


def test_label_kfold_is_deterministic_without_a_seed():
    data = synthetic(n_students=12, n_kcs=4, n_items=24, seed=17)
    a = make_folds(data, "item_blocked", 3, seed=None, convention="per_fold")
    b = make_folds(data, "item_blocked", 3, seed=None, convention="per_fold")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_seeds_change_the_partition():
    data = synthetic(n_students=12, n_kcs=4, n_items=24, seed=18)
    a = make_folds(data, "item_blocked", 3, seed=1, convention="pooled")
    b = make_folds(data, "item_blocked", 3, seed=2, convention="pooled")
    assert any(len(x) != len(y) or not np.array_equal(x, y) for x, y in zip(a, b))


def test_the_two_conventions_give_different_numbers():
    """Averaging fold RMSEs is not the RMSE of the pooled residuals."""
    data = synthetic(n_students=20, n_kcs=4, n_items=30, seed=19, n_reps=8)
    design = build_afm_design(data)
    kw = {"scheme": "item_blocked", "n_folds": 3, "seed": 3, "method": "L-BFGS-B"}
    per_fold = cross_validate(design, data, convention="per_fold", **kw)
    pooled = cross_validate(design, data, convention="pooled", **kw)

    assert 0.0 < per_fold.rmse < 1.0
    assert per_fold.rmse != pooled.rmse, (
        "the conventions coincide only when every fold has identical size and "
        "error; if this ever passes trivially the test has stopped testing"
    )
    # Jensen: sqrt is concave, so the mean of fold RMSEs sits at or below the
    # RMSE of the pooled residuals whenever the folds are equally sized.
    assert abs(per_fold.rmse - pooled.rmse) < 0.05, "same quantity, different estimator"


@pytest.mark.parametrize("convention", ["per_fold", "pooled"])
def test_worker_count_does_not_move_a_single_digit(convention):
    """``n_jobs`` is a wall-clock knob, not a modelling one.

    Partitions are drawn in the parent before any fit starts and results are
    collected in submission order, so the only thing a second process can
    change is how long the answer takes. Asserting on the repr rather than
    ``approx`` is deliberate: 'close enough' is the failure mode this guards
    against, since KC-model comparisons are decided in the fourth decimal.
    """
    data = synthetic(n_students=14, n_kcs=4, n_items=20, seed=44, n_reps=5)
    design = build_afm_design(data)
    kw = {"scheme": "item_blocked", "n_folds": 3, "convention": convention,
          "method": "L-BFGS-B"}

    one = cross_validate(design, data, seed=7, n_jobs=1, **kw)
    two = cross_validate(design, data, seed=7, n_jobs=2, **kw)
    assert one == two, "the whole CVResult, per-fold detail included"
    assert repr(one.rmse) == repr(two.rmse)

    seeds = (0, 1, 2)
    pd.testing.assert_frame_equal(
        repeated_cross_validate(design, data, seeds=seeds, n_jobs=1, **kw),
        repeated_cross_validate(design, data, seeds=seeds, n_jobs=2, **kw),
        check_exact=True,
    )

    models = {"afm": design, "compat": build_afm_design(data, learnsphere_compat=True)}
    pd.testing.assert_frame_equal(
        paired_cross_validate(models, data, seeds=seeds, n_jobs=1, **kw),
        paired_cross_validate(models, data, seeds=seeds, n_jobs=2, **kw),
        check_exact=True,
    )


def test_item_blocked_cv_reports_unseen_columns():
    """A KC carried by a single item must be unseen when that item is held out."""
    data = synthetic(n_students=20, n_kcs=30, n_items=30, seed=20, n_reps=4)
    assert len(data.kc_names) == len(set(data.items)), "one KC per item by construction"
    design = build_afm_design(data)
    result = cross_validate(design, data, scheme="item_blocked", n_folds=3,
                            seed=4, method="L-BFGS-B")
    unseen = np.mean([f.unseen_column_fraction for f in result.folds])
    assert unseen > 0.9, (
        "with one KC per item, every held-out row should hit an unseen KC column — "
        "this is the cold-start mechanism behind fine-grained models' item-blocked RMSE"
    )


def test_student_blocked_cv_flags_unseen_students():
    """Held-out students have no intercept in training — except the reference.

    Under identification one student is the reference level and has no column
    at all, so rows belonging to that student touch nothing unseen. Every other
    held-out student's rows do.
    """
    data = synthetic(n_students=12, n_kcs=3, n_items=20, seed=21, n_reps=6)
    result = cross_validate(build_afm_design(data, learnsphere_compat=True), data,
                            scheme="student_blocked", n_folds=3, seed=5,
                            method="L-BFGS-B")
    assert all(f.unseen_column_fraction == pytest.approx(1.0) for f in result.folds)

    identified = cross_validate(build_afm_design(data), data,
                                scheme="student_blocked", n_folds=3, seed=5,
                                method="L-BFGS-B")
    assert all(f.unseen_column_fraction > 0.0 for f in identified.folds)


# --------------------------------------------------------------------------
# Paired cross-validation
# --------------------------------------------------------------------------


def test_paired_cv_scores_every_model_on_identical_folds():
    data = synthetic(n_students=16, n_kcs=4, n_items=20, seed=41, n_reps=6)
    models = {"afm": build_afm_design(data),
              "compat": build_afm_design(data, learnsphere_compat=True)}
    folds = paired_cross_validate(models, data, n_folds=3, seeds=(0, 1),
                                  method="L-BFGS-B")
    assert len(folds) == 3 * 2 * 2
    sizes = folds.pivot_table(index=["seed", "fold"], columns="model", values="n_test")
    np.testing.assert_array_equal(sizes["afm"].to_numpy(), sizes["compat"].to_numpy())

    contrasts = paired_contrasts(folds, baseline="compat")
    assert list(contrasts["model"]) == ["afm"]
    assert contrasts["n_folds"].iloc[0] == 6


def test_paired_cv_rejects_designs_over_different_rows():
    a = synthetic(n_students=6, n_kcs=3, n_items=12, seed=42, n_reps=4)
    b = synthetic(n_students=6, n_kcs=3, n_items=12, seed=43, n_reps=5)
    with pytest.raises(ValueError, match="different numbers of rows"):
        paired_cross_validate({"a": build_afm_design(a), "b": build_afm_design(b)},
                              a, n_folds=2)


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_paired_scores_reconstruct_repeated_cv(convention):
    """One paired run carries both conventions' per-seed scores exactly.

    With seeded folds the partitions are the same either way, so aggregating
    the paired table must reproduce what :func:`repeated_cross_validate`
    reports — the claim that lets the CLI source its ``cv_rmse`` columns and
    the contrasts from a single set of fits.
    """
    data = synthetic(n_students=14, n_kcs=4, n_items=20, seed=45, n_reps=5)
    design = build_afm_design(data)
    seeds = (0, 1, 2)
    kw = {"scheme": "item_blocked", "n_folds": 3, "convention": convention,
          "method": "L-BFGS-B"}

    scores = paired_scores(
        paired_cross_validate({"m": design}, data, seeds=seeds, **kw), convention)
    repeated = repeated_cross_validate(design, data, seeds=seeds, **kw)

    assert scores["seed"].tolist() == repeated["seed"].tolist()
    np.testing.assert_allclose(scores["rmse"], repeated["rmse"], rtol=1e-12)
    np.testing.assert_allclose(scores["unseen_column_fraction"],
                               repeated["unseen_column_fraction"], rtol=1e-12)
    assert scores["all_converged"].tolist() == repeated["all_converged"].tolist()


def test_paired_scores_reject_an_unknown_convention():
    with pytest.raises(ValueError, match="convention"):
        paired_scores(pd.DataFrame(), "median")
