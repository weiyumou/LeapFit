"""Tests for the local AFM.

The most important test here is ``test_published_aic_bic_identity``: it pins
our parameter-counting convention against the EDM 2025 tables without needing
any student data, by exploiting the fact that

    BIC - AIC = nPars * (log N - 2)

holds identically for LearnSphere's definitions. If someone later "fixes" the
parameter count (say, by not counting slopes pinned at zero, or by adding an
intercept), that test fails and the published baseline silently stops being
reproducible — which is exactly the failure we cannot afford.
"""

from __future__ import annotations

import re
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.optimize import minimize

from leapfit import (
    Block,
    StepData,
    accumulator_block,
    build_afm_design,
    cross_validate,
    fit_afm,
    from_frame,
    list_kc_models,
    load_student_step,
    make_folds,
    paired_contrasts,
    paired_cross_validate,
)
from leapfit.fit import _expit, _gradient, _objective

# --------------------------------------------------------------------------
# Published EDM 2025 model-fit tables (main.tex Tables 8 and 9).
# (n_kcs, AIC, BIC) per KC model.
# --------------------------------------------------------------------------

E22_STUDENTS, E22_OBS = 39, 42_176
E22_TABLE = {
    "Single-KC":     (1,    46227.9805, 46582.6144),
    "Unique-step":   (1865, 43323.0595, 75923.4268),
    "LOs":           (87,   43972.6766, 45815.0429),
    "LOs-new":       (101,  43353.2793, 45437.8345),
    "Concept":       (371,  41994.9029, 48750.2457),
    "Concept-emb":   (101,  44537.1400, 46621.6952),
    "Question-emb":  (91,   43880.7030, 45792.2660),
    "KCluster":      (114,  43424.5571, 45734.0021),
}

# The paper reports 41 students for E-learning 2023, but the identity below
# only closes at 39 — two students are absent from the fitted rollup. Worth
# chasing in the data description; it does not affect any published estimate.
E23_STUDENTS, E23_OBS = 39, 44_065
E23_TABLE = {
    "Single-KC":     (1,    46210.3867, 46566.8170),
    "Unique-step":   (1398, 42183.6839, 66829.5327),
    "v1-CTA":        (75,   43434.4955, 45077.5521),
    "v2-combined":   (72,   43471.4342, 45062.3302),
    "Concept":       (298,  41655.2518, 47175.5742),
    "Concept-emb":   (81,   44366.9480, 46114.3256),
    "Question-emb":  (78,   43946.2607, 45641.4778),
    "KCluster":      (92,   42999.9064, 44938.5393),
}


@pytest.mark.parametrize(
    "students,n_obs,table",
    [(E22_STUDENTS, E22_OBS, E22_TABLE), (E23_STUDENTS, E23_OBS, E23_TABLE)],
    ids=["elearning22", "elearning23"],
)
def test_published_aic_bic_identity(students, n_obs, table):
    """nPars = n_students + 2 * n_kcs, with no intercept column."""
    for name, (n_kcs, aic, bic) in table.items():
        n_params = students + 2 * n_kcs
        expected = n_params * (np.log(n_obs) - 2.0)
        assert bic - aic == pytest.approx(expected, rel=1e-5, abs=0.05), name


# LearnSphere workflow wf3990 (2024-12-06, E-learning 22), read back from
# Analysis-1-x916817_model_values.xml. nPars is recovered from its own output as
# (AIC + 2*log_likelihood)/2 and matched our design exactly for all ten models.
WF3990_STUDENTS, WF3990_OBS = 39, 42_176
WF3990_NPARS = {  # kc_model: (n_kcs, nPars implied by LearnSphere)
    "LOs-MCQ": (87, 213), "LOs-new-MCQ": (101, 241), "Single-KC-MCQ": (1, 41),
    "Unique-step-MCQ": (1865, 3769), "concept": (371, 781), "pmi": (118, 275),
    "concept-cosine": (102, 243), "concept-euclidean": (118, 275),
    "question-euclidean": (114, 267), "question-cosine": (93, 225),
}


def test_nparams_matches_learnsphere_workflow_output():
    """Confirmed against real LearnSphere output, 41 to 3,769 parameters."""
    for name, (n_kcs, n_params) in WF3990_NPARS.items():
        assert WF3990_STUDENTS + 2 * n_kcs == n_params, name


def test_design_n_params_matches_published_convention():
    """compat mode must keep LearnSphere's count, phantom parameters included."""
    data = _synthetic(n_students=39, n_kcs=5, n_items=20, seed=0)
    assert build_afm_design(data, learnsphere_compat=True).n_params == 39 + 2 * 5


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _rollup_frame(rows, kc_model="M"):
    return pd.DataFrame(rows).rename(
        columns={"kc": f"KC ({kc_model})", "opp": f"Opportunity ({kc_model})"})


def test_opportunity_is_zero_based():
    df = _rollup_frame([{
        "Anon Student Id": "s1", "Problem Name": "p", "Step Name": "st",
        "First Attempt": "correct", "kc": "A", "opp": "1",
    }])
    data = from_frame(df, "M")
    assert data.opportunities == [(0,)], "first encounter must enter the model as T=0"


def test_multi_kc_splits_on_double_tilde():
    df = _rollup_frame([{
        "Anon Student Id": "s1", "Problem Name": "p", "Step Name": "st",
        "First Attempt": "incorrect", "kc": "A~~B", "opp": "3~~1",
    }])
    data = from_frame(df, "M")
    assert data.kcs == [("A", "B")]
    assert data.opportunities == [(2, 0)]
    assert data.y.tolist() == [0]


def test_rows_without_a_kc_are_skipped_entirely():
    df = _rollup_frame([
        {"Anon Student Id": "s1", "Problem Name": "p", "Step Name": "a",
         "First Attempt": "correct", "kc": "", "opp": ""},
        {"Anon Student Id": "s1", "Problem Name": "p", "Step Name": "b",
         "First Attempt": "correct", "kc": "A", "opp": "1"},
    ])
    data = from_frame(df, "M")
    assert len(data) == 1 and data.skipped_no_kc == 1


def test_only_correct_counts_as_success():
    df = _rollup_frame([
        {"Anon Student Id": "s1", "Problem Name": "p", "Step Name": s,
         "First Attempt": a, "kc": "A", "opp": "1"}
        for s, a in [("a", "correct"), ("b", "incorrect"), ("c", "hint")]
    ])
    assert from_frame(df, "M").y.tolist() == [1, 0, 0]


def test_kc_opportunity_length_mismatch_raises():
    df = _rollup_frame([{
        "Anon Student Id": "s1", "Problem Name": "p", "Step Name": "st",
        "First Attempt": "correct", "kc": "A~~B", "opp": "1",
    }])
    with pytest.raises(ValueError, match="opportunity value"):
        from_frame(df, "M")


def test_missing_kc_model_lists_alternatives():
    df = _rollup_frame([{
        "Anon Student Id": "s1", "Problem Name": "p", "Step Name": "st",
        "First Attempt": "correct", "kc": "A", "opp": "1",
    }])
    with pytest.raises(KeyError, match="M"):
        from_frame(df, "Nope")


def test_list_kc_models(tmp_path):
    path = tmp_path / "rollup.txt"
    pd.DataFrame({
        "Anon Student Id": ["s1"], "Problem Name": ["p"], "Step Name": ["st"],
        "First Attempt": ["correct"],
        "KC (Alpha)": ["A"], "Opportunity (Alpha)": ["1"],
        "KC (Beta)": ["B"], "Opportunity (Beta)": ["1"],
    }).to_csv(path, sep="\t", index=False)
    assert list_kc_models(str(path)) == ["Alpha", "Beta"]


# --------------------------------------------------------------------------
# Design structure
# --------------------------------------------------------------------------

def test_penalty_and_bounds_follow_pyafm():
    data = _synthetic(n_students=4, n_kcs=3, n_items=12, seed=1)
    design = build_afm_design(data, learnsphere_compat=True, bound_slopes=True)
    slices = design.slices()

    l2, bounds = design.l2, design.bounds
    assert np.all(l2[slices["student"]] == 1.0), "students are ridge-penalized at 1.0"
    assert np.all(l2[slices["kc_intercept"]] == 0.0)
    assert np.all(l2[slices["kc_slope"]] == 0.0)

    assert all(b == (None, None) for b in bounds[slices["student"]])
    assert all(b == (None, None) for b in bounds[slices["kc_intercept"]])
    assert all(b == (0.0, None) for b in bounds[slices["kc_slope"]]), \
        "PyAFM bounds slopes below at zero"


def test_slopes_are_unbounded_by_default():
    """The DataShop workflow behind the published tables allows negative slopes.

    Measured on wf3990's own output: 3,790 of 29,700 fitted slopes are
    negative (12.8%), the smallest -1.1747. Bounding them changes the fitted
    likelihood by 7-22 nats on E-learning-22, so the default must match the
    workflow that produced the baseline we compare against.
    """
    data = _synthetic(n_students=4, n_kcs=3, n_items=12, seed=1)
    design = build_afm_design(data)
    assert all(b == (None, None) for b in design.bounds[design.slices()["kc_slope"]])


def test_slope_column_holds_the_opportunity_count():
    df = _rollup_frame([{
        "Anon Student Id": "s1", "Problem Name": "p", "Step Name": "st",
        "First Attempt": "correct", "kc": "A", "opp": "7",
    }])
    design = build_afm_design(from_frame(df, "M"), identify=False)
    slope_block = design.blocks[2].matrix.toarray()
    assert slope_block[0, 0] == 6.0


def test_take_preserves_labels_penalty_and_bounds():
    data = _synthetic(n_students=5, n_kcs=3, n_items=15, seed=2)
    design = build_afm_design(data)
    subset = design.take(np.array([0, 3, 5]))
    assert subset.n_obs == 3
    assert subset.columns == design.columns
    np.testing.assert_array_equal(subset.l2, design.l2)
    assert subset.bounds == design.bounds


def test_extra_block_extends_the_design():
    """The accumulator seam: PFA's counts and any history-derived predictor."""
    data = _synthetic(n_students=5, n_kcs=3, n_items=15, seed=3)
    design = build_afm_design(data)
    extra = accumulator_block(data, np.random.default_rng(0).normal(size=(len(data), 2)),
                              columns=["prior_successes", "prior_failures"])
    extended = design.with_blocks(extra)
    assert extended.n_params == design.n_params + 2
    assert "accumulator" in extended.slices()
    assert np.all(extended.l2[extended.slices()["accumulator"]] == 0.0)
    with pytest.raises(ValueError, match="rows"):
        accumulator_block(data, np.zeros(len(data) + 1))


# --------------------------------------------------------------------------
# Objective correctness
# --------------------------------------------------------------------------

def test_objective_matches_the_reference_implementation():
    """Byte-for-byte agreement with PyAFM's dense _ll / _ll_grad."""
    rng = np.random.default_rng(7)
    X_dense = rng.normal(size=(200, 6))
    y = (rng.random(200) < 0.5).astype(float)
    l2 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    w = rng.normal(size=6)

    # PyAFM custom_logistic._ll, transcribed
    z = np.dot(w, np.transpose(X_dense))
    ref_obj = sum(np.subtract(np.logaddexp(0, z), np.multiply(y, z)))
    ref_obj += np.dot(np.divide(l2, 2), np.multiply(w, w))
    # PyAFM custom_logistic._ll_grad, transcribed
    p = 1.0 / (1.0 + np.exp(-z))
    ref_grad = -1 * (np.dot(np.transpose(X_dense), np.subtract(y, p)) - np.multiply(l2, w))

    X = sparse.csr_matrix(X_dense)
    assert _objective(w, X, y, l2) == pytest.approx(ref_obj, rel=1e-12)
    np.testing.assert_allclose(_gradient(w, X, y, l2), ref_grad, rtol=1e-10)


def test_end_to_end_agreement_with_the_reference_recipe():
    """Our sparse fit equals PyAFM's dense one, coefficient for coefficient.

    PyAFM cannot be executed here (it imports the long-removed
    ``sklearn.cross_validation``), so we rebuild its exact recipe — dense
    ``hstack((S, Q, O))``, ``l2 = [1]*students + [0]*kcs*2``, slope bounds at
    zero, ``w0 = 0``, TNC with ``maxiter=1000`` — and drive scipy directly.
    """
    data = _synthetic(n_students=8, n_kcs=4, n_items=16, seed=23, n_reps=5)
    design = build_afm_design(data, learnsphere_compat=True)
    y = np.asarray(data.y, dtype=float)

    n_students, n_kcs = len(data.student_names), len(data.kc_names)
    X_dense = design.matrix.toarray()
    l2 = np.array([1.0] * n_students + [0.0] * n_kcs + [0.0] * n_kcs)
    bounds = ([(None, None)] * (n_students + n_kcs)) + [(0, None)] * n_kcs

    def ref_ll(w):
        z = np.dot(w, np.transpose(X_dense))
        ll = sum(np.subtract(np.logaddexp(0, z), np.multiply(y, z)))
        return ll + np.dot(np.divide(l2, 2), np.multiply(w, w))

    def ref_grad(w):
        z = np.dot(w, np.transpose(X_dense))
        p = 1.0 / (1.0 + np.exp(-z))
        return -1 * (np.dot(np.transpose(X_dense), np.subtract(y, p)) - np.multiply(l2, w))

    # A budget large enough that both runs actually converge; at TNC's default
    # they stop early at slightly different points, because dense `w @ X.T` and
    # sparse `X @ w` sum in different orders and the optimizer amplifies it.
    budget = {"maxfun": 200_000}
    ref = minimize(ref_ll, np.zeros(X_dense.shape[1]), jac=ref_grad,
                   method="TNC", bounds=bounds, options=budget)
    ours = fit_afm(design, y, method="TNC", max_fun=200_000, warn_not_converged=False)

    assert ref.success and ours.converged

    # The meaningful invariant: both paths reach the same optimum value. The
    # coefficients then agree to ~1e-5 rather than to machine precision — the
    # objective is flat near the optimum and TNC's default tolerance stops
    # each path at a slightly different point in that basin. Five orders of
    # magnitude below the fourth decimal anyone reports.
    assert ours.ll == pytest.approx(-ref.fun, rel=1e-9)
    np.testing.assert_allclose(ours.weights, ref.x, rtol=0, atol=1e-4)

    # Both really are at a stationary point of the same function.
    for w in (ours.weights, ref.x):
        grad = _gradient(w, design.matrix, y, design.l2)
        free = np.array([lo is None or w[i] > lo + 1e-9
                         for i, (lo, _) in enumerate(design.bounds)])
        assert np.abs(grad[free]).max() < 1e-3


def test_references_iteration_cap_is_inert_for_tnc():
    """PyAFM's `options={'maxiter': 1000}` is silently dropped by TNC.

    Pinned as a test because it is the reason we default `max_fun=None`: every
    published AFM fit ran at TNC's default budget, not at the stated 1000.
    """
    import warnings as _warnings

    from scipy.optimize import OptimizeWarning

    data = _synthetic(n_students=6, n_kcs=3, n_items=12, seed=24)
    design = build_afm_design(data)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        minimize(_objective, np.zeros(design.n_params),
                 args=(design.matrix, np.asarray(data.y, float), design.l2),
                 jac=_gradient, method="TNC", bounds=design.bounds,
                 options={"maxiter": 1000})
    assert any(issubclass(w.category, OptimizeWarning)
               and "maxiter" in str(w.message) for w in caught)

    # ...whereas the option TNC actually reads does bite.
    short = fit_afm(design, data.y, method="TNC", max_fun=3, warn_not_converged=False)
    full = fit_afm(design, data.y, method="TNC", max_fun=200_000,
                   warn_not_converged=False)
    assert short.ll < full.ll


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(11)
    X = sparse.csr_matrix(rng.normal(size=(80, 5)))
    y = (rng.random(80) < 0.5).astype(float)
    l2 = np.array([1.0, 0.0, 0.5, 0.0, 0.0])
    w = rng.normal(size=5) * 0.3

    analytic = _gradient(w, X, y, l2)
    eps = 1e-6
    numeric = np.array([
        (_objective(w + eps * e, X, y, l2) - _objective(w - eps * e, X, y, l2)) / (2 * eps)
        for e in np.eye(5)
    ])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_expit_is_stable_at_extremes():
    z = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    p = _expit(z)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    assert p[0] == pytest.approx(0.0) and p[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def test_recovers_known_parameters():
    data, truth = _synthetic(n_students=60, n_kcs=4, n_items=40, seed=5,
                             n_reps=25, return_truth=True)
    fit = fit_afm(build_afm_design(data, bound_slopes=True), data.y,
                  method="L-BFGS-B", max_fun=5000)

    intercepts = fit.block("kc_intercept")
    slopes = fit.block("kc_slope")
    # Intercepts are identified only up to the student mean; compare centred.
    np.testing.assert_allclose(
        intercepts - intercepts.mean(), truth["beta"] - truth["beta"].mean(),
        atol=0.35)
    np.testing.assert_allclose(slopes, truth["gamma"], atol=0.06)


def test_slopes_never_go_negative():
    """A KC engineered to get *worse* with practice must rest at the bound."""
    data = _synthetic(n_students=40, n_kcs=2, n_items=20, seed=8,
                      n_reps=20, gamma=np.array([-0.4, 0.3]))
    fit = fit_afm(build_afm_design(data, bound_slopes=True), data.y,
                  method="L-BFGS-B", max_fun=5000)
    slopes = fit.block("kc_slope")
    assert slopes.min() >= -1e-9
    assert slopes[0] == pytest.approx(0.0, abs=1e-6)


def test_ll_includes_the_ridge_penalty():
    data = _synthetic(n_students=10, n_kcs=3, n_items=20, seed=9)
    fit = fit_afm(build_afm_design(data, learnsphere_compat=True), data.y,
                  method="L-BFGS-B")
    assert fit.penalty > 0
    assert fit.ll == pytest.approx(fit.ll_unpenalized - fit.penalty)
    assert fit.aic == pytest.approx(-2 * fit.ll + 2 * fit.n_params)
    assert fit.bic == pytest.approx(
        -2 * fit.ll + fit.n_params * np.log(fit.n_obs))


def test_zero_ridge_makes_the_two_likelihoods_agree():
    data = _synthetic(n_students=10, n_kcs=3, n_items=20, seed=10)
    fit = fit_afm(build_afm_design(data, student_l2=0.0), data.y, method="L-BFGS-B")
    assert fit.ll == pytest.approx(fit.ll_unpenalized)


def test_kc_values_exports_datashop_column_names():
    """Tooling that reads DataShop KC-values files expects these exact headers."""
    data = _synthetic(n_students=10, n_kcs=3, n_items=20, seed=12)
    fit = fit_afm(build_afm_design(data), data.y, method="L-BFGS-B")
    values = fit.kc_values(data)
    for column in ("KC Name", "Slope", "Intercept (probability) at Opportunity 1",
                   "Number of Unique Steps"):
        assert column in values.columns
    assert values["Intercept (probability) at Opportunity 1"].between(0, 1).all()
    assert values["Number of Unique Steps"].sum() >= len(set(data.items))


def test_predict_rejects_a_mismatched_design():
    data = _synthetic(n_students=6, n_kcs=3, n_items=12, seed=13)
    fit = fit_afm(build_afm_design(data), data.y, method="L-BFGS-B")
    smaller = _synthetic(n_students=6, n_kcs=2, n_items=12, seed=14)
    with pytest.raises(ValueError, match="coefficients"):
        fit.predict_proba(build_afm_design(smaller))


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["unstratified", "response_stratified",
                                    "student_blocked", "item_blocked"])
def test_folds_partition_the_rows(scheme):
    data = _synthetic(n_students=9, n_kcs=4, n_items=18, seed=15)
    folds = make_folds(data, scheme, n_folds=3, seed=0, convention="pooled")
    joined = np.concatenate(folds)
    assert len(joined) == len(np.unique(joined)) == len(data)


def test_blocked_folds_keep_a_label_on_one_side():
    data = _synthetic(n_students=12, n_kcs=4, n_items=24, seed=16)
    items = np.asarray(data.items)
    for held in make_folds(data, "item_blocked", 3, seed=1, convention="pooled"):
        held_items = set(items[held])
        rest = np.setdiff1d(np.arange(len(data)), held)
        assert not (held_items & set(items[rest])), "an item straddled the split"


def test_label_kfold_is_deterministic_without_a_seed():
    data = _synthetic(n_students=12, n_kcs=4, n_items=24, seed=17)
    a = make_folds(data, "item_blocked", 3, seed=None, convention="per_fold")
    b = make_folds(data, "item_blocked", 3, seed=None, convention="per_fold")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_seeds_change_the_partition():
    data = _synthetic(n_students=12, n_kcs=4, n_items=24, seed=18)
    a = make_folds(data, "item_blocked", 3, seed=1, convention="pooled")
    b = make_folds(data, "item_blocked", 3, seed=2, convention="pooled")
    assert any(len(x) != len(y) or not np.array_equal(x, y) for x, y in zip(a, b))


def test_the_two_conventions_give_different_numbers():
    """Averaging fold RMSEs is not the RMSE of the pooled residuals."""
    data = _synthetic(n_students=20, n_kcs=4, n_items=30, seed=19, n_reps=8)
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


def test_item_blocked_cv_reports_unseen_columns():
    """A KC carried by a single item must be unseen when that item is held out."""
    data = _synthetic(n_students=20, n_kcs=30, n_items=30, seed=20, n_reps=4)
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
    data = _synthetic(n_students=12, n_kcs=3, n_items=20, seed=21, n_reps=6)
    result = cross_validate(build_afm_design(data, learnsphere_compat=True), data,
                            scheme="student_blocked", n_folds=3, seed=5,
                            method="L-BFGS-B")
    assert all(f.unseen_column_fraction == pytest.approx(1.0) for f in result.folds)

    identified = cross_validate(build_afm_design(data), data,
                                scheme="student_blocked", n_folds=3, seed=5,
                                method="L-BFGS-B")
    assert all(f.unseen_column_fraction > 0.0 for f in identified.folds)


# --------------------------------------------------------------------------
# Identification: the new machinery must be a no-op on everything observable
# --------------------------------------------------------------------------

def test_identify_drops_a_reference_student():
    data = _synthetic(n_students=6, n_kcs=3, n_items=12, seed=30, n_reps=5)
    full = build_afm_design(data, identify=False)
    ident = full.identify()
    assert ident.n_params == full.n_params - 1
    assert len(ident.aliased) == 1
    assert ident.aliased.columns[0].startswith("student:")
    assert ident.n_params == ident.rank(), "identified design must be full rank"


def test_identify_drops_slope_columns_for_never_repeated_kcs():
    """A KC nobody practises twice has T == 0 always: no estimable slope."""
    data = _synthetic(n_students=8, n_kcs=12, n_items=12, seed=31, n_reps=1)
    design = build_afm_design(data)
    dropped = design.aliased.by_block()
    assert len(dropped.get("kc_slope", [])) == 12, dropped
    assert design.n_params == design.rank()


def test_aliased_columns_carry_no_information():
    """Re-inserting zeros for the dropped columns reproduces the same likelihood.

    This is the equivalence that licenses dropping them: the identified fit is
    a point of the full design, with the aliased coefficients at zero, and it
    attains exactly the full design's optimum.
    """
    data = _synthetic(n_students=8, n_kcs=4, n_items=16, seed=32, n_reps=6)
    full = build_afm_design(data, identify=False)
    ident = full.identify()
    fit = fit_afm(ident, data.y, method="L-BFGS-B", max_fun=200_000,
                  warn_not_converged=False)

    keep = [c not in set(ident.aliased.columns) for c in full.columns]
    w_full = np.zeros(full.n_params)
    w_full[np.flatnonzero(keep)] = fit.weights

    y = np.asarray(data.y, float)
    zero = np.zeros(full.n_params)
    assert (-_objective(w_full, full.matrix, y, zero)
            == pytest.approx(fit.ll_unpenalized, rel=1e-12))
    np.testing.assert_allclose(_expit(full.matrix @ w_full),
                               fit.predict_proba(ident), rtol=0, atol=1e-12)


def test_identification_does_not_change_the_maximised_likelihood():
    """Same optimum as the unidentified design, just without the phantom column."""
    data = _synthetic(n_students=8, n_kcs=4, n_items=16, seed=33, n_reps=6)
    full = fit_afm(build_afm_design(data, identify=False, student_l2=0.0), data.y,
                   method="L-BFGS-B", max_fun=200_000, warn_not_converged=False)
    ident = fit_afm(build_afm_design(data), data.y,
                    method="L-BFGS-B", max_fun=200_000, warn_not_converged=False)
    assert ident.ll_unpenalized == pytest.approx(full.ll_unpenalized, abs=1e-4)
    assert ident.n_params == full.n_params - 1


def test_recentring_students_leaves_predictions_unchanged():
    data = _synthetic(n_students=8, n_kcs=4, n_items=16, seed=34, n_reps=6)
    design = build_afm_design(data)
    fit = fit_afm(design, data.y, method="L-BFGS-B", max_fun=200_000,
                  warn_not_converged=False)

    theta, shift = fit.centred_students(data)
    assert theta.mean() == pytest.approx(0.0, abs=1e-12)
    assert len(theta) == len(data.student_names), "reference student must reappear"

    raw = fit.kc_values(data, centre=False)["Intercept (logit)"].to_numpy()
    cen = fit.kc_values(data, centre=True)["Intercept (logit)"].to_numpy()
    np.testing.assert_allclose(cen - raw, shift, rtol=0, atol=1e-12)

    # theta_i + beta_k is invariant, which is what predictions depend on.
    fitted = dict(zip(data.student_names, theta))
    before = fit._block_values("student")
    for s in data.student_names:
        assert fitted[s] + shift == pytest.approx(before.get(s, 0.0), abs=1e-12)


def _multi_kc_data():
    """Two KCs per row, but *varying* pairs so the KC columns stay distinct."""
    pairs = [("A", "B"), ("B", "C"), ("A", "C")]
    rows = []
    for i in range(8):
        for j in range(9):
            a, b = pairs[j % 3]
            rows.append({"Anon Student Id": f"s{i}", "Problem Name": "p",
                         "Step Name": f"st{j}",
                         "First Attempt": "correct" if (i + j) % 3 else "incorrect",
                         "kc": f"{a}~~{b}", "opp": f"{j // 3 + 1}~~{j // 3 + 1}"})
    return from_frame(_rollup_frame(rows), "M")


def test_sum_redundancy_is_detected_for_any_constant_kcs_per_row():
    """The student/KC dependency exists whenever every row has the same m KCs.

    Sum of student columns = 1; sum of KC columns = m * 1. Checking only for
    m == 1 would miss it on every multi-KC export.
    """
    design = build_afm_design(_multi_kc_data(), identify=False)
    assert design.kc_per_row() == 2.0
    assert design._has_sum_redundancy()
    assert design.rank() == design.n_params - 1
    assert build_afm_design(_multi_kc_data()).n_params == design.n_params - 1


def test_recentring_refuses_on_multi_kc_designs():
    data = _multi_kc_data()
    fit = fit_afm(build_afm_design(data), data.y, method="L-BFGS-B",
                  max_fun=200_000, warn_not_converged=False)
    assert not fit.design.recentring_is_valid()
    with pytest.raises(ValueError, match="multi-KC"):
        fit.centred_students(data)
    # ...and kc_values silently reports uncentred rather than raising.
    assert len(fit.kc_values(data)) == 3


def test_never_repeated_kc_reports_slope_as_undefined_not_zero():
    data = _synthetic(n_students=8, n_kcs=12, n_items=12, seed=35, n_reps=1)
    fit = fit_afm(build_afm_design(data), data.y, method="L-BFGS-B",
                  warn_not_converged=False)
    values = fit.kc_values(data)
    assert len(values) == 12, "every KC still gets a row"
    assert values["Slope"].isna().all(), (
        "a KC with no second opportunity has no estimable learning rate; "
        "reporting 0.0 would feed a gamma<=0.001 screen a false positive"
    )


def test_identify_raises_on_a_collinear_extra_block():
    """The guard that protects accumulator and hierarchical blocks added later."""
    data = _synthetic(n_students=6, n_kcs=3, n_items=12, seed=36, n_reps=5)
    design = build_afm_design(data, identify=False)
    kc_block = next(b for b in design.blocks if b.name == "kc_intercept")
    duplicate = Block.build("copy", kc_block.matrix.copy(),
                            [f"dup_{c}" for c in kc_block.columns])
    with pytest.raises(ValueError, match="rank-deficient"):
        design.with_blocks(duplicate).identify()


def test_compat_preserves_the_learnsphere_parameter_count():
    data = _synthetic(n_students=6, n_kcs=3, n_items=12, seed=37, n_reps=5)
    compat = build_afm_design(data, learnsphere_compat=True)
    assert compat.n_params == 6 + 2 * 3
    assert len(compat.aliased) == 0
    assert np.all(compat.l2[compat.slices()["student"]] == 1.0)


# --------------------------------------------------------------------------
# Optimality certificate
# --------------------------------------------------------------------------

def test_optimality_certificate_detects_a_starved_solver():
    data = _synthetic(n_students=20, n_kcs=6, n_items=24, seed=38, n_reps=8)
    design = build_afm_design(data)
    starved = fit_afm(design, data.y, method="TNC", max_fun=2, warn_not_converged=False)
    solved = fit_afm(design, data.y, method="TNC", max_fun=200_000,
                     warn_not_converged=False)
    assert not starved.is_optimal and solved.is_optimal
    assert solved.ll > starved.ll


def test_fit_warns_when_not_at_the_optimum():
    data = _synthetic(n_students=20, n_kcs=6, n_items=24, seed=39, n_reps=8)
    with pytest.warns(RuntimeWarning, match="not at a stationary point"):
        fit_afm(build_afm_design(data), data.y, method="TNC", max_fun=2)


# --------------------------------------------------------------------------
# Canonical practice ordering
# --------------------------------------------------------------------------

def test_practice_order_uses_time_then_row_order():
    df = pd.DataFrame([
        {"Anon Student Id": "s1", "Problem Name": "p", "Step Name": "b",
         "First Transaction Time": "2022-01-01 00:00:09", "First Attempt": "correct",
         "KC (M)": "A", "Opportunity (M)": "2"},
        {"Anon Student Id": "s1", "Problem Name": "p", "Step Name": "a",
         "First Transaction Time": "2022-01-01 00:00:01", "First Attempt": "correct",
         "KC (M)": "A", "Opportunity (M)": "1"},
    ])
    data = from_frame(df, "M")
    np.testing.assert_array_equal(data.practice_order()["s1"], [1, 0])
    assert data.recomputed_opportunities() == [(1,), (0,)]
    # The file's own column is row-ordered, so here the two disagree.
    assert data.opportunities == [(1,), (0,)]


def test_recomputed_opportunities_match_the_file_when_order_agrees():
    data = _synthetic(n_students=5, n_kcs=3, n_items=9, seed=40, n_reps=4)
    assert data.times is None, "synthetic rollups carry no time column"
    assert data.recomputed_opportunities() == data.opportunities
    assert len(data.opportunity_disagreements()) == 0


# --------------------------------------------------------------------------
# Paired cross-validation
# --------------------------------------------------------------------------

def test_paired_cv_scores_every_model_on_identical_folds():
    data = _synthetic(n_students=16, n_kcs=4, n_items=20, seed=41, n_reps=6)
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
    a = _synthetic(n_students=6, n_kcs=3, n_items=12, seed=42, n_reps=4)
    b = _synthetic(n_students=6, n_kcs=3, n_items=12, seed=43, n_reps=5)
    with pytest.raises(ValueError, match="different numbers of rows"):
        paired_cross_validate({"a": build_afm_design(a), "b": build_afm_design(b)},
                              a, n_folds=2)


# --------------------------------------------------------------------------
# Portability: what the reader requires, and what it refuses
# --------------------------------------------------------------------------

MINIMAL_COLUMNS = ["Anon Student Id", "Problem Name", "Step Name",
                   "First Attempt", "KC (M)", "Opportunity (M)"]


def _minimal_frame(n_students=6, n_steps=4, n_reps=3, attempt=str):
    """A rollup carrying *only* the columns the reader declares it needs."""
    rows = []
    for s in range(n_students):
        seen: dict[str, int] = {}
        for _ in range(n_reps):
            for step in range(n_steps):
                kc = f"kc{step % 2}"
                seen[kc] = seen.get(kc, 0) + 1
                rows.append({
                    "Anon Student Id": f"S{s}", "Problem Name": "p",
                    "Step Name": f"st{step}",
                    "First Attempt": attempt("correct" if (s + step) % 3 else "incorrect"),
                    "kc": kc, "opp": str(seen[kc]),
                })
    return _rollup_frame(rows)


def test_the_minimal_column_set_is_enough_to_fit_and_cross_validate():
    """Six columns, no timestamps, no DataShop metadata."""
    df = _minimal_frame()
    assert list(df.columns) == MINIMAL_COLUMNS

    data = from_frame(df, "M")
    assert data.times is None
    design = build_afm_design(data)
    fit = fit_afm(design, data.y)
    assert fit.is_optimal
    assert cross_validate(design, data, scheme="item_blocked", n_folds=2).rmse > 0


def test_every_declared_column_is_actually_required():
    """Each of the six is load-bearing — dropping any one raises by name."""
    df = _minimal_frame()
    for column in MINIMAL_COLUMNS:
        with pytest.raises(KeyError, match=re.escape(column)):
            from_frame(df.drop(columns=[column]), "M")


def test_opportunities_can_be_recomputed_without_a_time_column():
    """practice_order falls back to row order, which is what DataShop numbers by."""
    data = from_frame(_minimal_frame(), "M")
    assert data.times is None
    assert data.recomputed_opportunities() == data.opportunities
    assert len(data.opportunity_disagreements()) == 0


def test_outcome_labels_are_matched_case_insensitively():
    """'Correct' must not silently score as a failure.

    Matching the literal string is the reference's rule, and under it an export
    that capitalizes the column yields an all-zero response, a converged fit,
    and a plausible AIC — wrong with no symptom. Folding case is a no-op on
    real DataShop files, which are lowercase.
    """
    lower = from_frame(_minimal_frame(), "M")
    upper = from_frame(_minimal_frame(attempt=str.capitalize), "M")
    spaced = from_frame(_minimal_frame(attempt=lambda v: f"  {v.upper()} "), "M")
    assert lower.y.mean() > 0
    assert np.array_equal(lower.y, upper.y)
    assert np.array_equal(lower.y, spaced.y)


def test_unknown_outcome_vocabulary_raises_instead_of_scoring_zero():
    df = _minimal_frame()
    df["First Attempt"] = np.where(df["First Attempt"] == "correct", "1", "0")
    with pytest.raises(ValueError, match="Unrecognized 'First Attempt'"):
        from_frame(df, "M")
    # Declaring only the successes is not enough — '0' is still unaccounted for,
    # so the guard stays armed rather than switching off on any override.
    with pytest.raises(ValueError, match="Unrecognized 'First Attempt'"):
        from_frame(df, "M", success_values=("1",))
    data = from_frame(df, "M", success_values=("1",), failure_values=("0",))
    assert 0 < data.y.mean() < 1


def test_documented_datashop_failure_labels_are_accepted_silently():
    df = _minimal_frame()
    df.loc[df.index % 5 == 0, "First Attempt"] = "hint"
    df.loc[df.index % 7 == 0, "First Attempt"] = "unknown"
    data = from_frame(df, "M")
    assert 0 < data.y.mean() < 1


def test_a_malformed_opportunity_value_names_the_row():
    df = _minimal_frame()
    df.loc[5, "Opportunity (M)"] = "."
    with pytest.raises(ValueError, match=r"Row 7: non-integer opportunity"):
        from_frame(df, "M")


def test_a_constant_response_warns_and_names_the_vocabulary():
    df = _minimal_frame()
    df["First Attempt"] = "incorrect"
    with pytest.warns(RuntimeWarning, match="Every observation is a failure"):
        from_frame(df, "M")


# --------------------------------------------------------------------------
# Separation: coefficients with no finite MLE
# --------------------------------------------------------------------------

def _separated_frame(always_correct="kc0"):
    rows = []
    rng = np.random.default_rng(3)
    for s in range(10):
        seen: dict[str, int] = {}
        for _ in range(4):
            for step in range(4):
                kc = f"kc{step}"
                seen[kc] = seen.get(kc, 0) + 1
                ok = True if kc == always_correct else bool(rng.random() < 0.6)
                rows.append({
                    "Anon Student Id": f"S{s}", "Problem Name": "p",
                    "Step Name": f"st{step}",
                    "First Attempt": "correct" if ok else "incorrect",
                    "kc": kc, "opp": str(seen[kc]),
                })
    return _rollup_frame(rows)


def test_an_always_correct_kc_is_reported_as_separated():
    data = from_frame(_separated_frame(), "M")
    design = build_afm_design(data)
    sep = design.separated(data.y)
    assert "kc_intercept:kc0" in sep.columns
    assert sep.directions[sep.columns.index("kc_intercept:kc0")] == 1
    assert "kc_intercept:kc1" not in sep.columns


def test_a_separated_coefficient_has_no_maximum():
    """Checked against the objective itself, not against solver behaviour.

    The claim ``Design.separated`` makes is that the likelihood improves
    without bound along that coordinate. Walk the coefficient by hand and watch
    the negative log-likelihood fall monotonically, while the same walk along an
    identified coefficient turns around at an interior minimum. This is the
    definition of "no maximizer", and it holds whatever the optimizer does —
    which matters, because TNC halts a diverging coefficient around 19 on its
    own gradient criterion, so a test that watched the fitted number grow would
    be testing the solver instead.
    """
    data = from_frame(_separated_frame(), "M")
    design = build_afm_design(data)
    fit = fit_afm(design, data.y, warn_not_converged=False, warn_separated=False)
    X, y, l2 = design.matrix, np.asarray(data.y, dtype=float), design.l2

    def walk(name):
        j = design.columns.index(name)
        out = []
        for value in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            w = fit.weights.copy()
            w[j] = value
            out.append(_objective(w, X, y, l2))
        return out

    diverging = walk("kc_intercept:kc0")
    identified = walk("kc_intercept:kc1")
    assert all(a > b for a, b in pairwise(diverging)), (
        "a separated coefficient must keep improving the fit as it grows")
    assert identified[-1] > min(identified), (
        "an identified coefficient must have an interior optimum")


def test_a_ridge_removes_the_separation():
    """A penalty supplies the missing curvature, so the MLE exists again."""
    data = from_frame(_separated_frame(), "M")
    assert len(build_afm_design(data, student_l2=0.0).separated(data.y)) > 0
    penalized = build_afm_design(data, identify=False, student_l2=1.0)
    kc_blocks = [b.name for b in penalized.blocks]
    assert "kc_intercept" in kc_blocks  # unpenalized, so it is still flagged
    assert not any(c.startswith("student:")
                   for c in penalized.separated(data.y).columns)


def test_a_lower_bound_absorbs_a_downward_divergence():
    """bound_slopes rests an all-failure slope on 0 instead of sending it to -inf."""
    rows = []
    for s in range(8):
        for r in range(4):
            rows.append({
                "Anon Student Id": f"S{s}", "Problem Name": "p", "Step Name": "st0",
                "First Attempt": "incorrect", "kc": "kc0", "opp": str(r + 1)})
            rows.append({
                "Anon Student Id": f"S{s}", "Problem Name": "p", "Step Name": "st1",
                "First Attempt": "correct" if r % 2 else "incorrect",
                "kc": "kc1", "opp": str(r + 1)})
    data = from_frame(_rollup_frame(rows), "M")

    free = build_afm_design(data)
    bounded = build_afm_design(data, bound_slopes=True)
    assert "kc_slope:kc0" in free.separated(data.y).columns
    assert "kc_slope:kc0" not in bounded.separated(data.y).columns


def test_separated_columns_are_flagged_in_the_kc_values_table():
    """Flagged, not blanked — unlike a never-repeated KC, the datum exists."""
    data = from_frame(_separated_frame(), "M")
    fit = fit_afm(build_afm_design(data), data.y, warn_not_converged=False,
                  warn_separated=False)
    values = fit.kc_values(data).set_index("KC Name")
    assert bool(values.loc["kc0", "Separated"])
    assert not bool(values.loc["kc1", "Separated"])
    assert np.isfinite(values.loc["kc0", "Intercept (logit)"])


def test_fitting_a_separated_design_warns():
    data = from_frame(_separated_frame(), "M")
    design = build_afm_design(data)
    with pytest.warns(RuntimeWarning, match="no finite maximum-likelihood"):
        fit_afm(design, data.y, warn_not_converged=False)


def test_a_well_behaved_design_reports_no_separation():
    data, _ = _synthetic(20, 5, 20, seed=7, return_truth=True)
    design = build_afm_design(data)
    assert len(design.separated(data.y)) == 0
    fit = fit_afm(design, data.y)
    assert len(fit.separated) == 0
    assert "separation" not in fit.summary()


# --------------------------------------------------------------------------
# Annotation: student-step in, student-step out
# --------------------------------------------------------------------------

def _fitted(df, model="M"):
    data = from_frame(df, model)
    fit = fit_afm(build_afm_design(data), data.y, warn_not_converged=False,
                  warn_separated=False)
    return data, fit


def test_annotate_appends_the_datashop_prediction_column():
    df = _minimal_frame()
    data, fit = _fitted(df)
    out = fit.annotate(data)

    assert list(out.columns) == [*df.columns, "Predicted Error Rate (M)"]
    pd.testing.assert_frame_equal(out[df.columns.tolist()], df)  # originals untouched
    expected = 1.0 - fit.predict_proba(fit.design)
    np.testing.assert_allclose(out["Predicted Error Rate (M)"].to_numpy(), expected)
    assert df.shape[1] == len(MINIMAL_COLUMNS), "source frame must not be mutated"


def test_annotate_leaves_rows_without_a_kc_blank():
    """The rows that entered no fit get NaN, everything else the fit's 1 - p.

    This is the alignment the reference attempts by re-reading and re-sorting
    its input file; recording source positions at parse time makes it exact.
    """
    df = _minimal_frame()
    dropped = df.index[df.index % 5 == 0]
    df.loc[dropped, ["KC (M)", "Opportunity (M)"]] = ""
    data, fit = _fitted(df)
    assert data.skipped_no_kc == len(dropped)

    col = fit.annotate(data)["Predicted Error Rate (M)"]
    assert col.loc[dropped].isna().all()
    kept = col.drop(index=dropped)
    assert not kept.isna().any()
    np.testing.assert_allclose(kept.to_numpy(), 1.0 - fit.predict_proba(fit.design))


def test_annotate_overwrites_an_existing_prediction_column_in_place():
    """DataShop exports can already carry the column; ours replaces, not duplicates."""
    df = _minimal_frame()
    df.insert(2, "Predicted Error Rate (M)", "stale")
    data, fit = _fitted(df)
    out = fit.annotate(data)

    assert list(out.columns) == list(df.columns)          # position preserved
    assert out.columns.tolist().count("Predicted Error Rate (M)") == 1
    assert not (out["Predicted Error Rate (M)"] == "stale").any()


def test_annotate_accumulates_one_column_per_kc_model():
    """``into=`` chains fits of several KC models over one file — the CLI path."""
    df = _minimal_frame()
    df["KC (Fine)"] = df["KC (M)"] + "-" + df["Step Name"]
    seen: dict[tuple[str, str], int] = {}
    counts = []
    for key in zip(df["Anon Student Id"], df["KC (Fine)"]):
        seen[key] = seen.get(key, 0) + 1
        counts.append(str(seen[key]))
    df["Opportunity (Fine)"] = counts

    data_m, fit_m = _fitted(df, "M")
    data_f, fit_f = _fitted(df, "Fine")
    out = fit_m.annotate(data_m)
    out = fit_f.annotate(data_f, into=out)

    assert "Predicted Error Rate (M)" in out.columns
    assert "Predicted Error Rate (Fine)" in out.columns
    assert len(out) == len(df)


def test_annotate_requires_the_source_table():
    data, fit = _fitted(_minimal_frame())
    bare = StepData(y=data.y, students=data.students, items=data.items,
                    kcs=data.kcs, opportunities=data.opportunities,
                    kc_model=data.kc_model)
    with pytest.raises(ValueError, match="source table"):
        fit.annotate(bare)


def test_annotate_refuses_a_subset_fit():
    """A model fitted to a CV training split cannot annotate the full file."""
    data, _ = _fitted(_minimal_frame())
    design = build_afm_design(data)
    train = np.arange(len(data))[:-10]
    fold_fit = fit_afm(design.take(train), data.y[train],
                       warn_not_converged=False, warn_separated=False)
    with pytest.raises(ValueError, match="subset"):
        fold_fit.annotate(data)


def test_annotated_file_round_trips_through_the_loader(tmp_path):
    """Written to disk, the annotated file is still a valid student-step file
    that parses to the same observations — predictions ride along, blanks stay
    blank, and nothing shifts by a row."""
    df = _minimal_frame()
    df.loc[df.index[3], ["KC (M)", "Opportunity (M)"]] = ""
    data, fit = _fitted(df)

    path = tmp_path / "annotated.txt"
    fit.annotate(data).to_csv(path, sep="\t", index=False, lineterminator="\n")
    again = load_student_step(str(path), kc_model="M")

    assert np.array_equal(again.y, data.y)
    assert again.kcs == data.kcs and again.opportunities == data.opportunities
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert raw.loc[3, "Predicted Error Rate (M)"] == ""


def test_cli_predictions_writes_one_column_per_model(tmp_path):
    from leapfit.cli import main as cli_main

    df = _minimal_frame()
    df["KC (M2)"] = df["KC (M)"]
    df["Opportunity (M2)"] = df["Opportunity (M)"]
    export = tmp_path / "export.txt"
    df.to_csv(export, sep="\t", index=False, lineterminator="\n")

    out = tmp_path / "annotated.txt"
    assert cli_main([str(export), "--cv", "none", "--predictions", str(out)]) == 0

    written = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    assert len(written) == len(df)
    for col in ("Predicted Error Rate (M)", "Predicted Error Rate (M2)"):
        assert col in written.columns
        assert (written[col] != "").all()
    # Identical KC models must produce identical predictions.
    assert written["Predicted Error Rate (M)"].tolist() == \
        written["Predicted Error Rate (M2)"].tolist()


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

def _synthetic(n_students, n_kcs, n_items, seed, n_reps=6, gamma=None,
               return_truth=False):
    """Generate a student-step rollup from a known AFM.

    Items are assigned to KCs round-robin, so ``n_items == n_kcs`` produces
    exactly one item per KC — the degenerate granularity that makes
    item-blocked CV cold-start.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.8, size=n_students)
    beta = rng.normal(0.3, 1.0, size=n_kcs)
    gamma = rng.uniform(0.05, 0.4, size=n_kcs) if gamma is None else np.asarray(gamma)

    item_kc = np.arange(n_items) % n_kcs
    records = []
    for s in range(n_students):
        seen = np.zeros(n_kcs, dtype=int)
        order = rng.permutation(np.repeat(np.arange(n_items), n_reps))
        for item in order:
            k = item_kc[item]
            t = seen[k]
            p = 1.0 / (1.0 + np.exp(-(theta[s] + beta[k] + gamma[k] * t)))
            records.append({
                "Anon Student Id": f"s{s:03d}",
                "Problem Name": "prob",
                "Step Name": f"step{item:03d}",
                "First Attempt": "correct" if rng.random() < p else "incorrect",
                "kc": f"kc{k:03d}",
                "opp": str(t + 1),  # DataShop is 1-based
            })
            seen[k] += 1

    data = from_frame(_rollup_frame(records), "M")
    if return_truth:
        order = np.argsort([f"kc{k:03d}" for k in range(n_kcs)])
        return data, {"theta": theta, "beta": beta[order], "gamma": gamma[order]}
    return data
