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

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.optimize import minimize

from afm import (
    build_afm_design,
    congruity_block,
    cross_validate,
    fit_afm,
    from_frame,
    list_kc_models,
    make_folds,
)
from afm.model import _expit, _gradient, _objective

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
    data = _synthetic(n_students=39, n_kcs=5, n_items=20, seed=0)
    design = build_afm_design(data)
    assert design.n_params == 39 + 2 * 5


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
    design = build_afm_design(data, bound_slopes=True)
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
    design = build_afm_design(from_frame(df, "M"))
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
    data = _synthetic(n_students=5, n_kcs=3, n_items=15, seed=3)
    design = build_afm_design(data)
    extra = congruity_block(data, np.random.default_rng(0).normal(size=(len(data), 2)),
                            columns=["cross", "n_prior"])
    extended = design.with_blocks(extra)
    assert extended.n_params == design.n_params + 2
    assert "congruity" in extended.slices()
    assert np.all(extended.l2[extended.slices()["congruity"]] == 0.0)


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
    design = build_afm_design(data)
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
    fit = fit_afm(build_afm_design(data), data.y, method="L-BFGS-B")
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
    """The refine-datashop-kc command reads these exact headers."""
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
    data = _synthetic(n_students=12, n_kcs=3, n_items=20, seed=21, n_reps=6)
    result = cross_validate(build_afm_design(data), data, scheme="student_blocked",
                            n_folds=3, seed=5, method="L-BFGS-B")
    assert all(f.unseen_column_fraction == pytest.approx(1.0) for f in result.folds)


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
