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
import pytest
from scipy.optimize import minimize

from leapfit import build_afm_design, fit_afm, from_frame
from leapfit.fit import _gradient

from helpers import co_occurring_kc_data, rollup, separated_frame, synthetic

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
    data = synthetic(n_students=39, n_kcs=5, n_items=20, seed=0)
    assert build_afm_design(data, learnsphere_compat=True).n_params == 39 + 2 * 5


# --------------------------------------------------------------------------
# Design structure
# --------------------------------------------------------------------------

def test_penalty_and_bounds_follow_pyafm():
    data = synthetic(n_students=4, n_kcs=3, n_items=12, seed=1)
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
    data = synthetic(n_students=4, n_kcs=3, n_items=12, seed=1)
    design = build_afm_design(data)
    assert all(b == (None, None) for b in design.bounds[design.slices()["kc_slope"]])


def test_slope_column_holds_the_opportunity_count():
    df = rollup([{
        "Anon Student Id": "s1", "Problem Name": "p", "Step Name": "st",
        "First Attempt": "correct", "kc": "A", "opp": "7",
    }])
    design = build_afm_design(from_frame(df, "M"), identify=False)
    slope_block = design.blocks[2].matrix.toarray()
    assert slope_block[0, 0] == 6.0


# --------------------------------------------------------------------------
# Agreement with PyAFM's own recipe
# --------------------------------------------------------------------------

def test_end_to_end_agreement_with_the_reference_recipe():
    """Our sparse fit equals PyAFM's dense one, coefficient for coefficient.

    PyAFM cannot be executed here (it imports the long-removed
    ``sklearn.cross_validation``), so we rebuild its exact recipe — dense
    ``hstack((S, Q, O))``, ``l2 = [1]*students + [0]*kcs*2``, slope bounds at
    zero, ``w0 = 0``, TNC with ``maxiter=1000`` — and drive scipy directly.
    """
    data = synthetic(n_students=8, n_kcs=4, n_items=16, seed=23, n_reps=5)
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


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def test_recovers_known_parameters():
    data, truth = synthetic(n_students=60, n_kcs=4, n_items=40, seed=5,
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
    data = synthetic(n_students=40, n_kcs=2, n_items=20, seed=8,
                     n_reps=20, gamma=np.array([-0.4, 0.3]))
    fit = fit_afm(build_afm_design(data, bound_slopes=True), data.y,
                  method="L-BFGS-B", max_fun=5000)
    slopes = fit.block("kc_slope")
    assert slopes.min() >= -1e-9
    assert slopes[0] == pytest.approx(0.0, abs=1e-6)


def test_kc_values_exports_datashop_column_names():
    """Tooling that reads DataShop KC-values files expects these exact headers."""
    data = synthetic(n_students=10, n_kcs=3, n_items=20, seed=12)
    fit = fit_afm(build_afm_design(data), data.y, method="L-BFGS-B")
    values = fit.kc_values(data)
    for column in ("KC Name", "Slope", "Intercept (probability) at Opportunity 1",
                   "Number of Unique Steps"):
        assert column in values.columns
    assert values["Intercept (probability) at Opportunity 1"].between(0, 1).all()
    assert values["Number of Unique Steps"].sum() >= len(set(data.items))


# --------------------------------------------------------------------------
# Identification, as AFM reports it
# --------------------------------------------------------------------------

def test_never_repeated_kc_reports_slope_as_undefined_not_zero():
    data = synthetic(n_students=8, n_kcs=12, n_items=12, seed=35, n_reps=1)
    fit = fit_afm(build_afm_design(data), data.y, method="L-BFGS-B",
                  warn_not_converged=False)
    values = fit.kc_values(data)
    assert len(values) == 12, "every KC still gets a row"
    assert values["Slope"].isna().all(), (
        "a KC with no second opportunity has no estimable learning rate; "
        "reporting 0.0 would feed a gamma<=0.001 screen a false positive"
    )


def test_a_duplicate_kc_reports_no_estimate_rather_than_its_twins():
    """The dropped half of a duplicate pair is not estimable; NaN says so."""
    data = co_occurring_kc_data()
    fit = fit_afm(build_afm_design(data), data.y, method="L-BFGS-B",
                  warn_not_converged=False, warn_separated=False)
    values = fit.kc_values(data).set_index("KC Name")
    assert not np.isnan(values.loc["A", "Intercept (logit)"])
    assert np.isnan(values.loc["B", "Intercept (logit)"])
    assert np.isnan(values.loc["B", "Slope"])


# --------------------------------------------------------------------------
# Separation: coefficients with no finite MLE
# --------------------------------------------------------------------------

def test_separated_columns_are_flagged_in_the_kc_values_table():
    """Flagged, not blanked — unlike a never-repeated KC, the datum exists."""
    data = from_frame(separated_frame(), "M")
    fit = fit_afm(build_afm_design(data), data.y, warn_not_converged=False,
                  warn_separated=False)
    values = fit.kc_values(data).set_index("KC Name")
    assert bool(values.loc["kc0", "Separated"])
    assert not bool(values.loc["kc1", "Separated"])
    assert np.isfinite(values.loc["kc0", "Intercept (logit)"])
