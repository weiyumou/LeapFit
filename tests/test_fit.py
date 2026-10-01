"""The penalized logistic fit that every family shares.

The objective and its gradient, against PyAFM's own code and against finite
differences; the likelihoods and the optimality certificate a fit reports;
recentring the student intercepts; and writing predictions back into the
export the fit came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.optimize import minimize

from leapfit import StepData, build_afm_design, fit_afm, from_frame, load_student_step
from leapfit.fit import _expit, _gradient, _objective

from helpers import MINIMAL_COLUMNS, minimal_frame, multi_kc_data, separated_frame, synthetic

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


def test_references_iteration_cap_is_inert_for_tnc():
    """PyAFM's `options={'maxiter': 1000}` is silently dropped by TNC.

    Pinned as a test because it is the reason we default `max_fun=None`: every
    published AFM fit ran at TNC's default budget, not at the stated 1000.
    """
    import warnings as _warnings

    from scipy.optimize import OptimizeWarning

    data = synthetic(n_students=6, n_kcs=3, n_items=12, seed=24)
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


def test_ll_includes_the_ridge_penalty():
    data = synthetic(n_students=10, n_kcs=3, n_items=20, seed=9)
    fit = fit_afm(build_afm_design(data, learnsphere_compat=True), data.y,
                  method="L-BFGS-B")
    assert fit.penalty > 0
    assert fit.ll == pytest.approx(fit.ll_unpenalized - fit.penalty)
    assert fit.aic == pytest.approx(-2 * fit.ll + 2 * fit.n_params)
    assert fit.bic == pytest.approx(
        -2 * fit.ll + fit.n_params * np.log(fit.n_obs))


def test_zero_ridge_makes_the_two_likelihoods_agree():
    data = synthetic(n_students=10, n_kcs=3, n_items=20, seed=10)
    fit = fit_afm(build_afm_design(data, student_l2=0.0), data.y, method="L-BFGS-B")
    assert fit.ll == pytest.approx(fit.ll_unpenalized)


def test_predict_rejects_a_mismatched_design():
    data = synthetic(n_students=6, n_kcs=3, n_items=12, seed=13)
    fit = fit_afm(build_afm_design(data), data.y, method="L-BFGS-B")
    smaller = synthetic(n_students=6, n_kcs=2, n_items=12, seed=14)
    with pytest.raises(ValueError, match="coefficients"):
        fit.predict_proba(build_afm_design(smaller))


def test_fitting_a_separated_design_warns():
    data = from_frame(separated_frame(), "M")
    design = build_afm_design(data)
    with pytest.warns(RuntimeWarning, match="no finite maximum-likelihood"):
        fit_afm(design, data.y, warn_not_converged=False)


# --------------------------------------------------------------------------
# Optimality certificate
# --------------------------------------------------------------------------


def test_optimality_certificate_detects_a_starved_solver():
    data = synthetic(n_students=20, n_kcs=6, n_items=24, seed=38, n_reps=8)
    design = build_afm_design(data)
    starved = fit_afm(design, data.y, method="TNC", max_fun=2, warn_not_converged=False)
    solved = fit_afm(design, data.y, method="TNC", max_fun=200_000,
                     warn_not_converged=False)
    assert not starved.is_optimal and solved.is_optimal
    assert solved.ll > starved.ll


def test_fit_warns_when_not_at_the_optimum():
    data = synthetic(n_students=20, n_kcs=6, n_items=24, seed=39, n_reps=8)
    with pytest.warns(RuntimeWarning, match="not at a stationary point"):
        fit_afm(build_afm_design(data), data.y, method="TNC", max_fun=2)


# --------------------------------------------------------------------------
# Recentring the student intercepts
# --------------------------------------------------------------------------


def test_recentring_students_leaves_predictions_unchanged():
    data = synthetic(n_students=8, n_kcs=4, n_items=16, seed=34, n_reps=6)
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


def test_recentring_refuses_on_multi_kc_designs():
    data = multi_kc_data()
    fit = fit_afm(build_afm_design(data), data.y, method="L-BFGS-B",
                  max_fun=200_000, warn_not_converged=False)
    assert not fit.design.recentring_is_valid()
    with pytest.raises(ValueError, match="multi-KC"):
        fit.centred_students(data)
    # ...and kc_values silently reports uncentred rather than raising.
    assert len(fit.kc_values(data)) == 3


# --------------------------------------------------------------------------
# Annotation: student-step in, student-step out
# --------------------------------------------------------------------------


def _fitted(df, model="M"):
    data = from_frame(df, model)
    fit = fit_afm(build_afm_design(data), data.y, warn_not_converged=False,
                  warn_separated=False)
    return data, fit


def test_annotate_appends_the_datashop_prediction_column():
    df = minimal_frame()
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
    df = minimal_frame()
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
    df = minimal_frame()
    df.insert(2, "Predicted Error Rate (M)", "stale")
    data, fit = _fitted(df)
    out = fit.annotate(data)

    assert list(out.columns) == list(df.columns)          # position preserved
    assert out.columns.tolist().count("Predicted Error Rate (M)") == 1
    assert not (out["Predicted Error Rate (M)"] == "stale").any()


def test_annotate_accumulates_one_column_per_kc_model():
    """``into=`` chains fits of several KC models over one file — the CLI path."""
    df = minimal_frame()
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
    data, fit = _fitted(minimal_frame())
    bare = StepData(y=data.y, students=data.students, items=data.items,
                    kcs=data.kcs, opportunities=data.opportunities,
                    kc_model=data.kc_model)
    with pytest.raises(ValueError, match="source table"):
        fit.annotate(bare)


def test_annotate_refuses_a_subset_fit():
    """A model fitted to a CV training split cannot annotate the full file."""
    data, _ = _fitted(minimal_frame())
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
    df = minimal_frame()
    df.loc[df.index[3], ["KC (M)", "Opportunity (M)"]] = ""
    data, fit = _fitted(df)

    path = tmp_path / "annotated.txt"
    fit.annotate(data).to_csv(path, sep="\t", index=False, lineterminator="\n")
    again = load_student_step(str(path), kc_model="M")

    assert np.array_equal(again.y, data.y)
    assert again.kcs == data.kcs and again.opportunities == data.opportunities
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert raw.loc[3, "Predicted Error Rate (M)"] == ""
