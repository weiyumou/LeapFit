"""Tests for Performance Factors Analysis.

The load-bearing tests are the count-semantics ones: PFA is AFM with practice
split by outcome, so everything hangs on the success/failure counts being
strictly *prior* and accumulated over the same canonical ordering as ``T``.
The reference implementation this family is grounded against gets exactly that
wrong (``AnalysisPfaStepBased/program/PFA.R:33-34`` uses an inclusive cumsum),
and ``test_inclusive_counts_manufacture_learning_rates_from_noise`` pins what
that defect does, as a permanent demonstration.
"""

from __future__ import annotations

import numpy as np
import pytest

from leapfit import (
    build_afm_design,
    build_pfa_design,
    cross_validate,
    fit_afm,
    fit_pfa,
    from_frame,
    success_failure_counts,
)
from leapfit.fit import _expit

from helpers import rollup, simulate_pfa, step_data, step_row

# --------------------------------------------------------------------------
# Count semantics — the part the reference got wrong
# --------------------------------------------------------------------------


def test_counts_are_strictly_prior():
    """correct, incorrect, correct -> s = (0,1,1), f = (0,0,1)."""
    df = rollup([step_row("s1", f"st{i}", y, "A", i + 1)
                 for i, y in enumerate([1, 0, 1])])
    data = from_frame(df, "M")
    s, f = success_failure_counts(data)
    assert s == [(0,), (1,), (1,)]
    assert f == [(0,), (0,), (1,)]


def test_the_inclusive_mode_reproduces_the_reference_leak_identity():
    """AnalysisPfaStepBased's cumsum: s = s_prior + y, f = f_prior + (1 - y),
    row by row — the identity that puts the response on both sides."""
    data, _ = simulate_pfa(n_students=8, n_reps=6)
    s_prior, f_prior = success_failure_counts(data)
    s_incl, f_incl = success_failure_counts(data, inclusive=True)
    for i in range(len(data)):
        y = int(data.y[i])
        assert all(a == b + y for a, b in zip(s_incl[i], s_prior[i]))
        assert all(a == b + (1 - y) for a, b in zip(f_incl[i], f_prior[i]))


def test_multi_kc_steps_feed_every_kc_on_the_row():
    df = rollup([
        step_row("s1", "st1", 1, "A~~B", "1~~1"),
        step_row("s1", "st2", 0, "A~~B", "2~~2"),
        step_row("s1", "st3", 1, "A", "3"),
    ])
    data = from_frame(df, "M")
    s, f = success_failure_counts(data)
    assert s == [(0, 0), (1, 1), (1,)]
    assert f == [(0, 0), (0, 0), (1,)]


def test_counts_follow_transaction_time_not_row_order():
    """A file listing attempts out of order still accumulates by time —
    the validation export really contains such inversions."""
    df = rollup([
        step_row("s1", "late", 1, "A", 2, time="2024-01-01 01:44:41"),
        step_row("s1", "early", 0, "A", 1, time="2024-01-01 01:44:36"),
    ])
    data = from_frame(df, "M")
    s, f = success_failure_counts(data)
    assert (s[0], f[0]) == ((0,), (1,)), "the later attempt saw the earlier failure"
    assert (s[1], f[1]) == ((0,), (0,))


# --------------------------------------------------------------------------
# The design
# --------------------------------------------------------------------------


def test_per_kc_design_shape_and_blocks():
    data, truth = simulate_pfa(n_students=12, n_reps=8)
    design = build_pfa_design(data, identify=False)
    assert [b.name for b in design.blocks] == ["kc_intercept", "kc_success", "kc_failure"]
    assert design.n_params == 3 * len(truth)
    assert np.all(design.l2 == 0.0)


def test_pooled_design_shares_two_slopes():
    data, truth = simulate_pfa(n_students=12, n_reps=8)
    design = build_pfa_design(data, slopes="pooled", identify=False)
    assert [b.name for b in design.blocks] == ["kc_intercept", "success", "failure"]
    assert design.n_params == len(truth) + 2


def test_student_intercepts_recreate_the_sum_redundancy_and_identify_fixes_it():
    data, _ = simulate_pfa(n_students=12, n_reps=8)
    design = build_pfa_design(data, student_intercepts=True)
    dropped = design.aliased.by_block()
    assert dropped.get("student"), "one student must fall out as the reference level"
    assert design.n_params == design.rank()


def test_invalid_options_raise():
    data, _ = simulate_pfa(n_students=4, n_reps=4)
    with pytest.raises(ValueError, match="slopes"):
        build_pfa_design(data, slopes="banana")
    with pytest.raises(ValueError, match="counts"):
        build_pfa_design(data, counts="lagged")


def test_a_kc_with_no_prior_successes_has_no_estimable_success_slope():
    """The PFA analogue of AFM's never-practised-twice KC."""
    rows = [step_row(f"s{i}", f"st{r}", 0, "Z", r + 1)
            for i in range(6) for r in range(3)]
    rows += [step_row(f"s{i}", f"stA{r}", r % 2, "A", r + 1)
             for i in range(6) for r in range(4)]
    data = step_data(rows)
    design = build_pfa_design(data)
    assert "kc_success:Z" in design.aliased.columns

    fit = fit_pfa(design, data.y, warn_not_converged=False, warn_separated=False)
    values = fit.kc_values(data).set_index("KC Name")
    assert np.isnan(values.loc["Z", "Success Slope"])
    assert not np.isnan(values.loc["Z", "Failure Slope"])
    # Z is also all-failure, so its intercept diverges — flagged, not hidden.
    assert bool(values.loc["Z", "Separated"])


# --------------------------------------------------------------------------
# Fitting: recovery, nesting, and the leak demonstration
# --------------------------------------------------------------------------


def test_recovers_the_generating_parameters():
    data, truth = simulate_pfa(n_students=60, n_reps=14, seed=11)
    fit = fit_pfa(build_pfa_design(data), data.y, method="L-BFGS-B",
                  max_fun=200_000)
    assert fit.is_optimal
    values = fit.kc_values(data).set_index("KC Name")
    for kc, (beta, gamma, rho) in truth.items():
        assert values.loc[kc, "Success Slope"] == pytest.approx(gamma, abs=0.12), kc
        assert values.loc[kc, "Failure Slope"] == pytest.approx(rho, abs=0.12), kc
        assert values.loc[kc, "Intercept (logit)"] == pytest.approx(beta, abs=0.35), kc


def test_pfa_nests_afm():
    """With T recomputed, AFM's slope column is exactly kc_success + kc_failure,
    so PFA's likelihood can never be worse — and splitting one slope into two
    costs exactly one parameter per KC."""
    data, truth = simulate_pfa(n_students=25, n_reps=10, seed=3, student_sd=0.6)
    afm_design = build_afm_design(data, recompute_opportunities=True)
    pfa_design = build_pfa_design(data, student_intercepts=True)

    afm = fit_afm(afm_design, data.y, method="L-BFGS-B", max_fun=200_000)
    pfa = fit_pfa(pfa_design, data.y, method="L-BFGS-B", max_fun=200_000)
    assert afm.is_optimal and pfa.is_optimal
    assert pfa.ll_unpenalized >= afm.ll_unpenalized - 1e-4
    assert pfa.n_params == afm.n_params + len(truth)


def test_inclusive_counts_manufacture_learning_rates_from_noise():
    """The reference's defect, as a permanent demonstration.

    Responses are i.i.d. coin flips: the true slopes are exactly zero and
    nothing is learnable. Prior counts recover that. Inclusive counts —
    the reference's construction — produce large opposite-signed "learning
    rates" and a far better in-sample likelihood, because each response
    predicts itself.
    """
    rng = np.random.default_rng(0)
    rows = [step_row(f"s{i:02d}", f"st{t}", int(rng.random() < 0.6), "K", t + 1)
            for i in range(60) for t in range(30)]
    data = step_data(rows)

    prior = fit_pfa(build_pfa_design(data, slopes="pooled"), data.y,
                    method="L-BFGS-B", max_fun=100_000)
    with pytest.warns(UserWarning, match="inside its own predictor"):
        leaky_design = build_pfa_design(data, slopes="pooled", counts="inclusive")
    leaky = fit_pfa(leaky_design, data.y, method="L-BFGS-B", max_fun=100_000)

    def slopes(fit):
        return fit.block("success")[0], fit.block("failure")[0]

    g0, r0 = slopes(prior)
    g1, r1 = slopes(leaky)
    assert abs(g0) < 0.05 and abs(r0) < 0.05, "prior counts must find nothing"
    assert g1 > 0.05 and r1 < -0.10, "inclusive counts invent opposite-signed slopes"
    assert leaky.ll_unpenalized - prior.ll_unpenalized > 30.0, (
        "the leak buys a large spurious likelihood gain")


def test_a_fit_that_stops_short_names_its_family():
    """Regression: the warning said AFM whatever was fitted."""
    data, _ = simulate_pfa(n_students=8, n_reps=6)
    with pytest.warns(RuntimeWarning, match="The PFA fit is not at a stationary point"):
        fit_pfa(build_pfa_design(data), data.y, method="TNC", max_fun=2)


def test_a_pfa_fit_can_start_from_another():
    """``w0``, as fit_afm and fit_lkt take it: the same optimum from a warm start."""
    data, _ = simulate_pfa(n_students=8, n_reps=6)
    design = build_pfa_design(data)
    cold = fit_pfa(design, data.y, warn_not_converged=False)
    warm = fit_pfa(design, data.y, w0=cold.weights, warn_not_converged=False)
    assert warm.ll == pytest.approx(cold.ll, abs=1e-6)


def test_pooled_kc_values_broadcast_the_shared_slopes():
    data, _ = simulate_pfa(n_students=15, n_reps=8)
    fit = fit_pfa(build_pfa_design(data, slopes="pooled"), data.y,
                  warn_not_converged=False)
    values = fit.kc_values(data)
    assert values["Success Slope"].nunique() == 1
    assert values["Failure Slope"].nunique() == 1
    assert not values["Success Slope"].isna().any()


def test_probability_column_matches_the_intercept():
    data, _ = simulate_pfa(n_students=15, n_reps=8)
    fit = fit_pfa(build_pfa_design(data), data.y, warn_not_converged=False)
    values = fit.kc_values(data)
    np.testing.assert_allclose(
        values["Intercept (probability) at first attempt"],
        _expit(values["Intercept (logit)"].to_numpy()))


# --------------------------------------------------------------------------
# The shared machinery applies unchanged
# --------------------------------------------------------------------------


def test_cross_validation_and_annotation_work_for_pfa():
    data, _ = simulate_pfa(n_students=15, n_reps=8)
    design = build_pfa_design(data)
    cv = cross_validate(design, data, scheme="item_blocked", n_folds=3)
    assert 0.0 < cv.rmse < 1.0

    fit = fit_pfa(design, data.y, warn_not_converged=False)
    out = fit.annotate(data)
    col = out["Predicted Error Rate (M)"]
    assert len(out) == len(data) and not col.isna().any()
    np.testing.assert_allclose(col.to_numpy(), 1.0 - fit.predict_proba(design))
