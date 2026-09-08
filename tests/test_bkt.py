"""Tests for Bayesian Knowledge Tracing.

BKT is the first non-logistic family in the package, so it shares none of the
solver machinery the other tests exercise and has to pin its own. Three groups
carry the weight:

* **the recursion is BKT** — predict / update / transition checked step for step
  against a plain-Python implementation of the textbook equations, plus the
  degenerate cases where the answer is known by hand (no guess and no slip
  makes the latent observable; mastery is monotone under correct responses
  when forgetting is off, and can fall when it is on);
* **EM maximizes what it claims to** — monotone by construction, and landing
  where a bounded multi-start direct optimizer on the same likelihood lands.
  ``test_forget_is_not_pinned_at_zero`` is a regression: ``F = 0`` is an
  absorbing fixed point of its own M-step, so an implementation that starts it
  there makes ``forget=True`` silently identical to classical BKT while still
  charging a parameter for it;
* **the honest-statistics contract holds** — a parameter with no data behind it
  is not counted and not printed as a number, degeneracy is flagged rather than
  hidden, and padding is inert.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from leapfit import from_frame, load_student_step
from leapfit.bkt import (
    DEFAULT_INIT,
    BKTFit,
    Sequences,
    bkt_filter,
    bkt_from_data,
    extract_sequences,
    fit_bkt,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

PARAM_SETS = [(0.3, 0.1, 0.2, 0.1), (0.5, 0.25, 0.05, 0.3), (0.1, 0.4, 0.35, 0.02)]


def reference_bkt(answers, prior, learn, guess, slip, forget=0.0):
    """Textbook BKT, written directly — the oracle for the array recursion."""
    known, p_correct, trajectory = prior, [], [prior]
    for y in answers:
        pc = known * (1 - slip) + (1 - known) * guess
        p_correct.append(pc)
        post = known * (1 - slip) / pc if y == 1 else known * slip / (1 - pc)
        known = post * (1 - forget) + (1 - post) * learn
        trajectory.append(known)
    return np.array(p_correct), np.array(trajectory)


def _row(student, step, y, kc, opp, time=None):
    out = {"Anon Student Id": student, "Problem Name": "p", "Step Name": step,
           "First Attempt": "correct" if y else "incorrect", "kc": kc, "opp": str(opp)}
    if time is not None:
        out["First Transaction Time"] = time
    return out


def _rollup(rows, kc_model="M"):
    return pd.DataFrame(rows).rename(
        columns={"kc": f"KC ({kc_model})", "opp": f"Opportunity ({kc_model})"})


def _simulate_bkt(n_students=30, n_reps=12, seed=3, truth=None):
    """Sequentially simulate from a known BKT, one KC per step."""
    truth = truth or {"A": (0.2, 0.25, 0.15, 0.10), "B": (0.5, 0.10, 0.25, 0.20)}
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_students):
        for kc, (prior, learn, guess, slip) in truth.items():
            known = rng.random() < prior
            for t in range(n_reps):
                p = (1 - slip) if known else guess
                y = int(rng.random() < p)
                rows.append(_row(f"s{i}", f"{kc}-{t}", y, kc, t + 1))
                if not known:
                    known = rng.random() < learn
    return from_frame(_rollup(rows), kc_model="M")


@pytest.fixture(scope="module")
def example():
    return load_student_step("examples/student-step.txt", kc_model="Topics")


@pytest.fixture(scope="module")
def simulated():
    return _simulate_bkt()


def _direct_optimum(y, mask, *, fit_forget, ceiling=1.0 - 1e-9, n_starts=12, seed=0):
    """Best log-likelihood a bounded multi-start L-BFGS-B finds on the same objective.

    Twelve starts on a five-parameter problem: verified against 50 during
    development, which found nothing better on any KC of the shipped example.
    The assertion tolerance is 1e-3 against observed gaps of ~1e-5, so the
    margin absorbs a start that lands badly.
    """
    def nll(p):
        forget = p[4] if fit_forget else 0.0
        state = bkt_filter(y, mask, p[0], p[1], p[2], p[3], forget)
        return -float(np.sum(np.log(state["scale"]) * mask))

    bounds = [(1e-9, 1 - 1e-9)] * 2 + [(1e-9, ceiling)] * 2 + [(1e-9, 1 - 1e-9)]
    rng = np.random.default_rng(seed)
    best = -np.inf
    for _ in range(n_starts):
        start = rng.uniform(0.05, min(0.95, ceiling), size=5)
        best = max(best, -minimize(nll, start, method="L-BFGS-B", bounds=bounds).fun)
    return best


# ==========================================================================
# The recursion is BKT
# ==========================================================================

@pytest.mark.parametrize("params", PARAM_SETS)
@pytest.mark.parametrize("forget", [0.0, 0.05])
def test_recursion_matches_a_direct_implementation(params, forget):
    answers = [1, 1, 0, 1, 0, 0, 1]
    y = np.array([answers])
    mask = np.ones_like(y, dtype=bool)

    state = bkt_filter(y, mask, *params, forget=forget)
    expected_p, expected_known = reference_bkt(answers, *params, forget=forget)

    assert state["p_correct"][0] == pytest.approx(expected_p, abs=1e-12)
    # `predicted` is mastery *before* each response, i.e. the trajectory without
    # its final post-transition entry.
    assert state["predicted"][0, :, 1] == pytest.approx(expected_known[:-1], abs=1e-12)


def test_the_two_response_classes_are_a_distribution():
    y = np.array([[1, 0, 1, 1]])
    state = bkt_filter(y, np.ones_like(y, dtype=bool), 0.3, 0.1, 0.2, 0.1)
    assert np.all(state["p_correct"] > 0) and np.all(state["p_correct"] < 1)
    # `scale` is the predictive of the class actually observed.
    expected = np.where(y == 1, state["p_correct"], 1 - state["p_correct"])
    assert state["scale"] == pytest.approx(expected, abs=1e-12)


def test_without_guess_or_slip_the_latent_is_observable():
    """With G = S = 0 a response reveals the state: one correct implies mastery."""
    y = np.array([[0, 1, 1]])
    state = bkt_filter(y, np.ones_like(y, dtype=bool), 0.4, 0.2, 1e-12, 1e-12)
    # After the correct response at t=1 the student is known to know it, and with
    # no forgetting stays there.
    assert state["predicted"][0, 2, 1] == pytest.approx(1.0, abs=1e-9)
    assert state["p_correct"][0, 2] == pytest.approx(1.0, abs=1e-9)


def test_mastery_is_monotone_under_correct_responses_without_forgetting():
    y = np.ones((1, 8), dtype=int)
    known = bkt_filter(y, np.ones_like(y, dtype=bool), 0.2, 0.1, 0.2, 0.1)["predicted"][0, :, 1]
    assert np.all(np.diff(known) > 0)


def test_mastery_can_fall_under_forgetting():
    y = np.ones((1, 8), dtype=int)
    known = bkt_filter(y, np.ones_like(y, dtype=bool), 0.9, 0.05, 0.2, 0.1,
                       forget=0.4)["predicted"][0, :, 1]
    assert np.any(np.diff(known) < 0), "a large forget rate must be able to lower mastery"


def test_padding_is_inert():
    y = np.array([[1, 1, 0, 1, 0], [1, 0, 1, 0, 0]])
    mask = np.array([[True] * 5, [True, True, True, False, False]])
    padded = bkt_filter(y, mask, *PARAM_SETS[0])
    alone = bkt_filter(y[1:2, :3], mask[1:2, :3], *PARAM_SETS[0])

    assert padded["p_correct"][1, :3] == pytest.approx(alone["p_correct"][0], abs=1e-12)
    # A padded step contributes exactly nothing to the log-likelihood.
    assert np.log(padded["scale"][1])[3:] == pytest.approx(np.zeros(2), abs=0)
    # ...and the first row is unaffected by having a short neighbour.
    assert padded["p_correct"][0] == pytest.approx(
        bkt_filter(y[0:1], mask[0:1], *PARAM_SETS[0])["p_correct"][0], abs=1e-12)


# ==========================================================================
# Reshaping the export into sequences
# ==========================================================================

def test_sequences_follow_the_canonical_practice_order():
    """Rows are ordered by transaction time, not by position in the file."""
    rows = [_row("s", "b", 1, "A", 2, time="2024-01-01 00:00:10"),
            _row("s", "a", 0, "A", 1, time="2024-01-01 00:00:05")]
    data = from_frame(_rollup(rows), kc_model="M")
    sequences = extract_sequences(data)

    assert len(sequences) == 1
    assert sequences.y[0, :2].tolist() == [0, 1]      # the 00:00:05 row first
    assert sequences.rows[0, :2].tolist() == [1, 0]   # ...which is file row 1


def test_one_sequence_per_student_kc_pair():
    rows = [_row(s, f"{kc}{t}", 1, kc, t + 1)
            for s in ("s0", "s1") for kc in ("A", "B") for t in range(3)]
    sequences = extract_sequences(from_frame(_rollup(rows), kc_model="M"))
    assert len(sequences) == 4
    assert sequences.n_obs == 12
    assert sorted(sequences.lengths.tolist()) == [3, 3, 3, 3]


def test_rows_map_back_to_the_source_positions(example):
    sequences = extract_sequences(example)
    mapped = sequences.rows[sequences.mask]
    assert sorted(mapped.tolist()) == list(range(len(example)))


def test_multi_kc_steps_are_refused_not_guessed():
    """The convention is undecided, so it raises rather than picking one."""
    rows = [_row("s", "x", 1, "A~~B", "1~~1"), _row("s", "y", 0, "A", "2")]
    data = from_frame(_rollup(rows), kc_model="M")
    with pytest.raises(ValueError, match="one KC per step"):
        extract_sequences(data)


def test_take_subsets_and_retrims():
    rows = ([_row("s0", f"a{t}", 1, "A", t + 1) for t in range(6)]
            + [_row("s1", "a0", 0, "A", 1)])
    sequences = extract_sequences(from_frame(_rollup(rows), kc_model="M"))
    assert sequences.y.shape[1] == 6

    short = sequences.take(np.array([1]))          # only the length-1 sequence
    assert short.y.shape == (1, 1) and short.n_obs == 1
    assert short.kc_names == sequences.kc_names    # the vocabulary is not resized


# ==========================================================================
# EM maximizes what it claims to
# ==========================================================================

@pytest.mark.parametrize("forget", [False, True])
def test_em_is_monotone(simulated, forget):
    fit = fit_bkt(extract_sequences(simulated), forget=forget, n_restarts=3)
    assert np.all(fit.monotone), "EM decreased the likelihood — that is a bug, not a fit"


@pytest.mark.parametrize("forget", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_em_reaches_what_a_direct_optimizer_reaches(example, forget, bounded):
    """EM against multi-start L-BFGS-B on the same likelihood, KC by KC.

    Not a tautology: EM and a quasi-Newton method on the raw objective share
    nothing but the objective, so agreement is evidence the E- and M-steps are
    the right ones.
    """
    sequences = extract_sequences(example)
    fit = fit_bkt(sequences, forget=forget, bound_degenerate=bounded, n_restarts=10)
    ceiling = 0.5 if bounded else 1.0 - 1e-9

    for k, name in enumerate(sequences.kc_names):
        y, mask = sequences.for_kc(k)
        direct = _direct_optimum(y, mask, fit_forget=forget, ceiling=ceiling)
        state = bkt_filter(y, mask, fit.prior[k], fit.learn[k], fit.guess[k],
                           fit.slip[k], fit.forget[k])
        theirs = float(np.sum(np.log(state["scale"]) * mask))
        assert direct - theirs < 1e-3, f"EM stopped {direct - theirs:.4g} nats short on {name}"


def test_forget_is_not_pinned_at_zero(simulated):
    """Regression: ``F = 0`` is an absorbing fixed point of its own M-step.

    ``F' = sum_t xi_t(known, unknown) / sum_t gamma_t(known)``, and every
    ``xi_t(known, unknown)`` carries a factor of ``A[known, unknown] = F``. An
    implementation that initialises ``F`` at 0 therefore re-estimates 0 forever:
    ``forget=True`` becomes classical BKT wearing an extra parameter.
    """
    assert DEFAULT_INIT[4] > 0, "the forget initializer must not start at the fixed point"

    sequences = extract_sequences(simulated)
    fitted = fit_bkt(sequences, forget=True, n_restarts=3)
    assert np.nanmax(fitted.forget) > 1e-4, "forget never moved off its initial value"

    # Being a strict superset of classical BKT, it cannot fit worse.
    classical = fit_bkt(sequences, forget=False, n_restarts=3)
    assert fitted.ll > classical.ll - 1e-6
    assert fitted.n_params > classical.n_params


def test_forget_off_pins_it_at_zero_and_does_not_charge_for_it(simulated):
    sequences = extract_sequences(simulated)
    fit = fit_bkt(sequences, forget=False, n_restarts=2)
    assert np.all(fit.forget == 0.0)
    assert fit.n_params == 4 * len(sequences.kc_names)   # L0, T, G, S — no F


def test_one_restart_is_deterministic_and_ignores_the_seed(simulated):
    sequences = extract_sequences(simulated)
    a = fit_bkt(sequences, n_restarts=1, seed=0)
    b = fit_bkt(sequences, n_restarts=1, seed=99)
    assert a.ll == b.ll and np.array_equal(a.guess, b.guess)


def test_restarts_never_lose_to_a_single_run(simulated):
    sequences = extract_sequences(simulated)
    one = fit_bkt(sequences, n_restarts=1)
    many = fit_bkt(sequences, n_restarts=8, seed=1)
    assert many.ll >= one.ll - 1e-9
    assert many.restart_lls.shape[1] == 8
    assert np.all(many.restart_spread >= 0)


def test_bound_degenerate_keeps_guess_and_slip_below_a_half(simulated):
    fit = fit_bkt(extract_sequences(simulated), bound_degenerate=True, n_restarts=5)
    assert np.all(np.nan_to_num(fit.guess) <= 0.5 + 1e-9)
    assert np.all(np.nan_to_num(fit.slip) <= 0.5 + 1e-9)
    assert fit.degenerate == []


def test_em_recovers_planted_parameters():
    """A long simulation from known parameters is fitted back to them."""
    truth = {"A": (0.15, 0.20, 0.10, 0.08)}
    data = _simulate_bkt(n_students=400, n_reps=15, seed=11, truth=truth)
    fit = fit_bkt(extract_sequences(data), n_restarts=6, seed=2)

    prior, learn, guess, slip = truth["A"]
    assert fit.prior[0] == pytest.approx(prior, abs=0.10)
    assert fit.learn[0] == pytest.approx(learn, abs=0.06)
    assert fit.guess[0] == pytest.approx(guess, abs=0.06)
    assert fit.slip[0] == pytest.approx(slip, abs=0.06)


# ==========================================================================
# The honest-statistics contract
# ==========================================================================

def test_a_kc_never_practised_twice_has_no_learn_rate():
    """No transition observed means no learn rate to estimate — and none charged."""
    rows = [_row(f"s{i}", "only", i % 2, "Once", 1) for i in range(8)]
    rows += [_row(f"s{i}", f"rep{t}", 1, "Twice", t + 1) for i in range(8) for t in range(3)]
    sequences = extract_sequences(from_frame(_rollup(rows), kc_model="M"))
    fit = fit_bkt(sequences, n_restarts=2)

    assert fit.unestimable == ["Once"]
    # 3 parameters for 'Once' (L0, G, S) + 4 for 'Twice'.
    assert fit.n_params == 7
    values = fit.kc_values().set_index("KC Name")
    assert np.isnan(values.loc["Once", "Learn (T)"])
    assert np.isfinite(values.loc["Twice", "Learn (T)"])


def test_a_kc_with_no_data_is_unfitted_not_zero():
    sequences = extract_sequences(_simulate_bkt(n_students=4, n_reps=3))
    ghost = Sequences(y=sequences.y, mask=sequences.mask, rows=sequences.rows,
                      kc_of_seq=sequences.kc_of_seq, students=sequences.students,
                      kc_names=[*sequences.kc_names, "NeverSeen"],
                      kc_model=sequences.kc_model, n_source_rows=sequences.n_source_rows)
    fit = fit_bkt(ghost, n_restarts=1)

    assert fit.unfitted == ["NeverSeen"]
    assert np.isnan(fit.prior[-1]), "an unfitted KC must not report 0 as an estimate"
    assert fit.n_params == 4 * (len(ghost.kc_names) - 1)


def test_degeneracy_is_flagged_rather_than_repaired():
    """G + S >= 1 means 'known' predicts failure; the flag says so."""
    fit = BKTFit(kc_names=["A", "B"], prior=np.array([0.3, 0.3]),
                 learn=np.array([0.1, 0.1]), guess=np.array([0.2, 0.7]),
                 slip=np.array([0.1, 0.4]), forget=np.zeros(2), ll=-1.0, n_obs=10,
                 n_seq=2, n_seq_per_kc=np.array([1, 1]), n_transitions=np.array([1, 1]),
                 converged=np.ones(2, bool), monotone=np.ones(2, bool),
                 n_iter=np.array([1, 1]), final_improvement=np.zeros(2),
                 restart_lls=np.zeros((2, 1)), has_forget=False,
                 bound_degenerate=False, kc_model="M")
    assert fit.degenerate == ["B"]
    assert "degenerate" in fit.summary()


def test_degeneracy_warns_when_it_happens():
    """A KC answered mostly wrong drives the unconstrained fit into the corner."""
    rows = [_row(f"s{i}", f"t{t}", 0, "Bad", t + 1) for i in range(6) for t in range(4)]
    rows += [_row(f"s{i}", f"u{t}", 1, "Bad", t + 5) for i in range(6) for t in range(4)]
    sequences = extract_sequences(from_frame(_rollup(rows), kc_model="M"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_bkt(sequences, n_restarts=12, seed=4)
    if fit.degenerate:  # data-dependent; when it happens it must be announced
        assert any("G + S >= 1" in str(w.message) for w in caught)


def test_there_is_no_optimality_certificate(example):
    """BKT's likelihood is not concave, so the AFM guarantee must not be implied."""
    fit = fit_bkt(extract_sequences(example), n_restarts=3)
    assert not hasattr(fit, "is_optimal")
    assert fit.worst_final_improvement >= 0
    assert "restarts" in fit.summary()


# ==========================================================================
# Reporting and prediction
# ==========================================================================

def test_predictions_are_row_aligned_and_prequential(example):
    sequences, fit = bkt_from_data(example, n_restarts=2)
    predicted = fit.predict_proba(sequences)

    assert predicted.shape == (len(example),)
    assert np.all(np.isfinite(predicted))
    assert np.all((predicted > 0) & (predicted < 1))
    # The first response in a sequence is predicted from the prior alone.
    first = sequences.rows[:, 0]
    for seq, row in enumerate(first):
        k = sequences.kc_of_seq[seq]
        expected = (fit.prior[k] * (1 - fit.slip[k]) + (1 - fit.prior[k]) * fit.guess[k])
        assert predicted[row] == pytest.approx(expected, abs=1e-12)


def test_mastery_starts_at_the_prior(example):
    sequences, fit = bkt_from_data(example, n_restarts=2)
    mastery = fit.mastery(sequences)
    for seq, row in enumerate(sequences.rows[:, 0]):
        assert mastery[row] == pytest.approx(fit.prior[sequences.kc_of_seq[seq]], abs=1e-12)


def test_predicting_with_a_different_kc_model_raises(example):
    _, fit = bkt_from_data(example, n_restarts=1)
    other = extract_sequences(load_student_step("examples/student-step.txt",
                                                kc_model="Skills"))
    with pytest.raises(ValueError, match="different KC models"):
        fit.predict_proba(other)


def test_rmse_beats_the_base_rate(example):
    sequences, fit = bkt_from_data(example, n_restarts=4)
    base = float(np.sqrt(np.mean((example.y - example.y.mean()) ** 2)))
    assert fit.rmse(sequences, example.y) < base


def test_annotate_returns_the_source_table_plus_one_column(example):
    _, fit = bkt_from_data(example, n_restarts=1)
    out = fit.annotate(example)
    column = f"Predicted Error Rate ({example.kc_model})"

    assert len(out) == len(example.source)
    assert column in out.columns
    assert list(out.columns[:-1]) == list(example.source.columns)
    assert np.all((out[column].dropna() > 0) & (out[column].dropna() < 1))


def test_fit_statistics_are_self_consistent(example):
    sequences, fit = bkt_from_data(example, n_restarts=2)
    assert fit.n_obs == len(example)
    assert fit.aic == pytest.approx(-2 * fit.ll + 2 * fit.n_params)
    assert fit.bic == pytest.approx(-2 * fit.ll + fit.n_params * np.log(fit.n_obs))
    # The reported log-likelihood is the one the filter computes.
    predicted = fit.predict_proba(sequences)
    y = example.y.astype(float)
    direct = float(np.sum(np.log(np.where(y == 1, predicted, 1 - predicted))))
    assert fit.ll == pytest.approx(direct, abs=1e-8)


def test_params_frame_covers_every_kc(example):
    _, fit = bkt_from_data(example, n_restarts=2)
    frame = fit.params()
    assert list(frame["KC Name"]) == sorted(example.kc_names)
    assert set(frame.columns) >= {"Prior (L0)", "Learn (T)", "Guess (G)", "Slip (S)",
                                  "Forget (F)", "Degenerate", "Restart Spread"}


# ==========================================================================
# Cross-validation and the command line
# ==========================================================================

@pytest.mark.parametrize("scheme", ["student_blocked", "item_blocked", "unstratified"])
def test_cross_validation_runs_under_every_scheme(example, scheme):
    from leapfit.bkt import cross_validate_bkt

    result = cross_validate_bkt(example, scheme=scheme, n_folds=3, seed=0, n_restarts=2)
    assert result.n_folds == 3 and len(result.folds) == 3
    assert 0.0 < result.rmse < 1.0
    assert all(f.n_train + f.n_test == len(example) for f in result.folds)


def test_cross_validation_beats_the_base_rate(example):
    from leapfit.bkt import cross_validate_bkt

    base = float(np.sqrt(np.mean((example.y - example.y.mean()) ** 2)))
    result = cross_validate_bkt(example, scheme="student_blocked", n_folds=3,
                                seed=0, n_restarts=3)
    assert result.rmse < base


def test_the_fold_seed_and_the_restart_seed_are_separate(example):
    """Two randomnesses with two jobs: the partition, and EM's restarts.

    Collapsing them would tie a model's held-out score to the partition it was
    scored on *through the optimizer*, so a change of folds would move the
    number for two reasons at once.
    """
    from leapfit.bkt import cross_validate_bkt

    def run(seed, restart_seed):
        return cross_validate_bkt(example, scheme="student_blocked", n_folds=3,
                                  seed=seed, restart_seed=restart_seed, n_restarts=2)

    base = run(0, 0)
    same_folds = run(0, 7)      # different restarts, same partition
    other_folds = run(1, 0)     # different partition, same restarts

    sizes = [(f.n_train, f.n_test) for f in base.folds]
    assert [(f.n_train, f.n_test) for f in same_folds.folds] == sizes, \
        "the restart seed must not touch the partition"
    assert [f.rmse for f in other_folds.folds] != [f.rmse for f in base.folds], \
        "the fold seed must actually repartition"


def test_restricting_to_a_training_split_closes_the_gaps():
    from leapfit.bkt import _restrict

    rows = [_row("s", f"t{t}", t % 2, "A", t + 1) for t in range(6)]
    sequences = extract_sequences(from_frame(_rollup(rows), kc_model="M"))
    keep = np.array([True, False, True, True, False, True])

    trimmed = _restrict(sequences, keep)
    assert trimmed.n_obs == 4
    assert trimmed.lengths.tolist() == [4]                    # one sequence, compacted
    assert trimmed.rows[0, :4].tolist() == [0, 2, 3, 5]       # order preserved
    assert trimmed.y[0, :4].tolist() == [0, 0, 1, 1]          # responses follow their rows


def test_a_sequence_emptied_by_the_split_is_dropped():
    from leapfit.bkt import _restrict

    rows = ([_row("s0", f"a{t}", 1, "A", t + 1) for t in range(3)]
            + [_row("s1", "a0", 0, "A", 1)])
    sequences = extract_sequences(from_frame(_rollup(rows), kc_model="M"))
    keep = np.array([True, True, True, False])       # s1 disappears entirely

    trimmed = _restrict(sequences, keep)
    assert len(trimmed) == 1 and trimmed.students == ["s0"]
    assert trimmed.kc_names == sequences.kc_names    # vocabulary is not resized


def test_the_cli_reports_bkt_columns(tmp_path, capsys):
    from leapfit.cli import main_bkt

    out = tmp_path / "bkt.csv"
    code = main_bkt(["examples/student-step.txt", "--kc-model", "Topics",
                     "--restarts", "2", "--cv", "student_blocked",
                     "--out", str(out)])
    assert code == 0
    table = pd.read_csv(out)
    assert list(table["kc_model"]) == ["Topics"]
    assert {"n_params", "log_likelihood", "aic", "bic", "restart_spread",
            "n_degenerate", "n_unestimable", "cv_rmse"} <= set(table.columns)
    # No design, so no design-shaped columns pretending otherwise.
    assert "n_aliased" not in table.columns and "is_optimal" not in table.columns


def test_the_cli_skips_a_multi_kc_model_rather_than_failing(tmp_path, capsys):
    """A file with one usable KC model and one multi-KC model still produces a table."""
    from leapfit.cli import main_bkt

    frame = pd.read_csv("examples/student-step.txt", sep="\t", dtype=str,
                        keep_default_na=False)
    frame["KC (Both)"] = frame["KC (Topics)"] + "~~" + frame["KC (Skills)"]
    frame["Opportunity (Both)"] = (frame["Opportunity (Topics)"] + "~~"
                                   + frame["Opportunity (Skills)"])
    export = tmp_path / "two-models.txt"
    frame.to_csv(export, sep="\t", index=False)

    out = tmp_path / "bkt.csv"
    code = main_bkt([str(export), "--kc-model", "Topics", "--kc-model", "Both",
                     "--restarts", "1", "--cv", "none", "--out", str(out)])
    assert code == 0
    assert list(pd.read_csv(out)["kc_model"]) == ["Topics"]
    assert "skipped" in capsys.readouterr().err
