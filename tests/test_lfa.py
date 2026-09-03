"""Learning Factors Analysis: the operators, the screens, and the search.

Two tests here are load-bearing and must not be allowed to fail quietly.

``test_a_separated_kc_is_refused_and_the_reason_is_recorded`` and
``test_without_the_screens_the_reference_pathology_reproduces`` are a matched
pair. On the validation export the reference LFA selected a KC model whose
slope has no finite estimate, and that one bad split then appeared in all 99
states it reported. The pair pins both halves: that leapfit refuses the move,
and that turning the screens off puts the pathology back — so the screen is
demonstrably what prevents it, not an accident of this data.

``test_the_evidence_screen_counts_prior_practice_not_observations`` pins the
*unit* of the evidence screen. Observations at ``T >= 1`` are the only rows a
slope column touches, and on the validation export that count is what
separates a structurally dead KC (0), the reference's degenerate pick (5, all
failures), and its clean siblings (201 and 475). Counting plain observations
instead would have passed all three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leapfit import build_afm_design, fit_afm, from_frame, load_student_step
from leapfit.lfa import (
    HEURISTICS,
    FactorMatrix,
    Move,
    build_factor_matrix,
    lfa_search,
    relabel,
    replay,
    split,
    validate_top,
)
from leapfit.lfa import _partition as partition

EXAMPLE = "examples/student-step.txt"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _frame(rows, kc_model="F"):
    return pd.DataFrame(rows).rename(
        columns={"kc": f"KC ({kc_model})", "opp": f"Opportunity ({kc_model})"})


def _attempts(student, step, kc, outcomes):
    """One student's repeated attempts at one step, in practice order."""
    return [{"Anon Student Id": student, "Problem Name": "p", "Step Name": step,
             "First Attempt": outcome, "kc": kc, "opp": str(i + 1)}
            for i, outcome in enumerate(outcomes)]


def _mixed(student, step, kc, n):
    """``n`` attempts alternating correct/incorrect — never separated."""
    return _attempts(student, step, kc,
                     ["correct" if i % 2 else "incorrect" for i in range(n)])


def _screen_data():
    """Three factors, each a single step, chosen to trip a different screen.

    ``dead``   one attempt per student, so no row has ``T >= 1``
    ``thin``   two attempts per student, so one row per student has ``T >= 1``
    ``uniform`` four attempts per student, all incorrect: enough rows to clear
               the evidence screen, but its slope column separates the data
    ``mixedN`` ordinary steps, so the residual skill is always well behaved
    """
    rows = []
    for student in ("s1", "s2", "s3"):
        rows += _attempts(student, "dead", "dead", ["correct"])
        rows += _mixed(student, "thin", "thin", 2)
        rows += _attempts(student, "uniform", "uniform", ["incorrect"] * 4)
        for k in range(4):
            rows += _mixed(student, f"m{k}", f"m{k}", 6)
    return from_frame(_frame(rows), "F")


def _constant_student_data():
    """One student who never got anything wrong — separated in every state."""
    rows = []
    for student in ("s1", "s2"):
        for k in range(3):
            rows += _mixed(student, f"m{k}", f"m{k}", 6)
    rows += _attempts("always", "m0", "m0", ["correct", "correct"])
    return from_frame(_frame(rows), "F")


def _example_factors():
    models = {m: load_student_step(EXAMPLE, kc_model=m) for m in ("Topics", "Skills")}
    return models, build_factor_matrix(models)


# --------------------------------------------------------------------------
# The difficulty-factor matrix
# --------------------------------------------------------------------------


def test_factor_matrix_drops_a_column_identical_to_one_already_taken():
    rows = _mixed("s1", "a", "X", 2) + _mixed("s1", "b", "Y", 2)
    coarse = from_frame(_frame(rows, "coarse"), "coarse")
    same = from_frame(_frame(rows, "coarse").rename(
        columns={"KC (coarse)": "KC (twin)",
                 "Opportunity (coarse)": "Opportunity (twin)"}), "twin")
    P = build_factor_matrix({"coarse": coarse, "twin": same})
    assert len(P) == 2, "the twin's two columns duplicate the first model's exactly"
    assert len(P.dropped) == 2
    assert set(P.reasons) == {"identical to X", "identical to Y"}
    assert "dropped" in P.summary()


def test_a_skill_name_present_in_two_models_is_prefixed_with_its_model():
    a = from_frame(_frame(_mixed("s1", "a", "S", 2) + _mixed("s1", "b", "T", 2),
                          "one"), "one")
    b = from_frame(_frame(_mixed("s1", "a", "S", 2) + _mixed("s1", "b", "S", 2),
                          "two"), "two")
    P = build_factor_matrix({"one": a, "two": b})
    assert "S" in P.factors, "the first model keeps the bare name"
    assert "two-S" in P.factors, "the collision is prefixed, not silently merged"


def test_a_multi_kc_model_cannot_serve_as_a_difficulty_factor():
    df = _frame([{"Anon Student Id": "s1", "Problem Name": "p", "Step Name": "a",
                  "First Attempt": "correct", "kc": "A~~B", "opp": "1~~1"}])
    with pytest.raises(ValueError, match="more than one"):
        build_factor_matrix({"wide": from_frame(df, "F")})


def test_models_covering_different_row_sets_cannot_be_combined():
    full = _mixed("s1", "a", "X", 2) + _mixed("s1", "b", "Y", 2)
    short = _frame(full, "short")
    short.loc[0, "KC (short)"] = ""          # drops that observation entirely
    short.loc[0, "Opportunity (short)"] = ""
    with pytest.raises(ValueError, match="different numbers of observations"):
        build_factor_matrix({"full": from_frame(_frame(full, "full"), "full"),
                             "short": from_frame(short, "short")})


def test_build_factor_matrix_refuses_an_empty_mapping():
    with pytest.raises(ValueError, match="at least one"):
        build_factor_matrix({})


def test_a_factor_matrix_checks_its_parallel_tuples():
    with pytest.raises(ValueError, match="members has"):
        FactorMatrix(("a",), ("f", "g"), (frozenset({"a"}),))


# --------------------------------------------------------------------------
# The operators
# --------------------------------------------------------------------------


STEPS = ("a", "b", "c")


def test_split_carves_the_factors_steps_out_of_the_skill():
    got = split(("all",) * 3, STEPS, "all", "f", frozenset({"a"}))
    assert got == ("all*f", "all", "all")


def test_a_split_that_would_empty_either_side_is_degenerate():
    labels = ("all",) * 3
    assert split(labels, STEPS, "all", "f", frozenset()) is None, "covers nothing"
    assert split(labels, STEPS, "all", "f", frozenset(STEPS)) is None, "covers all"


def test_a_split_only_touches_the_skill_it_names():
    labels = ("all*f", "all", "all")
    got = split(labels, STEPS, "all", "g", frozenset({"a", "b"}))
    assert got == ("all*f", "all*g", "all"), "step a already belongs to all*f"


def test_a_derived_skill_can_be_split_again():
    labels = split(("all",) * 3, STEPS, "all", "f", frozenset({"a", "b"}))
    got = split(labels, STEPS, "all*f", "g", frozenset({"a"}))
    assert got == ("all*f*g", "all*f", "all")


def test_replay_reconstructs_a_labelling_from_its_history():
    P = FactorMatrix(STEPS, ("f", "g"),
                     (frozenset({"a"}), frozenset({"b"})))
    history = (Move("split", "all", "f"), Move("split", "all", "g"))
    assert replay(history, P) == ("all*f", "all*g", "all")


def test_replaying_a_history_without_a_move_is_how_merge_undoes_a_split():
    P = FactorMatrix(STEPS, ("f", "g"),
                     (frozenset({"a"}), frozenset({"b"})))
    history = (Move("split", "all", "f"), Move("split", "all", "g"))
    assert replay(history[1:], P) == ("all", "all*g", "all")


def test_transposed_split_orders_are_one_state():
    P = FactorMatrix(STEPS, ("f", "g"),
                     (frozenset({"a"}), frozenset({"b"})))
    forward = replay((Move("split", "all", "f"), Move("split", "all", "g")), P)
    reverse = replay((Move("split", "all", "g"), Move("split", "all", "f")), P)
    assert partition(forward) == partition(reverse), (
        "the search must not fit the same partition twice under two labellings")


def test_a_stale_history_raises_rather_than_replaying_a_degenerate_split():
    P = FactorMatrix(STEPS, ("f",), (frozenset(STEPS),))
    with pytest.raises(ValueError, match="degenerate"):
        replay((Move("split", "all", "f"),), P)


# --------------------------------------------------------------------------
# Opportunity counts
# --------------------------------------------------------------------------


def test_a_searched_kc_model_recomputes_its_own_opportunity_counts():
    """The export's column belongs to the model it was written for.

    ``relabel`` must ignore it: a searched labelling did not exist when the
    file was written, so reading the column would carry the *old* model's
    practice counts into the new one.
    """
    rows = _mixed("s1", "a", "X", 2) + _mixed("s1", "b", "X", 2)
    data = from_frame(_frame(rows), "F")
    assert [o[0] for o in data.opportunities] == [0, 1, 0, 1], "as exported, one KC"
    got = relabel(data, ("p##a", "p##b"), ("split_a", "split_b"))
    assert [o[0] for o in got.opportunities] == [0, 1, 0, 1]
    assert got.kcs[0] == ("split_a",) and got.kcs[2] == ("split_b",)


def test_relabelling_two_steps_apart_restarts_each_kcs_practice_count():
    rows = _attempts("s1", "a", "X", ["correct", "correct"]) + \
        _attempts("s1", "b", "X", ["correct", "correct"])
    data = from_frame(_frame(rows), "F")
    merged = relabel(data, ("p##a", "p##b"), ("one", "one"))
    assert [o[0] for o in merged.opportunities] == [0, 1, 2, 3], (
        "one KC over both steps accumulates across them")
    apart = relabel(data, ("p##a", "p##b"), ("one", "two"))
    assert [o[0] for o in apart.opportunities] == [0, 1, 0, 1], (
        "two KCs count their own practice only")


# --------------------------------------------------------------------------
# The screens
# --------------------------------------------------------------------------


def test_the_evidence_screen_counts_prior_practice_not_observations():
    """The screen counts rows at ``T >= 1``, not rows.

    In this fixture ``thin`` has **6 observations but only 3 at T >= 1** — two
    attempts each by three students. A screen counting observations would let
    it through a threshold of 4; the slope it would estimate rests on three
    rows. ``dead`` has 3 observations and none at ``T >= 1`` at all.
    """
    data = _screen_data()
    P = build_factor_matrix({"F": data})
    res = lfa_search(data, P, max_iterations=1, min_opportunities=4,
                     screen_separation=False)
    refused = res.rejected.by_reason().get("too little practice", [])
    assert any("by dead" in m for m in refused), "0 rows at T >= 1"
    assert any("by thin" in m for m in refused), (
        "6 observations, but only 3 of them carry prior practice")
    assert not any("by m0" in m for m in refused), (
        "an ordinary step has 15 rows at T >= 1 and must survive")


def test_the_evidence_screen_compares_against_the_threshold_inclusively():
    data = _screen_data()
    P = build_factor_matrix({"F": data})
    res = lfa_search(data, P, max_iterations=1, min_opportunities=3,
                     screen_separation=False)
    refused = res.rejected.by_reason().get("too little practice", [])
    assert any("by dead" in m for m in refused)
    assert not any("by thin" in m for m in refused), (
        "exactly 3 rows at T >= 1 clears a threshold of 3")


def test_a_separated_kc_is_refused_and_the_reason_is_recorded():
    data = _screen_data()
    P = build_factor_matrix({"F": data})
    res = lfa_search(data, P, max_iterations=1)
    refused = res.rejected.by_reason().get("separated", [])
    assert any("by uniform" in m for m in refused), (
        "all of this KC's repeat attempts fail, so its slope has no finite MLE")
    assert not any("uniform" in "".join(str(m) for m in s.history)
                   for s in res.states), "no accepted state may contain it"


def test_without_the_screens_the_reference_pathology_reproduces():
    """Turning the screens off must put the degenerate split back.

    This is the control for the screen: it shows the refusal is what keeps the
    non-estimable KC out, not a property of this fixture.
    """
    data = _screen_data()
    P = build_factor_matrix({"F": data})
    loose = lfa_search(data, P, max_iterations=1, min_opportunities=0,
                       screen_separation=False)
    accepted = [s for s in loose.states
                if any(m.factor == "uniform" for m in s.history)]
    assert accepted, "with no screen the move is offered and fitted"
    assert any(s.n_separated for s in accepted), (
        "and the state it produces carries a coefficient with no finite estimate")


def test_refusals_are_reported_rather_than_silently_dropped():
    data = _screen_data()
    P = build_factor_matrix({"F": data})
    res = lfa_search(data, P, max_iterations=1)
    assert len(res.rejected), "a screen that shrinks the space must say so"
    assert len(res.rejected.moves) == len(res.rejected.reasons)
    assert "refused" in res.rejected.summary()
    assert "refused" in res.summary()


def test_a_search_that_refuses_nothing_says_so():
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=1)
    assert res.rejected.summary() == "no moves refused"


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------


def test_the_search_improves_on_the_root_by_its_own_heuristic():
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=8)
    assert res.best.score("bic") < res.root.score("bic")
    assert res.best.depth >= 1
    assert res.states[0] is res.best, "states are ranked, best first"


def test_every_evaluated_state_carries_an_optimality_certificate():
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=6)
    assert all(s.is_optimal for s in res.states), (
        "a frontier ordered by uncertified fits is ordered by optimizer noise")
    assert "certified optimum" in res.summary(), (
        "and the count of uncertified states is reported, not left implicit")


def test_the_search_beats_a_hand_authored_kc_model_on_its_own_criterion():
    models, P = _example_factors()
    fit = fit_afm(build_afm_design(models["Topics"]), models["Topics"].y,
                  warn_not_converged=False, warn_separated=False)
    res = lfa_search(models["Topics"], P, max_iterations=8)
    assert res.best.bic < fit.bic, (
        f"searched BIC {res.best.bic:.3f} must beat the authored model's {fit.bic:.3f}")


def test_neither_criterion_recovers_the_planted_model_on_this_example():
    """An honest negative, pinned so nobody "fixes" the search to match hope.

    ``examples/student-step.txt`` is generated from ``Topics``, but at 480
    observations BIC's ``log N`` penalty prefers a coarser model than the truth
    and AIC's prefers a finer one. Recovering the generating model is a
    property of the data, not something the search can promise.
    """
    models, P = _example_factors()
    planted = frozenset(frozenset(np.flatnonzero(
        [kc == (name,) for kc in models["Topics"].kcs]).tolist())
        for name in models["Topics"].kc_names)
    for heuristic in HEURISTICS:
        res = lfa_search(models["Topics"], P, heuristic=heuristic,
                         max_iterations=12)
        assert partition(res.best.labels) != planted


def test_bic_stops_no_deeper_than_aic():
    models, P = _example_factors()
    bic = lfa_search(models["Topics"], P, heuristic="bic", max_iterations=12)
    aic = lfa_search(models["Topics"], P, heuristic="aic", max_iterations=12)
    assert bic.best.depth <= aic.best.depth, (
        "BIC charges log(N) per parameter against AIC's 2, so it stops earlier")


def test_the_search_is_deterministic():
    models, P = _example_factors()
    runs = [lfa_search(models["Topics"], P, max_iterations=5) for _ in range(2)]
    assert [s.bic for s in runs[0].states] == [s.bic for s in runs[1].states]
    assert runs[0].frame().equals(runs[1].frame())


def test_warm_starting_does_not_move_the_optimum():
    models, P = _example_factors()
    warm = lfa_search(models["Topics"], P, max_iterations=5, warm_start=True)
    cold = lfa_search(models["Topics"], P, max_iterations=5, warm_start=False)
    assert warm.n_evaluated == cold.n_evaluated
    assert warm.best.labels == cold.best.labels
    assert warm.best.ll == pytest.approx(cold.best.ll, abs=1e-6), (
        "the objective is convex, so the start point cannot change the optimum")


def test_max_iterations_bounds_the_expansions():
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=2, patience=0)
    assert res.n_iterations == 2
    assert res.stopped == "max iterations"


def test_the_search_stops_on_a_no_improvement_streak():
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=50, patience=2)
    assert res.stopped == "no improvement"
    assert res.n_iterations < 50


def test_the_frontier_can_be_exhausted():
    P = FactorMatrix(("p##a", "p##b"), ("only",), (frozenset({"p##a"}),))
    rows = _mixed("s1", "a", "X", 4) + _mixed("s1", "b", "X", 4) + \
        _mixed("s2", "a", "X", 4) + _mixed("s2", "b", "X", 4)
    data = from_frame(_frame(rows), "F")
    res = lfa_search(data, P, max_iterations=20, patience=0,
                     min_opportunities=1)
    assert res.stopped == "exhausted", "one factor affords exactly one split"


def test_lineage_merge_changes_nothing_when_the_frontier_never_fills():
    """The documented scope of Stage 1's merge, pinned.

    Undoing a split reaches a state that was already scored as a child, so it
    is already on an unbounded frontier. Merge earns its keep only when the
    beam has evicted something.
    """
    models, P = _example_factors()
    with_merge = lfa_search(models["Topics"], P, max_iterations=6, merge=True)
    without = lfa_search(models["Topics"], P, max_iterations=6, merge=False)
    assert with_merge.best.labels == without.best.labels
    assert with_merge.n_evaluated == without.n_evaluated


def test_the_same_split_is_never_offered_twice_on_one_lineage():
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=10)
    for state in res.states:
        pairs = [(m.skill, m.factor) for m in state.history]
        assert len(pairs) == len(set(pairs))


def test_an_unknown_heuristic_raises():
    models, P = _example_factors()
    with pytest.raises(ValueError, match="heuristic must be one of"):
        lfa_search(models["Topics"], P, heuristic="rmse")


def test_a_factor_matrix_must_cover_every_step_being_fitted():
    models, P = _example_factors()
    short = FactorMatrix(P.steps[:-1], P.factors, P.members)
    with pytest.raises(ValueError, match="carry no row in the factor matrix"):
        lfa_search(models["Topics"], short)


def test_a_coefficient_separated_under_the_root_is_reported_once():
    """A student who never varied cannot be repaired by any split.

    Reporting it per state would read as though some move created it. It is
    read once, at the root, and named — because it also makes the optimality
    certificate borderline for every state, which is otherwise a mystery.
    """
    data = _constant_student_data()
    P = build_factor_matrix({"F": data})
    res = lfa_search(data, P, max_iterations=2, min_opportunities=1)
    assert res.persistent_separation, "the constant student must be detected"
    assert any("always" in column for column in res.persistent_separation)
    assert all(column.startswith("student:") for column in res.persistent_separation)
    assert "property of the data" in res.summary()


def test_a_state_distinguishes_kc_separation_from_the_rest():
    data = _constant_student_data()
    P = build_factor_matrix({"F": data})
    res = lfa_search(data, P, max_iterations=2, min_opportunities=1)
    assert all(s.n_separated_kc == 0 for s in res.states), (
        "the screen refuses KC-block separation, so no accepted state has any")
    assert any(s.n_separated for s in res.states), "but the student's is still there"
    carrier = next(s for s in res.states if s.n_separated)
    assert "in a KC block" in carrier.summary(), (
        "a bare separated=1 reads as a KC problem; the breakdown says otherwise")


def test_the_result_frame_is_one_row_per_evaluated_state():
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=4)
    frame = res.frame()
    assert len(frame) == len(res.states) == res.n_evaluated
    assert frame["rank"].tolist() == list(range(1, len(frame) + 1))
    assert frame["bic"].is_monotonic_increasing, "ranked by the heuristic"
    assert frame.loc[0, "history"] == " | ".join(str(m) for m in res.best.history)


def test_a_state_reports_its_kc_model_as_step_to_label():
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=2)
    mapping = res.best.kc_model(P.steps)
    assert set(mapping) == set(P.steps)
    assert set(mapping.values()) == set(res.best.labels)


# --------------------------------------------------------------------------
# Held-out validation of the top states
# --------------------------------------------------------------------------


def _validated(**kwargs):
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=8)
    kwargs.setdefault("n", 4)
    kwargs.setdefault("seeds", (0, 1, 2))
    return models, res, validate_top(res, models["Topics"], **kwargs)


def test_every_candidate_is_scored_on_the_same_folds():
    """Shared folds are what make the contrast paired rather than two means."""
    _, _, val = _validated()
    per_fold = val.folds.groupby(["seed", "fold"])["n_test"].nunique()
    assert (per_fold == 1).all(), (
        "one held-out row count per (seed, fold) means every model saw that fold")
    counts = val.folds.groupby("model").size()
    assert counts.nunique() == 1, "and every model was scored on all of them"


def test_the_root_and_authored_models_join_the_comparison():
    models, _, val = _validated(extra=None)
    assert "root" in set(val.frame()["model"]), "the search's starting point"
    _, _, with_extra = _validated(extra=models)
    named = set(with_extra.frame()["model"])
    assert {"Topics", "Skills", "root"} <= named, (
        "an authored KC model is the comparison that says whether searching helped")


def test_validation_reports_whether_the_criterion_agrees_with_held_out_rmse():
    """On this example it does not, and that is the point of the stage.

    BIC prefers a 2-KC model; pooled item-blocked RMSE prefers a 3-KC one. The
    disagreement is deterministic here, so it is pinned rather than described —
    if a change ever made the criterion and the held-out score agree on this
    data, that would be a result worth noticing, not a silent improvement.
    """
    _, _, val = _validated()
    assert not val.agrees
    assert val.winner != val.frame().iloc[0]["model"]
    assert "DISAGREE" in val.summary()


def test_the_authored_model_can_beat_the_criterions_pick_out_of_sample():
    authored, _ = _example_factors()
    _, _, val = _validated(extra=authored)
    frame = val.frame().set_index("model")
    assert frame.loc["Topics", "search_rank"] > frame.loc["Topics", "cv_rank"], (
        "the planted model is penalised in sample and rewarded out of sample")


def test_contrasts_difference_within_fold_and_exclude_the_baseline():
    _, _, val = _validated()
    assert val.baseline == "root", "the search's starting point by default"
    assert val.baseline not in set(val.contrasts["model"])
    assert (val.contrasts["n_folds"] == 9).all(), "3 folds x 3 seeds, paired"


def test_the_baseline_can_be_named():
    _, res, _ = _validated()
    models, _ = _example_factors()
    val = validate_top(res, models["Topics"], n=3, baseline="rank1")
    assert val.baseline == "rank1"
    assert "rank1" not in set(val.contrasts["model"])


def test_an_unknown_baseline_raises():
    _, res, _ = _validated()
    models, _ = _example_factors()
    with pytest.raises(KeyError, match="baseline"):
        validate_top(res, models["Topics"], n=2, baseline="nope")


def test_an_extra_model_over_other_rows_cannot_be_paired():
    models, res, _ = _validated()
    short = _constant_student_data()
    with pytest.raises(ValueError, match="different row set"):
        validate_top(res, models["Topics"], n=2, extra={"other": short})


def test_validation_scores_in_the_mode_the_search_used():
    """In-sample and held-out columns must describe the same models."""
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=4,
                     learnsphere_compat=True)
    assert res.learnsphere_compat
    val = validate_top(res, models["Topics"], n=3, include_root=False)
    frame = val.frame().set_index("model")
    assert frame.loc["rank1", "bic"] == pytest.approx(res.states[0].bic)
    assert frame.loc["rank1", "n_params"] == res.states[0].n_params


def test_n_must_be_at_least_one():
    models, res, _ = _validated()
    with pytest.raises(ValueError, match="n must be at least 1"):
        validate_top(res, models["Topics"], n=0)


def test_rank_correlation_needs_three_candidates():
    models, P = _example_factors()
    res = lfa_search(models["Topics"], P, max_iterations=3)
    val = validate_top(res, models["Topics"], n=1, include_root=True)
    assert val.rank_correlation() != val.rank_correlation(), "NaN for two"
    assert "undefined" in val.summary()
