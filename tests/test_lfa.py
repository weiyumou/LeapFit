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

import os

import numpy as np
import pandas as pd
import pytest

from leapfit import build_afm_design, fit_afm, from_frame, load_student_step
from leapfit.lfa import (
    HEURISTICS,
    MERGES,
    FactorMatrix,
    Move,
    build_factor_matrix,
    lfa_search,
    merge,
    relabel,
    replay,
    root_labels,
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


def test_merge_joins_two_skills():
    labels = ("all*f", "all*g", "all")
    assert merge(labels, "all*f", "all*g") == ("all*f+all*g", "all*f+all*g", "all")


def test_a_merge_label_does_not_depend_on_the_order_it_was_asked_in():
    """Which is what lets the two orders collapse to one state."""
    labels = ("x", "y", "z")
    assert merge(labels, "x", "y") == merge(labels, "y", "x")


def test_merging_an_absent_or_identical_skill_is_impossible():
    labels = ("x", "y", "z")
    assert merge(labels, "x", "nope") is None
    assert merge(labels, "nope", "x") is None
    assert merge(labels, "x", "x") is None, "a skill cannot merge with itself"


def test_a_merge_reads_as_a_merge_in_the_history():
    assert str(Move("merge", "a", "b")) == "merge a and b"
    assert str(Move("split", "a", "f")) == "split a by f"


def test_replay_applies_merges_as_well_as_splits():
    P = FactorMatrix(STEPS, ("f", "g"),
                     (frozenset({"a"}), frozenset({"b"})))
    history = (Move("split", "all", "f"), Move("split", "all", "g"),
               Move("merge", "all*f", "all*g"))
    assert replay(history, P) == ("all*f+all*g", "all*f+all*g", "all")


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
    with_merge = lfa_search(models["Topics"], P, max_iterations=6, merges="lineage")
    without = lfa_search(models["Topics"], P, max_iterations=6, merges="none")
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
# Scoring across processes
# --------------------------------------------------------------------------


def test_the_worker_count_does_not_move_a_single_digit():
    """``n_jobs`` is a wall-clock knob, not a modelling one.

    The candidates of one expansion are independent and their parent is fixed,
    so how the work is partitioned cannot reach a score. Results come back in
    submission order, so the frontier is built in the same sequence whatever
    the worker count — which is what makes this an equality rather than an
    approximation.
    """
    models, P = _example_factors()
    one = lfa_search(models["Topics"], P, max_iterations=4, n_jobs=1)
    two = lfa_search(models["Topics"], P, max_iterations=4, n_jobs=2)
    assert [s.bic for s in one.states] == [s.bic for s in two.states]
    assert [s.ll for s in one.states] == [s.ll for s in two.states]
    assert [s.labels for s in one.states] == [s.labels for s in two.states]
    assert one.rejected.moves == two.rejected.moves
    assert one.n_evaluated == two.n_evaluated
    assert one.stopped == two.stopped
    assert one.frame().equals(two.frame())


def test_n_jobs_follows_joblibs_convention():
    from leapfit.lfa import _worker_count
    cores = os.cpu_count() or 1
    assert _worker_count(1, 100) == 1
    assert _worker_count(None, 100) == 1, "None is serial, as in crossval"
    assert _worker_count(0, 100) == 1
    assert _worker_count(3, 100) == 3
    assert _worker_count(-1, 100) == cores, "-1 is every core"
    assert _worker_count(-2, 100) == max(1, cores - 1), "-2 is all but one"


def test_a_serial_search_leaves_no_observations_pinned_in_the_caller():
    """The serial path shares the worker's globals; it must not keep them."""
    from leapfit.lfa import _WORKER
    models, P = _example_factors()
    lfa_search(models["Topics"], P, max_iterations=2, n_jobs=1)
    assert not _WORKER, "the module-level scratch space is cleared on exit"


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


# --------------------------------------------------------------------------
# The leapfit-lfa console script
# --------------------------------------------------------------------------


def _lfa_cli(*extra, export=EXAMPLE):
    from leapfit.cli import main_lfa
    return main_lfa([export, "--max-iterations", "2", "--validate", "0", *extra])


def test_cli_lfa_writes_the_frontier_and_the_refusals(tmp_path):
    frontier, refusals = tmp_path / "f.csv", tmp_path / "r.csv"
    assert _lfa_cli("--out", str(frontier), "--refusals", str(refusals)) == 0
    table = pd.read_csv(frontier)
    assert {"rank", "n_kcs", "n_params", "bic", "is_optimal", "history"} <= set(table.columns)
    assert table["rank"].tolist() == list(range(1, len(table) + 1))
    # Written even when nothing was refused: an empty file is a result, a
    # missing one cannot be told from a run that never asked.
    assert set(pd.read_csv(refusals).columns) == {"move", "reason"}


def test_cli_lfa_writes_a_kc_model_ready_to_join(tmp_path):
    out = tmp_path / "q.txt"
    assert _lfa_cli("--qmatrix", str(out), "--kc-model-name", "Discovered") == 0
    table = pd.read_csv(out, sep="\t", dtype=str)
    assert list(table.columns) == ["Problem Name", "Step Name", "KC (Discovered)"]
    assert len(table) == 40, "one row per step, not per observation"
    assert table["KC (Discovered)"].nunique() >= 2, "the search split something"


def test_cli_lfa_annotates_the_export_under_the_discovered_model(tmp_path):
    out = tmp_path / "annotated.txt"
    assert _lfa_cli("--predictions", str(out), "--kc-model-name", "Found") == 0
    written = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    assert "Predicted Error Rate (Found)" in written.columns
    assert (written["Predicted Error Rate (Found)"] != "").all()
    assert "KC (Topics)" in written.columns, "the export round-trips verbatim"


def test_cli_lfa_validates_the_shortlist_and_writes_it(tmp_path):
    from leapfit.cli import main_lfa
    out = tmp_path / "v.csv"
    assert main_lfa([EXAMPLE, "--max-iterations", "2", "--validate", "2",
                     "--compare", "Topics", "--validation", str(out)]) == 0
    table = pd.read_csv(out)
    assert {"cv_rmse", "search_rank", "cv_rank"} <= set(table.columns)
    assert "Topics" in set(table["model"]), "--compare joins the comparison"
    assert "root" in set(table["model"])


def test_cli_lfa_says_so_rather_than_writing_an_empty_validation(tmp_path, capsys):
    out = tmp_path / "v.csv"
    assert _lfa_cli("--validation", str(out)) == 0
    assert not out.exists()
    assert "--validate 0 skipped it" in capsys.readouterr().err


def test_cli_lfa_excludes_a_multi_kc_model_from_the_factors(tmp_path, capsys):
    """The reference aborts; this reports and proceeds with what is eligible."""
    rows = _mixed("s1", "a", "X", 4) + _mixed("s1", "b", "Y", 4) + \
        _mixed("s2", "a", "X", 4) + _mixed("s2", "b", "Y", 4)
    df = _frame(rows, "clean")
    df["KC (wide)"] = "P~~Q"
    df["Opportunity (wide)"] = df["Opportunity (clean)"] + "~~1"
    export = tmp_path / "export.txt"
    df.to_csv(export, sep="\t", index=False, lineterminator="\n")

    assert _lfa_cli("--min-opportunities", "1", export=str(export)) == 0
    err = capsys.readouterr().err
    assert "excluding 'wide'" in err and "more than one KC" in err


def test_cli_lfa_refuses_an_unknown_model_name(tmp_path, capsys):
    assert _lfa_cli("--factors", "Nope") == 1
    assert "Unknown KC model" in capsys.readouterr().err
    assert _lfa_cli("--compare", "Nope") == 1
    assert "is not a KC model" in capsys.readouterr().err


def test_cli_lfa_lists_the_models_and_exits(capsys):
    from leapfit.cli import main_lfa
    assert main_lfa([EXAMPLE, "--list-models"]) == 0
    assert capsys.readouterr().out.split() == ["Skills", "Topics"]


# --------------------------------------------------------------------------
# What the pairwise merge buys
# --------------------------------------------------------------------------


def _three_step_data():
    rows = []
    for student in ("s1", "s2", "s3"):
        for step in ("a", "b", "c"):
            rows += _mixed(student, step, step, 6)
    return from_frame(_frame(rows), "F")


def _two_factor_matrix(data):
    """P with factors {a} and {b} only — no factor equals the union {a, b}."""
    steps = tuple(sorted(set(data.items)))
    a, b = (s for s in steps if s.endswith(("##a", "##b")))
    return FactorMatrix(steps, ("f", "g"), (frozenset({a}), frozenset({b})))


def test_a_union_of_two_factors_is_reachable_only_by_merging():
    """The justification for the operator, as an exhaustive search.

    Split only ever divides a part, so from the ``All`` root over factors
    ``{a}`` and ``{b}`` the reachable labellings are ``{abc}``, ``{a|bc}``,
    ``{b|ac}`` and ``{a|b|c}``. Putting ``a`` and ``b`` together in one KC and
    leaving ``c`` alone is not among them, and no factor equals ``{a, b}``.
    Carve them out separately and merge, and it is reachable.
    """
    data = _three_step_data()
    P = _two_factor_matrix(data)
    steps = P.steps
    a, b = (next(s for s in steps if s.endswith(f"##{x}")) for x in "ab")
    target = partition(tuple("ab" if s in (a, b) else "c" for s in steps))

    kw = {"max_iterations": 20, "patience": 0, "min_opportunities": 1}
    without = lfa_search(data, P, merges="none", **kw)
    assert without.stopped == "exhausted", "the whole space was enumerated"
    assert target not in {partition(s.labels) for s in without.states}

    with_merge = lfa_search(data, P, merges="pairwise", **kw)
    assert target in {partition(s.labels) for s in with_merge.states}, (
        "a KC that is the union of two factors needs the merge operator")


def test_pairwise_merge_reaches_strictly_more_states_than_splitting_alone():
    data = _three_step_data()
    P = _two_factor_matrix(data)
    kw = {"max_iterations": 20, "patience": 0, "min_opportunities": 1}
    split_only = lfa_search(data, P, merges="none", **kw)
    pairwise = lfa_search(data, P, merges="pairwise", **kw)
    seen = {partition(s.labels) for s in split_only.states}
    grown = {partition(s.labels) for s in pairwise.states}
    assert seen < grown, "a superset, and strictly bigger"


def test_offering_both_merge_operators_explores_at_least_as_much():
    models, P = _example_factors()
    kw = {"max_iterations": 4, "patience": 0}
    counts = {m: lfa_search(models["Topics"], P, merges=m, **kw).n_evaluated
              for m in MERGES}
    assert counts["none"] <= counts["lineage"]
    assert counts["none"] < counts["pairwise"]
    assert counts["both"] >= max(counts.values())


def test_a_merge_is_never_offered_twice_on_one_lineage():
    models, P = _example_factors()
    result = lfa_search(models["Topics"], P, max_iterations=6, merges="both")
    for state in result.states:
        moves = [(m.kind, m.skill, m.factor) for m in state.history]
        assert len(moves) == len(set(moves))


def test_merges_must_be_a_known_mode():
    models, P = _example_factors()
    with pytest.raises(ValueError, match="merges must be one of"):
        lfa_search(models["Topics"], P, merges="pairwise-ish")


def test_the_cli_offers_the_merge_operators(tmp_path):
    out = tmp_path / "f.csv"
    assert _lfa_cli("--merges", "both", "--out", str(out)) == 0
    assert len(pd.read_csv(out)) > 1


def test_merging_pays_off_from_a_fine_grained_model_not_from_the_root():
    """Why the operator's value is coupled to where the search starts.

    From the ``All`` root there is nothing to coarsen: BIC has already judged
    each accepted split worth more than its two parameters, so undoing one
    cannot win, and on a real export 26 scored merge moves changed no answer.
    From an authored *fine* model the same operator is immediately productive
    — which is precisely the situation Cen, Koedinger & Junker (2006) handled
    by merging into the starting model **by hand**.
    """
    data = load_student_step(EXAMPLE, kc_model="Skills")
    steps = tuple(sorted(set(data.items)))
    at = dict(zip(data.items, (kcs[0] for kcs in data.kcs)))
    labels = tuple(at[step] for step in steps)

    def bic(current):
        scored = relabel(data, steps, current)
        design = build_afm_design(scored)
        return fit_afm(design, scored.y, warn_not_converged=False,
                       warn_separated=False).bic

    authored = bic(labels)
    skills = sorted(set(labels))
    best = min(bic(merge(labels, a, b))
               for i, a in enumerate(skills) for b in skills[i + 1:])
    assert best < authored - 10, (
        f"one merge should buy real BIC here: {authored:.3f} -> {best:.3f}")


# --------------------------------------------------------------------------
# Starting from a KC model you already have
# --------------------------------------------------------------------------


def test_the_root_defaults_to_the_all_model():
    """Which is the ``Single-KC`` labelling, built rather than read."""
    models, P = _example_factors()
    assert root_labels(P) == ("all",) * len(P.steps)
    result = lfa_search(models["Topics"], P, max_iterations=1)
    assert result.root.n_kcs == 1


def test_a_step_data_and_a_mapping_describe_the_same_root():
    models, P = _example_factors()
    skills = models["Skills"]
    mapping = dict(zip(skills.items, (kcs[0] for kcs in skills.kcs)))
    assert root_labels(P, skills) == root_labels(P, mapping)
    assert len(set(root_labels(P, skills))) == 12


def test_a_root_must_label_every_step():
    _, P = _example_factors()
    partial = dict(zip(P.steps[:-1], ["x"] * (len(P.steps) - 1)))
    with pytest.raises(ValueError, match="A root must cover every step"):
        root_labels(P, partial)


def test_a_multi_kc_model_cannot_be_a_root():
    _, P = _example_factors()
    df = _frame([{"Anon Student Id": "s1", "Problem Name": "p", "Step Name": "a",
                  "First Attempt": "correct", "kc": "A~~B", "opp": "1~~1"}])
    with pytest.raises(ValueError, match="more than one"):
        root_labels(P, from_frame(df, "F"))


def test_replay_rebuilds_from_the_root_it_was_given():
    P = FactorMatrix(STEPS, ("f",), (frozenset({"a"}),))
    history = (Move("split", "x", "f"),)
    assert replay(history, P, ("x", "x", "y")) == ("x*f", "x", "y")
    with pytest.raises(ValueError, match="degenerate"):
        replay(history, P)          # no skill named "x" under the All root


def test_from_a_fine_root_splitting_can_do_nothing_and_merging_pays():
    """The whole reason ``root=`` and the merge operator belong together.

    Split only refines, and BIC will not buy a refinement of an already-fine
    model — so from the authored 12-KC ``Skills`` labelling a split-only search
    returns the root untouched. The same search with pairwise merge coarsens it
    to 6 KCs and gains 80-odd nats. Neither number is interesting alone; the
    pair is, because it shows the operator was idle from the ``All`` root for a
    structural reason rather than a data one.
    """
    models, P = _example_factors()
    kw = {"root": models["Skills"], "max_iterations": 6, "patience": 0,
          "min_opportunities": 1}
    split_only = lfa_search(models["Skills"], P, merges="none", **kw)
    assert split_only.best.history == (), "no refinement of Skills pays"
    assert split_only.best.bic == split_only.root.bic

    merging = lfa_search(models["Skills"], P, merges="pairwise", **kw)
    assert merging.best.bic < split_only.best.bic - 50
    assert merging.best.n_kcs < merging.root.n_kcs, "it coarsened the model"
    assert all(m.kind == "merge" for m in merging.best.history)


def test_a_separation_already_in_the_root_does_not_refuse_every_move():
    """The screens judge moves, not the state they were handed.

    A move can only make the KCs it *touched* separate, so a separation
    elsewhere is not its doing. Comparing against the whole design instead
    refused 4,251 of 4,277 merge candidates on a 91-KC root that carried three
    separated KCs — every move inherited them, so every move was refused and
    the search scored nothing.
    """
    data = _screen_data()
    root = dict(zip(data.items, (kcs[0] for kcs in data.kcs)))
    P = build_factor_matrix({"F": data})
    result = lfa_search(data, P, root=root, max_iterations=2, patience=0,
                        merges="pairwise", min_opportunities=1)
    assert result.root_separation, "the root's own separated KC is reported"
    assert result.n_evaluated > 1, (
        "and moves are still scored rather than blanket-refused")
    assert "came in with the root" in result.summary()


def test_the_cli_can_start_from_an_authored_kc_model(tmp_path):
    out = tmp_path / "f.csv"
    assert _lfa_cli("--root", "Skills", "--merges", "pairwise",
                    "--out", str(out)) == 0
    assert len(pd.read_csv(out)) > 1
