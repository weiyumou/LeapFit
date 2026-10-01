"""Tests for Logistic Knowledge Tracing.

The load-bearing tests are the two identities. LKT is a generalization of the
other families here, not a neighbour of them, so if
``(student, kc, kc) x (intercept, intercept, lineafm$)`` is not AFM's design
column for column, and ``(kc, kc, kc) x (intercept, linesuc$, linefail$)`` is
not PFA's, then the term algebra means something other than what the reference
says it means and every downstream comparison is between two different models.

Everything else falls into three groups: what a count *is* (strictly prior,
accumulated within student x level, over the one canonical practice order),
what the module refuses and why, and whether the parameter count stays honest
when a spec asks for something the shared identification pass cannot break.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from leapfit import (
    LOGITDEC_WINDOW,
    PARAMETER_TOLERANCE,
    REFERENCE_COST,
    LKTFit,
    Term,
    build_afm_design,
    build_lkt_design,
    build_pfa_design,
    component_labels,
    cross_validate,
    design_terms,
    fit_afm,
    fit_lkt,
    fit_lkt_pars,
    fit_pfa,
    history_counts,
    lkt_terms,
)
from leapfit.lkt import PARAMETER_STEP, _central_differences

from helpers import EPOCH, clocked_data, step_data, step_row

AFM_SPEC = (("student", "kc", "kc"), ("intercept", "intercept", "lineafm$"))
PFA_SPEC = (("kc", "kc", "kc"), ("intercept", "linesuc$", "linefail$"))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _dense(design):
    return design.matrix.toarray()


# --------------------------------------------------------------------------
# The identities: AFM and PFA are LKT specifications
# --------------------------------------------------------------------------


def test_the_afm_specification_reproduces_the_afm_design(example):
    """Same columns, same order, same aliasing, same likelihood."""
    afm = build_afm_design(example, recompute_opportunities=True)
    lkt = build_lkt_design(example, lkt_terms(*AFM_SPEC))

    assert [b.matrix.shape[1] for b in lkt.blocks] == [b.matrix.shape[1] for b in afm.blocks]
    assert [b.columns for b in lkt.blocks] == [b.columns for b in afm.blocks]
    np.testing.assert_array_equal(_dense(lkt), _dense(afm))
    assert lkt.aliased.columns == afm.aliased.columns
    assert lkt.n_params == afm.n_params

    assert (fit_lkt(lkt, example.y).ll_unpenalized
            == pytest.approx(fit_afm(afm, example.y).ll_unpenalized, abs=1e-9))


def test_the_pfa_specification_reproduces_the_pfa_design(example):
    pfa = build_pfa_design(example)
    lkt = build_lkt_design(example, lkt_terms(*PFA_SPEC))

    np.testing.assert_array_equal(_dense(lkt), _dense(pfa))
    assert [b.columns for b in lkt.blocks] == [b.columns for b in pfa.blocks]
    assert (fit_lkt(lkt, example.y).ll_unpenalized
            == pytest.approx(fit_pfa(pfa, example.y).ll_unpenalized, abs=1e-9))


def test_features_without_a_dollar_reproduce_pooled_pfa(example):
    """No ``$`` is one shared coefficient — PFA's ``slopes="pooled"``."""
    pooled = build_pfa_design(example, slopes="pooled")
    lkt = build_lkt_design(example, lkt_terms(("kc", "kc", "kc"),
                                              ("intercept", "linesuc", "linefail")))
    np.testing.assert_array_equal(_dense(lkt), _dense(pooled))


def test_the_afm_identity_is_against_recomputed_opportunities():
    """LKT counts practice itself; AFM reads DataShop's column by default.

    The two disagree wherever the export's row order contradicts its
    timestamps, which is the case this fixture builds: two attempts whose
    ``Opportunity`` values follow row order while the times run the other way.
    """
    rows = [step_row("s1", "st1", 1, "A", 1, time="2024-01-01 00:00:02"),
            step_row("s1", "st2", 0, "A", 2, time="2024-01-01 00:00:01")]
    data = step_data(rows)
    assert data.opportunity_disagreements().size == 2

    lkt = build_lkt_design(data, lkt_terms(("kc",), ("lineafm$",)), identify=False)
    from_file = build_afm_design(data, identify=False)
    recomputed = build_afm_design(data, recompute_opportunities=True, identify=False)

    slope = _dense(lkt)[:, 0]
    np.testing.assert_array_equal(slope, _dense(recomputed)[:, -1])
    assert not np.array_equal(slope, _dense(from_file)[:, -1])


# --------------------------------------------------------------------------
# Counts: what "prior" means, and over what
# --------------------------------------------------------------------------


def test_counts_on_the_student_component_span_that_students_whole_history():
    """A feature on the student counts everything that student did before,
    whatever KC it was on — the reference's index is (level, student), and for
    the student component the level *is* the student."""
    rows = [step_row("s1", "st1", 1, "A", 1), step_row("s1", "st2", 0, "B", 1),
            step_row("s1", "st3", 1, "A", 2), step_row("s2", "st4", 0, "A", 1)]
    data = step_data(rows)
    s, f = history_counts(data, component_labels(data, "student"))
    assert [row[0] for row in s] == [0, 1, 1, 0]
    assert [row[0] for row in f] == [0, 0, 1, 0]


def test_counts_on_an_item_never_repeated_stay_zero(example):
    """Which is why an item-level practice feature is not estimable here: the
    column is identically zero and identification drops it."""
    s, f = history_counts(example, component_labels(example, "item"))
    assert all(row == (0,) for row in s) and all(row == (0,) for row in f)

    design = build_lkt_design(example, [Term("kc", "intercept"),
                                        Term("item", "lineafm")])
    assert design.slices()["lineafm[item]"] == slice(4, 4)
    assert "lineafm[item]:lineafm" in design.aliased.columns


def test_history_counts_checks_that_the_labels_match_the_data(example):
    with pytest.raises(ValueError, match="observations but"):
        history_counts(example, example.kcs[:-1])


# --------------------------------------------------------------------------
# Feature values
# --------------------------------------------------------------------------


def _column(data, feature, pars=()):
    design = build_lkt_design(data, [Term("kc", feature, per_level=True, pars=pars)],
                              identify=False)
    return _dense(design)[:, 0]


@pytest.fixture
def streak():
    """One student, one KC, outcomes correct, incorrect, correct, correct."""
    return step_data([step_row("s1", f"st{i}", y, "A", i + 1)
                      for i, y in enumerate([1, 0, 1, 1])])


@pytest.mark.parametrize("feature,expected", [
    ("intercept", [1, 1, 1, 1]),
    ("lineafm", [0, 1, 2, 3]),
    ("linesuc", [0, 1, 1, 2]),
    ("linefail", [0, 0, 1, 1]),
    ("linecomp", [0, 1, 0, 1]),
    ("logafm", [math.log1p(t) for t in (0, 1, 2, 3)]),
    ("logsuc", [math.log1p(s) for s in (0, 1, 1, 2)]),
    ("logfail", [math.log1p(f) for f in (0, 0, 1, 1)]),
    ("prop", [0.5, 1.0, 0.5, 2 / 3]),  # the reference guards 0/0 with .5, not a dropped row
])
def test_static_feature_values_match_their_definitions(streak, feature, expected):
    np.testing.assert_allclose(_column(streak, feature), expected)


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


def test_components_resolve_from_the_parse_and_from_the_source_table(example):
    assert component_labels(example, "student")[0] == (example.students[0],)
    assert component_labels(example, "item")[0] == (example.items[0],)
    assert component_labels(example, "kc") == list(example.kcs)
    from_source = component_labels(example, "Problem Name")
    assert all(len(row) == 1 for row in from_source)
    assert len(set(from_source)) < len(example)


def test_kc_is_the_only_component_that_can_put_two_levels_on_one_row():
    data = step_data([step_row("s1", "st1", 1, "A~~B", "1~~1"),
                      step_row("s1", "st2", 0, "A", 2)])
    assert component_labels(data, "kc") == [("A", "B"), ("A",)]
    assert component_labels(data, "student") == [("s1",), ("s1",)]


def test_a_shared_coefficient_sums_a_rows_levels():
    """Two KCs on one step contribute additively, as they do in AFM — so the
    pooled column is the row total, matching PFA's pooled construction."""
    rows = [step_row("s1", "st1", 1, "A~~B", "1~~1"),
            step_row("s1", "st2", 0, "A~~B", "2~~2")]
    data = step_data(rows)
    pooled = _dense(build_lkt_design(data, [Term("kc", "lineafm")], identify=False))
    per_level = _dense(build_lkt_design(data, [Term("kc", "lineafm", per_level=True)],
                                        identify=False))
    np.testing.assert_allclose(pooled[:, 0], per_level.sum(axis=1))
    np.testing.assert_allclose(pooled[:, 0], [0.0, 2.0])


def test_an_unknown_component_names_the_columns_that_exist(example):
    with pytest.raises(KeyError, match="Available columns"):
        component_labels(example, "Not A Column")


def test_a_source_component_needs_the_source_table(example):
    stripped = replace(example, source=None, source_rows=None)
    with pytest.raises(ValueError, match="no source table"):
        component_labels(stripped, "Problem Name")


# --------------------------------------------------------------------------
# Terms, notation, round-tripping
# --------------------------------------------------------------------------


def test_terms_parse_the_references_dollar_notation():
    per_level = Term.parse("kc", "lineafm$")
    shared = Term.parse("kc", "lineafm")
    assert per_level.per_level and not shared.per_level
    assert per_level.notation() == "lineafm$"
    assert str(shared) == "kc:lineafm"


def test_the_student_alias_follows_the_references_column_name():
    assert Term.parse("Anon.Student.Id", "intercept").component == "student"


def test_intercepts_are_always_per_level():
    """A factor is expanded whether or not the spec writes the ``$``, and it is
    written without one."""
    assert Term("kc", "intercept").per_level
    assert Term.parse("kc", "intercept$") == Term("kc", "intercept")
    assert Term("kc", "intercept").notation() == "intercept"


def test_design_terms_round_trip_through_the_block_names(example):
    terms = lkt_terms(("student", "kc", "kc", "item"),
                      ("intercept", "intercept", "lineafm$", "logsuc"),
                      pars=(None, None, None, None))
    design = build_lkt_design(example, terms)
    assert design_terms(design) == terms


def test_a_fixed_parameter_is_part_of_the_terms_identity(example):
    terms = (Term("kc", "powafm", per_level=True, pars=0.5),
             Term("kc", "powafm", per_level=True, pars=0.8))
    design = build_lkt_design(example, terms, identify=False)
    assert [b.name for b in design.blocks] == ["powafm$(0.5)[kc]", "powafm$(0.8)[kc]"]
    assert design_terms(design) == terms


def test_the_fit_prints_the_specification_it_fitted(example):
    fit = fit_lkt(build_lkt_design(example, lkt_terms(*AFM_SPEC)), example.y)
    assert isinstance(fit, LKTFit)
    assert fit.notation() == "student:intercept + kc:intercept + kc:lineafm$"


# --------------------------------------------------------------------------
# Refusals — a spec that cannot mean what it says does not build
# --------------------------------------------------------------------------


@pytest.mark.parametrize("feature,reason", [
    ("errordec", "pred_ed"),
    ("recencystudy", "commented out"),
    ("recencytest", "commented out"),
    ("dashfail", "parlength"),
    ("baseratepropdec", "grouping this module"),
    ("logitdecevol", "grouping this module"),
    ("base5suc", "fifth parameter"),
    ("diffcor1", "prior fit"),
])
def test_deferred_features_are_refused_by_name_with_the_reason(feature, reason):
    """Three of these the reference cannot compute either — the refusal names
    the defect rather than pretending the feature is merely unimplemented."""
    with pytest.raises(NotImplementedError, match=reason):
        Term("kc", feature)


def test_an_unknown_feature_lists_the_ones_that_exist():
    with pytest.raises(ValueError, match="Implemented: "):
        Term("kc", "nonsense")


def test_a_random_effect_is_refused_rather_than_silently_fixed():
    with pytest.raises(NotImplementedError, match="glmer"):
        Term.parse("kc", "intercept@")


def test_a_shape_parameter_is_required_where_it_exists_and_refused_where_it_does_not():
    with pytest.raises(ValueError, match="takes 1 parameter, got 0"):
        Term("kc", "powafm")
    with pytest.raises(ValueError, match="takes no parameter"):
        Term("kc", "lineafm", pars=0.5)
    with pytest.raises(ValueError, match="takes 4 parameters, got 2"):
        Term("kc", "ppe", pars=(0.3, 0.2))


def test_a_repeated_term_is_refused(example):
    with pytest.raises(ValueError, match="specified twice"):
        build_lkt_design(example, [Term("kc", "lineafm"), Term("kc", "lineafm")])


def test_an_empty_specification_is_refused(example):
    with pytest.raises(ValueError, match="at least one term"):
        build_lkt_design(example, [])


def test_a_non_positive_cost_is_refused(example):
    with pytest.raises(ValueError, match="cost must be positive"):
        build_lkt_design(example, [Term("kc", "intercept")], cost=0.0)


# --------------------------------------------------------------------------
# Identification: the parameter count stays honest or the design does not build
# --------------------------------------------------------------------------


def test_the_student_kc_redundancy_is_broken_exactly_once(example):
    design = build_lkt_design(example, lkt_terms(*AFM_SPEC))
    assert design.n_params == design.rank()
    assert len(design.aliased) == 1
    assert design.aliased.columns[0].startswith("student:")


def test_intercepts_on_two_crossed_components_are_identified(example):
    """Students and items cross, so the only dependence between their two
    intercept blocks is the all-ones one, and one reference level breaks it."""
    design = build_lkt_design(example, [Term("student", "intercept"),
                                        Term("item", "intercept")])
    assert design.n_params == design.rank()
    assert len(design.aliased) == 1
    assert design.aliased.columns[0].startswith("student:")
    assert "sum redundancy across student, intercept[item]" in design.aliased.reasons[0]


def test_a_nested_component_gets_one_reference_level_per_component(example):
    """Every item here belongs to exactly one KC, so the KC/item graph falls
    apart into one component per KC and each carries its own redundancy. Four
    KCs, four reference levels — which is the whole point of doing this per
    component rather than once. The student block leads, then latest-declared
    first, so the KCs named before the items nested in them keep every level
    and the items give way."""
    design = build_lkt_design(example, [Term("kc", "intercept"),
                                        Term("item", "intercept")])
    assert design.n_params == design.rank()
    assert len(design.aliased) == len(example.kc_names) == 4
    assert all(c.startswith("intercept[item]:") for c in design.aliased.columns)
    assert all("component" in reason for reason in design.aliased.reasons)


def test_a_coarser_component_named_after_the_one_nested_in_it_is_refused(example):
    """Reversed, the same rule would take *every* KC: each KC intercept is the
    sum of its items', so the items already span the KC block. That is a
    hierarchical parent declared after the levels it groups — a block that
    adds nothing, not a factor with a reference level — and it is refused,
    naming the block that spans it, rather than silently dropped whole."""
    spec = [Term("item", "intercept"), Term("kc", "intercept")]
    unidentified = build_lkt_design(example, spec, identify=False)
    assert unidentified.rank() == unidentified.n_params - len(example.kc_names), "the premise"
    with pytest.raises(ValueError, match=r"kc_intercept adds nothing to this design: every "
                                         r"column of it lies in the span of intercept\[item\]"):
        build_lkt_design(example, spec)
    with pytest.raises(ValueError, match="kc_intercept adds nothing"):
        build_lkt_design(example, [Term("student", "intercept"), *spec])


def test_a_nested_component_is_identified_under_a_student_intercept_too(example):
    """Adding a student intercept to the nested pair above connects the whole
    design, but not the KC/item graph: the nesting still lives there, one
    redundancy per KC, beside the student's own. Cut from the whole design's
    graph instead, this looked like one component and was refused."""
    spec = [Term("student", "intercept"), Term("kc", "intercept"),
            Term("item", "intercept")]
    unidentified = build_lkt_design(example, spec, identify=False)
    n_kcs = len(example.kc_names)
    assert unidentified.rank() == unidentified.n_params - (n_kcs + 1), "the premise"

    design = build_lkt_design(example, spec)
    assert design.n_params == design.rank()
    dropped = design.aliased.by_block()
    assert len(dropped["student"]) == 1 and len(dropped["intercept[item]"]) == n_kcs
    assert "kc_intercept" not in dropped, "declared before the items, so kept whole"
    assert design.aliased.reasons[0] == "reference level (student/KC sum redundancy)"


def test_a_dependence_beyond_the_all_ones_one_still_raises(example):
    """The pass models sum redundancies between factors exactly and nothing
    else. ``lineafm`` is ``linesuc`` plus ``linefail``, level by level — a
    dependence among blocks that are not factors at all — and it says so
    rather than undercounting parameters."""
    spec = lkt_terms(("kc", "kc", "kc"), ("lineafm$", "linesuc$", "linefail$"))
    unidentified = build_lkt_design(example, spec, identify=False)
    assert unidentified.rank() < unidentified.n_params, "the premise"
    with pytest.raises(ValueError, match="still rank-deficient"):
        build_lkt_design(example, spec)


def test_the_same_factor_under_two_names_is_refused_rather_than_halved(example):
    """The export's own KC column read as a component partitions the rows
    exactly as the parsed KC does. That is nested in the extreme — every level
    pairs with one level of the other — and it is a mistake in the
    specification, not a property of the data, so it is refused, naming the
    block that already spans it, rather than resolved by silently dropping one
    copy whole."""
    spec = [Term("kc", "intercept"), Term("KC (Topics)", "intercept")]
    with pytest.raises(ValueError, match=r"intercept\[KC \(Topics\)\] adds nothing to this "
                                         r"design: every column of it lies in the span of "
                                         r"kc_intercept"):
        build_lkt_design(example, spec)


def _cohorts(tags, conditions=None, n_students=3, n_steps=30, seed=0):
    """Cohorts that share no students: cohort ``c`` draws each step's KC tag
    from ``tags[c]`` (``~~`` for a step carrying two) and, where given, its
    ``Condition`` from ``conditions[c]``."""
    rng = np.random.default_rng(seed)
    rows, seen = [], {}
    for c, kcs in enumerate(tags):
        for s in range(n_students):
            for j in range(n_steps):
                kc = str(rng.choice(kcs))
                n = seen[(c, s, kc)] = seen.get((c, s, kc), 0) + 1
                extra = {} if conditions is None else {"Condition": str(rng.choice(conditions[c]))}
                rows.append(step_row(f"c{c}s{s}", f"st{j}", int(rng.random() < 0.6), kc,
                                     "~~".join([str(n)] * len(kc.split("~~"))), **extra))
    return step_data(rows)


def test_a_feature_that_touches_every_row_does_not_merge_cohorts():
    """``propdec`` starts every level at a half, so its column is nonzero on
    every row and joins the graph over the blocks that cover every row. Two
    cohorts sharing no students and no KCs merge there — but not on the
    student/KC pair's own graph, which is where their redundancies live, one
    per cohort, exactly as without the feature."""
    data = _cohorts([["A", "B"], ["C", "D"]])
    afm = build_lkt_design(data, lkt_terms(*AFM_SPEC))
    terms = lkt_terms(("student", "kc", "kc"), ("intercept", "intercept", "propdec"),
                      (None, None, 0.9))
    assert len(set(build_lkt_design(data, terms, identify=False).row_components())) == 1

    design = build_lkt_design(data, terms)
    assert design.n_params == design.rank()
    assert design.aliased == afm.aliased
    assert design.aliased.reasons[1] == ("reference level (student/KC sum redundancy, "
                                         "component 2 of 2)")


def test_cohorts_joined_only_through_a_third_block_keep_a_reference_level_each():
    """Students and conditions never cross between the two cohorts, so their
    pair carries one redundancy per cohort, however well the KCs both cohorts
    share connect everything else — and the student/KC and condition/KC pairs
    carry one more between them. Cut from the whole design's graph, the
    cohorts looked like one and a reference level went missing."""
    spec = [Term("student", "intercept"), Term("Condition", "intercept"),
            Term("kc", "intercept")]
    conditions = [["c1", "c2"], ["c3", "c4"]]

    single = _cohorts([["K1", "K2", "K3"]] * 2, conditions)
    unidentified = build_lkt_design(single, spec, identify=False)
    assert unidentified.rank() == unidentified.n_params - 3, "the premise"
    design = build_lkt_design(single, spec)
    assert design.n_params == design.rank()
    dropped = design.aliased.by_block()
    assert sorted(s[:2] for s in dropped["student"]) == ["c0", "c1"], "one per cohort"
    assert len(dropped["kc_intercept"]) == 1
    assert "intercept[Condition]" not in dropped, "declared first, so kept whole"

    # Steps carrying one KC or two: the KC block covers every row without
    # partitioning it, so it has no redundancy of its own to add — but it
    # joined the cohorts all the same.
    multi = _cohorts([["K1", "K2", "K1~~K2"], ["K1", "K3", "K1~~K3"]], conditions)
    unidentified = build_lkt_design(multi, spec, identify=False)
    assert unidentified.rank() == unidentified.n_params - 2, "the premise"
    design = build_lkt_design(multi, spec)
    assert design.n_params == design.rank()
    assert sorted(design.aliased.by_block()) == ["student"]
    assert len(design.aliased) == 2


def test_a_single_intercept_on_any_component_is_fine(example):
    for component in ("student", "kc", "item", "Problem Name"):
        design = build_lkt_design(example, [Term(component, "intercept")])
        assert design.n_params == design.rank()


def test_a_level_with_no_second_opportunity_reports_nan_not_zero():
    """AFM's never-practised-twice rule, at the level of any component: the
    slope was not estimated, and printing 0.0 would invite it into a
    low-slope screen that reads zero as "students did not learn"."""
    rows = [step_row("s1", "st1", 1, "A", 1), step_row("s1", "st2", 0, "A", 2),
            step_row("s1", "st3", 1, "B", 1)]
    data = step_data(rows)
    design = build_lkt_design(data, lkt_terms(("kc", "kc"), ("intercept", "lineafm$")))
    assert "lineafm$[kc]:B" in design.aliased.columns

    values = fit_lkt(design, data.y, warn_separated=False,
                     warn_not_converged=False).component_values(data, "kc")
    row = values.set_index("Level").loc["B"]
    assert np.isnan(row["lineafm$"])
    assert np.isfinite(row["intercept"])


# --------------------------------------------------------------------------
# The reference's penalty
# --------------------------------------------------------------------------


def test_cost_is_the_reference_ridge_on_every_column(example):
    design = build_lkt_design(example, lkt_terms(*AFM_SPEC), cost=REFERENCE_COST)
    np.testing.assert_allclose(design.l2, 1.0 / REFERENCE_COST)

    fit = fit_lkt(design, example.y)
    assert fit.penalty > 0.0
    # The reference reports the *unpenalized* likelihood of a penalized fit —
    # the opposite convention to LearnSphere's AFM, which reports the penalized
    # objective as if it were a likelihood. Both are available here.
    assert fit.ll == pytest.approx(fit.ll_unpenalized - fit.penalty)


def test_the_reference_ridge_hides_a_separation_the_default_reports():
    """``Design.separated`` skips penalized columns, correctly: a ridge
    supplies the curvature the likelihood is missing. So the same spec on the
    same data reports a separated student without ``cost`` and none with it —
    which is what the reference's always-on penalty is doing."""
    rows = ([step_row("s1", f"st{i}", 1, "A", i + 1) for i in range(4)]
            + [step_row("s2", f"st{i}", i % 2, "A", i + 1) for i in range(4)])
    data = step_data(rows)
    terms = [Term("student", "intercept")]

    unpenalized = build_lkt_design(data, terms)
    assert list(unpenalized.separated(data.y).columns) == ["student:s1"]

    penalized = build_lkt_design(data, terms, cost=REFERENCE_COST)
    assert len(penalized.separated(data.y)) == 0


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_component_values_gives_one_row_per_level_and_one_column_per_feature(example):
    fit = fit_lkt(build_lkt_design(example, lkt_terms(*AFM_SPEC)), example.y)
    values = fit.component_values(example, "kc")
    assert list(values["Level"]) == example.kc_names
    assert list(values.columns) == ["Level", "intercept", "lineafm$", "Separated"]
    assert not values["Separated"].any()


def test_a_shared_coefficient_is_broadcast_to_every_level(example):
    fit = fit_lkt(build_lkt_design(example, lkt_terms(("kc", "kc"),
                                                      ("intercept", "lineafm"))),
                  example.y)
    values = fit.component_values(example, "kc")
    assert values["lineafm"].nunique() == 1
    assert values["intercept"].nunique() == len(example.kc_names)


def test_component_values_refuses_a_component_the_fit_does_not_carry(example):
    fit = fit_lkt(build_lkt_design(example, lkt_terms(*PFA_SPEC)), example.y)
    with pytest.raises(ValueError, match="No term on component 'student'"):
        fit.component_values(example, "student")


def test_the_reported_statistics_are_the_shared_ones(example):
    """LKT itself reports no parameter count, no AIC and no BIC; here they come
    from the same place every other family's do."""
    fit = fit_lkt(build_lkt_design(example, lkt_terms(*AFM_SPEC)), example.y)
    assert fit.label == "LKT"
    assert fit.n_params == fit.design.rank()
    assert fit.bic == pytest.approx(-2 * fit.ll + fit.n_params * np.log(fit.n_obs))
    assert fit.is_optimal


# --------------------------------------------------------------------------
# numer, the one feature that reads a column instead of a history
# --------------------------------------------------------------------------


def test_numer_reads_its_component_as_a_number():
    rows = [step_row("s1", f"st{i}", i % 2, "A", i + 1, **{"Prior Score": str(i * 2)})
            for i in range(3)]
    data = step_data(rows)
    design = build_lkt_design(data, [Term("Prior Score", "numer")], identify=False)
    np.testing.assert_allclose(_dense(design)[:, 0], [0.0, 2.0, 4.0])


def test_numer_refuses_a_column_that_is_not_a_number(example):
    with pytest.raises(ValueError, match="do not parse"):
        build_lkt_design(example, [Term("Problem Name", "numer")], identify=False)


def test_numer_refuses_the_dollar_suffix():
    with pytest.raises(ValueError, match="no levels to extend over"):
        Term("Prior Score", "numer", per_level=True)


# --------------------------------------------------------------------------
# Composition with the shared layer
# --------------------------------------------------------------------------


def test_cross_validation_runs_over_an_lkt_design(example):
    design = build_lkt_design(example, lkt_terms(*AFM_SPEC))
    result = cross_validate(design, example, scheme="item_blocked", n_folds=3, seed=0)
    assert len(result.folds) == 3
    assert 0.0 < result.rmse < 1.0


# --------------------------------------------------------------------------
# A term that needs a clock the export does not keep
# --------------------------------------------------------------------------


def test_a_term_that_needs_a_clock_names_itself_when_the_export_has_none(example):
    """``example`` has times but no durations, and ``base2`` needs the time on
    task that durations give."""
    with pytest.raises(ValueError, match=r"Term kc:base2\$\(0.3,0.5\) cannot be computed"):
        build_lkt_design(example, [Term("kc", "base2", per_level=True, pars=(0.3, 0.5))])


# --------------------------------------------------------------------------
# Feature semantics, against naive transcriptions of the reference's own
# definitions. The module computes these with recurrences and vectorized
# segment arithmetic; the oracles below are the formulas as written.
# --------------------------------------------------------------------------


def _expdec(v, d):
    return sum(v[j] * d ** (len(v) - 1 - j) for j in range(len(v)))


def _slide_expdec(x, d):
    running = [_expdec(x[:i + 1], d) for i in range(len(x))]
    return np.array([0.0, *running[:-1]])


def _ghost_counts(v, d):
    """``corv``/``incorv``: one ghost success and one ghost failure, decayed."""
    w = len(v)
    corv = sum([1.0, *v][j] * d ** (w - j) for j in range(w + 1))
    incorv = sum([1.0, *(1 - a for a in v)][j] * d ** (w - j) for j in range(w + 1))
    return corv, incorv


def _slide_propdec(x, d):
    running = [np.divide(*(lambda c, i: (c, c + i))(*_ghost_counts(x[:i + 1], d)))
               for i in range(len(x))]
    return np.array([0.5, *running[:-1]])


def _slide_logitdec(x, d, window=60):
    running = []
    for i in range(len(x)):
        corv, incorv = _ghost_counts(x[max(0, i - window):i + 1], d)
        running.append(math.log(corv / incorv))
    return np.array([0.0, *running[:-1]])


def _baselevel(age, d):
    return np.array([0.0, *(age[i] ** -d for i in range(1, len(age)))])


@pytest.fixture
def clocked():
    """Seven attempts, one of the gaps a whole day, one of them five seconds."""
    outcomes = [1, 0, 1, 1, 0, 1, 0]
    gaps = [0, 10, 20, 70, 300, 86400, 5]
    return clocked_data(outcomes, gaps, durations=[5, 7, 9, 4, 6, 8, 3]), \
        np.array(outcomes, dtype=float), np.cumsum(np.asarray(gaps, dtype=float))


def test_the_decay_features_lag_their_own_trial(clocked):
    """Every one of these is shifted by a position: the reference's slide
    functions return ``c(seed, v[1:n-1])``, so nothing regresses on itself.
    The ghost trials are what make position 0 defined instead of 0/0, which the
    oracles start at a half for ``propdec`` and at zero for ``logitdec``."""
    data, y, _ = clocked
    d = 0.85
    np.testing.assert_allclose(_column(data, "expdecafm", d), _slide_expdec(np.ones(7), d))
    np.testing.assert_allclose(_column(data, "expdecsuc", d), _slide_expdec(y, d))
    np.testing.assert_allclose(_column(data, "expdecfail", d), _slide_expdec(1 - y, d))
    np.testing.assert_allclose(_column(data, "propdec", d), _slide_propdec(y, d))
    np.testing.assert_allclose(_column(data, "logitdec", d), _slide_logitdec(y, d))


def test_logitdec_truncates_at_the_references_sixty_trial_window():
    """Undocumented in the paper, and it bites: at ``d = .97`` the window
    changes the feature by 0.06 logits over 200 trials, so a package that
    quietly used the whole history would not reproduce the reference."""
    rng = np.random.default_rng(0)
    outcomes = rng.integers(0, 2, 200).tolist()
    data = clocked_data(outcomes, [0] + [60] * 199)
    d = 0.97
    windowed = _slide_logitdec(np.asarray(outcomes, dtype=float), d)
    whole = _slide_logitdec(np.asarray(outcomes, dtype=float), d, window=10_000)

    np.testing.assert_allclose(_column(data, "logitdec", d), windowed, atol=1e-12)
    assert np.abs(windowed - whole).max() > 0.01, "the window has to matter here"
    assert LOGITDEC_WINDOW == 61


def test_recency_reads_only_the_last_interval(clocked):
    """Older practice is invisible to it — that is the whole feature."""
    data, y, times = clocked
    spacing = np.array([0.0, *np.diff(times)])
    expected = np.array([0.0, *(s ** -0.5 for s in spacing[1:])])
    np.testing.assert_allclose(_column(data, "recency", 0.5), expected)

    previous = np.array([0.0, *y[:-1]])
    np.testing.assert_allclose(_column(data, "recencysuc", 0.5), previous * expected)
    np.testing.assert_allclose(_column(data, "recencyfail", 0.5), (1 - previous) * expected)


def test_base_is_practice_scaled_by_a_power_law_decay_of_its_age(clocked):
    data, _, times = clocked
    n = np.arange(7, dtype=float)
    level = _baselevel(times - times[0], 0.3)
    np.testing.assert_allclose(_column(data, "base", 0.3), np.log1p(n) * level)


def test_base2_counts_time_away_from_the_system_at_its_second_parameter(clocked):
    """With the session weight at 1 it is ``base`` on real time; at 0 it is
    ``base`` on time spent working. Nothing else in the model changes."""
    data, _, _ = clocked
    on_task = data.time_on_task()
    n = np.arange(7, dtype=float)

    real = _column(data, "base2", (0.3, 1.0))
    np.testing.assert_allclose(real, _column(data, "base", 0.3), atol=1e-12)

    worked = _column(data, "base2", (0.3, 0.0))
    np.testing.assert_allclose(
        worked, np.log1p(n) * _baselevel(on_task - on_task[0], 0.3), atol=1e-12)


def test_dash_decays_its_count_of_prior_practice_with_real_time(clocked):
    """A day's gap at a one-day scale should cost a factor of ``e``."""
    data, _, _ = clocked
    value = _column(data, "dashafm", 1.0)
    assert value[0] == 0.0
    assert np.all(np.isfinite(value))
    # Position 5 follows the 86,400-second gap; position 4 follows 300 seconds.
    assert value[5] < value[4]


def test_the_mean_spacing_sentinel_selects_base4s_unspaced_branch(clocked):
    """The reference marks the second practice with a mean spacing of -1 —
    there is no interval between prior practices yet — and ``base4`` branches
    on that rather than on the position."""
    data, _, times = clocked
    n = np.arange(7, dtype=float)
    on_task = data.time_on_task()
    decay, session, power, unspaced = 0.19, 0.63, 0.055, 0.5

    intage = on_task - on_task[0]
    age = (times - times[0] - intage) * session + intage
    level = _baselevel(age, decay)
    value = _column(data, "base4", (decay, session, power, unspaced))

    # Positions 0 and 1 take the unspaced branch, which is a flat multiplier.
    assert value[0] == 0.0
    assert value[1] == pytest.approx(unspaced * math.log1p(n[1]) * level[1])
    assert value[2] != pytest.approx(unspaced * math.log1p(n[2]) * level[2])


def test_ppe_weights_recent_practice_more_heavily(clocked):
    """Its weighted mean time-since-practice has to sit inside the range of
    the individual ones, and below their unweighted mean once the spacing is
    uneven — that is what the weighting is for."""
    data, _, _ = clocked
    value = _column(data, "ppe", (0.35, 0.20, 0.30, 0.97))
    assert value[0] == 0.0                       # no prior practice
    assert np.all(np.isfinite(value))
    assert np.all(value[1:] > 0.0)


def test_a_feature_that_divides_by_a_zero_interval_is_refused_not_infinite():
    """Two attempts on one level at the same timestamp make an age of zero,
    and the reference raises it to a negative power and hands ``Inf`` to its
    solver."""
    data = clocked_data([1, 0, 1], [0, 0, 60])
    with pytest.raises(ValueError, match="non-finite"):
        build_lkt_design(data, [Term("kc", "base", per_level=True, pars=0.3)],
                         identify=False)


def test_powafm_and_logit_read_the_plain_counts(clocked):
    data, y, _ = clocked
    n = np.arange(7, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(y)[:-1]])
    np.testing.assert_allclose(_column(data, "powafm", 0.5), n ** 0.5)
    np.testing.assert_allclose(
        _column(data, "logit", 0.02),
        np.log((0.1 + 30 * 0.02 + s) / (0.1 + 30 * 0.02 + (n - s))))


# --------------------------------------------------------------------------
# Terms with more than one parameter
# --------------------------------------------------------------------------


def test_a_multi_parameter_term_round_trips_through_its_block_name(example):
    """All four parameters are part of the block name, so two ``ppe`` terms at
    different settings are two blocks rather than a collision — and they are
    written exactly, because the block name is where a design keeps its
    specification. Six significant digits would hand back different numbers
    from the ones fitted, and the vignette's own seeds already carry seven."""
    pars = (0.3491901, 0.2045801, 1e-05, 0.9734477)
    term = Term("kc", "ppe", per_level=True, pars=pars)
    assert term.block_name == "ppe$(0.3491901,0.2045801,1e-05,0.9734477)[kc]"

    design = build_lkt_design(example, [term], identify=False)
    recovered = design_terms(design)[0]
    assert recovered == term
    assert recovered.pars == pars

    # Exact is not the same as long: a value written plainly prints plainly.
    assert Term("kc", "powafm", pars=1.0).block_name == "powafm(1)[kc]"


def test_a_scalar_parameter_is_accepted_for_the_single_parameter_features():
    assert Term("kc", "recency", pars=0.5).pars == (0.5,)
    assert Term.parse("kc", "recency$", 0.5).pars == (0.5,)
    assert Term("kc", "recency", pars=(0.5,)) == Term("kc", "recency", pars=0.5)


def test_lkt_terms_takes_one_par_entry_per_term(example):
    terms = lkt_terms(("student", "kc", "kc"),
                      ("intercept", "logitdec", "ppe"),
                      (None, 0.9, (0.3, 0.2, 0.1, 0.97)))
    assert terms[1].pars == (0.9,)
    assert terms[2].pars == (0.3, 0.2, 0.1, 0.97)
    with pytest.raises(ValueError, match="par entr"):
        lkt_terms(("kc", "kc"), ("intercept", "logitdec"), (0.9,))


def test_time_on_task_refuses_a_duration_it_cannot_accumulate_past():
    """DataShop writes "." where it could not compute a step duration.
    Accumulating past it would make every later step of that student NaN, and
    the feature reading it would then fail complaining about timestamps."""
    rows = [step_row("s1", f"st{i}", i % 2, "A", i + 1,
                     time=str(EPOCH + pd.Timedelta(seconds=60 * i)),
                     **{"Step Duration (sec)": "." if i == 1 else 5})
            for i in range(3)]
    with pytest.raises(ValueError, match="1 of 3 rows have no 'Step Duration"):
        step_data(rows).time_on_task()


# --------------------------------------------------------------------------
# Fitting the feature parameters: the profile likelihood
# --------------------------------------------------------------------------


PROFILE_SPEC = (("student", "kc", "kc", "kc"),
                ("intercept", "intercept", "logitdec", "recency"))


def _profile_terms(logitdec=0.9, recency=0.5):
    return lkt_terms(*PROFILE_SPEC, (None, None, logitdec, recency))


@pytest.fixture(scope="module")
def profile(example):
    return fit_lkt_pars(example, _profile_terms())


def test_the_profile_improves_on_its_own_seed(example, profile):
    """The weakest thing a search must do, and the one that catches a sign
    error in the objective: end no worse than where it started."""
    seeded = fit_lkt(build_lkt_design(example, _profile_terms()), example.y,
                     warn_not_converged=False)
    assert profile.fit.ll >= seeded.ll - 1e-9
    assert profile.n_free == 2
    assert profile.n_evaluations > profile.n_free


def test_the_fitted_parameters_are_counted_as_parameters(example, profile):
    """The reference reports no parameter count at all, so this is ours to get
    right: a decay rate estimated from the data is a parameter, and AIC and BIC
    have to be charged for it."""
    columns = profile.fit.design.n_params
    assert profile.fit.n_fitted_pars == 2
    assert profile.fit.n_params == columns + 2
    assert profile.fit.aic == pytest.approx(-2 * profile.fit.ll + 2 * (columns + 2))
    assert "feature parameter(s) fitted too" in profile.fit.summary()


def test_identification_is_decided_at_the_seed_and_held(example, profile):
    """A parameter value that made one more column identically zero would
    change the parameter count mid-search, and the AIC of one evaluation would
    stop being comparable with the next."""
    seed = build_lkt_design(example, _profile_terms())
    assert profile.fit.design.n_params == seed.n_params
    assert len(profile.fit.design.aliased) == len(seed.aliased)


def _twins(n_students=8, n_steps=16, seed=3):
    """KCs ``A`` and ``B`` tag exactly the same steps, so every per-level block
    carries them as two identical columns; ``C`` tags the rest. At these sizes a
    ``propdec$`` search lands inside its bounds rather than on one, where six
    significant digits would happen to be exact."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_students):
        seen = {"A~~B": 0, "C": 0}
        for i in range(n_steps):
            kc = "A~~B" if i % 2 == 0 else "C"
            seen[kc] += 1
            opp = "~~".join([str(seen[kc])] * len(kc.split("~~")))
            rows.append(step_row(f"s{s}", f"st{i}", int(rng.random() < 0.6), kc, opp))
    return step_data(rows)


def test_a_profiles_design_reports_the_seeds_aliasing_in_its_own_names():
    """Identification is held from the seed, and so is its record of *why*. A
    duplicate's reason names the block it duplicates, and a parametric block's
    name moves with its parameter — so the profile's design has to report what
    a design built fresh at the estimate reports, not the seed's names and not
    a generic reason in place of the real one."""
    data = _twins()
    terms = lkt_terms(("student", "kc", "kc"), ("intercept", "intercept", "propdec$"),
                      (None, None, 0.9))
    assert "duplicate of propdec$(0.9)[kc]:A" in build_lkt_design(data, terms).aliased.reasons

    profile = fit_lkt_pars(data, terms)
    assert profile.pars[0] != 0.9, "the parameter has to move for its block name to"
    assert profile.fit.design.aliased == build_lkt_design(data, profile.terms).aliased
    # And the fit knows exactly what it fitted: the terms read back from its
    # block names are the estimate itself, not the estimate to six digits.
    assert profile.fit.terms == profile.terms


def test_the_outer_gradient_is_a_central_difference_as_optim_takes_it(example):
    """R's ``optim`` differences both ways at ``ndeps = 1e-3``; scipy's
    L-BFGS-B, left to itself, would step forward only. The trajectory shows
    which one ran: after the seed, each parameter is probed once above it and
    once below."""
    result = fit_lkt_pars(example, _profile_terms(), max_iterations=1)
    visited = result.trajectory[list(result.labels)].to_numpy()
    seed = np.array([0.9, 0.5])
    np.testing.assert_allclose(visited[0], seed)
    np.testing.assert_allclose(visited[1:5] - seed,
                               [[PARAMETER_STEP, 0.0], [-PARAMETER_STEP, 0.0],
                                [0.0, PARAMETER_STEP], [0.0, -PARAMETER_STEP]], atol=1e-12)


def test_a_difference_at_a_bound_is_taken_over_the_span_that_fits():
    """Each side stops at its bound and the quotient divides by the span it
    covered, as ``optim`` does. On a quadratic that quotient is exactly the
    derivative at the span's midpoint, which makes the rule checkable."""
    def f(x):
        return float(x[0] ** 2 + 3.0 * x[0])

    h = PARAMETER_STEP
    for x, midpoint in [(0.5, 0.5),                    # interior: central
                        (1.0, 1.0 - h / 2),            # on the bound: one-sided
                        (1.0 - h / 2, 1.0 - 3 * h / 4)]:  # partway: shortened
        gradient = _central_differences(f, np.array([x]), [(0.0, 1.0)])
        assert gradient[0] == pytest.approx(2.0 * midpoint + 3.0, abs=1e-9)
    assert _central_differences(f, np.array([0.5]), [(0.5, 0.5)])[0] == 0.0


def test_max_gain_bounds_what_any_single_parameter_step_actually_buys(example, profile):
    """The certificate is a measurement: every parameter is stepped both ways
    and refitted. So no step it did not take can beat the number it reports."""
    for j in range(profile.n_free):
        for direction in (-1.0, 1.0):
            probe = np.array(profile.pars, dtype=float)
            lower, upper = profile.bounds[j]
            probe[j] = float(np.clip(probe[j] + direction * 1e-3, lower, upper))
            if probe[j] == profile.pars[j]:
                continue
            terms = _profile_terms(*probe)
            stepped = fit_lkt(build_lkt_design(example, terms), example.y,
                              warn_not_converged=False)
            assert stepped.ll - profile.fit.ll <= profile.max_gain + 1e-9
    assert profile.is_stationary
    assert profile.max_gain <= PARAMETER_TOLERANCE


def test_the_optimizers_own_flag_is_not_the_certificate(example, profile):
    """They answer different questions, so both are reported. Capped at one
    outer iteration the optimizer says it stopped early — and on this data it
    had already reached a corner from which no step improves, which is what
    ``is_stationary`` is for and what ``converged`` cannot tell you."""
    stopped = fit_lkt_pars(example, _profile_terms(), max_iterations=1)
    assert not stopped.converged
    assert "ITERATIONS" in stopped.message.upper()
    assert stopped.is_stationary

    assert profile.converged
    assert profile.fit.ll == pytest.approx(stopped.fit.ll, abs=1e-9)


def test_free_holds_the_parameters_it_does_not_select(example):
    result = fit_lkt_pars(example, _profile_terms(logitdec=0.7, recency=0.4),
                          free=(True, False))
    assert result.n_free == 1
    assert result.labels == ("kc:logitdec",)
    held = next(t for t in result.terms if t.feature == "recency")
    assert held.pars == (0.4,)
    assert result.fit.n_fitted_pars == 1


def test_a_parameter_resting_on_a_bound_is_reported_as_such(example):
    """The estimate is then the bound, not an interior maximum, and a reader
    who cannot see that will read it as an estimate of the decay rate."""
    result = fit_lkt_pars(example, _profile_terms(), bounds=(0.80, 0.81))
    frame = result.frame()
    assert frame["at_bound"].any()
    assert "resting on a bound" in result.summary()
    assert np.all(frame["estimate"] >= 0.80 - 1e-12)
    assert np.all(frame["estimate"] <= 0.81 + 1e-12)


def test_restarts_measure_the_non_convexity_instead_of_assuming_it_away(example, profile):
    """One start says nothing about other basins; the summary says so, and
    several starts turn that into a measurement."""
    assert len(profile.restarts) == 1
    assert "says nothing about other basins" in profile.summary()

    several = fit_lkt_pars(example, _profile_terms(),
                           starts=[(0.9, 0.5), (0.2, 0.2), (0.99, 0.99)])
    assert len(several.restarts) == 3
    assert several.fit.ll == pytest.approx(several.restarts["objective"].max())
    assert "restarts       3 start(s)" in several.summary()


def test_the_two_objectives_coincide_when_nothing_is_penalized(example, profile):
    """``penalized`` profiles what the inner solver maximizes and
    ``likelihood`` profiles what the reference reports; with no ridge there is
    only one function. (The ``profile`` fixture is the ``penalized`` one.)"""
    likelihood = fit_lkt_pars(example, _profile_terms(), objective="likelihood")
    np.testing.assert_allclose(profile.pars, likelihood.pars, atol=1e-9)


def test_a_ridge_separates_the_two_objectives(example):
    """And then they are genuinely different questions: one maximizes a single
    function over parameters and coefficients together, the other maximizes one
    function over the coefficients and a different one over the parameters."""
    penalized = fit_lkt_pars(example, _profile_terms(), cost=2.0,
                             objective="penalized")
    likelihood = fit_lkt_pars(example, _profile_terms(), cost=2.0,
                              objective="likelihood")
    assert penalized.fit.ll >= likelihood.fit.ll - 1e-9
    assert likelihood.fit.ll_unpenalized >= penalized.fit.ll_unpenalized - 1e-9


def test_warm_starting_does_not_change_where_the_search_lands(example, profile):
    """Safe to do aggressively because the inner problem is convex and every
    inner fit certifies itself independently of where it started. (The
    ``profile`` fixture is warm-started.)"""
    cold = fit_lkt_pars(example, _profile_terms(), warm_start=False)
    np.testing.assert_allclose(profile.pars, cold.pars, atol=1e-6)
    assert profile.fit.ll == pytest.approx(cold.fit.ll, abs=1e-6)


def test_the_trajectory_records_every_evaluation(example, profile):
    trajectory = profile.trajectory
    assert len(trajectory) == profile.n_evaluations
    assert list(trajectory.columns[:1]) == ["evaluation"]
    assert set(profile.labels) <= set(trajectory.columns)
    assert trajectory["inner_optimal"].all()
    assert trajectory["objective"].max() == pytest.approx(profile.fit.ll)


def test_a_specification_with_nothing_to_fit_is_refused(example):
    with pytest.raises(ValueError, match="nothing to fit"):
        fit_lkt_pars(example, lkt_terms(("kc", "kc"), ("intercept", "lineafm$")))


@pytest.mark.parametrize("kwargs,message", [
    ({"objective": "nonsense"}, "objective must be one of"),
    ({"free": (True,)}, "free has 1 entr"),
    ({"free": (False, False)}, "selects no parameter"),
    ({"starts": [(0.5,)]}, "1 value"),
    ({"bounds": [(0.1, 0.9)]}, "1 bound pair"),
])
def test_the_search_checks_its_own_arguments(example, kwargs, message):
    with pytest.raises(ValueError, match=message):
        fit_lkt_pars(example, _profile_terms(), **kwargs)
