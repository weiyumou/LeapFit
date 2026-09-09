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

import numpy as np
import pandas as pd
import pytest

from leapfit import (
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
    fit_pfa,
    from_frame,
    history_counts,
    lkt_terms,
    load_student_step,
    success_failure_counts,
)

EXAMPLE = "examples/student-step.txt"

AFM_SPEC = (("student", "kc", "kc"), ("intercept", "intercept", "lineafm$"))
PFA_SPEC = (("kc", "kc", "kc"), ("intercept", "linesuc$", "linefail$"))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _rollup(rows, kc_model="M"):
    return pd.DataFrame(rows).rename(
        columns={"kc": f"KC ({kc_model})", "opp": f"Opportunity ({kc_model})"})


def _row(student, step, y, kc, opp, time=None, **extra):
    out = {
        "Anon Student Id": student, "Problem Name": "p", "Step Name": step,
        "First Attempt": "correct" if y else "incorrect", "kc": kc, "opp": str(opp),
    }
    if time is not None:
        out["First Transaction Time"] = time
    return out | extra


def _data(rows, kc_model="M"):
    return from_frame(_rollup(rows, kc_model), kc_model)


@pytest.fixture(scope="module")
def example():
    return load_student_step(EXAMPLE, kc_model="Topics")


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
    rows = [_row("s1", "st1", 1, "A", 1, time="2024-01-01 00:00:02"),
            _row("s1", "st2", 0, "A", 2, time="2024-01-01 00:00:01")]
    data = _data(rows)
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


def test_history_counts_on_kcs_is_the_pfa_count_function(example):
    """The generalization has to agree with what it generalizes."""
    assert history_counts(example, example.kcs) == success_failure_counts(example)


def test_history_counts_are_strictly_prior():
    """correct, incorrect, correct -> s = (0,1,1), f = (0,0,1)."""
    data = _data([_row("s1", f"st{i}", y, "A", i + 1)
                  for i, y in enumerate([1, 0, 1])])
    s, f = history_counts(data, data.kcs)
    assert s == [(0,), (1,), (1,)]
    assert f == [(0,), (0,), (1,)]


def test_counts_on_the_student_component_span_that_students_whole_history():
    """A feature on the student counts everything that student did before,
    whatever KC it was on — the reference's index is (level, student), and for
    the student component the level *is* the student."""
    rows = [_row("s1", "st1", 1, "A", 1), _row("s1", "st2", 0, "B", 1),
            _row("s1", "st3", 1, "A", 2), _row("s2", "st4", 0, "A", 1)]
    data = _data(rows)
    s, f = history_counts(data, component_labels(data, "student"))
    assert [row[0] for row in s] == [0, 1, 1, 0]
    assert [row[0] for row in f] == [0, 0, 1, 0]


def test_successes_and_failures_sum_to_the_recomputed_opportunities(example):
    s, f = history_counts(example, example.kcs)
    total = [tuple(a + b for a, b in zip(sr, fr)) for sr, fr in zip(s, f)]
    assert total == example.recomputed_opportunities()


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


def _single_kc_column(data, feature, **kwargs):
    design = build_lkt_design(data, [Term("kc", feature, per_level=True, **kwargs)],
                              identify=False)
    return _dense(design)[:, 0]


@pytest.fixture
def streak():
    """One student, one KC, outcomes correct, incorrect, correct, correct."""
    return _data([_row("s1", f"st{i}", y, "A", i + 1)
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
    ("prop", [0.5, 1.0, 0.5, 2 / 3]),
])
def test_static_feature_values_match_their_definitions(streak, feature, expected):
    np.testing.assert_allclose(_single_kc_column(streak, feature), expected)


def test_prop_seeds_an_unpractised_level_at_a_half(streak):
    """The reference guards 0/0 with .5 rather than dropping the row."""
    assert _single_kc_column(streak, "prop")[0] == 0.5


def test_powafm_raises_the_count_to_its_fixed_exponent(streak):
    np.testing.assert_allclose(_single_kc_column(streak, "powafm", par=0.5),
                               [0.0, 1.0, 2 ** 0.5, 3 ** 0.5])


def test_logafm_is_log1p_of_lineafm(example):
    line = _dense(build_lkt_design(example, [Term("kc", "lineafm")], identify=False))
    log = _dense(build_lkt_design(example, [Term("kc", "logafm")], identify=False))
    np.testing.assert_allclose(log[:, 0], np.log1p(line[:, 0]))


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
    data = _data([_row("s1", "st1", 1, "A~~B", "1~~1"),
                  _row("s1", "st2", 0, "A", 2)])
    assert component_labels(data, "kc") == [("A", "B"), ("A",)]
    assert component_labels(data, "student") == [("s1",), ("s1",)]


def test_a_shared_coefficient_sums_a_rows_levels():
    """Two KCs on one step contribute additively, as they do in AFM — so the
    pooled column is the row total, matching PFA's pooled construction."""
    rows = [_row("s1", "st1", 1, "A~~B", "1~~1"),
            _row("s1", "st2", 0, "A~~B", "2~~2")]
    data = _data(rows)
    pooled = _dense(build_lkt_design(data, [Term("kc", "lineafm")], identify=False))
    per_level = _dense(build_lkt_design(data, [Term("kc", "lineafm", per_level=True)],
                                        identify=False))
    np.testing.assert_allclose(pooled[:, 0], per_level.sum(axis=1))
    np.testing.assert_allclose(pooled[:, 0], [0.0, 2.0])


def test_an_unknown_component_names_the_columns_that_exist(example):
    with pytest.raises(KeyError, match="Available columns"):
        component_labels(example, "Not A Column")


def test_a_source_component_needs_the_source_table(example):
    stripped = type(example)(
        y=example.y, students=example.students, items=example.items,
        kcs=example.kcs, opportunities=example.opportunities,
        kc_model=example.kc_model,
    )
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
    terms = (Term("kc", "powafm", per_level=True, par=0.5),
             Term("kc", "powafm", per_level=True, par=0.8))
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
    ("recency", "numeric time"),
    ("base2", "numeric time"),
    ("ppe", "numeric time"),
    ("logitdec", "decay parameter"),
    ("propdec", "decay parameter"),
    ("diffcor1", "prior fit"),
])
def test_deferred_features_are_refused_by_name_with_the_reason(feature, reason):
    with pytest.raises(NotImplementedError, match=reason):
        Term("kc", feature)


def test_an_unknown_feature_lists_the_ones_that_exist():
    with pytest.raises(ValueError, match="Implemented: "):
        Term("kc", "nonsense")


def test_a_random_effect_is_refused_rather_than_silently_fixed():
    with pytest.raises(NotImplementedError, match="glmer"):
        Term.parse("kc", "intercept@")


def test_a_shape_parameter_is_required_where_it_exists_and_refused_where_it_does_not():
    with pytest.raises(ValueError, match="nothing here fits one"):
        Term("kc", "powafm")
    with pytest.raises(ValueError, match="takes no parameter"):
        Term("kc", "lineafm", par=0.5)


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


def test_intercepts_on_any_other_pair_of_components_are_refused(example):
    with pytest.raises(NotImplementedError, match="redundant direction"):
        build_lkt_design(example, [Term("student", "intercept"),
                                   Term("item", "intercept")])
    with pytest.raises(NotImplementedError, match="3 components"):
        build_lkt_design(example, [Term("student", "intercept"),
                                   Term("kc", "intercept"),
                                   Term("item", "intercept")])


def test_a_single_intercept_on_any_component_is_fine(example):
    for component in ("student", "kc", "item", "Problem Name"):
        design = build_lkt_design(example, [Term(component, "intercept")])
        assert design.n_params == design.rank()


def test_identify_false_accepts_what_identification_refuses(example):
    """The escape hatch the refusal advertises has to exist: the caller has
    taken the parameter count on themselves."""
    design = build_lkt_design(example, [Term("student", "intercept"),
                                        Term("item", "intercept")], identify=False)
    assert design.n_params > design.rank()


def test_a_level_with_no_second_opportunity_reports_nan_not_zero():
    """AFM's never-practised-twice rule, at the level of any component: the
    slope was not estimated, and printing 0.0 would invite it into a
    low-slope screen that reads zero as "students did not learn"."""
    rows = [_row("s1", "st1", 1, "A", 1), _row("s1", "st2", 0, "A", 2),
            _row("s1", "st3", 1, "B", 1)]
    data = _data(rows)
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
    assert fit.ll == pytest.approx(fit.ll_unpenalized - fit.penalty)


def test_the_reference_ridge_hides_a_separation_the_default_reports():
    """``Design.separated`` skips penalized columns, correctly: a ridge
    supplies the curvature the likelihood is missing. So the same spec on the
    same data reports a separated student without ``cost`` and none with it —
    which is what the reference's always-on penalty is doing."""
    rows = ([_row("s1", f"st{i}", 1, "A", i + 1) for i in range(4)]
            + [_row("s2", f"st{i}", i % 2, "A", i + 1) for i in range(4)])
    data = _data(rows)
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
    rows = [_row("s1", f"st{i}", i % 2, "A", i + 1, **{"Prior Score": str(i * 2)})
            for i in range(3)]
    data = _data(rows)
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


def test_an_lkt_design_takes_row_subsets_like_any_other(example):
    design = build_lkt_design(example, lkt_terms(*PFA_SPEC))
    rows = np.arange(0, len(example), 2)
    subset = design.take(rows)
    assert subset.n_obs == len(rows)
    assert subset.columns == design.columns
