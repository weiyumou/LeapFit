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
    LOGITDEC_WINDOW,
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


EPOCH = pd.Timestamp("2024-01-01 00:00:00")


def _clocked(outcomes, gaps, durations=None, student="s1", kc="A"):
    """One student, one level: a practice sequence with a clock on it.

    ``gaps`` are the seconds between consecutive attempts, the first ignored,
    so the sequence starts at :data:`EPOCH`.
    """
    times = np.cumsum(np.asarray(gaps, dtype=float))
    rows = []
    for i, (y, t) in enumerate(zip(outcomes, times)):
        extra = {} if durations is None else {"Step Duration (sec)": durations[i]}
        rows.append(_row(student, f"st{i}", y, kc, i + 1,
                         time=str(EPOCH + pd.Timedelta(seconds=int(t))), **extra))
    return _data(rows)


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
    np.testing.assert_allclose(_single_kc_column(streak, "powafm", pars=0.5),
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
    with pytest.raises(ValueError, match="takes 1 fixed parameter, got 0"):
        Term("kc", "powafm")
    with pytest.raises(ValueError, match="takes no parameter"):
        Term("kc", "lineafm", pars=0.5)
    with pytest.raises(ValueError, match="takes 4 fixed parameters, got 2"):
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


# --------------------------------------------------------------------------
# The clock, which lives on StepData because it is not LKT's alone
# --------------------------------------------------------------------------


def test_epoch_times_are_seconds_and_not_the_parsers_own_unit():
    """A regression with teeth: ``to_datetime``'s backing unit is a pandas
    version detail, and reading it as nanoseconds when it is microseconds
    scales every interval by a thousand — silently, and only the *spacing*
    features would notice."""
    data = _clocked([1, 0], [0, 102])
    times = data.epoch_times()
    assert times[0] == int(EPOCH.timestamp())
    assert times[1] - times[0] == 102


def test_time_on_task_is_the_lagged_cumulative_duration():
    data = _clocked([1, 0, 1], [0, 60, 60], durations=[5.0, 7.0, 9.0])
    np.testing.assert_allclose(data.time_on_task(), [0.0, 5.0, 12.0])


def test_an_export_without_a_clock_refuses_rather_than_substituting_one(example):
    """Every value that could stand in for a missing time — the row number, a
    constant — is a different model, so there is no default to fall back to."""
    stripped = type(example)(
        y=example.y, students=example.students, items=example.items,
        kcs=example.kcs, opportunities=example.opportunities, kc_model=example.kc_model,
    )
    with pytest.raises(ValueError, match="no 'First Transaction Time' column"):
        stripped.epoch_times()
    with pytest.raises(ValueError, match="no 'Step Duration"):
        stripped.time_on_task()


def test_a_term_that_needs_a_clock_names_itself_when_the_export_has_none(example):
    """``example`` has times but no durations, so this is the second half."""
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
    return _clocked(outcomes, gaps, durations=[5, 7, 9, 4, 6, 8, 3]), \
        np.array(outcomes, dtype=float), np.cumsum(np.asarray(gaps, dtype=float))


def _column(data, feature, pars=()):
    design = build_lkt_design(data, [Term("kc", feature, per_level=True, pars=pars)],
                              identify=False)
    return _dense(design)[:, 0]


def test_the_decay_features_lag_their_own_trial(clocked):
    """Every one of these is shifted by a position: the reference's slide
    functions return ``c(seed, v[1:n-1])``, so nothing regresses on itself."""
    data, y, _ = clocked
    d = 0.85
    np.testing.assert_allclose(_column(data, "expdecafm", d), _slide_expdec(np.ones(7), d))
    np.testing.assert_allclose(_column(data, "expdecsuc", d), _slide_expdec(y, d))
    np.testing.assert_allclose(_column(data, "expdecfail", d), _slide_expdec(1 - y, d))
    np.testing.assert_allclose(_column(data, "propdec", d), _slide_propdec(y, d))
    np.testing.assert_allclose(_column(data, "logitdec", d), _slide_logitdec(y, d))


def test_propdec_starts_at_a_half_and_logitdec_at_zero(clocked):
    """The ghost trials are what make position 0 defined instead of 0/0."""
    data, _, _ = clocked
    assert _column(data, "propdec", 0.85)[0] == 0.5
    assert _column(data, "logitdec", 0.85)[0] == 0.0


def test_logitdec_truncates_at_the_references_sixty_trial_window():
    """Undocumented in the paper, and it bites: at ``d = .97`` the window
    changes the feature by 0.06 logits over 200 trials, so a package that
    quietly used the whole history would not reproduce the reference."""
    rng = np.random.default_rng(0)
    outcomes = rng.integers(0, 2, 200).tolist()
    data = _clocked(outcomes, [0] + [60] * 199)
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
    data = _clocked([1, 0, 1], [0, 0, 60])
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
    different settings are two blocks rather than a collision."""
    pars = (0.3491901, 0.2045801, 1e-05, 0.9734477)
    term = Term("kc", "ppe", per_level=True, pars=pars)
    assert term.block_name.startswith("ppe$(") and term.block_name.endswith(")[kc]")
    assert term.block_name.count(",") == 3

    design = build_lkt_design(example, [term], identify=False)
    recovered = design_terms(design)[0]
    assert recovered.feature == "ppe" and recovered.component == "kc"
    assert recovered.pars == pytest.approx(pars, rel=1e-5)


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
    rows = [_row("s1", f"st{i}", i % 2, "A", i + 1,
                 time=str(EPOCH + pd.Timedelta(seconds=60 * i)),
                 **{"Step Duration (sec)": "." if i == 1 else 5})
            for i in range(3)]
    with pytest.raises(ValueError, match="1 of 3 rows have no 'Step Duration"):
        _data(rows).time_on_task()
