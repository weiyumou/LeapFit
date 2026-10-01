"""Reading a student-step export into ``StepData``.

What the reader requires and what it refuses, how it orders practice and
numbers opportunities, the clock it keeps for the families that need one,
and how a transaction export, or a file made for DataShop's import, is rolled
up into a student-step one first.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from leapfit import (
    build_afm_design,
    cross_validate,
    fit_afm,
    from_frame,
    list_kc_models,
    load_transactions,
    rollup_transactions,
)

from helpers import (
    EPOCH,
    MINIMAL_COLUMNS,
    as_transactions,
    clocked_data,
    minimal_frame,
    rollup,
    stamp,
    synthetic,
    tx_row,
)

# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_opportunity_is_zero_based():
    df = rollup([{
        "Anon Student Id": "s1", "Problem Name": "p", "Step Name": "st",
        "First Attempt": "correct", "kc": "A", "opp": "1",
    }])
    data = from_frame(df, "M")
    assert data.opportunities == [(0,)], "first encounter must enter the model as T=0"


def test_multi_kc_splits_on_double_tilde():
    df = rollup([{
        "Anon Student Id": "s1", "Problem Name": "p", "Step Name": "st",
        "First Attempt": "incorrect", "kc": "A~~B", "opp": "3~~1",
    }])
    data = from_frame(df, "M")
    assert data.kcs == [("A", "B")]
    assert data.opportunities == [(2, 0)]
    assert data.y.tolist() == [0]


def test_rows_without_a_kc_are_skipped_entirely():
    df = rollup([
        {"Anon Student Id": "s1", "Problem Name": "p", "Step Name": "a",
         "First Attempt": "correct", "kc": "", "opp": ""},
        {"Anon Student Id": "s1", "Problem Name": "p", "Step Name": "b",
         "First Attempt": "correct", "kc": "A", "opp": "1"},
    ])
    data = from_frame(df, "M")
    assert len(data) == 1 and data.skipped_no_kc == 1


def test_only_correct_counts_as_success():
    df = rollup([
        {"Anon Student Id": "s1", "Problem Name": "p", "Step Name": s,
         "First Attempt": a, "kc": "A", "opp": "1"}
        for s, a in [("a", "correct"), ("b", "incorrect"), ("c", "hint")]
    ])
    assert from_frame(df, "M").y.tolist() == [1, 0, 0]


def test_kc_opportunity_length_mismatch_raises():
    df = rollup([{
        "Anon Student Id": "s1", "Problem Name": "p", "Step Name": "st",
        "First Attempt": "correct", "kc": "A~~B", "opp": "1",
    }])
    with pytest.raises(ValueError, match="opportunity value"):
        from_frame(df, "M")


def test_missing_kc_model_lists_alternatives():
    df = rollup([{
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
# Canonical practice ordering
# --------------------------------------------------------------------------


def test_practice_order_uses_time_then_row_order():
    df = pd.DataFrame([
        {"Anon Student Id": "s1", "Problem Name": "p", "Step Name": "b",
         "First Transaction Time": "2022-01-01 00:00:09", "First Attempt": "correct",
         "KC (M)": "A", "Opportunity (M)": "2"},
        {"Anon Student Id": "s1", "Problem Name": "p", "Step Name": "a",
         "First Transaction Time": "2022-01-01 00:00:01", "First Attempt": "correct",
         "KC (M)": "A", "Opportunity (M)": "1"},
    ])
    data = from_frame(df, "M")
    np.testing.assert_array_equal(data.practice_order()["s1"], [1, 0])
    assert data.recomputed_opportunities() == [(1,), (0,)]
    # The file's own column is row-ordered, so here the two disagree.
    assert data.opportunities == [(1,), (0,)]


def test_recomputed_opportunities_match_the_file_when_order_agrees():
    """Without a time column practice_order falls back to row order, which is
    what DataShop numbers by."""
    data = synthetic(n_students=5, n_kcs=3, n_items=9, seed=40, n_reps=4)
    assert data.times is None, "synthetic rollups carry no time column"
    assert data.recomputed_opportunities() == data.opportunities
    assert len(data.opportunity_disagreements()) == 0


# --------------------------------------------------------------------------
# Portability: what the reader requires, and what it refuses
# --------------------------------------------------------------------------


def test_the_minimal_column_set_is_enough_to_fit_and_cross_validate():
    """Six columns, no timestamps, no DataShop metadata."""
    df = minimal_frame()
    assert list(df.columns) == MINIMAL_COLUMNS

    data = from_frame(df, "M")
    assert data.times is None
    design = build_afm_design(data)
    fit = fit_afm(design, data.y)
    assert fit.is_optimal
    assert cross_validate(design, data, scheme="item_blocked", n_folds=2).rmse > 0


def test_every_declared_column_is_actually_required():
    """Each of the six is load-bearing — dropping any one raises by name."""
    df = minimal_frame()
    for column in MINIMAL_COLUMNS:
        with pytest.raises(KeyError, match=re.escape(column)):
            from_frame(df.drop(columns=[column]), "M")


def test_outcome_labels_are_matched_case_insensitively():
    """'Correct' must not silently score as a failure.

    Matching the literal string is the reference's rule, and under it an export
    that capitalizes the column yields an all-zero response, a converged fit,
    and a plausible AIC — wrong with no symptom. Folding case is a no-op on
    real DataShop files, which are lowercase.
    """
    lower = from_frame(minimal_frame(), "M")
    upper = from_frame(minimal_frame(attempt=str.capitalize), "M")
    spaced = from_frame(minimal_frame(attempt=lambda v: f"  {v.upper()} "), "M")
    assert lower.y.mean() > 0
    assert np.array_equal(lower.y, upper.y)
    assert np.array_equal(lower.y, spaced.y)


def test_unknown_outcome_vocabulary_raises_instead_of_scoring_zero():
    df = minimal_frame()
    df["First Attempt"] = np.where(df["First Attempt"] == "correct", "1", "0")
    with pytest.raises(ValueError, match="Unrecognized 'First Attempt'"):
        from_frame(df, "M")
    # Declaring only the successes is not enough — '0' is still unaccounted for,
    # so the guard stays armed rather than switching off on any override.
    with pytest.raises(ValueError, match="Unrecognized 'First Attempt'"):
        from_frame(df, "M", success_values=("1",))
    data = from_frame(df, "M", success_values=("1",), failure_values=("0",))
    assert 0 < data.y.mean() < 1


def test_the_vocabulary_error_names_each_label_on_one_side_only():
    """Regression: the failures it listed included 'correct'."""
    df = minimal_frame()
    df["First Attempt"] = np.where(df["First Attempt"] == "correct", "1", "0")
    with pytest.raises(ValueError, match=re.escape("failures are ['hint', 'incorrect', 'unknown']")):
        from_frame(df, "M")
    # Declared as both, a label is scored as a success, so that is where it is listed.
    with pytest.raises(ValueError, match=re.escape("failures are ['incorrect', 'unknown']")):
        from_frame(df, "M", success_values=("correct", "hint"))
    # Nor is 'correct' a failure beside a success vocabulary of the file's own,
    # which the old default let through as one.
    df["First Attempt"] = np.where(df["First Attempt"] == "1", "1", "correct")
    with pytest.raises(ValueError, match=re.escape("value(s): 'correct'")):
        from_frame(df, "M", success_values=("1",))


def test_documented_datashop_failure_labels_are_accepted_silently():
    df = minimal_frame()
    df.loc[df.index % 5 == 0, "First Attempt"] = "hint"
    df.loc[df.index % 7 == 0, "First Attempt"] = "unknown"
    data = from_frame(df, "M")
    assert 0 < data.y.mean() < 1


def test_a_malformed_opportunity_value_names_the_row():
    df = minimal_frame()
    df.loc[5, "Opportunity (M)"] = "."
    with pytest.raises(ValueError, match=r"Row 7: non-integer opportunity"):
        from_frame(df, "M")


def test_a_constant_response_warns_and_names_the_vocabulary():
    df = minimal_frame()
    df["First Attempt"] = "incorrect"
    with pytest.warns(RuntimeWarning, match="Every observation is a failure"):
        from_frame(df, "M")


# --------------------------------------------------------------------------
# The clock, which lives on StepData because it is not LKT's alone
# --------------------------------------------------------------------------


def test_epoch_times_are_seconds_and_not_the_parsers_own_unit():
    """A regression with teeth: ``to_datetime``'s backing unit is a pandas
    version detail, and reading it as nanoseconds when it is microseconds
    scales every interval by a thousand — silently, and only the *spacing*
    features would notice."""
    data = clocked_data([1, 0], [0, 102])
    times = data.epoch_times()
    assert times[0] == int(EPOCH.timestamp())
    assert times[1] - times[0] == 102


def test_time_on_task_is_the_lagged_cumulative_duration():
    data = clocked_data([1, 0, 1], [0, 60, 60], durations=[5.0, 7.0, 9.0])
    np.testing.assert_allclose(data.time_on_task(), [0.0, 5.0, 12.0])


def test_an_export_without_a_clock_refuses_rather_than_substituting_one(example):
    """Every value that could stand in for a missing time — the row number, a
    constant — is a different model, so there is no default to fall back to."""
    stripped = replace(example, times=None, durations=None)
    with pytest.raises(ValueError, match="no 'First Transaction Time' column"):
        stripped.epoch_times()
    with pytest.raises(ValueError, match="no 'Step Duration"):
        stripped.time_on_task()


# --------------------------------------------------------------------------
# Transaction exports and import files, rolled up into student-steps
# --------------------------------------------------------------------------


def test_a_step_rolls_up_to_its_first_attempt():
    """Attempt 1 is the step's row. What has no attempt number, and every later
    attempt, leaves no trace on it but the time it took."""
    steps = rollup_transactions(pd.DataFrame([
        tx_row("s1", "st1", "", 0, attempt=""),  # a page view
        tx_row("s1", "st1", "INCORRECT", 5),
        tx_row("s1", "st1", "HINT", 9, attempt=2),
        tx_row("s1", "st1", "CORRECT", 14, attempt=3),
        tx_row("s1", "st2", "CORRECT", 20),
    ]))
    assert steps["Step Name"].tolist() == ["st1", "st2"]
    assert steps["First Attempt"].tolist() == ["incorrect", "correct"]
    assert steps["First Transaction Time"].tolist() == [stamp(5), stamp(20)]
    assert steps["Opportunity (M)"].tolist() == ["1", "2"]


def test_outcomes_take_the_student_step_vocabulary():
    """Where DataShop left a first attempt's outcome blank, its student-step
    export says 'unknown'."""
    steps = rollup_transactions(pd.DataFrame([
        tx_row("s1", f"st{i}", outcome, i)
        for i, outcome in enumerate(["CORRECT", "HINT", " Incorrect ", ""])]))
    assert steps["First Attempt"].tolist() == ["correct", "hint", "incorrect", "unknown"]
    assert from_frame(steps, "M").y.tolist() == [1, 0, 0, 0]


def test_each_problem_view_is_an_encounter_of_its_own():
    """A problem opened again straight away is a second encounter with its steps,
    not more attempts at the first."""
    steps = rollup_transactions(pd.DataFrame([
        tx_row("s1", "st1", "INCORRECT", 0),
        tx_row("s1", "st1", "INCORRECT", 5, **{"Problem View": "2"}),
        tx_row("s1", "st1", "CORRECT", 6, attempt=2, **{"Problem View": "2"}),
    ]))
    assert steps["Problem View"].tolist() == ["1", "2"]
    assert steps["First Attempt"].tolist() == ["incorrect", "incorrect"]
    assert steps["Opportunity (M)"].tolist() == ["1", "2"]


def test_the_hierarchy_tells_apart_problems_that_share_a_name():
    """The export's Level columns are part of what makes an encounter, and are kept."""
    steps = rollup_transactions(pd.DataFrame([
        tx_row("s1", "st1", "CORRECT", 0, **{"Level (Unit)": "one"}),
        tx_row("s1", "st1", "INCORRECT", 1, **{"Level (Unit)": "two"}),
    ]))
    assert steps["Level (Unit)"].tolist() == ["one", "two"]
    assert steps["Opportunity (M)"].tolist() == ["1", "2"]


def test_a_step_with_several_kcs_counts_each_of_them():
    """DataShop repeats a model's KC column once per KC, which pandas reads as
    'KC (M)', 'KC (M).1'. Each label is kept once, in column order, and a cell
    that is already '~~'-joined is split as well."""
    steps = rollup_transactions(pd.DataFrame([
        tx_row("s1", "st1", "CORRECT", 0, kc="A", **{"KC (M).1": "B"}),
        tx_row("s1", "st2", "INCORRECT", 1, kc="B", **{"KC (M).1": ""}),
        tx_row("s1", "st3", "CORRECT", 2, kc="A~~C", **{"KC (M).1": "C"}),
    ]))
    assert "KC (M).1" not in steps.columns
    assert steps["KC (M)"].tolist() == ["A~~B", "B", "A~~C"]
    assert steps["Opportunity (M)"].tolist() == ["1~~1", "2", "2~~1"]
    data = from_frame(steps, "M")
    assert data.kcs == [("A", "B"), ("B",), ("A", "C")]
    assert data.opportunities == [(0, 0), (1,), (1, 0)]
    assert data.duplicate_kc_rows == 0


def test_a_repeated_kc_header_in_the_file_reads_as_one_model(tmp_path):
    """The export as DataShop writes it: the same header twice."""
    header = ["Anon Student Id", "Problem Name", "Problem View", "Step Name",
              "Attempt At Step", "Outcome", "Time", "KC (M)", "KC (M)"]
    lines = [header,
             ["s1", "p", "1", "st1", "1", "CORRECT", stamp(0), "A", "B"],
             ["s1", "p", "1", "st2", "1", "INCORRECT", stamp(1), "B", ""]]
    path = tmp_path / "tx.txt"
    path.write_text("".join("\t".join(line) + "\n" for line in lines))
    assert list_kc_models(str(path)) == ["M"]
    data = load_transactions(str(path), "M")
    assert data.kcs == [("A", "B"), ("B",)]
    assert data.opportunities == [(0, 0), (1,)]


def test_opportunities_count_in_practice_order_and_the_rows_come_back_in_it():
    """By time; within a second by problem view, as DataShop counts a problem's
    earlier view first; then by the file's own order. Returned in that order,
    the rows give the reader the same practice order, so a recount agrees."""
    def at(t, view, step, outcome="CORRECT"):
        return tx_row("s1", step, outcome, t, **{"Problem View": str(view)})

    steps = rollup_transactions(pd.DataFrame([
        at(30, 1, "late", "INCORRECT"),
        at(10, 2, "second-view"), at(10, 2, "second-view-too"),
        at(20, 1, "same-1"), at(20, 1, "same-2"),
        at(10, 1, "first-view"),
        at(1, 1, "early"),
    ]))
    assert steps["Step Name"].tolist() == ["early", "first-view", "second-view",
                                           "second-view-too", "same-1", "same-2", "late"]
    assert steps["Opportunity (M)"].tolist() == ["1", "2", "3", "4", "5", "6", "7"]
    assert len(from_frame(steps, "M").opportunity_disagreements()) == 0


def test_step_duration_sums_the_encounter_unless_its_first_attempt_has_none():
    """DataShop sums the step's transaction durations, writes '.' where the
    first attempt's own is undefined, and leaves a later undefined one out."""
    def seconds(value):
        return {"Duration (sec)": value}

    steps = rollup_transactions(pd.DataFrame([
        tx_row("s1", "st1", "", 0, attempt="", **seconds("2")),
        tx_row("s1", "st1", "INCORRECT", 1, **seconds("3")),
        tx_row("s1", "st1", "CORRECT", 2, attempt=2, **seconds("4.25")),
        tx_row("s1", "st2", "INCORRECT", 3, **seconds(".")),
        tx_row("s1", "st2", "CORRECT", 4, attempt=2, **seconds("5")),
        tx_row("s1", "st3", "INCORRECT", 5, **seconds("1.5")),
        tx_row("s1", "st3", "CORRECT", 6, attempt=2, **seconds(".")),
        tx_row("s1", "st4", "CORRECT", 7, **seconds("1")),
    ]))
    assert steps["Step Duration (sec)"].tolist() == ["9.25", ".", "1.5", "1"]
    durations = from_frame(steps, "M").durations
    assert durations[0] == 9.25 and np.isnan(durations[1]) and durations[2] == 1.5

    bare = rollup_transactions(pd.DataFrame([tx_row("s1", "st1", "CORRECT", 0)]))
    assert "Step Duration (sec)" not in bare.columns


def test_a_step_without_a_kc_keeps_empty_cells_and_is_skipped():
    steps = rollup_transactions(pd.DataFrame([
        tx_row("s1", "st1", "CORRECT", 0, kc=""),
        tx_row("s1", "st2", "CORRECT", 1, kc="A"),
        tx_row("s1", "st3", "INCORRECT", 2, kc="A"),
    ]))
    assert steps[["KC (M)", "Opportunity (M)"]].values.tolist() == [
        ["", ""], ["A", "1"], ["A", "2"]]
    data = from_frame(steps, "M")
    assert len(data) == 2 and data.skipped_no_kc == 1


@pytest.mark.parametrize("change, error, match", [
    (lambda tx: tx.assign(**{"Sample Name": ["All Data", "Other"]}), ValueError, "2 samples"),
    (lambda tx: tx.assign(**{"Step Name": "st1"}), ValueError, "share their encounter"),
    (lambda tx: tx.assign(**{"Attempt At Step": ""}), ValueError, "No transaction has"),
    (lambda tx: tx.drop(columns="Attempt At Step").assign(**{"Step Name": ""}), ValueError,
     "No transaction names a step"),
    (lambda tx: tx.drop(columns="Problem View"), KeyError, "Problem View"),
    (lambda tx: tx.assign(**{"Event Type": ["assess", ""]}), ValueError,
     "'Event Type' is set on 1 of 2"),
    (lambda tx: tx.assign(Time=["soon", stamp(1)]), ValueError, "no format DataShop imports"),
])
def test_transactions_that_cannot_be_rolled_up_faithfully_are_refused(change, error, match):
    tx = pd.DataFrame([tx_row("s1", "st1", "CORRECT", 0), tx_row("s1", "st2", "CORRECT", 1)])
    with pytest.raises(error, match=match):
        rollup_transactions(change(tx))


def test_reading_a_transaction_export_as_student_steps_points_to_the_rollup():
    with pytest.raises(KeyError, match="transaction export"):
        from_frame(pd.DataFrame([tx_row("s1", "st1", "CORRECT", 0)]), "M")


def test_the_example_comes_back_from_a_transaction_export_of_itself(example):
    """Every step's first attempt, time, KCs and opportunities come back, so AFM
    fits the rolled-up table exactly as it fits the student-step export."""
    source = example.source
    steps = rollup_transactions(as_transactions(source))
    both = source.merge(steps, on=["Anon Student Id", "Problem Name", "Step Name"],
                        suffixes=("", " rolled up"))
    assert len(both) == len(source) == len(steps)
    for column in ["First Attempt", "First Transaction Time", "KC (Topics)",
                   "Opportunity (Topics)", "KC (Skills)", "Opportunity (Skills)"]:
        assert both[column].tolist() == both[f"{column} rolled up"].tolist(), column

    again = from_frame(steps, "Topics")
    fit = fit_afm(build_afm_design(example), example.y)
    refit = fit_afm(build_afm_design(again), again.y)
    assert refit.ll == pytest.approx(fit.ll, rel=1e-9)
    assert refit.n_params == fit.n_params


def test_a_file_made_for_import_rolls_up_as_its_export_does(example):
    """DataShop ignores an import's attempt numbers and makes its own, so a file
    without them gives the table its export gives. A blank Event Type, as
    DataShop's exports carry it, changes nothing."""
    export = as_transactions(example.source).assign(**{"Event Type": ""})
    imported = export.drop(columns="Attempt At Step")
    pd.testing.assert_frame_equal(rollup_transactions(imported), rollup_transactions(export))


def test_an_import_numbers_the_attempts_at_each_step_by_time():
    """Wherever the file lists them: the earliest transaction that names the
    step in its problem view is the first attempt, and one that names no step
    is no attempt at all."""
    steps = rollup_transactions(pd.DataFrame([
        tx_row("s1", "st1", "CORRECT", 9),
        tx_row("s1", "", "", 0),
        tx_row("s1", "st1", "HINT", 4),
        tx_row("s1", "st1", "INCORRECT", 6, **{"Problem View": "2"}),
    ]).drop(columns="Attempt At Step"))
    assert steps[["Problem View", "First Attempt", "First Transaction Time"]].values.tolist() == [
        ["1", "hint", stamp(4)], ["2", "incorrect", stamp(6)]]


def test_unix_milliseconds_are_ordered_as_times_and_written_in_utc():
    """As text, 1000000000000 sorts before 999999999999."""
    steps = rollup_transactions(pd.DataFrame([
        tx_row("s1", "st1", "CORRECT", 0, Time="1000000000000"),
        tx_row("s1", "st2", "INCORRECT", 0, Time="999999999999"),
        tx_row("s1", "st3", "CORRECT", 0, Time="1411017161123"),
    ]))
    assert steps["Step Name"].tolist() == ["st2", "st1", "st3"]
    assert steps["First Transaction Time"].tolist() == [
        "2001-09-09 01:46:39.999", "2001-09-09 01:46:40", "2014-09-18 05:12:41.123"]
    assert steps["Opportunity (M)"].tolist() == ["1", "2", "3"]


@pytest.mark.parametrize("written", [
    "2015-09-01 00:21:02", "2015-09-01 00:21:02:000", "2015/09/01 00:21:02.000",
    "09/01/2015 00:21:02", "09/01/15 00:21:02:000", "September 01, 2015 12:21:02 AM",
    "1441066862000", "1.441066862E12",
])
def test_each_time_format_datashop_imports_is_written_as_its_exports_write_it(written):
    steps = rollup_transactions(pd.DataFrame([tx_row("s1", "st1", "CORRECT", 0, Time=written)]))
    assert steps["First Transaction Time"].tolist() == ["2015-09-01 00:21:02"]
