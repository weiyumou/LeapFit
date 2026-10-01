"""Reading a student-step export into ``StepData``.

What the reader requires and what it refuses, how it orders practice and
numbers opportunities, and the clock it keeps for the families that need one.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from leapfit import build_afm_design, cross_validate, fit_afm, from_frame, list_kc_models

from helpers import EPOCH, MINIMAL_COLUMNS, clocked_data, minimal_frame, rollup, synthetic

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


def test_opportunities_can_be_recomputed_without_a_time_column():
    """practice_order falls back to row order, which is what DataShop numbers by."""
    data = from_frame(minimal_frame(), "M")
    assert data.times is None
    assert data.recomputed_opportunities() == data.opportunities
    assert len(data.opportunity_disagreements()) == 0


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
    stripped = type(example)(
        y=example.y, students=example.students, items=example.items,
        kcs=example.kcs, opportunities=example.opportunities, kc_model=example.kc_model,
    )
    with pytest.raises(ValueError, match="no 'First Transaction Time' column"):
        stripped.epoch_times()
    with pytest.raises(ValueError, match="no 'Step Duration"):
        stripped.time_on_task()
