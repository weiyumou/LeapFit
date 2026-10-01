"""The design matrix: its blocks, its identification, and separation.

Identification is the load-bearing part. A column it drops must be one the
data cannot estimate, so dropping it may change neither a prediction nor the
maximised likelihood, and the tests here check both.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from leapfit import Block, Design, accumulator_block, build_afm_design, fit_afm, from_frame
from leapfit.fit import _expit, _objective

from helpers import co_occurring_kc_data, multi_kc_data, separated_frame, step_data, synthetic

# --------------------------------------------------------------------------
# Design structure
# --------------------------------------------------------------------------


def test_from_levels_puts_each_rows_values_in_its_levels_columns():
    block = Block.from_levels("kc", [("B",), ("A", "B"), ("A",)],
                              values=[(2.0,), (0.0, 3.0), (1.0,)])
    assert block.columns == ["A", "B"]
    np.testing.assert_array_equal(block.matrix.toarray(), [[0, 2], [0, 3], [1, 0]])
    assert block.matrix.nnz == 3, "a zero value is a structural zero, not stored"
    assert Block.from_levels("f", [("x",), ("y",)]).matrix.toarray().tolist() == [[1, 0], [0, 1]]


def test_take_preserves_labels_penalty_and_bounds():
    data = synthetic(n_students=5, n_kcs=3, n_items=15, seed=2)
    design = build_afm_design(data)
    subset = design.take(np.array([0, 3, 5]))
    assert subset.n_obs == 3
    assert subset.columns == design.columns
    np.testing.assert_array_equal(subset.l2, design.l2)
    assert subset.bounds == design.bounds


def test_extra_block_extends_the_design():
    """The accumulator seam: PFA's counts and any history-derived predictor."""
    data = synthetic(n_students=5, n_kcs=3, n_items=15, seed=3)
    design = build_afm_design(data)
    extra = accumulator_block(data, np.random.default_rng(0).normal(size=(len(data), 2)),
                              columns=["prior_successes", "prior_failures"])
    extended = design.with_blocks(extra)
    assert extended.n_params == design.n_params + 2
    assert "accumulator" in extended.slices()
    assert np.all(extended.l2[extended.slices()["accumulator"]] == 0.0)
    with pytest.raises(ValueError, match="rows"):
        accumulator_block(data, np.zeros(len(data) + 1))


# --------------------------------------------------------------------------
# Identification: the new machinery must be a no-op on everything observable
# --------------------------------------------------------------------------


def test_identify_drops_a_reference_student():
    data = synthetic(n_students=6, n_kcs=3, n_items=12, seed=30, n_reps=5)
    full = build_afm_design(data, identify=False)
    ident = full.identify()
    assert ident.n_params == full.n_params - 1
    assert len(ident.aliased) == 1
    assert ident.aliased.columns[0].startswith("student:")
    assert ident.n_params == ident.rank(), "identified design must be full rank"
    # A regression guard: one component, so no component bookkeeping in the reason.
    assert ident.aliased.reasons == ("reference level (student/KC sum redundancy)",)


def test_identify_drops_slope_columns_for_never_repeated_kcs():
    """A KC nobody practises twice has T == 0 always: no estimable slope."""
    data = synthetic(n_students=8, n_kcs=12, n_items=12, seed=31, n_reps=1)
    design = build_afm_design(data)
    dropped = design.aliased.by_block()
    assert len(dropped.get("kc_slope", [])) == 12, dropped
    assert design.n_params == design.rank()


def test_aliased_columns_carry_no_information():
    """Re-inserting zeros for the dropped columns reproduces the same likelihood.

    This is the equivalence that licenses dropping them: the identified fit is
    a point of the full design, with the aliased coefficients at zero, and it
    attains exactly the full design's optimum.
    """
    data = synthetic(n_students=8, n_kcs=4, n_items=16, seed=32, n_reps=6)
    full = build_afm_design(data, identify=False)
    ident = full.identify()
    fit = fit_afm(ident, data.y, method="L-BFGS-B", max_fun=200_000,
                  warn_not_converged=False)

    keep = [c not in set(ident.aliased.columns) for c in full.columns]
    w_full = np.zeros(full.n_params)
    w_full[np.flatnonzero(keep)] = fit.weights

    y = np.asarray(data.y, float)
    zero = np.zeros(full.n_params)
    assert (-_objective(w_full, full.matrix, y, zero)
            == pytest.approx(fit.ll_unpenalized, rel=1e-12))
    np.testing.assert_allclose(_expit(full.matrix @ w_full),
                               fit.predict_proba(ident), rtol=0, atol=1e-12)


@pytest.mark.parametrize("make_data,n_aliased", [
    pytest.param(lambda: synthetic(n_students=8, n_kcs=4, n_items=16, seed=33, n_reps=6),
                 1, id="reference-student"),
    pytest.param(co_occurring_kc_data, 3, id="duplicate-kcs"),
])
def test_identification_does_not_change_the_maximised_likelihood(make_data, n_aliased):
    """Same optimum as the unidentified design, just without the columns that
    carry nothing: the phantom reference student, or a KC that tags exactly the
    steps another does (with its slope, and then a reference student). That
    equivalence is what licenses the drop."""
    data = make_data()
    kwargs = {"method": "L-BFGS-B", "max_fun": 200_000,
              "warn_not_converged": False, "warn_separated": False}
    full = fit_afm(build_afm_design(data, identify=False, student_l2=0.0),
                   data.y, **kwargs)
    ident = fit_afm(build_afm_design(data), data.y, **kwargs)
    assert ident.ll_unpenalized == pytest.approx(full.ll_unpenalized, abs=1e-4)
    assert ident.n_params == full.n_params - n_aliased


def test_sum_redundancy_is_detected_for_any_constant_kcs_per_row():
    """The student/KC dependency exists whenever every row has the same m KCs.

    Sum of student columns = 1; sum of KC columns = m * 1. Checking only for
    m == 1 would miss it on every multi-KC export.
    """
    design = build_afm_design(multi_kc_data(), identify=False)
    assert design.kc_per_row() == 2.0
    assert design._has_sum_redundancy()
    assert design.rank() == design.n_params - 1
    assert build_afm_design(multi_kc_data()).n_params == design.n_params - 1


def test_identify_raises_on_a_collinear_extra_block():
    """The guard that protects blocks added later. A copy of the KC intercepts
    under new names is one factor entered twice — every level pairs with
    exactly one of the other's — and is refused by name, rather than resolved
    like a nested factor by silently dropping one copy whole."""
    data = synthetic(n_students=6, n_kcs=3, n_items=12, seed=36, n_reps=5)
    design = build_afm_design(data, identify=False)
    kc_block = design.get("kc_intercept")
    duplicate = Block.build("copy", kc_block.matrix.copy(),
                            [f"dup_{c}" for c in kc_block.columns])
    with pytest.raises(ValueError, match="kc_intercept and copy partition the rows "
                                         "identically"):
        design.with_blocks(duplicate).identify()


def test_identify_raises_on_a_collinear_accumulator():
    """And the same guard for a block that is not a factor at all: a copy of
    the slopes is zero wherever a KC is met for the first time, so no sum
    redundancy can explain it."""
    data = synthetic(n_students=6, n_kcs=3, n_items=12, seed=36, n_reps=5)
    design = build_afm_design(data, identify=False)
    slopes = design.get("kc_slope")
    duplicate = Block.build("counts", slopes.matrix.copy(),
                            [f"dup_{c}" for c in slopes.columns])
    with pytest.raises(ValueError, match="a block added to this design is collinear"):
        design.with_blocks(duplicate).identify()


def test_identify_raises_on_a_hierarchical_parent_block():
    """The hook's other use. A parent appended over the KCs it groups is the
    sum of their intercept columns, so the KC block already spans it, and the
    elimination would take every column of it — as silent as dropping one copy
    of a repeated factor, and refused the same way. Declared before the KCs,
    the same parent keeps every level and its KCs give up one each instead."""
    data = synthetic(n_students=6, n_kcs=4, n_items=12, seed=36, n_reps=5)
    design = build_afm_design(data, identify=False)
    parent = _one_hot([f"g{int(kcs[0][2:]) // 2}" for kcs in data.kcs], "parent")
    with pytest.raises(ValueError, match="parent adds nothing to this design: every column "
                                         "of it lies in the span of kc_intercept"):
        design.with_blocks(parent).identify()

    student, kc_intercept, kc_slope = design.blocks
    identified = Design((student, parent, kc_intercept, kc_slope)).identify()
    assert identified.n_params == identified.rank()
    dropped = identified.aliased.by_block()
    assert len(dropped["student"]) == 1 and len(dropped["kc_intercept"]) == 2
    assert "parent" not in dropped


def test_identify_drops_kcs_that_tag_identical_steps():
    data = co_occurring_kc_data()
    full = build_afm_design(data, identify=False)
    ident = full.identify()

    dropped = dict(zip(ident.aliased.columns, ident.aliased.reasons))
    assert dropped["kc_intercept:B"] == "duplicate of kc_intercept:A"
    assert dropped["kc_slope:B"] == "duplicate of kc_slope:A"
    assert ident.n_params == ident.rank(), "identified design must be full rank"
    assert full.rank() == full.n_params - len(ident.aliased)


def test_duplicate_removal_can_create_the_sum_redundancy_and_it_is_still_broken():
    """Every row here carries the pair, so dedup turns 2 KCs per row into 1.

    The redundancy does not exist in the design as built — KCs per row is 2 —
    and comes into being only once ``B`` is dropped. Deciding reference levels
    before that would leave the design rank-deficient.
    """
    data = co_occurring_kc_data(pair_steps=4, solo_steps=0)
    design = build_afm_design(data, identify=False)
    assert design.kc_per_row() == 2.0, "no sum redundancy to break as built"
    ident = design.identify()
    assert [c for c in ident.aliased.columns if c.startswith("student:")]
    assert ident.n_params == ident.rank()


def _two_cohort_data(n_per_cohort=4, n_steps=4):
    """Two groups of students that share no items and no KCs."""
    rows = []
    for cohort, kcs in enumerate((("A", "B"), ("C", "D"))):
        for i in range(n_per_cohort):
            for j in range(n_steps):
                rows.append({"Anon Student Id": f"c{cohort}s{i}",
                             "Problem Name": f"p{cohort}",
                             "Step Name": f"c{cohort}st{j}",
                             "First Attempt": "correct" if (i + j) % 3 else "incorrect",
                             "kc": kcs[j % 2], "opp": f"{j // 2 + 1}"})
    return step_data(rows)


def _one_hot(labels: list[str], name: str) -> Block:
    """A crossed factor as a design block: one column per level, one per row."""
    return Block.from_levels(name, [(v,) for v in labels])


def test_a_third_partitioning_block_carries_a_second_redundancy():
    """``m`` blocks that each cover every row span the all-ones direction ``m``
    times over, so they carry ``m - 1`` dependencies rather than one.

    The third block here is crossed with both students and KCs — every
    combination occurs — so the all-ones relation is the *only* thing relating
    it to them, which is exactly what the pass models. The KC block still keeps
    every level: the reason for dropping a student rather than a KC does not
    weaken when a third factor joins, since the KC intercepts are still the
    reported output.
    """
    data = synthetic(n_students=6, n_kcs=3, n_items=12, seed=11, n_reps=6)
    third = _one_hot([f"g{i % 4}" for i in range(len(data))], "cohort")

    two = build_afm_design(data, identify=False)
    three = two.with_blocks(third)
    assert three.rank() == three.n_params - 2, "two redundancies to break, not one"

    identified = three.identify()
    assert identified.n_params == identified.rank()
    assert len(identified.aliased) == 2
    blocks = {c.split(":")[0] for c in identified.aliased.columns}
    assert blocks == {"student", "cohort"}, "the student block first, then latest-declared"


def test_partitioning_is_detected_from_the_row_sums_not_from_a_block_name():
    """The old pass looked for blocks literally named ``student`` and
    ``kc_intercept``. Two blocks named neither, both covering every row, are
    just as dependent and are now identified as such."""
    labels_a = [f"a{i % 3}" for i in range(60)]
    labels_b = [f"b{i % 4}" for i in range(60)]
    design = Design((_one_hot(labels_a, "left"), _one_hot(labels_b, "right")))
    assert design.rank() == design.n_params - 1

    identified = design.identify()
    assert identified.n_params == identified.rank()
    assert len(identified.aliased) == 1
    assert identified.aliased.columns[0].startswith("right:"), "latest-declared gives way"
    assert "sum redundancy across left, right" in identified.aliased.reasons[0]


def test_a_block_that_leaves_a_row_at_zero_does_not_partition():
    """Covering every row is what makes a block span the all-ones direction.
    One that misses a row cannot, so it neither carries a redundancy nor joins
    the graph the components are cut from."""
    labels = [f"a{i % 3}" for i in range(30)]
    partial = np.zeros((30, 2))
    partial[: 20, 0] = 1.0
    partial[20:29, 1] = 1.0          # row 29 is left at zero
    design = Design((_one_hot(labels, "left"),
                     Block.build("sparse", partial, ["p", "q"])))
    assert design._covering_blocks() == ["left"]
    assert not design._has_sum_redundancy()
    assert design.identify().n_params == design.n_params


def test_row_components_separates_cohorts_that_share_no_material():
    design = build_afm_design(_two_cohort_data(), identify=False)
    labels = design.row_components()
    assert set(labels.tolist()) == {0, 1}
    assert len(np.unique(labels[:16])) == 1, "one cohort's rows come first here"


def test_identify_drops_one_reference_student_per_component():
    """Each component shifts independently, so each carries its own redundancy."""
    data = _two_cohort_data()
    full = build_afm_design(data, identify=False)
    assert full.rank() == full.n_params - 2, "two redundancies, not one"

    ident = full.identify()
    assert len(ident.aliased.by_block()["student"]) == 2
    assert all("component" in reason for reason in ident.aliased.reasons
               if "reference level" in reason)
    assert ident.n_params == ident.rank()


def test_a_cohort_of_one_student_gives_up_its_only_student():
    """The student block is the one block allowed to go whole. One student per
    cohort is still one reference level per component, as it always was;
    refusing it would refuse every single-student export."""
    data = _two_cohort_data(n_per_cohort=1, n_steps=6)
    ident = build_afm_design(data, identify=False).identify()
    assert ident.n_params == ident.rank()
    assert len(ident.aliased.by_block()["student"]) == 2
    assert ident.get("student").matrix.shape[1] == 0


def test_a_component_without_the_sum_redundancy_keeps_every_student():
    """One cohort tags some steps with two KCs: no redundancy there to break.

    Observed on the spacing-exp2 export's ``Question Group`` model, where nine
    of the ten courses contribute a reference student and the tenth does not.
    """
    rows = []
    for j in range(4):  # cohort 0: one KC per row -> redundant
        for i in range(4):
            rows.append({"Anon Student Id": f"c0s{i}", "Problem Name": "p0",
                         "Step Name": f"c0st{j}",
                         "First Attempt": "correct" if (i + j) % 3 else "incorrect",
                         "kc": ("A", "B")[j % 2], "opp": f"{j // 2 + 1}"})
    # cohort 1: one KC on some rows, two on others -> no redundancy to break
    tagging = [("C", "1"), ("D", "1"), ("C~~D", "2~~2"),
               ("C", "3"), ("D", "3"), ("C~~D", "4~~4")]
    for j, (kc, opp) in enumerate(tagging):
        for i in range(4):
            rows.append({"Anon Student Id": f"c1s{i}", "Problem Name": "p1",
                         "Step Name": f"c1st{j}",
                         "First Attempt": "correct" if (i + j) % 2 else "incorrect",
                         "kc": kc, "opp": opp})
    data = step_data(rows)
    full = build_afm_design(data, identify=False)
    assert full.rank() == full.n_params - 1, "only cohort 0 carries a redundancy"

    ident = full.identify()
    students = ident.aliased.by_block()["student"]
    assert len(students) == 1 and students[0].startswith("c0")
    assert ident.n_params == ident.rank()


# --------------------------------------------------------------------------
# Separation: coefficients with no finite MLE
# --------------------------------------------------------------------------


def test_an_always_correct_kc_is_reported_as_separated():
    data = from_frame(separated_frame(), "M")
    design = build_afm_design(data)
    sep = design.separated(data.y)
    assert "kc_intercept:kc0" in sep.columns
    assert sep.directions[sep.columns.index("kc_intercept:kc0")] == 1
    assert "kc_intercept:kc1" not in sep.columns


def test_a_separated_coefficient_has_no_maximum():
    """Checked against the objective itself, not against solver behaviour.

    The claim ``Design.separated`` makes is that the likelihood improves
    without bound along that coordinate. Walk the coefficient by hand and watch
    the negative log-likelihood fall monotonically, while the same walk along an
    identified coefficient turns around at an interior minimum. This is the
    definition of "no maximizer", and it holds whatever the optimizer does —
    which matters, because TNC halts a diverging coefficient around 19 on its
    own gradient criterion, so a test that watched the fitted number grow would
    be testing the solver instead.
    """
    data = from_frame(separated_frame(), "M")
    design = build_afm_design(data)
    fit = fit_afm(design, data.y, warn_not_converged=False, warn_separated=False)
    X, y, l2 = design.matrix, np.asarray(data.y, dtype=float), design.l2

    def walk(name):
        j = design.columns.index(name)
        out = []
        for value in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            w = fit.weights.copy()
            w[j] = value
            out.append(_objective(w, X, y, l2))
        return out

    diverging = walk("kc_intercept:kc0")
    identified = walk("kc_intercept:kc1")
    assert all(a > b for a, b in pairwise(diverging)), (
        "a separated coefficient must keep improving the fit as it grows")
    assert identified[-1] > min(identified), (
        "an identified coefficient must have an interior optimum")


def test_a_ridge_removes_the_separation():
    """A penalty supplies the missing curvature, so the MLE exists again."""
    data = from_frame(separated_frame(), "M")
    assert len(build_afm_design(data, student_l2=0.0).separated(data.y)) > 0
    penalized = build_afm_design(data, identify=False, student_l2=1.0)
    kc_blocks = [b.name for b in penalized.blocks]
    assert "kc_intercept" in kc_blocks  # unpenalized, so it is still flagged
    assert not any(c.startswith("student:")
                   for c in penalized.separated(data.y).columns)


def test_a_lower_bound_absorbs_a_downward_divergence():
    """bound_slopes rests an all-failure slope on 0 instead of sending it to -inf."""
    rows = []
    for s in range(8):
        for r in range(4):
            rows.append({
                "Anon Student Id": f"S{s}", "Problem Name": "p", "Step Name": "st0",
                "First Attempt": "incorrect", "kc": "kc0", "opp": str(r + 1)})
            rows.append({
                "Anon Student Id": f"S{s}", "Problem Name": "p", "Step Name": "st1",
                "First Attempt": "correct" if r % 2 else "incorrect",
                "kc": "kc1", "opp": str(r + 1)})
    data = step_data(rows)

    free = build_afm_design(data)
    bounded = build_afm_design(data, bound_slopes=True)
    assert "kc_slope:kc0" in free.separated(data.y).columns
    assert "kc_slope:kc0" not in bounded.separated(data.y).columns


def test_a_well_behaved_design_reports_no_separation():
    data, _ = synthetic(20, 5, 20, seed=7, return_truth=True)
    design = build_afm_design(data)
    assert len(design.separated(data.y)) == 0
    fit = fit_afm(design, data.y)
    assert len(fit.separated) == 0
    assert "separation" not in fit.summary()
