"""LFA against DataShop's own LFA, on a run of the reference tool.

The LFA search engine ships only as a binary, so unlike AFM and PFA there is no
reference source to read. What there is instead is a *run*: DataShop's offline
`lfa-6.0` tool was executed locally against the KCluster-tagged E-learning 2022
export, and its inputs, its `allModels.txt`, its Q matrices and a per-model
comparison were kept under `results/lfa-learnsphere-20260902/`. This suite
turns that run into a fixture. Point it elsewhere with ``LFA_BUNDLE_DIR``.

Three kinds of claim are pinned here, and they fail for different reasons.

**The conventions** are checked from the reference's own reported numbers
alone, with no refitting — the same trick as
``test_afm.py::test_published_aic_bic_identity``. If ``nPars`` or the ``BIC``
formula ever drifts, these break without needing any data at all.

**The agreement** is the load-bearing one.
``test_leapfit_reproduces_the_reference_or_beats_it_certified`` encodes the
acceptance criterion the AFM suite uses: for every model, either our fit
statistics match within ``NAT_TOLERANCE``, **or** our likelihood is strictly
better *and* our fit carries a KKT optimality certificate. On a convex
objective nothing can beat a certified optimum, so "better and certified" is a
pass, not a discrepancy. `docs/DESIGN.md` quotes 1.2e-9 for the top model;
``test_the_reference_top_model_matches_to_numerical_precision`` is what keeps
that number honest.

**The defect** is pinned too, in
``test_the_reference_selection_carries_a_kc_with_no_finite_estimate`` and
``test_leapfit_refuses_the_move_the_reference_took``. The reference selected a
KC model containing a slope with no finite MLE, and that move then appeared in
all 99 states it reported. Both halves are asserted, because a regression that
made leapfit accept that move would otherwise look like a search that simply
found a different answer.
"""

from __future__ import annotations

import gzip
import os
import re

import numpy as np
import pytest

from leapfit import build_afm_design, fit_afm, load_student_step
from leapfit.lfa import build_factor_matrix, lfa_search, relabel

BUNDLE = os.environ.get("LFA_BUNDLE_DIR", "results/lfa-learnsphere-20260902")
EXPORT = os.path.join(BUNDLE, "inputs", "student-step-tagged.txt.gz")
RUNS = ("run-smoke", "run-bic", "run-aic")

#: The corpus the run was made on, asserted rather than assumed — a fixture
#: pointing at different data would otherwise fail in confusing ways.
N_OBS, N_STUDENTS = 20687, 39

#: Agreement band on the fit statistics. Far tighter than the AFM suite's 20
#: nats, because here we reproduce the reference's *inner model* exactly rather
#: than a differently-optimized fit of the same model. It is not arbitrary: a
#: KKT-certified fit still permits a likelihood shortfall of order ``g^2/(2H)``
#: — about 2e-5 nats at this sample size, per the solver notes in
#: `docs/DESIGN.md` — so two certified fits of the same model may disagree by
#: that much in either direction. Measured worst case over the three runs:
#: 4.3e-05 nats, on a state both sides certify.
NAT_TOLERANCE = 1e-4

#: The KC model whose skills supplied P, and the factor the reference chose at
#: its second expansion — the move with no finite estimate.
FACTOR_MODEL = "LOs-new-MCQ"
BAD_FACTOR = "5.2a recall_goals_cta"

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(BUNDLE) and os.path.exists(EXPORT)),
    reason=f"LFA reference-run bundle not found under {BUNDLE!r}",
)


# --------------------------------------------------------------------------
# Reading the fixture
# --------------------------------------------------------------------------


def _equivalence(run):
    import pandas as pd
    return pd.read_csv(os.path.join(BUNDLE, run, "equivalence.csv"))


def _step_keys(data):
    """The step identity the bundle's Q matrices are indexed by.

    ``StepData.items`` is ``Problem Name ## Step Name``; the tool's inputs were
    generated with DataShop's fuller ``hierarchy;problem;step``, so the fixture
    carries that form and the mapping is rebuilt from the source table.
    """
    source, rows = data.source, data.source_rows
    return [f"{h};{p};{s}" for h, p, s in zip(
        source["Problem Hierarchy"][rows], source["Problem Name"][rows],
        source["Step Name"][rows])]


def _read_qmatrix(path):
    """``step -> KC label`` from a `NamedMatrix` text file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        skills = fh.readline().rstrip("\n").split("\t")[1:]
        out = {}
        for line in fh:
            cells = line.rstrip("\n").split("\t")
            if len(cells) < 2:
                continue
            on = [skills[j] for j, v in enumerate(cells[1:])
                  if v not in ("0", "0.0", "")]
            assert len(on) == 1, f"{path} tags a step with {len(on)} KCs"
            out[cells[0]] = on[0]
    return out


def _reported(run):
    """``model index -> (ll, aic, bic, rmse, change history)`` from allModels."""
    with open(os.path.join(BUNDLE, run, "allModels.txt")) as fh:
        text = fh.read()
    parts = re.split(r"\nModel (\d+):\n", text)
    out = {}
    for i in range(1, len(parts), 2):
        body = parts[i + 1]
        stats = re.search(
            r"LL: ([-\d.E]+)\tAIC: ([-\d.E]+)\tBIC: ([-\d.E]+)\tRMSE: ([-\d.E]+)",
            body)
        history = re.search(r"Change history: \t(.*)", body)
        if stats:
            out[int(parts[i])] = (*(float(x) for x in stats.groups()),
                                  history.group(1).strip() if history else "")
    return out


@pytest.fixture(scope="module")
def data():
    return load_student_step(EXPORT, kc_model=FACTOR_MODEL)


def _refit(data, labels_by_step):
    """Refit a fixture Q matrix under the reference's own conventions.

    The bundle indexes steps as ``hierarchy;problem;step`` while
    :attr:`~leapfit.data.StepData.items` is ``problem ## step``, so the
    labelling is translated into leapfit's own item space first — the two are
    in correspondence observation by observation.
    """
    by_item = dict(zip(data.items, (labels_by_step[k] for k in _step_keys(data))))
    steps = tuple(by_item)
    scored = relabel(data, steps, tuple(by_item[s] for s in steps))
    design = build_afm_design(scored, learnsphere_compat=True,
                              recompute_opportunities=False)
    return fit_afm(design, scored.y, warn_not_converged=False,
                   warn_separated=False), scored


# --------------------------------------------------------------------------
# The conventions, from the reference's own numbers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("run", RUNS)
def test_the_reference_parameter_count_is_students_plus_two_per_kc(run):
    """Recovered from the reference's output without refitting anything."""
    table = _equivalence(run)
    recovered = (table["ref_aic"] + 2 * table["ref_ll"]) / 2
    expected = N_STUDENTS + 2 * table["n_kcs"]
    assert np.allclose(recovered, expected, atol=1e-6), (
        "nPars = n_students + 2*n_KCs, with no intercept column")
    assert (table["n_params"] == expected).all(), "and leapfit agrees with it"


@pytest.mark.parametrize("run", RUNS)
def test_the_reference_bic_minus_aic_closes_on_its_own_numbers(run):
    table = _equivalence(run)
    predicted = table["n_params"] * (np.log(N_OBS) - 2)
    assert np.allclose(table["ref_bic"] - table["ref_aic"], predicted, atol=1e-6)


def test_the_aic_and_bic_runs_diverge_only_where_the_frontier_mixes_depths():
    """Where states share ``nPars`` the two criteria differ by a constant and
    cannot order them differently; where a state has *fewer* parameters, BIC's
    heavier per-parameter charge favours it and AIC's does not.

    Measured: the two runs share 98 of their 99 reported states and agree on
    the whole head of the ranking. They part over exactly one state — the only
    2-split model either run reports, which BIC keeps and AIC drops for a
    3-split one. This is the same mechanism as the 9.94-against-2.00 charge in
    `docs/DESIGN.md`, caught at the one place in this run where it can act.
    """
    bic, aic = _equivalence("run-bic"), _equivalence("run-aic")
    shared = set(bic["change_history"]) & set(aic["change_history"])
    assert len(shared) == 98, "the two criteria are close, not interchangeable"

    head = 50
    assert bic["change_history"][:head].equals(aic["change_history"][:head]), (
        "and they agree on the head of the ranking, where every state shares p")

    only_bic = set(bic["change_history"]) - shared
    only_aic = set(aic["change_history"]) - shared
    assert len(only_bic) == len(only_aic) == 1
    depth = lambda h: h.count("split ")  # noqa: E731
    assert depth(next(iter(only_bic))) < depth(next(iter(only_aic))), (
        "the state BIC keeps and AIC drops is the shallower, cheaper one")
    assert sorted(bic["n_params"].unique()) == [45, 47], "two parameter counts"
    assert aic["n_params"].nunique() == 1, "AIC reports only the deeper states"


# --------------------------------------------------------------------------
# The agreement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("run", RUNS)
def test_leapfit_reproduces_the_reference_or_beats_it_certified(run, data):
    """The two-sided criterion: match, or be strictly better and certified."""
    folder = os.path.join(BUNDLE, run, "qmatrix-top5")
    checked = 0
    for name in sorted(os.listdir(folder)):
        index = int(re.search(r"QMatrix_(\d+)", name).group(1))
        reference = _reported(run)[index]
        fit, _ = _refit(data, _read_qmatrix(os.path.join(folder, name)))
        delta = fit.ll - reference[0]
        assert delta > -NAT_TOLERANCE, (
            f"{run} model {index}: our likelihood is {-delta:.3g} nats worse")
        if abs(delta) > NAT_TOLERANCE:
            assert fit.is_optimal, (
                f"{run} model {index}: we claim {delta:.3g} nats better but "
                "the fit is not at a certified optimum, so the claim is unbacked")
        checked += 1
    assert checked == 5, "the bundle keeps the top five Q matrices per run"


def test_the_reference_top_model_matches_to_numerical_precision(data):
    """The 1.2e-9 that `docs/DESIGN.md` quotes, kept honest."""
    fit, _ = _refit(
        data, _read_qmatrix(os.path.join(BUNDLE, "run-smoke", "qmatrix-top5",
                                         "QMatrix_1.txt")))
    ll, aic, bic, _rmse, _ = _reported("run-smoke")[1]
    assert fit.n_obs == N_OBS
    assert fit.ll == pytest.approx(ll, abs=1e-7)
    assert fit.aic == pytest.approx(aic, abs=1e-7)
    assert fit.bic == pytest.approx(bic, abs=1e-7)


@pytest.mark.parametrize("run", RUNS)
def test_leapfit_is_never_worse_than_the_reference(run):
    """Recorded over all 99 models per run, not just the five kept in full."""
    table = _equivalence(run)
    assert len(table) == 99
    assert (table["d_ll"] > -NAT_TOLERANCE).all(), (
        "beyond the certificate's own tolerance, every discrepancy runs one "
        "way: the reference stops early, we do not")
    material = table[table["d_ll"].abs() > NAT_TOLERANCE]
    assert (material["d_ll"] > 0).all(), (
        "and no material discrepancy favours the reference")


def test_the_reference_understates_more_likelihoods_as_the_search_deepens():
    """21 of 99 at depth 1, all 99 at depth 3 — the reason the frontier moves."""
    shallow = _equivalence("run-smoke")
    deep = _equivalence("run-bic")
    assert (shallow["d_ll"] > 1e-4).sum() == 21
    assert (deep["d_ll"] > 1e-4).sum() == 99
    assert shallow["d_ll"].max() > 6.0, "up to 6.57 nats at depth 1"


# --------------------------------------------------------------------------
# The defect, and that leapfit does not reproduce it
# --------------------------------------------------------------------------


def test_the_reference_selection_carries_a_kc_with_no_finite_estimate(data):
    fit, _ = _refit(
        data, _read_qmatrix(os.path.join(BUNDLE, "run-bic", "qmatrix-top5",
                                         "QMatrix_1.txt")))
    by_block = fit.separated.by_block()
    assert by_block.get("kc_slope"), (
        "the model the reference selected has a slope with no finite MLE")
    assert any(BAD_FACTOR in column for column in by_block["kc_slope"])


def test_the_bad_move_reached_every_state_the_reference_reported():
    """Split-only search cannot back out, so one bad move contaminates all."""
    table = _equivalence("run-bic")
    assert table["change_history"].str.contains(BAD_FACTOR, regex=False).all()


def test_leapfit_refuses_the_move_the_reference_took(data):
    """The end-to-end claim: on this data, the screens change the trajectory."""
    factors = build_factor_matrix({FACTOR_MODEL: data})
    result = lfa_search(data, factors, max_iterations=2, patience=0)
    refused = [move for move in result.rejected.moves if BAD_FACTOR in move]
    assert refused, f"{BAD_FACTOR!r} must be offered and refused"
    assert result.rejected.by_reason().get("separated"), "and refused for that"
    for state in result.states:
        history = " ".join(str(move) for move in state.history)
        assert BAD_FACTOR not in history, "no accepted state may contain it"
