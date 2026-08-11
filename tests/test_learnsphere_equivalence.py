"""Equivalence against LearnSphere's own output, on LearnSphere's own input.

Everything in ``test_afm.py`` runs without data. This file is the other half:
it fits the real E-learning 2022 export that DataShop workflow ``wf3990``
processed, and checks our numbers against the ``model_values.xml`` that same
workflow produced. Skipped when the artifacts are absent, so a bare clone still
has a green suite.

Point it somewhere else with ``AFM_WF3990_DIR``.

The acceptance criterion for compat mode is deliberately two-sided. For each
KC model we require **either**

* our fit statistics match LearnSphere within ``NAT_TOLERANCE``, **or**
* our likelihood is strictly *better* and our fit carries a KKT optimality
  certificate — i.e. the gap is LearnSphere stopping early, not us diverging.

Only the second branch is allowed to be large, and only where it is provably
their optimizer: on a convex objective a certified stationary point is the
global optimum, so nothing can beat it.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from itertools import pairwise

import pytest

from leapfit import build_afm_design, fit_afm, load_student_step

WF3990_DIR = os.environ.get(
    "AFM_WF3990_DIR", "results/wf3990_results_2024_1206-10-runs")
EXPORT = os.path.join(WF3990_DIR, "Data-1-x435245_all-kc-merged.txt")
MODEL_VALUES = os.path.join(WF3990_DIR, "Analysis-1-x916817_model_values.xml")

NAT_TOLERANCE = 20.0   # agreement band on AIC/BIC, in nats
MAX_FUN = 500_000      # enough for our side to reach the optimum

pytestmark = pytest.mark.skipif(
    not (os.path.exists(EXPORT) and os.path.exists(MODEL_VALUES)),
    reason=f"wf3990 artifacts not found under {WF3990_DIR!r}",
)


def learnsphere_values() -> dict[str, dict[str, float]]:
    """The unsuffixed model instance from each of the workflow's 10 CV replicates."""
    root = ET.parse(MODEL_VALUES).getroot()
    out = {}
    for model in root.findall("model"):
        name = model.find("name").text
        base = re.match(r"KC \((?P<n>.+?)(-\d+)?\)", name).group("n")
        if base == name[4:-1]:
            out[base] = {t: float(model.find(t).text)
                         for t in ("AIC", "BIC", "log_likelihood")}
    return out


@pytest.fixture(scope="module")
def reference() -> dict[str, dict[str, float]]:
    return learnsphere_values()


def test_the_export_is_elearning_22(reference):
    data = load_student_step(EXPORT, kc_model="LOs-new-MCQ")
    assert len(data) == 42_176
    assert len(data.student_names) == 39
    assert len(set(data.items)) == 1_865
    assert len(reference) == 10


def test_compat_reproduces_the_learnsphere_parameter_count(reference):
    """nPars recovered from their own output as (AIC + 2*ll)/2."""
    for name, ref in reference.items():
        data = load_student_step(EXPORT, kc_model=name)
        implied = (ref["AIC"] + 2 * ref["log_likelihood"]) / 2
        ours = build_afm_design(data, learnsphere_compat=True).n_params
        assert ours == pytest.approx(implied, abs=1e-6), name
        assert ours == len(data.student_names) + 2 * len(data.kc_names), name


def test_compat_reproduces_the_learnsphere_fit(reference):
    """Match within tolerance, or beat them from a certified optimum."""
    beaten = []
    for name, ref in reference.items():
        data = load_student_step(EXPORT, kc_model=name)
        design = build_afm_design(data, learnsphere_compat=True)
        fit = fit_afm(design, data.y, method="TNC", max_fun=MAX_FUN,
                      warn_not_converged=False)

        d_aic = fit.aic - ref["AIC"]
        if abs(d_aic) <= NAT_TOLERANCE:
            assert abs(fit.bic - ref["BIC"]) <= NAT_TOLERANCE, name
            continue

        assert fit.ll_unpenalized > ref["log_likelihood"], (
            f"{name}: we differ from LearnSphere by {d_aic:.1f} nats of AIC and "
            f"our likelihood is worse — that is our bug, not their optimizer."
        )
        assert fit.is_optimal, (
            f"{name}: we claim a better likelihood but our own fit is not at a "
            f"stationary point (max|grad| {fit.max_free_gradient:.3g})."
        )
        beaten.append((name, -d_aic / 2))

    # Only the heavily over-parameterized models should land here.
    assert {n for n, _ in beaten} <= {"concept", "Unique-step-MCQ"}, beaten


def test_identification_only_removes_phantom_parameters(reference):
    """Analysis mode drops columns that cannot be estimated, and nothing else.

    The likelihood is unchanged (up to the ridge that compat also carries), and
    the parameter count falls by exactly the measured rank deficiency.
    """
    for name in ("LOs-new-MCQ", "pmi", "concept", "Unique-step-MCQ"):
        data = load_student_step(EXPORT, kc_model=name)
        full = build_afm_design(data, identify=False, student_l2=0.0)
        ident = build_afm_design(data)

        assert ident.n_params == full.rank(), name
        assert full.n_params - ident.n_params == len(ident.aliased), name

        a = fit_afm(full, data.y, method="TNC", max_fun=MAX_FUN,
                    warn_not_converged=False)
        b = fit_afm(ident, data.y, method="TNC", max_fun=MAX_FUN,
                    warn_not_converged=False)
        assert b.ll_unpenalized == pytest.approx(a.ll_unpenalized, abs=0.5), (
            f"{name}: dropping aliased columns changed the likelihood, so they "
            "were not aliased after all."
        )
        assert b.is_optimal, name


def test_phantom_parameters_scale_with_granularity(reference):
    """The overcount is not uniform — it tracks how fine the KC model is."""
    counts = {}
    for name in ("Single-KC-MCQ", "LOs-new-MCQ", "pmi", "concept", "Unique-step-MCQ"):
        data = load_student_step(EXPORT, kc_model=name)
        counts[name] = (len(data.kc_names),
                        len(build_afm_design(data).aliased))
    assert counts["Unique-step-MCQ"][1] == 959
    assert counts["concept"][1] == 14
    assert counts["Single-KC-MCQ"][1] == 1
    fine_to_coarse = [counts[n][1] for n in
                      ("Unique-step-MCQ", "concept", "pmi", "Single-KC-MCQ")]
    assert fine_to_coarse == sorted(fine_to_coarse, reverse=True)


def test_never_repeated_kcs_report_undefined_slopes(reference):
    data = load_student_step(EXPORT, kc_model="Unique-step-MCQ")
    fit = fit_afm(build_afm_design(data), data.y, method="TNC", max_fun=MAX_FUN,
                  warn_not_converged=False)
    values = fit.kc_values(data)
    assert len(values) == len(data.kc_names) == 1_865
    assert int(values["Slope"].isna().sum()) == 958


def test_file_and_recomputed_opportunities_agree_almost_everywhere():
    """DataShop's Opportunity column follows *row* order, and row order is not
    sorted by time.

    The E-learning 2022 export contains 12 within-student inversions of
    ``First Transaction Time`` — an attempt at 01:44:41 listed before one at
    01:44:36 — and the shipped counts follow the file, so 28 rows carry the
    wrong opportunity number. Every disagreement traces to one of those
    inversions; none is a same-second tie.
    """
    data = load_student_step(EXPORT, kc_model="LOs-new-MCQ")
    disagree = set(data.opportunity_disagreements().tolist())
    assert 0 < len(disagree) / len(data) < 0.001

    inverted: set[int] = set()
    rows_by_student: dict[str, list[int]] = {}
    for i, s in enumerate(data.students):
        rows_by_student.setdefault(s, []).append(i)
    for rows in rows_by_student.values():
        for a, b in pairwise(rows):
            if data.times[b] < data.times[a]:
                inverted.update((a, b))
    assert inverted, "expected the export's row order to contain time inversions"

    for i in disagree:
        student = data.students[i]
        near = [j for j in rows_by_student[student] if abs(j - i) <= 2]
        assert inverted & set(near), (
            f"row {i} disagrees but no time inversion sits next to it"
        )


def test_recomputed_opportunities_barely_move_the_fit():
    """The column is wrong on 0.07% of rows, and it does not matter numerically.

    Worth pinning: it licenses keeping DataShop's column as the default (so the
    published baseline stays reproducible) while offering the corrected
    ordering for new work.
    """
    data = load_student_step(EXPORT, kc_model="LOs-new-MCQ")
    a = fit_afm(build_afm_design(data), data.y, method="TNC",
                max_fun=MAX_FUN, warn_not_converged=False)
    b = fit_afm(build_afm_design(data, recompute_opportunities=True), data.y,
                method="TNC", max_fun=MAX_FUN, warn_not_converged=False)
    assert a.n_params == b.n_params
    assert abs(a.ll_unpenalized - b.ll_unpenalized) < 1.0
