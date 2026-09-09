"""Equivalence against the LKT package's own published output.

The reference's ``Examples`` vignette is precompiled, so the CRAN tarball ships
both halves of a fixture: ``data/largerawsample.rda`` is the input, and
``inst/doc/Examples.html`` carries the log-likelihood each model reached,
printed to eight decimals. That pair does for LKT what ``model_values.xml``
does for LearnSphere — it lets us check the numbers without an R interpreter,
and without the R package's dependency stack (``LiblineaR``, ``SparseM``,
``data.table``, ``lme4``, ``glmnet``).

Skipped when the converted input is absent, so a bare clone stays green. The
dataset is not vendored: it is GPL-3, this repository is MIT, and it is 1.5 MB.
To produce it::

    curl -O https://cran.r-project.org/src/contrib/LKT_1.7.0.tar.gz
    uv run --with pyreadr python results/lkt-vignette/scripts/make_fixture.py \\
        --tarball LKT_1.7.0.tar.gz

Point it somewhere else with ``LKT_VIGNETTE_DIR``.

**The acceptance criterion is the two-sided one this suite uses elsewhere**
(see ``test_learnsphere_equivalence.py``): either we agree within a tolerance,
or our likelihood is strictly *better* and our fit carries a KKT optimality
certificate — on a convex objective a certified stationary point is the global
optimum, so nothing can beat it and a gap in that direction is the reference's
optimizer stopping early. Which is what happens: the reference calls
``LiblineaR`` at its default ``epsilon = 1e-4``, and the published fit sits
about half a nat below the optimum of the objective it is solving.
``test_the_gap_is_the_solver_not_the_parameterization`` pins the decomposition
so that "half a nat" cannot quietly become something else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest

from leapfit import (
    REFERENCE_COST,
    build_lkt_design,
    fit_lkt,
    lkt_terms,
    load_student_step,
)
from leapfit.design import Design

VIGNETTE_DIR = os.environ.get("LKT_VIGNETTE_DIR", "results/lkt-vignette")
EXPORT = os.path.join(VIGNETTE_DIR, "lkt-largerawsample-step.txt")

#: What the vignette's own preparation leaves: 58,316 raw rows less 3,194 STUDY
#: events. If the fixture disagrees, it was not built the same way and no
#: comparison below means anything.
N_OBS = 55_122
N_STUDENTS = 478
N_KCS = 72


@dataclass(frozen=True)
class Chunk:
    """One ``LKT()`` call from the vignette and the numbers it printed."""

    components: tuple[str, ...]
    features: tuple[str, ...]
    pars: tuple
    log_likelihood: float
    mcfadden: float
    n_params: int

    def terms(self):
        return lkt_terms(self.components, self.features, self.pars)


_STUDENT_AND_KC = ("student", "kc", "kc", "kc")
_TWO_INTERCEPTS = ("intercept", "intercept")

#: The vignette chunks this suite reproduces, each identified by its own
#: heading in ``inst/doc/Examples.html``. The three parametric ones fix every
#: parameter through ``fixedpars``, which matters: the reference's parameter
#: *search* silently stops recomputing features that share a parameter or whose
#: first parameter is fixed (``LKTfunctions.R:151`` against ``:163-218``), so a
#: chunk with a free parameter is not a sound target to reproduce.
#:
#: All but the first pass ``interc=TRUE``. A global intercept is not
#: implemented here, and on these specs it cannot change the answer by more
#: than the ridge does: each already carries a per-level intercept on two
#: components, so the all-ones column is inside the span either way and the
#: rank is the same. What differs is which columns carry it — a
#: reparameterization, and under a ridge that moves the objective a little.
#: ``test_the_gap_is_the_solver_not_the_parameterization`` measures how much.
#:
#: Two details of the specs themselves are easy to misread. ``lineafm`` in the
#: AFM chunk carries no ``$``, so its slope is one shared coefficient — the
#: chunk is labelled AFM but fits AFM's pooled variant. And ``KC..Default.``
#: has been overwritten with the problem name before any of these run, so the
#: "KC" component throughout is really the problem.
CHUNKS = {
    "AFM": Chunk(
        ("student", "kc", "kc"), ("intercept", "intercept", "lineafm"),
        (None, None, None), -27347.20717315, 0.280024, N_KCS + N_STUDENTS - 1 + 1),
    "logitdec + recency": Chunk(
        _STUDENT_AND_KC, (*_TWO_INTERCEPTS, "logitdec", "recency"),
        (None, None, 0.9, 0.5), -25474.53126101, 0.329326, N_KCS + N_STUDENTS - 1 + 2),
    "PPE": Chunk(
        _STUDENT_AND_KC, (*_TWO_INTERCEPTS, "ppe", "logitdec"),
        (None, None, (0.3491901, 0.2045801, 1e-05, 0.9734477), 0.4443027),
        -24695.58586712, 0.349833, N_KCS + N_STUDENTS - 1 + 2),
    "base4": Chunk(
        _STUDENT_AND_KC, (*_TWO_INTERCEPTS, "base4", "logitdec"),
        (None, None, (0.1890747, 0.6309054, 0.05471752, 0.5), 0.2160748),
        -25969.92489408, 0.316284, N_KCS + N_STUDENTS - 1 + 2),
}

#: Enough for our side to reach a certified optimum on 55k rows.
TIGHT = {"method": "L-BFGS-B", "max_fun": 500_000, "tol": 1e-14}

#: The published fit is short of its own optimum by ~0.48 nats. The band is
#: wide enough not to fail on a solver detail and tight enough that a real
#: divergence — a mis-specified term, a count off by one — could not hide in it:
#: one misplaced opportunity count moves the likelihood by far more.
NAT_TOLERANCE = 2.0

pytestmark = pytest.mark.skipif(
    not os.path.exists(EXPORT),
    reason=f"LKT vignette input not found at {EXPORT!r}; see this module's docstring",
)


@pytest.fixture(scope="module")
def data():
    return load_student_step(EXPORT, kc_model="Default")


@pytest.fixture(scope="module")
def fit(data):
    """The AFM chunk, which the decomposition tests below take apart."""
    design = build_lkt_design(data, CHUNKS["AFM"].terms(), cost=REFERENCE_COST)
    return design, fit_lkt(design, data.y, **TIGHT)


def _null_ll(y) -> float:
    p = float(np.mean(y))
    return float(np.sum(np.where(y == 1, np.log(p), np.log1p(-p))))


def test_the_fixture_is_the_one_the_vignette_fitted(data):
    """Guard the comparison before making it."""
    assert len(data) == N_OBS
    assert len(data.student_names) == N_STUDENTS
    assert len(data.kc_names) == N_KCS


def test_the_specification_reproduces_the_references_parameterization(data, fit):
    """``sparse.model.matrix`` on ``~ lineafm + kc + student + 0`` gives the
    first factor full dummies and the second treatment contrasts: every KC, one
    student dropped, one pooled slope. Identification arrives at the same count
    from the other direction — by removing the redundancy rather than by
    choosing a contrast."""
    design, fitted = fit
    assert design.n_params == N_KCS + (N_STUDENTS - 1) + 1
    assert design.n_params == design.rank()
    assert len(design.aliased) == 1
    assert design.aliased.columns[0].startswith("student:")
    assert fitted.n_params == design.n_params


@pytest.mark.parametrize("name", list(CHUNKS))
def test_the_published_log_likelihood_is_reproduced(data, name):
    """Every vignette chunk whose parameters are all fixed, in one pass.

    Between them these exercise the whole implemented surface: prior counts
    (``AFM``), a decayed outcome history with the reference's undocumented
    60-trial window and an interval since last practice (``logitdec +
    recency``), and the two four-parameter spacing features, one of which
    (``base4``) additionally reads time on task.
    """
    chunk = CHUNKS[name]
    design = build_lkt_design(data, chunk.terms(), cost=REFERENCE_COST)
    assert design.n_params == chunk.n_params
    assert design.n_params == design.rank()

    fitted = fit_lkt(design, data.y, **TIGHT)
    gap = fitted.ll_unpenalized - chunk.log_likelihood
    assert fitted.is_optimal, "our own fit must be certified before we judge theirs"
    assert gap > -NAT_TOLERANCE, (
        f"we are {-gap:.4f} nats *worse* than the reference, which a certified "
        "optimum cannot be on a convex objective — the design must differ"
    )
    assert abs(gap) < NAT_TOLERANCE, (
        f"log-likelihood differs by {gap:.4f} nats "
        f"(ours {fitted.ll_unpenalized:.8f}, published {chunk.log_likelihood:.8f})"
    )
    mcfadden = 1.0 - fitted.ll_unpenalized / _null_ll(data.y)
    assert mcfadden == pytest.approx(chunk.mcfadden, abs=5e-5)


def test_every_chunk_lands_on_the_better_side_of_its_published_value(data):
    """Not one comparison but the shape of all four.

    A sign that flipped between chunks would say the agreement is noise around
    a wrong design. All four sitting *above* their published value, by a
    fraction of a nat, is the signature of one optimizer stopping early.
    """
    gaps = {}
    for name, chunk in CHUNKS.items():
        design = build_lkt_design(data, chunk.terms(), cost=REFERENCE_COST)
        fitted = fit_lkt(design, data.y, **TIGHT)
        gaps[name] = fitted.ll_unpenalized - chunk.log_likelihood
    assert all(g > 0 for g in gaps.values()), gaps
    assert max(gaps.values()) < NAT_TOLERANCE, gaps


def test_the_gap_is_the_solver_not_the_parameterization(data, fit):
    """Decompose the half-nat, so the explanation is checked and not asserted.

    Three things could account for a difference: the ridge ``cost`` puts on
    every column, which reference level the redundancy is broken at (under a
    ridge that is not a reparameterization — it moves the objective), and the
    reference's own stopping rule. The first two are measured here and are an
    order of magnitude too small; what is left is ``LiblineaR`` at
    ``epsilon = 1e-4``.
    """
    _, penalized = fit
    terms = CHUNKS["AFM"].terms()

    unpenalized = fit_lkt(build_lkt_design(data, terms), data.y,
                          warn_separated=False, **TIGHT)
    ridge_effect = abs(penalized.ll_unpenalized - unpenalized.ll_unpenalized)

    # R's contrasts drop the first student level; identification drops the last.
    full = build_lkt_design(data, terms, cost=REFERENCE_COST, identify=False)
    student = next(b for b in full.blocks if b.name == "student")
    keep = np.ones(student.matrix.shape[1], dtype=bool)
    keep[0] = False
    other_reference = fit_lkt(
        Design(tuple(b.keep(keep) if b.name == "student" else b for b in full.blocks)),
        data.y, **TIGHT)
    reference_effect = abs(other_reference.ll_unpenalized - penalized.ll_unpenalized)

    gap = abs(penalized.ll_unpenalized - CHUNKS["AFM"].log_likelihood)
    assert ridge_effect < 0.1 * gap, f"ridge moves {ridge_effect:.4f} of {gap:.4f} nats"
    assert reference_effect < 0.1 * gap, (
        f"reference level moves {reference_effect:.4f} of {gap:.4f} nats")
    assert other_reference.is_optimal and unpenalized.is_optimal


def test_prediction_clipping_does_not_explain_the_gap(data, fit):
    """The reference clamps fitted probabilities to ``[1e-5, 1-1e-5]`` before
    summing (``LKTfunctions.R:447-453``), which can only lower its reported
    likelihood. On this fit it costs less than a ten-thousandth of a nat, so it
    is not where the half-nat went."""
    design, fitted = fit
    p = fitted.predict_proba(design)
    clipped = np.clip(p, 1e-5, 1 - 1e-5)
    ll_clipped = float(np.sum(np.where(data.y == 1, np.log(clipped), np.log1p(-clipped))))
    assert abs(fitted.ll_unpenalized - ll_clipped) < 1e-4


def test_the_cost_parameter_is_the_references_ridge(data, fit):
    design, fitted = fit
    np.testing.assert_allclose(design.l2, 1.0 / REFERENCE_COST)
    assert fitted.penalty > 0.0
    # The reference reports the *unpenalized* likelihood of a penalized fit —
    # the opposite convention to LearnSphere's AFM, which reports the penalized
    # objective as if it were a likelihood. Both are available here.
    assert fitted.ll == pytest.approx(fitted.ll_unpenalized - fitted.penalty)
    assert fitted.ll_unpenalized > fitted.ll
