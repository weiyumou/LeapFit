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

#: From the AFM chunk of ``inst/doc/Examples.html`` in ``LKT_1.7.0.tar.gz``::
#:
#:     modelob <- LKT(data = val, interc = FALSE,
#:                    components = c("Anon.Student.Id","KC..Default.","KC..Default."),
#:                    features   = c("intercept", "intercept", "lineafm"))
#:     #> McFadden's R2 logistic: 0.280024
#:     #> LogLike logistic: -27347.20717315
#:
#: ``lineafm`` carries no ``$``, so the slope is one shared coefficient rather
#: than one per KC — the vignette's chunk is labelled AFM but fits AFM's pooled
#: variant. ``KC..Default.`` has been overwritten with the problem name.
PUBLISHED_LL = -27347.20717315
PUBLISHED_MCFADDEN = 0.280024

#: What the vignette's own preparation leaves: 58,316 raw rows less 3,194 STUDY
#: events. If the fixture disagrees, it was not built the same way and no
#: comparison below means anything.
N_OBS = 55_122
N_STUDENTS = 478
N_KCS = 72

#: Enough for our side to reach a certified optimum on 55k rows.
TIGHT = {"method": "L-BFGS-B", "max_fun": 500_000, "tol": 1e-14}

#: The published fit is short of its own optimum by ~0.48 nats. The band is
#: wide enough not to fail on a solver detail and tight enough that a real
#: divergence — a mis-specified term, a count off by one — could not hide in it:
#: one misplaced opportunity count moves the likelihood by far more.
NAT_TOLERANCE = 2.0

AFM_SPEC = (("student", "kc", "kc"), ("intercept", "intercept", "lineafm"))

pytestmark = pytest.mark.skipif(
    not os.path.exists(EXPORT),
    reason=f"LKT vignette input not found at {EXPORT!r}; see this module's docstring",
)


@pytest.fixture(scope="module")
def data():
    return load_student_step(EXPORT, kc_model="Default")


@pytest.fixture(scope="module")
def fit(data):
    design = build_lkt_design(data, lkt_terms(*AFM_SPEC), cost=REFERENCE_COST)
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


def test_the_published_log_likelihood_is_reproduced(data, fit):
    _, fitted = fit
    gap = fitted.ll_unpenalized - PUBLISHED_LL
    assert fitted.is_optimal, "our own fit must be certified before we judge theirs"
    assert gap > -NAT_TOLERANCE, (
        f"we are {-gap:.4f} nats *worse* than the reference, which a certified "
        "optimum cannot be on a convex objective — the design must differ"
    )
    assert abs(gap) < NAT_TOLERANCE, (
        f"log-likelihood differs by {gap:.4f} nats "
        f"(ours {fitted.ll_unpenalized:.8f}, published {PUBLISHED_LL:.8f})"
    )


def test_the_published_mcfadden_r2_is_reproduced(data, fit):
    """The reference's headline statistic, against an intercept-only null."""
    _, fitted = fit
    mcfadden = 1.0 - fitted.ll_unpenalized / _null_ll(data.y)
    assert mcfadden == pytest.approx(PUBLISHED_MCFADDEN, abs=5e-5)


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
    terms = lkt_terms(*AFM_SPEC)

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

    gap = abs(penalized.ll_unpenalized - PUBLISHED_LL)
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
