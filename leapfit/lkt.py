"""Logistic Knowledge Tracing: a term algebra over components and features.

LKT (Pavlik, Eglington & Harrel-Williams 2021) generalizes the families in this
package rather than sitting beside them. A model is a list of **terms**, each
pairing a *component* — a factor whose levels partition the data, such as the
student, the item, or the KC — with a *feature*, a function of that component's
practice history for that student::

    logit P(Y_ij = 1) = sum over terms of  sum_l q_jl * coef * feature(history_il)
                                           \\____/          \\__________________/
                                       component level        what is counted

A feature is fitted with one shared coefficient, or — the reference's ``$``
suffix — with one coefficient per level of its component. In that notation
this package's two other families are single LKT specifications::

    AFM  components (student, kc, kc)  features (intercept, intercept, lineafm$)
    PFA  components (kc, kc, kc)       features (intercept, linesuc$, linefail$)

and :func:`build_lkt_design` produces designs identical to
:func:`leapfit.afm.build_afm_design` (with ``recompute_opportunities=True``,
because LKT derives its counts from the practice order rather than reading
DataShop's ``Opportunity`` column) and to
:func:`leapfit.pfa.build_pfa_design`. Both identities are pinned by tests.

**Provenance, and how it differs from the rest of this package.** The reference
is the CRAN package
`LKT <https://CRAN.R-project.org/package=LKT>`_ 1.7.0 (Pavlik & Eglington,
2024-07-01), one file, ``R/LKTfunctions.R``. It is **GPL-3**; leapfit is MIT.
So unlike :mod:`leapfit.afm` and :mod:`leapfit.pfa`, which are *adapted from*
their reference, this module is written from the published equations and the
reference's documented feature semantics and then **validated against** its
output. Nothing here is a translation of its source, and the distinction is
load-bearing rather than decorative.

**Features.** Everything the reference computes from prior counts, from decayed
outcome histories, or from the clock, at parameters the caller **fixes**:

===============  ======  ========================================================
family           pars    features
===============  ======  ========================================================
counts           0       ``intercept`` ``lineafm`` ``logafm`` ``linesuc``
                         ``logsuc`` ``linefail`` ``logfail`` ``linecomp`` ``prop``
counts, shaped   1       ``powafm`` ``logit``
decayed history  1       ``expdecafm`` ``expdecsuc`` ``expdecfail`` ``propdec``
                         ``propdec2`` ``logitdec``
recency          1       ``recency`` ``recencysuc`` ``recencyfail``
forgetting       1       ``base`` ``basesuc`` ``basefail`` ``dashafm`` ``dashsuc``
forgetting, 2    2       ``base2`` ``base2suc`` ``base2fail``
spacing          4       ``base4`` ``ppe``
covariate        0       ``numer``
===============  ======  ========================================================

**Nothing here searches for a parameter.** The reference fits its decay rates
with an outer ``optim`` that rebuilds every feature and refits the whole
regression at each evaluation; that is a different fitter and it is not in this
module. Every parametric feature therefore *requires* ``pars=``, and a fit at
fixed parameters is what it says it is. Features the reference computes that
are not here are refused **by name with the reason** — see :data:`DEFERRED` —
because a spec silently missing a term is worse than one that will not build.

Also deferred: a global intercept (``interc=TRUE``), the ``*`` and ``:``
connectors, ``interacts``, ``autoKC`` clustering, and ``@`` random effects.

**The reference's penalty is expressible here exactly.** LKT solves with
``LiblineaR(type = 0, cost = 512)``, which minimizes ``0.5 w'w + C * sum nll``;
dividing by ``C`` gives this package's objective with ``l2 = 1/C`` on **every**
column. Pass ``cost=512`` (:data:`REFERENCE_COST`) to reproduce it. Worth
knowing what that ridge is doing: with a per-level intercept on both the
student and the KC the design is rank-deficient, and ``cost`` is what keeps the
fit from sliding along that flat direction — the same identification-by-penalty
that :meth:`leapfit.design.Design.identify` replaces with an exact drop. The
default here is ``l2 = 0`` and an identified design, as everywhere else.

DIVERGENCE (block names): two block names are fixed by the shared layer rather
than by this module's scheme. ``Design.identify`` *detects* sum redundancies
from the row sums, so any number of intercept components identifies correctly;
but it takes its reference level from the block named ``prefer_drop``
(``"student"``), and ``Design.recentring_is_valid`` looks for ``kc_intercept``.
A per-level intercept on those two components therefore uses those names, and
every other term is named ``feature[component]`` — which also makes an LKT AFM
spec produce a design identical to :func:`~leapfit.afm.build_afm_design`'s,
block names included.

DIVERGENCE (fit statistics): the reference reports the unpenalized
log-likelihood of predictions clipped to ``[1e-5, 1-1e-5]``, and reports no
parameter count, no AIC and no BIC. Here the likelihood is unclipped and
``n_params`` is the rank of the design, so AIC and BIC exist and mean what they
mean for every other family in this package.

DIVERGENCE (non-finite features are refused): several of the reference's
time-based features divide by an elapsed time. Where two attempts on one level
carry the same timestamp that elapsed time is zero, and the reference's
``baselevel`` raises it to a negative power and produces ``Inf`` — which
``LiblineaR`` will happily consume. :func:`build_lkt_design` checks every
assembled column and raises instead, naming the feature and the rows.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from leapfit.data import StepData
from leapfit.design import Aliased, Block, Design
from leapfit.fit import DEFAULT_METHOD, LogisticFit, fit_logistic

#: ``LiblineaR``'s cost in the reference's own default call, as ``l2 = 1/cost``.
REFERENCE_COST = 512.0

#: Components this module resolves itself. Anything else is looked up as a
#: column of the source export (see :func:`component_labels`). ``"kc"`` is the
#: only one that can carry several levels on one row.
COMPONENTS = ("student", "item", "kc")

#: ``slidelogitdec`` looks back over ``x[max(1, i - 60):i]`` — at most 61
#: trials. Undocumented in the paper and load-bearing at slow decay rates
#: (``d = .99`` still weights the 61st trial back at 0.54), so it is named here
#: rather than buried in the implementation.
LOGITDEC_WINDOW = 61

#: ``dash``'s decay scale is expressed in days.
_DAY = 86_400.0

#: The component's own column read as a number rather than as a factor.
#: Handled apart from the registry because it consults no history at all.
NUMERIC_FEATURE = "numer"


class _View:
    """One (student, component level) practice sequence, in order.

    Everything a feature is allowed to see about one level of one student:
    the outcomes, the prior counts, and — where the export supports it — the
    clock. Positions are practice positions, so index 0 is that student's
    first encounter with that level.
    """

    __slots__ = ("f", "on_task", "s", "time", "y")

    def __init__(self, y, s, f, time, on_task):
        self.y, self.s, self.f = y, s, f
        self.time, self.on_task = time, on_task

    def __len__(self) -> int:
        return len(self.y)


# --------------------------------------------------------------------------
# Sequence primitives. Each takes one practice sequence and returns one value
# per position, and each is written from the reference function it is named
# after rather than translated from it.
# --------------------------------------------------------------------------


def _lag(values: np.ndarray, seed: float = 0.0) -> np.ndarray:
    """``c(seed, head(x, -1))`` — the value from the previous position."""
    out = np.empty_like(values, dtype=float)
    out[0] = seed
    out[1:] = values[:-1]
    return out


def _slide_expdec(x: np.ndarray, d: float) -> np.ndarray:
    """``slideexpdec``: position ``i`` decays ``x[:i]``, most recent weighted 1.

    Zero at the first position — the reference lags the running total by one,
    so nothing here sees its own trial.
    """
    out = np.zeros(len(x), dtype=float)
    acc = 0.0
    for i in range(1, len(x)):
        acc = d * acc + x[i - 1]
        out[i] = acc
    return out


def _decayed_counts(y: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """The reference's ``corv``/``incorv``: decayed successes and failures over
    ``y[:i]``, each seeded with one ghost trial.

    ``propdec`` and ``logitdec`` are two readings of the same pair. The ghosts
    are what make position 0 well defined — a proportion of 0.5 and a logit of
    0 — rather than 0/0.
    """
    n = len(y)
    corv = np.ones(n, dtype=float)
    incorv = np.ones(n, dtype=float)
    c = i_ = 1.0
    for k in range(1, n):
        c = d * c + y[k - 1]
        i_ = d * i_ + (1.0 - y[k - 1])
        corv[k], incorv[k] = c, i_
    return corv, incorv


def _windowed_decayed_counts(y: np.ndarray, d: float,
                             window: int) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_decayed_counts` over a trailing window of at most ``window``.

    The recurrence cannot express a window, so this is the direct sum. It is
    only reached on sequences longer than the window: below that the window
    never truncates and the recurrence is exact.
    """
    n = len(y)
    corv = np.ones(n, dtype=float)
    incorv = np.ones(n, dtype=float)
    for k in range(1, n):
        a = max(0, k - window)
        chunk = y[a:k]
        w = len(chunk)
        weights = d ** np.arange(w - 1, -1, -1.0)
        ghost = d ** w
        corv[k] = ghost + float(chunk @ weights)
        incorv[k] = ghost + float((1.0 - chunk) @ weights)
    return corv, incorv


def _baselevel(age: np.ndarray, d: float) -> np.ndarray:
    """``baselevel``: age since the level's first encounter, to the power ``-d``.

    Zero at the first position, where the age is zero and the power undefined.
    """
    out = np.zeros(len(age), dtype=float)
    if len(age) > 1:
        with np.errstate(divide="ignore"):
            out[1:] = np.asarray(age[1:], dtype=float) ** -d
    return out


def _spacing(time: np.ndarray) -> np.ndarray:
    """``componentspacing``: elapsed time since this level's previous encounter."""
    out = np.zeros(len(time), dtype=float)
    out[1:] = np.diff(np.asarray(time, dtype=float))
    return out


def _mean_spacing(spacing: np.ndarray) -> np.ndarray:
    """``meanspacingf``: the running mean of prior spacings, with a sentinel.

    Position 0 is 0 and position 1 is **-1**, both of which the features that
    read this treat as "no spacing to speak of yet" — the sentinel is the
    reference's, and ``base4`` branches on it rather than on the position.
    From position 2 on it is the mean of the spacings at positions 1..i-1;
    position 0's spacing is structurally zero and is excluded.
    """
    n = len(spacing)
    out = np.zeros(n, dtype=float)
    if n > 1:
        out[1] = -1.0
    if n > 2:
        out[2:] = np.cumsum(spacing[1:n - 1]) / np.arange(1, n - 1)
    return out


def _dash(time: np.ndarray, increments: np.ndarray, scale: float) -> np.ndarray:
    """``countOutcomeDash``: a count of prior trials that decays with real time.

    ``scale`` is in days. ``increments`` is 1 per trial for ``dashafm`` and the
    outcome for ``dashsuc``.
    """
    n = len(time)
    out = np.zeros(n, dtype=float)
    carried = float(increments[0])
    for i in range(1, n):
        out[i] = carried * math.exp(-(time[i] - time[i - 1]) / (scale * _DAY))
        carried = out[i] + float(increments[i])
    return out


def _ppe_weighted_time(age: np.ndarray, d: float) -> np.ndarray:
    """``slideppetw``: PPE's recency-weighted mean time since prior practice.

    At each position the times back to every earlier practice are weighted by
    themselves to the power ``-d`` — recent practice counts for more — and
    averaged. The first position, and any position where some earlier practice
    carries the same timestamp, returns 1: the weights are then ``Inf/Inf`` and
    the reference's own ``is.nan`` guard falls through to 1.
    """
    n = len(age)
    out = np.ones(n, dtype=float)
    age = np.asarray(age, dtype=float)
    for i in range(1, n):
        elapsed = age[i] - age[:i]
        if not np.all(elapsed > 0.0):
            continue
        weights = elapsed ** -d
        total = weights.sum()
        if not np.isfinite(total) or total == 0.0:
            continue
        out[i] = float((weights / total) @ elapsed)
    return out


# --------------------------------------------------------------------------
# The feature registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Feature:
    """How one feature is computed, and what it needs to be computable.

    ``flat`` features are pure functions of the prior counts and evaluate over
    every observation at once. ``walk`` features read a whole practice sequence
    — a decay, a spacing, an age — and are evaluated one sequence at a time.
    """

    arity: int
    flat: Callable[[np.ndarray, np.ndarray, tuple[float, ...]], np.ndarray] | None = None
    walk: Callable[[_View, tuple[float, ...]], np.ndarray] | None = None
    needs: tuple[str, ...] = ()


def _prop(s, f, _):
    n = s + f
    return np.where(n == 0, 0.5, s / np.where(n == 0, 1.0, n))


def _propdec(view, pars):
    corv, incorv = _decayed_counts(view.y, pars[0])
    return corv / (corv + incorv)


def _propdec2(view, pars):
    d = pars[0]
    n = len(view.y)
    running = _slide_expdec(view.y, d)
    w = np.arange(n, dtype=float)
    # sum(d^0 .. d^(w+2)), the reference's three ghost failures.
    denom = (np.full(n, w + 3.0) if d == 1.0
             else (1.0 - d ** (w + 3.0)) / (1.0 - d))
    out = running / denom
    out[0] = 0.0
    return out


def _logitdec(view, pars):
    d = pars[0]
    counts = (_decayed_counts if len(view.y) <= LOGITDEC_WINDOW + 1
              else lambda y, dd: _windowed_decayed_counts(y, dd, LOGITDEC_WINDOW))
    corv, incorv = counts(view.y, d)
    return np.log(corv / incorv)


def _recency(view, pars, weight=None):
    spacing = _spacing(view.time)
    with np.errstate(divide="ignore"):
        value = np.where(spacing == 0.0, 0.0, spacing ** -pars[0])
    return value if weight is None else value * weight(view)


def _age(view, pars) -> np.ndarray:
    """``base``'s clock: real time since this level's first encounter."""
    time = np.asarray(view.time, dtype=float)
    return time - time[0]


def _blended_age(view, pars) -> np.ndarray:
    """``base2``'s clock: time away from the system counted at ``pars[1]``.

    ``(real age - age on task) * w + age on task`` — with ``w = 1`` this is
    real time and with ``w = 0`` it is time spent working, so the parameter
    says how much a gap between sessions counts towards forgetting.
    """
    on_task = np.asarray(view.on_task, dtype=float)
    intage = on_task - on_task[0]
    return (_age(view, pars) - intage) * pars[1] + intage


def _base(clock, amount):
    def feature(view, pars):
        return amount(view) * _baselevel(clock(view, pars), pars[0])
    return feature


def _base4(view, pars):
    decay, session, spacing_power, unspaced = pars
    level = _baselevel(_blended_age(view, pars), decay)
    practice = np.log1p(view.s + view.f)
    mean_real = _mean_spacing(_spacing(view.time))
    mean_task = _mean_spacing(_spacing(view.on_task))
    blended = session * (mean_real - mean_task) + mean_task
    unspaced_here = mean_real <= 0.0
    # Positions on the sentinel branch never reach the power, so a NaN out of
    # it is a real negative mean spacing rather than the sentinel's own -1.
    spaced = np.where(unspaced_here, 1.0, blended) ** spacing_power
    return np.where(unspaced_here, unspaced, spaced) * practice * level


def _ppe(view, pars):
    count_power, base_decay, spacing_decay, weight_decay = pars
    n = view.s + view.f
    lagged = _lag(_spacing(view.time))
    with np.errstate(divide="ignore"):
        contribution = np.where(lagged == 0.0, 0.0, 1.0 / np.log(lagged + math.e))
    spacing = np.cumsum(contribution)
    spacing = np.where(n <= 1, 0.0, spacing / np.where(n <= 1, 1.0, n - 1))
    weighted = _ppe_weighted_time(_age(view, pars), weight_decay)
    return n.astype(float) ** count_power * weighted ** -(base_decay + spacing_decay * spacing)


#: Every implemented feature, keyed by the reference's own name.
_FEATURES: dict[str, _Feature] = {
    # Pure functions of the prior counts.
    "intercept": _Feature(0, flat=lambda s, f, p: np.ones(len(s))),
    "lineafm": _Feature(0, flat=lambda s, f, p: (s + f).astype(float)),
    "logafm": _Feature(0, flat=lambda s, f, p: np.log1p(s + f)),
    "linesuc": _Feature(0, flat=lambda s, f, p: s.astype(float)),
    "logsuc": _Feature(0, flat=lambda s, f, p: np.log1p(s)),
    "linefail": _Feature(0, flat=lambda s, f, p: f.astype(float)),
    "logfail": _Feature(0, flat=lambda s, f, p: np.log1p(f)),
    "linecomp": _Feature(0, flat=lambda s, f, p: (s - f).astype(float)),
    "prop": _Feature(0, flat=_prop),
    "powafm": _Feature(1, flat=lambda s, f, p: (s + f).astype(float) ** p[0]),
    "logit": _Feature(1, flat=lambda s, f, p: np.log(
        (0.1 + 30.0 * p[0] + s) / (0.1 + 30.0 * p[0] + f))),

    # Decayed outcome histories — no clock, just practice order.
    "expdecafm": _Feature(1, walk=lambda v, p: _slide_expdec(np.ones(len(v)), p[0])),
    "expdecsuc": _Feature(1, walk=lambda v, p: _slide_expdec(v.y, p[0])),
    "expdecfail": _Feature(1, walk=lambda v, p: _slide_expdec(1.0 - v.y, p[0])),
    "propdec": _Feature(1, walk=_propdec),
    "propdec2": _Feature(1, walk=_propdec2),
    "logitdec": _Feature(1, walk=_logitdec),

    # Recency: the interval since the previous encounter, and nothing older.
    "recency": _Feature(1, walk=_recency, needs=("time",)),
    "recencysuc": _Feature(
        1, walk=lambda v, p: _recency(v, p, weight=lambda v: _lag(v.y)), needs=("time",)),
    "recencyfail": _Feature(
        1, walk=lambda v, p: _recency(v, p, weight=lambda v: 1.0 - _lag(v.y)),
        needs=("time",)),

    # Forgetting: practice scaled by a power-law decay of its age.
    "base": _Feature(1, walk=_base(_age, lambda v: np.log1p(v.s + v.f)), needs=("time",)),
    "basesuc": _Feature(1, walk=_base(_age, lambda v: np.log1p(v.s)), needs=("time",)),
    "basefail": _Feature(1, walk=_base(_age, lambda v: np.log1p(v.f)), needs=("time",)),
    "base2": _Feature(2, walk=_base(_blended_age, lambda v: np.log1p(v.s + v.f)),
                      needs=("time", "on_task")),
    "base2suc": _Feature(2, walk=_base(_blended_age, lambda v: np.log1p(v.s)),
                         needs=("time", "on_task")),
    "base2fail": _Feature(2, walk=_base(_blended_age, lambda v: np.log1p(v.f)),
                          needs=("time", "on_task")),
    "dashafm": _Feature(1, needs=("time",), walk=lambda v, p: np.log1p(
        _dash(v.time, np.ones(len(v)), p[0]))),
    "dashsuc": _Feature(1, needs=("time",), walk=lambda v, p: np.log1p(
        _dash(v.time, v.y, p[0]))),

    # Spacing: forgetting scaled by how spread out the practice was.
    "base4": _Feature(4, walk=_base4, needs=("time", "on_task")),
    "ppe": _Feature(4, walk=_ppe, needs=("time",)),
}

#: Every feature name this module implements, including :data:`NUMERIC_FEATURE`.
FEATURE_NAMES = (*sorted(_FEATURES), NUMERIC_FEATURE)

#: Reference features this module refuses, and why. Named individually so a
#: spec copied out of a paper fails with the reason rather than with
#: "unknown feature". Three of these are refusals to reproduce a defect: the
#: reference cannot compute them either.
DEFERRED: dict[str, str] = {
    "errordec": ("the reference reads data$pred_ed (LKTfunctions.R:891), which nothing "
                 "in the package ever assigns"),
    "recencystudy": ("the reference reads <component>previousstudy, whose assignment is "
                     "commented out in computeSpacingPredictors (LKTfunctions.R:25)"),
    "recencytest": ("the reference reads <component>previousstudy, whose assignment is "
                    "commented out in computeSpacingPredictors (LKTfunctions.R:25)"),
    "dashfail": ("the reference counts it in parlength (LKTfunctions.R:543) but has no "
                 "branch for it in computefeatures"),
    "logitdecevol": ("indexes by component level across students, a grouping this module "
                     "does not build"),
    "baseratepropdec": ("indexes by student over component labels, a grouping this module "
                        "does not build"),
    "base5suc": "not implemented; base4 with a fifth parameter",
    "base5fail": "not implemented; base4 with a fifth parameter",
    **dict.fromkeys(
        ("diffcor1", "diffcor2", "diffincor1", "diffincor2", "diffall1", "diffall2",
         "diffcorComp", "diffincorComp", "diffallComp", "diffrelcor1", "diffrelcor2"),
        "needs predictions from a prior fit as an input column",
    ),
}

_BLOCK_NAME = re.compile(r"^(?P<feature>[A-Za-z][A-Za-z0-9]*)(?P<per_level>\$?)"
                         r"(?:\((?P<pars>[^)]*)\))?\[(?P<component>.+)]$")

#: Per-level intercepts on these two components are named the way the shared
#: identification pass expects; see the module docstring's block-name divergence.
_SHARED_BLOCK_NAME = {"student": "student", "kc": "kc_intercept"}

#: Component names the reference uses, mapped onto this package's. Only the
#: student is fixed by LKT's own contract; a KC model is chosen when the export
#: is loaded, so ``KC..Default.`` has no general translation and stays a
#: source-column lookup.
_ALIASES = {"Anon.Student.Id": "student", "Anon Student Id": "student"}


@dataclass(frozen=True)
class Term:
    """One component paired with one feature — a group of design columns.

    ``per_level`` is the reference's ``$``: one coefficient per level of the
    component instead of one shared across them. Intercepts are always
    per-level (a factor is expanded whether or not it is written with a ``$``),
    so the flag is forced on for them, as in the reference.

    ``pars`` are the feature's shape parameters, **fixed** — nothing in this
    module fits one. A scalar is accepted for the single-parameter features.
    They are part of the term's identity and of its block name, so two
    ``powafm`` terms on one component at different exponents do not collide.
    """

    component: str
    feature: str
    per_level: bool = False
    pars: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.feature in DEFERRED:
            raise NotImplementedError(
                f"Feature {self.feature!r} is not implemented here: "
                f"{DEFERRED[self.feature]}. Implemented features are "
                f"{', '.join(FEATURE_NAMES)}."
            )
        if self.feature not in _FEATURES and self.feature != NUMERIC_FEATURE:
            raise ValueError(
                f"Unknown feature {self.feature!r}. Implemented: "
                f"{', '.join(FEATURE_NAMES)}."
            )
        if self.feature == NUMERIC_FEATURE and self.per_level:
            raise ValueError(
                "'numer' reads its component as a number, not as a factor, so "
                "'numer$' has no levels to extend over. Drop the '$'."
            )
        if not isinstance(self.pars, tuple):
            scalar = (float(self.pars),) if np.isscalar(self.pars) else tuple(self.pars)
            object.__setattr__(self, "pars", scalar)

        arity = self.arity
        if len(self.pars) != arity:
            fits = ("takes no parameter" if arity == 0
                    else f"takes {arity} fixed parameter{'s' if arity > 1 else ''}")
            raise ValueError(
                f"Feature {self.feature!r} {fits}, got {len(self.pars)}. Nothing in "
                "this module searches for a parameter — the reference fits its decay "
                "rates with an outer optimizer that is not implemented here — so a "
                "parametric feature has to be held at a value you choose."
            )
        # Frozen, but the reference's own normalization: a factor is expanded
        # whether or not the spec writes the '$'.
        if self.feature == "intercept" and not self.per_level:
            object.__setattr__(self, "per_level", True)

    @property
    def arity(self) -> int:
        """How many fixed parameters this term's feature takes."""
        spec = _FEATURES.get(self.feature)
        return 0 if spec is None else spec.arity

    @classmethod
    def parse(cls, component: str, feature: str,
              pars: float | Sequence[float] | None = None) -> Term:
        """Build from the reference's own strings, ``$`` suffix included."""
        text = feature.strip()
        per_level = text.endswith("$")
        if text.endswith("@"):
            raise NotImplementedError(
                f"{feature!r}: '@' asks for a random effect, which the reference fits "
                "with lme4::glmer. This package fits fixed effects throughout — a "
                "mixed-effects fit is a different estimator whose AIC/BIC are not "
                "comparable with these."
            )
        if pars is None:
            pars = ()
        elif np.isscalar(pars):
            pars = (float(pars),)
        return cls(component=_ALIASES.get(component, component),
                   feature=text.rstrip("$"), per_level=per_level, pars=tuple(pars))

    def notation(self) -> str:
        """The feature as the reference writes it: ``lineafm$``.

        An intercept is written without the ``$`` even though it is per-level,
        because that is how the reference writes it — the expansion is implicit
        for a factor. :meth:`parse` reads it back to the same term.
        """
        marker = "$" if self.per_level and self.feature != "intercept" else ""
        return f"{self.feature}{marker}"

    def _pars_suffix(self) -> str:
        return "" if not self.pars else "(" + ",".join(f"{p:g}" for p in self.pars) + ")"

    def __str__(self) -> str:
        return f"{self.component}:{self.notation()}{self._pars_suffix()}"

    @property
    def block_name(self) -> str:
        """Design block this term contributes — see the block-name divergence."""
        if self.feature == "intercept" and self.component in _SHARED_BLOCK_NAME:
            return _SHARED_BLOCK_NAME[self.component]
        return f"{self.notation()}{self._pars_suffix()}[{self.component}]"


def lkt_terms(components: Sequence[str], features: Sequence[str],
              pars: Sequence[float | Sequence[float] | None] | None = None,
              ) -> tuple[Term, ...]:
    """Terms from the reference's parallel ``components``/``features`` vectors.

    Exists so a specification can be copied out of a paper unchanged::

        lkt_terms(components=("student", "kc", "kc"),
                  features=("intercept", "intercept", "lineafm$"))

    ``pars`` parallels the same vectors: one entry per term, holding that
    feature's fixed parameters (a scalar for the single-parameter features, a
    tuple for the rest, ``None`` where the feature takes none). Deliberately
    *per term* rather than the reference's single flat vector consumed in
    feature order — the flat form is what makes its own parameter bookkeeping
    hard to read, and hard enough that two of its branches
    (``LKTfunctions.R:155-156``, ``:163-218``) silently stop recomputing the
    features that share a parameter.
    """
    if len(components) != len(features):
        raise ValueError(
            f"{len(components)} component(s) but {len(features)} feature(s); the two "
            "vectors are positional and must be the same length"
        )
    if pars is None:
        pars = [None] * len(features)
    elif len(pars) != len(features):
        raise ValueError(f"{len(pars)} par entr(ies) for {len(features)} feature(s)")
    return tuple(Term.parse(c, f, p) for c, f, p in zip(components, features, pars))


def design_terms(design: Design) -> tuple[Term, ...]:
    """Recover the terms of an LKT design from its block names.

    The spec is stored in the design rather than beside it, so a fit knows what
    it fitted without carrying extra state and without being told twice.
    """
    out = []
    for block in design.blocks:
        if block.name == "student":
            out.append(Term("student", "intercept"))
        elif block.name == "kc_intercept":
            out.append(Term("kc", "intercept"))
        elif m := _BLOCK_NAME.match(block.name):
            pars = () if m["pars"] is None else tuple(float(p) for p in m["pars"].split(","))
            out.append(Term(component=m["component"], feature=m["feature"],
                            per_level=bool(m["per_level"]), pars=pars))
        else:
            raise ValueError(f"Block {block.name!r} was not built by leapfit.lkt")
    return tuple(out)


# --------------------------------------------------------------------------
# Components and their practice histories
# --------------------------------------------------------------------------


def component_labels(data: StepData, component: str) -> list[tuple[str, ...]]:
    """Each observation's levels of one component, as
    :attr:`~leapfit.data.StepData.kcs` is shaped.

    ``"student"``, ``"item"`` and ``"kc"`` are resolved from the parsed data;
    anything else is read from the export the data came from, which
    :attr:`~leapfit.data.StepData.source` retains together with each
    observation's position in it. ``"kc"`` is the only component that can put
    several levels on one row.
    """
    if component == "student":
        return [(s,) for s in data.students]
    if component == "item":
        return [(i,) for i in data.items]
    if component == "kc":
        return list(data.kcs)
    return [(v,) for v in _source_column(data, component).astype(str)]


def _source_column(data: StepData, component: str) -> np.ndarray:
    if data.source is None or data.source_rows is None:
        raise ValueError(
            f"Component {component!r} is not one of {COMPONENTS} and this StepData "
            "carries no source table to look it up in. Load with load_student_step "
            "or from_frame, which retain it."
        )
    if component not in data.source.columns:
        available = ", ".join(map(repr, data.source.columns))
        raise KeyError(
            f"Component {component!r} is neither one of {COMPONENTS} nor a column of "
            f"the export. Available columns: {available}"
        )
    return data.source[component].to_numpy()[data.source_rows]


@dataclass(frozen=True)
class _Layout:
    """Every (observation, component level) pair, grouped into practice sequences.

    The pairs are stored flat and ordered so that each (student, level) history
    is one contiguous run — ``starts[k]:starts[k+1]``, in practice order. That
    is the shape both kinds of feature want: a count feature reads
    :attr:`prior_s` and :attr:`prior_f` straight across, and a decay or
    forgetting feature walks one run at a time.

    ``slots`` records which of an observation's own labels a pair came from, so
    a per-observation view can be rebuilt in the row's label order — the order
    everything outside this module aligns to.
    """

    n_obs: int
    rows: np.ndarray          # (m,) observation index
    columns: np.ndarray       # (m,) level index into `levels`
    slots: np.ndarray         # (m,) label position within that observation
    starts: np.ndarray        # (n_sequences + 1,)
    levels: list[str]
    y: np.ndarray             # (m,) outcome at each pair
    prior_s: np.ndarray       # (m,) prior successes within its own sequence
    prior_f: np.ndarray       # (m,)


def _layout(data: StepData, labels: Sequence[tuple[str, ...]]) -> _Layout:
    """Group every (observation, level) pair by (student, level), in practice order.

    The reference forms the same grouping by pasting the level onto the student
    id and grouping on the concatenated string (``LKTfunctions.R:296``), which
    is why two (student, level) pairs whose names differ only in where the
    boundary falls collide there and not here.
    """
    levels = sorted({label for row in labels for label in row})
    index = {label: j for j, label in enumerate(levels)}

    sequences: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for student, rows in data.practice_order().items():
        for i in rows:
            for slot, label in enumerate(labels[i]):
                sequences.setdefault((student, label), []).append((i, slot))

    pairs = [pair for run in sequences.values() for pair in run]
    rows_flat = np.fromiter((i for i, _ in pairs), dtype=int, count=len(pairs))
    slots = np.fromiter((s for _, s in pairs), dtype=int, count=len(pairs))
    columns = np.fromiter(
        (index[label] for (_, label), run in sequences.items() for _ in run),
        dtype=int, count=len(pairs))
    starts = np.concatenate(
        [[0], np.cumsum([len(run) for run in sequences.values()], dtype=int)]).astype(int)

    y = np.asarray(data.y, dtype=float)[rows_flat]

    # Prior counts: an exclusive cumulative sum that restarts at every sequence.
    totals = np.concatenate([[0.0], np.cumsum(y)])
    at_start = np.repeat(starts[:-1], np.diff(starts))
    prior_s = totals[:-1] - totals[at_start]
    prior_f = (np.arange(len(y)) - at_start) - prior_s

    return _Layout(n_obs=len(data), rows=rows_flat, columns=columns, slots=slots,
                   starts=starts, levels=levels, y=y,
                   prior_s=prior_s, prior_f=prior_f)


def history_counts(data: StepData, labels: Sequence[tuple[str, ...]],
                   ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Prior successes and failures per observation, per level of a component.

    :func:`leapfit.pfa.success_failure_counts` with the KC hardcoding lifted:
    the reference accumulates within ``paste(component level, student)``
    (``LKTfunctions.R:981``), so a count is a student's own history with that
    level and nothing else. Strictly prior — the current attempt is not inside
    its own predictor — and accumulated over
    :meth:`~leapfit.data.StepData.practice_order`, the one ordering everything
    in this package agrees on.

    Passing ``data.kcs`` reproduces
    :func:`~leapfit.pfa.success_failure_counts` exactly; passing the student's
    own labels gives that student's whole prior history, which is what a
    feature on the student component means. Element ``n`` is aligned with
    ``labels[n]`` by position, as :attr:`~leapfit.data.StepData.opportunities`
    is aligned with :attr:`~leapfit.data.StepData.kcs`.
    """
    if len(labels) != len(data):
        raise ValueError(f"{len(data)} observations but {len(labels)} label rows")
    layout = _layout(data, labels)
    s_out: list[list[int]] = [[0] * len(row) for row in labels]
    f_out: list[list[int]] = [[0] * len(row) for row in labels]
    for i, slot, s, f in zip(layout.rows, layout.slots, layout.prior_s, layout.prior_f):
        s_out[i][slot] = int(s)
        f_out[i][slot] = int(f)
    return [tuple(v) for v in s_out], [tuple(v) for v in f_out]


# --------------------------------------------------------------------------
# Assembling a design
# --------------------------------------------------------------------------


class _Clock:
    """The time series a term may need, resolved once per design and shared."""

    def __init__(self, data: StepData):
        self._data = data
        self._cache: dict[str, np.ndarray] = {}

    def get(self, what: str, term: Term) -> np.ndarray:
        if what not in self._cache:
            source = {"time": self._data.epoch_times,
                      "on_task": self._data.time_on_task}[what]
            try:
                self._cache[what] = np.asarray(source(), dtype=float)
            except ValueError as exc:
                raise ValueError(f"Term {term} cannot be computed: {exc}") from exc
        return self._cache[what]


def _term_values(term: Term, data: StepData, layout: _Layout, clock: _Clock) -> np.ndarray:
    """One value per (observation, level) pair, in ``layout`` order."""
    if term.feature == NUMERIC_FEATURE:
        column = pd.to_numeric(pd.Series(_source_column(data, term.component)),
                               errors="coerce").to_numpy(dtype=float)
        if np.isnan(column).any():
            raise ValueError(
                f"'numer' reads component {term.component!r} as a number, but "
                f"{int(np.isnan(column).sum()):,} of its values do not parse as one."
            )
        return column[layout.rows]

    spec = _FEATURES[term.feature]
    if spec.flat is not None:
        return np.asarray(spec.flat(layout.prior_s, layout.prior_f, term.pars), dtype=float)

    time = clock.get("time", term)[layout.rows] if "time" in spec.needs else None
    on_task = clock.get("on_task", term)[layout.rows] if "on_task" in spec.needs else None

    out = np.empty(len(layout.rows), dtype=float)
    for a, b in zip(layout.starts[:-1], layout.starts[1:]):
        view = _View(layout.y[a:b], layout.prior_s[a:b], layout.prior_f[a:b],
                     None if time is None else time[a:b],
                     None if on_task is None else on_task[a:b])
        out[a:b] = spec.walk(view, term.pars)
    return out


def _term_block(term: Term, layout: _Layout, values: np.ndarray, l2: float) -> Block:
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"Term {term} produced {bad:,} non-finite value(s). The usual cause is two "
            "attempts on one level carrying the same timestamp, which makes an elapsed "
            "time zero and a negative power of it infinite. The reference produces Inf "
            "here and fits on it; check the export's times before choosing between "
            "dropping those rows and using a feature that does not divide by an interval."
        )

    if not term.per_level:
        # One shared coefficient. A row carrying several levels contributes the
        # sum over them, which is what an additive multi-KC step means here and
        # what leapfit.pfa's pooled slopes already do.
        totals = np.zeros(layout.n_obs, dtype=float)
        np.add.at(totals, layout.rows, values)
        return Block.build(term.block_name, totals[:, None], [term.notation()], l2=l2)

    matrix = sparse.csr_matrix((values, (layout.rows, layout.columns)),
                               shape=(layout.n_obs, len(layout.levels)))
    matrix.eliminate_zeros()  # a zero count is a structural zero, not a datum
    return Block.build(term.block_name, matrix, layout.levels, l2=l2)


def build_lkt_design(data: StepData, terms: Iterable[Term], *,
                     l2: float = 0.0, cost: float | None = None,
                     identify: bool = True) -> Design:
    """Assemble an LKT design from parsed student-step data.

    :param terms: the specification, in order. Blocks appear in this order, so
        writing ``(student:intercept, kc:intercept, kc:lineafm$)`` reproduces
        AFM's column layout as well as its columns.
    :param l2: ridge on every coefficient. Zero by default, as elsewhere in
        this package: a design that needs a penalty to be identified is one
        whose parameter count is wrong, and
        :meth:`~leapfit.design.Design.identify` fixes that exactly.
    :param cost: the reference's ``LiblineaR`` cost, as ``l2 = 1/cost``.
        Overrides ``l2``. ``cost=REFERENCE_COST`` is the reference's own
        default and reproduces the penalty its published fits carry.
    :param identify: drop aliased columns, so ``n_params == rank(X)``. A level
        nobody practises twice has an identically-zero ``lineafm`` column and
        no estimable coefficient, exactly as a never-repeated KC does in AFM;
        two levels tagging the same rows share a column; and a reference level
        is dropped for each redundancy between blocks that partition the rows.
        Per-level intercepts on ``m`` components carry ``m - 1`` of those, and
        all of them are broken — the student's level goes first, then the
        latest-declared component's, so the component named first in ``terms``
        keeps every level.

    :raises ValueError: for a term whose feature needs a clock the export does
        not carry, or whose values come out non-finite.
    """
    terms = tuple(terms)
    if not terms:
        raise ValueError("An LKT model needs at least one term")
    if cost is not None:
        if cost <= 0:
            raise ValueError(f"cost must be positive (it enters as l2 = 1/cost), got {cost!r}")
        l2 = 1.0 / float(cost)

    seen: dict[str, Term] = {}
    for term in terms:
        if (first := seen.get(term.block_name)) is not None:
            raise ValueError(f"Term {term} is specified twice (first as {first})")
        seen[term.block_name] = term
    clock = _Clock(data)
    layouts: dict[str, _Layout] = {}
    blocks = []
    for term in terms:
        layout = layouts.get(term.component)
        if layout is None:
            layout = layouts[term.component] = _layout(
                data, component_labels(data, term.component))
        blocks.append(_term_block(term, layout,
                                  _term_values(term, data, layout, clock), l2))

    design = Design(tuple(blocks))
    return design.identify() if identify else design


@dataclass
class LKTFit(LogisticFit):
    """A fitted LKT model: everything :class:`~leapfit.fit.LogisticFit`
    reports, plus a per-component view of the coefficients."""

    #: Nonlinear feature parameters fitted alongside the coefficients, by
    #: :func:`fit_lkt_pars`. Zero when the parameters were held where the
    #: caller put them. They are **counted in** :attr:`n_params`, and therefore
    #: in AIC and BIC, because they were estimated from the same data.
    n_fitted_pars: int = 0

    @property
    def terms(self) -> tuple[Term, ...]:
        return design_terms(self.design)

    def summary(self) -> str:
        text = super().summary()
        if self.n_fitted_pars:
            text += (f"\n  profile        {self.n_fitted_pars} feature parameter(s) "
                     "fitted too, and counted above; the profile is not convex, so "
                     "the inner certificate does not extend to them")
        return text

    def notation(self) -> str:
        """The fitted model as the reference would write it."""
        return " + ".join(str(t) for t in self.terms)

    def component_values(self, data: StepData, component: str) -> pd.DataFrame:
        """One row per level of ``component``, one column per feature on it.

        The generalization of :meth:`leapfit.afm.AFMFit.kc_values`, and it
        keeps that method's conventions. Every level in ``data`` gets a row,
        including levels whose column was aliased away — those report ``NaN``
        rather than ``0.0``, because they were never estimated and a zero
        would be read as an estimate of zero. A feature fitted with one shared
        coefficient is broadcast to every level, as
        :meth:`leapfit.pfa.PFAFit.kc_values` broadcasts pooled slopes.

        ``Separated`` flags levels whose coefficient has no finite maximum —
        every observation touching them came out the same way — so the printed
        number is wherever the optimizer stopped. It is reported for per-level
        features only; a shared coefficient that separates shows up in
        :meth:`~leapfit.fit.LogisticFit.summary` instead.
        """
        here = [t for t in self.terms if t.component == component]
        if not here:
            fitted = sorted({t.component for t in self.terms})
            raise ValueError(
                f"No term on component {component!r}; this fit has {', '.join(fitted)}"
            )

        levels = sorted({label for row in component_labels(data, component)
                         for label in row})
        by_block = self.separated.by_block()
        frame: dict[str, object] = {"Level": levels}
        diverging: set[str] = set()

        for term in here:
            values = self._block_values(term.block_name)
            header = term.notation() + term._pars_suffix()
            if term.per_level:
                frame[header] = [values.get(level, np.nan) for level in levels]
                diverging |= set(by_block.get(term.block_name, ()))
            else:
                frame[header] = [next(iter(values.values()), np.nan)] * len(levels)

        frame["Separated"] = [level in diverging for level in levels]
        return pd.DataFrame(frame)


def fit_lkt(design: Design, y, *, method: str = DEFAULT_METHOD,
            max_fun: int | None = None, tol: float | None = None,
            w0: np.ndarray | None = None,
            warn_not_converged: bool = True,
            warn_separated: bool = True) -> LKTFit:
    """Fit an LKT design by penalized maximum likelihood —
    :func:`leapfit.fit.fit_logistic` with an LKT reporting view.

    The reference solves the same objective through ``LiblineaR`` at
    ``epsilon = 1e-4``; this uses the same solver, certificate and conventions
    as every other family here. See :func:`~leapfit.fit.fit_logistic` for the
    parameters.
    """
    return fit_logistic(design, y, method=method, max_fun=max_fun, tol=tol,
                        w0=w0, warn_not_converged=warn_not_converged,
                        warn_separated=warn_separated, result_type=LKTFit,
                        label="LKT", stacklevel=3)  # 3: attribute past this wrapper


# --------------------------------------------------------------------------
# Fitting the feature parameters: a profile likelihood over the design builder
# --------------------------------------------------------------------------

#: The reference's own bounds on every nonlinear parameter (``lowb``, ``highb``),
#: recycled across all of them regardless of what they mean. Kept as the default
#: so a search here starts from the same feasible set the published searches did.
PARAMETER_BOUNDS = (1e-5, 0.99999)

#: Step for the central differences the outer optimizer works from. R's
#: ``optim`` uses ``ndeps = 1e-3``; scipy's L-BFGS-B defaults to ``1e-8``, which
#: is the wrong order here — the profile is only as smooth as the inner solve is
#: tight, so a step that small differentiates the inner optimizer's own noise.
PARAMETER_STEP = 1e-3

#: A step of :data:`PARAMETER_STEP` that buys less than this many nats is
#: treated as no improvement. Nats rather than a gradient norm because it is
#: the quantity a reader can act on: "no single parameter moved by 0.001
#: improves the fit by more than this".
PARAMETER_TOLERANCE = 1e-2

#: What the outer loop maximizes. ``"penalized"`` profiles the objective the
#: inner solver actually maximizes, so the pair (parameters, coefficients) is a
#: maximizer of one function. ``"likelihood"`` profiles the plain Bernoulli
#: log-likelihood of a penalized fit, which is what the reference does and is
#: not a single objective at all; the two coincide whenever ``l2 = 0``.
OBJECTIVES = ("penalized", "likelihood")


def _flat_parameters(terms: Sequence[Term]) -> tuple[list[tuple[int, int]], list[str]]:
    """Every parameter of every term, flattened, with a label apiece."""
    slots, labels = [], []
    for t_index, term in enumerate(terms):
        for p_index in range(len(term.pars)):
            slots.append((t_index, p_index))
            suffix = f"[{p_index}]" if len(term.pars) > 1 else ""
            labels.append(f"{term.component}:{term.notation()}{suffix}")
    return slots, labels


def _with_parameters(terms: Sequence[Term], slots: Sequence[tuple[int, int]],
                     values: np.ndarray) -> tuple[Term, ...]:
    """``terms`` with the parameters at ``slots`` replaced by ``values``."""
    pars = [list(term.pars) for term in terms]
    for (t_index, p_index), value in zip(slots, values):
        pars[t_index][p_index] = float(value)
    return tuple(replace(term, pars=tuple(row)) for term, row in zip(terms, pars))


def _conform(design: Design, template: Design) -> Design:
    """Cut ``design`` down to the columns ``template`` kept, block by block.

    Identification runs **once**, at the seed, and the column set it chose is
    then held fixed for every other parameter value. Two reasons, and they
    point the same way. It is a dense ``p x p`` eigendecomposition, so running
    it per evaluation would dominate the search; and a parameter value that
    happened to make one more column identically zero would change the
    parameter count mid-search, which would make the AIC of one evaluation
    incomparable with the next. This is the choice
    :meth:`~leapfit.design.Design.take` already makes for cross-validation
    folds, for the same reason.

    Blocks are matched by position rather than by name, because a term's
    parameters are part of its block name and these are exactly the blocks
    whose parameters are moving.
    """
    blocks, dropped, reasons = [], [], []
    for block, kept in zip(design.blocks, template.blocks):
        wanted = set(kept.columns)
        mask = np.array([c in wanted for c in block.columns], dtype=bool)
        for column in np.asarray(block.columns, dtype=object)[~mask]:
            dropped.append(f"{block.name}:{column}")
            reasons.append("not estimable at the seed parameters (identification "
                           "is decided once and held)")
        blocks.append(block.keep(mask))
    return Design(tuple(blocks), Aliased(tuple(dropped), tuple(reasons)))


@dataclass(frozen=True)
class LKTProfile:
    """A fitted LKT model **including** its feature parameters.

    What separates this from :func:`fit_lkt` is one certificate. The inner
    problem is convex, so :attr:`LogisticFit.is_optimal` certifies a *global*
    maximum over the coefficients. The profile over the feature parameters is
    **not** convex — a decay rate enters the design nonlinearly — so
    :attr:`is_stationary` certifies only that no small step improves the fit.
    Whether some distant parameter value is better is a question this cannot
    answer, and :attr:`restarts` is how to ask it.
    """

    terms: tuple[Term, ...]
    fit: LKTFit
    labels: tuple[str, ...]
    seeds: np.ndarray
    pars: np.ndarray
    bounds: tuple[tuple[float, float], ...]
    objective: str
    n_evaluations: int
    converged: bool
    message: str
    max_gain: float
    gains: np.ndarray
    trajectory: pd.DataFrame
    restarts: pd.DataFrame
    inner_not_optimal: int

    @property
    def n_free(self) -> int:
        return len(self.labels)

    @property
    def is_stationary(self) -> bool:
        """No single-parameter step of :data:`PARAMETER_STEP` improves the fit.

        A local statement, and deliberately weaker than the inner certificate.
        """
        return bool(self.max_gain <= PARAMETER_TOLERANCE)

    def frame(self) -> pd.DataFrame:
        """One row per fitted parameter: where it started, where it landed."""
        lower, upper = zip(*self.bounds)
        return pd.DataFrame({
            "parameter": list(self.labels),
            "seed": self.seeds,
            "estimate": self.pars,
            "lower": lower,
            "upper": upper,
            "at_bound": [p <= lo + 1e-12 or p >= hi - 1e-12
                         for p, (lo, hi) in zip(self.pars, self.bounds)],
            "gain_from_stepping": self.gains,
        })

    def summary(self) -> str:
        flag = "" if self.is_stationary else "  *** NOT STATIONARY ***"
        at_bound = int(self.frame()["at_bound"].sum())
        lines = [
            (f"LKT profile over {self.n_free} feature parameter(s), "
             f"{self.n_evaluations} evaluation(s), objective '{self.objective}'"),
            "  " + " | ".join(f"{name} {value:.6g}"
                              for name, value in zip(self.labels, self.pars)),
            (f"  stationarity   best single-parameter step buys {self.max_gain:.3g} "
             f"nats (tol {PARAMETER_TOLERANCE:g}){flag}"),
        ]
        if at_bound:
            lines.append(f"  {at_bound} parameter(s) resting on a bound — the estimate is "
                         "the bound, not an interior maximum")
        if len(self.restarts) > 1:
            spread = self.restarts["objective"].max() - self.restarts["objective"].min()
            reached = self.restarts["objective"].round(6).nunique()
            lines.append(f"  restarts       {len(self.restarts)} start(s) reached "
                         f"{reached} distinct optimum/optima, spread {spread:.4g} nats")
        else:
            lines.append("  restarts       1 start; the profile is not convex, so this "
                         "says nothing about other basins")
        if self.inner_not_optimal:
            lines.append(f"  *** {self.inner_not_optimal} inner fit(s) did not reach a "
                         "certified optimum; the profile is noisy where they did not")
        lines.append(self.fit.summary())
        return "\n".join(lines)


def fit_lkt_pars(data: StepData, terms: Iterable[Term], *,
                 free: Sequence[bool] | None = None,
                 bounds: tuple[float, float] | Sequence[tuple[float, float]] = PARAMETER_BOUNDS,
                 starts: Sequence[Sequence[float]] | None = None,
                 objective: str = "penalized",
                 l2: float = 0.0, cost: float | None = None, identify: bool = True,
                 max_iterations: int = 100, warm_start: bool = True,
                 method: str = DEFAULT_METHOD, max_fun: int | None = None,
                 tol: float | None = None) -> LKTProfile:
    """Fit an LKT model's feature parameters along with its coefficients.

    A profile likelihood. For each candidate parameter vector the features are
    recomputed, the design is rebuilt, and the coefficients are fitted to
    convergence; the outer optimizer moves the parameters over the resulting
    surface. That is the reference's own procedure
    (``optim(method = "L-BFGS-B")`` around a full refit), and it is expensive
    for the same reason: every outer evaluation is an entire model fit.

    Three things this does that the reference does not, all of which follow
    from the same place — the profile is not the convex problem the inner
    solver is certified on:

    * **The parameters are counted.** ``fit.n_params`` is ``rank(X)`` plus the
      number fitted here, so the AIC and BIC of a searched model are charged
      for the search. The reference reports no parameter count at all.
    * **Stationarity is checked, not assumed.** After the optimizer stops,
      every parameter is stepped by :data:`PARAMETER_STEP` in both directions
      and refitted; :attr:`LKTProfile.max_gain` is the best improvement any of
      those found. See :attr:`LKTProfile.is_stationary` for what that does and
      does not certify.
    * **Restarts are available and reported.** Pass several ``starts`` and the
      spread of the optima they reach is a measurement of how much the
      non-convexity matters on your data, rather than an assumption that it
      does not.

    :param terms: the specification. Each term's ``pars`` are its **seeds**.
    :param free: a boolean per parameter, flattened over terms in order — the
        analogue of the reference's ``fixedpars``, where ``NA`` means "fit it".
        Defaults to fitting every parameter. :meth:`LKTProfile.frame` names
        them in the same order.
    :param bounds: one ``(lower, upper)`` pair for every parameter, or one per
        fitted parameter. Defaults to :data:`PARAMETER_BOUNDS`, the reference's,
        which are recycled across parameters whatever they mean — an exponent
        and a decay rate get the same box.
    :param starts: parameter vectors to start from, each of length ``free``.
        Defaults to a single start at the seeds. The best optimum is returned
        and all of them are reported in :attr:`LKTProfile.restarts`.
    :param objective: which surface to profile — see :data:`OBJECTIVES`.
    :param warm_start: seed each inner fit from the previous one's
        coefficients. Safe to do aggressively because the inner problem is
        convex and every inner fit certifies itself independently of where it
        started; worth doing because neighbouring parameter values give
        neighbouring fits.
    :param max_iterations: outer iteration cap, the reference's ``maxitv``.
    """
    terms = tuple(terms)
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    slots, all_labels = _flat_parameters(terms)
    if not slots:
        raise ValueError(
            "No term in this specification takes a parameter, so there is nothing to "
            "fit here — use fit_lkt(build_lkt_design(...), y)."
        )

    if free is None:
        free = [True] * len(slots)
    if len(free) != len(slots):
        raise ValueError(
            f"free has {len(free)} entr(ies) for {len(slots)} parameter(s): "
            f"{', '.join(all_labels)}"
        )
    chosen = [i for i, keep in enumerate(free) if keep]
    if not chosen:
        raise ValueError("free selects no parameter; every one is held fixed")

    fitted_slots = [slots[i] for i in chosen]
    labels = tuple(all_labels[i] for i in chosen)
    seeds = np.array([terms[t].pars[p] for t, p in fitted_slots], dtype=float)

    pairs = list(bounds)
    box = ([tuple(map(float, bounds))] * len(chosen)
           if len(pairs) == 2 and np.isscalar(pairs[0]) else
           [tuple(map(float, b)) for b in pairs])
    if len(box) != len(chosen):
        raise ValueError(f"{len(box)} bound pair(s) for {len(chosen)} fitted parameter(s)")

    y = np.asarray(data.y, dtype=float)
    inner = {"method": method, "max_fun": max_fun, "tol": tol,
             "warn_not_converged": False, "warn_separated": False}

    # Identification is decided here, once, and held for every evaluation.
    template = build_lkt_design(data, _with_parameters(terms, fitted_slots, seeds),
                                l2=l2, cost=cost, identify=identify)

    history: list[dict] = []
    state = {"w": None, "not_optimal": 0}

    def evaluate(values: np.ndarray) -> tuple[float, LKTFit, Design]:
        at = _with_parameters(terms, fitted_slots, values)
        design = _conform(build_lkt_design(data, at, l2=l2, cost=cost, identify=False),
                          template)
        fit = fit_lkt(design, y, w0=state["w"] if warm_start else None, **inner)
        if warm_start:
            state["w"] = fit.weights
        if not fit.is_optimal:
            state["not_optimal"] += 1
        value = fit.ll if objective == "penalized" else fit.ll_unpenalized
        history.append({"evaluation": len(history), **dict(zip(labels, map(float, values))),
                        "objective": value, "inner_optimal": fit.is_optimal})
        return value, fit, design

    def negated(values: np.ndarray) -> float:
        return -evaluate(np.asarray(values, dtype=float))[0]

    attempts = [np.asarray(s, dtype=float) for s in (starts or [seeds])]
    for start in attempts:
        if start.shape != (len(chosen),):
            raise ValueError(f"a start has {start.size} value(s) for {len(chosen)} parameter(s)")

    outcomes, best = [], None
    for start in attempts:
        state["w"] = None
        clipped = np.clip(start, [lo for lo, _ in box], [hi for _, hi in box])
        result = minimize(negated, clipped, method="L-BFGS-B", bounds=box,
                          options={"maxiter": max_iterations, "eps": PARAMETER_STEP})
        value, fit, design = evaluate(result.x)
        outcomes.append({"start": tuple(float(v) for v in start),
                         "estimate": tuple(float(v) for v in result.x),
                         "objective": value, "converged": bool(result.success),
                         "message": str(result.message)})
        if best is None or value > best[0]:
            best = (value, result, fit, design)

    _, result, fit, design = best

    # Stationarity, measured rather than inferred: step each parameter both ways
    # and see whether anything better is within reach.
    gains = np.zeros(len(chosen))
    for j in range(len(chosen)):
        for direction in (-1.0, 1.0):
            probe = np.array(result.x, dtype=float)
            probe[j] = float(np.clip(probe[j] + direction * PARAMETER_STEP, *box[j]))
            if probe[j] == result.x[j]:
                continue
            gains[j] = max(gains[j], evaluate(probe)[0] - best[0])

    fit.n_fitted_pars = len(chosen)
    fit.n_params = design.n_params + len(chosen)

    return LKTProfile(
        terms=_with_parameters(terms, fitted_slots, result.x),
        fit=fit, labels=labels, seeds=seeds,
        pars=np.asarray(result.x, dtype=float), bounds=tuple(box),
        objective=objective, n_evaluations=len(history),
        converged=bool(result.success), message=str(result.message),
        max_gain=float(gains.max()), gains=gains,
        trajectory=pd.DataFrame(history), restarts=pd.DataFrame(outcomes),
        inner_not_optimal=state["not_optimal"],
    )
