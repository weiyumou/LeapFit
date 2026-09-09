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

**What this module covers, and what it does not.** The reference computes
roughly fifty features. Implemented here are the ones that are pure functions
of prior success and failure counts — no clock, no fitted shape parameter:

    intercept  lineafm  logafm  powafm  linesuc  logsuc  linefail  logfail
    linecomp  prop  numer

The rest are refused by name with the reason, because a spec silently missing a
term is worse than one that will not build. Two things they need that this
package does not yet have:

* **Numeric time.** ``recency``, ``base``, ``base2/4/5``, ``dash*``, ``ppe`` and
  the spacing features need ``CF..Time.`` in seconds, and the ``base2`` family
  additionally needs time-on-task accumulated from ``Step Duration (sec)``.
  :attr:`leapfit.data.StepData.times` holds timestamp *strings*, read only to
  order :meth:`~leapfit.data.StepData.practice_order`.
* **A fitted shape parameter.** ``propdec``, ``logitdec``, ``expdec*``,
  ``recency`` and friends make the design a function of a decay rate, which the
  reference fits by an outer ``optim`` refitting the whole regression at every
  evaluation. ``powafm`` is here only because its exponent can be *fixed*
  (pass ``par=``); nothing in this module searches for one.

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
than by this module's scheme. ``Design.identify`` finds the student/KC sum
redundancy by looking for blocks literally named ``student`` and
``kc_intercept``, so a per-level intercept on those two components uses those
names and every other term is named ``feature[component]``. The shared pass
breaks exactly *one* such redundancy, so a spec carrying per-level intercepts on
any other combination of components is refused rather than fitted with a
parameter count that overstates the model — see :func:`build_lkt_design`.

DIVERGENCE (fit statistics): the reference reports the unpenalized
log-likelihood of predictions clipped to ``[1e-5, 1-1e-5]``, and reports no
parameter count, no AIC and no BIC. Here the likelihood is unclipped and
``n_params`` is the rank of the design, so AIC and BIC exist and mean what they
mean for every other family in this package.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from leapfit.data import StepData
from leapfit.design import Block, Design
from leapfit.fit import DEFAULT_METHOD, LogisticFit, fit_logistic

#: ``LiblineaR``'s cost in the reference's own default call, as ``l2 = 1/cost``.
REFERENCE_COST = 512.0

#: Components this module resolves itself. Anything else is looked up as a
#: column of the source export (see :func:`component_labels`). ``"kc"`` is the
#: only one that can carry several levels on one row.
COMPONENTS = ("student", "item", "kc")

#: Feature value from prior success and failure counts, with an optional fixed
#: parameter. Written from the reference's documented semantics; the branch
#: names match ``computefeatures`` so a spec reads the same in both systems.
FEATURES: dict[str, Callable[[int, int, float | None], float]] = {
    "intercept": lambda s, f, p: 1.0,
    "lineafm": lambda s, f, p: float(s + f),
    "logafm": lambda s, f, p: math.log1p(s + f),
    "powafm": lambda s, f, p: float(s + f) ** p,
    "linesuc": lambda s, f, p: float(s),
    "logsuc": lambda s, f, p: math.log1p(s),
    "linefail": lambda s, f, p: float(f),
    "logfail": lambda s, f, p: math.log1p(f),
    "linecomp": lambda s, f, p: float(s - f),
    # The reference seeds an unpractised level at .5 (a 0/0 guarded by ifelse).
    "prop": lambda s, f, p: 0.5 if s + f == 0 else s / (s + f),
}

#: Features taking a fixed shape parameter. Required for these, refused for the
#: rest; nothing here searches for one.
PARAMETRIC = ("powafm",)

#: The component's own column read as a number rather than as a factor.
#: Handled outside :data:`FEATURES` because it consults no history at all.
NUMERIC_FEATURE = "numer"

STATIC_FEATURES = (*sorted(FEATURES), NUMERIC_FEATURE)

#: Reference features this module refuses, and why. Named individually so a
#: spec copied out of a paper fails with the reason rather than with
#: "unknown feature".
DEFERRED: dict[str, str] = {
    **dict.fromkeys(
        ("recency", "recencysuc", "recencyfail", "recencystudy", "recencytest",
         "base", "base2", "base4", "basesuc", "basefail", "base2suc", "base2fail",
         "base5suc", "base5fail", "ppe", "dashafm", "dashsuc"),
        "needs numeric time; StepData carries timestamp strings only",
    ),
    **dict.fromkeys(
        ("propdec", "propdec2", "logitdec", "logitdecevol", "logit", "errordec",
         "expdecafm", "expdecsuc", "expdecfail", "baseratepropdec"),
        "needs a fitted decay parameter; the design would depend on it",
    ),
    **dict.fromkeys(
        ("diffcor1", "diffcor2", "diffincor1", "diffincor2", "diffall1", "diffall2",
         "diffcorComp", "diffincorComp", "diffallComp", "diffrelcor1", "diffrelcor2"),
        "needs predictions from a prior fit as an input column",
    ),
}

_BLOCK_NAME = re.compile(r"^(?P<feature>[A-Za-z]+)(?P<per_level>\$?)"
                         r"(?:\((?P<par>[^)]*)\))?\[(?P<component>.+)]$")

#: Per-level intercepts on these two components are named the way the shared
#: identification pass expects; see the module docstring's block-name divergence.
_SHARED_BLOCK_NAME = {"student": "student", "kc": "kc_intercept"}


@dataclass(frozen=True)
class Term:
    """One component paired with one feature — a group of design columns.

    ``per_level`` is the reference's ``$``: one coefficient per level of the
    component instead of one shared across them. Intercepts are always
    per-level (a factor is expanded whether or not it is written with a ``$``),
    so the flag is forced on for them, as in the reference.

    ``par`` is a *fixed* shape parameter, accepted only by the features in
    :data:`PARAMETRIC`. It is part of the term's identity and of its block
    name, so two ``powafm`` terms on one component at different exponents do
    not collide.
    """

    component: str
    feature: str
    per_level: bool = False
    par: float | None = None

    def __post_init__(self) -> None:
        if self.feature in DEFERRED:
            raise NotImplementedError(
                f"Feature {self.feature!r} is not implemented here: "
                f"{DEFERRED[self.feature]}. Implemented features are "
                f"{', '.join(STATIC_FEATURES)}."
            )
        if self.feature not in FEATURES and self.feature != NUMERIC_FEATURE:
            raise ValueError(
                f"Unknown feature {self.feature!r}. Implemented: "
                f"{', '.join(STATIC_FEATURES)}."
            )
        if self.feature == NUMERIC_FEATURE and self.per_level:
            raise ValueError(
                "'numer' reads its component as a number, not as a factor, so "
                "'numer$' has no levels to extend over. Drop the '$'."
            )
        needs_par = self.feature in PARAMETRIC
        if needs_par and self.par is None:
            raise ValueError(
                f"Feature {self.feature!r} takes a shape parameter and nothing here "
                "fits one. Pass par=<value> to hold it fixed."
            )
        if not needs_par and self.par is not None:
            raise ValueError(f"Feature {self.feature!r} takes no parameter, got par={self.par!r}")
        # Frozen, but the reference's own normalization: a factor is expanded
        # whether or not the spec writes the '$'.
        if self.feature == "intercept" and not self.per_level:
            object.__setattr__(self, "per_level", True)

    @classmethod
    def parse(cls, component: str, feature: str, par: float | None = None) -> Term:
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
        return cls(component=_ALIASES.get(component, component),
                   feature=text.rstrip("$"), per_level=per_level, par=par)

    def notation(self) -> str:
        """The feature as the reference writes it: ``lineafm$``.

        An intercept is written without the ``$`` even though it is per-level,
        because that is how the reference writes it — the expansion is implicit
        for a factor. :meth:`parse` reads it back to the same term.
        """
        marker = "$" if self.per_level and self.feature != "intercept" else ""
        return f"{self.feature}{marker}"

    def __str__(self) -> str:
        par = "" if self.par is None else f"({self.par:g})"
        return f"{self.component}:{self.notation()}{par}"

    @property
    def block_name(self) -> str:
        """Design block this term contributes — see the block-name divergence."""
        if self.feature == "intercept" and self.component in _SHARED_BLOCK_NAME:
            return _SHARED_BLOCK_NAME[self.component]
        par = "" if self.par is None else f"({self.par:g})"
        return f"{self.notation()}{par}[{self.component}]"


#: Component names the reference uses, mapped onto this package's. Only the
#: student is fixed by LKT's own contract; a KC model is chosen when the export
#: is loaded, so ``KC..Default.`` has no general translation and stays a
#: source-column lookup.
_ALIASES = {"Anon.Student.Id": "student", "Anon Student Id": "student"}


def lkt_terms(components: Sequence[str], features: Sequence[str],
              pars: Sequence[float | None] | None = None) -> tuple[Term, ...]:
    """Terms from the reference's parallel ``components``/``features`` vectors.

    Exists so a specification can be copied out of a paper unchanged::

        lkt_terms(components=("student", "kc", "kc"),
                  features=("intercept", "intercept", "lineafm$"))

    ``pars`` parallels the same vectors and holds a fixed parameter for the
    features that take one (``None`` elsewhere) — deliberately positional like
    the reference's ``fixedpars``, but *per term* rather than a single flat
    vector consumed in feature order, because the flat form is what makes the
    reference's own parameter bookkeeping hard to read.
    """
    if len(components) != len(features):
        raise ValueError(
            f"{len(components)} component(s) but {len(features)} feature(s); the two "
            "vectors are positional and must be the same length"
        )
    if pars is None:
        pars = [None] * len(features)
    elif len(pars) != len(features):
        raise ValueError(f"{len(pars)} par(s) for {len(features)} feature(s)")
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
            out.append(Term(component=m["component"], feature=m["feature"],
                            per_level=bool(m["per_level"]),
                            par=None if m["par"] is None else float(m["par"])))
        else:
            raise ValueError(f"Block {block.name!r} was not built by leapfit.lkt")
    return tuple(out)


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
    feature on the student component means.
    """
    if len(labels) != len(data):
        raise ValueError(f"{len(data)} observations but {len(labels)} label rows")
    s_out: list[tuple[int, ...]] = [()] * len(data)
    f_out: list[tuple[int, ...]] = [()] * len(data)
    for rows in data.practice_order().values():
        s_seen: dict[str, int] = {}
        f_seen: dict[str, int] = {}
        for i in rows:
            correct = int(data.y[i])
            s_row, f_row = [], []
            for label in labels[i]:
                s_row.append(s_seen.get(label, 0))
                f_row.append(f_seen.get(label, 0))
                s_seen[label] = s_seen.get(label, 0) + correct
                f_seen[label] = f_seen.get(label, 0) + (1 - correct)
            s_out[i], f_out[i] = tuple(s_row), tuple(f_row)
    return s_out, f_out


def _term_values(term: Term, data: StepData, labels: Sequence[tuple[str, ...]],
                 counts: tuple[list[tuple[int, ...]], list[tuple[int, ...]]] | None,
                 ) -> list[tuple[float, ...]]:
    """One value per (observation, level), aligned with ``labels``."""
    if term.feature == NUMERIC_FEATURE:
        column = pd.to_numeric(pd.Series(_source_column(data, term.component)),
                               errors="coerce").to_numpy(dtype=float)
        if np.isnan(column).any():
            bad = int(np.isnan(column).sum())
            raise ValueError(
                f"'numer' reads component {term.component!r} as a number, but {bad:,} "
                "of its values do not parse as one."
            )
        return [(v,) * len(row) for v, row in zip(column, labels)]

    value = FEATURES[term.feature]
    s_counts, f_counts = counts  # type: ignore[misc]
    return [tuple(value(s, f, term.par) for s, f in zip(s_row, f_row))
            for s_row, f_row in zip(s_counts, f_counts)]


def _term_block(term: Term, labels: Sequence[tuple[str, ...]],
                values: Sequence[tuple[float, ...]], l2: float) -> Block:
    n = len(labels)
    if not term.per_level:
        # One shared coefficient. A row carrying several levels contributes the
        # sum over them, which is what an additive multi-KC step means here and
        # what leapfit.pfa's pooled slopes already do.
        totals = np.array([float(sum(row)) for row in values])
        return Block.build(term.block_name, totals[:, None], [term.notation()], l2=l2)

    levels = sorted({label for row in labels for label in row})
    index = {label: j for j, label in enumerate(levels)}
    rows, cols, vals = [], [], []
    for i, (row_labels, row_values) in enumerate(zip(labels, values)):
        for label, value in zip(row_labels, row_values):
            rows.append(i)
            cols.append(index[label])
            vals.append(float(value))
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(levels)))
    matrix.eliminate_zeros()  # a zero count is a structural zero, not a datum
    return Block.build(term.block_name, matrix, levels, l2=l2)


def _refuse_unbreakable_redundancy(terms: Sequence[Term]) -> None:
    """Refuse specs whose parameter count this package cannot state honestly.

    Per-level intercept blocks each sum to the all-ones vector, so ``m`` of them
    carry ``m - 1`` redundant directions per connected component.
    :meth:`~leapfit.design.Design.identify` breaks exactly one, between the
    blocks named ``student`` and ``kc_intercept``. Anything else would leave a
    design whose ``rank`` is below its column count — caught by ``identify``'s
    own check, but caught there with a message about collinear blocks rather
    than about the spec that produced them.
    """
    intercepts = {t.component for t in terms if t.feature == "intercept"}
    if len(intercepts) > 1 and intercepts != {"student", "kc"}:
        listed = ", ".join(sorted(intercepts))
        extra = len(intercepts) - 1
        raise NotImplementedError(
            f"Per-level intercepts on {len(intercepts)} components ({listed}) carry "
            f"{extra} redundant direction{'s' if extra > 1 else ''}, and the shared "
            "identification pass breaks only the student/KC one. Reduce to a single "
            "intercept component, use student+kc, or pass identify=False and count "
            "parameters yourself — leaving them in would make AIC and BIC charge for "
            "parameters that do not exist."
        )


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
        two levels tagging the same rows share a column; and one student is
        dropped as the reference level when a student and a KC intercept make
        the design redundant.

    :raises NotImplementedError: for a spec whose intercepts carry a redundancy
        the shared pass cannot break. See :func:`_refuse_unbreakable_redundancy`.
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
    if identify:
        # Only when we are the ones promising n_params == rank(X). Under
        # identify=False the caller has taken the parameter count on themselves.
        _refuse_unbreakable_redundancy(terms)

    label_cache: dict[str, list[tuple[str, ...]]] = {}
    count_cache: dict[str, tuple[list[tuple[int, ...]], list[tuple[int, ...]]]] = {}

    blocks = []
    for term in terms:
        labels = label_cache.get(term.component)
        if labels is None:
            labels = label_cache[term.component] = component_labels(data, term.component)
        counts = None
        if term.feature != NUMERIC_FEATURE:
            counts = count_cache.get(term.component)
            if counts is None:
                counts = count_cache[term.component] = history_counts(data, labels)
        blocks.append(_term_block(term, labels,
                                  _term_values(term, data, labels, counts), l2))

    design = Design(tuple(blocks))
    return design.identify() if identify else design


@dataclass
class LKTFit(LogisticFit):
    """A fitted LKT model: everything :class:`~leapfit.fit.LogisticFit`
    reports, plus a per-component view of the coefficients."""

    @property
    def terms(self) -> tuple[Term, ...]:
        return design_terms(self.design)

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
            if term.per_level:
                frame[term.notation()] = [values.get(level, np.nan) for level in levels]
                diverging |= set(by_block.get(term.block_name, ()))
            else:
                shared = next(iter(values.values()), np.nan)
                frame[term.notation()] = [shared] * len(levels)

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
