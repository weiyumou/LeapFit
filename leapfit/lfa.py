"""Learning Factors Analysis: a search over KC models, scored by AFM.

LFA is not a model. AFM is the model; LFA is a *search* over the KC labelling
AFM conditions on, ranked by AFM's own BIC or AIC. So this module sits
**above** :mod:`leapfit.afm` and adds no estimator: a state is a KC model,
scoring one is :func:`~leapfit.afm.build_afm_design` followed by
:func:`~leapfit.afm.fit_afm`, and everything those carry — the identification
pass, the separation check, the KKT optimality certificate — applies unchanged
to every state the search visits::

    state   a Q matrix: one KC label per step
    root    the "All" model, one skill on every step
    P       which steps carry which difficulty factor, built from KC models
            that already exist on the export (:func:`build_factor_matrix`)
    split   carve the steps of skill k that carry factor f into a new skill
            "k*f", leaving the rest of k behind
    merge   undo a recorded split, rejoining its two children
    rank    greedy best-first on the fitted state's BIC (or AIC)

Provenance. The reference is DataShop's LFA — ``AnalysisLFASearch`` in
LearnSphere/WorkflowComponents, and the offline ``lfa-6.0`` tool — whose
search engine ships only as a binary. Its behaviour was recovered from
bytecode and from running it locally, and is recorded with measurements in
``results/lfa-learnsphere-20260902/README.md``. Two findings there are
load-bearing. leapfit reproduces the reference's fit statistics to **1.2e-9**
under ``learnsphere_compat=True``, so this is a search layer over an
already-validated scorer. And the reference's own optimizer stops early on
some states — understating 21 of 99 log-likelihoods at depth 1 by up to 6.6
nats, and **all 99** at depth 3 — which reorders its frontier (132 pairwise
rank inversions among the states it reported). That is why
:attr:`LFAState.is_optimal` rides on every state instead of being assumed: a
search that ranks by a fit statistic is only as good as its fits.

DIVERGENCE (screening). The reference accepts whatever split its heuristic
prefers, and on the validated export the model it selected contained a KC
whose slope has no finite estimate — reported as ``-17.05``, which is merely
where its optimizer stopped. Carving off a small, outcome-homogeneous set of
steps buys real likelihood while AIC/BIC charge one or two parameters, so the
criterion prefers such states systematically: **34%** of the 941 candidate
moves at one measured node were separated. Neither honest parameter counting
(``n_params = rank(X)``) nor bounding the slopes removes the preference — both
were measured, and the same single-step split still ranked first of 941, so
this is a property of the *criterion*, not of the estimator. Every candidate
move is therefore screened on two independent axes (:func:`screen`) and what
was refused is kept (:class:`Rejected`) rather than dropped silently. Both
screens run *before* the fit, so a refused move costs no optimization.

DIVERGENCE (opportunity counts). A searched KC model did not exist when the
export was written, so it has no ``Opportunity (...)`` column to read. ``T``
is always recomputed from :meth:`~leapfit.data.StepData.practice_order`; the
flag that makes recomputation optional for AFM is not optional here.

DIVERGENCE (merge). The published method has only Binary Split — Cen,
Koedinger & Junker (2006) merge by hand into the *starting* model rather than
as a search move. :func:`lfa_search` offers merge as lineage-undo, and it
matters only because the frontier is bounded: undoing a split reaches a state
that was scored earlier and may since have been evicted from the beam. It adds
search *paths*, not reachable states — split alone can reach any refinement
the factors express. Arbitrary pairwise merge over all KC pairs is a larger
operator and is deliberately not implemented here.

Not implemented, deliberately: Firth-penalized likelihood, which would remove
the unbounded reward at its source instead of screening the symptom. Its
penalty is ``0.5 * log det I(beta)`` — a different estimator, not a per-column
ridge, so it belongs to a project extending this one rather than to this
module.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leapfit.afm import build_afm_design, fit_afm
from leapfit.crossval import paired_contrasts, paired_cross_validate, paired_scores
from leapfit.data import StepData
from leapfit.design import Design
from leapfit.fit import DEFAULT_METHOD

#: Search heuristics. ``"bic"`` is the default because it is the reference's
#: and because it charges ``log N`` per parameter against AIC's 2 — on a
#: 20,687-observation export, 9.94 against 2.00, so a two-parameter split must
#: buy 19.9 nats rather than 4.0. Within one expansion in compat mode the two
#: rank identically (all siblings share ``n_params``); they differ in *where
#: the search stops*, not in which child it picks.
HEURISTICS = ("bic", "aic")

#: Minimum observations at ``T >= 1`` a newly created KC must have. Those are
#: the rows that identify a slope, and the count separates the three cases
#: measured on the validation export: a factor isolating a step nobody
#: repeated gives 0 (its slope column is identically zero and identification
#: removes it), the reference's degenerate pick gives 5 (all failures, so the
#: slope diverges), and the clean siblings give 201 and 475.
MIN_OPPORTUNITIES = 3

#: Consecutive expansions without improving the incumbent before the search
#: stops. The reference's ``stopRepetitionCount`` default; 0 disables it.
PATIENCE = 5

#: Expansion budget. The reference's web component defaults to 50 and caps at
#: 200; its offline tool ships 200.
MAX_ITERATIONS = 50

#: Unexpanded states retained. The reference caps its queues at 1,000, and the
#: cap is what gives :func:`lfa_search`'s merge operator anything to do.
BEAM = 1000

#: Joins a skill to the factor it was split by, as the reference names them.
SPLIT_SEP = "*"

ROOT_SKILL = "all"


# --------------------------------------------------------------------------
# The difficulty-factor matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorMatrix:
    """P: which steps carry which difficulty factor.

    ``members`` parallels ``factors`` and holds each factor's step set, so a
    split is a set intersection. ``dropped`` and ``reasons`` parallel each
    other and record factor columns the build removed — the same audit-record
    shape as :class:`~leapfit.design.Aliased`, and for the same reason: a
    dropped factor is one the search will never try, which is a fact about the
    search space rather than a detail of how P was assembled.
    """

    steps: tuple[str, ...]
    factors: tuple[str, ...]
    members: tuple[frozenset[str], ...]
    dropped: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.members) != len(self.factors):
            raise ValueError(
                f"members has {len(self.members)} entries for "
                f"{len(self.factors)} factors"
            )
        if len(self.dropped) != len(self.reasons):
            raise ValueError(
                f"dropped has {len(self.dropped)} entries for "
                f"{len(self.reasons)} reasons"
            )

    def __len__(self) -> int:
        return len(self.factors)

    def summary(self) -> str:
        if not self.factors:
            return "no difficulty factors"
        out = f"{len(self.factors)} factor(s) over {len(self.steps)} steps"
        if self.dropped:
            out += f"; {len(self.dropped)} column(s) dropped"
        return out


def build_factor_matrix(models: Mapping[str, StepData]) -> FactorMatrix:
    """Assemble P from several KC models parsed off the same export.

    The reference's rules, reproduced: a factor model must tag exactly one KC
    per row, every model must cover the same rows, exact-duplicate factor
    columns are dropped, and a skill name appearing in two models is prefixed
    with its model name. On the validation export that took 951 columns from
    three models to 943.

    Violations raise rather than warn, following :mod:`leapfit.data`: a model
    with multi-KC rows has no single "the factor value" for a step, and models
    covering different rows cannot index one P matrix, so neither has a
    sensible reading to fall back on.

    :param models: KC-model name to its parsed observations. Every value must
        come from the same export, which is what makes their steps comparable.
    """
    if not models:
        raise ValueError("build_factor_matrix needs at least one KC model")

    coverage: dict[str, tuple] = {}
    for name, data in models.items():
        wide = [i for i, row in enumerate(data.kcs) if len(row) != 1]
        if wide:
            raise ValueError(
                f"KC model {name!r} tags {len(wide)} row(s) with more than one "
                f"KC (first at observation {wide[0]}). A difficulty factor has "
                "one value per step, so multi-KC models cannot be used as one; "
                "convert or exclude the model."
            )
        coverage[name] = tuple(data.items)

    sizes = {name: len(items) for name, items in coverage.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(
            f"KC models cover different numbers of observations: {sizes}. "
            "Factors must be aligned to one row set; models whose KC columns "
            "are blank on different rows cannot be combined."
        )
    reference = next(iter(coverage.values()))
    for name, items in coverage.items():
        if items != reference:
            raise ValueError(
                f"KC model {name!r} covers a different row set from "
                f"{next(iter(coverage))!r}; factors must share one row set."
            )

    steps = tuple(sorted(set(reference)))
    seen: dict[frozenset[str], str] = {}
    names: list[str] = []
    members: list[frozenset[str]] = []
    dropped: list[str] = []
    reasons: list[str] = []

    taken: dict[str, str] = {}  # skill name -> the model that claimed it
    for model, data in models.items():
        by_skill: dict[str, set[str]] = {}
        for item, row in zip(data.items, data.kcs):
            by_skill.setdefault(row[0], set()).add(item)
        for skill in sorted(by_skill):
            column = frozenset(by_skill[skill])
            if (first := seen.get(column)) is not None:
                dropped.append(f"{model}:{skill}")
                reasons.append(f"identical to {first}")
                continue
            label = skill if skill not in taken else f"{model}-{skill}"
            taken.setdefault(skill, model)
            seen[column] = label
            names.append(label)
            members.append(column)

    return FactorMatrix(steps, tuple(names), tuple(members),
                        tuple(dropped), tuple(reasons))


# --------------------------------------------------------------------------
# Moves, refusals, states
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Move:
    """One model operator application, as the reference records it."""

    kind: str
    skill: str
    factor: str

    def __str__(self) -> str:
        return f"{self.kind} {self.skill} by {self.factor}"


@dataclass(frozen=True)
class Rejected:
    """Candidate moves the screens refused, and why.

    ``reasons`` parallel ``moves``. Kept because a screen that silently
    shrinks the search space is indistinguishable from a search space that was
    always that small — and because the refusals are the interesting output:
    on the validation export a third of the moves at one node were refused.
    """

    moves: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.moves) != len(self.reasons):
            raise ValueError(
                f"moves has {len(self.moves)} entries for "
                f"{len(self.reasons)} reasons"
            )

    def __len__(self) -> int:
        return len(self.moves)

    def by_reason(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for move, reason in zip(self.moves, self.reasons):
            out.setdefault(reason, []).append(move)
        return out

    def summary(self) -> str:
        if not self.moves:
            return "no moves refused"
        counts = {k: len(v) for k, v in self.by_reason().items()}
        detail = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
        return f"{len(self.moves)} move(s) refused ({detail})"


@dataclass(frozen=True)
class LFAState:
    """One KC model the search evaluated, with its fit and its certificate."""

    labels: tuple[str, ...]
    history: tuple[Move, ...]
    ll: float
    aic: float
    bic: float
    n_kcs: int
    n_params: int
    is_optimal: bool
    n_separated: int
    n_separated_kc: int = 0
    iteration: int = -1
    weights: np.ndarray | None = field(default=None, repr=False)
    columns: tuple[str, ...] = field(default=(), repr=False)

    @property
    def depth(self) -> int:
        return len(self.history)

    def score(self, heuristic: str) -> float:
        """The value the search ranks by. Lower is better."""
        return self.bic if heuristic == "bic" else self.aic

    def kc_model(self, steps: tuple[str, ...]) -> dict[str, str]:
        """Step to KC label, ready to write back as a new KC model."""
        return dict(zip(steps, self.labels))

    def summary(self) -> str:
        moves = " | ".join(str(m) for m in self.history) or "root"
        flag = "" if self.is_optimal else "  NOT AT OPTIMUM"
        sep = (f"  separated={self.n_separated} "
               f"({self.n_separated_kc} in a KC block)") if self.n_separated else ""
        return (f"{self.n_kcs} KCs, {self.n_params} params  "
                f"ll={self.ll:.4f}  AIC={self.aic:.3f}  BIC={self.bic:.3f}"
                f"{sep}{flag}\n  {moves}")


@dataclass(frozen=True)
class LFAResult:
    """A finished search: what it found, how it got there, what it refused."""

    root: LFAState
    states: tuple[LFAState, ...]
    trajectory: tuple[LFAState, ...]
    rejected: Rejected
    factors: FactorMatrix
    heuristic: str
    n_evaluated: int
    n_iterations: int
    stopped: str
    persistent_separation: tuple[str, ...] = ()
    learnsphere_compat: bool = False

    @property
    def best(self) -> LFAState:
        return self.states[0] if self.states else self.root

    def frame(self) -> pd.DataFrame:
        """The ranked frontier, one row per evaluated state."""
        return pd.DataFrame({
            "rank": range(1, len(self.states) + 1),
            "depth": [s.depth for s in self.states],
            "n_kcs": [s.n_kcs for s in self.states],
            "n_params": [s.n_params for s in self.states],
            "log_likelihood": [s.ll for s in self.states],
            "aic": [s.aic for s in self.states],
            "bic": [s.bic for s in self.states],
            "is_optimal": [s.is_optimal for s in self.states],
            "n_separated": [s.n_separated for s in self.states],
            "found_at_iteration": [s.iteration for s in self.states],
            "history": [" | ".join(str(m) for m in s.history) for s in self.states],
        })

    def summary(self) -> str:
        gain = self.root.score(self.heuristic) - self.best.score(self.heuristic)
        uncertified = sum(1 for s in self.states if not s.is_optimal)
        return "\n".join([
            (f"LFA search over {len(self.factors)} difficulty factor(s), "
             f"heuristic={self.heuristic.upper()}"),
            (f"  {self.n_iterations} expansion(s), {self.n_evaluated} state(s) "
             f"evaluated, stopped: {self.stopped}"),
            f"  root  {self.root.score(self.heuristic):.3f}",
            (f"  best  {self.best.score(self.heuristic):.3f}  "
             f"(improvement {gain:.3f} over {self.best.depth} move(s))"),
            f"  {self.rejected.summary()}",
            (f"  {uncertified} of {len(self.states)} evaluated state(s) not at "
             "a certified optimum"),
            *([(f"  note: {len(self.persistent_separation)} coefficient(s) "
                "outside the KC blocks have no finite estimate in *every* "
                f"state ({', '.join(self.persistent_separation[:3])}); that is "
                "a property of the data, not of any move, and it makes "
                "certification borderline throughout")]
              if self.persistent_separation else []),
            self.best.summary(),
        ])


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------


def split(labels: tuple[str, ...], steps: tuple[str, ...], skill: str,
          factor: str, members: frozenset[str]) -> tuple[str, ...] | None:
    """Carve ``skill``'s steps that carry ``factor`` into ``"skill*factor"``.

    ``None`` when the move is degenerate — the factor covers none of the
    skill's steps, or all of them, so one side would be empty. The reference
    prunes the same two cases before fitting ("delete all zero skill").
    """
    owned = {s for s, label in zip(steps, labels) if label == skill}
    carved = owned & members
    if not carved or carved == owned:
        return None
    new = f"{skill}{SPLIT_SEP}{factor}"
    return tuple(new if (label == skill and step in members) else label
                 for step, label in zip(steps, labels))


def replay(history: tuple[Move, ...], factors: FactorMatrix) -> tuple[str, ...]:
    """The labelling a move sequence produces from the root.

    Merge is implemented as replay-without-a-move rather than as an inverse
    operator, which makes an undone split unreachable by construction: there
    is no state that a history cannot express.
    """
    at = dict(zip(factors.factors, factors.members))
    labels = (ROOT_SKILL,) * len(factors.steps)
    for move in history:
        got = split(labels, factors.steps, move.skill, move.factor,
                    at[move.factor])
        if got is None:
            raise ValueError(f"replaying {move} is degenerate; history is stale")
        labels = got
    return labels


def _partition(labels: tuple[str, ...]) -> frozenset[frozenset[int]]:
    """A label-independent state key, so transposed split orders collapse."""
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    return frozenset(frozenset(v) for v in groups.values())


# --------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------


def relabel(data: StepData, steps: tuple[str, ...],
            labels: tuple[str, ...]) -> StepData:
    """``data`` under a searched KC labelling, with ``T`` recomputed.

    Recomputation is not optional: the labelling did not exist when the export
    was written, so its ``Opportunity`` column does not either.
    """
    at = dict(zip(steps, labels))
    out = dataclasses.replace(data, kcs=[(at[item],) for item in data.items],
                              opportunities=[()] * len(data))
    return dataclasses.replace(out, opportunities=out.recomputed_opportunities())


def screen(data: StepData, design: Design | None, touched: tuple[str, ...], *,
           min_opportunities: int = MIN_OPPORTUNITIES,
           separation: bool = True) -> str | None:
    """Why a candidate state should be refused, or ``None`` to accept it.

    Two independent axes, both required, because neither implies the other:

    **Evidence.** A KC needs observations at ``T >= 1`` before its learning
    rate means anything; those are the only rows the slope column touches.

    **Estimability.** A KC whose slope or intercept column separates the
    responses has no finite estimate, and the search would be rewarded for
    creating it. This is checked on the *design*, before any fit, so a refused
    move costs no optimization.

    :param design: ``None`` when identification already refused the state.
    :param touched: the KC labels the move created or changed.
    """
    counts = dict.fromkeys(touched, 0)
    for row, opportunities in zip(data.kcs, data.opportunities):
        if row[0] in counts and opportunities[0] >= 1:
            counts[row[0]] += 1
    thin = sorted(k for k, n in counts.items() if n < min_opportunities)
    if thin:
        return "too little practice"

    if design is None:
        return "not identifiable"
    if separation:
        by_block = design.separated(data.y).by_block()
        if by_block.get("kc_intercept") or by_block.get("kc_slope"):
            return "separated"
    return None


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------


def _seed(design: Design, parent: LFAState | None) -> np.ndarray | None:
    """Start a child from its parent's coefficients.

    Columns are matched by their qualified name; a column the parent lacks
    (``kc_slope:k*f``) inherits from the skill it was carved out of
    (``kc_slope:k``), which is the coefficient it is closest to. Anything
    still unmatched starts at zero, as a cold fit would.
    """
    if parent is None or parent.weights is None:
        return None
    at = dict(zip(parent.columns, parent.weights))
    out = np.zeros(design.matrix.shape[1])
    for j, column in enumerate(design.columns):
        if column in at:
            out[j] = at[column]
            continue
        block, _, name = column.partition(":")
        origin = f"{block}:{name.rsplit(SPLIT_SEP, 1)[0]}"
        if origin in at:
            out[j] = at[origin]
    return out


def _evaluate(data: StepData, factors: FactorMatrix, labels: tuple[str, ...],
              history: tuple[Move, ...], parent: LFAState | None,
              iteration: int, *, touched: tuple[str, ...],
              min_opportunities: int, separation: bool, learnsphere_compat: bool,
              warm_start: bool, method: str,
              max_fun: int | None) -> tuple[LFAState | None, str | None]:
    """Screen a candidate state and, if it survives, fit it."""
    scored = relabel(data, factors.steps, labels)
    try:
        design = build_afm_design(scored, learnsphere_compat=learnsphere_compat,
                                  recompute_opportunities=False)
    except ValueError:
        # identify() refuses a rank deficiency it cannot attribute. On a
        # machine-generated KC model that is a property of the move, not a
        # modelling mistake, so it refuses the move rather than the search.
        design = None

    if (reason := screen(scored, design, touched,
                         min_opportunities=min_opportunities,
                         separation=separation)) is not None:
        return None, reason

    fit = fit_afm(design, scored.y, method=method, max_fun=max_fun,
                  w0=_seed(design, parent) if warm_start else None,
                  warn_not_converged=False, warn_separated=False)
    by_block = fit.separated.by_block()
    state = LFAState(
        labels=labels, history=history, ll=fit.ll, aic=fit.aic, bic=fit.bic,
        n_kcs=len(scored.kc_names), n_params=fit.n_params,
        is_optimal=fit.is_optimal, n_separated=len(fit.separated),
        n_separated_kc=(len(by_block.get("kc_intercept", ()))
                        + len(by_block.get("kc_slope", ()))),
        iteration=iteration, weights=fit.weights,
        columns=tuple(design.columns),
    )
    return state, None


def lfa_search(data: StepData, factors: FactorMatrix, *,
               heuristic: str = "bic",
               max_iterations: int = MAX_ITERATIONS,
               patience: int = PATIENCE,
               min_opportunities: int = MIN_OPPORTUNITIES,
               screen_separation: bool = True,
               merge: bool = True,
               beam: int = BEAM,
               warm_start: bool = True,
               learnsphere_compat: bool = False,
               method: str = DEFAULT_METHOD,
               max_fun: int | None = None) -> LFAResult:
    """Search KC models by repeated splitting, ranked by AFM's BIC or AIC.

    Greedy best-first: each iteration expands the best *unexpanded* state,
    fits every surviving child, and returns the frontier ranked by
    ``heuristic``. The frontier is global, so the search can return to a
    shallower branch when a deep one stops paying — the behaviour the
    published method describes as "A* does not always go down".

    Deterministic by construction. Factors are iterated in ``factors``' order,
    skills in sorted order, and ties in the frontier break on
    ``(score, depth, history)``, so two runs on one input agree exactly.

    :param heuristic: one of :data:`HEURISTICS`.
    :param patience: stop after this many expansions that do not improve the
        incumbent; 0 to disable and run to ``max_iterations``.
    :param min_opportunities: evidence screen; see :func:`screen`.
    :param screen_separation: estimability screen; see :func:`screen`. Leaving
        this off reproduces the reference, including its selection of KC models
        whose parameters have no finite estimate.
    :param merge: offer lineage-undo merges. Only reachable states already
        evicted from ``beam`` can come back this way, so it does nothing when
        the frontier never fills.
    :param beam: unexpanded states retained.
    :param learnsphere_compat: score with the reference's conventions (student
        ridge, ``n_params = n_students + 2 * n_KCs``) rather than
        ``rank(X)``. Reproduction only; note that the two disagree about
        parameter counts *between siblings*, which is why they can order a
        frontier differently.
    """
    if heuristic not in HEURISTICS:
        raise ValueError(f"heuristic must be one of {HEURISTICS}, got {heuristic!r}")
    if beam < 1:
        raise ValueError(f"beam must be at least 1, got {beam!r}")

    missing = set(data.items) - set(factors.steps)
    if missing:
        raise ValueError(
            f"{len(missing)} step(s) in the data carry no row in the factor "
            f"matrix (first: {sorted(missing)[0]!r}). P must cover every step "
            "being fitted, or the searched labelling is undefined on some rows."
        )

    at = dict(zip(factors.factors, factors.members))
    root_labels = (ROOT_SKILL,) * len(factors.steps)
    root, reason = _evaluate(
        data, factors, root_labels, (), None, 0, touched=(ROOT_SKILL,),
        min_opportunities=0, separation=False,
        learnsphere_compat=learnsphere_compat, warm_start=False,
        method=method, max_fun=max_fun)
    if root is None:  # pragma: no cover - the All model is always identifiable
        raise ValueError(f"the root 'All' model was refused: {reason}")

    # A coefficient that separates the responses under the "All" model — a
    # student who never varied, say — separates under every refinement of it,
    # so no move can repair it and every state inherits a flat direction. Read
    # it once, at the root, and report it rather than letting it surface as an
    # unexplained ``separated=1`` on each state in turn.
    root_design = build_afm_design(relabel(data, factors.steps, root_labels),
                                   learnsphere_compat=learnsphere_compat,
                                   recompute_opportunities=False)
    persistent = tuple(
        column for column in root_design.separated(data.y).columns
        if not column.startswith(("kc_intercept:", "kc_slope:"))
    )

    root_key = _partition(root_labels)
    cache: dict[frozenset[frozenset[int]], LFAState] = {root_key: root}
    # The frontier holds *keys*, not states: an LFAState carries a coefficient
    # vector, so ``state in frontier`` would compare numpy arrays.
    frontier: set[frozenset[frozenset[int]]] = {root_key}
    expanded: set[frozenset[frozenset[int]]] = set()
    moves: list[str] = []
    reasons: list[str] = []
    trajectory: list[LFAState] = [root]
    incumbent = root
    stale = 0
    iteration = 0
    stopped = "exhausted"

    def rank(state: LFAState) -> tuple:
        return (state.score(heuristic), state.depth,
                tuple(str(m) for m in state.history))

    while iteration < max_iterations:
        frontier -= expanded
        if not frontier:
            stopped = "exhausted"
            break
        parent_key = min(frontier, key=lambda k: rank(cache[k]))
        parent = cache[parent_key]
        expanded.add(parent_key)
        iteration += 1

        tried: list[tuple[tuple[str, ...], tuple[Move, ...], tuple[str, ...]]] = []
        done = {(m.skill, m.factor) for m in parent.history}
        for skill in sorted(set(parent.labels)):
            for factor in factors.factors:
                if (skill, factor) in done:
                    continue
                child = split(parent.labels, factors.steps, skill, factor,
                              at[factor])
                if child is None:
                    continue
                move = Move("split", skill, factor)
                tried.append((child, (*parent.history, move),
                              (f"{skill}{SPLIT_SEP}{factor}", skill)))
        if merge:
            for i, move in enumerate(parent.history):
                if move.kind != "split":
                    continue
                history = parent.history[:i] + parent.history[i + 1:]
                try:
                    child = replay(history, factors)
                except ValueError:
                    continue
                tried.append((child, history, tuple(sorted(set(child)))))

        for child, history, touched in tried:
            key = _partition(child)
            if key in cache:
                # Already scored. Re-offering it is what makes merge useful:
                # the state may have been evicted from the beam since.
                if key not in expanded:
                    frontier.add(key)
                continue
            state, reason = _evaluate(
                data, factors, child, history, parent, iteration,
                touched=touched, min_opportunities=min_opportunities,
                separation=screen_separation,
                learnsphere_compat=learnsphere_compat, warm_start=warm_start,
                method=method, max_fun=max_fun)
            if state is None:
                moves.append(str(history[-1]) if history else "root")
                reasons.append(reason)
                continue
            cache[key] = state
            frontier.add(key)

        if len(frontier) > beam:
            ordered = sorted(frontier, key=lambda k: rank(cache[k]))
            frontier = set(ordered[:beam])

        best = min(cache.values(), key=rank)
        if rank(best) < rank(incumbent):
            incumbent = best
            stale = 0
        else:
            stale += 1
        trajectory.append(incumbent)
        if patience and stale >= patience:
            stopped = "no improvement"
            break
    else:
        stopped = "max iterations"

    states = tuple(sorted(cache.values(), key=rank))
    return LFAResult(
        root=root, states=states, trajectory=tuple(trajectory),
        rejected=Rejected(tuple(moves), tuple(reasons)), factors=factors,
        heuristic=heuristic, n_evaluated=len(cache), n_iterations=iteration,
        stopped=stopped, persistent_separation=persistent,
        learnsphere_compat=learnsphere_compat,
    )


# --------------------------------------------------------------------------
# Held-out validation of what the search chose
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LFAValidation:
    """Held-out scores for a search's top states, on identical folds.

    The search ranks by an information criterion because cross-validating
    inside the loop is unaffordable — the reference's own follow-up says so
    outright (Koedinger, McLaughlin & Stamper, EDM 2012: "too computationally
    expensive to run inside the LFA search... After the search is complete, we
    test the best models using cross validation"). This is that second step,
    and the interesting output is not a score but an *agreement*: whether the
    criterion that drove the search also picks the winner out of sample.

    Because every searched KC model covers the same rows, the folds are drawn
    once and shared, so the comparison is a within-fold paired contrast rather
    than a set of independently resampled means — see
    :func:`~leapfit.crossval.paired_cross_validate`.
    """

    folds: pd.DataFrame
    contrasts: pd.DataFrame
    searched: pd.DataFrame
    baseline: str
    heuristic: str
    scheme: str
    convention: str
    n_folds: int
    seeds: tuple[int, ...]

    def frame(self) -> pd.DataFrame:
        """One row per candidate: its in-sample criterion and its held-out RMSE."""
        per_seed = paired_scores(self.folds, self.convention)
        scores = per_seed.groupby("model", as_index=False).agg(
            cv_rmse=("rmse", "mean"),
            unseen_column_fraction=("unseen_column_fraction", "mean"),
            all_converged=("all_converged", "all"),
        )
        out = self.searched.merge(scores, on="model", how="left")
        out["search_rank"] = out[self.heuristic].rank(method="min").astype(int)
        out["cv_rank"] = out["cv_rmse"].rank(method="min").astype(int)
        return out.sort_values("search_rank", ignore_index=True)

    @property
    def winner(self) -> str:
        """The model with the lowest held-out RMSE."""
        frame = self.frame()
        return str(frame.loc[frame["cv_rmse"].idxmin(), "model"])

    @property
    def agrees(self) -> bool:
        """Whether the criterion's pick also wins out of sample."""
        frame = self.frame()
        return bool(frame.loc[frame[self.heuristic].idxmin(), "model"] == self.winner)

    def rank_correlation(self) -> float:
        """Spearman correlation between the search order and the held-out order.

        A description of how well the criterion tracked generalization on these
        candidates, not an inferential claim — with a handful of models and a
        handful of folds there is not enough to test.
        """
        frame = self.frame()
        if len(frame) < 3:
            return float("nan")
        return float(frame["search_rank"].corr(frame["cv_rank"], method="spearman"))

    def summary(self) -> str:
        frame = self.frame()
        best = frame.iloc[0]
        rho = self.rank_correlation()
        lines = [
            (f"Paired CV of {len(frame)} candidate(s): {self.scheme}, "
             f"{self.n_folds} folds x {len(self.seeds)} seed(s), "
             f"{self.convention} RMSE"),
            (f"  {self.heuristic.upper()} picked {best['model']!r} "
             f"(cv_rmse {best['cv_rmse']:.4f}); "
             f"held-out picks {self.winner!r}"),
            ("  the criterion and held-out RMSE agree on the winner"
             if self.agrees else
             "  they DISAGREE — the criterion's pick is not the best predictor"),
            f"  rank correlation {rho:.3f}" if rho == rho else
            "  rank correlation undefined for fewer than three candidates",
        ]
        if len(self.contrasts):
            lines.append(f"  paired contrasts against {self.baseline!r} "
                         "(negative mean_diff beats it):")
            lines.append("    " + self.contrasts.to_string(
                index=False).replace("\n", "\n    "))
        return "\n".join(lines)


def validate_top(result: LFAResult, data: StepData, *, n: int = 5,
                 include_root: bool = True,
                 extra: Mapping[str, StepData] | None = None,
                 baseline: str | None = None,
                 scheme: str = "item_blocked", n_folds: int = 3,
                 seeds=(0,), convention: str = "pooled",
                 method: str = DEFAULT_METHOD, max_fun: int | None = None,
                 n_jobs: int | None = 1) -> LFAValidation:
    """Cross-validate a search's top ``n`` states on identical folds.

    Candidates are named ``"rank1"`` .. ``"rankN"`` in search order, plus
    ``"root"`` for the "All" model the search began from, plus whatever is
    passed in ``extra`` under its own name. Scoring uses the mode the search
    used, taken from ``result``, so the in-sample and held-out columns of
    :meth:`LFAValidation.frame` describe the same models.

    Opportunity counts are recomputed for **every** candidate, including any in
    ``extra`` that ships its own ``Opportunity`` column. The searched models
    have no such column, so recomputing throughout is what makes the
    comparison a comparison; call
    :func:`~leapfit.crossval.cross_validate` directly to reproduce a published
    number for an authored model under its own convention.

    :param n: how many of the ranked states to score. The frontier can hold
        hundreds; the point of the exercise is the top handful.
    :param extra: hand-authored KC models to score alongside — the comparison
        that says whether searching beat the labelling you already had. Each
        must cover the same rows as ``data``.
    :param baseline: what the contrasts difference against. Defaults to
        ``"root"`` when it is included, else the search's own pick, so the
        headline reads "did refining help out of sample".
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n!r}")

    steps = result.factors.steps
    compat = result.learnsphere_compat
    designs: dict[str, Design] = {}
    rows: list[dict] = []

    def record(name: str, state: LFAState | None, design: Design,
               scored: StepData) -> None:
        designs[name] = design
        if state is None:
            fit = fit_afm(design, scored.y, method=method, max_fun=max_fun,
                          warn_not_converged=False, warn_separated=False)
            state = LFAState(
                labels=(), history=(), ll=fit.ll, aic=fit.aic, bic=fit.bic,
                n_kcs=len(scored.kc_names), n_params=fit.n_params,
                is_optimal=fit.is_optimal, n_separated=len(fit.separated))
        rows.append({
            "model": name, "depth": state.depth, "n_kcs": state.n_kcs,
            "n_params": state.n_params, "log_likelihood": state.ll,
            "aic": state.aic, "bic": state.bic,
            "is_optimal": state.is_optimal,
            "history": " | ".join(str(m) for m in state.history),
        })

    def design_for(state: LFAState) -> tuple[Design, StepData]:
        scored = relabel(data, steps, state.labels)
        return build_afm_design(scored, learnsphere_compat=compat,
                                recompute_opportunities=False), scored

    if include_root:
        design, scored = design_for(result.root)
        record("root", result.root, design, scored)
    for rank, state in enumerate(result.states[:n], start=1):
        if include_root and state.labels == result.root.labels:
            continue
        design, scored = design_for(state)
        record(f"rank{rank}", state, design, scored)
    for name, authored in (extra or {}).items():
        if list(authored.items) != list(data.items):
            raise ValueError(
                f"extra model {name!r} covers a different row set from the "
                "search data, so folds cannot be shared and the comparison "
                "would not be paired."
            )
        design = build_afm_design(authored, learnsphere_compat=compat,
                                  recompute_opportunities=True)
        record(name, None, design, authored)

    if baseline is None:
        baseline = "root" if include_root else next(iter(designs))
    if baseline not in designs:
        raise KeyError(f"baseline {baseline!r} not among {list(designs)}")

    folds = paired_cross_validate(
        designs, data, scheme=scheme, n_folds=n_folds, seeds=seeds,
        convention=convention, method=method, max_fun=max_fun, n_jobs=n_jobs)
    return LFAValidation(
        folds=folds, contrasts=paired_contrasts(folds, baseline),
        searched=pd.DataFrame(rows), baseline=baseline,
        heuristic=result.heuristic, scheme=scheme, convention=convention,
        n_folds=n_folds, seeds=tuple(seeds),
    )
