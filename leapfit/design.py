"""Design matrices as labelled, individually-penalized blocks.

The additive models in this package are all a concatenation of column blocks.
AFM's is three (:func:`leapfit.afm.build_afm_design`)::

    logit P(Y_ij = 1) = theta_i + sum_k q_jk * beta_k + sum_k q_jk * gamma_k * T_ik
                        \\_____/   \\______________/   \\___________________________/
                         student        kc_intercept              kc_slope

and PFA's replaces the last block with prior-success and prior-failure counts.
Nothing in this module knows which of those it is holding.

Blocks exist because a plain matrix cannot describe the model. LearnSphere
treats each group differently — students ridge-penalized at 1.0, KC parameters
unpenalized, slopes sometimes bounded below at zero — so a :class:`Block`
carries its columns *together with* the per-column penalty and bounds that
belong to them, and :class:`Design` concatenates blocks while keeping the
coefficient labels attached.

This is deliberately the extension point for everything downstream:

* **A new model family** is a new set of blocks and nothing else; the solver,
  the identification pass, and the separation check all come for free.
* **A new predictor over practice history** — spacing gaps, similarity-weighted
  practice, anything accumulated over :meth:`leapfit.data.StepData.practice_order`
  — is one more :class:`Block` (see :func:`accumulator_block`).
* **Hierarchical shrinkage** is a reparameterization, not new machinery:
  write ``beta_k = beta_parent(k) + b_k`` as an unpenalized parent block plus
  a ridge-penalized deviation block, and the ridge weight *is* the prior
  precision ``1/sigma_b^2``. Estimating ``sigma_b^2`` needs an outer loop, but
  the inner fit is this same solver.
"""

from __future__ import annotations

from collections.abc import Sequence, Sized
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph


@dataclass(frozen=True)
class Block:
    """One labelled group of design columns with its penalty and bounds."""

    name: str
    matrix: sparse.csr_matrix
    columns: list[str]
    l2: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        n_cols = self.matrix.shape[1]
        for attr in ("columns", "l2", "lower", "upper"):
            if len(getattr(self, attr)) != n_cols:
                raise ValueError(
                    f"Block '{self.name}': {attr} has {len(getattr(self, attr))} "
                    f"entries for {n_cols} columns"
                )

    @classmethod
    def build(cls, name: str, matrix, columns: list[str], *, l2: float = 0.0,
              lower: float = -np.inf, upper: float = np.inf) -> Block:
        n = len(columns)
        return cls(
            name=name,
            matrix=sparse.csr_matrix(matrix),
            columns=list(columns),
            l2=np.full(n, float(l2)),
            lower=np.full(n, float(lower)),
            upper=np.full(n, float(upper)),
        )

    def keep(self, mask: np.ndarray) -> Block:
        """Column subset, preserving labels, penalty, and bounds."""
        idx = np.flatnonzero(mask)
        return Block(self.name, self.matrix[:, idx],
                     [self.columns[i] for i in idx],
                     self.l2[idx], self.lower[idx], self.upper[idx])

    def duplicate_columns(self) -> list[tuple[int, int]]:
        """``(j, first)`` for every column that repeats an earlier one exactly.

        Equality is bitwise on the stored pattern and values, so a pair is
        reported only when the two columns *are* the same vector — no
        tolerance, nothing to tune. Near-duplicates are left in place for
        :meth:`Design.rank` to catch, because dropping one would change the
        fit rather than only its parameterization.

        Exact repetition is what real KC models produce: two KCs tagging the
        same steps share an intercept column, and their opportunity counts,
        accumulated over those same rows, coincide too. All-zero columns are
        skipped — those are dead, a separate and better-named reason.
        """
        M = self.matrix.tocsc()
        seen: dict[bytes, int] = {}
        out = []
        for j in range(M.shape[1]):
            lo, hi = M.indptr[j], M.indptr[j + 1]
            if lo == hi:
                continue
            key = M.indices[lo:hi].tobytes() + M.data[lo:hi].tobytes()
            if (first := seen.get(key)) is None:
                seen[key] = j
            else:
                out.append((j, first))
        return out


@dataclass(frozen=True)
class Aliased:
    """Columns removed from a design because they are not estimable.

    ``columns`` are fully-qualified (``"kc_slope:KC-17"``) and ``reasons``
    parallel them. A dropped column carries no information: it is either
    identically zero or an exact linear combination of the columns kept, so
    removing it leaves every fitted value unchanged while making the
    parameter count honest. This is what R's ``glm`` does when it reports
    coefficients as ``NA`` "because of singularities" and uses the rank for
    its degrees of freedom.
    """

    columns: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.columns)

    def by_block(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for full in self.columns:
            block, _, col = full.partition(":")
            out.setdefault(block, []).append(col)
        return out

    def summary(self) -> str:
        if not self.columns:
            return "no aliased columns"
        counts = {b: len(v) for b, v in self.by_block().items()}
        detail = ", ".join(f"{n} from {b}" for b, n in sorted(counts.items()))
        return f"{len(self)} aliased column(s) dropped ({detail})"


@dataclass(frozen=True)
class Separated:
    """Columns whose maximum-likelihood estimate runs off to infinity.

    Distinct from :class:`Aliased`, and the difference matters. An aliased
    column carries *no* information, so dropping it costs nothing. A separated
    column carries the strongest information there is — every observation it
    touches came out the same way — and precisely for that reason no finite
    coefficient maximizes the likelihood. The optimizer stops somewhere out on
    the plateau and returns whatever it reached, so the printed estimate is an
    artefact of the evaluation budget rather than a property of the data.

    ``directions`` parallel ``columns``: ``+1`` where the estimate diverges to
    ``+inf``, ``-1`` to ``-inf``.
    """

    columns: tuple[str, ...] = ()
    directions: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.columns)

    def by_block(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for full in self.columns:
            block, _, col = full.partition(":")
            out.setdefault(block, []).append(col)
        return out

    def summary(self) -> str:
        if not self.columns:
            return "no separated columns"
        counts = {b: len(v) for b, v in self.by_block().items()}
        detail = ", ".join(f"{n} from {b}" for n, b in
                           ((n, b) for b, n in sorted(counts.items())))
        up = sum(d > 0 for d in self.directions)
        return (f"{len(self)} column(s) with no finite MLE ({detail}); "
                f"{up} diverge to +inf, {len(self) - up} to -inf")


@dataclass(frozen=True)
class Design:
    """A concatenation of blocks, with row subsetting for cross-validation."""

    blocks: tuple[Block, ...]
    aliased: Aliased = Aliased()

    def __post_init__(self) -> None:
        heights = {b.matrix.shape[0] for b in self.blocks}
        if len(heights) > 1:
            raise ValueError(f"Blocks disagree on row count: {heights}")
        names = [b.name for b in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate block names: {names}")

    @property
    def n_obs(self) -> int:
        return self.blocks[0].matrix.shape[0]

    @property
    def n_params(self) -> int:
        """Free coefficients — the ``nPars`` that AIC and BIC penalize.

        After :meth:`identify` this equals ``rank(X)``: every column left is
        estimable. Without it, this is every column, which is LearnSphere's
        convention and overcounts by the rank deficiency — 959 phantom
        parameters for E-learning-22's ``Unique-step`` model, or 25% of its
        BIC penalty.
        """
        return sum(b.matrix.shape[1] for b in self.blocks)

    @property
    def matrix(self) -> sparse.csr_matrix:
        return sparse.hstack([b.matrix for b in self.blocks], format="csr")

    @property
    def l2(self) -> np.ndarray:
        return np.concatenate([b.l2 for b in self.blocks])

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        lo = np.concatenate([b.lower for b in self.blocks])
        hi = np.concatenate([b.upper for b in self.blocks])
        return [(None if np.isneginf(a) else a, None if np.isposinf(b) else b)
                for a, b in zip(lo, hi)]

    @property
    def columns(self) -> list[str]:
        return [f"{b.name}:{c}" for b in self.blocks for c in b.columns]

    def slices(self) -> dict[str, slice]:
        out, start = {}, 0
        for b in self.blocks:
            width = b.matrix.shape[1]
            out[b.name] = slice(start, start + width)
            start += width
        return out

    def take(self, rows: np.ndarray) -> Design:
        """Row subset, preserving labels, penalty, bounds, and identification.

        Identification is deliberately *not* recomputed per subset. A
        cross-validation fold can make a column locally constant without that
        column being unidentifiable in the model being scored; recomputing
        would change the parameter count between folds. Columns absent from a
        training fold simply keep their zero start (see crossval).
        """
        return Design(tuple(
            Block(b.name, b.matrix[rows], b.columns, b.l2, b.lower, b.upper)
            for b in self.blocks
        ), self.aliased)

    def with_blocks(self, *extra: Block) -> Design:
        """Append blocks — the hook for accumulator terms and hierarchies.

        The existing aliasing record carries over, but the new columns are
        unchecked: call :meth:`identify` again afterwards. An accumulator or
        hierarchical-parent block can easily be collinear with what is already
        there, and that is exactly the failure this machinery exists to catch.
        """
        return Design(self.blocks + tuple(extra), self.aliased)

    def rank(self, tol: float | None = None) -> int:
        """Numerical rank of the design, via the column-scaled Gram matrix.

        Scaling to unit column norm first is not optional. An opportunity
        column reaches into the thousands while the indicator columns are 0/1,
        so the unscaled Gram spans ~10 orders of magnitude and a relative
        eigenvalue threshold misreads well-identified models as deficient —
        measured on E-learning-22's ``Single-KC``, which reads as rank 37 of 41
        unscaled and the correct 40 of 41 scaled.

        The Gram is ``p x p``, so this stays cheap where a dense ``n x p``
        factorization would not: 63 MB at ``p = 2811`` against 948 MB dense.
        """
        X = self.matrix
        norms = sparse.linalg.norm(X, axis=0)
        norms[norms == 0.0] = 1.0
        D = sparse.diags(1.0 / norms)
        gram = ((X @ D).T @ (X @ D)).toarray()
        ev = np.linalg.eigvalsh(gram)
        if tol is None:
            tol = max(self.n_obs, gram.shape[0]) * np.finfo(float).eps * max(ev.max(), 0.0)
        return int((ev > tol).sum())

    def separated(self, y) -> Separated:
        """Columns whose coefficient has no finite maximizer, given ``y``.

        A sign-homogeneous, unpenalized column whose active rows are *all*
        successes can be pushed up forever: raising its coefficient strictly
        increases the fitted probability of exactly those rows, all of which
        want a higher probability, and touches nothing else. The likelihood is
        therefore strictly increasing in that coefficient with no maximizer —
        an exact argument, not a numerical threshold. All-failure columns are
        the mirror image.

        Three things make a column safe and are checked: a nonzero ridge
        (which supplies the missing curvature), a finite bound in the diverging
        direction (the estimate rests on the bound instead), and mixed signs
        within the column (where the argument does not apply).

        This is a *lower bound* on the separated set. Detecting every case —
        including groups of columns that separate the data only in combination
        — is a linear-programming problem; this catches the single-column form,
        which is what fine-grained KC models actually produce.
        """
        y = np.asarray(y, dtype=float)
        if y.shape[0] != self.n_obs:
            raise ValueError(f"{self.n_obs} design rows but {y.shape[0]} responses")

        X = self.matrix
        active = (X != 0)
        n_active = np.asarray(active.sum(axis=0)).ravel()
        n_success = np.asarray(active.astype(float).T @ y).ravel()

        nonneg = np.asarray((X < 0).sum(axis=0)).ravel() == 0
        nonpos = np.asarray((X > 0).sum(axis=0)).ravel() == 0

        lower = np.concatenate([b.lower for b in self.blocks])
        upper = np.concatenate([b.upper for b in self.blocks])
        can_rise, can_fall = np.isposinf(upper), np.isneginf(lower)

        live = (n_active > 0) & (self.l2 <= 0.0)
        all_success = live & (n_success == n_active)
        all_failure = live & (n_success == 0.0)

        rises = ((all_success & nonneg) | (all_failure & nonpos)) & can_rise
        falls = ((all_failure & nonneg) | (all_success & nonpos)) & can_fall

        idx = np.flatnonzero(rises | falls)
        columns = self.columns
        return Separated(
            tuple(columns[j] for j in idx),
            tuple(1 if rises[j] else -1 for j in idx),
        )

    def identify(self, *, prefer_drop: str = "student", check: bool = True) -> Design:
        """Drop columns that are not estimable, so ``n_params == rank(X)``.

        Three sources of aliasing are removed, all exactly rather than
        numerically:

        1. **Dead columns.** A column of all zeros carries no information and
           its coefficient is arbitrary. In AFM these are KCs that no student
           ever practises twice, so ``T`` is always 0 and the learning rate is
           not estimable at all. E-learning-22's ``Unique-step`` model has 958.
        2. **Duplicate columns within a block.** Two KCs that tag exactly the
           same steps have identical intercept columns, and — because
           opportunity counts are accumulated over those same rows — identical
           slope columns too. Only one of each group is estimable. Real KC
           models do this: FoundationalASSIST's ``CCSS`` labelling has three
           such groups (``{3.OA.B.5, 5.OA.A.1}``, ``{4.NBT.A.1, 4.NF.B.4b,
           4.OA.A.1}``, ``{5.NF.B.7b, 5.NF.B.7c}``), which is exactly its rank
           deficiency of 7. The first column of each group keeps the estimate
           and the rest are dropped, so they report ``NaN`` in ``kc_values``
           rather than a number that is really some other KC's.

           Deliberately *within* a block only. A whole block that duplicates
           another — an accumulator or hierarchical-parent term collinear with
           what is already there, or one factor entered twice under two names —
           is a modelling error, not a property of the data, and still raises
           under ``check``. A factor *nested* in another, items within KCs, is
           a property of the data, and is identified under point 3 — provided
           the coarser factor is declared first. Declared after the levels it
           groups, it is the hierarchical parent just described: the finer
           block already spans it, point 3 would drop it whole, and it is
           refused instead.
        3. **Sum redundancies, pair by pair.** A block whose rows all sum to
           the same positive constant spans the all-ones direction: its
           columns add up to ``c * 1``. The student block always does (one
           student per row), and so does a KC-intercept block whenever every
           row carries the same number of KCs. Two such blocks are therefore
           linearly dependent — add a constant to every student, subtract it
           from every KC intercept, and no prediction moves.

           The dependency lives on the two blocks' *own* graph, and on each of
           its connected components separately. Each component shifts
           independently, so cohorts that share no material carry one apiece:
           on the spacing-exp2 export the ten components are its courses, and
           one reference student is dropped per course. A component where one
           of the two blocks' row sums varies — rows that do not all carry the
           same number of KCs, say — has no redundancy to break and keeps
           every column. And no third block enters a pair's graph: a
           decayed-history feature touches every row without relating two
           cohorts to each other, and a nested factor — items within KCs —
           splits its pair's graph into one component per KC however well
           connected the students make everything else.

           With three or more blocks the pairwise dependencies overlap —
           three crossed blocks carry two, not three — so the columns to drop
           are chosen by exact elimination over them rather than one per
           dependency: ``prefer_drop`` first, then the latest-declared blocks,
           each from its last level back, so that the earliest block in the
           design keeps every level. Exactly as many columns go as the
           dependencies span, and with two blocks this is one per component.
           A block other than ``prefer_drop`` that this would take whole is
           the collinear block of point 2, and is refused rather than
           dropped. ``prefer_drop`` alone may go whole: a cohort of one
           student gives up its only student, as it always has.

        A student is dropped rather than a KC because the KC intercepts are
        the reported output — learning curves, difficulty tables, low-slope
        screens — and a KC missing from that table would be worse than an
        arbitrary reference student. The same reasoning orders the general
        case: whichever block a spec declares first is the one that keeps every
        level. Use :meth:`~leapfit.afm.AFMFit.centred_students` to move the
        fit to the reference-free sum-to-zero point afterwards. Note that with
        several components that recentring is one *global* shift, so intercept
        levels stay comparable only within a component — nothing in the data
        relates two cohorts that never met the same material.

        :param check: verify numerically that the result is full rank, and
            raise if it is not, or if it would drop a whole block. Leave this
            on: it is the guard that catches aliasing introduced by blocks
            added later.
        """
        keep = {b.name: np.ones(b.matrix.shape[1], dtype=bool) for b in self.blocks}
        dropped, reasons = [], []

        for b in self.blocks:
            nnz = np.asarray((b.matrix != 0).sum(axis=0)).ravel()
            for j in np.flatnonzero(nnz == 0):
                keep[b.name][j] = False
                dropped.append(f"{b.name}:{b.columns[j]}")
                reasons.append("column is identically zero (not estimable)")

        for b in self.blocks:
            for j, first in b.duplicate_columns():
                keep[b.name][j] = False
                dropped.append(f"{b.name}:{b.columns[j]}")
                reasons.append(f"duplicate of {b.name}:{b.columns[first]}")

        reduced = Design(
            tuple(b.keep(keep[b.name]) for b in self.blocks),
            Aliased(tuple(self.aliased.columns) + tuple(dropped),
                    tuple(self.aliased.reasons) + tuple(reasons)),
        )
        identified = reduced._drop_reference_levels(prefer_drop, check=check)

        if check:
            r = identified.rank()
            if r != identified.n_params:
                if repeated := reduced._coinciding_pairs():
                    named = "; ".join(f"{a} and {b}" for a, b in repeated)
                    raise ValueError(
                        f"Design still rank-deficient after identification: "
                        f"{identified.n_params} columns, rank {r}. {named} partition "
                        f"the rows identically — one factor under two names — and a "
                        f"repeated factor is refused rather than one copy of it "
                        f"silently dropped. Keep one of them."
                    )
                raise ValueError(
                    f"Design still rank-deficient after identification: "
                    f"{identified.n_params} columns, rank {r}. Either a block added to "
                    f"this design is collinear with the others, or the KC model "
                    f"carries a dependency this pass does not model exactly — a KC "
                    f"that tags every row of its component, say. Drop or "
                    f"reparameterize the offending columns before fitting, or "
                    f"AIC/BIC will count parameters that do not exist."
                )
        return identified

    def _drop_reference_levels(self, prefer_drop: str, *, check: bool = True) -> Design:
        """Break every sum redundancy, on the columns that survive.

        Deliberately decided *after* dead and duplicate columns are gone: a row
        that carried two KCs carries one once a duplicate of the pair is
        dropped, and that is when the sum redundancy comes into being. Asking
        the question of the original matrix would miss it.

        Each column is read as its vector of coefficients across the
        redundancies, and the columns are visited in drop order. A column is
        dropped when its vector is not already spanned by those of the columns
        visited before it — decided exactly, in rationals. The dropped vectors
        then span every column's, so no combination of the redundancies
        survives on the columns kept, and exactly as many columns go as there
        are independent redundancies. Each drop is reported against the
        smallest redundancy it takes part in, the one it most specifically
        stands for.

        Under ``check``, a block other than ``prefer_drop`` that would lose
        every column raises instead, naming the blocks those drops are
        reported against: they span it, so it is not a factor with a reference
        level but a block that adds nothing — a parent declared after the
        levels it groups.
        """
        redundancies = self._sum_redundancies()
        if not redundancies:
            return self

        # Column (block, j) -> {redundancy index: its coefficient there}.
        vectors: dict[tuple[str, int], dict[int, Fraction]] = {}
        for k, redundancy in enumerate(redundancies):
            for name, columns, weight in zip(redundancy.blocks, redundancy.columns,
                                             redundancy.weights):
                for j in columns:
                    vectors.setdefault((name, int(j)), {})[k] = Fraction(weight)
        support = [sum(map(len, r.columns)) for r in redundancies]

        by_name = {b.name: b for b in self.blocks}
        involved = [b.name for b in self.blocks
                    if any(b.name in r.blocks for r in redundancies)]
        basis: dict[int, dict[int, Fraction]] = {}
        seen: set[frozenset] = set()
        drops = []
        for name in self._drop_order(involved, prefer_drop):
            for j in range(by_name[name].matrix.shape[1] - 1, -1, -1):
                vector = vectors.get((name, j))
                if not vector or (key := frozenset(vector.items())) in seen:
                    continue  # in no redundancy, or identical to a column visited
                seen.add(key)
                if residual := _reduce(vector, basis):
                    pivot = min(residual)
                    basis[pivot] = {k: v / residual[pivot] for k, v in residual.items()}
                    drops.append((min(vector, key=lambda k: (support[k], k)), name, j))

        if check:
            spanned_by: dict[str, set[str]] = {}
            for k, name, _ in drops:
                spanned_by.setdefault(name, set()).update(redundancies[k].blocks)
            for name, blocks in spanned_by.items():
                taken = sum(drop[1] == name for drop in drops)
                if name != prefer_drop and taken == by_name[name].matrix.shape[1]:
                    raise ValueError(_whole_block_refusal(
                        name, [b for b in by_name if b in blocks and b != name]))

        keep = {b.name: np.ones(b.matrix.shape[1], dtype=bool) for b in self.blocks}
        dropped, reasons = [], []
        for k, name, j in sorted(drops, key=lambda drop: drop[0]):
            keep[name][j] = False
            dropped.append(f"{name}:{by_name[name].columns[j]}")
            redundancy = redundancies[k]
            reasons.append(_reference_reason(list(redundancy.blocks), redundancy.label,
                                             redundancy.n_components))
        return Design(
            tuple(b.keep(keep[b.name]) if not keep[b.name].all() else b
                  for b in self.blocks),
            Aliased(tuple(self.aliased.columns) + tuple(dropped),
                    tuple(self.aliased.reasons) + tuple(reasons)),
        )

    @staticmethod
    def _drop_order(names: list[str], prefer_drop: str) -> list[str]:
        """Which blocks give up a level first.

        ``prefer_drop`` leads where it applies, and the rest follow
        latest-declared first, so the block a specification names earliest is
        the one left whole. For an AFM design that is exactly the old rule —
        drop a student, keep every KC — and it generalizes the reason for it
        rather than the two block names it was written in.
        """
        ordered = [n for n in names if n == prefer_drop]
        return ordered + [n for n in reversed(names) if n != prefer_drop]

    def _row_sums(self, name: str) -> np.ndarray | None:
        b = next((x for x in self.blocks if x.name == name), None)
        return None if b is None else np.asarray(b.matrix.sum(axis=1)).ravel()

    def kc_per_row(self) -> float | None:
        """The constant number of KCs per row, or None if it varies."""
        sums = self._row_sums("kc_intercept")
        if sums is None or sums.size == 0 or not np.allclose(sums, sums[0]):
            return None
        return float(sums[0]) if sums[0] > 0 else None

    def _covering_blocks(self) -> list[str]:
        """Blocks that touch every row, in design order.

        The candidates for a sum redundancy, and the blocks
        :meth:`row_components` builds its graph over by default. A block that
        leaves some row at zero cannot span the all-ones direction, so it can
        be neither.
        """
        out = []
        for b in self.blocks:
            sums = self._row_sums(b.name)
            if sums is not None and sums.size and bool(np.all(sums > 0)):
                out.append(b.name)
        return out

    def _has_sum_redundancy(self) -> bool:
        """Whether any two blocks span the same direction over some rows.

        Every row carries exactly one student, so the student columns sum to
        the all-ones vector. If every row also carries the *same* number ``m``
        of KCs, the KC-intercept columns sum to ``m * 1``, and the two blocks
        are linearly dependent whatever ``m`` is — not only for the usual
        one-KC-per-row partition. Nothing here is specific to those two blocks;
        see :meth:`_sum_redundancies`.
        """
        return bool(self._sum_redundancies())

    def row_components(self, names: Sequence[str] | None = None) -> np.ndarray:
        """Component label per row, from the graph over some blocks' levels.

        Two rows land in the same component when a chain of shared levels
        connects them. ``names`` picks the blocks, by default every one that
        covers every row — shared students, shared KCs, shared anything — so
        that several components mean the export holds cohorts that never met
        the same material. :meth:`identify` asks this of one *pair* of blocks
        at a time instead, because a sum redundancy between two blocks lives
        on their own graph: a third block that touches every row can join
        components that no redundancy joins. All-zero when fewer than two
        blocks are given, since then there is nothing for components to
        localize.
        """
        names = self._covering_blocks() if names is None else list(names)
        if len(names) < 2:
            return np.zeros(self.n_obs, dtype=np.int64)

        by_name = {b.name: b for b in self.blocks}
        incidence = sparse.hstack([by_name[n].matrix for n in names], format="csr")
        incidence = (incidence != 0).astype(np.int8)
        n_rows = incidence.shape[0]
        graph = sparse.bmat([[None, incidence], [incidence.T, None]], format="csr")
        _, labels = csgraph.connected_components(graph, directed=False)
        # Renumber consecutively: a level nobody touches is its own graph
        # component, and would otherwise leave a gap in the labels.
        return np.unique(labels[:n_rows], return_inverse=True)[1].astype(np.int64)

    def _column_components(self, name: str, rows: np.ndarray) -> np.ndarray:
        """Component label per column of a block: the label of any row it touches.

        A column touches one component only — that is what a component is — so
        the first stored row decides it. Empty columns get ``-1`` and match no
        component.
        """
        M = next(b for b in self.blocks if b.name == name).matrix.tocsc()
        starts, ends = M.indptr[:-1], M.indptr[1:]
        out = np.full(M.shape[1], -1, dtype=np.int64)
        occupied = starts < ends
        out[occupied] = rows[M.indices[starts[occupied]]]
        return out

    def _pair_graphs(self):
        """Each pair of covering blocks with its own graph, in design order.

        Yields the two names, the component of every row, how many components
        there are, and each block's columns grouped by component.
        """
        names = self._covering_blocks()
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                rows = self.row_components((first, second))
                if rows.size:
                    yield (first, second, rows, int(rows.max()) + 1,
                           (_members(self._column_components(first, rows)),
                            _members(self._column_components(second, rows))))

    def _sum_redundancies(self) -> list[_SumRedundancy]:
        """Every sum redundancy between two covering blocks: pairs in design
        order, and within a pair its graph's components in label order.

        A component carries one wherever both blocks' row sums are constant on
        it, and none where either varies — a block can be constant within one
        cohort and not across the export, so the question is asked per
        component rather than of the block. Rows and columns are grouped by one
        sort rather than one scan per component, so this stays linear-ish
        however many components there are — a design where no two students
        share an item has as many components as students.

        Two blocks that partition the rows identically are left out: that is
        one factor under two names, a mistake in the specification rather than
        a property of the data, and :meth:`identify` refuses it rather than
        silently dropping one of them whole.
        """
        sums = {name: self._row_sums(name) for name in self._covering_blocks()}
        out = []
        for first, second, rows, n_components, columns in self._pair_graphs():
            if _coincide(n_components, columns):
                continue
            for label, group in _members(rows).items():
                a, b = sums[first][group], sums[second][group]
                if np.allclose(a, a[0]) and np.allclose(b, b[0]):
                    empty = np.array([], dtype=np.int64)
                    out.append(_SumRedundancy(
                        (first, second), label, n_components,
                        (columns[0].get(label, empty), columns[1].get(label, empty)),
                        (float(b[0]), -float(a[0]))))
        return out

    def _coinciding_pairs(self) -> list[tuple[str, str]]:
        """Pairs of covering blocks that partition the rows identically."""
        return [(first, second) for first, second, _, n_components, columns
                in self._pair_graphs() if _coincide(n_components, columns)]

    def recentring_is_valid(self) -> bool:
        """Whether shifting students into KC intercepts leaves predictions fixed.

        Adding ``c`` to every student and subtracting it from every KC
        intercept cancels only if each row picks up ``+c`` exactly once from
        the student side and ``-c`` exactly once from the KC side. The student
        side always holds (one student per row, and a dropped reference level
        counts as a student whose effect is zero). The KC side needs exactly
        one KC per row.
        """
        return self.kc_per_row() == 1.0


def _reference_reason(names: list[str], label: int, n_components: int) -> str:
    """Why a reference level was dropped, naming the blocks it stood between.

    The student/KC wording is preserved exactly: it is the case every other
    docstring, changelog entry and design note in this package refers to, and
    a reader following one of those to ``Aliased.reasons`` should find the
    words they were sent to look for.
    """
    kind = ("student/KC sum redundancy" if set(names) == {"student", "kc_intercept"}
            else "sum redundancy across " + ", ".join(names))
    where = "" if n_components == 1 else f", component {label + 1} of {n_components}"
    return f"reference level ({kind}{where})"


def _whole_block_refusal(name: str, spanning: list[str]) -> str:
    """Why a block that identification would drop whole is refused instead."""
    return (
        f"{name} adds nothing to this design: every column of it lies in the span "
        f"of {' and '.join(spanning)}, the way a parent block's columns are sums of "
        f"the levels it groups, so identification would drop it whole. A block "
        f"collinear with what is already there is refused rather than silently "
        f"dropped. Remove it; or, to keep it, declare it before the finer blocks, "
        f"which then give up one reference level per level of {name} instead; and "
        f"where a ridge is what identifies the hierarchy, leave identification out."
    )


@dataclass(frozen=True)
class _SumRedundancy:
    """Two blocks that span the same direction over one component of their graph.

    Where the two blocks' rows sum to constants ``c1`` and ``c2`` there, adding
    ``c2`` to each of the first block's levels in the component and taking
    ``c1`` from each of the second's leaves every prediction where it was: that
    null vector is ``weights`` on ``columns``.
    """

    blocks: tuple[str, str]
    label: int                              # the component, in the pair's own graph
    n_components: int                       # how many components that graph has
    columns: tuple[np.ndarray, np.ndarray]  # each block's columns in the component
    weights: tuple[float, float]


def _members(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Indices grouped by their label, by one sort; negative labels left out."""
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    return {int(labels[g[0]]): g for g in groups if g.size and labels[g[0]] >= 0}


def _coincide(n_components: int, columns: tuple[dict, dict]) -> bool:
    """Whether a pair's graph pairs every level of one block with exactly one of
    the other's — the same partition of the rows, under two sets of names.

    One component does not count: a single student and a single KC are two
    constant columns, and that is the ordinary reference level, not a
    repeated factor.
    """
    return n_components > 1 and all(
        len(columns[0].get(label, ())) == len(columns[1].get(label, ())) == 1
        for label in range(n_components))


def _reduce(vector: dict[int, Fraction],
            basis: dict[int, dict[int, Fraction]]) -> dict[int, Fraction]:
    """What is left of ``vector`` once the span of ``basis`` is taken out.

    ``basis`` is in echelon form: each vector is keyed by its smallest index,
    where it is 1. Eliminating the smallest index the two share therefore
    never reintroduces one already eliminated, and each step moves strictly
    rightwards.
    """
    residual = dict(vector)
    while hits := [k for k in residual if k in basis]:
        k = min(hits)
        coefficient = residual[k]
        for m, value in basis[k].items():
            updated = residual.get(m, 0) - coefficient * value
            if updated:
                residual[m] = updated
            else:
                residual.pop(m, None)
    return residual


def accumulator_block(data: Sized, values: np.ndarray, *,
                      name: str = "accumulator",
                      columns: list[str] | None = None,
                      l2: float = 0.0) -> Block:
    """Wrap per-observation accumulator columns as a design block.

    An *accumulator* is any predictor computed over a student's practice
    history — prior success/failure counts (PFA's terms), spacing gaps,
    similarity-weighted practice. Build it over
    :meth:`leapfit.data.StepData.practice_order` so it agrees with the
    opportunity counts about what "before" means, then attach it with
    :meth:`Design.with_blocks` and re-run :meth:`Design.identify`, which will
    refuse the design if the new columns are collinear with what is already
    there.

    ``values`` is ``(n_obs,)`` or ``(n_obs, p)`` — one column per accumulator
    that gets its own coefficient. ``data`` is only consulted for its length,
    so passing the :class:`~leapfit.data.StepData` the design was built from
    keeps the row-count check honest.
    """
    acc = np.asarray(values, dtype=float)
    if acc.ndim == 1:
        acc = acc[:, None]
    if acc.shape[0] != len(data):
        raise ValueError(f"Expected {len(data)} rows, got {acc.shape[0]}")
    labels = columns or ([name] if acc.shape[1] == 1
                         else [f"{name}_{i}" for i in range(acc.shape[1])])
    return Block.build(name, acc, labels, l2=l2)


def coefficient_frame(design: Design, weights: np.ndarray) -> pd.DataFrame:
    """Fitted weights as a tidy table of (block, column, estimate)."""
    return pd.DataFrame({
        "block": [b.name for b in design.blocks for _ in b.columns],
        "column": [c for b in design.blocks for c in b.columns],
        "estimate": weights,
    })
