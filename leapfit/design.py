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

from collections.abc import Sized
from dataclasses import dataclass

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
           what is already there — is a modelling error, not a property of the
           data, and still raises under ``check``.
        3. **Sum redundancies between partitioning blocks, per component.**
           A block whose rows all sum to the same positive constant spans the
           all-ones direction: its columns add up to ``c * 1``. The student
           block always does (one student per row), and so does a KC-intercept
           block whenever every row carries the same number of KCs. Two such
           blocks are therefore linearly dependent — add a constant to every
           student, subtract it from every KC intercept, and no prediction
           moves — and ``m`` of them carry ``m - 1`` dependencies, not one. One
           column is removed from each of ``m - 1`` blocks to break them,
           taking ``prefer_drop`` first and then the latest-declared blocks, so
           that the earliest block in the design keeps every level.

           The dependencies are *not* global when the graph over those blocks'
           levels is disconnected. Each component shifts independently, so a
           design with ``c`` components carries its own set per component, and
           breaking one leaves the rest behind. Cohorts that share no material
           do this: on the spacing-exp2 export the ten components are its
           courses, and one reference student is dropped per course. A
           component where only one block partitions its rows — one where the
           rows do not all carry the same number of KCs, say — has no
           redundancy to break and keeps every column.

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
            raise if it is not. Leave this on: it is the guard that catches
            aliasing introduced by blocks added later.
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
        reduced = reduced._drop_reference_levels(prefer_drop)

        if check:
            r = reduced.rank()
            if r != reduced.n_params:
                raise ValueError(
                    f"Design still rank-deficient after identification: "
                    f"{reduced.n_params} columns, rank {r}. Either a block added to "
                    f"this design is collinear with the others, or the KC model "
                    f"carries a dependency this pass does not model exactly — a KC "
                    f"that tags every row of its component, say. Drop or "
                    f"reparameterize the offending columns before fitting, or "
                    f"AIC/BIC will count parameters that do not exist."
                )
        return reduced

    def _drop_reference_levels(self, prefer_drop: str) -> Design:
        """Break every sum redundancy, on the columns that survive.

        Deliberately decided *after* dead and duplicate columns are gone: a row
        that carried two KCs carries one once a duplicate of the pair is
        dropped, and that is when the sum redundancy comes into being. Asking
        the question of the original matrix would miss it.
        """
        rows = self.row_components()
        redundancies = self._sum_redundancies(rows)
        if not redundancies:
            return self
        n_components = int(rows.max()) + 1 if rows.size else 0

        by_name = {b.name: b for b in self.blocks}
        keep = {b.name: np.ones(b.matrix.shape[1], dtype=bool) for b in self.blocks}
        column_of: dict[str, np.ndarray] = {}
        dropped, reasons = [], []

        for label, names in sorted(redundancies.items()):
            for name in self._drop_order(names, prefer_drop)[:len(names) - 1]:
                if name not in column_of:
                    column_of[name] = self._column_components(name, rows)
                live = np.flatnonzero(keep[name] & (column_of[name] == label))
                if not live.size:
                    continue
                j = int(live[-1])
                keep[name][j] = False
                dropped.append(f"{name}:{by_name[name].columns[j]}")
                reasons.append(_reference_reason(names, label, n_components))

        if not dropped:
            return self
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

        The candidates for a sum redundancy, and the blocks whose levels the
        connected components are built over. A block that leaves some row at
        zero cannot span the all-ones direction, so it can be neither.
        """
        out = []
        for b in self.blocks:
            sums = self._row_sums(b.name)
            if sums is not None and sums.size and bool(np.all(sums > 0)):
                out.append(b.name)
        return out

    def _partition_blocks(self, rows: np.ndarray | None = None) -> list[str]:
        """Covering blocks whose row sums are *one* positive constant on ``rows``.

        Each of these spans the all-ones direction over those rows — its
        columns add to ``c * 1`` — so any two of them are linearly dependent
        there. Asked per component rather than globally, because a block can
        be constant within one cohort and not across the export: a design
        where one cohort's steps carry two KCs and another's carry one has the
        redundancy in each cohort separately and nowhere globally.
        """
        out = []
        for name in self._covering_blocks():
            sums = self._row_sums(name)
            sums = sums if rows is None else sums[rows]
            if sums.size and sums[0] > 0 and np.allclose(sums, sums[0]):
                out.append(name)
        return out

    def _has_sum_redundancy(self, rows: np.ndarray | None = None) -> bool:
        """Whether two or more blocks span the all-ones direction over ``rows``.

        Every row carries exactly one student, so the student columns sum to
        the all-ones vector. If every row also carries the *same* number ``m``
        of KCs, the KC-intercept columns sum to ``m * 1``, and the two blocks
        are linearly dependent whatever ``m`` is — not only for the usual
        one-KC-per-row partition. Nothing here is specific to those two blocks;
        see :meth:`_partition_blocks`.
        """
        return len(self._partition_blocks(rows)) >= 2

    def row_components(self) -> np.ndarray:
        """Component label per row, from the graph over the partitioning blocks.

        Two rows land in the same component when a chain of shared levels —
        shared students, shared KCs, shared anything that covers every row —
        connects them. One component is the ordinary case; several mean the
        export holds cohorts that never met the same material, and each of
        them carries its own sum redundancies and its own reference levels
        (see :meth:`identify`). All-zero when fewer than two blocks cover the
        rows, since then there is no redundancy for components to localize.
        """
        names = self._covering_blocks()
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

    def _sum_redundancies(self, rows: np.ndarray) -> dict[int, list[str]]:
        """Per component, the blocks that partition it — two or more or nothing.

        Rows are grouped by one sort rather than one scan per component, so
        this stays linear-ish however many components there are — a design
        where no two students share an item has as many components as students.
        """
        if rows.size == 0:
            return {}
        order = np.argsort(rows, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(rows[order])) + 1)
        out = {}
        for group in groups:
            names = self._partition_blocks(group)
            if len(names) >= 2:
                out[int(rows[group[0]])] = names
        return out

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
