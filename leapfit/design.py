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
        3. **The student/KC sum redundancy, once per connected component.**
           When every row carries exactly one student and the same number of
           KCs, the student columns and the KC intercept columns span a common
           direction: adding a constant to every student and subtracting it
           from every KC intercept leaves every prediction unchanged. One
           column from ``prefer_drop`` is removed to break it.

           That redundancy is *not* unique when the student x KC bipartite
           graph is disconnected. Each component shifts independently, so a
           design with ``c`` components carries ``c`` of them, and dropping a
           single reference level leaves ``c - 1`` behind. Cohorts that share
           no material do this: on the spacing-exp2 export the ten components
           are its courses, and one reference student is dropped per course.
           A component whose rows do not all carry the same number of KCs has
           no redundancy to break and keeps every column.

        A student is dropped rather than a KC because the KC intercepts are
        the reported output — learning curves, difficulty tables, low-slope
        screens — and a KC missing from that table would be worse than an
        arbitrary reference student. Use :meth:`~leapfit.afm.AFMFit.centred_students` to move the
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
        """Break one sum redundancy per component, on the columns that survive.

        Deliberately decided *after* dead and duplicate columns are gone: a row
        that carried two KCs carries one once a duplicate of the pair is
        dropped, and that is when the student/KC sum redundancy comes into
        being. Asking the question of the original matrix would miss it.
        """
        block = next((b for b in self.blocks if b.name == prefer_drop), None)
        if block is None or block.matrix.shape[1] == 0:
            return self

        rows = self.row_components()
        n_components = int(rows.max()) + 1 if rows.size else 0
        column_of = self._column_components(prefer_drop, rows)

        keep = np.ones(block.matrix.shape[1], dtype=bool)
        dropped, reasons = [], []
        for label in self._sum_redundant_components(rows):
            live = np.flatnonzero(keep & (column_of == label))
            if not live.size:
                continue
            j = int(live[-1])
            keep[j] = False
            dropped.append(f"{prefer_drop}:{block.columns[j]}")
            reasons.append(
                "reference level (student/KC sum redundancy)" if n_components == 1
                else ("reference level (student/KC sum redundancy, component "
                      f"{label + 1} of {n_components})"))

        if not dropped:
            return self
        return Design(
            tuple(b.keep(keep) if b.name == prefer_drop else b for b in self.blocks),
            Aliased(tuple(self.aliased.columns) + tuple(dropped),
                    tuple(self.aliased.reasons) + tuple(reasons)),
        )

    def _row_sums(self, name: str) -> np.ndarray | None:
        b = next((x for x in self.blocks if x.name == name), None)
        return None if b is None else np.asarray(b.matrix.sum(axis=1)).ravel()

    def kc_per_row(self) -> float | None:
        """The constant number of KCs per row, or None if it varies."""
        sums = self._row_sums("kc_intercept")
        if sums is None or sums.size == 0 or not np.allclose(sums, sums[0]):
            return None
        return float(sums[0]) if sums[0] > 0 else None

    def _has_sum_redundancy(self, rows: np.ndarray | None = None) -> bool:
        """True when the student columns and the KC columns span the same vector.

        Every row carries exactly one student, so the student columns sum to
        the all-ones vector. If every row also carries the *same* number ``m``
        of KCs, the KC-intercept columns sum to ``m * 1``, and the two blocks
        are linearly dependent whatever ``m`` is — not only for the usual
        one-KC-per-row partition.

        ``rows`` restricts the question to a subset — one connected component,
        where the redundancy actually lives (see :meth:`row_components`). The
        whole design is one component's worth of rows in the common case, and
        then this is the global test it has always been.
        """
        students = self._row_sums("student")
        kcs = self._row_sums("kc_intercept")
        if students is None or kcs is None:
            return False
        if rows is not None:
            students, kcs = students[rows], kcs[rows]
        return bool(students.size and np.allclose(students, 1.0)
                    and np.allclose(kcs, kcs[0]) and kcs[0] > 0)

    def row_components(self) -> np.ndarray:
        """Component label per row, from the student x KC bipartite graph.

        Two rows land in the same component when a chain of shared students and
        shared KCs connects them. One component is the ordinary case; several
        mean the export holds cohorts that never met the same material, and
        each of them carries its own sum redundancy and its own intercept
        level (see :meth:`identify`). All-zero if either block is missing.
        """
        student = next((b for b in self.blocks if b.name == "student"), None)
        kc = next((b for b in self.blocks if b.name == "kc_intercept"), None)
        if student is None or kc is None:
            return np.zeros(self.n_obs, dtype=np.int64)

        incidence = (student.matrix.T @ kc.matrix).tocsr()
        incidence.data[:] = 1.0
        graph = sparse.bmat([[None, incidence], [incidence.T, None]], format="csr")
        _, labels = csgraph.connected_components(graph, directed=False)
        # Each row carries exactly one student, so its student's label is its own.
        by_student = labels[: student.matrix.shape[1]]
        by_row = (student.matrix @ (by_student + 1.0)).astype(np.int64) - 1
        # Renumber consecutively: a KC nobody practises is its own graph
        # component, and would otherwise leave a gap in the labels.
        return np.unique(by_row, return_inverse=True)[1].astype(np.int64)

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

    def _sum_redundant_components(self, rows: np.ndarray) -> list[int]:
        """The components that carry a student/KC sum redundancy to break.

        Rows are grouped by one sort rather than one scan per component, so
        this stays linear-ish however many components there are — a design
        where no two students share an item has as many components as students.
        """
        if rows.size == 0:
            return []
        order = np.argsort(rows, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(rows[order])) + 1)
        return [int(rows[group[0]]) for group in groups
                if self._has_sum_redundancy(group)]

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
