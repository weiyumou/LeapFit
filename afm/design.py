"""Design matrices as labelled, individually-penalized blocks.

AFM's linear predictor is a concatenation of three column blocks::

    logit P(Y_ij = 1) = theta_i + sum_k q_jk * beta_k + sum_k q_jk * gamma_k * T_ik
                        \\_____/   \\______________/   \\___________________________/
                         student        kc_intercept              kc_slope

LearnSphere treats each block differently — students are ridge-penalized at
1.0, KC parameters are unpenalized, and slopes are bounded below at zero — so
a plain matrix is not enough to describe the model. A :class:`Block` carries
its columns *together with* the per-column penalty and bounds that belong to
them, and :class:`Design` concatenates blocks while keeping the coefficient
labels attached.

This is deliberately the extension point for everything downstream:

* **A congruity-weighted practice term** is one more :class:`Block` holding a
  single column of accumulated congruity — nothing else changes.
* **Hierarchical shrinkage** is a reparameterization, not new machinery:
  write ``beta_k = beta_parent(k) + b_k`` as an unpenalized parent block plus
  a ridge-penalized deviation block, and the ridge weight *is* the prior
  precision ``1/sigma_b^2``. Estimating ``sigma_b^2`` needs an outer loop, but
  the inner fit is this same solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from afm.data import StepData

STUDENT_L2 = 1.0  # LearnSphere's ridge on student intercepts (PyAFM: l2 = 1.0)


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
        """Append blocks — the hook for congruity terms and hierarchies.

        The existing aliasing record carries over, but the new columns are
        unchecked: call :meth:`identify` again afterwards. A congruity or
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

    def identify(self, *, prefer_drop: str = "student", check: bool = True) -> Design:
        """Drop columns that are not estimable, so ``n_params == rank(X)``.

        Two sources of aliasing are removed, both exactly rather than
        numerically:

        1. **Dead columns.** A column of all zeros carries no information and
           its coefficient is arbitrary. In AFM these are KCs that no student
           ever practises twice, so ``T`` is always 0 and the learning rate is
           not estimable at all. E-learning-22's ``Unique-step`` model has 958.
        2. **The student/KC sum redundancy.** When every row carries exactly
           one student and exactly one KC, the student columns and the KC
           intercept columns both sum to the all-ones vector, so one column is
           redundant: adding a constant to every student and subtracting it
           from every KC intercept leaves every prediction unchanged. One
           column from ``prefer_drop`` is removed to break it.

        A student is dropped rather than a KC because the KC intercepts are
        the reported output — learning curves and the RQ-3 screen — and a KC
        missing from that table would be worse than an arbitrary reference
        student. Use :meth:`~afm.model.AFMFit.recentre_students` to move the
        fit to the reference-free sum-to-zero point afterwards.

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

        if self._has_sum_redundancy() and keep.get(prefer_drop) is not None:
            live = np.flatnonzero(keep[prefer_drop])
            if live.size:
                j = int(live[-1])
                keep[prefer_drop][j] = False
                block = next(b for b in self.blocks if b.name == prefer_drop)
                dropped.append(f"{prefer_drop}:{block.columns[j]}")
                reasons.append("reference level (student/KC sum redundancy)")

        reduced = Design(
            tuple(b.keep(keep[b.name]) for b in self.blocks),
            Aliased(tuple(self.aliased.columns) + tuple(dropped),
                    tuple(self.aliased.reasons) + tuple(reasons)),
        )

        if check:
            r = reduced.rank()
            if r != reduced.n_params:
                raise ValueError(
                    f"Design still rank-deficient after identification: "
                    f"{reduced.n_params} columns, rank {r}. Some block added to this "
                    f"design is collinear with the others; drop or reparameterize it "
                    f"before fitting, or AIC/BIC will count parameters that do not exist."
                )
        return reduced

    def _row_sums(self, name: str) -> np.ndarray | None:
        b = next((x for x in self.blocks if x.name == name), None)
        return None if b is None else np.asarray(b.matrix.sum(axis=1)).ravel()

    def kc_per_row(self) -> float | None:
        """The constant number of KCs per row, or None if it varies."""
        sums = self._row_sums("kc_intercept")
        if sums is None or sums.size == 0 or not np.allclose(sums, sums[0]):
            return None
        return float(sums[0]) if sums[0] > 0 else None

    def _has_sum_redundancy(self) -> bool:
        """True when the student columns and the KC columns span the same vector.

        Every row carries exactly one student, so the student columns sum to
        the all-ones vector. If every row also carries the *same* number ``m``
        of KCs, the KC-intercept columns sum to ``m * 1``, and the two blocks
        are linearly dependent whatever ``m`` is — not only for the usual
        one-KC-per-row partition.
        """
        students = self._row_sums("student")
        m = self.kc_per_row()
        return students is not None and m is not None and np.allclose(students, 1.0)

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


def build_afm_design(data: StepData, *, learnsphere_compat: bool = False,
                     student_l2: float | None = None, bound_slopes: bool = False,
                     identify: bool | None = None,
                     recompute_opportunities: bool = False) -> Design:
    """Assemble the AFM design from parsed student-step data.

    Two settings, because reproduction and analysis want opposite defaults.

    **Analysis (default).** No ridge, aliased columns removed, so
    ``n_params == rank(X)`` and the reported log-likelihood is a likelihood.
    Use this for anything new.

    **``learnsphere_compat=True``.** Student ridge at 1.0, no identification,
    ``n_params = n_students + 2 * n_KCs``. Reproduces the published baseline
    and nothing else — treat it as a fixture, not a model.

    :param student_l2: ridge on student intercepts. Defaults to 0.0, or 1.0
        under ``learnsphere_compat``. LearnSphere uses this penalty to pick a
        point on the flat direction described in :meth:`Design.identify`; it
        costs only ~2.4 nats on E-learning-22, which is the tell that it is an
        identification device rather than regularization. Identifying the
        design properly does the same job without contaminating the likelihood.
    :param bound_slopes: constrain learning rates to be non-negative.

        The two LearnSphere AFMs disagree, and the default follows the one that
        produced the published numbers. ``AnalysisPyAfm`` bounds slopes at
        ``(0, None)``; the ``wf3990`` DataShop workflow
        (``AnalysisFastAfmAndCv``) does not, and 3,790 of its 29,700 fitted
        slopes are negative. Bounding also puts the MLE on the boundary for
        every non-learning KC, where likelihood-ratio statistics are no longer
        chi-squared — which would break the nested tests this package exists
        to support. If non-negativity is wanted, impose it as a prior on the
        slope, not as a wall.

        A consequence worth keeping straight: because real fits admit negative
        slopes, the EDM 2025 RQ-3 screen (``gamma <= 0.001``) selects KCs where
        students did not learn *or got worse*.
    :param identify: drop aliased columns. Defaults to ``not learnsphere_compat``.
    :param recompute_opportunities: derive ``T`` from
        :meth:`~afm.data.StepData.practice_order` instead of reading DataShop's
        ``Opportunity`` column.

        The column is not always right. It follows the export's *row* order,
        and on ds5426 that order contains 12 within-student inversions of
        ``First Transaction Time`` — an attempt at 01:44:41 listed before one at
        01:44:36 — which mis-numbers 28 rows (0.07%). Recomputing uses the
        timestamps, which is what "prior practice" actually means. Off by
        default because the column is what LearnSphere's published fits used.
    """
    if student_l2 is None:
        student_l2 = STUDENT_L2 if learnsphere_compat else 0.0
    if identify is None:
        identify = not learnsphere_compat

    students = data.student_names
    kcs = data.kc_names
    s_index = {s: i for i, s in enumerate(students)}
    k_index = {k: i for i, k in enumerate(kcs)}
    n = len(data)

    rows = np.arange(n)
    student_mat = sparse.csr_matrix(
        (np.ones(n), (rows, [s_index[s] for s in data.students])),
        shape=(n, len(students)),
    )

    opportunities = (data.recomputed_opportunities() if recompute_opportunities
                     else data.opportunities)
    q_rows, q_cols, q_vals, t_vals = [], [], [], []
    for i, (labels, counts) in enumerate(zip(data.kcs, opportunities)):
        for label, count in zip(labels, counts):
            q_rows.append(i)
            q_cols.append(k_index[label])
            q_vals.append(1.0)
            t_vals.append(float(count))

    shape = (n, len(kcs))
    kc_mat = sparse.csr_matrix((q_vals, (q_rows, q_cols)), shape=shape)
    opp_mat = sparse.csr_matrix((t_vals, (q_rows, q_cols)), shape=shape)
    opp_mat.eliminate_zeros()  # a T=0 entry is a structural zero, not a datum

    design = Design((
        Block.build("student", student_mat, students, l2=student_l2),
        Block.build("kc_intercept", kc_mat, kcs),
        Block.build("kc_slope", opp_mat, kcs,
                    lower=0.0 if bound_slopes else -np.inf),
    ))
    return design.identify() if identify else design


def congruity_block(data: StepData, accumulated: np.ndarray,
                    *, name: str = "congruity", columns: list[str] | None = None) -> Block:
    """Wrap precomputed congruity accumulators as an unpenalized design block.

    ``accumulated`` is ``(n_obs,)`` or ``(n_obs, p)`` — one column per
    accumulator you want a separate coefficient for (e.g. cross-KC and
    within-KC congruity, or a plain off-KC attempt count alongside them, which
    is what identifies the transfer-neutral congruity level).
    """
    acc = np.asarray(accumulated, dtype=float)
    if acc.ndim == 1:
        acc = acc[:, None]
    if acc.shape[0] != len(data):
        raise ValueError(f"Expected {len(data)} rows, got {acc.shape[0]}")
    labels = columns or ([name] if acc.shape[1] == 1
                         else [f"{name}_{i}" for i in range(acc.shape[1])])
    return Block.build(name, acc, labels)


def coefficient_frame(design: Design, weights: np.ndarray) -> pd.DataFrame:
    """Fitted weights as a tidy table of (block, column, estimate)."""
    return pd.DataFrame({
        "block": [b.name for b in design.blocks for _ in b.columns],
        "column": [c for b in design.blocks for c in b.columns],
        "estimate": weights,
    })
