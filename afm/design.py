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


@dataclass(frozen=True)
class Design:
    """A concatenation of blocks, with row subsetting for cross-validation."""

    blocks: tuple[Block, ...]

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
        """Total free coefficients — the ``nPars`` that AIC and BIC penalize.

        LearnSphere counts every column, including slopes resting on their
        lower bound of zero, and does not add an intercept (PyAFM passes
        ``fit_intercept=False``). Reproduced exactly; see tests.
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
        """Row subset, preserving every block's labels, penalty, and bounds."""
        return Design(tuple(
            Block(b.name, b.matrix[rows], b.columns, b.l2, b.lower, b.upper)
            for b in self.blocks
        ))

    def with_blocks(self, *extra: Block) -> Design:
        """Append blocks — the hook for congruity terms and hierarchies."""
        return Design(self.blocks + tuple(extra))


def build_afm_design(data: StepData, *, student_l2: float = STUDENT_L2,
                     bound_slopes: bool = False) -> Design:
    """Assemble LearnSphere's AFM design from parsed student-step data.

    :param student_l2: ridge on student intercepts; 1.0 is PyAFM's value.
    :param bound_slopes: constrain learning rates to be non-negative.

    **The two LearnSphere AFMs disagree here, and the default follows the one
    that produced the published numbers.** PyAFM bounds slopes at ``(0, None)``,
    but the DataShop workflow behind the EDM 2025 tables (wf3990) does not:
    3,790 of its 29,700 fitted slopes are negative, down to -1.17. Bounding
    optimizes over a strictly smaller feasible set and therefore reports a
    worse likelihood on the same data, by 7-22 nats on E-learning-22.

    So ``bound_slopes=False`` is the default — it reproduces the published
    baseline. Pass ``True`` for PyAFM's variant.

    A consequence worth keeping straight: because real fits admit negative
    slopes, the EDM 2025 RQ-3 screen (``gamma <= 0.001``) selects KCs where
    students did not learn *or got worse*, not only the flat ones.
    """
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

    q_rows, q_cols, q_vals, t_vals = [], [], [], []
    for i, (labels, counts) in enumerate(zip(data.kcs, data.opportunities)):
        for label, count in zip(labels, counts):
            q_rows.append(i)
            q_cols.append(k_index[label])
            q_vals.append(1.0)
            t_vals.append(float(count))

    shape = (n, len(kcs))
    kc_mat = sparse.csr_matrix((q_vals, (q_rows, q_cols)), shape=shape)
    opp_mat = sparse.csr_matrix((t_vals, (q_rows, q_cols)), shape=shape)

    return Design((
        Block.build("student", student_mat, students, l2=student_l2),
        Block.build("kc_intercept", kc_mat, kcs),
        Block.build("kc_slope", opp_mat, kcs,
                    lower=0.0 if bound_slopes else -np.inf),
    ))


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
