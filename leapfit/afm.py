"""The Additive Factors Model: its design matrix and its reporting layer.

Everything *generic* lives elsewhere — :mod:`leapfit.data` parses the export,
:mod:`leapfit.design` holds the block algebra, :mod:`leapfit.fit` holds the
penalized-logistic solver and its optimality certificate. What is left here is
only what makes AFM AFM::

    logit P(Y_ij = 1) = theta_i + sum_k q_jk * beta_k + sum_k q_jk * gamma_k * T_ik
                        \\_____/   \\______________/   \\___________________________/
                         student        kc_intercept              kc_slope

which is three column blocks (:func:`build_afm_design`) plus a KC-shaped view of
the fitted coefficients (:meth:`AFMFit.kc_values`).

The split is the point. PFA differs from AFM only in the last two blocks —
prior successes and prior failures per KC instead of a single opportunity
count — so it is a sibling module, not a fork of the fitter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from leapfit.data import StepData
from leapfit.design import Block, Design
from leapfit.fit import DEFAULT_METHOD, LogisticFit, _expit, fit_logistic

STUDENT_L2 = 1.0  # LearnSphere's ridge on student intercepts (PyAFM: l2 = 1.0)


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
        point on the flat direction described in
        :meth:`~leapfit.design.Design.identify`; it costs only ~2.4 nats on
        E-learning-22, which is the tell that it is an identification device
        rather than regularization. Identifying the design properly does the
        same job without contaminating the likelihood.
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
        slopes, a low-slope screen (``gamma <= 0.001``) selects KCs where
        students did not learn *or got worse*.
    :param identify: drop aliased columns. Defaults to ``not learnsphere_compat``.
    :param recompute_opportunities: derive ``T`` from
        :meth:`~leapfit.data.StepData.practice_order` instead of reading
        DataShop's ``Opportunity`` column.

        The column is not always right. It follows the export's *row* order,
        and on the E-learning 2022 validation export that order contains 12
        within-student inversions of ``First Transaction Time`` — an attempt at
        01:44:41 listed before one at 01:44:36 — which mis-numbers 28 rows
        (0.07%). Recomputing uses the timestamps, which is what "prior
        practice" actually means. Off by default because the column is what
        LearnSphere's published fits used.
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


@dataclass
class AFMFit(LogisticFit):
    """A fitted AFM: everything :class:`~leapfit.fit.LogisticFit` reports, in
    KC-shaped form."""

    def centred_students(self, data: StepData) -> tuple[pd.Series, float]:
        """Student effects at the sum-to-zero point, and the shift applied.

        Reference coding leaves ``beta_k`` meaning "for the reference student",
        which is an arbitrary choice. Moving to ``mean(theta) = 0`` makes it
        "for the average student" instead. This is a slide along the flat
        direction of :meth:`~leapfit.design.Design.identify`, so every fitted
        value is unchanged — the caller must add the same shift to the KC
        intercepts.

        Only valid when each row carries exactly one KC: with two KCs on a row,
        subtracting the shift from every KC intercept would remove it twice.
        """
        if not self.design.recentring_is_valid():
            raise ValueError(
                "Sum-to-zero recentring needs exactly one KC per row; this design "
                f"has {self.design.kc_per_row()} KCs per row (multi-KC), where the "
                "shift does not cancel."
            )
        fitted = self._block_values("student")
        # Columns dropped as the reference level sit at zero by construction.
        full = pd.Series({s: fitted.get(s, 0.0) for s in data.student_names})
        shift = float(full.mean())
        return full - shift, shift

    def kc_values(self, data: StepData, *, centre: bool = True) -> pd.DataFrame:
        """KC parameters in DataShop's model-values layout.

        Column names match what DataShop exports and what the existing
        ``refine-datashop-kc`` command reads back (``KC Name``, ``Slope``,
        ``Intercept (probability) at Opportunity 1``), so a local fit is a
        drop-in replacement for a downloaded KC-values file.

        Every KC in ``data`` gets a row, including ones whose columns were
        aliased away. **A KC that no student ever practises twice reports
        ``Slope = NaN``, not 0.** Its learning rate is not estimable — there is
        no second opportunity to estimate it from — and printing ``0.000``
        would invite it into an RQ-3-style screen that reads zero as "students
        did not learn".

        :param centre: report intercepts for the *average* student (sum-to-zero)
            rather than for the arbitrary reference student left by
            identification. Silently skipped on multi-KC designs, where the
            recentring identity does not hold.
        """
        intercepts = self._block_values("kc_intercept")
        slopes = self._block_values("kc_slope")

        shift = 0.0
        if centre and self.design.recentring_is_valid():
            _, shift = self.centred_students(data)

        steps: dict[str, set[str]] = {}
        for labels, item in zip(data.kcs, data.items):
            for label in labels:
                steps.setdefault(label, set()).add(item)

        by_block = self.separated.by_block()
        diverging = set(by_block.get("kc_intercept", ())) | set(by_block.get("kc_slope", ()))

        names = data.kc_names
        beta = np.array([intercepts.get(n, np.nan) + shift for n in names])
        return pd.DataFrame({
            "KC Name": names,
            "Intercept (logit)": beta,
            "Intercept (probability) at Opportunity 1": _expit(np.nan_to_num(beta)) * np.where(np.isnan(beta), np.nan, 1.0),
            "Slope": [slopes.get(n, np.nan) for n in names],
            "Number of Unique Steps": [len(steps.get(n, ())) for n in names],
            # Kept as a flag rather than blanked: unlike a never-repeated KC,
            # a separated one *is* informative — every attempt went the same
            # way — but its estimate is wherever the optimizer stopped.
            "Separated": [n in diverging for n in names],
        }).sort_values("KC Name", ignore_index=True)


def fit_afm(design: Design, y, *, method: str = DEFAULT_METHOD,
            max_fun: int | None = None, tol: float | None = None,
            warn_not_converged: bool = True,
            warn_separated: bool = True) -> AFMFit:
    """Fit AFM by penalized maximum likelihood under box constraints.

    A thin specialization of :func:`~leapfit.fit.fit_logistic` — the objective,
    the solver, and the optimality certificate are shared with every other
    logistic-family model in this package. Only the result type differs, so that
    :meth:`AFMFit.kc_values` is available on what comes back.

    :param method: ``"TNC"`` reproduces LearnSphere. ``"L-BFGS-B"`` accepts
        the same bounds and usually converges tighter in fewer evaluations,
        but will differ from published values in the last decimals.
    :param max_fun: budget in function evaluations. ``None`` uses the solver's
        default, which is what every published AFM fit effectively used (see
        :mod:`leapfit.fit` on the reference's inert ``maxiter``).
    :param warn_separated: warn when some coefficient has no finite MLE (see
        :meth:`~leapfit.design.Design.separated`).
    """
    return fit_logistic(design, y, method=method, max_fun=max_fun, tol=tol,
                        warn_not_converged=warn_not_converged,
                        warn_separated=warn_separated, result_type=AFMFit,
                        label="AFM", stacklevel=3)  # 3: attribute past this wrapper


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
