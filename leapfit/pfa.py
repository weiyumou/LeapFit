"""Performance Factors Analysis: its design matrix and its reporting layer.

The model is Pavlik, Cen & Koedinger's (AIED 2009), as fixed-effects logistic
regression::

    logit P(Y_ij = 1) = [theta_i] + sum_k q_jk * (beta_k + gamma_k * s_ik + rho_k * f_ik)
                        \\_______/   \\________________________________________________/
                         student      kc_intercept + kc_success + kc_failure

where ``s_ik`` and ``f_ik`` count student ``i``'s **prior** successes and
failures on KC ``k`` — strictly before the current attempt, over the same
canonical practice ordering AFM's opportunity counts use. Canonical PFA has no
student term (that is its point: usable for adaptive scheduling without an
ability estimate); ``student_intercepts=True`` adds one.

PFA relates to AFM by splitting practice by outcome: ``s_ik + f_ik = T_ik``
identically, so AFM is the restriction ``gamma_k = rho_k``. That identity is
pinned by a test, and it is what makes AIC/BIC/LRT comparisons between the two
families meaningful on one dataset.

**Provenance, and where we deliberately differ.** LearnSphere ships two PFA
components, and neither fits the canonical model:

* ``AnalysisPfa`` ("Full"/"Simple", Pavlik 2016) fits per-KC or pooled slopes
  with *random* intercepts for KC and student (``lme4::glmer``), and its
  ``info.xml`` says so. Its prior counts are correctly lagged upstream
  (``GeneratePfaFeatures/program/PFA-features.R:96-108``).
* ``AnalysisPfaStepBased`` — the one that reads student-step files — fits
  pooled slopes with random KC slopes and random student intercepts
  (``PFA.R:125``, ``nAGQ=0``), and builds its counts with an **inclusive**
  ``cumsum`` (``PFA.R:33-34``): each attempt's own outcome is inside its own
  predictor, so the response regresses on itself. Its manifest promises
  "prior" counts; the sibling component lags them correctly.

We fit fixed effects throughout (same solver, certificates, and comparability
as :mod:`leapfit.afm`; a mixed-effects fit is a different estimator whose
AIC/BIC are not comparable with these). Counts are strictly prior by default;
``counts="inclusive"`` reproduces the step-based component's construction for
demonstration and warns, because on pure noise it manufactures opposite-signed
"learning rates" and ~0.15 of in-sample AUC. It reproduces only their count
*timing* — not their estimator, and not their treatment of multi-KC cells as
atomic labels (leapfit always splits on ``~~``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from leapfit.data import StepData
from leapfit.design import Block, Design
from leapfit.fit import DEFAULT_METHOD, LogisticFit, _expit, fit_logistic

COUNT_MODES = ("prior", "inclusive")


def success_failure_counts(
    data: StepData, *, inclusive: bool = False,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Per-observation success/failure counts for each of its KCs.

    Shaped exactly like ``data.opportunities``: element ``n`` holds one count
    per KC of observation ``n``, aligned by position. Accumulated over
    :meth:`~leapfit.data.StepData.practice_order`, so "before" means the same
    thing it means for AFM's ``T`` — and ``s + f`` equals the recomputed
    opportunity count identically (a pinned invariant).

    :param inclusive: include the current attempt's own outcome in its own
        counts, replicating ``AnalysisPfaStepBased``'s ``cumsum``
        (``PFA.R:33-34``). Under it ``s = s_prior + y`` and
        ``f = f_prior + (1 - y)`` hold row by row — the label-leak identity.
        Exists so the defect is reproducible; never use it for analysis.
    """
    s_out: list[tuple[int, ...]] = [()] * len(data)
    f_out: list[tuple[int, ...]] = [()] * len(data)
    for rows in data.practice_order().values():
        s_seen: dict[str, int] = {}
        f_seen: dict[str, int] = {}
        for i in rows:
            correct = int(data.y[i])
            s_row, f_row = [], []
            for kc in data.kcs[i]:
                s, f = s_seen.get(kc, 0), f_seen.get(kc, 0)
                if inclusive:
                    s, f = s + correct, f + (1 - correct)
                s_row.append(s)
                f_row.append(f)
                s_seen[kc] = s_seen.get(kc, 0) + correct
                f_seen[kc] = f_seen.get(kc, 0) + (1 - correct)
            s_out[i], f_out[i] = tuple(s_row), tuple(f_row)
    return s_out, f_out


def build_pfa_design(data: StepData, *, slopes: str = "per_kc",
                     student_intercepts: bool = False, student_l2: float = 0.0,
                     counts: str = "prior",
                     identify: bool = True) -> Design:
    """Assemble the PFA design from parsed student-step data.

    :param slopes: ``"per_kc"`` (canonical: one ``gamma_k``/``rho_k`` per KC)
        or ``"pooled"`` (one shared ``gamma``/``rho``, the analogue of
        ``AnalysisPfa``'s "Simple" variant and of what
        ``AnalysisPfaStepBased`` fits as fixed effects).
    :param student_intercepts: add a fixed student block. Canonical PFA omits
        it; both LearnSphere components include a (random) one. With one KC
        per row this recreates the student/KC sum redundancy, which
        ``identify`` resolves exactly as for AFM.
    :param student_l2: ridge on the student block when present.
    :param counts: ``"prior"`` or ``"inclusive"`` — see
        :func:`success_failure_counts`. Inclusive warns: it exists to
        reproduce a defect, and every statistic of such a fit describes a
        model whose predictors contain the response.
    :param identify: drop aliased columns afterwards. A KC with no prior
        successes anywhere has an identically-zero success column — its
        ``gamma_k`` is not estimable, the analogue of AFM's never-practised-
        twice slopes — and one student is dropped as the reference level when
        the student block makes the design redundant.
    """
    if slopes not in ("per_kc", "pooled"):
        raise ValueError(f"slopes must be 'per_kc' or 'pooled', got {slopes!r}")
    if counts not in COUNT_MODES:
        raise ValueError(f"counts must be one of {COUNT_MODES}, got {counts!r}")
    if counts == "inclusive":
        warnings.warn(
            "counts='inclusive' replicates AnalysisPfaStepBased's cumsum, which "
            "puts each response inside its own predictor (PFA.R:33-34). Fit "
            "statistics under it describe a model that has seen its own labels. "
            "For demonstration only.",
            UserWarning, stacklevel=2,
        )

    s_counts, f_counts = success_failure_counts(data, inclusive=(counts == "inclusive"))

    kcs = data.kc_names
    k_index = {k: i for i, k in enumerate(kcs)}
    n = len(data)

    q_rows, q_cols, s_vals, f_vals = [], [], [], []
    for i, (labels, s_row, f_row) in enumerate(zip(data.kcs, s_counts, f_counts)):
        for label, s, f in zip(labels, s_row, f_row):
            q_rows.append(i)
            q_cols.append(k_index[label])
            s_vals.append(float(s))
            f_vals.append(float(f))

    shape = (n, len(kcs))
    kc_mat = sparse.csr_matrix((np.ones(len(q_rows)), (q_rows, q_cols)), shape=shape)

    blocks: list[Block] = []
    if student_intercepts:
        students = data.student_names
        s_index = {s: i for i, s in enumerate(students)}
        student_mat = sparse.csr_matrix(
            (np.ones(n), (np.arange(n), [s_index[s] for s in data.students])),
            shape=(n, len(students)),
        )
        blocks.append(Block.build("student", student_mat, students, l2=student_l2))

    blocks.append(Block.build("kc_intercept", kc_mat, kcs))

    if slopes == "per_kc":
        s_mat = sparse.csr_matrix((s_vals, (q_rows, q_cols)), shape=shape)
        f_mat = sparse.csr_matrix((f_vals, (q_rows, q_cols)), shape=shape)
        s_mat.eliminate_zeros()  # a zero count is a structural zero, not a datum
        f_mat.eliminate_zeros()
        blocks.append(Block.build("kc_success", s_mat, kcs))
        blocks.append(Block.build("kc_failure", f_mat, kcs))
    else:
        # Pooled: gamma * sum_k q_jk s_ik — the row totals across the step's KCs.
        s_tot = np.array([float(sum(row)) for row in s_counts])
        f_tot = np.array([float(sum(row)) for row in f_counts])
        blocks.append(Block.build("success", s_tot[:, None], ["success"]))
        blocks.append(Block.build("failure", f_tot[:, None], ["failure"]))

    design = Design(tuple(blocks))
    return design.identify() if identify else design


@dataclass
class PFAFit(LogisticFit):
    """A fitted PFA: everything :class:`~leapfit.fit.LogisticFit` reports, in
    KC-shaped form."""

    def kc_values(self, data: StepData, *, centre: bool = True) -> pd.DataFrame:
        """Per-KC parameters: difficulty, success slope, failure slope.

        Mirrors :meth:`leapfit.afm.AFMFit.kc_values`. Every KC in ``data``
        gets a row. A slope whose column was aliased away — a KC with no prior
        successes (or failures) anywhere — reports ``NaN``, not ``0.0``: it was
        never estimated. Under ``slopes="pooled"`` the shared slope is
        broadcast to every KC. ``Separated`` flags coefficients with no finite
        MLE, whose printed values are artefacts of where the optimizer stopped.

        :param centre: with a student block on a one-KC-per-row design, report
            intercepts for the average student (sum-to-zero) rather than the
            reference student. A no-op for canonical student-free PFA.
        """
        intercepts = self._block_values("kc_intercept")
        s_per = self._block_values("kc_success")
        f_per = self._block_values("kc_failure")
        pooled_s = self._block_values("success")
        pooled_f = self._block_values("failure")

        shift = 0.0
        has_students = any(b.name == "student" for b in self.design.blocks)
        if centre and has_students and self.design.recentring_is_valid():
            _, shift = self.centred_students(data)

        steps: dict[str, set[str]] = {}
        for labels, item in zip(data.kcs, data.items):
            for label in labels:
                steps.setdefault(label, set()).add(item)

        by_block = self.separated.by_block()
        diverging = (set(by_block.get("kc_intercept", ()))
                     | set(by_block.get("kc_success", ()))
                     | set(by_block.get("kc_failure", ())))

        names = data.kc_names
        beta = np.array([intercepts.get(n, np.nan) + shift for n in names])

        def slope(per: dict[str, float], pooled: dict[str, float]) -> list[float]:
            if pooled:  # one shared coefficient, broadcast
                value = next(iter(pooled.values()))
                return [value] * len(names)
            return [per.get(n, np.nan) for n in names]

        return pd.DataFrame({
            "KC Name": names,
            "Intercept (logit)": beta,
            "Intercept (probability) at first attempt":
                _expit(np.nan_to_num(beta)) * np.where(np.isnan(beta), np.nan, 1.0),
            "Success Slope": slope(s_per, pooled_s),
            "Failure Slope": slope(f_per, pooled_f),
            "Number of Unique Steps": [len(steps.get(n, ())) for n in names],
            "Separated": [n in diverging for n in names],
        }).sort_values("KC Name", ignore_index=True)


def fit_pfa(design: Design, y, *, method: str = DEFAULT_METHOD,
            max_fun: int | None = None, tol: float | None = None,
            warn_not_converged: bool = True,
            warn_separated: bool = True) -> PFAFit:
    """Fit PFA by penalized maximum likelihood — :func:`leapfit.fit.fit_logistic`
    with a PFA reporting view. See that function for the parameters."""
    return fit_logistic(design, y, method=method, max_fun=max_fun, tol=tol,
                        warn_not_converged=warn_not_converged,
                        warn_separated=warn_separated, result_type=PFAFit,
                        label="PFA", stacklevel=3)  # 3: attribute past this wrapper
