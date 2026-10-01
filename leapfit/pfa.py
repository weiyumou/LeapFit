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
identically, so AFM is the restriction ``gamma_k = rho_k``. That identity holds
by construction, since the recomputed ``T`` is the two counts' sum from the
same pass over the practice order, and it is what makes AIC/BIC/LRT
comparisons between the two families meaningful on one dataset.

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

from leapfit.data import StepData
from leapfit.design import Block, Design
from leapfit.fit import DEFAULT_METHOD, LogisticFit, _expit, _steps_per_kc, fit_logistic

COUNT_MODES = ("prior", "inclusive")


def success_failure_counts(
    data: StepData, *, inclusive: bool = False,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Per-observation success/failure counts for each of its KCs.

    Shaped exactly like ``data.opportunities``: element ``n`` holds one count
    per KC of observation ``n``, aligned by position. These are
    :meth:`~leapfit.data.StepData.prior_counts`, accumulated over
    :meth:`~leapfit.data.StepData.practice_order`, so "before" means the same
    thing it means for AFM's ``T``: ``s + f`` is the recomputed opportunity
    count, which is computed from the same pass.

    :param inclusive: include the current attempt's own outcome in its own
        counts, replicating ``AnalysisPfaStepBased``'s ``cumsum``
        (``PFA.R:33-34``). Under it ``s = s_prior + y`` and
        ``f = f_prior + (1 - y)`` hold row by row — the label-leak identity.
        Exists so the defect is reproducible; never use it for analysis.
    """
    s, f = data.prior_counts()
    if not inclusive:
        return s, f
    return ([tuple(c + int(y) for c in row) for row, y in zip(s, data.y)],
            [tuple(c + 1 - int(y) for c in row) for row, y in zip(f, data.y)])


def build_pfa_design(data: StepData, *, slopes: str = "per_kc",
                     student_intercepts: bool = False, counts: str = "prior",
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

    blocks: list[Block] = []
    if student_intercepts:
        blocks.append(Block.from_levels("student", [(s,) for s in data.students]))
    blocks.append(Block.from_levels("kc_intercept", data.kcs))
    if slopes == "per_kc":
        blocks.append(Block.from_levels("kc_success", data.kcs, values=s_counts))
        blocks.append(Block.from_levels("kc_failure", data.kcs, values=f_counts))
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
        names = data.kc_names
        beta = self._kc_intercepts(data, centre)
        diverging = set(self.separated.in_blocks("kc_intercept", "kc_success", "kc_failure"))

        def slope(per_kc: str, pooled: str) -> list[float]:
            if shared := self._block_values(pooled):  # one coefficient, broadcast
                return [next(iter(shared.values()))] * len(names)
            per = self._block_values(per_kc)
            return [per.get(n, np.nan) for n in names]

        return pd.DataFrame({
            "KC Name": names,
            "Intercept (logit)": beta,
            "Intercept (probability) at first attempt": _expit(beta),
            "Success Slope": slope("kc_success", "success"),
            "Failure Slope": slope("kc_failure", "failure"),
            "Number of Unique Steps": _steps_per_kc(data),
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
