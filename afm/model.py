"""The AFM fitter: LearnSphere's objective, sparse and instrumented.

The optimization problem is PyAFM's, term for term
(``AnalysisPyAfm/program/custom_logistic.py``). Minimize over ``w``::

    f(w) = sum_n [ log(1 + exp(z_n)) - y_n * z_n ]  +  sum_p (l2_p / 2) * w_p^2
    z = X w

subject to per-coefficient box constraints, from ``w0 = 0``, with TNC.

Two things about the reference are load-bearing and are reproduced rather
than corrected, because correcting either would break comparability with
every AIC/BIC DataShop has already published:

**The reported log-likelihood includes the ridge penalty.** PyAFM sets
``self.ll = -w.fun``, and ``w.fun`` is the *penalized* objective. So the
"log likelihood" in a DataShop model-values table is really a penalized
objective, and the AIC and BIC built from it inherit that. We expose both:
:attr:`AFMFit.ll` follows LearnSphere, :attr:`AFMFit.ll_unpenalized` is the
actual Bernoulli log-likelihood.

**Every column counts toward nPars**, including slopes resting on their lower
bound of zero, and there is no intercept column. Verified against the EDM
2025 tables via the identity ``BIC - AIC = nPars * (log N - 2)``; see
``tests/test_afm.py::test_published_aic_bic_identity``.

What we do change:

* The design stays sparse. PyAFM calls ``X.toarray()``, which is 1.2 GB for
  E-learning-22's ``Unique-step`` model.
* The optimizer's convergence report is kept on the result instead of
  discarded, because a silent non-convergence is indistinguishable from a bad
  KC model in the fit statistics.

DIVERGENCE (inert iteration cap): PyAFM passes ``options={'maxiter': 1000}``
to ``scipy.optimize.minimize(method="TNC")``, but TNC's budget option is
``maxfun`` — scipy raises ``OptimizeWarning: Unknown solver options: maxiter``
and ignores it. Every published AFM fit therefore ran at TNC's *default*
budget of ``max(100, 10 * nPars)`` function evaluations, which scales with
model size: 410 for E22's ``Single-KC`` (41 parameters) against 37,690 for
``Unique-step`` (3,769). We default ``max_fun=None``, reproducing that
behaviour so published numbers stay comparable, and route an explicit
``max_fun`` to the option name the solver actually reads.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from afm.data import StepData
from afm.design import Design, coefficient_frame

DEFAULT_METHOD = "TNC"  # PyAFM's choice
GRADIENT_TOL_SCALE = 3e-3  # see AFMFit.gradient_tolerance

# The option name each solver reads for its evaluation/iteration budget. TNC
# ignores 'maxiter' entirely, which is how the reference's stated cap of 1000
# came to be inert; see the module docstring.
_BUDGET_OPTION = {"TNC": "maxfun"}


def _budget_options(method: str, max_fun: int | None) -> dict:
    if max_fun is None:
        return {}
    return {_BUDGET_OPTION.get(method, "maxiter"): int(max_fun)}


def _objective(w, X, y, l2):
    z = X @ w
    nll = float(np.sum(np.logaddexp(0.0, z) - y * z))
    return nll + 0.5 * float(np.dot(l2, w * w))


def _gradient(w, X, y, l2):
    z = X @ w
    p = _expit(z)
    return X.T @ (p - y) + l2 * w


def _expit(z: np.ndarray) -> np.ndarray:
    """Stable logistic. Matches util.invlogit's branch on the sign of z."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class AFMFit:
    """A fitted AFM: coefficients, fit statistics, and optimizer diagnostics."""

    weights: np.ndarray
    design: Design
    n_obs: int
    n_params: int
    ll: float                 # LearnSphere convention: -(NLL + penalty)
    ll_unpenalized: float     # the actual Bernoulli log-likelihood
    penalty: float
    converged: bool
    n_iter: int
    message: str
    max_free_gradient: float = float("nan")

    @property
    def is_optimal(self) -> bool:
        """Whether the fit satisfies the KKT conditions of its convex problem.

        The objective is convex, so a stationary point that respects the bounds
        *is* the global optimum — this is a certificate, not a heuristic, and
        it is independent of whatever the optimizer reported about itself. It
        is worth checking: LearnSphere's own AFM stops 1,618 nats short of the
        optimum on E-learning-22's ``Unique-step`` model while reporting no
        error at all.
        """
        return bool(np.isfinite(self.max_free_gradient)
                    and self.max_free_gradient < self.gradient_tolerance)

    @property
    def gradient_tolerance(self) -> float:
        """Threshold bounding the *likelihood* left on the table, not the gradient.

        Near the optimum the shortfall is ``g^2 / (2H)``, and a logistic
        Hessian diagonal is ``sum_n p(1-p) x^2 <= n/4``. Taking
        ``g = GRADIENT_TOL_SCALE * sqrt(n)`` bounds the shortfall at
        ``2 * GRADIENT_TOL_SCALE^2`` nats — about 2e-5 — **independently of
        sample size**, which is what makes one constant usable from a 40-row
        test to a 42,000-row course.
        """
        return GRADIENT_TOL_SCALE * max(1.0, self.n_obs) ** 0.5

    @property
    def aic(self) -> float:
        return -2.0 * self.ll + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.ll + self.n_params * np.log(self.n_obs)

    @property
    def aic_unpenalized(self) -> float:
        return -2.0 * self.ll_unpenalized + 2.0 * self.n_params

    @property
    def bic_unpenalized(self) -> float:
        return -2.0 * self.ll_unpenalized + self.n_params * np.log(self.n_obs)

    def predict_proba(self, design: Design | sparse.spmatrix) -> np.ndarray:
        X = design.matrix if isinstance(design, Design) else sparse.csr_matrix(design)
        if X.shape[1] != len(self.weights):
            raise ValueError(
                f"Design has {X.shape[1]} columns but the fit has {len(self.weights)} "
                "coefficients — the KC sets probably differ between fit and predict."
            )
        return _expit(X @ self.weights)

    def brier(self, design, y) -> float:
        """Mean squared error on the probability scale (PyAFM's score)."""
        resid = np.asarray(y, dtype=float) - self.predict_proba(design)
        return float(np.mean(resid ** 2))

    def rmse(self, design, y) -> float:
        return float(np.sqrt(self.brier(design, y)))

    def coefficients(self) -> pd.DataFrame:
        return coefficient_frame(self.design, self.weights)

    def block(self, name: str) -> np.ndarray:
        return self.weights[self.design.slices()[name]]

    def _block_values(self, name: str) -> dict[str, float]:
        """Fitted value per column label for one block, aliased columns absent."""
        block = next((b for b in self.design.blocks if b.name == name), None)
        if block is None:
            return {}
        return dict(zip(block.columns, self.weights[self.design.slices()[name]]))

    def centred_students(self, data: StepData) -> tuple[pd.Series, float]:
        """Student effects at the sum-to-zero point, and the shift applied.

        Reference coding leaves ``beta_k`` meaning "for the reference student",
        which is an arbitrary choice. Moving to ``mean(theta) = 0`` makes it
        "for the average student" instead. This is a slide along the flat
        direction of :meth:`~afm.design.Design.identify`, so every fitted value
        is unchanged — the caller must add the same shift to the KC intercepts.

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

        names = data.kc_names
        beta = np.array([intercepts.get(n, np.nan) + shift for n in names])
        return pd.DataFrame({
            "KC Name": names,
            "Intercept (logit)": beta,
            "Intercept (probability) at Opportunity 1": _expit(np.nan_to_num(beta)) * np.where(np.isnan(beta), np.nan, 1.0),
            "Slope": [slopes.get(n, np.nan) for n in names],
            "Number of Unique Steps": [len(steps.get(n, ())) for n in names],
        }).sort_values("KC Name", ignore_index=True)

    def summary(self) -> str:
        flag = "" if self.is_optimal else "  *** NOT AT THE OPTIMUM ***"
        lines = [
            f"AFM | n = {self.n_obs:,} | params = {self.n_params:,}",
            (f"  log-likelihood {self.ll:12.4f}   "
             f"(unpenalized {self.ll_unpenalized:.4f}, ridge {self.penalty:.4f})"),
            f"  AIC            {self.aic:12.4f}",
            f"  BIC            {self.bic:12.4f}",
            (f"  optimality     max|grad| {self.max_free_gradient:.3g} "
             f"(tol {self.gradient_tolerance:.3g}){flag}"),
            f"  optimizer      {self.n_iter} iterations — {self.message}",
        ]
        if len(self.design.aliased):
            lines.append(f"  identification {self.design.aliased.summary()}")
        return "\n".join(lines)


def fit_afm(design: Design, y, *, method: str = DEFAULT_METHOD,
            max_fun: int | None = None, tol: float | None = None,
            warn_not_converged: bool = True) -> AFMFit:
    """Fit AFM by penalized maximum likelihood under box constraints.

    :param method: ``"TNC"`` reproduces LearnSphere. ``"L-BFGS-B"`` accepts
        the same bounds and usually converges tighter in fewer evaluations,
        but will differ from published values in the last decimals.
    :param max_fun: budget in function evaluations. ``None`` uses the solver's
        default, which is what every published AFM fit effectively used (see
        the module docstring on the reference's inert ``maxiter``).
    """
    X = design.matrix
    y = np.asarray(y, dtype=float)
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"{X.shape[0]} design rows but {y.shape[0]} responses")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Responses must be coded 0/1")

    l2 = design.l2
    w0 = np.zeros(X.shape[1])

    result = minimize(
        _objective, w0, args=(X, y, l2), jac=_gradient,
        method=method, bounds=design.bounds,
        options=_budget_options(method, max_fun) | ({} if tol is None else {"ftol": tol}),
    )

    w = result.x
    penalty = 0.5 * float(np.dot(l2, w * w))
    penalized_nll = float(result.fun)

    grad = _gradient(w, X, y, l2)
    lower = np.array([-np.inf if b[0] is None else b[0] for b in design.bounds])
    upper = np.array([np.inf if b[1] is None else b[1] for b in design.bounds])
    free = (w > lower + 1e-9) & (w < upper - 1e-9)
    max_free_grad = float(np.abs(grad[free]).max()) if free.any() else 0.0

    fit = AFMFit(
        weights=w, design=design, n_obs=X.shape[0], n_params=design.n_params,
        ll=-penalized_nll,
        ll_unpenalized=-(penalized_nll - penalty),
        penalty=penalty,
        converged=bool(result.success),
        n_iter=int(getattr(result, "nit", -1)),
        message=str(result.message),
        max_free_gradient=max_free_grad,
    )

    if warn_not_converged and not fit.is_optimal:
        warnings.warn(
            f"AFM is not at a stationary point: max |gradient| on free coefficients "
            f"is {fit.max_free_gradient:.3g} (tolerance {fit.gradient_tolerance:.3g}) "
            f"after {fit.n_iter} iterations ({fit.message}). Fit statistics for this "
            f"{fit.n_params:,}-parameter model are not the optimum; raise max_fun or "
            "try method='L-BFGS-B'.",
            RuntimeWarning, stacklevel=2,
        )
    return fit
