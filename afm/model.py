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

    def kc_values(self, data: StepData) -> pd.DataFrame:
        """KC parameters in DataShop's model-values layout.

        Column names match what DataShop exports and what the existing
        ``refine-datashop-kc`` command reads back (``KC Name``, ``Slope``,
        ``Intercept (probability) at Opportunity 1``), so a local fit is a
        drop-in replacement for a downloaded KC-values file.
        """
        slices = self.design.slices()
        names = self.design.blocks[[b.name for b in self.design.blocks].index("kc_intercept")].columns
        intercepts = self.weights[slices["kc_intercept"]]
        slopes = self.weights[slices["kc_slope"]]

        steps: dict[str, set[str]] = {}
        for labels, item in zip(data.kcs, data.items):
            for label in labels:
                steps.setdefault(label, set()).add(item)

        return pd.DataFrame({
            "KC Name": names,
            "Intercept (logit)": intercepts,
            "Intercept (probability) at Opportunity 1": _expit(np.asarray(intercepts)),
            "Slope": slopes,
            "Number of Unique Steps": [len(steps.get(n, ())) for n in names],
        }).sort_values("KC Name", ignore_index=True)

    def summary(self) -> str:
        flag = "" if self.converged else "  *** DID NOT CONVERGE ***"
        return (
            f"AFM | n = {self.n_obs:,} | params = {self.n_params:,}\n"
            f"  log-likelihood {self.ll:12.4f}   (unpenalized {self.ll_unpenalized:.4f}, "
            f"ridge penalty {self.penalty:.4f})\n"
            f"  AIC            {self.aic:12.4f}\n"
            f"  BIC            {self.bic:12.4f}\n"
            f"  optimizer      {self.n_iter} iterations — {self.message}{flag}"
        )


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

    fit = AFMFit(
        weights=w, design=design, n_obs=X.shape[0], n_params=design.n_params,
        ll=-penalized_nll,
        ll_unpenalized=-(penalized_nll - penalty),
        penalty=penalty,
        converged=bool(result.success),
        n_iter=int(getattr(result, "nit", -1)),
        message=str(result.message),
    )

    if warn_not_converged and not fit.converged:
        warnings.warn(
            f"AFM did not converge after {fit.n_iter} iterations ({fit.message}). "
            f"Fit statistics for a {fit.n_params:,}-parameter model are unreliable; "
            "raise max_fun or try method='L-BFGS-B'.",
            RuntimeWarning, stacklevel=2,
        )
    return fit
