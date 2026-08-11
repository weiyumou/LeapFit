"""The shared logistic-family fitter: LearnSphere's objective, sparse and instrumented.

Every additive model in this package — AFM today, PFA next — is a penalized
logistic regression over a labelled :class:`~leapfit.design.Design`. They differ
only in which columns the design holds, so the solver, the fit statistics, and
the optimality certificate live here once and the algorithm modules stay thin.
Models that are *not* logistic (BKT's EM over an HMM, IRT's nonlinear
likelihood) get their own fitters and share only :mod:`leapfit.data`.

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
bound of zero, and there is no intercept column. Verified against published
model-fit tables via the identity ``BIC - AIC = nPars * (log N - 2)``; see
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
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from leapfit.data import StepData
from leapfit.design import Design, Separated, coefficient_frame

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
class LogisticFit:
    """A fitted logistic model: coefficients, fit statistics, and diagnostics.

    Algorithm modules subclass this to add their own reporting view — see
    :class:`leapfit.afm.AFMFit` — but everything that does not depend on what
    the columns *mean* is here.
    """

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
    separated: Separated = field(default_factory=Separated)
    label: str = "logistic"  # what summary() calls the model

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

    def annotate(self, data: StepData, *, into: pd.DataFrame | None = None) -> pd.DataFrame:
        """The source table plus a ``Predicted Error Rate (<model>)`` column.

        Student-step in, student-step out: this returns the exact table the
        data was parsed from with one column added (or overwritten in place if
        the file already carries it, as DataShop exports can), following
        DataShop's conventions — the value is the predicted **error** rate
        ``1 - P(correct)``, and rows that entered no fit because they carry no
        KC stay blank (``NaN``, written as an empty cell by ``to_csv``).

        Predictions are in-sample: the fitted model evaluated on the rows it
        was fitted to, which is what LearnSphere's components write back too.
        For held-out predictions use :mod:`leapfit.crossval`.

        :param into: add the column to this existing table (mutated and
            returned) instead of a fresh copy of ``data.source`` — how one file
            fitted under several KC models accumulates one column per model.
        """
        if data.source is None or data.source_rows is None:
            raise ValueError(
                "This StepData does not carry its source table, so there is "
                "nothing to annotate. Build it with load_student_step() or "
                "from_frame() rather than constructing StepData directly."
            )
        if self.design.n_obs != len(data):
            raise ValueError(
                f"The fit covers {self.design.n_obs} rows but the data has "
                f"{len(data)} — a fit on a subset (e.g. a CV training split) "
                "cannot annotate the full table."
            )

        out = data.source.copy() if into is None else into
        if len(out) != len(data.source):
            raise ValueError(
                f"'into' has {len(out)} rows but the source table has "
                f"{len(data.source)}; the frames cannot describe the same file."
            )

        values = np.full(len(data.source), np.nan)
        values[data.source_rows] = 1.0 - self.predict_proba(self.design)
        out[f"Predicted Error Rate ({data.kc_model})"] = values
        return out

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

    def summary(self) -> str:
        flag = "" if self.is_optimal else "  *** NOT AT THE OPTIMUM ***"
        lines = [
            f"{self.label} | n = {self.n_obs:,} | params = {self.n_params:,}",
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
        if len(self.separated):
            lines.append(f"  separation     {self.separated.summary()}")
        return "\n".join(lines)


def fit_logistic(design: Design, y, *, method: str = DEFAULT_METHOD,
                 max_fun: int | None = None, tol: float | None = None,
                 warn_not_converged: bool = True, warn_separated: bool = True,
                 result_type: type[LogisticFit] = LogisticFit,
                 label: str = "logistic", stacklevel: int = 2) -> LogisticFit:
    """Fit any :class:`~leapfit.design.Design` by penalized maximum likelihood
    under box constraints.

    Algorithm-agnostic: what is being modelled is entirely a property of the
    blocks in ``design``. :func:`leapfit.afm.fit_afm` is this function with
    ``result_type=AFMFit``.

    :param method: ``"TNC"`` reproduces LearnSphere. ``"L-BFGS-B"`` accepts
        the same bounds and usually converges tighter in fewer evaluations,
        but will differ from published values in the last decimals.
    :param max_fun: budget in function evaluations. ``None`` uses the solver's
        default, which is what every published AFM fit effectively used (see
        the module docstring on the reference's inert ``maxiter``).
    :param warn_separated: warn when some coefficient has no finite MLE (see
        :meth:`~leapfit.design.Design.separated`). The check itself always runs
        and its result is on the fit; this only controls the warning, which
        cross-validation silences because it would fire once per fold.
    :param result_type: the :class:`LogisticFit` subclass to construct, so a
        model family can attach its own reporting view to the same solve.
    :param stacklevel: how many frames up to attribute warnings to. Thin
        wrappers such as :func:`leapfit.afm.fit_afm` pass ``3`` so the warning
        points at the user's call, not at the wrapper.
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

    fit = result_type(
        weights=w, design=design, n_obs=X.shape[0], n_params=design.n_params,
        ll=-penalized_nll,
        ll_unpenalized=-(penalized_nll - penalty),
        penalty=penalty,
        converged=bool(result.success),
        n_iter=int(getattr(result, "nit", -1)),
        message=str(result.message),
        max_free_gradient=max_free_grad,
        separated=design.separated(y),
        label=label,
    )

    if warn_separated and len(fit.separated):
        blocks = ", ".join(f"{len(v)} in {k}" for k, v in fit.separated.by_block().items())
        warnings.warn(
            f"{len(fit.separated)} coefficient(s) have no finite maximum-likelihood "
            f"estimate ({blocks}): every observation they touch has the same outcome, "
            "so the likelihood keeps improving as they run to +/-inf. Their reported "
            "values reflect where the optimizer stopped, and AIC/BIC count them as "
            "estimated parameters. Merge or drop the affected levels, or penalize them.",
            RuntimeWarning, stacklevel=stacklevel,
        )
    if warn_not_converged and not fit.is_optimal:
        warnings.warn(
            f"AFM is not at a stationary point: max |gradient| on free coefficients "
            f"is {fit.max_free_gradient:.3g} (tolerance {fit.gradient_tolerance:.3g}) "
            f"after {fit.n_iter} iterations ({fit.message}). Fit statistics for this "
            f"{fit.n_params:,}-parameter model are not the optimum; raise max_fun or "
            "try method='L-BFGS-B'.",
            RuntimeWarning, stacklevel=stacklevel,
        )
    return fit
