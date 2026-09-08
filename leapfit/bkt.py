"""Bayesian Knowledge Tracing: a two-state HMM per knowledge component, fitted by EM.

BKT is the first model in this package that is not a logistic regression, so it
brings its own fitter and shares only :mod:`leapfit.data`. What it does *not*
bring is a different input format: it reads the same
:class:`~leapfit.data.StepData` every other family reads, and reshapes it here.

The model, per knowledge component, over one student's responses to that KC in
practice order. The latent ``K_t`` is binary -- the student either knows the KC
at opportunity ``t`` or does not -- and each response is a noisy read of it::

    P(K_1 = 1)            = L0                     prior
    P(K_t+1 = 1 | K_t = 0) = T                     learn
    P(K_t+1 = 0 | K_t = 1) = F                     forget   (0 in classical BKT)
    P(y_t = 1 | K_t = 0)   = G                     guess
    P(y_t = 1 | K_t = 1)   = 1 - S                 slip

Read forward, one step is predict -> update -> transition, with ``p = P(known)``:

1. **predict**    ``P(y = 1) = p (1 - S) + (1 - p) G``
2. **update**     Bayes on the response, e.g. ``p' = p (1 - S) / P(y = 1)`` when correct
3. **transition** ``p'' = p' (1 - F) + (1 - p') T``

which is the scaled forward pass of the HMM written out. :func:`bkt_filter`
computes exactly this, and it is what both the predictions and the E-step use.

## Provenance

The recursion is the textbook one (Corbett & Anderson 1995). The
implementation is adapted from the ``BKTFilter`` in the companion ``betabern``
project, which fits the same recursion by gradient descent on the prequential
NLL; the arithmetic, the logit-space guards and the treatment of ``forget``
carry over unchanged, the fitter does not. Validation follows that project's
*transplantation* protocol against pyBKT -- load the reference's fitted
parameters into this implementation and compare predictions row by row -- which
separates "is this the same model" from "is this the same fitting procedure".
See ``scripts/validate_bkt.py``.

DIVERGENCE (fitting procedure): the reference implementations fit by EM, and so
does this one, but pyBKT's EM is compiled over its own sequence layout and
initialises from ``num_fits`` random restarts. We expose restarts explicitly
(``n_restarts``, seeded) and report the spread of final log-likelihoods across
them, because BKT's likelihood is genuinely multimodal and a single fit that
does not say so is a fit you cannot audit.

DIVERGENCE (no optimality certificate): every other fit in this package carries
one -- the logistic objective is convex, so satisfying the KKT conditions *is*
a proof of global optimality. BKT's likelihood is not concave, and no such
certificate exists for it under EM or under anything else. :attr:`BKTFit.converged`
therefore means only that EM stopped on its tolerance, and
:attr:`BKTFit.restart_spread` is the evidence offered in place of a proof. Do
not read the two as equivalent.

DIVERGENCE (multi-KC steps): unresolved, and raised rather than guessed. A
DataShop step may be tagged with several KCs, which AFM absorbs additively;
BKT is defined on one skill at a time and has no additive form. The available
conventions -- replicate the row once per KC (pyBKT's, which inflates ``N`` and
so makes AIC/BIC incomparable with AFM's on the same file), or model only the
step's first KC, or refuse -- are not interchangeable, and picking one silently
would put an uncomparable number in a comparison table. Until that is decided,
:func:`extract_sequences` raises on a multi-KC export naming the offending
rows. See ``docs/DESIGN.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from leapfit.data import StepData

#: Keeps probabilities off 0 and 1 so the Bayes ratio and the logs stay finite.
#: The same guard the reference implementation uses.
EPS = 1e-6

#: EM stops when the relative log-likelihood improvement falls below this.
#:
#: Set tight, and the cap set high to match, because the stopping rule measures
#: the *step* rather than the distance left to travel. BKT's likelihood has flat
#: ridges -- ``learn`` and ``forget`` trade off along one -- and on a ridge EM's
#: steps shrink long before it arrives. At a looser 1e-8 the example dataset
#: stops 0.046 nats short of the optimum on one KC while reporting convergence,
#: which is the exact failure mode this package exists to catch elsewhere.
#: :attr:`BKTFit.final_improvement` reports the step actually achieved, so the
#: claim is auditable rather than taken on trust.
DEFAULT_TOL = 1e-10

DEFAULT_MAX_ITER = 2000

#: Textbook starting point, and the reference implementation's default. The
#: guess/slip pair sits well inside the ``G + S < 1`` region that keeps BKT
#: identified, so restart 0 begins somewhere defensible rather than at random.
#:
#: ``forget`` starts at 0.01 rather than 0 **and must**: zero is an absorbing
#: fixed point of its own M-step. The update is
#: ``F' = sum_t xi_t(known, unknown) / sum_t gamma_t(known)`` and every
#: ``xi_t(known, unknown)`` carries a factor of ``A[known, unknown] = F``, so
#: ``F = 0`` re-estimates as ``F = 0`` forever. Starting there would make
#: ``forget=True`` silently identical to classical BKT while still charging a
#: parameter for it.
DEFAULT_INIT = (0.3, 0.1, 0.2, 0.1, 0.01)  # (prior, learn, guess, slip, forget)

#: Range random restarts sample from. Bounded away from 0 and 1 because EM's
#: fixed points include the degenerate corners, and a restart that starts in
#: one never leaves it.
_RESTART_RANGE = (0.05, 0.95)


def _clip(x):
    return np.clip(x, EPS, 1.0 - EPS)


# ==================================================================================
# Reshaping the export into sequences
# ==================================================================================

@dataclass(frozen=True)
class Sequences:
    """Per-(student, KC) response sequences, padded to a rectangle.

    BKT's unit of observation is a *sequence*, not a row: the latent state
    carries along one student's practice of one KC, in the order they practised
    it. This is the shape :class:`~leapfit.data.StepData` is reshaped into, and
    the only place the reshaping happens.

    ``rows[s, t]`` is the position in the originating ``StepData`` of the
    response at ``y[s, t]``, or ``-1`` where padded -- which is what lets a
    prediction computed in sequence space be written back against the right
    row of the export.
    """

    y: np.ndarray             # (n_seq, max_len) int8 in {0, 1}, 0-padded
    mask: np.ndarray          # (n_seq, max_len) bool, True where a real response
    rows: np.ndarray          # (n_seq, max_len) int, -1 where padded
    kc_of_seq: np.ndarray     # (n_seq,) index into kc_names
    students: list[str]       # (n_seq,)
    kc_names: list[str]
    kc_model: str
    n_source_rows: int        # len(StepData) this came from

    def __len__(self) -> int:
        return int(self.mask.shape[0])

    @property
    def n_obs(self) -> int:
        """Total real responses. Equals ``len(StepData)`` for a single-KC export."""
        return int(self.mask.sum())

    @property
    def lengths(self) -> np.ndarray:
        return self.mask.sum(axis=1)

    def for_kc(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The ``(y, mask)`` block for one KC, trimmed to its own longest sequence."""
        sel = self.kc_of_seq == k
        if not sel.any():
            return np.zeros((0, 0), dtype=np.int8), np.zeros((0, 0), dtype=bool)
        y, mask = self.y[sel], self.mask[sel]
        width = int(mask.sum(axis=1).max())
        return y[:, :width], mask[:, :width]

    def take(self, seq_index: np.ndarray) -> Sequences:
        """Whole sequences by index, re-trimmed; the KC vocabulary is not resized.

        The sequence-level analogue of :meth:`leapfit.design.Design.take`, for
        splitting by student or by KC. Cross-validation does *not* use it: a
        fold holds out individual responses, which shortens a sequence rather
        than removing it, and that is :func:`_restrict`.
        """
        seq_index = np.asarray(seq_index, dtype=int)
        mask = self.mask[seq_index]
        width = max(1, int(mask.sum(axis=1).max(initial=0)))
        return Sequences(
            y=self.y[seq_index][:, :width], mask=mask[:, :width],
            rows=self.rows[seq_index][:, :width], kc_of_seq=self.kc_of_seq[seq_index],
            students=[self.students[i] for i in seq_index], kc_names=self.kc_names,
            kc_model=self.kc_model, n_source_rows=self.n_source_rows,
        )


def extract_sequences(data: StepData) -> Sequences:
    """Reshape parsed student-step data into per-(student, KC) sequences.

    Order comes from :meth:`~leapfit.data.StepData.practice_order` -- the same
    canonical ordering AFM's opportunity counts and PFA's prior-success counts
    are built over, so "before" means one thing across the package.

    :raises ValueError: if any step carries more than one KC. See the module
        docstring: the convention for that case is deliberately unresolved
        rather than silently chosen.
    """
    multi = [i for i, labels in enumerate(data.kcs) if len(labels) > 1]
    if multi:
        shown = ", ".join(str(i) for i in multi[:5])
        more = f" (and {len(multi) - 5} more)" if len(multi) > 5 else ""
        raise ValueError(
            f"BKT needs one KC per step, but {len(multi)} of {len(data)} rows in KC "
            f"model {data.kc_model!r} carry several -- observations {shown}{more}. "
            "BKT is defined on a single skill and has no additive form, so the "
            "convention for multi-KC steps changes N and therefore AIC/BIC; leapfit "
            "does not pick one for you yet. Fit a single-KC model, or see "
            "docs/DESIGN.md for the open question."
        )
    empty = [i for i, labels in enumerate(data.kcs) if not labels]
    if empty:
        raise ValueError(
            f"{len(empty)} observation(s) carry no KC at all; load_student_step() drops "
            "those, so this StepData was probably built by hand."
        )

    kc_names = data.kc_names
    kc_index = {k: i for i, k in enumerate(kc_names)}

    # One sequence per (student, KC), rows in practice order.
    grouped: dict[tuple[str, str], list[int]] = {}
    for student, order in data.practice_order().items():
        for i in order:
            grouped.setdefault((student, data.kcs[i][0]), []).append(int(i))

    keys = sorted(grouped)  # deterministic sequence order, whatever dict insertion did
    max_len = max((len(grouped[k]) for k in keys), default=1)
    n_seq = len(keys)

    y = np.zeros((n_seq, max_len), dtype=np.int8)
    mask = np.zeros((n_seq, max_len), dtype=bool)
    rows = np.full((n_seq, max_len), -1, dtype=int)
    kc_of_seq = np.empty(n_seq, dtype=int)
    students = []

    for s, (student, kc) in enumerate(keys):
        idx = grouped[(student, kc)]
        y[s, :len(idx)] = data.y[idx]
        mask[s, :len(idx)] = True
        rows[s, :len(idx)] = idx
        kc_of_seq[s] = kc_index[kc]
        students.append(student)

    return Sequences(y=y, mask=mask, rows=rows, kc_of_seq=kc_of_seq, students=students,
                     kc_names=kc_names, kc_model=data.kc_model, n_source_rows=len(data))


# ==================================================================================
# The recursion
# ==================================================================================

def _transition(learn: float, forget: float) -> np.ndarray:
    """``A[i, j] = P(K_t+1 = j | K_t = i)`` over ``(unknown, known)``."""
    return np.array([[1.0 - learn, learn], [forget, 1.0 - forget]])


def _emissions(y: np.ndarray, guess: float, slip: float) -> np.ndarray:
    """``b[s, t, i] = P(y_st | K_t = i)``, shape ``(n_seq, len, 2)``."""
    correct = np.array([guess, 1.0 - slip])  # P(y = 1 | unknown), P(y = 1 | known)
    return np.where(y[..., None] == 1, correct, 1.0 - correct)


def bkt_filter(y: np.ndarray, mask: np.ndarray, prior: float, learn: float,
               guess: float, slip: float, forget: float = 0.0) -> dict:
    """Run BKT forward over padded sequences: predict, update, transition.

    This is the scaled forward pass of the HMM, written in BKT's own terms. It
    is the single place the recursion lives -- predictions read ``p_correct``
    off it, and the E-step reads ``filtered`` and ``scale``.

    Padding is inert: past a sequence's length the state is frozen and the
    scale is 1, so a padded step contributes nothing to any sum and nothing to
    the log-likelihood.

    :return: a dict with

        ``p_correct`` (n, L)
            ``P(y_t = 1 | y_<t)`` -- the one-step-ahead prediction, which is
            what "a BKT prediction" means and what pyBKT's ``predict`` returns.
        ``predicted`` (n, L, 2)
            ``P(K_t | y_<t)`` -- mastery *before* the response is seen.
        ``filtered`` (n, L, 2)
            ``P(K_t | y_<=t)`` -- after the Bayes update. The scaled forward
            variable the E-step needs.
        ``scale`` (n, L)
            ``P(y_t | y_<t)`` for the response actually observed; its masked
            log-sum is the log-likelihood.
    """
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    n, length = y.shape

    step = _transition(learn, forget)
    correct = np.array([guess, 1.0 - slip])
    b = _emissions(y, guess, slip)

    predicted = np.empty((n, length, 2))
    filtered = np.empty((n, length, 2))
    p_correct = np.zeros((n, length))
    scale = np.ones((n, length))

    state = np.tile(np.array([1.0 - prior, prior]), (n, 1))  # P(K_1) = L0
    for t in range(length):
        live = mask[:, t]
        keep = live[:, None]

        # --- 1. Predict ---
        predicted[:, t] = state
        p_correct[:, t] = state @ correct

        # --- 2. Update: Bayes on the observed response. ---
        unnormalized = state * b[:, t]
        evidence = np.where(live, unnormalized.sum(axis=1), 1.0).clip(EPS)
        filtered[:, t] = np.where(keep, unnormalized / evidence[:, None], state)
        scale[:, t] = evidence

        # --- 3. Transition: learn out of unknown, forget out of known. ---
        state = np.where(keep, filtered[:, t] @ step, state)

    return {"p_correct": p_correct, "predicted": predicted,
            "filtered": filtered, "scale": scale}


def _backward(b: np.ndarray, mask: np.ndarray, scale: np.ndarray,
              step: np.ndarray) -> np.ndarray:
    """Scaled backward variable ``beta_t(i) = P(y_>t | K_t = i) / prod(scale_>t)``.

    Scaled by the same constants the forward pass divided by, so that
    ``filtered * beta`` is already the posterior and needs no renormalizing.
    """
    n, length, _ = b.shape
    beta = np.ones((n, length, 2))
    for t in range(length - 2, -1, -1):
        ahead = b[:, t + 1] * beta[:, t + 1] / scale[:, t + 1][:, None]
        beta[:, t] = np.where(mask[:, t + 1][:, None], ahead @ step.T, beta[:, t + 1])
    return beta


# ==================================================================================
# EM
# ==================================================================================

@dataclass
class _KCFit:
    """One KC's EM run: the parameters, and what the run itself did."""

    prior: float
    learn: float
    guess: float
    slip: float
    forget: float
    ll: float
    n_iter: int
    converged: bool
    monotone: bool          # EM never decreased the likelihood -- a numerics check
    n_transitions: int      # observed opportunities to learn or forget
    final_improvement: float  # the last step EM took, in nats


def _em(y: np.ndarray, mask: np.ndarray,
        init: tuple[float, float, float, float, float], *, fit_forget: bool,
        bound_degenerate: bool, tol: float, max_iter: int) -> _KCFit:
    """Baum-Welch for the two-state BKT chain over one KC's sequences.

    EM is monotone by construction, so the loop tracks whether that actually
    held: a decrease is not a bad fit, it is a bug, and it is reported rather
    than tolerated.
    """
    prior, learn, guess, slip, start_forget = (float(v) for v in init)
    # Pinned at 0 for classical BKT; otherwise started strictly inside (0, 1),
    # because both ends are absorbing fixed points of the M-step below.
    forget = _clip(start_forget) if fit_forget else 0.0
    ceiling = 0.5 - EPS if bound_degenerate else 1.0 - EPS

    transitions = mask[:, 1:]
    n_transitions = int(transitions.sum())

    ll = -np.inf
    previous = -np.inf
    improvement = np.inf
    monotone = True
    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        step = _transition(learn, forget)
        b = _emissions(y, guess, slip)
        state = bkt_filter(y, mask, prior, learn, guess, slip, forget)
        beta = _backward(b, mask, state["scale"], step)

        # --- E-step. `ll` is the likelihood of the *current* parameters. ---
        ll = float(np.sum(np.log(state["scale"]) * mask))
        if iterations > 1:
            if ll < previous - 1e-9 * max(1.0, abs(previous)):
                monotone = False
            improvement = abs(ll - previous)
            if improvement <= tol * max(1.0, abs(previous)):
                converged = True
                break
        previous = ll

        gamma = state["filtered"] * beta * mask[:, :, None]        # P(K_t | y)
        ahead = b[:, 1:] * beta[:, 1:] / state["scale"][:, 1:, None]
        xi = (state["filtered"][:, :-1, :, None] * step[None, None, :, :]
              * ahead[:, :, None, :] * transitions[:, :, None, None])  # P(K_t, K_t+1 | y)

        # --- M-step. Each parameter is a ratio of expected counts. ---
        prior = _clip(float(gamma[:, 0, 1].mean()))

        leaving = (gamma[:, :-1] * transitions[:, :, None]).sum(axis=(0, 1))
        if leaving[0] > EPS:
            learn = _clip(float(xi[:, :, 0, 1].sum() / leaving[0]))
        if fit_forget and leaving[1] > EPS:
            forget = _clip(float(xi[:, :, 1, 0].sum() / leaving[1]))

        occupancy = gamma.sum(axis=(0, 1))
        if occupancy[0] > EPS:
            guess = float(np.clip((gamma[:, :, 0] * (y == 1)).sum() / occupancy[0],
                                  EPS, ceiling))
        if occupancy[1] > EPS:
            slip = float(np.clip((gamma[:, :, 1] * (y == 0)).sum() / occupancy[1],
                                 EPS, ceiling))

    if not converged:  # the loop ended on the iteration cap, one M-step past `ll`
        state = bkt_filter(y, mask, prior, learn, guess, slip, forget)
        ll = float(np.sum(np.log(state["scale"]) * mask))

    return _KCFit(prior=prior, learn=learn, guess=guess, slip=slip, forget=forget,
                  ll=ll, n_iter=iterations, converged=converged, monotone=monotone,
                  n_transitions=n_transitions, final_improvement=float(improvement))


def _fit_one_kc(y: np.ndarray, mask: np.ndarray, *, fit_forget: bool,
                bound_degenerate: bool, n_restarts: int,
                rng: np.random.Generator, tol: float,
                max_iter: int) -> tuple[_KCFit, np.ndarray]:
    """Run EM from several starting points and keep the best.

    Restart 0 is always :data:`DEFAULT_INIT`, so a single-restart fit is
    deterministic and reproduces the textbook starting point. The rest are
    drawn from a seeded generator. Returns the winner and *every* restart's
    final log-likelihood -- the spread is the only honest statement available
    about whether this optimum is the optimum.
    """
    scores, best = [], None
    for restart in range(max(1, n_restarts)):
        init = DEFAULT_INIT if restart == 0 else tuple(rng.uniform(*_RESTART_RANGE, size=5))
        run = _em(y, mask, init, fit_forget=fit_forget, bound_degenerate=bound_degenerate,
                  tol=tol, max_iter=max_iter)
        scores.append(run.ll)
        if best is None or run.ll > best.ll:
            best = run
    return best, np.asarray(scores)


# ==================================================================================
# The fitted model
# ==================================================================================

PARAM_NAMES = ("prior", "learn", "guess", "slip", "forget")


@dataclass
class BKTFit:
    """A fitted BKT: five probabilities per knowledge component, and the diagnostics.

    The reporting surface deliberately mirrors :class:`~leapfit.fit.LogisticFit`
    -- ``n_obs``, ``n_params``, ``ll``, ``aic``, ``bic``, ``rmse``, ``annotate``
    all mean the same things -- so a BKT row and an AFM row can sit in one
    table. Two things do **not** carry over, and both are named differently on
    purpose:

    * there is no ``is_optimal``. See :attr:`restart_spread`.
    * there is no penalized/unpenalized pair. Nothing here is penalized, so
      ``ll`` is the Bernoulli log-likelihood and nothing else.
    """

    kc_names: list[str]
    prior: np.ndarray          # (n_kcs,) L0
    learn: np.ndarray          # (n_kcs,) T
    guess: np.ndarray          # (n_kcs,) G
    slip: np.ndarray           # (n_kcs,) S
    forget: np.ndarray         # (n_kcs,) F -- all zero unless fitted
    ll: float
    n_obs: int
    n_seq: int
    n_seq_per_kc: np.ndarray   # (n_kcs,)
    n_transitions: np.ndarray  # (n_kcs,) observed chances to learn or forget
    converged: np.ndarray      # (n_kcs,) bool
    monotone: np.ndarray       # (n_kcs,) bool -- EM never went downhill
    n_iter: np.ndarray         # (n_kcs,)
    final_improvement: np.ndarray  # (n_kcs,) the last step EM took, in nats
    restart_lls: np.ndarray    # (n_kcs, n_restarts)
    has_forget: bool
    bound_degenerate: bool
    kc_model: str
    label: str = "BKT"

    # -- what exists, and what does not ------------------------------------------

    @property
    def unfitted(self) -> list[str]:
        """KCs with no sequences at all. Their parameters are ``NaN``, not zero."""
        return [n for n, c in zip(self.kc_names, self.n_seq_per_kc) if c == 0]

    @property
    def unestimable(self) -> list[str]:
        """KCs no student ever practises twice.

        With no second opportunity there is no transition to observe, so
        ``learn`` (and ``forget``) have nothing to estimate from. Reported, and
        **not charged for** in :attr:`n_params` -- the same treatment AFM gives
        a KC whose slope cannot exist.
        """
        return [n for n, t, c in zip(self.kc_names, self.n_transitions, self.n_seq_per_kc)
                if c > 0 and t == 0]

    @property
    def degenerate(self) -> list[str]:
        """KCs whose fit says guessing beats knowing (``G + S >= 1``).

        A documented failure mode of unconstrained BKT (Baker, Corbett & Aleven
        2008): the likelihood is happy, but the fitted "knowledge" state is
        anti-correlated with correctness and the parameters no longer mean what
        their names say. Flagged rather than silently repaired; pass
        ``bound_degenerate=True`` to :func:`fit_bkt` to keep the fit out of that
        region instead.
        """
        total = self.guess + self.slip
        return [n for n, t in zip(self.kc_names, total) if np.isfinite(t) and t >= 1.0]

    @property
    def n_params(self) -> int:
        """Free parameters that actually exist.

        Three per fitted KC (``L0``, ``G``, ``S``), plus ``T`` -- and ``F`` when
        fitted -- only for KCs that have an observed transition to estimate them
        from. An unfitted KC contributes nothing.
        """
        fitted = np.asarray(self.n_seq_per_kc) > 0
        estimable = fitted & (np.asarray(self.n_transitions) > 0)
        per_transition = 2 if self.has_forget else 1
        return int(3 * fitted.sum() + per_transition * estimable.sum())

    @property
    def restart_spread(self) -> np.ndarray:
        """Per KC, the range of final log-likelihoods across EM restarts.

        This is what stands in for AFM's optimality certificate, and it is
        weaker on purpose. BKT's likelihood is not concave, so no run can
        prove it found the global optimum. A spread of 0 across many restarts
        is evidence that the surface has one basin worth reaching; a large
        spread says the reported fit is one of several and the choice among
        them was made by the seed.
        """
        if self.restart_lls.size == 0:
            return np.zeros(len(self.kc_names))
        return self.restart_lls.max(axis=1) - self.restart_lls.min(axis=1)

    @property
    def worst_final_improvement(self) -> float:
        """The largest last-step EM took across KCs, in nats.

        The closest thing BKT has to AFM's ``max_free_gradient``, and a weaker
        thing: it bounds the improvement of the *next* iteration, not the
        likelihood left on the table. On a flat ridge those differ by orders of
        magnitude, which is why it is reported rather than turned into a
        pass/fail. Read it together with :attr:`restart_spread`.
        """
        live = np.asarray(self.final_improvement)[np.asarray(self.n_seq_per_kc) > 0]
        return float(np.max(live, initial=0.0))

    @property
    def n_restarts(self) -> int:
        return int(self.restart_lls.shape[1]) if self.restart_lls.size else 1

    @property
    def aic(self) -> float:
        return -2.0 * self.ll + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.ll + self.n_params * np.log(max(self.n_obs, 1))

    # -- reporting ----------------------------------------------------------------

    def params(self) -> pd.DataFrame:
        """One row per KC: the five probabilities and the flags that qualify them."""
        return pd.DataFrame({
            "KC Name": self.kc_names,
            "Prior (L0)": self.prior,
            "Learn (T)": self.learn,
            "Guess (G)": self.guess,
            "Slip (S)": self.slip,
            "Forget (F)": self.forget,
            "Sequences": self.n_seq_per_kc,
            "Transitions": self.n_transitions,
            "Degenerate": [n in set(self.degenerate) for n in self.kc_names],
            "Unestimable": [n in set(self.unestimable) for n in self.kc_names],
            "Restart Spread": self.restart_spread,
            "Final EM Step": self.final_improvement,
        }).sort_values("KC Name", ignore_index=True)

    def kc_values(self) -> pd.DataFrame:
        """The parameter table with ``learn`` blanked where it cannot exist.

        A KC nobody practises twice reports ``Learn = NaN``, not ``0`` -- the
        same rule :meth:`leapfit.afm.AFMFit.kc_values` applies to a slope with
        no second opportunity behind it, and for the same reason: printing a
        number invites it into a screen that reads zero as "did not learn".
        """
        frame = self.params()
        blank = frame["Transitions"].to_numpy() == 0
        frame.loc[blank, ["Learn (T)", "Forget (F)"]] = np.nan
        return frame

    def summary(self) -> str:
        # An unfitted KC has no restarts, so its spread is NaN rather than 0 —
        # nothing was measured there, which is not the same as "measured, and flat".
        measured = self.restart_spread[np.isfinite(self.restart_spread)]
        spread = float(np.max(measured, initial=0.0))
        fitted = int(np.sum(np.asarray(self.n_seq_per_kc) > 0))
        lines = [
            (f"{self.label} | n = {self.n_obs:,} | params = {self.n_params:,} | "
             f"{fitted} KC(s), {self.n_seq:,} sequence(s)"),
            f"  log-likelihood {self.ll:12.4f}",
            f"  AIC            {self.aic:12.4f}",
            f"  BIC            {self.bic:12.4f}",
            (f"  EM             {int(np.max(self.n_iter, initial=0))} iteration(s) max, "
             f"{int(np.sum(~np.asarray(self.converged, dtype=bool)))} KC(s) hit the cap, "
             f"last step {self.worst_final_improvement:.3g} nats"),
            f"  restarts       {self.n_restarts}, worst spread {spread:.3g} nats"
            + ("  (multimodal — the fit depends on the seed)" if spread > 1e-3 else ""),
        ]
        if not np.all(self.monotone):
            bad = int(np.sum(~np.asarray(self.monotone, dtype=bool)))
            lines.append(f"  *** EM decreased the likelihood on {bad} KC(s) — this is a bug ***")
        if self.unfitted:
            lines.append(f"  unfitted       {len(self.unfitted)} KC(s) with no data: "
                         f"{', '.join(self.unfitted[:4])}")
        if self.unestimable:
            lines.append(f"  unestimable    {len(self.unestimable)} KC(s) never practised "
                         f"twice — no learn rate: {', '.join(self.unestimable[:4])}")
        if self.degenerate:
            lines.append(f"  degenerate     {len(self.degenerate)} KC(s) with G + S >= 1: "
                         f"{', '.join(self.degenerate[:4])}")
        return "\n".join(lines)

    # -- prediction ---------------------------------------------------------------

    def predict_proba(self, sequences: Sequences) -> np.ndarray:
        """One-step-ahead ``P(correct)`` per source row, ``NaN`` where unpredictable.

        The prediction for a response is made from everything *before* it in the
        same sequence, which is what a BKT prediction is. Note the consequence
        for cross-validation: a held-out response is still predicted using its
        own student's earlier responses, including ones that were in the
        training split. That is pyBKT's convention and the only one the model
        admits -- a filter has no way to predict a mid-sequence response
        without the history that precedes it -- but it means BKT's held-out
        score is not measuring the same independence a design-matrix model's
        is. What is held out is the *parameter estimation*, not the history.
        """
        if list(sequences.kc_names) != list(self.kc_names):
            raise ValueError(
                f"The fit covers {len(self.kc_names)} KCs and these sequences carry "
                f"{len(sequences.kc_names)} — they come from different KC models."
            )
        out = np.full(sequences.n_source_rows, np.nan)
        for k in range(len(self.kc_names)):
            selected = sequences.kc_of_seq == k
            if not selected.any() or not np.isfinite(self.prior[k]):
                continue
            mask = sequences.mask[selected]
            width = int(mask.sum(axis=1).max(initial=0))
            if width == 0:
                continue
            mask = mask[:, :width]
            state = bkt_filter(sequences.y[selected][:, :width], mask,
                               self.prior[k], self.learn[k], self.guess[k],
                               self.slip[k], self.forget[k])
            out[sequences.rows[selected][:, :width][mask]] = state["p_correct"][mask]
        return out

    def mastery(self, sequences: Sequences) -> np.ndarray:
        """``P(known)`` before each response, per source row. ``NaN`` where unpredictable."""
        out = np.full(sequences.n_source_rows, np.nan)
        for k in range(len(self.kc_names)):
            selected = sequences.kc_of_seq == k
            if not selected.any() or not np.isfinite(self.prior[k]):
                continue
            mask = sequences.mask[selected]
            width = int(mask.sum(axis=1).max(initial=0))
            if width == 0:
                continue
            mask = mask[:, :width]
            state = bkt_filter(sequences.y[selected][:, :width], mask,
                               self.prior[k], self.learn[k], self.guess[k],
                               self.slip[k], self.forget[k])
            out[sequences.rows[selected][:, :width][mask]] = state["predicted"][:, :, 1][mask]
        return out

    def brier(self, sequences: Sequences, y) -> float:
        """Mean squared error on the probability scale, over predictable rows."""
        predicted = self.predict_proba(sequences)
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(predicted)
        return float(np.mean((y[keep] - predicted[keep]) ** 2))

    def rmse(self, sequences: Sequences, y) -> float:
        return float(np.sqrt(self.brier(sequences, y)))

    def annotate(self, data: StepData, *, into: pd.DataFrame | None = None) -> pd.DataFrame:
        """The source table plus a ``Predicted Error Rate (<model>)`` column.

        Student-step in, student-step out, following the same DataShop
        convention as :meth:`leapfit.fit.LogisticFit.annotate`: the value is
        the predicted **error** rate ``1 - P(correct)``, and rows that entered
        no fit stay blank.
        """
        if data.source is None or data.source_rows is None:
            raise ValueError(
                "This StepData does not carry its source table, so there is nothing to "
                "annotate. Build it with load_student_step() or from_frame()."
            )
        sequences = extract_sequences(data)
        if sequences.n_source_rows != len(data):
            raise ValueError("The sequences and the data describe different files.")

        out = data.source.copy() if into is None else into
        if len(out) != len(data.source):
            raise ValueError(
                f"'into' has {len(out)} rows but the source table has {len(data.source)}."
            )
        values = np.full(len(data.source), np.nan)
        values[data.source_rows] = 1.0 - self.predict_proba(sequences)
        out[f"Predicted Error Rate ({data.kc_model})"] = values
        return out


# ==================================================================================
# The fitter
# ==================================================================================

def fit_bkt(sequences: Sequences, *, forget: bool = False,
            bound_degenerate: bool = False, n_restarts: int = 1,
            seed: int | None = 0, tol: float = DEFAULT_TOL,
            max_iter: int = DEFAULT_MAX_ITER, warn_not_converged: bool = True,
            warn_degenerate: bool = True) -> BKTFit:
    """Fit BKT by expectation-maximization, one independent chain per KC.

    Every parameter is per-KC and no KC's data touches another's, so this is
    ``n_kcs`` separate Baum-Welch runs and the total log-likelihood is their
    sum. That is also why a bad KC cannot contaminate a good one: a degenerate
    or unestimable KC is flagged on its own row.

    :param forget: fit a forgetting probability ``F``. Classical BKT pins it at
        0, which is the default; a model being compared against one with an
        explicit forgetting mechanism should be allowed it.
    :param bound_degenerate: keep ``G`` and ``S`` at or below 0.5 so that
        ``G + S < 1`` holds by construction. Off by default because the
        reference implementations fit unconstrained, and because a bound that
        binds is more honestly reported as :attr:`BKTFit.degenerate` than
        hidden inside the fit.
    :param n_restarts: EM runs per KC. Restart 0 always starts from
        :data:`DEFAULT_INIT`, so ``n_restarts=1`` is deterministic and ignores
        ``seed``; further restarts are drawn from ``seed``. BKT's likelihood is
        multimodal, and :attr:`BKTFit.restart_spread` is the only evidence
        available about whether that mattered here — running one restart does
        not make the problem unimodal, only unmeasured.
    :param seed: seeds the random restarts. Fixed at 0 rather than ``None`` so
        that a multi-restart fit is reproducible unless you ask otherwise.
    :param warn_not_converged: warn when some KC's EM stops on the iteration cap.
    :param warn_degenerate: warn when some KC fits ``G + S >= 1``.
    """
    n_kcs = len(sequences.kc_names)
    root = np.random.default_rng(seed)
    streams = root.spawn(n_kcs) if n_kcs else []

    prior = np.full(n_kcs, np.nan)
    learn = np.full(n_kcs, np.nan)
    guess = np.full(n_kcs, np.nan)
    slip = np.full(n_kcs, np.nan)
    forgetting = np.full(n_kcs, np.nan)
    n_seq_per_kc = np.zeros(n_kcs, dtype=int)
    n_transitions = np.zeros(n_kcs, dtype=int)
    converged = np.ones(n_kcs, dtype=bool)
    monotone = np.ones(n_kcs, dtype=bool)
    n_iter = np.zeros(n_kcs, dtype=int)
    final_improvement = np.zeros(n_kcs)
    restart_lls = np.full((n_kcs, max(1, n_restarts)), np.nan)
    total_ll = 0.0

    for k in range(n_kcs):
        y, mask = sequences.for_kc(k)
        n_seq_per_kc[k] = y.shape[0]
        if y.shape[0] == 0 or not mask.any():
            # No data: leave the parameters NaN. A zero here would read as a
            # fitted probability of zero, which is a claim, not a gap.
            continue
        best, scores = _fit_one_kc(y, mask, fit_forget=forget,
                                   bound_degenerate=bound_degenerate,
                                   n_restarts=n_restarts, rng=streams[k],
                                   tol=tol, max_iter=max_iter)
        prior[k], learn[k] = best.prior, best.learn
        guess[k], slip[k], forgetting[k] = best.guess, best.slip, best.forget
        n_transitions[k] = best.n_transitions
        converged[k], monotone[k], n_iter[k] = best.converged, best.monotone, best.n_iter
        final_improvement[k] = best.final_improvement
        restart_lls[k, :len(scores)] = scores
        total_ll += best.ll

    fit = BKTFit(
        kc_names=list(sequences.kc_names), prior=prior, learn=learn, guess=guess,
        slip=slip, forget=forgetting, ll=float(total_ll), n_obs=sequences.n_obs,
        n_seq=len(sequences), n_seq_per_kc=n_seq_per_kc, n_transitions=n_transitions,
        converged=converged, monotone=monotone, n_iter=n_iter,
        final_improvement=final_improvement, restart_lls=restart_lls,
        has_forget=forget, bound_degenerate=bound_degenerate,
        kc_model=sequences.kc_model,
    )

    if warn_not_converged and not converged.all():
        warnings.warn(
            f"EM hit the {max_iter}-iteration cap on {int((~converged).sum())} of "
            f"{n_kcs} KC(s); their parameters are wherever it stopped. Raise max_iter "
            "or loosen tol.", RuntimeWarning, stacklevel=2)
    if warn_degenerate and fit.degenerate:
        warnings.warn(
            f"{len(fit.degenerate)} KC(s) fitted G + S >= 1, where 'known' is the state "
            f"that predicts failure: {', '.join(fit.degenerate[:4])}. The likelihood is "
            "fine; the parameter names are not. Pass bound_degenerate=True to exclude "
            "that region.", RuntimeWarning, stacklevel=2)
    if not monotone.all():
        warnings.warn(
            f"EM decreased the log-likelihood on {int((~monotone).sum())} KC(s). EM is "
            "monotone by construction, so this is a numerical defect in the fit, not a "
            "property of the data — please report it.", RuntimeWarning, stacklevel=2)
    return fit


def bkt_from_data(data: StepData, **kwargs) -> tuple[Sequences, BKTFit]:
    """Reshape and fit in one call, returning both halves.

    The sequences come back because everything downstream of the fit needs them
    -- :meth:`BKTFit.predict_proba`, :meth:`BKTFit.rmse` and cross-validation
    all take sequences, not a ``StepData``.
    """
    sequences = extract_sequences(data)
    return sequences, fit_bkt(sequences, **kwargs)


# ==================================================================================
# Cross-validation
#
# BKT does not go through :func:`leapfit.crossval.cross_validate`, which fits a
# :class:`~leapfit.design.Design` per fold. It reuses that module's fold schemes
# and result types so the numbers land in the same table, and runs its own fits.
# The import direction is deliberate and one-way: a model family may use the
# shared layer, never the reverse.
# ==================================================================================

def _restrict(sequences: Sequences, keep: np.ndarray) -> Sequences:
    """The same sequences with non-kept observations removed and the gaps closed.

    Removing a mid-sequence response shortens that student's history rather than
    punching a hole in it -- there is no "missing" state for a filter to carry
    across. This is what makes a training split well defined for every fold
    scheme, not only ``student_blocked``.
    """
    keep = np.asarray(keep, dtype=bool)
    live = sequences.mask
    y_by_row = np.zeros(sequences.n_source_rows, dtype=np.int8)
    y_by_row[sequences.rows[live]] = sequences.y[live]

    surviving, kept = [], []
    for s in range(len(sequences)):
        rows = sequences.rows[s][live[s]]
        rows = rows[keep[rows]]
        if rows.size:
            surviving.append(s)
            kept.append(rows)

    width = max((len(r) for r in kept), default=1)
    n = len(surviving)
    y = np.zeros((n, width), dtype=np.int8)
    mask = np.zeros((n, width), dtype=bool)
    rows_out = np.full((n, width), -1, dtype=int)
    for i, rows in enumerate(kept):
        y[i, :len(rows)] = y_by_row[rows]
        mask[i, :len(rows)] = True
        rows_out[i, :len(rows)] = rows

    return Sequences(y=y, mask=mask, rows=rows_out,
                     kc_of_seq=sequences.kc_of_seq[surviving],
                     students=[sequences.students[i] for i in surviving],
                     kc_names=sequences.kc_names, kc_model=sequences.kc_model,
                     n_source_rows=sequences.n_source_rows)


def cross_validate_bkt(data: StepData, *, scheme: str = "student_blocked",
                       n_folds: int = 3, seed: int | None = None,
                       convention: str = "per_fold", restart_seed: int | None = 0,
                       **fit_kwargs):
    """Refit BKT on each training split and score the held-out rows.

    Returns a :class:`~leapfit.crossval.CVResult`, so a BKT row sits in the same
    comparison table as an AFM or PFA one -- but read the ``scheme`` carefully,
    because they do not mean quite the same thing here.

    **What is held out.** Only the *parameter estimation*. Held-out responses are
    removed from the sequences EM sees, so they contribute nothing to ``L0``,
    ``T``, ``G`` or ``S``. They are then predicted by running the filter over the
    full sequence, which means a mid-sequence held-out response is still
    predicted from its own student's earlier responses. That is not a leak of
    the response being scored, but it is a weaker form of independence than a
    design-matrix model gets from the same fold, and comparing the two RMSEs
    is comparing two slightly different questions.

    ``student_blocked`` is the default for exactly that reason: a held-out
    student contributes no history to anyone, so it is the one scheme under
    which BKT's held-out score means what it appears to mean.

    :param seed: seeds the **fold partition**, as it does everywhere else in the
        package.
    :param restart_seed: seeds EM's **random restarts**, which are a different
        randomness with a different job. They are named apart because
        collapsing them would tie a model's held-out score to the partition it
        was scored on through the optimizer, and a change in folds would then
        move the number for two reasons at once.
    :param fit_kwargs: passed to :func:`fit_bkt` (``forget``, ``n_restarts``, ...).
    """
    from leapfit.crossval import CONVENTIONS, CVResult, FoldResult, make_folds

    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")

    full = extract_sequences(data)
    y = np.asarray(data.y, dtype=float)
    folds = make_folds(data, scheme, n_folds, seed, convention)

    results, residuals = [], []
    for index, test_idx in enumerate(folds):
        keep = np.ones(len(data), dtype=bool)
        keep[test_idx] = False
        if not keep.any() or len(test_idx) == 0:
            raise ValueError(
                f"Fold {index} is degenerate: {int(keep.sum())} train / {len(test_idx)} test")

        fit = fit_bkt(_restrict(full, keep), seed=restart_seed,
                      warn_not_converged=False, warn_degenerate=False, **fit_kwargs)
        predicted = fit.predict_proba(full)[test_idx]

        # A held-out row whose KC never appeared in training has no parameters to
        # be predicted from. Counted, not quietly imputed — the same role
        # ``unseen_column_fraction`` plays for a design.
        unseen = ~np.isfinite(predicted)
        scored = ~unseen
        residual = y[test_idx][scored] - predicted[scored]
        residuals.append(residual)
        results.append(FoldResult(
            fold=index, n_train=int(keep.sum()), n_test=len(test_idx),
            rmse=float(np.sqrt(np.mean(residual ** 2))) if residual.size else float("nan"),
            unseen_column_fraction=float(unseen.mean()),
            converged=bool(np.all(fit.converged))))

    if convention == "per_fold":
        rmse = float(np.mean([f.rmse for f in results]))
    else:
        pooled = np.concatenate(residuals)
        rmse = float(np.sqrt(np.mean(pooled ** 2)))
    return CVResult(scheme=scheme, convention=convention, n_folds=n_folds,
                    seed=seed, rmse=rmse, folds=tuple(results))
