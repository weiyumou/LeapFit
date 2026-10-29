"""Bayesian Knowledge Tracing: a two-state HMM per knowledge component, fitted by EM.

BKT is the first model in this package that is not a logistic regression, so it
brings its own fitter and shares only :mod:`leapfit.data` and the fold schemes
of :mod:`leapfit.crossval`. What it does *not* bring is a different input
format: it reads the same :class:`~leapfit.data.StepData` every other family
reads, and reshapes it here.

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
NLL; the arithmetic and the logit-space guards carry over, the fitter does not.
Validation follows that project's *transplantation* protocol against two
references -- pyBKT (Badrinath, Wang & Pardos 2021) and Yudelson's
``hmm-scalable``, the C++ tool behind LearnSphere's ``AnalysisBkt`` component:
load each reference's fitted parameters into this recursion and compare
predictions row by row, which separates "is this the same model" from "is this
the same fitting procedure". ``scripts/validate_bkt.py`` runs both.

The fitter is Baum-Welch, as in both references, and there is one of it: every
(KC, restart) pair is an independent two-state chain, so all of them are run
through the same forward-backward pass at once, as rows of one padded array,
and a chain that has converged is dropped from the array while the others go
on. That is where the speed comes from -- the Python-level loop is over the
*length* of a sequence, never over KCs or restarts -- and ``n_jobs`` then
spreads batches of chains over processes. Neither changes a number: the
batches are cut deterministically and the per-chain arithmetic is the same
whichever batch a chain lands in.

DIVERGENCE (``G + S < 1``, always): both references fit guess and slip without
a joint constraint (``hmm-scalable`` caps each at 0.3 by default; pyBKT caps
neither), and both can return fits in which the "known" state is the one that
predicts failure. The names then no longer mean what they say, and the
likelihood is no help: on real data the label-swapped optimum is routinely the
higher one. This fitter requires ``G + S < 1`` for every KC -- the state called
"known" is the state more likely to answer correctly -- by maximizing the
M-step's objective *on* the boundary when the unconstrained update would cross
it, which keeps EM monotone. A KC whose fit sits on that boundary is reported
in :attr:`BKTFit.at_bound`: there, the data prefer the swap, and the
constrained fit is a model in which knowing the KC makes no difference. That
is a finding about the KC, not a repaired estimate, and it is printed as one.

DIVERGENCE (no optimality certificate): every other fit in this package carries
one -- the logistic objective is convex, so satisfying the KKT conditions *is*
a proof of global optimality. BKT's likelihood is not concave, and no such
certificate exists for it under EM or under anything else. :attr:`BKTFit.converged`
therefore means only that EM stopped on its tolerance, and
:attr:`BKTFit.restart_spread` is the evidence offered in place of a proof. Do
not read the two as equivalent.

DIVERGENCE (multi-KC steps): a DataShop step may be tagged with several KCs,
which AFM absorbs additively; BKT is defined on one skill at a time and has no
additive form. Two conventions are offered, named, and neither is silent:

* ``multi_kc="refuse"`` (default): :func:`extract_sequences` raises and names
  the offending rows. A step with several KCs is one observation for AFM and
  would be several for BKT, so ``N`` -- and with it AIC and BIC -- would differ
  between two rows of the same table.
* ``multi_kc="replicate"``: LearnSphere's convention. Each (step, KC) pair is
  one observation in that KC's sequence, and a step's prediction is the mean of
  its KCs' predictions. ``N`` is then the number of pairs, which
  :attr:`BKTFit.n_obs` reports next to :attr:`BKTFit.n_rows`, so the inflation
  is visible wherever the statistics are.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from leapfit.crossval import CONVENTIONS, CVResult, FoldResult, make_folds
from leapfit.data import StepData, _sequences

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
#: guess/slip pair sits well inside the ``G + S < 1`` region, so restart 0
#: begins somewhere defensible rather than at random.
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
#: fixed points include the corners, and a restart that starts in one never
#: leaves it. Guess/slip pairs are redrawn until ``G + S < 1``.
_RESTART_RANGE = (0.05, 0.95)

#: The conventions for a step tagged with several KCs. See the module docstring.
MULTI_KC = ("refuse", "replicate")

#: Cells -- sequence x step x restart -- per EM batch. Bounds the memory one
#: forward-backward pass touches (about a hundred bytes per cell) and gives
#: ``n_jobs`` something to spread; it does not change any number. Measured on
#: the E-learning 2022 export, 100k is the fastest budget in one process (a
#: smaller batch pads less and sheds its converged chains sooner) and within
#: a third of the fastest with four; 500k was 30% slower serially.
BATCH_CELLS = 100_000

PARAM_NAMES = ("prior", "learn", "guess", "slip", "forget")


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
    row of the export. Under ``multi_kc="replicate"`` a row appears once per
    KC it carries, in that many sequences.
    """

    y: np.ndarray             # (n_seq, max_len) int8 in {0, 1}, 0-padded
    mask: np.ndarray          # (n_seq, max_len) bool, True where a real response
    rows: np.ndarray          # (n_seq, max_len) int, -1 where padded
    kc_of_seq: np.ndarray     # (n_seq,) index into kc_names
    students: list[str]       # (n_seq,)
    kc_names: list[str]
    kc_model: str
    n_source_rows: int        # len() of the StepData these came from
    multi_kc: str = "refuse"

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_obs(self) -> int:
        """Observations: (row, KC) pairs, which is rows unless replicated."""
        return int(self.mask.sum())

    @property
    def n_rows(self) -> int:
        """Source rows that appear in at least one sequence."""
        return int(np.unique(self.rows[self.mask]).size)

    @property
    def lengths(self) -> np.ndarray:
        return self.mask.sum(axis=1)

    def for_kc(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """This KC's sequences, trimmed to their own longest."""
        selected = self.kc_of_seq == k
        mask = self.mask[selected]
        width = int(mask.sum(axis=1).max(initial=0))
        return self.y[selected][:, :width], mask[:, :width]

    def take(self, seq_index: np.ndarray) -> Sequences:
        """A subset of sequences, re-trimmed to the longest kept."""
        seq_index = np.asarray(seq_index)
        mask = self.mask[seq_index]
        width = max(int(mask.sum(axis=1).max(initial=0)), 1)
        return Sequences(
            y=self.y[seq_index][:, :width], mask=mask[:, :width],
            rows=self.rows[seq_index][:, :width], kc_of_seq=self.kc_of_seq[seq_index],
            students=[self.students[i] for i in seq_index], kc_names=self.kc_names,
            kc_model=self.kc_model, n_source_rows=self.n_source_rows,
            multi_kc=self.multi_kc,
        )


def extract_sequences(data: StepData, *, multi_kc: str = "refuse") -> Sequences:
    """Reshape parsed student-step data into per-(student, KC) sequences.

    Order comes from :meth:`~leapfit.data.StepData.practice_order` -- the same
    canonical ordering AFM's opportunity counts and PFA's prior-success counts
    are built over, so "before" means one thing across the package.

    :param multi_kc: what to do with a step that carries several KCs:
        ``"refuse"`` raises and names the rows; ``"replicate"`` puts the step
        into each KC's sequence, LearnSphere's convention. See the module
        docstring for why the default is to refuse.
    :raises ValueError: on a multi-KC step under ``"refuse"``, or on an
        observation with no KC at all.
    """
    if multi_kc not in MULTI_KC:
        raise ValueError(f"multi_kc must be one of {MULTI_KC}, got {multi_kc!r}")
    multi = [i for i, labels in enumerate(data.kcs) if len(labels) > 1]
    if multi and multi_kc == "refuse":
        shown = ", ".join(str(i) for i in multi[:5])
        more = f" (and {len(multi) - 5} more)" if len(multi) > 5 else ""
        raise ValueError(
            f"BKT needs one KC per step, but {len(multi)} of {len(data)} rows in KC "
            f"model {data.kc_model!r} carry several -- observations {shown}{more}. "
            "BKT is defined on a single skill and has no additive form, so the "
            "convention for multi-KC steps changes N and therefore AIC/BIC. Pass "
            "multi_kc='replicate' for LearnSphere's convention (one observation per "
            "(step, KC) pair, predictions averaged per step), or fit a single-KC model."
        )
    empty = [i for i, labels in enumerate(data.kcs) if not labels]
    if empty:
        raise ValueError(
            f"{len(empty)} observation(s) carry no KC at all; load_student_step() drops "
            "those, so this StepData was probably built by hand."
        )

    # Every (observation, KC) pair, grouped into (student, KC) runs in practice
    # order — the same layout LKT's features are built over.
    layout = _sequences(data, data.kcs)
    starts = np.asarray(layout.starts, dtype=int)
    n_seq = max(len(starts) - 1, 0)
    lengths = np.diff(starts) if n_seq else np.zeros(0, dtype=int)
    max_len = int(lengths.max(initial=1))

    run = np.repeat(np.arange(n_seq), lengths)
    position = np.arange(len(layout.rows)) - starts[run]

    y = np.zeros((n_seq, max_len), dtype=np.int8)
    mask = np.zeros((n_seq, max_len), dtype=bool)
    rows = np.full((n_seq, max_len), -1, dtype=int)
    y[run, position] = layout.y.astype(np.int8)
    mask[run, position] = True
    rows[run, position] = layout.rows
    kc_of_seq = layout.columns[starts[:-1]] if n_seq else np.zeros(0, dtype=int)
    students = [data.students[layout.rows[s]] for s in starts[:-1]]

    return Sequences(y=y, mask=mask, rows=rows, kc_of_seq=np.asarray(kc_of_seq, dtype=int),
                     students=students, kc_names=list(layout.levels),
                     kc_model=data.kc_model, n_source_rows=len(data), multi_kc=multi_kc)


# ==================================================================================
# The recursion
# ==================================================================================

def _per_sequence(value, n: int) -> np.ndarray:
    """A parameter given once, or once per sequence, as a ``(n,)`` float array."""
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        return np.full(n, float(value))
    if value.shape != (n,):
        raise ValueError(f"expected a scalar or {n} per-sequence values, got shape {value.shape}")
    return value


def bkt_filter(y: np.ndarray, mask: np.ndarray, prior, learn, guess, slip,
               forget=0.0) -> dict:
    """Run BKT forward over padded sequences: predict, update, transition.

    This is the scaled forward pass of the HMM, written in BKT's own terms. It
    is the single place the recursion lives -- predictions read ``p_correct``
    off it, and the E-step reads ``filtered`` and ``scale``.

    Each parameter is a scalar shared by every row of ``y``, or one value per
    row, which is how one call runs many KCs (or many restarts of one KC) at
    once. Padding is inert: past a sequence's length the state is frozen and
    the scale is 1, so a padded step contributes nothing to any sum and
    nothing to the log-likelihood.

    :return: a dict with

        ``p_correct`` (n, L)
            ``P(y_t = 1 | y_<t)`` -- the one-step-ahead prediction, which is
            what "a BKT prediction" means and what both references return.
        ``predicted`` (n, L, 2)
            ``P(K_t | y_<t)`` -- mastery *before* the response is seen.
        ``filtered`` (n, L, 2)
            ``P(K_t | y_<=t)`` -- after the Bayes update. The scaled forward
            variable the E-step needs.
        ``scale`` (n, L)
            ``P(y_t | y_<t)`` for the response actually observed; its masked
            log-sum is the log-likelihood.
        ``emission`` (n, L, 2)
            ``P(y_t | K_t)``, kept so the backward pass need not rebuild it.
    """
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    n, length = y.shape
    prior, learn, guess, slip, forget = (
        _per_sequence(v, n) for v in (prior, learn, guess, slip, forget))

    correct = np.stack([guess, 1.0 - slip], axis=1)         # P(y = 1 | unknown), (.. | known)
    emission = np.where(y[..., None] == 1, correct[:, None, :], 1.0 - correct[:, None, :])
    stay_unknown, stay_known = 1.0 - learn, 1.0 - forget

    predicted = np.empty((n, length, 2))
    filtered = np.empty((n, length, 2))
    p_correct = np.zeros((n, length))
    scale = np.ones((n, length))

    state = np.stack([1.0 - prior, prior], axis=1)          # P(K_1) = L0
    for t in range(length):
        live = mask[:, t]
        keep = live[:, None]

        # --- 1. Predict ---
        predicted[:, t] = state
        p_correct[:, t] = state[:, 0] * correct[:, 0] + state[:, 1] * correct[:, 1]

        # --- 2. Update: Bayes on the observed response. ---
        unnormalized = state * emission[:, t]
        evidence = np.where(live, unnormalized.sum(axis=1), 1.0).clip(EPS)
        now = np.where(keep, unnormalized / evidence[:, None], state)
        filtered[:, t] = now
        scale[:, t] = evidence

        # --- 3. Transition: learn out of unknown, forget out of known. ---
        moved = np.stack([now[:, 0] * stay_unknown + now[:, 1] * forget,
                          now[:, 0] * learn + now[:, 1] * stay_known], axis=1)
        state = np.where(keep, moved, state)

    return {"p_correct": p_correct, "predicted": predicted, "filtered": filtered,
            "scale": scale, "emission": emission}


def _backward(emission: np.ndarray, mask: np.ndarray, scale: np.ndarray,
              learn: np.ndarray, forget: np.ndarray) -> np.ndarray:
    """Scaled backward variable ``beta_t(i) = P(y_>t | K_t = i) / prod(scale_>t)``.

    Scaled by the same constants the forward pass divided by, so that
    ``filtered * beta`` is already the posterior and needs no renormalizing.
    """
    n, length, _ = emission.shape
    stay_unknown, stay_known = 1.0 - learn, 1.0 - forget
    beta = np.ones((n, length, 2))
    for t in range(length - 2, -1, -1):
        ahead = emission[:, t + 1] * beta[:, t + 1] / scale[:, t + 1][:, None]
        back = np.stack([stay_unknown * ahead[:, 0] + learn * ahead[:, 1],
                         forget * ahead[:, 0] + stay_known * ahead[:, 1]], axis=1)
        beta[:, t] = np.where(mask[:, t + 1][:, None], back, beta[:, t + 1])
    return beta


# ==================================================================================
# EM over a batch of chains
#
# A *chain* is one (KC, restart) pair: its own five parameters, its own
# sequences, its own stopping rule. Chains share nothing but the array they
# are stacked in, which is what lets one forward-backward pass serve all of
# them. ``chain[s]`` says which chain sequence ``s`` belongs to.
# ==================================================================================

@dataclass
class _ChainFits:
    """What EM produced for every chain of one batch, as arrays over chains."""

    prior: np.ndarray
    learn: np.ndarray
    guess: np.ndarray
    slip: np.ndarray
    forget: np.ndarray
    ll: np.ndarray
    n_iter: np.ndarray
    converged: np.ndarray
    monotone: np.ndarray        # EM never decreased the likelihood -- a numerics check
    final_improvement: np.ndarray  # the last step EM took, in nats
    at_bound: np.ndarray        # the G + S < 1 constraint was active at the end


def _identified(guess: np.ndarray, slip: np.ndarray, a: np.ndarray, b: np.ndarray,
                c: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The M-step for (G, S), held to ``G + S < 1``.

    The unconstrained update ``G = a / (a + b)``, ``S = c / (c + d)`` -- ``a``
    and ``b`` the expected correct and incorrect counts in the unknown state,
    ``c`` and ``d`` the expected incorrect and correct counts in the known
    state -- maximizes ``Q = a log G + b log(1 - G) + c log S + d log(1 - S)``,
    which is concave and separable. Where that update lands on or past the
    line ``G + S = 1``, the constrained maximum lies *on* the line, so it is
    found there by bisection on the (monotone) derivative. Because the
    replacement maximizes the same ``Q`` over a set containing the current
    parameters, EM's monotonicity survives the constraint.
    """
    ceiling = 1.0 - EPS                        # the boundary G + S = ceiling
    over = (guess + slip) >= ceiling
    if not over.any():
        return guess, slip
    a, b, c, d = (v[over] for v in (a, b, c, d))
    lo = np.full(a.shape, EPS)
    hi = np.full(a.shape, ceiling - EPS)
    for _ in range(100):                       # 1e-30 of the interval; more than enough
        g = 0.5 * (lo + hi)
        s = ceiling - g
        slope = a / g - b / (1.0 - g) - c / s + d / (1.0 - s)   # dQ/dG along the line
        lo = np.where(slope > 0, g, lo)
        hi = np.where(slope > 0, hi, g)
    g = 0.5 * (lo + hi)
    guess, slip = guess.copy(), slip.copy()
    guess[over], slip[over] = g, ceiling - g
    return guess, slip


def _em_chains(y: np.ndarray, mask: np.ndarray, chain: np.ndarray, n_chains: int,
               init: np.ndarray, *, fit_forget: bool, tol: float,
               max_iter: int) -> _ChainFits:
    """Baum-Welch for a batch of independent chains, stopping each on its own.

    ``init`` is ``(n_chains, 5)`` in :data:`PARAM_NAMES` order. A chain that
    has converged (or has no sequences) is frozen and its rows are dropped from
    the arrays, so the batch shrinks to the chains still moving.

    EM is monotone by construction, so the loop tracks whether that actually
    held: a decrease is not a bad fit, it is a bug, and it is reported rather
    than tolerated.
    """
    init = np.asarray(init, dtype=float)
    prior, learn, guess, slip = (_clip(init[:, i]) for i in range(4))
    # Pinned at 0 for classical BKT; otherwise started strictly inside (0, 1),
    # because both ends are absorbing fixed points of the M-step.
    forget = _clip(init[:, 4]) if fit_forget else np.zeros(n_chains)
    guess, slip = _identified(guess, slip, *(np.ones(n_chains),) * 4)

    ll = np.full(n_chains, -np.inf)
    previous = np.full(n_chains, -np.inf)
    improvement = np.full(n_chains, np.inf)
    monotone = np.ones(n_chains, dtype=bool)
    converged = np.zeros(n_chains, dtype=bool)
    n_iter = np.zeros(n_chains, dtype=int)
    at_bound = np.zeros(n_chains, dtype=bool)

    ys, ms, ch = y, mask, chain
    for iteration in range(1, max_iter + 1):
        if not ch.size:
            break
        pr, le, gu, sl, fo = prior[ch], learn[ch], guess[ch], slip[ch], forget[ch]
        state = bkt_filter(ys, ms, pr, le, gu, sl, fo)
        beta = _backward(state["emission"], ms, state["scale"], le, fo)

        # --- The likelihood of the *current* parameters, per chain. ---
        now = np.bincount(ch, weights=(np.log(state["scale"]) * ms).sum(axis=1),
                          minlength=n_chains)
        live = np.bincount(ch, minlength=n_chains) > 0
        ll[live] = now[live]
        n_iter[live] = iteration
        done = np.zeros(n_chains, dtype=bool)
        if iteration > 1:
            floor = 1e-9 * np.maximum(1.0, np.abs(previous))
            monotone[live & (ll < previous - floor)] = False
            step = np.abs(ll - previous)
            improvement[live] = step[live]
            done = live & (step <= tol * np.maximum(1.0, np.abs(previous)))
            converged[done] = True
        previous[live] = ll[live]
        if iteration == max_iter:
            break           # the cap: parameters stay where `ll` measured them

        # --- E-step: expected counts, summed within each chain. ---
        filtered, scale, emission = state["filtered"], state["scale"], state["emission"]
        transitions = ms[:, 1:]
        gamma = filtered * beta * ms[:, :, None]                 # P(K_t | y)
        ahead = emission[:, 1:] * beta[:, 1:] / scale[:, 1:, None]
        xi01 = (filtered[:, :-1, 0] * le[:, None] * ahead[:, :, 1] * transitions).sum(axis=1)
        xi10 = (filtered[:, :-1, 1] * fo[:, None] * ahead[:, :, 0] * transitions).sum(axis=1)
        leaving0 = (gamma[:, :-1, 0] * transitions).sum(axis=1)
        leaving1 = (gamma[:, :-1, 1] * transitions).sum(axis=1)
        right = ys == 1
        a = (gamma[:, :, 0] * right).sum(axis=1)      # correct while unknown
        b = (gamma[:, :, 0] * ~right * ms).sum(axis=1)  # incorrect while unknown
        c = (gamma[:, :, 1] * ~right * ms).sum(axis=1)  # incorrect while known
        d = (gamma[:, :, 1] * right).sum(axis=1)      # correct while known

        counts = np.bincount(ch, minlength=n_chains).astype(float)
        a, b, c, d, leaving0, leaving1, xi01, xi10, first_known = (
            np.bincount(ch, weights=values, minlength=n_chains)
            for values in (a, b, c, d, leaving0, leaving1, xi01, xi10, gamma[:, 0, 1]))

        # --- M-step: each parameter is a ratio of expected counts. ---
        update = live & ~done
        new_prior = _clip(np.where(counts > 0, first_known / np.maximum(counts, 1), prior))
        new_learn = _clip(np.where(leaving0 > EPS, xi01 / np.maximum(leaving0, EPS), learn))
        new_forget = (_clip(np.where(leaving1 > EPS, xi10 / np.maximum(leaving1, EPS), forget))
                      if fit_forget else forget)
        new_guess = _clip(np.where(a + b > EPS, a / np.maximum(a + b, EPS), guess))
        new_slip = _clip(np.where(c + d > EPS, c / np.maximum(c + d, EPS), slip))
        crossed = (new_guess + new_slip) >= 1.0 - EPS
        new_guess, new_slip = _identified(new_guess, new_slip, a, b, c, d)

        prior[update], learn[update], forget[update] = (
            new_prior[update], new_learn[update], new_forget[update])
        guess[update], slip[update] = new_guess[update], new_slip[update]
        at_bound[update] = crossed[update]

        if done.any():      # drop the frozen chains' rows from the next pass
            keep = ~done[ch]
            ys, ms, ch = ys[keep], ms[keep], ch[keep]
            width = int(ms.sum(axis=1).max(initial=0))
            ys, ms = ys[:, :width], ms[:, :width]

    # A chain that hit the cap broke out before its M-step, so `ll` is the
    # likelihood of the parameters it carries, same as a converged one.
    return _ChainFits(prior=prior, learn=learn, guess=guess, slip=slip, forget=forget,
                      ll=ll, n_iter=n_iter, converged=converged, monotone=monotone,
                      final_improvement=improvement, at_bound=at_bound)


def _run_batch(job: tuple) -> _ChainFits:
    """One batch of chains, in whichever process it lands."""
    y, mask, chain, n_chains, init, fit_forget, tol, max_iter = job
    return _em_chains(y, mask, chain, n_chains, init, fit_forget=fit_forget,
                      tol=tol, max_iter=max_iter)


def _worker_count(n_jobs: int | None, n_tasks: int) -> int:
    """joblib's convention: ``-1`` is every core, ``-2`` all but one."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        n_jobs = (os.cpu_count() or 1) + 1 + n_jobs
    return max(1, min(n_jobs, n_tasks))


def _run_batches(jobs: list[tuple], n_jobs: int | None) -> list[_ChainFits]:
    workers = _worker_count(n_jobs, len(jobs))
    if workers == 1:
        return [_run_batch(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_batch, jobs))


# ==================================================================================
# The fitted model
# ==================================================================================

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
    n_obs: int                 # (row, KC) pairs the fit saw
    n_rows: int                # source rows among them
    n_seq: int
    n_seq_per_kc: np.ndarray   # (n_kcs,)
    n_transitions: np.ndarray  # (n_kcs,) observed chances to learn or forget
    converged: np.ndarray      # (n_kcs,) bool
    monotone: np.ndarray       # (n_kcs,) bool -- EM never went downhill
    n_iter: np.ndarray         # (n_kcs,)
    final_improvement: np.ndarray  # (n_kcs,) the last step EM took, in nats
    restart_lls: np.ndarray    # (n_kcs, n_restarts)
    bound_active: np.ndarray   # (n_kcs,) bool -- G + S < 1 held the final M-step
    has_forget: bool
    kc_model: str
    multi_kc: str = "refuse"
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
    def at_bound(self) -> list[str]:
        """KCs whose fit sits on the ``G + S = 1`` boundary.

        The fitter holds every KC to ``G + S < 1``, so that "known" is the
        state more likely to answer correctly. A KC listed here is one where
        the unconstrained M-step wanted to cross that line -- the data prefer
        the label swap -- and the constrained fit landed on it, where the two
        states predict alike and BKT reduces to the KC's base rate. Not a
        repaired estimate: a finding that BKT's story does not fit this KC.
        """
        return [n for n, hit in zip(self.kc_names, self.bound_active) if hit]

    @property
    def degenerate(self) -> list[str]:
        """KCs with ``G > 0.5`` or ``S > 0.5``, Baker, Corbett & Aleven's (2008) criterion.

        Weaker than :attr:`at_bound` and reported separately because it is
        the criterion the literature screens by: a guess above a half can be
        legitimate (two-option items), a slip above a half rarely is. The
        fitter does not act on it.
        """
        return [n for n, g, s in zip(self.kc_names, self.guess, self.slip)
                if np.isfinite(g) and (g > 0.5 or s > 0.5)]

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
            "At Bound": np.asarray(self.bound_active, dtype=bool),
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
        n = (f"n = {self.n_obs:,}" if self.n_obs == self.n_rows
             else f"n = {self.n_obs:,} (step, KC) pairs over {self.n_rows:,} steps")
        lines = [
            (f"{self.label} | {n} | params = {self.n_params:,} | "
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
        if self.at_bound:
            lines.append(f"  at bound       {len(self.at_bound)} KC(s) held at G + S = 1, "
                         f"where knowing makes no difference: {', '.join(self.at_bound[:4])}")
        if self.degenerate:
            lines.append(f"  degenerate     {len(self.degenerate)} KC(s) with G > 0.5 or "
                         f"S > 0.5: {', '.join(self.degenerate[:4])}")
        return "\n".join(lines)

    # -- prediction ---------------------------------------------------------------

    def _filter(self, sequences: Sequences) -> tuple[dict, np.ndarray]:
        """Run the recursion over every sequence with a fitted KC, at once."""
        if list(sequences.kc_names) != list(self.kc_names):
            raise ValueError(
                f"The fit covers {len(self.kc_names)} KCs and these sequences carry "
                f"{len(sequences.kc_names)} — they come from different KC models."
            )
        k = sequences.kc_of_seq
        usable = np.isfinite(self.prior[k]) if k.size else np.zeros(0, dtype=bool)
        k = k[usable]
        state = bkt_filter(sequences.y[usable], sequences.mask[usable], self.prior[k],
                           self.learn[k], self.guess[k], self.slip[k], self.forget[k])
        return state, usable

    def _per_row(self, sequences: Sequences, values: np.ndarray,
                 usable: np.ndarray) -> np.ndarray:
        """Sequence-space values back onto source rows, averaged where a row recurs."""
        mask = sequences.mask[usable]
        rows = sequences.rows[usable][mask]
        total = np.bincount(rows, weights=values[mask], minlength=sequences.n_source_rows)
        count = np.bincount(rows, minlength=sequences.n_source_rows)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    def predict_proba(self, sequences: Sequences) -> np.ndarray:
        """One-step-ahead ``P(correct)`` per source row, ``NaN`` where unpredictable.

        The prediction for a response is made from everything *before* it in the
        same sequence, which is what a BKT prediction is. Note the consequence
        for cross-validation: a held-out response is still predicted using its
        own student's earlier responses, including ones that were in the
        training split. That is both references' convention and the only one
        the model admits -- a filter has no way to predict a mid-sequence
        response without the history that precedes it -- but it means BKT's
        held-out score is not measuring the same independence a design-matrix
        model's is. What is held out is the *parameter estimation*, not the
        history.

        A row that appears in several sequences (``multi_kc="replicate"``)
        gets the mean of its KCs' predictions, as LearnSphere's component
        reports it.
        """
        state, usable = self._filter(sequences)
        return self._per_row(sequences, state["p_correct"], usable)

    def mastery(self, sequences: Sequences) -> np.ndarray:
        """``P(known)`` before each response, per source row. ``NaN`` where unpredictable."""
        state, usable = self._filter(sequences)
        return self._per_row(sequences, state["predicted"][:, :, 1], usable)

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
        sequences = extract_sequences(data, multi_kc=self.multi_kc)
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

def _feasible_start(rng: np.random.Generator) -> np.ndarray:
    """A random starting point with ``G + S < 1``, redrawn until it is."""
    while True:
        start = rng.uniform(*_RESTART_RANGE, size=5)
        if start[2] + start[3] < 1.0 - EPS:
            return start


def _batches(sequences: Sequences, n_restarts: int) -> list[list[int]]:
    """KCs grouped so that one forward-backward pass stays under :data:`BATCH_CELLS`.

    KCs are sorted by their longest sequence first, so a batch pads its rows
    to a length its members nearly fill rather than to the longest KC in the
    data. A KC too big for the budget on its own gets a batch of its own.
    """
    lengths = sequences.lengths
    n_kcs = len(sequences.kc_names)
    longest = np.zeros(n_kcs, dtype=int)
    np.maximum.at(longest, sequences.kc_of_seq, lengths)
    n_seq = np.bincount(sequences.kc_of_seq, minlength=n_kcs)

    batches, current, rows = [], [], 0
    for k in sorted(np.flatnonzero(n_seq > 0), key=lambda k: (longest[k], k)):
        cells = (rows + n_seq[k]) * n_restarts * longest[k]
        if current and cells > BATCH_CELLS:
            batches.append(current)
            current, rows = [], 0
        current.append(int(k))
        rows += int(n_seq[k])
    if current:
        batches.append(current)
    return batches


def fit_bkt(sequences: Sequences, *, forget: bool = False, n_restarts: int = 1,
            seed: int | None = 0, tol: float = DEFAULT_TOL,
            max_iter: int = DEFAULT_MAX_ITER, n_jobs: int | None = 1,
            warn_not_converged: bool = True, warn_at_bound: bool = True) -> BKTFit:
    """Fit BKT by expectation-maximization, one independent chain per KC and restart.

    Every parameter is per-KC and no KC's data touches another's, so this is
    ``n_kcs * n_restarts`` separate Baum-Welch runs and the total
    log-likelihood is the sum of each KC's best. The runs are batched through
    one vectorized forward-backward pass and the batches spread over
    ``n_jobs`` processes; neither changes any number. Every KC is held to
    ``G + S < 1`` -- see the module docstring and :attr:`BKTFit.at_bound`.

    :param forget: fit a forgetting probability ``F``. Classical BKT pins it at
        0, which is the default; a model being compared against one with an
        explicit forgetting mechanism should be allowed it.
    :param n_restarts: EM runs per KC. Restart 0 always starts from
        :data:`DEFAULT_INIT`, so ``n_restarts=1`` is deterministic and ignores
        ``seed``; further restarts are drawn from ``seed``. BKT's likelihood is
        multimodal, and :attr:`BKTFit.restart_spread` is the only evidence
        available about whether that mattered here — running one restart does
        not make the problem unimodal, only unmeasured.
    :param seed: seeds the random restarts. Fixed at 0 rather than ``None`` so
        that a multi-restart fit is reproducible unless you ask otherwise.
    :param n_jobs: worker processes to run EM batches in, ``-1`` for every
        core. The result does not depend on it.
    :param warn_not_converged: warn when some KC's EM stops on the iteration cap.
    :param warn_at_bound: warn when some KC is held at ``G + S = 1``.
    """
    n_kcs = len(sequences.kc_names)
    restarts = max(1, int(n_restarts))
    root = np.random.default_rng(seed)
    streams = root.spawn(n_kcs) if n_kcs else []

    prior = np.full(n_kcs, np.nan)
    learn = np.full(n_kcs, np.nan)
    guess = np.full(n_kcs, np.nan)
    slip = np.full(n_kcs, np.nan)
    forgetting = np.full(n_kcs, np.nan)
    n_seq_per_kc = np.bincount(sequences.kc_of_seq, minlength=n_kcs) if n_kcs else np.zeros(0, int)
    n_transitions = np.zeros(n_kcs, dtype=int)
    if n_kcs:
        np.add.at(n_transitions, sequences.kc_of_seq, sequences.mask[:, 1:].sum(axis=1))
    converged = np.ones(n_kcs, dtype=bool)
    monotone = np.ones(n_kcs, dtype=bool)
    n_iter = np.zeros(n_kcs, dtype=int)
    final_improvement = np.zeros(n_kcs)
    restart_lls = np.full((n_kcs, restarts), np.nan)
    bound_active = np.zeros(n_kcs, dtype=bool)

    # Starting points are drawn per KC from its own stream, in KC order, so
    # they do not depend on how the KCs are later batched.
    inits = np.full((n_kcs, restarts, 5), np.nan)
    for k in range(n_kcs):
        if n_seq_per_kc[k] == 0:
            continue   # no data: parameters stay NaN, a gap rather than a claim
        inits[k, 0] = DEFAULT_INIT
        for r in range(1, restarts):
            inits[k, r] = _feasible_start(streams[k])

    jobs = []
    batches = _batches(sequences, restarts)
    for members in batches:
        local = np.full(n_kcs, -1, dtype=int)
        local[members] = np.arange(len(members))
        selected = local[sequences.kc_of_seq] >= 0
        mask = sequences.mask[selected]
        width = max(int(mask.sum(axis=1).max(initial=0)), 1)
        y, mask = sequences.y[selected][:, :width], mask[:, :width]
        kc_local = local[sequences.kc_of_seq[selected]]
        # Restart r of KC k is chain k*R + r; its rows are a copy of k's rows.
        y = np.repeat(y[None], restarts, axis=0).reshape(-1, width)
        mask = np.repeat(mask[None], restarts, axis=0).reshape(-1, width)
        chain = (kc_local[None, :] * restarts + np.arange(restarts)[:, None]).reshape(-1)
        init = inits[members].reshape(-1, 5)
        jobs.append((y, mask, chain, len(members) * restarts, init, forget, tol, max_iter))

    for members, result in zip(batches, _run_batches(jobs, n_jobs)):
        for i, k in enumerate(members):
            chains = np.arange(i * restarts, (i + 1) * restarts)
            restart_lls[k] = result.ll[chains]
            best = int(chains[np.argmax(result.ll[chains])])
            prior[k], learn[k] = result.prior[best], result.learn[best]
            guess[k], slip[k], forgetting[k] = (result.guess[best], result.slip[best],
                                                result.forget[best])
            converged[k], monotone[k] = result.converged[best], result.monotone[best]
            n_iter[k], final_improvement[k] = result.n_iter[best], result.final_improvement[best]
            bound_active[k] = result.at_bound[best]

    fitted = n_seq_per_kc > 0
    fit = BKTFit(
        kc_names=list(sequences.kc_names), prior=prior, learn=learn, guess=guess,
        slip=slip, forget=forgetting,
        ll=float(restart_lls[fitted].max(axis=1).sum()) if fitted.any() else 0.0,
        n_obs=sequences.n_obs, n_rows=sequences.n_rows, n_seq=len(sequences),
        n_seq_per_kc=n_seq_per_kc, n_transitions=n_transitions, converged=converged,
        monotone=monotone, n_iter=n_iter, final_improvement=final_improvement,
        restart_lls=restart_lls, bound_active=bound_active, has_forget=forget,
        kc_model=sequences.kc_model, multi_kc=sequences.multi_kc,
    )

    if warn_not_converged and not converged.all():
        warnings.warn(
            f"EM hit the {max_iter}-iteration cap on {int((~converged).sum())} of "
            f"{n_kcs} KC(s); their parameters are wherever it stopped. Raise max_iter "
            "or loosen tol.", RuntimeWarning, stacklevel=2)
    if warn_at_bound and fit.at_bound:
        warnings.warn(
            f"{len(fit.at_bound)} KC(s) held at G + S = 1, where the data prefer a fit in "
            f"which 'known' predicts failure: {', '.join(fit.at_bound[:4])}. For these the "
            "fitted BKT is the KC's base rate; see BKTFit.at_bound.",
            RuntimeWarning, stacklevel=2)
    if not monotone.all():
        warnings.warn(
            f"EM decreased the log-likelihood on {int((~monotone).sum())} KC(s). EM is "
            "monotone by construction, so this is a numerical defect in the fit, not a "
            "property of the data — please report it.", RuntimeWarning, stacklevel=2)
    return fit


def bkt_from_data(data: StepData, *, multi_kc: str = "refuse",
                  **kwargs) -> tuple[Sequences, BKTFit]:
    """Reshape and fit in one call, returning both halves.

    The sequences come back because everything downstream of the fit needs them
    -- :meth:`BKTFit.predict_proba`, :meth:`BKTFit.rmse` and cross-validation
    all take sequences, not a ``StepData``.
    """
    sequences = extract_sequences(data, multi_kc=multi_kc)
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
    live = sequences.mask & keep[np.where(sequences.mask, sequences.rows, 0)]
    lengths = live.sum(axis=1)
    surviving = np.flatnonzero(lengths > 0)
    width = max(int(lengths.max(initial=0)), 1)
    n = len(surviving)

    y = np.zeros((n, width), dtype=np.int8)
    mask = np.zeros((n, width), dtype=bool)
    rows_out = np.full((n, width), -1, dtype=int)
    # Kept responses slide left within their row; their order is unchanged.
    seq, _ = np.nonzero(live[surviving])
    position = np.arange(len(seq)) - np.repeat(np.cumsum(lengths[surviving]) - lengths[surviving],
                                               lengths[surviving])
    y[seq, position] = sequences.y[surviving][live[surviving]]
    mask[seq, position] = True
    rows_out[seq, position] = sequences.rows[surviving][live[surviving]]

    return Sequences(y=y, mask=mask, rows=rows_out,
                     kc_of_seq=sequences.kc_of_seq[surviving],
                     students=[sequences.students[i] for i in surviving],
                     kc_names=sequences.kc_names, kc_model=sequences.kc_model,
                     n_source_rows=sequences.n_source_rows, multi_kc=sequences.multi_kc)


_WORKER: dict = {}


def _init_fold_worker(full: Sequences, y: np.ndarray, fit_kwargs: dict) -> None:
    _WORKER.update(full=full, y=y, fit_kwargs=fit_kwargs)


def _score_fold(job: tuple) -> dict:
    """Refit on one training split and score its held-out rows. Runs in a worker."""
    index, test_idx = job
    full, y, fit_kwargs = _WORKER["full"], _WORKER["y"], _WORKER["fit_kwargs"]
    keep = np.ones(full.n_source_rows, dtype=bool)
    keep[test_idx] = False
    fit = fit_bkt(_restrict(full, keep), n_jobs=1, warn_not_converged=False,
                  warn_at_bound=False, **fit_kwargs)
    predicted = fit.predict_proba(full)[test_idx]

    # A held-out row whose KC never appeared in training has no parameters to
    # be predicted from. Counted, not quietly imputed — the same role
    # ``unseen_column_fraction`` plays for a design.
    unseen = ~np.isfinite(predicted)
    residual = y[test_idx][~unseen] - predicted[~unseen]
    return {"fold": index, "n_train": int(keep.sum()), "n_test": len(test_idx),
            "resid": residual,
            "rmse": float(np.sqrt(np.mean(residual ** 2))) if residual.size else float("nan"),
            "unseen_column_fraction": float(unseen.mean()),
            "converged": bool(np.all(fit.converged))}


def cross_validate_bkt(data: StepData, *, scheme: str = "student_blocked",
                       n_folds: int = 3, seed: int | None = None,
                       convention: str = "per_fold", restart_seed: int | None = 0,
                       multi_kc: str = "refuse", n_jobs: int | None = 1,
                       **fit_kwargs) -> CVResult:
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
    :param multi_kc: passed to :func:`extract_sequences`.
    :param n_jobs: worker processes to fit folds in, ``-1`` for every core.
        Folds are drawn in the parent and collected in order, so the result
        does not depend on it.
    :param fit_kwargs: passed to :func:`fit_bkt` (``forget``, ``n_restarts``, ...).
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")

    full = extract_sequences(data, multi_kc=multi_kc)
    y = np.asarray(data.y, dtype=float)
    folds = make_folds(data, scheme, n_folds, seed, convention)
    for index, test_idx in enumerate(folds):
        if len(test_idx) == 0 or len(test_idx) == len(data):
            raise ValueError(
                f"Fold {index} is degenerate: {len(data) - len(test_idx)} train / "
                f"{len(test_idx)} test")

    jobs = [(index, np.asarray(test_idx)) for index, test_idx in enumerate(folds)]
    fit_kwargs = {"seed": restart_seed, **fit_kwargs}
    workers = _worker_count(n_jobs, len(jobs))
    if workers == 1:
        try:
            _init_fold_worker(full, y, fit_kwargs)
            scored = [_score_fold(job) for job in jobs]
        finally:
            _WORKER.clear()
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_fold_worker,
                                 initargs=(full, y, fit_kwargs)) as pool:
            scored = list(pool.map(_score_fold, jobs))

    results = tuple(FoldResult(fold=s["fold"], n_train=s["n_train"], n_test=s["n_test"],
                               rmse=s["rmse"], unseen_column_fraction=s["unseen_column_fraction"],
                               converged=s["converged"]) for s in scored)
    if convention == "per_fold":
        rmse = float(np.mean([s["rmse"] for s in scored]))
    else:
        pooled = np.concatenate([s["resid"] for s in scored])
        rmse = float(np.sqrt(np.mean(pooled ** 2)))
    return CVResult(scheme=scheme, convention=convention, n_folds=n_folds,
                    seed=seed, rmse=rmse, folds=results)
