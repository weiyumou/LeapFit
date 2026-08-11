# A local AFM, matched to LearnSphere

LearnSphere fits vanilla AFM from an uploaded Q-matrix and nothing else. Every
model this project is heading toward — a congruity-weighted practice term,
hierarchical shrinkage over KC granularity, kernel-smoothed item parameters —
is a change to the design matrix or the penalty, so all of them need a fitter
we control. This is that fitter, built to reproduce the reference exactly so
that "AFM" stays a trustworthy baseline column.

```bash
python run_afm.py ds5426_student_step.txt --list-models

python run_afm.py ds5426_student_step.txt \
    --kc-model Single-KC --kc-model LOs-new --kc-model KCluster \
    --cv item_blocked --seeds 0:50 --out results/afm-e22.csv
```

```python
from afm import load_student_step, build_afm_design, fit_afm, cross_validate

data   = load_student_step("ds5426_student_step.txt", kc_model="LOs-new")
design = build_afm_design(data)                 # analysis default
fit    = fit_afm(design, data.y)
print(fit.summary())
print(cross_validate(design, data, scheme="item_blocked").summary())
```

## What the input file must contain

Six columns. Everything else in a DataShop student-step export is ignored.

| column | used for |
|---|---|
| `Anon Student Id` | the student block; the unit of student-blocked CV |
| `Problem Name` | half of the item label |
| `Step Name` | the other half — items are `Problem Name ## Step Name`, the unit of item-blocked CV |
| `First Attempt` | the response: `correct` is a success, `incorrect` / `hint` / `unknown` are failures |
| `KC (<model>)` | the Q-matrix row, `~~`-separated for multi-KC steps |
| `Opportunity (<model>)` | prior practice `T`, `~~`-separated and aligned by position, numbered from 1 |

One optional column changes behaviour:

| column | effect if present |
|---|---|
| `First Transaction Time` | defines the canonical practice order, so `recompute_opportunities=True` can correct `Opportunity`. Absent, practice order falls back to file row order — which is what DataShop numbers by anyway |

A file may carry any number of KC models; `list_kc_models(path)` enumerates
them and each is fitted independently. A missing required column raises and
names it. Values outside DataShop's `First Attempt` vocabulary raise rather
than being scored as failures — a file writing `1`/`0` needs
`success_values=("1",), failure_values=("0",)`, stated explicitly, because the
alternative is an all-zero response that fits, converges, and reports a
plausible AIC.

Practical ceiling: identification runs a dense `p × p` eigendecomposition, so
it is 0.3 s at 1,600 parameters, 22 s at 6,000, and 108 s at 10,000. Above
roughly 10,000 parameters pass `identify=False` and count parameters yourself.
Parsing and fitting are sparse and scale with the number of observations.

## Two settings, because reproduction and analysis want opposite defaults

| | `build_afm_design(data)` — **default** | `learnsphere_compat=True` |
|---|---|---|
| purpose | the baseline in new work | reproduce the published table |
| student ridge | none | 1.0 |
| identification | aliased columns dropped | none |
| `nPars` | `rank(X)` | `n_students + 2·n_KCs` |
| slope bound | free | free |

Compat is a **fixture, not a model** — its only job is to hit LearnSphere's
numbers, which `tests/test_learnsphere_equivalence.py` checks against wf3990's
own `model_values.xml`. Everything scientific runs in the default.

### Why the default drops the ridge

With one student and one KC per row, the student columns and the KC-intercept
columns both sum to the all-ones vector, so the design is rank-deficient: you
can add any constant to every student, subtract it from every KC intercept, and
every prediction is unchanged. LearnSphere's ridge on students picks one point
on that flat line — it costs 2.4 nats out of 21,000, which is the tell that it
is an **identification device, not regularization**. The cost is a "log
likelihood" that isn't one, and a parameter count that overstates the model.

`Design.identify()` removes the redundancy directly instead, and finds a second
kind the ridge cannot touch: **a KC that no student ever practises twice has
`T ≡ 0`, so its slope column is identically zero and `γ_k` is not estimable at
all.** E-learning 22:

| KC model | columns | rank | phantom parameters |
|---|---|---|---|
| `Single-KC` | 41 | 40 | 1 |
| `LOs-new` | 241 | 239 | 2 |
| `pmi` (KCluster) | 275 | 274 | 1 |
| `concept` | 781 | 767 | 14 |
| `Unique-step` | 3769 | 2810 | **959** |

BIC charges `log(42,176) = 10.65` per parameter, so `Unique-step`'s published
BIC carries **10,213 nats of penalty for parameters that do not exist** — 25% of
its total. The overcount scales with granularity, because fine-grained models
are the ones with singleton KCs, which matters for any argument about how BIC
penalizes granularity. It does not reorder the published table.

Consequences, all pinned by tests: a dropped student becomes the reference
level, so `kc_values()` reports intercepts at the **sum-to-zero** point (the
average student) rather than an arbitrary one; and a KC with no second
opportunity reports `Slope = NaN`, not `0.0`, so it cannot be mistaken for
"students did not learn" by a `γ ≤ 0.001` screen.

### A third kind of non-parameter: separation

Identification is about the design; this one is about the design *and the
responses*. A KC every student always gets right has an intercept whose MLE is
`+∞` — raising it always improves the fit and touches nothing else — so no
finite estimate exists. `Design.separated(y)` reports these exactly (the
single-column case; full detection is an LP), respecting ridge and bounds,
which both remove the divergence. E-learning 22:

| KC model | KCs | separated | intercepts | slopes |
|---|---|---|---|---|
| `Single-KC` | 1 | 0 | 0 | 0 |
| `LOs-new` | 101 | 1 | 0 | 1 |
| `pmi` (KCluster) | 118 | 2 | 0 | 2 |
| `concept` | 371 | 42 | 1 | 41 |
| `Unique-step` | 1865 | **746** | 253 | 493 |

The **likelihood** is fine: the gradient certificate still holds, so the
reported value is essentially the supremum and AIC is not distorted through that
channel. Two other things are wrong. First, 746 of `Unique-step`'s 2,810
remaining parameters are quantities that were never estimated, and BIC charges
10.65 nats for each. Second, their printed values are artefacts of where TNC
stopped: across those 746 the magnitude runs from 0.96 to 647 with a median of
23.5 — a spread that carries no information about the KCs.

Detection is a lower bound and visibly so: the largest coefficient *not* flagged
is 307, which is a group of columns separating the data only in combination.
That is the LP case this deliberately does not attempt.

`fit_afm` warns, `fit.summary()` reports it, `run_afm.py` adds an `n_separated`
column, and `kc_values()` carries a `Separated` flag — flagged rather than
blanked, because unlike a never-repeated KC the datum is real and strong; it is
the *estimate* that does not exist.

## The reference

Two components in [LearnSphere/WorkflowComponents](https://github.com/LearnSphere/WorkflowComponents),
and it matters which:

- **`AnalysisFastAfmAndCv`** (`program/learner_performance_prediction.py`) —
  **this is what produced the EDM 2025 numbers** (workflow `wf3990`). It is the
  only component emitting the `student_blocked_cv` / `item_blocked_cv` schema and
  the `Skill`/`Student` parameter types found in the run artifacts. It fits
  `sklearn.linear_model.LogisticRegression(solver="lbfgs")` — **unconstrained
  slopes** — and reports the *unpenalized* log-likelihood as `-log_loss * n`.
- **`AnalysisPyAfm`** (`program/process_datashop.py`, `program/custom_logistic.py`) —
  the documented Python AFM. Bounds slopes at `(0, None)` and reports a
  *penalized* objective as its log-likelihood.

They fit different models. Where the two disagree, this package follows
`AnalysisFastAfmAndCv`, because that is the published baseline. See
[learnsphere-issue-afm.md](learnsphere-issue-afm.md) for the full comparison and
two defects found along the way.

One further quirk of `AnalysisFastAfmAndCv`: it **recomputes opportunity counts
internally** (`df_to_sparse_afm`, a per-student `np.cumsum` over the Q-matrix) and
never reads the `Opportunity (...)` columns of its input. Ours reads the columns
by default, per PyAFM; pass `recompute_opportunities=True` for the other
convention. They agree on 99.93% of ds5426's rows. The 0.07% that differ are not
ties: **the export's row order contains 12 within-student inversions of
`First Transaction Time`** — an attempt at 01:44:41 listed before one at
01:44:36 — and DataShop's column follows row order, so 28 rows carry the wrong
opportunity number. Recomputing fixes them and moves the log-likelihood by less
than 1 nat.

**The EDM 2025 numbers use LearnSphere's parameter-counting convention**
(`nPars = n_students + 2·n_KCs`, no intercept column). Verified without touching
any data, from the identity `BIC − AIC = nPars · (log N − 2)`. This pins the
convention, *not* which component ran — the fits themselves came from the
`wf3990` DataShop workflow, which differs from PyAFM (see below):

| E-learning 2022 model | KCs | nPars = 39 + 2·KCs | predicted BIC−AIC | published |
|---|---|---|---|---|
| `Single-KC` | 1 | 41 | 354.63 | 354.634 |
| `LOs-new` | 101 | 241 | 2084.55 | 2084.555 |
| `Concept` | 371 | 781 | 6755.32 | 6755.343 |
| `Unique-step` | 1865 | 3769 | 32600.34 | 32600.367 |

at `N = 42,176`. Every row of both tables closes. Pinned in
`tests/test_afm.py::test_published_aic_bic_identity`, so a later "improvement"
to the parameter count breaks the build rather than the comparability.

## Semantics reproduced

| | rule |
|---|---|
| observation | one row of the student-step rollup; **rows with no KC are dropped**, so they leave `N` too |
| response | `First Attempt == "correct"`; `hint`, `incorrect` and `unknown` are failures. Matched case-insensitively, and an unrecognized value raises — see [What the input file must contain](#what-the-input-file-must-contain) |
| multi-KC | `KC (model)` and `Opportunity (model)` split on `~~`, aligned **by position** |
| opportunity | DataShop's 1-based count **minus 1**, so a first encounter enters as `T = 0` |
| ordering | one canonical practice sequence (`StepData.practice_order`), shared by `T` and any accumulator built over a student's history |
| item label | `Problem Name ## Step Name` (the unit of item-blocked CV) |
| design | `[student one-hot | KC indicator | KC × T]`, **no intercept column** |
| penalty | ridge 1.0 on student intercepts only; KC intercepts and slopes unpenalized |
| bounds | intercepts free; **slopes unbounded by default** — see below |
| optimizer | `scipy.optimize.minimize(method="TNC")` from `w0 = 0` |
| nPars | every column, including slopes resting at the bound |

## The reference is two different models

`AnalysisPyAfm` (Python) **bounds slopes at `(0, None)`**. The DataShop workflow
that produced the EDM 2025 tables (`wf3990`, 2024-12-06) **does not**: 3,790 of
its 29,700 fitted slopes are negative, down to −1.17. Bounding optimizes over a
strictly smaller feasible set, so it reports a worse likelihood on identical
data — 7–22 nats across E-learning-22's KC models.

`build_afm_design(..., bound_slopes=False)` is therefore the **default**, because
it reproduces the published baseline. Pass `True` for PyAFM's variant.

A consequence worth keeping straight: since real fits admit negative slopes, the
RQ-3 screen `γ ≤ 0.001` selects KCs where students did not learn **or got
worse** — not only flat ones.

### Validation against wf3990 (E-learning 22)

`nPars` recovered from LearnSphere's own output as `(AIC + 2·ll)/2` matches
`design.n_params` **exactly for all ten models**, from 41 to 3,769 parameters.
With slopes unbounded, fitted AIC agrees to within ±18 nats on eight of ten:

| model | our AIC − LearnSphere |
|---|---|
| `question-cosine` | −1.3 |
| `Single-KC` | −3.7 |
| `LOs` | +4.8 |
| `pmi` (KCluster) | −9.9 |
| `concept-cosine` | −10.6 |
| `LOs-new` | +11.5 |
| `concept-euclidean` | −18.0 |
| **`concept`** (781 par.) | **−145.8** |
| **`Unique-step`** (3,769 par.) | **−1618.2** |

The last two are not disagreements about the model — our solutions carry a KKT
optimality certificate on a convex objective, so LearnSphere's optimizer is
stopping early exactly where the parameter count is largest. The BIC ordering
still reproduces the paper's headline: `LOs-new` best, `pmi`/KCluster second.

Caveat on `parameters.xml`: its coefficients score 540–610 nats *worse* than the
same file's reported `log_likelihood` when evaluated on the full data, so those
values are not the full-data MLE (most likely a CV-fold fit). Use
`model_values.xml` for fit statistics; do not treat `parameters.xml` as the
fitted model.

## Three more things about the reference worth knowing

**1. The two components report different likelihoods.** PyAFM sets
`self.ll = -w.fun` where `w.fun` is the *penalized* objective, so its AIC and BIC
are built on a penalized objective rather than a likelihood. `AnalysisFastAfmAndCv`
instead reports the true Bernoulli log-likelihood (`-log_loss * n`) — but computes
it from a fit that sklearn silently L2-regularizes at `C=1.0`, and counts
parameters without the fitted global intercept. `AFMFit` exposes both conventions:
`.ll` follows PyAFM, `.ll_unpenalized` / `.aic_unpenalized` / `.bic_unpenalized`
are the textbook quantities and match FastAfmAndCv's definition.

**2. PyAFM's iteration cap has never taken effect.** It passes
`options={'maxiter': 1000}` to TNC, but TNC's budget option is `maxfun`;
scipy raises `OptimizeWarning: Unknown solver options: maxiter` and ignores it.
Every published AFM fit therefore ran at TNC's *default* budget of
`max(100, 10·nPars)` — 410 evaluations for E22's `Single-KC` against 37,690 for
`Unique-step`. We default `max_fun=None` to preserve that behaviour and route an
explicit `max_fun` to the option the solver actually reads. Pinned in
`test_references_iteration_cap_is_inert_for_tnc`.

**3. There are two incompatible CV conventions in the repo**, differing in the
third or fourth decimal — where KC-model comparisons are decided:

- `convention="per_fold"` (PyAFM): RMSE within each fold, then average the fold
  RMSEs. Item/student folds come from `LabelKFold`, which is **deterministic**
  and accepts no seed.
- `convention="pooled"` (FastAfmAndCv): shuffle labels, cut into contiguous
  blocks, pool all held-out predictions, one RMSE over the pooled vector. The
  shuffle is seeded, which is what makes a 50-seed protocol meaningful.

The EDM 2025 protocol ("three folds, 50 random seeds, item-stratified") is the
second. Both are here and the choice is explicit.

## Divergences from the reference

- **Sparse design.** PyAFM calls `X.toarray()`; that is 1.27 GB for E22's
  `Unique-step`. Sparse it is 2 MB. Objective and gradient agree with the dense
  reference to 1e-9 in value (`test_end_to_end_agreement_with_the_reference_recipe`);
  coefficients agree to ~1e-5, the width of TNC's basin, not a discrepancy.
- **Convergence is reported, not discarded.** A silent non-convergence is
  indistinguishable from a bad KC model in the fit statistics.
- **Malformed rows raise.** A row whose KC and opportunity fields differ in
  length silently misaligns skills with counts in the reference; here it errors
  with the row number, as does a non-integer opportunity value.
- **Outcome labels are validated.** The reference compares `First Attempt` to
  the literal `"correct"`, so an export writing `"Correct"` yields an all-zero
  response, a converged fit, and a plausible AIC. Case is folded first (a no-op
  on real DataShop files) and unrecognized values raise.
- **Separation is detected and reported.** See above; the reference reports
  diverging coefficients as ordinary estimates.
- **`unseen_column_fraction` per fold.** Blocking by item or student leaves some
  design columns with no training rows; their coefficients stay at 0 and those
  predictions fall back to the student intercept. That is the reference's
  behaviour and it is *the* mechanism by which a fine-grained KC model posts a
  worse item-blocked RMSE than `Single-KC`. Measured at E22 scale: 0.0% of
  held-out rows for a 101-KC model, 1.0% at 371 KCs, 100% at `Unique-step`.

## Open question for the data

The `BIC − AIC` identity closes at **39 students for E-learning 2023**, but the
paper reports 41. Two students are missing from the fitted rollup. Nothing
published depends on it; the data description may.

## Layout and the extension seam

```
afm/data.py      DataShop student-step rollup -> StepData (parsing rules)
afm/design.py    Block / Design: columns carrying their own penalty and bounds
afm/model.py     fit_afm: the objective, AIC/BIC, DataShop-format KC values
afm/crossval.py  fold schemes and the two conventions
run_afm.py       CLI: one row per KC model
tests/test_afm.py
```

`Design` is a list of labelled `Block`s, each carrying its own per-column ridge
and bounds. That is deliberately where the next models attach:

- **Congruity-weighted practice** is one more block —
  `design.with_blocks(congruity_block(data, accumulated))` — where `accumulated`
  has a column per accumulator (cross-KC congruity, plus the plain off-KC
  attempt count that identifies the transfer-neutral congruity level).
- **Hierarchical shrinkage** is a reparameterization, not new machinery: write
  `β_k = β_parent(k) + b_k` as an unpenalized parent block plus a ridge-penalized
  deviation block. The ridge weight *is* the prior precision `1/σ²_b`. Estimating
  `σ²_b` needs an outer loop; the inner fit is this same solver.

## Status

Validated end to end against `wf3990` on E-learning 22 — parsing, parameter
counting, fitted AIC/BIC, identification, and the opportunity ordering. The
equivalence suite requires the run artifacts and skips without them:

```bash
python -m pytest tests/test_afm.py -q            # 68 tests, ~3s, no data
AFM_WF3990_DIR=/path/to/wf3990 python -m pytest tests/ -q   # 76 tests
```

Runs on any DataShop student-step export carrying the six columns above; the
portability tests build a six-column file from scratch and fit it, so that claim
is checked rather than asserted. Not yet run against E-learning 23.
