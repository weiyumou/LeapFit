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
design = build_afm_design(data)
fit    = fit_afm(design, data.y)
print(fit.summary())
print(cross_validate(design, data, scheme="item_blocked").summary())
```

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
never reads the `Opportunity (...)` columns of its input. Ours reads the columns,
per PyAFM. On ds5426 the two agree on 99.93% of rows — they differ only where the
file's row order and the timestamp order disagree.

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
| response | `First Attempt == "correct"`; `hint` and `incorrect` are both failures |
| multi-KC | `KC (model)` and `Opportunity (model)` split on `~~`, aligned **by position** |
| opportunity | DataShop's 1-based count **minus 1**, so a first encounter enters as `T = 0` |
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
  with the row number.
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

Validated end to end against `wf3990` on E-learning 22: parsing, parameter
counting, and fitted AIC/BIC (see above). Not yet checked against
E-learning 23.

```bash
python -m pytest tests/ -q      # 37 tests, ~2s, no data required
```
