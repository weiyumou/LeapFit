# Changelog

Notable changes per release. Versions follow [semantic versioning](https://semver.org);
while the major version is 0, a minor bump may change the public API.

## 0.5.0 — 2026-09-03

### Added

- **Learning Factors Analysis: a search over KC models** (`leapfit/lfa.py`,
  `leapfit-lfa`). Not another student model — the states *are* KC labellings
  and each is scored by fitting AFM to it, so `leapfit.lfa` sits above
  `leapfit.afm` and reuses the identification pass, the separation check and
  the KKT certificate unchanged. `build_factor_matrix` assembles the
  difficulty-factor matrix from KC models already on the export, `lfa_search`
  runs greedy best-first on BIC (default) or AIC, and `LFAResult` carries the
  trajectory, the ranked frontier, every state's optimality certificate, and
  every refused move with its reason.

  Grounded against DataShop's own LFA, whose search engine ships only as a
  binary: leapfit reproduces the offline `lfa-6.0` tool's fit statistics to
  **1.2e-9** under `learnsphere_compat=True`. A run of that tool is kept as a
  fixture and `tests/test_lfa_equivalence.py` pins the agreement, the
  parameter-count convention recovered from the reference's own output, and
  the two defects below.

- **Two screens on every candidate move**, both before any fit, so a refused
  move costs no optimization. *Evidence*: a new KC needs
  `min_opportunities` observations at `T >= 1`, the only rows a slope column
  touches. *Estimability*: `Design.separated` restricted to the KCs the move
  touched.

  These are not defensive detail. On the validated export the reference
  selected a KC model whose slope has no finite estimate, and because its
  operator set is split-only that one move then appeared in **all 99** states
  it reported. 34% of the 941 candidate moves at one measured node were
  separated. Neither honest parameter counting (`n_params = rank(X)`) nor
  bounding the slopes removes the preference — both measured, and the same
  single-step split still ranked first of 941 — so this is a property of the
  criterion rather than the estimator.

- **Held-out validation of the shortlist** (`validate_top`, `LFAValidation`),
  scoring the top states and any authored models on folds shared by every
  candidate. This is the protocol the reference's own follow-up prescribes:
  search by an information criterion because cross-validation is unaffordable
  inside the loop, then test the best models out of sample. The interesting
  output is the *agreement*, and on the shipped example there is none — BIC's
  pick is third by held-out RMSE, and the planted `Topics` model is fifth of
  six in sample and second of six held out.

- **Merge operators** (`merges`: `"none"`, `"lineage"`, `"pairwise"`,
  `"both"`), which the published method has only as a manual step. Pairwise
  merge *enlarges the reachable set* — a KC that is the union of two factors'
  steps is not expressible as any sequence of splits — while lineage-undo adds
  only paths, mattering when `beam` has evicted a state.

- **`root=`**, to start the search from a KC model you already have rather
  than from the "All" model (one skill on every step, which is what an
  export's `Single-KC` column holds). This is the published setup, and it is
  what makes merging useful: from the All root neither merge operator changes
  the answer, because there is nothing to coarsen, while from an authored
  91-KC root pairwise merge finds a model **38.4 nats** of BIC better than
  splitting alone, with the merge in the winning lineage.

- **Warm starts.** `fit_logistic` and `fit_afm` take `w0` (default unchanged),
  and `lfa_search` seeds each child from its parent. 2.2–2.9× fewer function
  evaluations for identical optima — safe to use aggressively *because* the
  objective is convex and `is_optimal` certifies each fit independently of
  where it started.

- **Parallel scoring.** `lfa_search` takes `n_jobs`. An eight-expansion search
  on a 20,687-row export goes from 72.8 s to 14.7 s on eight workers, and the
  worker count is a wall-clock knob only: `test_the_worker_count_does_not_move_a_single_digit`
  asserts equality of every score, every refusal and the whole ranking. The
  pool is held open for the search rather than per expansion, because shipping
  the observations is the fixed cost (3.9 s to twelve workers) and paying it
  per iteration erased the gain entirely.

### Changed

- `tests/test_package.py` gains a third tier. A search over KC models is
  neither shared infrastructure nor a model family but a *consumer* of one, so
  `leapfit.lfa` imports `leapfit.afm` deliberately and a new test asserts the
  edge runs one way.

### Fixed

- The stated test counts in `README.md` were wrong in both places (`135 pass`,
  `118 pass, 11 skip`). Measured on a fresh clone: 180 pass, 29 skip in ~21 s.

## 0.4.0 — 2026-08-18

### Added

- **Paired model comparison is the CLI default.** When several KC models are
  fitted and cover the same export rows, folds are now drawn once per scheme
  and seed and every model is scored on those identical partitions. The
  `cv_rmse` columns are then comparable by construction, and a contrasts table
  (stdout, and `--contrasts FILE`) reports each model's within-fold RMSE
  difference against a baseline — the best-scoring model, or `--baseline NAME`.
  Differencing with the fold held fixed removes the partition from the
  between-model variance, which is both sounder and more powerful than
  t-testing two independently repeated CV means. Models covering different
  rows fall back to independent per-model CV with an explanation; `--no-paired`
  forces that protocol.
- `paired_scores` aggregates a `paired_cross_validate` table to per-(model,
  seed) scores under either RMSE convention — one paired run carries both, plus
  everything `repeated_cross_validate` reports, asserted equal in
  `test_paired_scores_reconstruct_repeated_cv`.
- `paired_cross_validate` tables now carry `unseen_column_fraction` and
  `converged` per (seed, fold, model), matching the other entry points.

### Changed

- With `--seeds`, reported `cv_rmse` values are unchanged: fold drawing
  depends only on the data, so same-seed partitions were already identical
  across models. Without `--seeds`, a multi-model run now uses seed 0 (shared
  folds must be seeded) instead of the deterministic `LabelKFold` partition,
  so those RMSEs move within their partition sensitivity. Single-model runs
  and `--no-paired` keep the previous behaviour exactly, including
  `LabelKFold` for `per_fold` without `--seeds`.
- The CLI reads the export once and fits every KC model before
  cross-validating (shared folds need all designs up front), so per-model CV
  progress now prints after all fit summaries rather than interleaved.

## 0.3.0 — 2026-08-16

### Added

- **Parallel cross-validation.** `cross_validate`, `repeated_cross_validate`
  and `paired_cross_validate` take `n_jobs` (joblib's convention: `-1` is every
  core), and the CLI takes `-j/--jobs`. Fold fits are independent, so they run
  in a process pool; the designs and responses cross to each worker once
  through the pool initializer rather than once per fold, and `StepData` — with
  its source table — never crosses at all. Measured on E-learning-22, a 50-seed
  3-fold protocol goes from 54 s to 12 s (`LOs-MCQ`) and 104 s to 17 s
  (`Unique-step-MCQ`) on 12 cores.

  Partitions are drawn in the parent before any fit starts and results are
  collected in submission order, so the worker count is a wall-clock knob only:
  `n_jobs=-1` returns bitwise what `n_jobs=1` returns, asserted in
  `test_worker_count_does_not_move_a_single_digit`.

### Changed

- `repeated_cross_validate` takes its cross-validation arguments explicitly
  instead of forwarding `**kwargs` to `cross_validate`, so that all
  (seed, fold) pairs can be dispatched as one pool rather than one pool per
  seed. Keyword callers are unaffected; the accepted names are unchanged.

## 0.2.0 — 2026-08-16

First tagged release. 0.1.0 existed only as a version string in the working
tree and was never published, so everything the package does is listed here.

### Models

- **AFM** — the Additive Factors Model, grounded in and validated for
  equivalence against LearnSphere's `AnalysisFastAfmAndCv` workflow output.
  `build_afm_design` / `fit_afm`, with per-KC intercepts and learning rates
  from `AFMFit.kc_values`.
- **PFA** — Performance Factors Analysis (Pavlik, Cen & Koedinger 2009) as
  fixed-effects logistic regression with strictly-prior success/failure counts;
  per-KC or pooled slopes, optional student intercepts. The audited reference
  builds its counts including each attempt's own outcome; that construction is
  reproducible through an explicit option that warns rather than silently
  changing the model.
- Both families read the same six-column DataShop student-step export, so
  switching families never means reshaping data.

### Statistics

- **Identified parameter counts.** Aliased columns are removed exactly rather
  than numerically, so `n_params = rank(X)` and AIC/BIC never charge for
  parameters that do not exist. Three sources are handled — never-repeated KCs,
  KCs tagging identical steps, and the student/KC sum redundancy once per
  connected component. Anything left over raises instead of being counted.
- **Separation detection.** Coefficients with no finite maximum-likelihood
  estimate are flagged (`fit.separated`, a `Separated` column in `kc_values`)
  instead of being printed as if they were real numbers.
- **A convergence certificate.** The objective is convex, so `fit.is_optimal`
  reports a KKT check, independent of what the optimizer claims about itself.
- **Cross-validation.** Unstratified, response-stratified, student-blocked and
  item-blocked schemes; per-fold and pooled RMSE conventions; seeded repeats;
  and `paired_cross_validate` for scoring several KC models on identical folds.
- **LearnSphere compatibility.** `build_afm_design(data, learnsphere_compat=True)`
  reproduces LearnSphere's ridge and parameter-counting conventions for
  matching a published table; the default is the statistically clean variant.

### Interfaces

- `leapfit-afm` and `leapfit-pfa` console scripts, including `--list-models`,
  repeated seeded CV, several `--cv` schemes in one run, `--out`/`--cv-folds`
  exports, and `--predictions`.
- `fit.annotate(data)` returns the input file unchanged except for an appended
  `Predicted Error Rate (<model>)` column per fitted KC model, following
  DataShop's convention, so learning-curve tooling can consume the result.

### Packaging

- Requires Python 3.11+; CI runs lint and the suite on 3.11, 3.12 and 3.13,
  then installs the built wheel into a clean environment and fits a model from
  outside the source tree.
- Dependencies are numpy, pandas and scipy only.
- Not on PyPI. Install from a tag; see the README.
