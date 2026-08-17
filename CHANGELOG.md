# Changelog

Notable changes per release. Versions follow [semantic versioning](https://semver.org);
while the major version is 0, a minor bump may change the public API.

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
