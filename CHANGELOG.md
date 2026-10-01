# Changelog

Notable changes per release. Versions follow [semantic versioning](https://semver.org);
while the major version is 0, a minor bump may change the public API.

## Unreleased

### Changed

- **Fits are about 1.6-1.9x faster, with the same results.** The objective
  and its gradient are evaluated together, from one product with the design
  and one exponential per row, where they used to be computed separately.
  On the E-learning 2022 export a TNC fit of `LOs-new-MCQ` takes 0.24 s
  instead of 0.39 s, and of `Unique-step-MCQ` 0.46 s instead of 0.86 s, with
  the same number of evaluations and the same optimum to the last bit on the
  machine measured. Every cross-validation fold, LFA candidate and LKT
  evaluation inherits it. `fit_lkt_pars` also builds what does not depend
  on a parameter (each component's practice sequences, and the clock) once
  per search rather than once per evaluation. Together these take the
  vignette's RPFA search from 10.6 s to 5.0 s.
- `leapfit-lfa` reads the export once, as `leapfit-afm` does, rather than
  once per KC model it loads: the factor models, `--root` and each
  `--compare`. On the E-learning 2022 export, which has 100 KC models,
  loading 10 of them took 5.6 s and now takes 1.0 s, and the default
  `--factors` loads all 100. An unknown `--root` is now refused before
  anything is loaded, like an unknown `--compare`.
- The unit tests are organised by the module they exercise. `test_data`,
  `test_design`, `test_fit`, `test_crossval` and `test_cli` join the family
  files, which keep only what is specific to their family. The builders the
  files share (`rollup`, `step_data`, `synthetic` and the rest) live in
  `tests/helpers.py`; each family file used to define its own copy. The
  example export is loaded once per session, by a fixture in
  `tests/conftest.py`.
- 17 test cases that repeated what another test already checks are removed
  or folded into that test, and parametrized tests replace near-identical
  ones. Expensive results are now computed once per module: the LFA search
  and validation on the example, the LKT vignette's chunk fits and parameter
  search, and the LearnSphere export, which was read 33 times. The suite
  without data runs 314 tests in about 18 s, down from 330 in 28 s; with every
  fixture it takes 83 s, down from 137 s. The console-script check now covers
  `leapfit-pfa` too.

### Added

- `StepData.prior_counts(labels=None)`: each observation's prior successes
  and failures per label, strictly before the attempt and over
  `practice_order`. It is now the one implementation behind
  `StepData.recomputed_opportunities`, `pfa.success_failure_counts` and
  `lkt.history_counts`, which each had their own copy of the loop; what
  they return is unchanged. LKT builds its features over the same grouping
  of each student's practice, so the tests that only checked the copies
  agree with each other are gone.
- `Block.from_levels(name, labels, values=None)`: a block with one column
  per level, from each row's labels and, optionally, the value each label
  carries. The AFM and PFA builders construct every block through it, where
  each built its own sparse triplets, and give the same matrices.
- `Aliased.in_blocks(*blocks)` and `Separated.in_blocks(*blocks)`: the
  columns in the named blocks, by their names within the block. The KC and
  component tables and the LFA screens read separation through it, where
  each combined `by_block()` entries by hand; the two records now share
  their column handling instead of each defining it.
- `LFAState.separated`, the separated columns that `n_separated` counts, and
  `LFAState.path`, its moves as the frontier and validation tables print
  them. The search reads the root's separation off its state, where it
  rebuilt the root's design a second time to find it.
- `Design.get(name)`: the block of that name, or `None`. It replaces five
  hand-written lookups across the package and the tests.
- `Design.lower` and `Design.upper`: the coefficient bounds as arrays, with
  `-inf` and `inf` where a coefficient is unbounded. `Design.bounds` keeps
  the `(min, max)` pairs with `None`; `fit_logistic` no longer converts them
  back and forth on every fit.

### Fixed

- The source distribution ships `examples/README.md`. The repository's
  `.gitignore` drops every `*.md` and hatchling honours it, so every release
  so far packed the examples without their README. It is now named in
  `artifacts`, as `CHANGELOG.md` already was.
- `uv build` in a working copy no longer packs a second virtual environment,
  such as `.venv-pybkt`, into the sdist. uv hides each environment from git
  with a `.gitignore` inside it, which hatchling does not read, so building
  0.6.0 that way failed on the environment's link to its interpreter.
  `.venv*/` is now ignored at the root.

## 0.6.0 — 2026-09-30

### Changed

- **The sum-redundancy pass is no longer written in terms of students and
  KCs.** `Design.identify` detected its one redundancy by looking for blocks
  literally named `student` and `kc_intercept`; it now detects *every* such
  redundancy from the row sums, pair by pair. A block whose rows all sum to the
  same positive constant spans the all-ones direction, so any two of them are
  dependent — on each connected component of *their own* graph on which both
  blocks' row sums are constant. Where three or more blocks take part those
  pairwise dependencies overlap (three crossed blocks carry two, not three), so
  the columns to drop are chosen by exact elimination over them, in rationals:
  `prefer_drop` first, then latest-declared, each block from its last level
  back, so the block a design names first keeps every level, and exactly as
  many go as the dependencies span.

  **AFM, PFA and LFA are unaffected, bit for bit.** With two blocks the rule is
  one reference level per component: the same columns, and the same
  `Aliased.reasons` strings, including the per-component wording. Checked
  against the previous pass on 201 designs from every shipped builder — AFM
  with and without recomputed opportunities, PFA per-KC, pooled and with
  student intercepts — over 103 KC models, all identical, and all three
  equivalence suites still pass. `Design.row_components()` now takes the blocks
  to build its graph over, defaulting to every block that covers the rows as
  before. It also no longer sends a dropped reference student's rows to a
  phantom component of their own on an *already identified* design, which the
  old mapping did by labelling each row through its student's now-empty column.

  What this unlocks is per-level intercepts on more than two components in the
  shapes real exports have. Crossed factors: students, KCs and a cohort or
  condition column. Nested ones: items declared after the KCs they sit in
  identify with one reference item per KC, under a student intercept too,
  which used to be refused. Cohorts that one pair of blocks separates while
  another connects them — students and conditions never shared between
  classes that share KCs — each keep a reference level of their own. And a
  feature that touches every row no longer
  merges cohorts: `propdec` is positive from the first attempt, so the graph
  over the whole design counted its one column as a level every row shares, and
  `student + kc + kc:propdec` on an export with two cohorts was refused.

  What it still refuses is what 0.5.0 refused as a mistake in the
  specification rather than a property of the data: a block the others
  already span. The first case is one factor under two names — two blocks
  that partition the rows identically, such as the export's own `KC (...)`
  column beside the parsed KC, or the vignette's KC, which *is* its problem
  name, beside `Problem Name` — and `identify` names the pair instead of
  silently dropping one copy whole. The second is a hierarchical parent
  declared after the levels it groups: topics appended with `with_blocks`
  over the KCs they group, or `kc` after `item` in an LKT spec. Elimination
  would take every column of it, so it is refused, naming the block that
  spans it; declared first, the parent keeps every level and the finer block
  gives up one per parent instead. `prefer_drop` alone may still go whole, so
  a cohort of one student gives up its only student as before. A dependence
  that is not a sum redundancy between factors still raises as it always did:
  an accumulator collinear with what is there, or `lineafm$` beside `linesuc$`
  and `linefail$`.

- `Term.par` -> `Term.pars` (a tuple; a scalar is accepted for the
  single-parameter features), and `STATIC_FEATURES` -> `FEATURE_NAMES`. Both
  from the same release, both unreleased.

### Added

- **`fit_lkt_pars`: fitting LKT's feature parameters, not just choosing them.**
  A profile likelihood over the design builder — each candidate parameter
  vector recomputes the features, rebuilds the design and fits the
  coefficients to convergence, and an outer L-BFGS-B walks the resulting
  surface. That is the reference's own procedure; three things around it are
  not.

  *The parameters are counted.* `LKTFit.n_params` becomes `rank(X) + k`, so a
  searched model's AIC and BIC are charged for the search. The reference
  reports no parameter count at all.

  *Stationarity is measured, not inferred.* The profile is **not** convex, so
  the inner KKT certificate — which does prove a global optimum over the
  coefficients — says nothing about the parameters. After the optimizer stops,
  every parameter is stepped by `PARAMETER_STEP` in both directions and
  refitted; `LKTProfile.max_gain` is the best improvement any of those found,
  in nats, and `is_stationary` compares it against `PARAMETER_TOLERANCE`. It is
  reported next to, not instead of, the optimizer's own `converged` flag,
  because they answer different questions.

  *Restarts turn an assumption into a measurement.* Pass several `starts` and
  the summary reports how many distinct optima they reached and how far apart.
  One start says nothing about other basins, and the summary says so.

  Identification is decided once at the seed and held for every evaluation
  (`_conform`), for the reason `Design.take` already holds it across CV folds:
  a parameter value that made one more column identically zero would change
  the parameter count mid-search and make one evaluation's AIC incomparable
  with the next's. The seed's record of what it dropped and why is held with
  it, renamed to the blocks' current names, so a searched fit reports the real
  reason each column is missing rather than a generic one. And parameters are
  written into block names exactly, so `fit.terms` and `fit.notation()` give
  back the estimate itself rather than six significant digits of it.

- **`objective=`, because the reference profiles a surface it is not
  maximizing.** `"penalized"` (the default) profiles the objective the inner
  solver actually maximizes, so the pair (parameters, coefficients) maximizes
  one function. `"likelihood"` profiles the plain Bernoulli log-likelihood of a
  *ridged* fit, which is the reference's choice and is not a single objective;
  the two coincide whenever `l2 = 0`, and part company as soon as `cost` is
  finite.

- **The reference's RPFA search reproduced, path and endpoint.** Its vignette
  prints every evaluation, so the whole `optim` trajectory is visible.
  `test_the_references_search_path_is_reproduced_point_by_point` checks four
  of the points it visited (agreement around 0.05 nats — tighter than any
  other chunk, because the spec has no time features), and
  `test_the_parameter_search_reaches_the_references_optimum` runs the loop:
  the reference stops at `propdec2 = 0.3736667`, leapfit reaches `0.37352`
  with a better likelihood from a stationary point. Worth knowing why it can:
  R's `optim` at `factr = 1e12` stops once the objective improves by less than
  about 6 nats, and the reference's own next probe already reported a better
  value than the one it returned.

  Two defaults follow the reference deliberately. `PARAMETER_BOUNDS` is its
  `(1e-5, 0.99999)`, recycled across every parameter whatever it means; and
  the outer gradient is taken the way R's `optim` takes it — central
  differences at `ndeps = 1e-3`, each side stopped at its bound — rather than
  by scipy's own forward differences at `1e-8`, because the profile is only as
  smooth as the inner solve is tight and a step that small differentiates the
  inner optimizer's own noise.

- **LKT features that read the clock and the outcome history.** Thirty-one
  features now, up from eleven: decayed histories
  (`expdecafm`, `expdecsuc`, `expdecfail`, `propdec`, `propdec2`, `logitdec`),
  recency (`recency`, `recencysuc`, `recencyfail`), power-law forgetting
  (`base`, `basesuc`, `basefail`, `base2`, `base2suc`, `base2fail`, `dashafm`,
  `dashsuc`) and the two four-parameter spacing models (`base4`, `ppe`).

  A parametric feature *requires* its `pars` — `build_lkt_design` holds them
  there, and `fit_lkt_pars` starts its search from them — and `Term.par`
  becomes `Term.pars`, a tuple, scalar accepted.

  Two reference behaviours are reproduced deliberately and are worth knowing
  about. `logitdec` truncates at a **60-trial window** (`slidelogitdec`'s
  `max(1, i - 60)`), undocumented in the paper and worth 0.06 logits at
  `d = .97` over 200 trials; `LOGITDEC_WINDOW` names it. And the mean-spacing
  sentinel of `-1` at a level's second practice is what selects `base4`'s
  unspaced branch, rather than the position doing it.

- **A clock on `StepData`.** `epoch_times()` parses `First Transaction Time` to
  seconds, and `time_on_task()` accumulates `Step Duration (sec)` — lagged, per
  student, over `practice_order()` — into the clock that ignores gaps between
  sessions. Both **refuse** when the export lacks the column instead of
  substituting a row number or a constant, because every substitute is a
  different model. Neither column is required by anything that ran before.

- **Three more of the reference's published chunks reproduced**, spanning the
  whole implemented surface: `logitdec + recency` at **-25474.531**, `PPE` at
  **-24695.586** and `base4` at **-25969.925**, each within 0.2 nats and each
  on the better side of the published value from a KKT-certified optimum. With
  the AFM chunk that is four, and `test_every_chunk_lands_on_the_better_side_of_its_published_value`
  checks the *shape* of all four together: a sign that flipped between them
  would say the agreement is noise around a wrong design.

  The equivalence fixture now also carries `Step Duration (sec)`, derived the
  way the vignette derives it — `(end latency + review latency + 500)/1000`,
  overwriting the export's own duration column — because that is what the
  published `base4` number was produced from.

- **Features are refused by name with the reason, and three of those reasons
  are reference defects.** `errordec` reads `data$pred_ed`, which nothing in
  the package ever assigns; `recencystudy` and `recencytest` read
  `<component>previousstudy`, whose assignment is commented out in
  `computeSpacingPredictors`; `dashfail` is counted in `parlength` but has no
  branch in `computefeatures`. The reference cannot compute any of them either.

- **Logistic Knowledge Tracing, static features** (`leapfit/lkt.py`). A model
  is a list of `Term`s, each pairing a *component* (any factor the export
  carries — student, item, KC, or a column of the source table) with a
  *feature* of that component's practice history for that student; the
  reference's `$` suffix fits one coefficient per level instead of one shared.
  `lkt_terms` takes the reference's own parallel `components`/`features`
  vectors so a specification can be copied out of a paper unchanged, and
  `LKTFit.component_values` generalizes `AFMFit.kc_values` to any component.

  AFM and PFA turn out to be single LKT specifications, and
  `build_lkt_design` reproduces both designs column for column — including the
  aliasing, the parameter count and the likelihood. Both identities are pinned
  by tests, with the AFM one against `recompute_opportunities=True`, because
  LKT counts practice from the ordering rather than reading DataShop's
  `Opportunity` column.

  The features here are the pure functions of prior success and failure
  counts: `intercept`, `lineafm`, `logafm`, `powafm`, `linesuc`, `logsuc`,
  `linefail`, `logfail`, `linecomp`, `prop` and `numer`. `history_counts` is
  `success_failure_counts` with the KC hardcoding lifted to an arbitrary
  component. The clock and the decayed histories are the entries above, and
  whatever the reference computes that none of them implements is **refused
  by name with the reason**, because a spec silently missing a term is worse
  than one that will not build.

- **`cost=`, the reference's penalty, expressed exactly.** LKT solves through
  `LiblineaR(type = 0, cost = 512)`, which is this package's objective with
  `l2 = 1/cost` on every column. The default stays `l2 = 0` with an identified
  design: that ridge is an identification device, and `Design.identify`
  removes the redundancy exactly instead. `test_the_reference_ridge_hides_a_separation_the_default_reports`
  pins what the difference costs.

- **Equivalence against the reference's published output**
  (`tests/test_lkt_equivalence.py`), without an R interpreter. The CRAN
  tarball ships both halves of a fixture — `largerawsample.rda` and a
  precompiled vignette printing each model's log-likelihood to eight decimals
  — which is the same kind of artifact as LearnSphere's `model_values.xml`,
  and the module builds its own fixture from the tarball when run as a script.
  On the vignette's AFM chunk leapfit reaches **-27346.740** against its
  published **-27347.207**: 0.47 nats better, from a KKT-certified optimum, so
  by the two-sided criterion this suite already uses the gap is the
  reference's optimizer stopping early. The test decomposes it rather than
  asserting it — the ridge accounts for 0.036 nats and the choice of reference
  level for 0.015, leaving `LiblineaR`'s default `epsilon = 1e-4`.

### Fixed

- A term whose feature divides by an elapsed time now **raises** when the
  assembled column is non-finite, naming the feature. Two attempts on one level
  sharing a timestamp make an age of zero, and the reference raises it to a
  negative power and hands `Inf` to its solver.
- The development install in `README.md` was `uv sync`, which leaves out the
  `dev` extra, so `uv run pytest` had no pytest of its own to run. It is now
  `uv sync --extra dev`, as CI runs it.

### Known limitations

- No global intercept (`interc=TRUE`), no `*` or `:` connectors, no
  `interacts`, no `autoKC`, no `@` random effects, and no `leapfit-lkt`
  console script yet. `interc` is now only a small step — the generalized
  identification pass above is what it was waiting on, since an all-ones column
  is just one more block that covers every row — but it is not implemented.
  Where a spec already carries a per-level intercept it changes nothing but the
  parameterization anyway, which is why the three `interc=TRUE` chunks
  reproduced here match without it.
- No search over *terms*. `fit_lkt_pars` fits a specification's parameters;
  choosing which features on which components to include is the reference's
  `buildLKTModel`, and that is a consumer of the family — the shape
  `leapfit.lfa` already has over `leapfit.afm` — rather than more of
  `leapfit.lkt`.
- Steps tagged with several KCs are not checked against the reference, whose
  sample data has one KC per step. leapfit enters them additively, as AFM
  does: a per-level term gives each of the step's KCs its own column, and a
  shared coefficient takes their sum
  (`test_a_shared_coefficient_sums_a_rows_levels`).

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
