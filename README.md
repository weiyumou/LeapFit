# leapfit

Student models for learning analytics and educational data mining, fitted
directly from [DataShop](https://pslcdatashop.web.cmu.edu/) student-step
exports. Today that is the **Additive Factors Model (AFM)** and **Performance
Factors Analysis (PFA)**; Bayesian Knowledge Tracing (BKT) is on the
[roadmap](#roadmap).

- **One input format.** Every model reads the same six columns of a
  student-step file, so switching model families never means reshaping data.
- **Grounded.** The implementation is adapted from LearnSphere's reference
  components and validated for equivalence against their output, so results
  stay comparable with numbers DataShop already reports.
- **Honest statistics.** Parameter counts equal the rank of the design,
  coefficients with no finite estimate are flagged instead of printed as if
  real, and every fit carries a convergence certificate — checked, not assumed.

## Install

```bash
# Not on PyPI — install from a release tag:
uv pip install "git+https://github.com/weiyumou/LeapFit@v0.4.0"

# ...or for development:
git clone https://github.com/weiyumou/LeapFit && cd LeapFit
uv sync                # or: uv pip install -e ".[dev]"
uv run pytest          # 135 pass in ~50s; equivalence extras skip without R / artifacts
```

Another project can depend on leapfit with the same direct reference —
`"leapfit @ git+https://github.com/weiyumou/LeapFit@v0.4.0"` in its
`dependencies` or in an extra. Two consequences worth knowing before you do:
PyPI refuses distributions whose metadata carries a direct URL, so a package
that is itself published to PyPI cannot declare leapfit this way even in an
extra nobody installs; and a direct reference pins one commit rather than
resolving a range, so every upgrade is an edit downstream.

## Quickstart

The repository ships a small synthetic dataset
([`examples/student-step.txt`](examples/)) tagged with two KC models — `Topics`,
the model the responses were actually generated from, and `Skills`, a finer
relabelling of the same steps.

```python
from leapfit import load_student_step, build_afm_design, fit_afm, cross_validate

data   = load_student_step("examples/student-step.txt", kc_model="Topics")
design = build_afm_design(data)
fit    = fit_afm(design, data.y)

print(fit.summary())                     # log-likelihood, AIC/BIC, optimality
print(fit.kc_values(data))               # per-KC intercepts and learning rates
print(cross_validate(design, data, scheme="item_blocked").summary())
```

Switching model families never means reshaping data — PFA is the same calls on
the same `data`:

```python
from leapfit import build_pfa_design, fit_pfa

pfa = fit_pfa(build_pfa_design(data), data.y)   # per-KC success/failure slopes
```

The same comparisons from the command line:

```bash
leapfit-afm examples/student-step.txt --list-models
leapfit-afm examples/student-step.txt --cv item_blocked --seeds 0:5
leapfit-pfa examples/student-step.txt --cv item_blocked --seeds 0:5

# both blocking schemes on one set of fits, with the runs behind the means
leapfit-afm examples/student-step.txt --cv student_blocked --cv item_blocked \
    --seeds 0:10 --out comparison.csv --cv-folds cv-folds.csv
```

```
kc_model  n_kcs  n_obs  n_params  n_separated  log_likelihood      aic      bic  is_optimal  cv_rmse
  Skills     12    480        35            0       -267.8217 605.6435 751.7260        True   0.4988
  Topics      4    480        19            0       -268.9846 575.9691 655.2711        True   0.4586

paired contrasts, within-fold (negative mean_diff = model beats baseline):
      scheme  model baseline  mean_diff  sd_diff  n_folds  folds_better
item_blocked Skills   Topics   0.040176 0.024314       15             0
```
*(columns abridged)*

AIC, BIC, and held-out RMSE all prefer `Topics` — the planted true model — over
the finer `Skills`. Every fitted learning rate is positive, as generated.

When several KC models cover the same rows — the usual case — they are scored
on **identical folds** and the contrasts table reports each model's held-out
RMSE difference against the best one, *paired by fold*: here `Skills` loses to
`Topics` on all 15 shared partitions, a far sharper statement than comparing
two independently resampled means. `--baseline NAME` picks the reference
model, `--contrasts FILE` writes the table, and `--no-paired` restores
independent per-model folds.

## Input format

A tab-separated student-step file with six required columns. Everything else in
a full DataShop export is ignored.

| column | meaning |
|---|---|
| `Anon Student Id` | the learner; the unit of student-blocked CV |
| `Problem Name`, `Step Name` | together the item label; the unit of item-blocked CV |
| `First Attempt` | `correct` is a success; `incorrect` / `hint` / `unknown` are failures |
| `KC (<model>)` | the step's knowledge component(s), `~~`-separated when there are several |
| `Opportunity (<model>)` | how many times the student has met each KC, numbered from 1, aligned by position |

Optional: `First Transaction Time` defines the practice order, letting
`recompute_opportunities=True` correct a miscounted `Opportunity` column.

A file may carry any number of KC models; `list_kc_models(path)` enumerates
them, and each fits independently. Malformed input raises with the row number
rather than fitting something silently wrong — including outcome vocabularies
the parser does not recognize (a file coding `1`/`0` needs
`success_values=("1",), failure_values=("0",)` stated explicitly).
[`examples/generate.py`](examples/generate.py) is a minimal reference for
producing compatible files from your own data.

## What you get beyond point estimates

- **Identified parameter counts.** Aliased columns are removed, so
  `n_params = rank(X)` and AIC/BIC never charge for parameters that do not
  exist. On one real export's finest KC model that is 959 phantom parameters,
  25% of its BIC penalty. Three sources, all removed exactly rather than
  numerically: a KC no student practises twice; two KCs that tag identical
  steps (one keeps the estimate, the other reports `NaN` rather than a number
  that is really its twin's); and the student/KC sum redundancy — *once per
  connected component*, because an export whose cohorts never met the same
  material carries one of them per cohort, and their intercept levels are then
  comparable only within a cohort. Anything left over raises instead of being
  counted, so a collinear block added later cannot slip through.
- **Separation detection.** A KC answered correctly by everyone has no finite
  intercept estimate; leapfit reports it (`fit.separated`, a `Separated` flag
  in `kc_values`) instead of printing the arbitrary number the optimizer
  stopped at.
- **A convergence certificate.** The objective is convex, so the fit checks the
  KKT conditions and `fit.is_optimal` says whether this is *the* optimum —
  independent of what the optimizer claims about itself.
- **Reproducible cross-validation.** Unstratified, response-stratified,
  student-blocked, and item-blocked schemes; both per-fold and pooled RMSE
  conventions; seeded repeats. Several KC models over the same rows are scored
  on identical folds (`paired_cross_validate`, the CLI default), so model
  comparison is a paired within-fold contrast (`paired_contrasts`) rather than
  a t-test over non-independent resamples — and one paired run also yields
  each model's per-seed scores under either convention (`paired_scores`). Pass
  `n_jobs` (or `leapfit-afm ... -j -1`) to fit the folds across cores:
  partitions are drawn before any fit starts and results are collected in
  order, so the numbers are identical to a single-process run.
- **Student-step in, student-step out.** `fit.annotate(data)` — or
  `leapfit-afm ... --predictions out.txt` — returns your input file unchanged
  except for one appended `Predicted Error Rate (<model>)` column per fitted KC
  model, following DataShop's convention (error rate = `1 − P(correct)`, blank
  for rows without a KC), so learning-curve tooling that reads DataShop exports
  can consume the result directly.
- **A compatibility switch.** `build_afm_design(data, learnsphere_compat=True)`
  reproduces LearnSphere's exact conventions (ridge, parameter counting) when
  you need to match a published table; the default is the statistically clean
  variant. 

## Roadmap

| model | state | notes |
|---|---|---|
| **AFM** | shipped | validated for equivalence against LearnSphere workflow output |
| **PFA** | shipped | canonical fixed-effects PFA (Pavlik, Cen & Koedinger 2009) with strictly-prior counts; per-KC or pooled slopes, optional student intercepts. The audited reference builds its counts *including* each attempt's own outcome — that construction is reproducible here via an explicit option that warns, never silently |
| BKT | planned | to be validated against the standard `standard-bkt` C++ tool |

## Development

CI runs lint and the full suite on Python 3.11–3.13, then builds the wheel,
installs it into a clean environment, and fits a model from outside the source
tree. The equivalence tests require LearnSphere run artifacts and skip without
them, so a bare clone is always green:

```bash
uv run pytest                                       # 118 pass, 11 skip, ~25s
AFM_WF3990_DIR=/path/to/artifacts uv run pytest     # + 8 LearnSphere equivalence tests
```

With `Rscript` on `PATH`, three more tests fit the same designs through R's
`stats::glm` and require agreement to numerical precision (~1e-8 in
log-likelihood) — any R works, e.g.
`micromamba create -p /tmp/r-env -c conda-forge r-base`.

## License

[MIT](LICENSE)
