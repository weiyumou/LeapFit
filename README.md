# leapfit

Student models for learning analytics and educational data mining, fitted
directly from [DataShop](https://pslcdatashop.web.cmu.edu/) student-step
exports. Today that is the **Additive Factors Model (AFM)**, **Performance
Factors Analysis (PFA)** and **Logistic Knowledge Tracing (LKT)** — of which
the first two turn out to be single specifications — plus **Learning Factors
Analysis (LFA)**, a search for the KC model itself, scored by AFM. Bayesian
Knowledge Tracing (BKT) is on the [roadmap](#roadmap).

- **One input format.** Every model reads the same six columns of a
  student-step file, so switching model families never means reshaping data.
  Transactions, exported from DataShop or made for its import, are rolled up
  into those columns first, the way DataShop's own student-step export is.
- **Grounded.** AFM, PFA and LFA are adapted from LearnSphere's reference
  components and validated for equivalence against their output, so results
  stay comparable with numbers DataShop already reports. LKT is validated
  against the published output of the CRAN `LKT` package but written
  clean-room, because that package is GPL-3.
- **Honest statistics.** Parameter counts equal the rank of the design,
  coefficients with no finite estimate are flagged instead of printed as if
  real, and every fit carries a convergence certificate — checked, not assumed.

## Install

```bash
# Not on PyPI — install from a release tag:
uv pip install "git+https://github.com/weiyumou/LeapFit@v0.7.0"

# ...or for development:
git clone https://github.com/weiyumou/LeapFit && cd LeapFit
uv sync --extra dev    # or: uv pip install -e ".[dev]"
uv run pytest          # 315 pass, 44 skip in ~16s; extras need R / reference-run artifacts
```

Another project can depend on leapfit with the same direct reference —
`"leapfit @ git+https://github.com/weiyumou/LeapFit@v0.7.0"` in its
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

**A model is a list of terms.** LKT pairs a *component* — a factor whose levels
partition the data — with a *feature* of that component's practice history, and
`$` fits one coefficient per level instead of one shared. AFM and PFA are two
points in that space (`build_lkt_design` reproduces both designs column for
column, which is a test); the interesting specifications are the other ones.

```python
from leapfit import build_lkt_design, fit_lkt, lkt_terms

terms = lkt_terms(components=("student", "student", "kc", "kc"),
                  features=("intercept", "logsuc", "intercept", "lineafm$"))
lkt = fit_lkt(build_lkt_design(data, terms), data.y)

print(lkt.notation())                        # the spec it fitted, in LKT notation
print(lkt.component_values(data, "kc"))      # one row per level, one column per feature
```

Features that read the clock or a decayed history are here too — `recency`,
`base`, `base2`, `base4`, `ppe`, `dashafm`, `logitdec`, `propdec`, `expdec*`.
A parametric feature always takes its `pars`, and `build_lkt_design` holds
them where you put them:

```python
terms = lkt_terms(components=("student", "kc", "kc", "kc"),
                  features=("intercept", "intercept", "expdecafm", "recency"),
                  pars=(None, None, 0.9, 0.5))
```

Time features need `First Transaction Time`, and the `base2` family also needs
`Step Duration (sec)`; an export without one is refused rather than defaulted.
Features the reference computes that are not implemented are refused by name
with the reason — three of them because the reference cannot compute them
either.

To **fit** the decay rates instead of choosing them, `fit_lkt_pars` profiles
them, starting from those same `pars`: each candidate rebuilds the features,
refits the coefficients, and the outer optimizer walks the resulting surface.

```python
from leapfit import fit_lkt_pars

profile = fit_lkt_pars(data, terms, starts=[(0.9, 0.5), (0.3, 0.3)])
print(profile.summary())   # estimates, stationarity, restart spread
print(profile.frame())     # one row per parameter, with its seed and bounds
```

Two things that costs, and both are reported rather than assumed. A fitted
decay rate **is a parameter**, so it is counted in `n_params` and charged for
in AIC and BIC — the reference reports no parameter count at all. And the
profile is *not* convex, so unlike the coefficient fit its optimum carries no
global certificate: `is_stationary` says only that no small step improves it,
and passing several `starts` turns "probably fine" into a measurement.

**Which KC model?** LFA turns that into a search. It is not another student
model — the states *are* KC labellings, and each is scored by fitting AFM to
it, so the identification pass, the separation check and the optimality
certificate come along unchanged.

```python
from leapfit import build_factor_matrix, lfa_search, validate_top

authored = {m: load_student_step("examples/student-step.txt", kc_model=m)
            for m in ("Topics", "Skills")}
factors  = build_factor_matrix(authored)   # the difficulty factors to split by
search   = lfa_search(data, factors)       # greedy best-first on BIC

print(search.summary())                    # trajectory, refusals, certificates
print(search.frame().head())               # the ranked frontier, one row per state

# then check the shortlist out of sample, on folds shared by every candidate
print(validate_top(search, data, n=3, extra=authored, seeds=(0, 1, 2)).summary())
```

```
LFA search over 16 difficulty factor(s), heuristic=BIC
  6 expansion(s), 83 state(s) evaluated, stopped: no improvement
  root  644.299
  best  641.984  (improvement 2.315 over 1 move(s))
  no moves refused
  0 of 83 evaluated state(s) not at a certified optimum
2 KCs, 15 params  ll=-274.6886  AIC=579.377  BIC=641.984
  split all by fractions

Paired CV of 5 candidate(s): item_blocked, 3 folds x 3 seed(s), pooled RMSE
  BIC picked 'rank1' (cv_rmse 0.4590); held-out picks 'Topics'
  they DISAGREE — the criterion's pick is not the best predictor
  rank correlation 0.400
  paired contrasts against 'root' (negative mean_diff beats it):
     model baseline  mean_diff  sd_diff  n_folds  folds_better
    Topics     root  -0.005674 0.005766        9             7
     rank1     root  -0.004796 0.006096        9             7
     rank2     root  -0.000112 0.008829        9             6
    Skills     root   0.028462 0.025754        9             0
```

Read that as two answers, not one. BIC prefers a two-KC model to everything the
search reached; held-out RMSE prefers `Topics`, the planted truth, which BIC
ranks *below* it. Choosing a KC model by an information criterion and checking
it out of sample are different questions, and `validate_top` reports the
disagreement instead of hiding it — the protocol the reference's own follow-up
prescribes. Pass `n_jobs=-1` to score each expansion across cores — a 5×
wall-clock win on a real export, and bitwise the same answer.

The same search from the command line, writing the labelling it found back out
in a form DataShop can import:

```bash
leapfit-lfa examples/student-step.txt \
    --factors Topics --factors Skills --validate 5 --compare Topics \
    --seeds 0:3 -j -1 --out frontier.csv --qmatrix discovered-kc-model.txt
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

Two optional columns, read only by the models that need them.
`First Transaction Time` defines the practice order — which lets
`recompute_opportunities=True` correct a miscounted `Opportunity` column — and,
parsed to seconds by `StepData.epoch_times()`, supplies the intervals an LKT
recency, forgetting or spacing feature measures. `Step Duration (sec)`
accumulates into `StepData.time_on_task()`, the clock that ignores the gaps
between sessions. A model that needs one and does not have it is refused, not
defaulted: every substitute value is a different model.

A file may carry any number of KC models; `list_kc_models(path)` enumerates
them, and each fits independently. Malformed input raises with the row number
rather than fitting something silently wrong — including outcome vocabularies
the parser does not recognize (a file coding `1`/`0` needs
`success_values=("1",), failure_values=("0",)` stated explicitly).
[`examples/generate.py`](examples/generate.py) is a minimal reference for
producing compatible files from your own data.

### Transactions

A DataShop transaction export, one row per action rather than per step, is
rolled up into its student-step table first. So is a file made for DataShop's
import, which lacks the columns DataShop adds on import:

```python
import pandas as pd
from leapfit import from_frame, load_transactions, rollup_transactions

data = load_transactions("transactions.txt", kc_model="Topics")

# or roll it up once, for several KC models
steps  = rollup_transactions(pd.read_csv("transactions.txt", sep="\t", dtype=str,
                                         keep_default_na=False))
models = {m: from_frame(steps, m) for m in ("Topics", "Skills")}
```

The commands accept either, and say on stderr that they rolled it up. A
step's row is the student's first attempt at it in one problem view, the
transaction DataShop numbered `Attempt At Step` 1, and its `Outcome` becomes
`First Attempt`. A file made for import has no `Attempt At Step`, because
DataShop numbers the attempts itself on import, so they are numbered the same
way: every transaction that names a step, in time order. Its `Time` may be in
any format DataShop's import reads, and comes back as DataShop's exports
write it, with Unix milliseconds in UTC. It must carry `Problem View`, which
DataShop can derive on import and this does not. A step's KCs in a model are
every label in the model's `KC (<model>)` columns, which the export repeats
once per KC, and `Opportunity (<model>)` counts each student's encounters with
each KC, in practice order. Where the file has `Duration (sec)`, the step's
durations add up to `Step Duration (sec)`. Checked against DataShop's own
student-step exports of three datasets, one of them rolled up from the very
file imported to make it, every row, outcome and time agrees on all three,
and every column on one; the docstring of `rollup_transactions` records where
the others differ, and why.

## What you get beyond point estimates

- **Identified parameter counts.** Aliased columns are removed, so
  `n_params = rank(X)` and AIC/BIC never charge for parameters that do not
  exist. On one real export's finest KC model that is 959 phantom parameters,
  25% of its BIC penalty. Three sources, all removed exactly rather than
  numerically: a KC no student practises twice; two KCs that tag identical
  steps (one keeps the estimate, the other reports `NaN` rather than a number
  that is really its twin's); and sum redundancies between blocks that span
  the all-ones direction — *all of them, pair by pair and component by
  component*. Any block whose rows sum to one positive constant spans it, and
  so can a KC block whose steps carry different numbers of KCs, when one KC
  only ever appears beside another, say; that is decided exactly, as a linear
  system in rationals. So two such blocks (student and KC intercepts, or
  either beside an item or cohort factor) are dependent on each connected
  component of their own graph: once per cohort, where cohorts never met the
  same material, and once per KC, where items nest within KCs. With three or
  more blocks the dependencies overlap, and exact elimination drops just as
  many columns as they span. Intercept levels are then comparable only within
  a cohort. Anything left over raises instead of being counted, so a collinear
  block added later cannot slip through: one factor entered twice under two
  names is refused by name, and so is a parent block declared after the levels
  it groups, which they already span.
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
  can consume the result directly. Transactions come back as the
  student-step table they roll up to.
- **A compatibility switch.** `build_afm_design(data, learnsphere_compat=True)`
  reproduces LearnSphere's exact conventions (ridge, parameter counting) when
  you need to match a published table; the default is the statistically clean
  variant. 

## Recipes

A few statistics take a line or two of code rather than a method of their own.
Each needs only what every fit carries — `predict_proba`, `weights`, `design`,
`ll_unpenalized`, `n_params` and `n_obs` — and the snippets continue the
quickstart's `data`, `design` and `fit`:

```python
import numpy as np
import pandas as pd
from scipy.special import expit

p = fit.predict_proba(design)        # P(correct) per row, from any Design with the fit's columns
brier = np.mean((data.y - p) ** 2)   # Brier score: mean squared error on the probability scale
rmse = np.sqrt(brier)

# AIC and BIC without the ridge. They differ from fit.aic and fit.bic only
# when a block is penalized, as under learnsphere_compat=True.
aic_unpenalized = -2 * fit.ll_unpenalized + 2 * fit.n_params
bic_unpenalized = -2 * fit.ll_unpenalized + fit.n_params * np.log(fit.n_obs)

# Every coefficient by block and column, and one block's rows of it.
coefficients = pd.DataFrame({
    "block": [b.name for b in fit.design.blocks for _ in b.columns],
    "column": [c for b in fit.design.blocks for c in b.columns],
    "estimate": fit.weights,
})
slopes = coefficients[coefficients["block"] == "kc_slope"]

# Predictions from a bare sparse matrix with the fit's columns, in order.
X = design.matrix
p = expit(X @ fit.weights)
```

Held-out versions come from cross-validation. A fold's Brier score is its RMSE
squared, and under the pooled convention so is the overall one:

```python
cv = cross_validate(design, data, scheme="item_blocked", convention="pooled")
held_out_brier = cv.rmse ** 2
per_fold_brier = cv.frame["rmse"] ** 2
```

A ridge on PFA's student intercepts is a block you build yourself:

```python
from leapfit import Block, Design, build_pfa_design, fit_pfa

students = Block.from_levels("student", [(s,) for s in data.students], l2=1.0)
ridged = Design((students, *build_pfa_design(data, identify=False).blocks)).identify()
pfa = fit_pfa(ridged, data.y)
```

## Roadmap

| model | state | notes |
|---|---|---|
| **AFM** | shipped | validated for equivalence against LearnSphere workflow output |
| **PFA** | shipped | canonical fixed-effects PFA (Pavlik, Cen & Koedinger 2009) with strictly-prior counts; per-KC or pooled slopes, optional student intercepts. The audited reference builds its counts *including* each attempt's own outcome — that construction is reproducible here via an explicit option that warns, never silently |
| **LFA** | shipped | a search over KC models rather than a model: greedy best-first on AFM's BIC or AIC, reproducing the reference's fit statistics to 1e-9. Every candidate move is screened for estimability and evidence — the reference's own selection contained a KC whose slope has no finite estimate, and that one move then appeared in all 99 states it reported — and the top states are validated out of sample on identical folds |
| **LKT** | shipped | components x features, of which AFM and PFA are single specifications — both identities are pinned by tests. Thirty-one features over any component the export carries: prior counts, decayed outcome histories, recency, power-law forgetting, and the two four-parameter spacing models (`base4`, `ppe`), each parametric one held at the values you pass or fitted. Validated against the reference's own published output — the CRAN package's precompiled vignette prints the log-likelihood of each model it fits — on **four** of its chunks, spanning the whole implemented surface: **-27347.207** (AFM), **-25474.531** (logitdec + recency), **-24695.586** (PPE) and **-25969.925** (base4), each reproduced within 0.5 nats and each on the better side, from a certified optimum, because the reference's `LiblineaR` stops at `epsilon = 1e-4`. `fit_lkt_pars` fits the decay rates themselves by profile likelihood, counting them in `n_params` and certifying stationarity by measurement; it walks the reference's own published RPFA search to the same optimum. Written clean-room: the reference is GPL-3, so this module is validated against it rather than adapted from it |
| LKT, a search over terms | planned | the reference's `buildLKTModel` is greedy forward/backward selection over features x components on BIC. That is a *consumer* of the family, the shape `leapfit.lfa` already has, not more of `leapfit.lkt` |
| BKT | planned | to be validated against the standard `standard-bkt` C++ tool |

## Development

CI runs lint and the full suite on Python 3.11–3.13, then builds the wheel,
installs it into a clean environment, and fits a model from outside the source
tree. The equivalence tests require LearnSphere run artifacts and skip without
them, so a bare clone is always green:

```bash
uv run pytest                                       # 315 pass, 44 skip, ~16s
AFM_WF3990_DIR=/path/to/artifacts uv run pytest     # + 8 AFM equivalence tests
LFA_BUNDLE_DIR=/path/to/lfa-reference-run uv run pytest   # + 18 LFA equivalence tests
LKT_VIGNETTE_DIR=/path/to/converted uv run pytest   # + 15 LKT equivalence tests
```

The LKT fixture is built from the CRAN tarball rather than shipped, because the
reference package is GPL-3 and this one is MIT —
`tests/test_lkt_equivalence.py` converts it when run as a script, and its
docstring gives the two commands.

With `Rscript` on `PATH`, three more tests fit the same designs through R's
`stats::glm` and require agreement to numerical precision (~1e-8 in
log-likelihood) — any R works, e.g.
`micromamba create -p /tmp/r-env -c conda-forge r-base`.

## License

[MIT](LICENSE)

One provenance note, because the distinction matters. AFM, PFA and LFA are
*adapted from* LearnSphere's reference components. `leapfit.lkt` is not: its
reference, the CRAN package [LKT](https://CRAN.R-project.org/package=LKT), is
GPL-3, so that module is written from the published equations and the
reference's documented feature semantics, and then **validated against** its
output. No LKT source is translated here.
