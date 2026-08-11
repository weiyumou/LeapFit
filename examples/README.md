# Example data

`student-step.txt` is a **synthetic** student-step file — 12 simulated students,
480 rows, no real learner anywhere in it. It exists so the README's commands run
out of the box and so the expected input format has a concrete instance you can
open.

It carries two KC models over the same steps:

| KC model | granularity | truth |
|---|---|---|
| `Topics` | 4 KCs, 10 steps each | **the generating model** — responses were simulated from an AFM on these KCs, all learning rates positive |
| `Skills` | 12 KCs, 3–4 steps each | a finer relabelling of the same steps |

Because `Topics` is the true model, a correct fitter should prefer it — and it
does: AIC 576.0 vs 605.6. Recovering a planted answer is the closest thing to a
ground-truth check a model-comparison tool can offer.

Regenerate (deterministic, seeded) with:

```bash
python examples/generate.py
```

`generate.py` is also the minimal reference for producing leapfit-compatible
files from your own data: tab-separated, the required columns plus a
`KC (…)`/`Opportunity (…)` pair per KC model, opportunities numbered from 1,
and an optional `First Transaction Time`.
