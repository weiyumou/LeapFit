"""Reading DataShop student-step rollups into AFM observations.

The parsing rules in this module are not our own design: they replicate
LearnSphere's PyAFM component
(``AnalysisPyAfm/program/process_datashop.py::plot_datashop_student_step``)
so that anything we fit stays comparable with the numbers DataShop and the
EDM 2025 paper already report. Where the reference is ambiguous or silently
lossy we raise instead of guessing, and every such divergence is marked
``DIVERGENCE`` below.

The rules, in the reference's own order:

1. One observation per row of the student-step export.
2. ``KC (<model>)`` holds the step's knowledge components joined by ``~~``;
   empty strings are dropped. **A row with no KC is skipped entirely** — it
   contributes to neither the fit nor the observation count that BIC uses.
3. ``Opportunity (<model>)`` holds the matching opportunity counts, also
   ``~~``-joined, aligned to the KC list *by position*. DataShop numbers
   opportunities from 1, and the reference subtracts 1, so a student's first
   encounter with a KC enters the model with ``T = 0``. This is what makes
   the intercept interpretable as "log-odds at opportunity 1".
4. ``First Attempt == "correct"`` is a success; every other value (including
   ``hint`` and ``incorrect``) is a failure.
5. The item label — the unit of item-blocked cross-validation — is
   ``Problem Name ## Step Name``.

DIVERGENCE (unrecognized outcome labels): the reference compares ``First
Attempt`` to the literal ``"correct"``, so any export writing ``"Correct"``, or
any non-DataShop file coding the outcome as ``1``/``0``, silently yields a
response vector of all zeros — a fit that converges, reports a plausible AIC,
and means nothing. We fold case and whitespace first, then require every
surviving value to be either a declared success or one of DataShop's documented
failure labels, and raise naming the offenders. Pass ``success_values`` to read
a different vocabulary deliberately.

DIVERGENCE (length mismatch): the reference indexes ``kc_opps[i]`` after
filtering both lists independently, so a row whose KC and opportunity fields
disagree in length either raises IndexError or silently misaligns skills with
counts. We check the lengths and raise a clear error naming the row.

DIVERGENCE (duplicate KC on one step): the reference builds ``{kc: opp}``
dicts, so a KC listed twice on one step keeps only its last opportunity and
counts once in the Q-matrix. We reproduce that (it is the behaviour the
published fits were produced under) but count the occurrences so callers can
see whether it happened.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KC_COLUMN = re.compile(r"^KC \((?P<name>.+)\)$")

CORRECT = "correct"
MULTI_SEP = "~~"
ITEM_SEP = "##"

#: The values DataShop documents for ``First Attempt``. Everything that is not
#: a declared success counts as a failure, but a value outside this vocabulary
#: means the column is not what we think it is — see the module docstring.
FIRST_ATTEMPT_VALUES = frozenset({"correct", "incorrect", "hint", "unknown"})


@dataclass(frozen=True)
class StepData:
    """One AFM-ready observation per element: a student-step first attempt.

    ``kcs[n]`` are the knowledge components of observation ``n`` and
    ``opportunities[n]`` the matching prior-practice counts, already
    zero-based. The two lists are the same length by construction.
    """

    y: np.ndarray                       # (n_obs,) int8 in {0, 1}
    students: list[str]                 # (n_obs,)
    items: list[str]                    # (n_obs,)
    kcs: list[tuple[str, ...]]          # (n_obs,) KC labels per observation
    opportunities: list[tuple[int, ...]]  # (n_obs,) zero-based counts
    kc_model: str
    times: list[str] | None = None      # (n_obs,) First Transaction Time, if present
    skipped_no_kc: int = 0
    duplicate_kc_rows: int = 0

    #: The table these observations were parsed from, unmodified, and the
    #: *position* in it of each observation. Rows with no KC exist in ``source``
    #: but not here, so ``source_rows`` is what lets predictions be written back
    #: into the file against the right rows — the alignment LearnSphere's
    #: components attempt by re-reading and re-sorting the input, which is
    #: exactly where its step-based PFA component breaks.
    source: pd.DataFrame | None = field(default=None, repr=False)
    source_rows: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def student_names(self) -> list[str]:
        return sorted(set(self.students))

    @property
    def kc_names(self) -> list[str]:
        return sorted({kc for row in self.kcs for kc in row})

    def practice_order(self) -> dict[str, np.ndarray]:
        """Each student's row indices, in the order they practised.

        **One canonical ordering, used by everything.** The opportunity counts
        in the AFM design and any accumulator built over a student's history
        (congruity-weighted practice, spacing gaps) must agree on what "before"
        means, or a coefficient on one is measured against a different history
        than the other.

        Ordered by ``First Transaction Time`` where the export provides it,
        with the file's own row order breaking ties — the same tie-break
        DataShop's ``Opportunity`` columns use. Falls back to pure row order
        when no time column is present.
        """
        order: dict[str, list[int]] = {}
        for i, s in enumerate(self.students):
            order.setdefault(s, []).append(i)
        if self.times is None:
            return {s: np.asarray(v, dtype=int) for s, v in order.items()}
        return {
            s: np.asarray(sorted(v, key=lambda i: (self.times[i], i)), dtype=int)
            for s, v in order.items()
        }

    def recomputed_opportunities(self) -> list[tuple[int, ...]]:
        """Opportunity counts derived from :meth:`practice_order`.

        DataShop ships its own ``Opportunity`` columns and we use them by
        default, but ``AnalysisFastAfmAndCv`` ignores them and recomputes
        exactly this way. Use :meth:`opportunity_disagreements` to see whether
        the two differ on your export before it matters.
        """
        out: list[list[int]] = [[] for _ in range(len(self))]
        for rows in self.practice_order().values():
            seen: dict[str, int] = {}
            for i in rows:
                counts = []
                for kc in self.kcs[i]:
                    counts.append(seen.get(kc, 0))
                    seen[kc] = seen.get(kc, 0) + 1
                out[i] = counts
        return [tuple(v) for v in out]

    def opportunity_disagreements(self) -> np.ndarray:
        """Row indices where the file's counts differ from the recomputed ones."""
        mine = self.recomputed_opportunities()
        return np.array([i for i, (a, b) in enumerate(zip(self.opportunities, mine))
                         if tuple(a) != tuple(b)], dtype=int)

    def summary(self) -> str:
        return (
            f"{len(self):,} observations | {len(self.student_names)} students | "
            f"{len(self.kc_names):,} KCs | {len(set(self.items)):,} items | "
            f"{self.y.mean():.2%} correct | model '{self.kc_model}'"
            + (f" | {self.skipped_no_kc:,} rows skipped (no KC)" if self.skipped_no_kc else "")
        )


def list_kc_models(path: str) -> list[str]:
    """Return the KC model names available in a student-step export."""
    header = pd.read_csv(path, sep="\t", nrows=0)
    return sorted(
        m.group("name") for col in header.columns if (m := KC_COLUMN.match(col))
    )


def load_student_step(path: str, kc_model: str, **kwargs) -> StepData:
    """Load one KC model out of a DataShop student-step export."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return from_frame(df, kc_model, **kwargs)


def from_frame(df: pd.DataFrame, kc_model: str, *,
               success_values: tuple[str, ...] = (CORRECT,),
               failure_values: tuple[str, ...] = tuple(FIRST_ATTEMPT_VALUES),
               ) -> StepData:
    """Build :class:`StepData` from an already-loaded student-step table.

    :param success_values: ``First Attempt`` values that count as a success,
        matched after folding case and stripping whitespace.
    :param failure_values: values that count as a failure. Anything in neither
        list raises rather than being silently scored as a failure, so a column
        that is not the one we think it is fails loudly. A file with its own
        vocabulary needs both lists — ``success_values=("1",),
        failure_values=("0",)`` — which keeps the guard meaningful instead of
        disabling it whenever the default is overridden.
    """
    kc_col, opp_col = f"KC ({kc_model})", f"Opportunity ({kc_model})"
    required = ["Anon Student Id", "Problem Name", "Step Name", "First Attempt",
                kc_col, opp_col]
    time_col = "First Transaction Time" if "First Transaction Time" in df.columns else None
    if missing := [c for c in required if c not in df.columns]:
        available = sorted(m.group("name") for c in df.columns if (m := KC_COLUMN.match(c)))
        raise KeyError(
            f"Missing column(s) {missing}. KC models present: {available or 'none'}"
        )

    cols = {c: df[c].astype(str).to_list() for c in required}
    time_values = df[time_col].astype(str).to_list() if time_col else None

    successes = {v.strip().lower() for v in success_values}
    failures = {v.strip().lower() for v in failure_values}
    y, students, items, kcs, opps, times, src = [], [], [], [], [], [], []
    outcomes: dict[str, int] = {}
    skipped = duplicates = 0

    n_rows = len(cols[kc_col])
    rows = zip(cols[kc_col], cols[opp_col], cols["First Attempt"],
               cols["Anon Student Id"], cols["Problem Name"], cols["Step Name"],
               time_values if time_values is not None else [None] * n_rows)
    for row_no, (kc_cell, opp_cell, attempt, student, problem, step, when) in enumerate(rows, start=2):
        labels = [k for k in kc_cell.split(MULTI_SEP) if k]
        if not labels:
            skipped += 1
            continue

        counts = [o for o in opp_cell.split(MULTI_SEP) if o]
        if len(counts) != len(labels):
            raise ValueError(
                f"Row {row_no}: {len(labels)} KC label(s) but {len(counts)} opportunity "
                f"value(s) for model '{kc_model}'. KCs={labels!r} opportunities={counts!r}"
            )

        try:
            numbers = [int(c) - 1 for c in counts]
        except ValueError as exc:
            raise ValueError(
                f"Row {row_no}: non-integer opportunity value in "
                f"'Opportunity ({kc_model})' = {opp_cell!r}. Opportunity counts must "
                f"be whole numbers, one per KC in {labels!r}."
            ) from exc

        # Positional zip, last-wins on a repeated KC — the reference's dict
        # comprehension semantics, preserved deliberately (see module docstring).
        paired = dict(zip(labels, numbers))
        if len(paired) != len(labels):
            duplicates += 1

        outcome = attempt.strip().lower()
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

        kcs.append(tuple(paired))
        opps.append(tuple(paired.values()))
        y.append(1 if outcome in successes else 0)
        students.append(student)
        items.append(f"{problem}{ITEM_SEP}{step}")
        times.append(when)
        src.append(row_no - 2)  # enumerate starts at 2; this is the 0-based position

    if not y:
        raise ValueError(f"No observations carry a KC under model '{kc_model}'")
    if unexpected := {v: n for v, n in outcomes.items()
                      if v not in successes and v not in failures}:
        listed = ", ".join(f"{v!r} ({n:,} rows)" for v, n in sorted(unexpected.items()))
        raise ValueError(
            f"Unrecognized 'First Attempt' value(s): {listed}. Successes are "
            f"{sorted(successes)} and failures are {sorted(failures)}; anything "
            "else would be scored as a failure without warning. Pass "
            "success_values=(...) and failure_values=(...) if this file uses a "
            "different vocabulary."
        )
    if len(set(y)) == 1:
        # Not an error — a hard unit really can be all-incorrect — but it is
        # also the signature of a success vocabulary that does not match the
        # file, so name the vocabulary rather than leaving it to be diagnosed
        # from a degenerate fit.
        warnings.warn(
            f"Every observation is a {'success' if y[0] else 'failure'}. "
            f"'First Attempt' holds {sorted(outcomes)} and successes are "
            f"{sorted(successes)}; check that pairing before reading the fit, "
            "because a constant response makes every coefficient unbounded.",
            RuntimeWarning, stacklevel=2,
        )
    if any(t < 0 for row in opps for t in row):
        raise ValueError(
            "Negative opportunity count after the reference's -1 adjustment; "
            "DataShop numbers opportunities from 1, so the export looks malformed."
        )

    return StepData(
        y=np.asarray(y, dtype=np.int8),
        students=students, items=items, kcs=kcs, opportunities=opps,
        kc_model=kc_model, times=(times if time_col else None),
        skipped_no_kc=skipped, duplicate_kc_rows=duplicates,
        source=df, source_rows=np.asarray(src, dtype=int),
    )
