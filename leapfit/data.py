"""Reading DataShop exports into AFM observations.

Every model reads a student-step rollup. A transaction export is rolled up into
one first by :func:`rollup_transactions`, under the rules DataShop's own
student-step export follows; that function records them, and how closely they
reproduce DataShop's export of the same data.

The parsing rules in this module are not our own design: they replicate
LearnSphere's PyAFM component
(``AnalysisPyAfm/program/process_datashop.py::plot_datashop_student_step``)
so that anything we fit stays comparable with numbers DataShop already
reports. Where the reference is ambiguous or silently lossy we raise instead
of guessing, and every such divergence is marked ``DIVERGENCE`` below.

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

Two further columns are read when the export carries them, and only because
models beyond AFM need them: ``First Transaction Time`` orders
:meth:`StepData.practice_order` and, parsed by :meth:`StepData.epoch_times`,
supplies the intervals a spacing or forgetting term measures; ``Step Duration
(sec)`` accumulates into :meth:`StepData.time_on_task`. Neither is required,
and a model that needs one and does not have it is refused rather than
defaulted — every substitute value is a different model.

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
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain

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

#: The columns :func:`rollup_transactions` requires, all of which a DataShop
#: transaction export carries. ``Attempt At Step`` and ``Problem View`` are
#: DataShop's own counts, made when the data was imported.
TRANSACTION_COLUMNS = ("Anon Student Id", "Problem Name", "Problem View", "Step Name",
                       "Attempt At Step", "Outcome", "Time")

#: A transaction export's KC column. A transaction with several KCs in one model
#: repeats the column once per KC, and pandas reads the repeats as
#: ``KC (<model>).1``, ``KC (<model>).2``, and so on.
TX_KC_COLUMN = re.compile(r"^KC \((?P<name>.+)\)(?:\.\d+)?$")


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
    durations: np.ndarray | None = None  # (n_obs,) Step Duration (sec), if present
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
        (PFA's prior-success/failure counts, spacing gaps, similarity-weighted
        practice) must agree on what "before" means, or a coefficient on one is
        measured against a different history than the other.

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

    def epoch_times(self) -> np.ndarray:
        """``First Transaction Time`` as seconds, for models that need a clock.

        :meth:`practice_order` only ever needs to *order* the timestamps, so
        they are kept as the strings the export wrote and parsed here on
        demand. A model with a forgetting or spacing term needs the intervals
        themselves, and an export without the column cannot supply them —
        which is a refusal, not a default, because every value that could be
        substituted (row number, a constant) is a different model.

        :raises ValueError: when the export carried no time column, or when a
            value in it does not parse, naming the offenders.
        """
        if self.times is None:
            raise ValueError(
                "This export carried no 'First Transaction Time' column, so there "
                "are no times to measure intervals between. A spacing, recency or "
                "forgetting term cannot be computed from it."
            )
        parsed = pd.to_datetime(pd.Series(self.times), errors="coerce", format="mixed")
        if (bad := parsed.isna()).any():
            examples = ", ".join(map(repr, pd.Series(self.times)[bad].unique()[:3]))
            raise ValueError(
                f"{int(bad.sum()):,} 'First Transaction Time' value(s) do not parse as "
                f"a timestamp, for example {examples}."
            )
        # Explicitly to seconds: the parsed dtype's own unit is a pandas
        # version detail (nanoseconds once, microseconds now), and dividing an
        # int64 view by a hardcoded 10**9 silently scales every interval.
        return parsed.to_numpy(dtype="datetime64[s]").astype(np.int64)

    def time_on_task(self) -> np.ndarray:
        """Seconds a student had spent working *before* each observation.

        The cumulative ``Step Duration (sec)`` over :meth:`practice_order`,
        lagged so the first step of a student enters at zero. It is the clock
        that ignores the gaps between sessions, and the difference between it
        and :meth:`epoch_times` is what a model has to work with if it wants
        to treat time away from the system differently from time at it.

        :raises ValueError: when the export carried no duration column.
        """
        if self.durations is None:
            raise ValueError(
                "This export carried no 'Step Duration (sec)' column, so time on "
                "task cannot be accumulated."
            )
        if (missing := int(np.isnan(self.durations).sum())):
            # DataShop writes "." where it could not compute one. Accumulating
            # past it would make every later step of that student NaN, and the
            # model that reads this would then fail complaining about timestamps.
            raise ValueError(
                f"{missing:,} of {len(self):,} rows have no 'Step Duration (sec)', so "
                "time on task cannot be accumulated past them. Drop those rows or use "
                "a feature that reads the wall clock instead."
            )
        out = np.zeros(len(self), dtype=float)
        for rows in self.practice_order().values():
            spent = np.asarray(self.durations[rows], dtype=float)
            out[rows] = np.concatenate([[0.0], np.cumsum(spent)[:-1]])
        return out

    def prior_counts(self, labels: Sequence[tuple[str, ...]] | None = None,
                     ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Successes and failures before each observation, per label.

        A count is the student's own history with that label and nothing
        else: strictly prior, so the current attempt is not inside its own
        predictor, and accumulated over :meth:`practice_order`, the one
        ordering everything in this package agrees on. Shaped like
        :attr:`opportunities`: element ``n`` holds one count per label of
        ``labels[n]``, aligned by position.

        :param labels: what to count, by default :attr:`kcs`. Any labelling of
            the observations works; a student's own id gives that student's
            whole prior history.
        """
        labels = self.kcs if labels is None else labels
        if len(labels) != len(self):
            raise ValueError(f"{len(self)} observations but {len(labels)} label rows")
        sequences = _sequences(self, labels)
        return (sequences.per_row(sequences.prior_s, labels),
                sequences.per_row(sequences.prior_f, labels))

    def recomputed_opportunities(self) -> list[tuple[int, ...]]:
        """Opportunity counts derived from :meth:`practice_order`: the prior
        successes plus the prior failures on each KC.

        DataShop ships its own ``Opportunity`` columns and we use them by
        default, but ``AnalysisFastAfmAndCv`` ignores them and recomputes
        exactly this way. Use :meth:`opportunity_disagreements` to see whether
        the two differ on your export before it matters.
        """
        sequences = _sequences(self, self.kcs)
        return sequences.per_row(sequences.prior_s + sequences.prior_f, self.kcs)

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


@dataclass(frozen=True)
class _Sequences:
    """Every (observation, label) pair, grouped into practice sequences.

    The pairs are stored flat and ordered so that each (student, label) history
    is one contiguous run, ``starts[k]:starts[k+1]``, in practice order, and the
    runs come in the order each student first met each label. Prior counts are
    an exclusive cumulative sum that restarts at every run, and the same shape
    is what :mod:`leapfit.lkt` builds its features over: a count feature reads
    :attr:`prior_s` and :attr:`prior_f` straight across, and a decay or
    forgetting feature walks one run at a time.

    ``slots`` records which of an observation's own labels a pair came from, so
    :meth:`per_row` can put values back in the row's label order, the order
    everything outside this module aligns to.
    """

    n_obs: int
    rows: np.ndarray          # (m,) observation index
    slots: np.ndarray         # (m,) label position within that observation
    columns: np.ndarray       # (m,) index into `levels`
    starts: np.ndarray        # (n_sequences + 1,)
    levels: list[str]         # every label, sorted
    y: np.ndarray             # (m,) outcome at each pair
    prior_s: np.ndarray       # (m,) successes earlier in its own run
    prior_f: np.ndarray       # (m,) failures earlier in its own run

    def per_row(self, values: np.ndarray,
                labels: Sequence[tuple[str, ...]]) -> list[tuple[int, ...]]:
        """Integer ``values``, one per pair, back in the shape of ``labels``."""
        lengths = np.fromiter(map(len, labels), dtype=np.intp, count=len(labels))
        ends = np.cumsum(lengths)
        flat = np.zeros(int(ends[-1]) if len(ends) else 0, dtype=np.int64)
        flat[ends[self.rows] - lengths[self.rows] + self.slots] = values
        out = flat.tolist()
        return [tuple(out[end - n:end]) for end, n in zip(ends.tolist(), lengths.tolist())]


def _sequences(data: StepData, labels: Sequence[tuple[str, ...]]) -> _Sequences:
    """Group every (observation, label) pair by (student, label), in practice order."""
    flat = list(chain.from_iterable(labels))
    levels = sorted(set(flat))
    row_lengths = np.fromiter(map(len, labels), dtype=np.intp, count=len(labels))
    row_starts = np.cumsum(row_lengths) - row_lengths
    # Each pair's level, in the row-major order `flat` lists them.
    level_of = pd.Index(levels).get_indexer(flat).astype(int)

    order = data.practice_order()  # students in order of first appearance
    practised = (np.concatenate(list(order.values())) if order
                 else np.zeros(0, dtype=int))
    student = np.repeat(np.arange(len(order)), [len(v) for v in order.values()])
    lengths = row_lengths[practised]
    rows = np.repeat(practised, lengths)
    slots = np.arange(len(rows)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    columns = level_of[row_starts[rows] + slots]

    # A run per (student, label), numbered in the order the student first met
    # the label; a stable sort by that number keeps each run in practice order.
    run = pd.factorize(np.repeat(student, lengths) * max(len(levels), 1) + columns)[0]
    by_run = np.argsort(run, kind="stable")
    rows, slots, columns = rows[by_run], slots[by_run], columns[by_run]
    starts = np.concatenate([[0], np.cumsum(np.bincount(run), dtype=int)]).astype(int)

    y = np.asarray(data.y, dtype=float)[rows]
    totals = np.concatenate([[0.0], np.cumsum(y)])
    at_start = np.repeat(starts[:-1], np.diff(starts))
    prior_s = totals[:-1] - totals[at_start]
    prior_f = (np.arange(len(y)) - at_start) - prior_s

    return _Sequences(n_obs=len(data), rows=rows, slots=slots, columns=columns,
                      starts=starts, levels=levels, y=y,
                      prior_s=prior_s, prior_f=prior_f)


def list_kc_models(path: str) -> list[str]:
    """Return the KC model names available in a student-step or transaction export."""
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
               failure_values: tuple[str, ...] = tuple(sorted(FIRST_ATTEMPT_VALUES - {CORRECT})),
               ) -> StepData:
    """Build :class:`StepData` from an already-loaded student-step table.

    :param success_values: ``First Attempt`` values that count as a success,
        matched after folding case and stripping whitespace.
    :param failure_values: values that count as a failure, by default DataShop's
        other three. Anything in neither list raises rather than being silently
        scored as a failure, so a column that is not the one we think it is
        fails loudly. A file with its own vocabulary needs both lists —
        ``success_values=("1",), failure_values=("0",)`` — which keeps the guard
        meaningful instead of disabling it whenever the default is overridden.
        A value in both lists is a success.
    """
    kc_col, opp_col = f"KC ({kc_model})", f"Opportunity ({kc_model})"
    required = ["Anon Student Id", "Problem Name", "Step Name", "First Attempt",
                kc_col, opp_col]
    time_col = "First Transaction Time" if "First Transaction Time" in df.columns else None
    duration_col = "Step Duration (sec)" if "Step Duration (sec)" in df.columns else None
    if missing := [c for c in required if c not in df.columns]:
        available = sorted(m.group("name") for c in df.columns if (m := KC_COLUMN.match(c)))
        message = f"Missing column(s) {missing}. KC models present: {available or 'none'}"
        if "Attempt At Step" in df.columns and "First Attempt" not in df.columns:
            message += (". This looks like a transaction export: roll it up with "
                        "rollup_transactions(), or read it with load_transactions()")
        raise KeyError(message)

    cols = {c: df[c].astype(str).to_list() for c in required}
    time_values = df[time_col].astype(str).to_list() if time_col else None
    # Optional, and only some models read it. DataShop writes "." for a step
    # whose duration it could not compute, so this is deliberately permissive
    # here and strict in StepData.time_on_task, where a gap actually matters.
    durations = (pd.to_numeric(df[duration_col], errors="coerce").to_numpy(dtype=float)
                 if duration_col else None)

    successes = {v.strip().lower() for v in success_values}
    # A value declared both ways is scored as a success, so it is not a failure.
    failures = {v.strip().lower() for v in failure_values} - successes
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

    source_rows = np.asarray(src, dtype=int)
    return StepData(
        y=np.asarray(y, dtype=np.int8),
        students=students, items=items, kcs=kcs, opportunities=opps,
        kc_model=kc_model, times=(times if time_col else None),
        durations=(durations[source_rows] if durations is not None else None),
        skipped_no_kc=skipped, duplicate_kc_rows=duplicates,
        source=df, source_rows=source_rows,
    )


def load_transactions(path: str, kc_model: str, **kwargs) -> StepData:
    """Load one KC model out of a DataShop transaction export.

    The export is rolled up by :func:`rollup_transactions`, and the resulting
    student-step table is what :attr:`StepData.source` holds, so
    ``annotate`` writes its predictions into that. To read several KC models,
    roll the export up once and pass the table to :func:`from_frame` for each.
    ``kwargs`` go to :func:`from_frame`.
    """
    tx = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return from_frame(rollup_transactions(tx), kc_model, **kwargs)


def rollup_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Roll a DataShop transaction export up into its student-step table.

    One row per encounter of a student with a step, keyed as DataShop keys one
    (student, the ``Level (...)`` hierarchy, problem, problem view, step), with
    the columns :func:`from_frame` reads. Read the export with ``dtype=str,
    keep_default_na=False``, as :func:`load_transactions` does. The rules,
    checked against DataShop's own student-step export of the same data:

    1. **The first attempt is the transaction DataShop numbered
       ``Attempt At Step`` 1.** Later attempts, and anything without an attempt
       number (untutored actions, page views, saves), count only toward the
       step's duration. pyAFM's ``roll_up.py`` takes a step's first transaction
       of any kind instead, which on ds5426 turns 2,666 of those events into
       first attempts with no outcome.
    2. **Encounters are the export's own problem views.** pyAFM re-derives them
       from changes of problem name, and so merges a problem the student opens
       again straight away into the visit before it (6,881 views on ds5426,
       where DataShop has 9,452).
    3. **``First Attempt`` is that transaction's ``Outcome``,** lowercased, and
       ``unknown`` where DataShop left it blank. ``First Transaction Time`` is
       its ``Time``.
    4. **A step's KCs in a model are every label in the model's ``KC (<model>)``
       columns.** The export repeats the column once per KC. Each label is kept
       once, in column order, and a cell that is already ``~~``-joined is split
       as well. They are written ``~~``-joined, as the student-step export
       writes them, with the opportunity counts in the same order.
    5. **``Opportunity (<model>)`` numbers each student's encounters with a KC
       from 1, in practice order:** ``First Transaction Time``, then ``Problem
       Start Time`` (within one second DataShop counts the earlier problem view
       first), then the export's own order. The rows are returned in that
       order, so :meth:`StepData.practice_order` agrees with it and
       ``recompute_opportunities=True`` changes nothing.
    6. **``Step Duration (sec)``, where the export has ``Duration (sec)``, is the
       sum of the encounter's transaction durations,** as DataShop documents
       it, and ``.`` when the first attempt's own is undefined, as its export
       has it. A later undefined duration is left out of the sum.

    Checked against DataShop's student-step exports of two datasets. On ds6160
    (89,110 steps) every row, outcome, time, KC, opportunity and duration
    agrees, and so does AFM's fit, to every digit printed. On ds5426 (53,050
    steps) every row, outcome, time and KC does (one model's labels were
    renamed between the two exports, one for one), and every duration but
    one. About a quarter of its opportunity counts do not: 76% of its steps
    share their student's second with another step, and DataShop orders
    those by something neither of its exports records. Recounting DataShop's
    own student-step export from its times does not reproduce them either.

    :raises KeyError: when a column of :data:`TRANSACTION_COLUMNS` is missing.
    :raises ValueError: when the export holds more than one sample, when two
        transactions are both the first attempt of one encounter, or when none
        is a first attempt.
    """
    names = [str(c) for c in tx.columns]
    if missing := [c for c in TRANSACTION_COLUMNS if c not in names]:
        raise KeyError(f"Missing transaction column(s) {missing}")

    def column(i: int) -> pd.Series:
        # By position, since an export repeats some headers.
        return tx.iloc[:, i].fillna("").astype(str).reset_index(drop=True)

    def named(name: str) -> pd.Series:
        return column(names.index(name))

    if "Sample Name" in names and len(samples := sorted(set(named("Sample Name")) - {""})) > 1:
        raise ValueError(
            f"The export holds {len(samples)} samples ({', '.join(map(repr, samples))}), and a "
            "transaction is listed once in each sample that contains it, so rolling them up "
            "together would count it more than once. Keep one sample first, for example "
            f"tx[tx['Sample Name'] == {samples[0]!r}]."
        )

    levels = [i for i, name in enumerate(names) if name.startswith("Level (")]
    encounter = pd.DataFrame({
        "student": named("Anon Student Id"),
        **{f"level {i}": column(i) for i in levels},
        "problem": named("Problem Name"),
        "view": named("Problem View"),
        "step": named("Step Name"),
    })
    first = np.flatnonzero(pd.to_numeric(named("Attempt At Step"), errors="coerce") == 1)
    if not len(first):
        raise ValueError("No transaction has 'Attempt At Step' 1, so there is no first "
                         "attempt to roll up.")
    steps = encounter.iloc[first].reset_index(drop=True)
    if (twice := steps.duplicated(keep=False)).any():
        example = steps[twice].iloc[0]
        raise ValueError(
            f"{int(twice.sum()):,} transactions are first attempts that share their encounter "
            f"with another first attempt, for example student {example['student']!r} on step "
            f"{example['step']!r} of problem {example['problem']!r}, view {example['view']}. "
            "DataShop numbers one first attempt per step per problem view, so this export "
            "was assembled from more than one, or edited."
        )

    out = steps.rename(columns={"student": "Anon Student Id", "problem": "Problem Name",
                                "view": "Problem View", "step": "Step Name",
                                **{f"level {i}": names[i] for i in levels}})
    out["First Transaction Time"] = named("Time").iloc[first].to_numpy()
    out["First Attempt"] = (named("Outcome").iloc[first].str.strip().str.lower()
                            .replace("", "unknown").to_numpy())
    if "Duration (sec)" in names:
        seconds = pd.to_numeric(named("Duration (sec)"), errors="coerce")
        total = seconds.groupby([encounter[c] for c in encounter.columns],
                                sort=False).transform("sum")
        out["Step Duration (sec)"] = [
            "." if np.isnan(own) else np.format_float_positional(round(s, 3), trim="-")
            for s, own in zip(total.iloc[first], seconds.iloc[first])]

    started = (named("Problem Start Time").iloc[first].to_numpy() if "Problem Start Time" in names
               else np.full(len(first), ""))
    order = pd.DataFrame({"student": out["Anon Student Id"], "time": out["First Transaction Time"],
                          "started": started, "row": np.arange(len(first))}
                         ).sort_values(["student", "time", "started", "row"]).index.to_numpy()

    models: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        if m := TX_KC_COLUMN.match(name):
            models.setdefault(m.group("name"), []).append(i)
    students = out["Anon Student Id"].to_numpy()
    for model, positions in models.items():
        cells = zip(*(column(i).iloc[first] for i in positions))
        labels = [tuple(dict.fromkeys(kc for cell in row for kc in cell.split(MULTI_SEP) if kc))
                  for row in cells]
        seen: dict[tuple[str, str], int] = {}  # encounters so far, per (student, KC)
        counts = [""] * len(labels)
        for r in order:
            numbers = []
            for kc in labels[r]:
                n = seen[students[r], kc] = seen.get((students[r], kc), 0) + 1
                numbers.append(str(n))
            counts[r] = MULTI_SEP.join(numbers)
        out[f"KC ({model})"] = [MULTI_SEP.join(row) for row in labels]
        out[f"Opportunity ({model})"] = counts

    return out.iloc[order].reset_index(drop=True)
