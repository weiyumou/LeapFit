"""Builders shared by the test modules.

Most make a small student-step table, or the ``StepData`` parsed from one, so
that a test can spell out the rows it needs. A row is a dict in DataShop's
column names, except that ``kc`` and ``opp`` stand for the KC model's two
columns, which :func:`rollup` renames once the model is known. The synthetic
generators draw from a known AFM or PFA, for the tests that recover one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leapfit import from_frame

#: The example export that ships with the repository.
EXAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "student-step.txt")

#: Where :func:`clocked_data` starts its clock.
EPOCH = pd.Timestamp("2024-01-01 00:00:00")

#: Every column the reader requires, and the only ones :func:`minimal_frame` writes.
MINIMAL_COLUMNS = ["Anon Student Id", "Problem Name", "Step Name",
                   "First Attempt", "KC (M)", "Opportunity (M)"]


# --------------------------------------------------------------------------
# Rows and tables
# --------------------------------------------------------------------------


def rollup(rows, kc_model="M"):
    """A student-step table, with each row's ``kc`` and ``opp`` under ``kc_model``."""
    return pd.DataFrame(rows).rename(
        columns={"kc": f"KC ({kc_model})", "opp": f"Opportunity ({kc_model})"})


def step_data(rows, kc_model="M"):
    """:func:`rollup`, parsed under the KC model it names."""
    return from_frame(rollup(rows, kc_model), kc_model)


def step_row(student, step, y, kc, opp, time=None, **extra):
    """One row, correct when ``y`` is truthy; ``extra`` adds columns."""
    out = {
        "Anon Student Id": student, "Problem Name": "p", "Step Name": step,
        "First Attempt": "correct" if y else "incorrect", "kc": kc, "opp": str(opp),
    }
    if time is not None:
        out["First Transaction Time"] = time
    return out | extra


def attempts(student, step, kc, outcomes):
    """One student's repeated attempts at one step, in practice order."""
    return [{"Anon Student Id": student, "Problem Name": "p", "Step Name": step,
             "First Attempt": outcome, "kc": kc, "opp": str(i + 1)}
            for i, outcome in enumerate(outcomes)]


def mixed(student, step, kc, n):
    """``n`` attempts alternating correct/incorrect — never separated."""
    return attempts(student, step, kc,
                    ["correct" if i % 2 else "incorrect" for i in range(n)])


def clocked_data(outcomes, gaps, durations=None, student="s1", kc="A"):
    """One student, one level: a practice sequence with a clock on it.

    ``gaps`` are the seconds between consecutive attempts, the first ignored,
    so the sequence starts at :data:`EPOCH`.
    """
    times = np.cumsum(np.asarray(gaps, dtype=float))
    rows = []
    for i, (y, t) in enumerate(zip(outcomes, times)):
        extra = {} if durations is None else {"Step Duration (sec)": durations[i]}
        rows.append(step_row(student, f"st{i}", y, kc, i + 1,
                             time=str(EPOCH + pd.Timedelta(seconds=int(t))), **extra))
    return step_data(rows)


def minimal_frame(n_students=6, n_steps=4, n_reps=3, attempt=str):
    """A rollup carrying *only* the columns the reader declares it needs."""
    rows = []
    for s in range(n_students):
        seen: dict[str, int] = {}
        for _ in range(n_reps):
            for step in range(n_steps):
                kc = f"kc{step % 2}"
                seen[kc] = seen.get(kc, 0) + 1
                rows.append({
                    "Anon Student Id": f"S{s}", "Problem Name": "p",
                    "Step Name": f"st{step}",
                    "First Attempt": attempt("correct" if (s + step) % 3 else "incorrect"),
                    "kc": kc, "opp": str(seen[kc]),
                })
    return rollup(rows)


def separated_frame(always_correct="kc0"):
    """Ten students on four KCs, one of which (``always_correct``) is never failed."""
    rows = []
    rng = np.random.default_rng(3)
    for s in range(10):
        seen: dict[str, int] = {}
        for _ in range(4):
            for step in range(4):
                kc = f"kc{step}"
                seen[kc] = seen.get(kc, 0) + 1
                ok = True if kc == always_correct else bool(rng.random() < 0.6)
                rows.append({
                    "Anon Student Id": f"S{s}", "Problem Name": "p",
                    "Step Name": f"st{step}",
                    "First Attempt": "correct" if ok else "incorrect",
                    "kc": kc, "opp": str(seen[kc]),
                })
    return rollup(rows)


def multi_kc_data():
    """Two KCs per row, but *varying* pairs so the KC columns stay distinct."""
    pairs = [("A", "B"), ("B", "C"), ("A", "C")]
    rows = []
    for i in range(8):
        for j in range(9):
            a, b = pairs[j % 3]
            rows.append({"Anon Student Id": f"s{i}", "Problem Name": "p",
                         "Step Name": f"st{j}",
                         "First Attempt": "correct" if (i + j) % 3 else "incorrect",
                         "kc": f"{a}~~{b}", "opp": f"{j // 3 + 1}~~{j // 3 + 1}"})
    return step_data(rows)


def co_occurring_kc_data(pair_steps=3, solo_steps=3, n_students=6):
    """``A`` and ``B`` tag exactly the same steps; ``C`` tags the rest.

    The duplicate-KC case as it occurs in real labellings: two standards that
    never appear apart. Their intercept columns are identical, and so are their
    opportunity columns, since both are counted over the same rows.
    """
    rows = []
    for i in range(n_students):
        for j in range(pair_steps):
            rows.append({"Anon Student Id": f"s{i}", "Problem Name": "p",
                         "Step Name": f"pair{j}",
                         "First Attempt": "correct" if (i + j) % 3 else "incorrect",
                         "kc": "A~~B", "opp": f"{j + 1}~~{j + 1}"})
        for j in range(solo_steps):
            rows.append({"Anon Student Id": f"s{i}", "Problem Name": "p",
                         "Step Name": f"solo{j}",
                         "First Attempt": "correct" if (i + j) % 2 else "incorrect",
                         "kc": "C", "opp": f"{j + 1}"})
    return step_data(rows)


# --------------------------------------------------------------------------
# Transaction exports
# --------------------------------------------------------------------------


def stamp(seconds):
    """The time ``seconds`` past :data:`EPOCH`, written as DataShop writes it."""
    return str(EPOCH + pd.Timedelta(seconds=seconds))


def tx_row(student, step, outcome, t, attempt=1, kc="A", **extra):
    """One transaction, at ``t`` seconds past :data:`EPOCH`.

    ``outcome`` is written as DataShop writes it (``CORRECT``, ``HINT``, ...),
    ``attempt`` is its ``Attempt At Step`` (``""`` for one that is not an
    attempt), and ``kc`` is its label under KC model ``M``, or ``None`` for no
    such column. ``extra`` adds or replaces columns.
    """
    row = {"Anon Student Id": student, "Problem Name": "p", "Problem View": "1",
           "Step Name": step, "Attempt At Step": str(attempt), "Outcome": outcome,
           "Time": stamp(t)}
    if kc is not None:
        row["KC (M)"] = kc
    return row | extra


def as_transactions(steps):
    """A transaction export that rolls up to the student-step table ``steps``.

    Each row becomes its step's first attempt, after a page view that has no
    attempt number, and an incorrect first attempt is followed by a correct
    second one. A step the student meets again is met in a new problem view.
    The KC columns are copied but not the opportunity counts, which the rollup
    has to make. A table without times gets one step a minute, in row order.
    """
    kc_columns = [c for c in steps.columns if c.startswith("KC (")]
    views: dict[tuple, int] = {}
    rows = []
    for i, step in enumerate(steps.to_dict("records")):
        t = (pd.Timestamp(step["First Transaction Time"]) - EPOCH).total_seconds() \
            if "First Transaction Time" in step else 60 * i
        student, name, outcome = step["Anon Student Id"], step["Step Name"], step["First Attempt"]
        seen = views[student, step["Problem Name"], name] = views.get(
            (student, step["Problem Name"], name), 0) + 1
        common = {"Problem View": str(seen), "Problem Name": step["Problem Name"],
                  **{c: step[c] for c in kc_columns}}
        rows.append(tx_row(student, name, "", t - 1, attempt="", kc=None, **common))
        rows.append(tx_row(student, name, outcome.upper(), t, kc=None, **common))
        if outcome != "correct":
            rows.append(tx_row(student, name, "CORRECT", t + 5, attempt=2, kc=None, **common))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Synthetic data from a known model
# --------------------------------------------------------------------------


def synthetic(n_students, n_kcs, n_items, seed, n_reps=6, gamma=None,
              return_truth=False):
    """Generate a student-step rollup from a known AFM.

    Items are assigned to KCs round-robin, so ``n_items == n_kcs`` produces
    exactly one item per KC — the degenerate granularity that makes
    item-blocked CV cold-start.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.8, size=n_students)
    beta = rng.normal(0.3, 1.0, size=n_kcs)
    gamma = rng.uniform(0.05, 0.4, size=n_kcs) if gamma is None else np.asarray(gamma)

    item_kc = np.arange(n_items) % n_kcs
    records = []
    for s in range(n_students):
        seen = np.zeros(n_kcs, dtype=int)
        order = rng.permutation(np.repeat(np.arange(n_items), n_reps))
        for item in order:
            k = item_kc[item]
            t = seen[k]
            p = 1.0 / (1.0 + np.exp(-(theta[s] + beta[k] + gamma[k] * t)))
            records.append({
                "Anon Student Id": f"s{s:03d}",
                "Problem Name": "prob",
                "Step Name": f"step{item:03d}",
                "First Attempt": "correct" if rng.random() < p else "incorrect",
                "kc": f"kc{k:03d}",
                "opp": str(t + 1),  # DataShop is 1-based
            })
            seen[k] += 1

    data = step_data(records)
    if return_truth:
        order = np.argsort([f"kc{k:03d}" for k in range(n_kcs)])
        return data, {"theta": theta, "beta": beta[order], "gamma": gamma[order]}
    return data


def simulate_pfa(n_students=40, n_reps=12, seed=5, truth=None, student_sd=0.0):
    """Sequentially simulate from a known PFA — counts feed back into outcomes."""
    truth = truth or {"A": (0.3, 0.30, -0.25), "B": (-0.4, 0.20, -0.10),
                      "C": (0.0, 0.10, -0.30)}
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_students):
        theta = rng.normal(0.0, student_sd)
        s_cnt = dict.fromkeys(truth, 0)
        f_cnt = dict.fromkeys(truth, 0)
        order = [k for _ in range(n_reps) for k in truth]
        rng.shuffle(order)
        for t, kc in enumerate(order):
            beta, gamma, rho = truth[kc]
            p = 1.0 / (1.0 + np.exp(-(theta + beta + gamma * s_cnt[kc] + rho * f_cnt[kc])))
            y = int(rng.random() < p)
            rows.append(step_row(f"S{i:03d}", f"st{kc}{t}", y, kc,
                                 s_cnt[kc] + f_cnt[kc] + 1))
            s_cnt[kc] += y
            f_cnt[kc] += 1 - y
    return step_data(rows), truth
