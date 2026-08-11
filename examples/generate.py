#!/usr/bin/env python3
"""Regenerate ``student-step.txt``, the synthetic example export.

Every row is simulated from a known Additive Factors Model — no student ever
produced this data. The file exists so the README's commands run out of the
box and so the expected input format has a concrete, inspectable instance.

Twelve students each attempt the same 40 steps once, in a per-student shuffled
order. Two KC models tag the steps at different granularities:

* ``KC (Topics)`` — 4 coarse topics, 10 steps each
* ``KC (Skills)`` — 12 finer skills, 3-4 steps each

Responses are drawn from ``logit P(correct) = theta_i + beta_k + gamma_k * T``
using the Topics model, with every learning rate positive — so fitting Topics
should recover clean positive slopes, while Skills illustrates comparing a
finer KC model on identical data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

TOPICS = ["fractions", "decimals", "ratios", "percents"]
SKILLS_PER_TOPIC = 3
N_STUDENTS = 12
STEPS_PER_TOPIC = 10

rng = np.random.default_rng(7)

theta = rng.normal(0.0, 0.7, N_STUDENTS)                    # student ability
beta = dict(zip(TOPICS, [0.4, -0.2, -0.6, 0.1]))            # topic easiness
gamma = dict(zip(TOPICS, [0.20, 0.15, 0.25, 0.10]))         # learning rate

# 40 steps: (problem, step, topic, skill). Skills subdivide their topic.
steps = []
for topic in TOPICS:
    for j in range(STEPS_PER_TOPIC):
        skill = f"{topic}-{'abc'[j % SKILLS_PER_TOPIC]}"
        steps.append((f"{topic.capitalize()}-P{j // 2 + 1}", f"step-{j % 2 + 1}", topic, skill))

rows = []
for i in range(N_STUDENTS):
    order = rng.permutation(len(steps))
    topic_seen: dict[str, int] = {}
    skill_seen: dict[str, int] = {}
    clock = 0
    for idx in order:
        problem, step, topic, skill = steps[idx]
        t = topic_seen.get(topic, 0)
        p = 1.0 / (1.0 + np.exp(-(theta[i] + beta[topic] + gamma[topic] * t)))
        correct = rng.random() < p
        topic_seen[topic] = t + 1
        skill_seen[skill] = skill_seen.get(skill, 0) + 1
        clock += int(rng.integers(30, 300))  # seconds between attempts
        rows.append({
            "Anon Student Id": f"Stu_{i + 1:02d}",
            "Problem Name": problem,
            "Step Name": step,
            "First Transaction Time": f"2024-03-0{i % 7 + 1} "
                                      f"{9 + clock // 3600:02d}:{clock // 60 % 60:02d}:{clock % 60:02d}",
            "First Attempt": "correct" if correct else "incorrect",
            "KC (Topics)": topic,
            "Opportunity (Topics)": str(topic_seen[topic]),
            "KC (Skills)": skill,
            "Opportunity (Skills)": str(skill_seen[skill]),
        })

out = Path(__file__).parent / "student-step.txt"
with out.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out} ({len(rows)} rows)")
