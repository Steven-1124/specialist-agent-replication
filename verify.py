"""Check the archived prompt digests, votes, and primary published counts offline."""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = [f"mode_{c}.{i}" for c, size in [(1, 5), (2, 6), (3, 3)] for i in range(1, size + 1)]
OUTCOMES = ["completion", *MODES]
KEYS = ["group", "task_id", "arm", "run_number"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def read_flags(row):
    values = tuple(int(row[name]) for name in OUTCOMES)
    require(all(x in (0, 1) for x in values), "A judgment must be binary")
    return values


def verify():
    prompts = ROOT / "prompts"
    entries = (prompts / "SHA256SUMS").read_text().splitlines()
    require(len(entries) == 9, "Expected nine prompt digests")
    for entry in entries:
        expected, filename = entry.split(maxsplit=1)
        actual = hashlib.sha256((prompts / filename).read_bytes()).hexdigest()
        require(actual == expected, f"Prompt checksum mismatch for {filename}")

    votes = defaultdict(dict)
    for row in read_csv("votes/standard_split_passes.csv"):
        key = tuple(row[name] for name in KEYS)
        draw = int(row["evaluator_pass"])
        require(draw not in votes[key], f"Duplicate evaluator pass {key}, {draw}")
        votes[key][draw] = read_flags(row)

    archived = {}
    for row in read_csv("data/executions.csv"):
        key = tuple(row[name] for name in KEYS)
        require(key not in archived, f"Duplicate archived execution {key}")
        archived[key] = read_flags(row)
    require(set(votes) == set(archived) and len(votes) == 900, "Expected 900 matched executions")

    cases = defaultdict(dict)
    totals = Counter()
    for key, draws in votes.items():
        require(set(draws) == {1, 2, 3}, f"Incorrect vote coverage for {key}")
        majority = tuple(int(sum(col) >= 2) for col in zip(*draws.values()))
        require(archived[key] == majority, f"Archived majority disagrees for {key}")
        group, task, arm, run = key
        case = cases[group, task]
        stats = case.setdefault(arm, {})
        stats[run] = (majority[0], sum(majority[1:]))
        totals[arm, "complete"] += majority[0]
        totals[arm, "modes"] += sum(majority[1:])

    require(len(cases) == 150, "Expected 150 group–task cases")
    require(len({task for group, task in cases}) == 30, "Expected 30 tasks")
    require({group for group, task in cases} == {f"G{g}" for g in range(1, 6)}, "Expected five groups")
    counts = Counter()
    risk_tasks = set()
    for (group, task), arms in cases.items():
        require(set(arms) == {"generalist", "specialist"}, "Expected matched arms")
        for runs in arms.values():
            require(set(runs) == {"1", "2", "3"}, "Expected three runs in each arm")
        control = [sum(col) for col in zip(*arms["generalist"].values())]
        treatment = [sum(col) for col in zip(*arms["specialist"].values())]
        completion, modes = (x - y for x, y in zip(treatment, control))
        if completion >= 0 and modes > 0:
            counts["risk"] += 1
            risk_tasks.add(task)
        elif completion >= 0 and modes < 0:
            counts["benefit"] += 1
        else:
            counts["other"] += 1
    require(counts == {"risk": 35, "benefit": 74, "other": 41}, f"State mismatch {counts}")
    require(len(risk_tasks) == 21, "Expected 21 tasks with a risk case")
    require(totals["generalist", "complete"] == 146 and totals["specialist", "complete"] == 220,
            "Completion totals differ from the paper")
    require(totals["generalist", "modes"] == 1257 and totals["specialist", "modes"] == 1148,
            "Flag totals differ from the paper")
    print("Verified 9 prompts, 2,700 passes, 900 executions and 150 paired group–task cases.")
    for arm in ("generalist", "specialist"):
        print(f"{arm.title()}: {totals[arm, 'complete'] / 450:.2%} judged complete; "
              f"{totals[arm, 'modes'] / 450:.2f} flagged modes per run.")
    print("Risk 35/150 (23.3%), benefit 74/150, other 41/150; risk occurs in 21 of 30 tasks.")


if __name__ == "__main__":
    verify()
