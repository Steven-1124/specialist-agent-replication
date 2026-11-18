"""Reconstruct the task-level results and check them against the published study."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = [f"mode_{c}.{i}" for c, size in [(1, 5), (2, 6), (3, 3)] for i in range(1, size + 1)]
OUTCOMES = ["completion", *MODES]
KEYS = ["group", "task_id", "arm", "run_number"]
ARMS = ("generalist", "specialist")
TASK_RESULTS = Path("data/task_results.csv")


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


def verify(write_task_results=False):
    prompts = ROOT / "prompts"
    entries = (prompts / "SHA256SUMS").read_text().splitlines()
    require(len(entries) == 9, "Expected nine prompt digests")
    for entry in entries:
        expected, filename = entry.split(maxsplit=1)
        actual = hashlib.sha256((prompts / filename).read_bytes()).hexdigest()
        require(actual == expected, f"Prompt checksum mismatch for {filename}")

    votes = defaultdict(dict)
    for row in read_csv("votes/evaluator_passes.csv"):
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
    for key, draws in votes.items():
        require(set(draws) == {1, 2, 3}, f"Incorrect vote coverage for {key}")
        majority = tuple(int(sum(col) >= 2) for col in zip(*draws.values()))
        require(archived[key] == majority, f"Archived majority disagrees for {key}")
        group, task, arm, run = key
        case = cases[group, task]
        stats = case.setdefault(arm, {})
        stats[run] = (majority[0], sum(majority[1:]))

    require(len(cases) == 150, "Expected 150 group–task cases")
    require(len({task for group, task in cases}) == 30, "Expected 30 tasks")
    require({group for group, task in cases} == {f"G{g}" for g in range(1, 6)}, "Expected five groups")
    counts = Counter()
    task_totals = defaultdict(Counter)
    for (group, task), arms in cases.items():
        require(set(arms) == set(ARMS), "Expected matched arms")
        totals = task_totals[task]
        totals["groups"] += 1
        for arm, runs in arms.items():
            require(set(runs) == {"1", "2", "3"}, "Expected three runs in each arm")
            for complete, flagged_modes in runs.values():
                totals[arm, "runs"] += 1
                totals[arm, "complete"] += complete
                totals[arm, "modes"] += flagged_modes
        control = [sum(col) for col in zip(*arms["generalist"].values())]
        treatment = [sum(col) for col in zip(*arms["specialist"].values())]
        completion, modes = (x - y for x, y in zip(treatment, control))
        if completion >= 0 and modes > 0:
            counts["risk"] += 1
            totals["risk_groups"] += 1
        elif completion >= 0 and modes < 0:
            counts["benefit"] += 1
        else:
            counts["other"] += 1
    require(counts == {"risk": 35, "benefit": 74, "other": 41}, f"State mismatch {counts}")
    require(sum(t["risk_groups"] > 0 for t in task_totals.values()) == 21, "Expected 21 tasks with a risk case")
    totals = Counter()
    for task in task_totals.values():
        totals.update(task)
    require(totals["generalist", "complete"] == 146 and totals["specialist", "complete"] == 220,
            "Completion totals differ from the paper")
    require(totals["generalist", "modes"] == 1257 and totals["specialist", "modes"] == 1148,
            "Flag totals differ from the paper")
    names = {row["task_id"]: row["display_task"] for row in read_csv("data/task_names.csv")}
    require(names.keys() == task_totals.keys(), "Task names do not match the results")
    rows = []
    for task in sorted(names, key=lambda key: names[key].casefold()):
        totals = task_totals[task]
        require(totals["groups"] == 5, f"Expected five groups for {task}")
        row = {"Task": names[task], "Task ID": task}
        for arm in ARMS:
            runs = totals[arm, "runs"]
            row[f"{arm.title()} completion (%)"] = f"{100 * totals[arm, 'complete'] / runs:.2f}"
        for arm in ARMS:
            row[f"{arm.title()} flagged modes per run"] = f"{totals[arm, 'modes'] / totals[arm, 'runs']:.2f}"
        row["Risk groups (of 5)"] = str(totals["risk_groups"])
        row["Risk frequency (%)"] = f"{100 * totals['risk_groups'] / totals['groups']:.0f}"
        rows.append(row)

    if write_task_results:
        with (ROOT / TASK_RESULTS).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    require(list(read_csv(TASK_RESULTS)) == rows, "Task results differ from the archived judgments")
    print(f"{'Task':22} {'Completion (%)':>16} {'Modes per run':>16}  Risk groups")
    print(f"{'':22} {'Gen. / Spec.':>16} {'Gen. / Spec.':>16}  (out of 5)")
    for row in rows:
        completion = f"{row['Generalist completion (%)']} / {row['Specialist completion (%)']}"
        modes = f"{row['Generalist flagged modes per run']} / {row['Specialist flagged modes per run']}"
        print(f"{row['Task']:22} {completion:>16} {modes:>16}  {row['Risk groups (of 5)']}")
    print("Verified all 30 tasks. Risk occurs in 35 of 150 group–task cases, spanning 21 tasks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-task-results", action="store_true", help="Regenerate data/task_results.csv")
    verify(parser.parse_args().write_task_results)
