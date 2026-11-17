"""Verify the public trace archive against its digests and the analysis inputs."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from read import records


HERE = Path(__file__).resolve().parent
KEY = ("group", "task_id", "arm", "run_number")


def row_key(row):
    return tuple(str(row[k]) for k in KEY)


def load_csv(name):
    with (HERE.parent / "data" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(test, message):
    if not test:
        raise ValueError(message)


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for filename, digest in manifest["sha256"].items():
        require(hashlib.sha256((HERE / filename).read_bytes()).hexdigest() == digest, filename)

    runs = {row_key(r): r for r in records("runs.jsonl.gz")}
    execution_keys = {row_key(r) for r in load_csv("executions.csv")}
    require(len(runs) == 900 and runs.keys() == execution_keys, "run keys differ from the analysis inputs")
    require(len({r["run_id"] for r in runs.values()}) == 900, "duplicated source run ID")
    all_steps = {}
    n_turns = 0
    for key, run in runs.items():
        steps = {t["sequence"]: t["output"] for t in run["turns"]}
        require(len(steps) == len(run["turns"]), "duplicate sequence")
        require(set(run["masked_turns"]) <= steps.keys(), "unknown masked sequence")
        all_steps[key] = steps
        n_turns += len(steps)
    require(n_turns == manifest["roles_turns"], "turn count")

    public_issues = load_csv("issue_associations.csv")
    seen = set()
    counts = Counter()
    for issue in records("issue_anchors.jsonl.gz"):
        index = issue["association"]
        require(1 <= index <= len(public_issues) and index not in seen, "invalid association number")
        seen.add(index)
        expected = public_issues[index - 1]
        key = row_key(issue)
        require(key == row_key(expected), "association run key")
        for field in ("label", "role", "event_number", "object_number"):
            value = issue[field]
            require(("" if value is None else str(value)) == expected[field], field)
        for a in issue["anchors"]:
            if a.get("status") == "unmatched":
                counts["unmatched"] += 1
                continue
            start, stop = a["start"], a["end"]
            require(0 <= start < stop, "invalid character span")
            field = a["field"]
            if field == "output":
                require(stop <= len(all_steps[key][a["sequence"]]), "span beyond output")
            else:
                require(field in ("input", "task_prompt") and len(a["excerpt"]) == stop - start, "excerpt length")
            counts["located " + field] += 1
    require(len(seen) == manifest["associations"] == len(public_issues), "association count")
    for name, count in counts.items():
        require(count == manifest["anchor_and_mask_counts"][name], name)
    print(f"Verified {len(runs)} executions, {n_turns} turns and {len(seen)} issue associations.")


if __name__ == "__main__":
    main()
