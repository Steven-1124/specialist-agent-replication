"""View an archived execution or its evidence crosswalk. Python standard library only."""
import argparse
import gzip
import json
from pathlib import Path


def records(filename):
    with gzip.open(Path(__file__).parent / filename, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=["G1", "G2", "G3", "G4", "G5"])
    parser.add_argument("task_id")
    parser.add_argument("arm", choices=["generalist", "specialist"])
    parser.add_argument("run_number", type=int, choices=[1, 2, 3])
    parser.add_argument("--issues", action="store_true", help="show issue annotations with the located excerpts")
    parser.add_argument("--label", help="show issue annotations with just this issue code")
    parser.add_argument("--sequence", type=int, help="show just this turn")
    args = parser.parse_args()
    key = ("group", "task_id", "arm", "run_number")

    def is_selected(item):
        return all(item[field] == getattr(args, field) for field in key)

    run = next((item for item in records("runs.jsonl.gz") if is_selected(item)), None)
    if run is None:
        parser.error("no archived run matches these identifiers")
    turns = {step["sequence"]: step["output"] for step in run["turns"]}
    if args.sequence is not None and args.sequence not in turns:
        parser.error("the specified turn does not occur in this run")
    if args.issues or args.label:
        result = []
        for item in records("issue_anchors.jsonl.gz"):
            if not is_selected(item) or (args.label and item["label"] != args.label):
                continue
            if args.sequence is not None:
                item["anchors"] = [a for a in item["anchors"] if a["sequence"] == args.sequence]
                if not item["anchors"] and args.sequence not in item["recorded_sequences"]:
                    continue
            for anchor in item["anchors"]:
                if anchor["field"] == "output" and "start" in anchor:
                    anchor["excerpt"] = turns[anchor["sequence"]][anchor["start"]:anchor["end"]]
            result.append(item)
    else:
        result = run
        if args.sequence is not None:
            result["turns"] = [step for step in run["turns"] if step["sequence"] == args.sequence]
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
