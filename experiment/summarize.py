"""Export fresh independent judgments and apply the paper's primary estimator."""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

from run import CHANNELS, HERE, MODE_IDS, digest, read_inputs


def checked_votes(folder, row):
    receipt_files = [folder / f"{channel}_pass_{i}.json"
                     for channel in ("completion", "standard") for i in (1, 2, 3)]
    if not all(f.is_file() for f in receipt_files):
        return None
    trace_digest = digest((folder / "simple_trace.json").read_text())
    generation = "archived_rejudged"
    if (folder / "generation.json").is_file():
        provenance = json.loads((folder / "generation.json").read_text())
        if provenance["configuration"]["row"] != row or provenance["sha256"] != trace_digest:
            raise ValueError(f"Run metadata disagrees with trace or design at {folder}.")
        generation = "regenerated"
    checked = {}
    models = set()
    for channel in ("completion", "standard"):
        values = []
        unique_settings = set()
        for index in (1, 2, 3):
            receipt = json.loads((folder / f"{channel}_pass_{index}.json").read_text())
            settings = receipt["settings"]
            if settings["trace_sha256"] != trace_digest or settings["historical_regime"] != row["evaluator_regime"]:
                raise ValueError(f"Judgment provenance mismatch at {folder}.")
            unique_settings.add(json.dumps(settings, sort_keys=True))
            models.add(settings["model"])
            values.append(CHANNELS[channel][1](receipt["raw_response"]))
        if len(unique_settings) != 1:
            raise ValueError(f"Pass settings differ within {channel} at {folder}.")
        checked[channel] = values
    if len(models) != 1:
        raise ValueError(f"Mixed judge models at {folder}.")
    return {"group": "G"+row["group"], "task_id": row["task"], "arm": row["arm"],
            "run_number": int(row["repeat"]),
            "completion": int(sum(v["task_completion"] for v in checked["completion"]) >= 2),
            **{"mode_"+m: int(sum(v["failure_modes"][m] for v in checked["standard"]) >= 2) for m in MODE_IDS},
            "generation_source": generation, "judge_regime": row["evaluator_regime"]}, models.pop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True, help="directory populated by run.py")
    p.add_argument("--tables", type=Path, required=True, help="new output directory")
    args = p.parse_args()
    if not args.runs.is_dir():
        p.error("--runs must name an existing run directory")
    if args.tables.exists():
        p.error("--tables must be a new directory")
    _, runs = read_inputs()
    execution_rows, models, sources = [], set(), set()
    for row in runs:
        folder = args.runs / ("g"+row["group"]) / row["arm"] / ("run_"+row["repeat"]) / row["task"]
        item = checked_votes(folder, row)
        if item:
            result, model = item
            execution_rows.append(result)
            models.add(model)
            sources.add(result["generation_source"])
    if not execution_rows:
        p.error("no runs have all six valid standard-split judgments")
    if len(models) != 1 or len(sources) != 1:
        p.error("use separate output trees for different judge models or generation sources")
    args.tables.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.tables.parent) as td:
        stage = Path(td) / "tables"
        stage.mkdir()
        with (stage / "executions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, list(execution_rows[0]))
            w.writeheader()
            w.writerows(execution_rows)
        if len(execution_rows) == len(runs):
            sys.path.insert(0, str(HERE.parent / "analysis"))
            import pandas as pd
            from reproduce import case_results
            outputs = case_results(pd.DataFrame(execution_rows))
            for filename, table in zip(("cases", "overall", "state_counts", "task_risk_frequency"), outputs):
                table.to_csv(stage / (filename + ".csv"), index=False)
        metadata = {"rated_runs": len(execution_rows), "planned_runs": len(runs), "judge_model": models.pop(),
                    "generation_source": sources.pop(), "estimate_full_panel": len(execution_rows) == len(runs)}
        (stage / "coverage.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(args.tables)
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
