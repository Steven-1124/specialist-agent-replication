"""Rerun the paper's six-stage workflow or the three DeepSeek judgment channels."""

import argparse
import csv
import datetime
import fcntl
import gzip
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

from responses import (MODE_IDS, _validate_required_outputs, parse_task_completion_response,
                       parse_failure_modes_response, parse_failure_modes_prompt_2_response)
from workflow_prompt_vars import (init_workflow_prompt_vars, substitute_workflow_prompt_vars,
                                  update_prompt_vars_after_agent)

HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / "prompts"
ROLES = ("Chief Executive Officer", "Chief Product Officer", "Chief Technology Officer",
         "Programmer", "Code Reviewer", "Chief Product Officer Manual")
ROLE_FILES = ("01_ceo", "02_cpo", "03_cto", "04_programmer", "05_reviewer", "06_cpo_manual")
CHANNELS = {
    "completion": ("01_task_completion", parse_task_completion_response),
    "standard": ("02_standard_failure_modes", parse_failure_modes_response),
    "p2": ("03_p2_issue_identification", parse_failure_modes_prompt_2_response),
}
TRACE_MARKER = "[EXECUTION-SPECIFIC MAS TRACE]"
CONFIG = json.loads((HERE / "models.json").read_text())
CANONICAL = "canonical_8192_chat_temp1_top_p095_deepseek-flash"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def serialize(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        try:
            f.write(serialize(value))
            f.flush()
            os.replace(f.name, path)
        finally:
            Path(f.name).unlink(missing_ok=True)


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def load_templates():
    return {role: (PROMPTS / "mas" / (name + ".txt")).read_text()
            for role, name in zip(ROLES, ROLE_FILES)}


def read_inputs():
    with (HERE / "tasks.csv").open(encoding="utf-8", newline="") as f:
        tasks = {r["task_id"]: r["task_prompt"] for r in csv.DictReader(f)}
    with (HERE / "protocol.csv").open(encoding="utf-8", newline="") as f:
        runs = list(csv.DictReader(f))
    return tasks, runs


def mask_local_path(text):
    return re.sub(r'(/Users/[^\s"\']+)', "[ABSOLUTE_PATH]", text)


def speak_to_team(task, task_id, templates, call):
    """Route the fixed graph. Each node retains its own chat history."""
    values = init_workflow_prompt_vars(task)
    histories = {role: [] for role in ROLES}
    turns = []

    def turn(role, user_text):
        system = substitute_workflow_prompt_vars(templates[role].replace("{task_id}", task_id), values)
        history = histories[role]
        history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": system}, *history]
        output = call(role, messages)
        history.append({"role": "assistant", "content": output})
        update_prompt_vars_after_agent(role, output, values)
        # The original logger concatenates the message contents, with no delimiter.
        turns.append({"sequence": len(turns) + 1, "role": role,
                      "input": mask_local_path("".join(m["content"] for m in messages)),
                      "output": mask_local_path(output)})
        return output

    text = task
    for role in ROLES[:3]:
        text = turn(role, text)
    for round_number in range(1, 4):
        code = turn(ROLES[3], text)
        text = turn(ROLES[4], code)
        if "<APPROVE>" in text:
            break
        if "<REVISE>" not in text:
            raise ValueError("The reviewer response triggered neither graph edge.")
        if round_number == 3:
            # A forced exit activates the manual node, without a carried message.
            # The final code/reviews still enter its dynamically expanded system prompt.
            text = ""
    turn(ROLES[5], text)
    _validate_required_outputs(turns)
    return [{"task": task_id, "task_prompt": task, "traces": turns}]


def replay(rows, tasks, out, write_replay):
    """Recreate the historical judge input from public turn outputs and compare its digest."""
    keys = {(r["group"], r["task"], r["arm"], r["repeat"]) for r in rows}
    seen = set()
    exact = masked = 0
    templates = load_templates()
    with gzip.open(HERE.parent / "traces" / "runs.jsonl.gz", "rt", encoding="utf-8") as src:
        for line in src:
            record = json.loads(line)
            key = (record["group"].removeprefix("G"), record["task_id"],
                   record["arm"], str(record["run_number"]))
            if key not in keys:
                continue
            if key in seen:
                raise ValueError(f"Duplicate trace: {key}")
            seen.add(key)
            if record["masked_turns"]:
                masked += 1
                continue
            turns = iter(record["turns"])

            def recorded_response(role, messages):
                turn = next(turns)
                if turn["role"] != role:
                    raise ValueError(f"Recorded role differs in {key}")
                return turn["output"]

            group, task, arm, repeat = key
            simple = speak_to_team(tasks[task], task, templates, recorded_response)
            if next(turns, None) is not None or digest(serialize(simple)) != record["simple_trace_sha256"]:
                raise ValueError(f"Historical judge input differs in {key}")
            exact += 1
            if write_replay:
                folder = out / ("g"+group) / arm / ("run_"+repeat) / task
                with run_lock(folder):
                    output = folder / "simple_trace.json"
                    if output.exists() and digest(output.read_text()) != record["simple_trace_sha256"]:
                        raise ValueError(f"Different existing trace at {output}")
                    write_json(output, simple)
    if seen != keys:
        raise ValueError("Missing selected trace keys")
    print(f"{exact} exact archived judge inputs; {masked} have redacted source turns and were skipped.")


def chat(endpoint, model, options, token, timeout):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    payload = dict(options, model=model)
    request = Request(endpoint.rstrip("/") + "/chat/completions",
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers)
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if result.get("model") != model:
        raise ValueError(f"Requested {model}, received {result.get('model')!r}; check the endpoint model identity.")
    choice = result["choices"][0]
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The endpoint returned an empty completion.")
    return result, content


def generate(row, task, folder, timeout):
    templates = load_templates()
    key = {"row": row, "task": task, "templates": {r: digest(v) for r,v in templates.items()},
           "models": CONFIG, "temperature": 1.0, "top_p": .95}
    meta = folder / "generation.json"
    trace_file = folder / "simple_trace.json"
    if trace_file.exists() and not meta.exists():
        raise ValueError(f"Existing trace lacks a fresh-generation receipt: {trace_file}. Choose a clean output directory.")
    if meta.exists():
        old = json.loads(meta.read_text())
        if old["configuration"] != key or digest(trace_file.read_text()) != old["sha256"]:
            raise ValueError(f"Existing run has conflicting inputs or bytes: {folder}")
        return
    calls = []

    def generate_turn(role, messages):
        is_coder = role == ROLES[3]
        model_key = CONFIG["coders"][row["arm"]] if is_coder else CONFIG["groups"][row["group"]]
        model = CONFIG["models"][model_key]
        endpoint = os.environ.get(model["endpoint_env"]) or os.environ.get("MAS_BASE_URL")
        if not endpoint:
            raise ValueError("Set " + model["endpoint_env"] + " or MAS_BASE_URL.")
        token_allowance = int(row["coder_max_tokens"]) if is_coder else 8192
        opts = dict(messages=messages, temperature=1., top_p=.95, max_tokens=token_allowance)
        result, text = chat(endpoint, model["id"], opts, os.environ.get("MAS_API_KEY"), timeout)
        calls.append(dict(role=role, model=model["id"], messages=messages, output=text,
                          max_tokens=token_allowance, usage=result.get("usage")))
        return text

    simple = speak_to_team(task, row["task"], templates, generate_turn)
    write_json(folder / "model_calls.json", calls)
    write_json(trace_file, simple)
    write_json(meta, {"configuration": key, "sha256": digest(trace_file.read_text()),
                      "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})


def evaluate(row, folder, channels, timeout):
    trace = (folder / "simple_trace.json").read_text()
    endpoint = os.environ.get("JUDGE_BASE_URL")
    if not endpoint:
        raise ValueError("Set JUDGE_BASE_URL to the DeepSeek chat API base URL.")
    alias = os.environ.get("JUDGE_MODEL", "deepseek-flash")
    canonical = row["evaluator_regime"] == CANONICAL
    for channel in channels:
        name, parse = CHANNELS[channel]
        template = (PROMPTS / "mast" / (name + ".txt")).read_text()
        if template.count(TRACE_MARKER) != 1:
            raise ValueError(f"Unrecognized trace placeholder in {name}")
        request = dict(messages=[{"role": "user", "content": template.replace(TRACE_MARKER, trace)}],
                       temperature=1., top_p=.95, response_format={"type": "json_object"},
                       thinking={"type": "disabled"}, reasoning_effort="none" if canonical else "high",
                       max_tokens=8192 if canonical else 262144)
        settings = {"model": alias, "trace_sha256": digest(trace), "prompt_sha256": digest(template),
                    "request_sha256": digest(serialize(request)), "historical_regime": row["evaluator_regime"]}
        passes = []
        for n in range(1, 4):
            out = folder / f"{channel}_pass_{n}.json"
            if out.exists():
                saved = json.loads(out.read_text())
                if saved["settings"] != settings:
                    raise ValueError(f"Existing judgments use different settings: {out}")
                parsed = parse(saved["raw_response"])
            else:
                response, content = chat(endpoint, alias, request, os.environ.get("JUDGE_API_KEY"), timeout)
                parsed = parse(content)
                write_json(out, {"settings": settings, "raw_response": content, "parsed": parsed,
                                 "usage": response.get("usage"), "finish_reason": response["choices"][0].get("finish_reason")})
            passes.append(parsed)
        if channel == "completion":
            voted = {"task_completion": int(sum(p["task_completion"] for p in passes) >= 2)}
        else:
            voted = {"failure_modes": {m: int(sum(p["failure_modes"][m] for p in passes) >= 2) for m in MODE_IDS}}
            voted["total_failures"] = sum(voted["failure_modes"].values())
        write_json(folder / f"{channel}_voted.json", voted)


def main():
    tasks, runs = read_inputs()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "replay", "generate", "judge"))
    p.add_argument("--group", choices=tuple(CONFIG["groups"]))
    p.add_argument("--arm", choices=tuple(CONFIG["coders"]))
    p.add_argument("--task", choices=tuple(tasks))
    p.add_argument("--repeat", choices=("1", "2", "3"))
    p.add_argument("--out", type=Path, default=Path("regenerated"))
    p.add_argument("--write-replay", action="store_true", help="save exact reconstructed judge inputs to --out")
    p.add_argument("--channel", choices=(*CHANNELS, "all"), default="all")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()
    selected = [r for r in runs if all(getattr(args, k) in (None, r[k])
                for k in ("group", "arm", "task", "repeat"))]
    channels = tuple(CHANNELS) if args.channel == "all" else [args.channel]
    print(f"{len(selected)} runs; {3*len(channels)*len(selected)} evaluation requests if judging from scratch.", flush=True)
    if args.action == "replay":
        replay(selected, tasks, args.out, args.write_replay)
        return
    if args.action == "plan":
        return
    for row in selected:
        folder = args.out / ("g"+row["group"]) / row["arm"] / ("run_"+row["repeat"]) / row["task"]
        print(folder, flush=True)
        with run_lock(folder):
            if args.action == "generate":
                generate(row, tasks[row["task"]], folder, args.timeout)
            else:
                evaluate(row, folder, channels, args.timeout)


if __name__ == "__main__":
    main()
