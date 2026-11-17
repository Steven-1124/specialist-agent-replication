"""Response and artifact validation routines extracted from the study code."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

MODE_IDS = tuple(f"{group}.{mode}" for group, count in ((1, 5), (2, 6), (3, 3)) for mode in range(1, count + 1))


class CompletionJudgeError(RuntimeError):
    """The completion judge did not return a complete, valid annotation."""


def parse_task_completion_response(response: str) -> dict[str, Any]:
    """Parse the strict two-field completion response."""

    text = (response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CompletionJudgeError(
            "Task-completion judge did not return valid JSON"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"summary", "task_completion"}:
        raise CompletionJudgeError(
            "Task-completion response must contain exactly summary and task_completion"
        )
    summary = payload["summary"]
    completion = payload["task_completion"]
    if not isinstance(summary, str) or not summary.strip():
        raise CompletionJudgeError("Task-completion summary must be non-empty")
    if type(completion) is not bool:
        raise CompletionJudgeError("task_completion must be true or false")
    return {"summary": summary.strip(), "task_completion": completion}


class FailureJudgeError(RuntimeError):
    """The failure-mode judge did not return a complete, valid annotation."""


def _coerce_binary(value: Any, *, field: str) -> int:
    if type(value) is bool:
        return int(value)
    if type(value) is int and value in (0, 1):
        return value
    raise FailureJudgeError(f"{field} must be true/false or 0/1")


def parse_failure_modes_response(response: str) -> dict[str, Any]:
    """Parse the strict summary plus 14-mode diagnostic response."""

    text = (response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FailureJudgeError("Failure-mode judge did not return valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"summary", "failure_modes"}:
        raise FailureJudgeError(
            "Failure-mode response must contain exactly summary and failure_modes"
        )
    summary = payload["summary"]
    raw_modes = payload["failure_modes"]
    if not isinstance(summary, str) or not summary.strip():
        raise FailureJudgeError("Failure-mode summary must be non-empty")
    if not isinstance(raw_modes, dict) or set(raw_modes) != set(MODE_IDS):
        raise FailureJudgeError("failure_modes must contain exactly the 14 MAST IDs")
    modes = {
        mode_id: _coerce_binary(raw_modes[mode_id], field=f"failure_modes.{mode_id}")
        for mode_id in MODE_IDS
    }
    return {
        "summary": summary.strip(),
        "failure_modes": modes,
        "total_failures": sum(modes.values()),
    }


class FailurePrompt2JudgeError(RuntimeError):
    """The prompt-2 failure judge did not return complete, grounded JSON."""


def parse_failure_modes_prompt_2_response(response: str) -> dict[str, Any]:
    """Parse modes plus all cited locations, enforcing their consistency."""

    text = (response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FailurePrompt2JudgeError("Failure prompt 2 did not return valid JSON") from exc
    expected_keys = {"summary", "failure_modes", "failure_mode_locations"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise FailurePrompt2JudgeError(
            "Failure prompt 2 response must contain exactly summary, failure_modes, "
            "and failure_mode_locations"
        )
    summary = payload["summary"]
    raw_modes = payload["failure_modes"]
    raw_locations = payload["failure_mode_locations"]
    if not isinstance(summary, str) or not summary.strip():
        raise FailurePrompt2JudgeError("Failure summary must be non-empty")
    if not isinstance(raw_modes, dict) or set(raw_modes) != set(MODE_IDS):
        raise FailurePrompt2JudgeError("failure_modes must contain exactly the 14 MAST IDs")
    modes = {
        mode_id: _coerce_binary(raw_modes[mode_id], field=f"failure_modes.{mode_id}")
        for mode_id in MODE_IDS
    }
    positive_modes = {mode_id for mode_id, value in modes.items() if value == 1}
    if not isinstance(raw_locations, dict) or set(raw_locations) != positive_modes:
        raise FailurePrompt2JudgeError(
            "failure_mode_locations must contain exactly the failure modes marked 1"
        )
    locations: dict[str, list[dict[str, str]]] = {}
    for mode_id in MODE_IDS:
        if mode_id not in raw_locations:
            continue
        examples = raw_locations[mode_id]
        if not isinstance(examples, list) or not examples:
            raise FailurePrompt2JudgeError(
                f"failure_mode_locations.{mode_id} must be a non-empty list"
            )
        parsed_examples: list[dict[str, str]] = []
        for index, example in enumerate(examples):
            if not isinstance(example, dict) or set(example) not in (
                {"issue", "reference trace"},
                {"issue", "reference_trace"},
            ):
                raise FailurePrompt2JudgeError(
                    f"failure_mode_locations.{mode_id}[{index}] must contain exactly "
                    "issue and reference trace (the JSON-safe reference_trace spelling "
                    "is also accepted and normalized)"
                )
            issue = example["issue"]
            reference = example.get("reference trace", example.get("reference_trace"))
            if not isinstance(issue, str) or not issue.strip():
                raise FailurePrompt2JudgeError(
                    f"failure_mode_locations.{mode_id}[{index}].issue must be non-empty"
                )
            if not isinstance(reference, str) or not reference.strip():
                raise FailurePrompt2JudgeError(
                    f"failure_mode_locations.{mode_id}[{index}].reference trace must be non-empty"
                )
            parsed_examples.append(
                {"issue": issue.strip(), "reference trace": reference.strip()}
            )
        locations[mode_id] = parsed_examples
    return {
        "summary": summary.strip(),
        "failure_modes": modes,
        "failure_mode_locations": locations,
        "total_failures": sum(modes.values()),
    }


def _escape_json_control_characters(payload: str) -> str:
    """Escape literal JSON control characters, but only inside strings.

    Some open-source chat models emit otherwise valid JSON with literal
    newlines or tabs in long code/manual strings.  Preserve existing JSON
    escapes verbatim and leave every character outside strings untouched so
    this cannot repair malformed structure, invalid escapes, or truncation.
    """
    escaped_payload: list[str] = []
    in_string = False
    escaped = False
    named_escapes = {
        "\b": r"\b",
        "\f": r"\f",
        "\n": r"\n",
        "\r": r"\r",
        "\t": r"\t",
    }

    for character in payload:
        if not in_string:
            escaped_payload.append(character)
            if character == '"':
                in_string = True
            continue

        if escaped:
            escaped_payload.append(character)
            escaped = False
        elif character == "\\":
            escaped_payload.append(character)
            escaped = True
        elif character == '"':
            escaped_payload.append(character)
            in_string = False
        elif ord(character) < 0x20:
            escaped_payload.append(
                named_escapes.get(character, f"\\u{ord(character):04x}")
            )
        else:
            escaped_payload.append(character)

    return "".join(escaped_payload)


def _escape_invalid_json_escapes_in_code_values(payload: str) -> str:
    r"""Escape non-JSON backslashes only inside ``files[].code`` strings.

    Source code frequently contains regexes (``\s``), terminal escapes
    (``\x1b``), or other language-level backslashes that a model emits only
    once inside its outer JSON response.  JSON permits only ``\"``, ``\\``,
    ``\/``, ``\b``, ``\f``, ``\n``, ``\r``, ``\t``, and four-hex-digit
    ``\u`` escapes.  Doubling any other backslash in a Coding ``code`` value
    recovers the intended source text without repairing object structure,
    truncated strings, or invalid escapes in other fields/phases.
    """
    marker = re.compile(r'"code"\s*:\s*"')
    valid_simple = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    hex_digits = set("0123456789abcdefABCDEF")
    output: list[str] = []
    cursor = 0
    changed = False
    while True:
        match = marker.search(payload, cursor)
        if match is None:
            output.append(payload[cursor:])
            break
        quote_position = match.end() - 1
        output.append(payload[cursor : quote_position + 1])
        position = quote_position + 1
        while position < len(payload):
            character = payload[position]
            if character == '"':
                output.append(character)
                position += 1
                break
            if character != "\\":
                output.append(character)
                position += 1
                continue
            if position + 1 >= len(payload):
                return payload
            following = payload[position + 1]
            if following in valid_simple:
                output.extend((character, following))
                position += 2
                continue
            if (
                following == "u"
                and position + 5 < len(payload)
                and all(value in hex_digits for value in payload[position + 2 : position + 6])
            ):
                output.append(payload[position : position + 6])
                position += 6
                continue
            output.append("\\\\")
            changed = True
            position += 1
        else:
            return payload
        cursor = position
    return "".join(output) if changed else payload


def _close_missing_files_array(payload: str) -> str | None:
    """Repair exactly one omitted ``]`` before the top-level notes field.

    Qwen Coder can occasionally finish an otherwise complete Coding payload as
    ``...last_file},\"notes\":...`` instead of
    ``...last_file}],\"notes\":...``.  Only accept that one unambiguous shape:
    immediately before an out-of-string top-level ``notes`` key, the parser
    stack must contain exactly the root object and its still-open files array.
    The caller still subjects the repaired value to strict ``json.loads``.
    """
    marker = ',"notes"'
    marker_positions: list[int] = []
    marker_stack: list[str] | None = None
    stack: list[str] = []
    in_string = False
    escaped = False
    after_marker = False

    for position, character in enumerate(payload):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if payload.startswith(marker, position):
            marker_positions.append(position)
            if marker_stack is None:
                marker_stack = list(stack)
            after_marker = True
        if after_marker:
            if character == '"':
                in_string = True
            continue

        if character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character == "}":
            if not stack or stack[-1] != "{":
                return None
            stack.pop()
        elif character == "]":
            if not stack or stack[-1] != "[":
                return None
            stack.pop()
    if (
        len(marker_positions) != 1
        or marker_stack != ["{", "["]
        or '"files":[' not in payload
    ):
        return None
    position = marker_positions[0]
    return payload[:position] + "]" + payload[position:]


def _extract_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    payload = text[start : end + 1]
    candidates = [payload]
    escaped_payload = _escape_json_control_characters(payload)
    if escaped_payload != payload:
        candidates.append(escaped_payload)
    if '"phase":"Coding"' in re.sub(r"\s+", "", payload) and '"files":[' in re.sub(r"\s+", "", payload):
        for candidate in tuple(candidates):
            escaped_code = _escape_invalid_json_escapes_in_code_values(candidate)
            if escaped_code != candidate and escaped_code not in candidates:
                candidates.append(escaped_code)
    for candidate in tuple(candidates):
        repaired = _close_missing_files_array(candidate)
        if repaired is not None:
            candidates.append(repaired)

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value.strip()) and not path.is_absolute() and ".." not in path.parts


def _validate_required_outputs(traces: list[dict[str, Any]]) -> None:
    """A completed workflow must contain real code and a real manual payload."""
    programmer_ok = False
    manual_ok = False
    for trace in traces:
        value = _extract_json_object(trace.get("output"))
        if not value:
            continue
        if trace.get("role") == "Programmer":
            files = value.get("files")
            if isinstance(files, list):
                programmer_ok = programmer_ok or any(
                    isinstance(item, dict)
                    and _is_safe_relative_path(str(item.get("file_name") or ""))
                    and bool(str(item.get("code") or "").strip())
                    for item in files
                )
        if trace.get("role") == "Chief Product Officer Manual":
            manual_ok = manual_ok or (
                _is_safe_relative_path(str(value.get("manual_file") or ""))
                and bool(str(value.get("manual_content") or "").strip())
            )
    missing: list[str] = []
    if not programmer_ok:
        missing.append("Programmer files[]")
    if not manual_ok:
        missing.append("CPO Manual manual_file/manual_content")
    if missing:
        raise RuntimeError("Workflow missing required output: " + ", ".join(missing))
