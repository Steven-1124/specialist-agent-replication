"""Runtime workflow prompt placeholders: parse agent outputs and fill {{VAR}} templates."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

# Stored on ExecutionContext.global_state; nested dict mutated across nodes.
WORKFLOW_PROMPT_VARS_KEY = "workflow_prompt_vars"

# Public placeholders use ALL_CAPS inside double braces, e.g. {{CEO_KPI}}.
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def substitute_workflow_prompt_vars(template: str, prompt_vars: Dict[str, Any]) -> str:
    """Replace {{VAR}} with string values from prompt_vars; missing keys become empty string."""

    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        val = prompt_vars.get(key)
        if val is None:
            return ""
        return str(val)

    return _PLACEHOLDER_RE.sub(repl, template)


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _fmt_lines(items: Any) -> str:
    if not items:
        return ""
    if isinstance(items, str):
        return items
    if isinstance(items, list):
        return "\n".join(f"- {x}" for x in items if x is not None)
    return str(items)


def init_workflow_prompt_vars(task_text: str) -> Dict[str, Any]:
    """Seed state before the first agent runs."""
    return {
        "TASK": task_text or "",
        "CEO_KPI": "",
        "CPO_MODALITY": "",
        "CPO_SPECIFIC_REQUIREMENTS": "",
        "CPO_SPEC": "",
        "CTO_LANGUAGE": "",
        "CTO_ARCHITECTURE": "",
        "CTO_PLAN": "",
        "CODE_VERSION_LATEST": "",
        "REVIEWER_FEEDBACK_LATEST": "",
        "ALL_CODE_VERSIONS": "",
        "ALL_REVIEWER_FEEDBACK": "",
        "__code_round": 0,
        "__review_round": 0,
    }


def update_prompt_vars_after_agent(node_id: str, output_text: str, pv: Dict[str, Any]) -> None:
    """Parse model output and store fields for downstream {{PLACEHOLDER}} substitution."""
    parsed = _extract_json_object(output_text)
    phase = str(parsed.get("phase", "")) if parsed else ""

    if node_id == "Chief Executive Officer" and parsed and phase == "KPIDefinition":
        pv["CEO_KPI"] = json.dumps(parsed, ensure_ascii=False, indent=2)

    elif node_id == "Chief Product Officer" and parsed and phase == "ProductSpec":
        pv["CPO_SPEC"] = json.dumps(parsed, ensure_ascii=False, indent=2)
        pv["CPO_MODALITY"] = str(parsed.get("product_modality") or "")
        pv["CPO_SPECIFIC_REQUIREMENTS"] = _fmt_lines(parsed.get("specific_requirements"))

    elif node_id == "Chief Technology Officer" and parsed and phase == "TechnologyPlan":
        pv["CTO_PLAN"] = json.dumps(parsed, ensure_ascii=False, indent=2)
        pv["CTO_LANGUAGE"] = str(parsed.get("language") or "")
        arch = parsed.get("architecture")
        if isinstance(arch, dict):
            pv["CTO_ARCHITECTURE"] = json.dumps(arch, ensure_ascii=False, indent=2)
        else:
            pv["CTO_ARCHITECTURE"] = str(arch or "")

    elif node_id == "Programmer":
        if not (parsed and phase == "Coding"):
            parsed = None
        n = int(pv.get("__code_round") or 0) + 1
        pv["__code_round"] = n
        body = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else output_text.strip()
        pv["CODE_VERSION_LATEST"] = body
        pv[f"CODE_VERSION_{n}"] = body
        versions: list[str] = []
        for i in range(1, n + 1):
            key = f"CODE_VERSION_{i}"
            if key in pv and pv[key]:
                versions.append(f"=== Code version {i} ===\n{pv[key]}")
        pv["ALL_CODE_VERSIONS"] = "\n\n".join(versions)

    elif node_id == "Code Reviewer":
        n = int(pv.get("__review_round") or 0) + 1
        pv["__review_round"] = n
        pv["REVIEWER_FEEDBACK_LATEST"] = output_text.strip()
        pv[f"REVIEWER_FEEDBACK_{n}"] = output_text.strip()
        feedbacks: list[str] = []
        for i in range(1, n + 1):
            key = f"REVIEWER_FEEDBACK_{i}"
            if key in pv and pv[key]:
                feedbacks.append(f"=== Review round {i} ===\n{pv[key]}")
        pv["ALL_REVIEWER_FEEDBACK"] = "\n\n".join(feedbacks)

    elif node_id == "Chief Product Officer Manual" and parsed and phase == "Manual":
        pv["CPO_MANUAL"] = str(parsed.get("manual_content") or "")
