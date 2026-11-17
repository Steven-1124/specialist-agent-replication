"""Bounded, content-free diagnostics for local model-service failures.

The model server can encounter exceptions whose raw messages contain request
content, filesystem paths, or credentials.  Runtime evidence therefore stores
only an allow-listed diagnosis derived from the exception text; the original
message is never returned by these helpers.
"""

from __future__ import annotations

import re
from typing import Any


DIAGNOSTIC_SCHEMA_VERSION = "os_mas_sanitized_generation_error_v1"
_SAFE_ATOM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,79}$")

_DIAGNOSTIC_MESSAGES = {
    "cuda_out_of_memory": "CUDA memory was exhausted during generation.",
    "cuda_device_mismatch": "CUDA tensors were placed on different devices during generation.",
    "cuda_runtime_error": "The CUDA runtime failed during generation.",
    "context_window_exhausted": "The request exhausted the declared context window.",
    "generation_max_tokens_without_eos": "Generation reached its request limit without EOS.",
    "missing_harmony_final_channel": "GPT-OSS returned no Harmony final channel.",
    "strict_json_contract_incomplete": "GPT-OSS returned no complete strict-JSON response.",
    "empty_completion": "The model returned no usable completion.",
    "generation_runtime_error": "The model runtime failed during generation.",
    "generation_exception": "Model generation failed.",
}


def _safe_exception_type(value: str) -> str:
    candidate = value.strip()
    return candidate if _SAFE_ATOM_RE.fullmatch(candidate) else "Exception"


def _diagnostic_code(exception_type: str, message: str) -> str:
    lowered = f"{exception_type} {message}".lower()
    if any(
        phrase in lowered
        for phrase in (
            "outofmemoryerror",
            "cuda out of memory",
            "cuda memory was exhausted",
        )
    ):
        return "cuda_out_of_memory"
    if any(
        phrase in lowered
        for phrase in (
            "all tensors to be on the same device",
            "found at least two devices",
            "not on the expected device",
            "different devices",
            "same device as the indexed tensor",
            "same device as indices",
            "device mismatch",
        )
    ):
        return "cuda_device_mismatch"
    if "cuda" in lowered and any(
        phrase in lowered
        for phrase in (
            "illegal memory access",
            "device-side assert",
            "cublas",
            "cudnn",
            "cuda error",
        )
    ):
        return "cuda_runtime_error"
    if "prompt exhausts" in lowered or "context window" in lowered:
        return "context_window_exhausted"
    if "max_tokens without eos" in lowered:
        return "generation_max_tokens_without_eos"
    if "harmony final channel" in lowered:
        return "missing_harmony_final_channel"
    if "strict" in lowered and "json" in lowered:
        return "strict_json_contract_incomplete"
    if "no new tokens" in lowered or "completion is empty" in lowered:
        return "empty_completion"
    if exception_type in {"RuntimeError", "ServiceError"}:
        return "generation_runtime_error"
    return "generation_exception"


def sanitized_generation_diagnostic(
    *,
    exception: BaseException | None = None,
    exception_type: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Return fixed-text diagnostic metadata without retaining raw content."""

    if exception is not None:
        resolved_type = type(exception).__name__
        resolved_message = str(exception)
    else:
        resolved_type = exception_type or "Exception"
        resolved_message = message or ""
    safe_type = _safe_exception_type(resolved_type)
    code = _diagnostic_code(safe_type, resolved_message)
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "exception_type": safe_type,
        "diagnostic_code": code,
        "diagnostic_message": _DIAGNOSTIC_MESSAGES[code],
    }
