#!/usr/bin/env python3
"""Serve one Hugging Face causal LM through a strict OpenAI-compatible API.

The service is deliberately small and dependency-light outside the model
stack: HTTP is implemented with the Python standard library.  A single model
is loaded once on the CUDA device or devices assigned to the process, quantized
with bitsandbytes NF4, and all generations are serialized.  Requests are
formatted only through the checkpoint's native ``apply_chat_template``.

The server never truncates a prompt, never offloads a model to CPU/disk, and
never reports a max-token cutoff as a normal stop.  These properties let the
client fail closed instead of silently accepting incomplete generations.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import signal
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from sanitized_diagnostics import sanitized_generation_diagnostic


SERVER_SCHEMA_VERSION = "3"
ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_REQUEST_BYTES = 16 * 1024 * 1024
GPT_OSS_FINAL_CHANNEL_MARKER = "<|channel|>final<|message|>"
GPT_OSS_ANALYSIS_CHANNEL_MARKER = "<|channel|>analysis<|message|>"
GPT_OSS_TERMINATORS = ("<|return|>", "<|end|>", "<|endoftext|>")
GPT_OSS_REVIEW_MARKERS = ("<APPROVE>", "<REVISE>")
GPT_OSS_GENERATION_BUDGET_POLICY = "request_max_tokens_with_context_guard"
GPT_OSS_ATTENTION_IMPLEMENTATION = "os_mas_chunked_eager"
GPT_OSS_ATTENTION_QUERY_CHUNK_SIZE = 256
GPT_OSS_STOPPING_POLICY = (
    "official_openai_harmony_parse_then_stop_after_final_strict_complete_json"
)
TOKENIZER_SENTINEL_THRESHOLD = 1_000_000_000


class ServiceError(RuntimeError):
    """A model-load or inference contract failure."""


class StructuredOutputError(ServiceError):
    """A fail-closed structured-output error with content-free diagnostics."""

    def __init__(self, message: str, *, safe_details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.safe_details = dict(safe_details)


class RequestContractError(ValueError):
    """An invalid chat-completions request."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ServiceError(f"Required HPC package is unavailable: {name}") from exc


def _optional_package_version(name: str) -> str | None:
    """Record optional loader packages without imposing them on older models."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _linux_process_memory_mib() -> tuple[int, int]:
    """Return current and peak resident memory with Linux-unit semantics."""

    if not sys.platform.startswith("linux"):
        raise ServiceError("Process-memory telemetry requires the Linux HPC runtime")
    try:
        status_lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
        vm_rss_line = next(line for line in status_lines if line.startswith("VmRSS:"))
        fields = vm_rss_line.split()
        if len(fields) != 3 or fields[2] != "kB":
            raise ValueError("unexpected VmRSS format")
        current_kib = int(fields[1])
        peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, StopIteration, TypeError, ValueError) as exc:
        raise ServiceError("Unable to collect Linux process-memory telemetry") from exc
    if current_kib <= 0 or peak_kib < current_kib:
        raise ServiceError("Linux process-memory telemetry is internally inconsistent")
    return (current_kib + 1023) // 1024, (peak_kib + 1023) // 1024


def _positive_context_limit(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _context_limit(model: Any, tokenizer: Any) -> int:
    configs: list[Any] = [model.config]
    for name in ("text_config", "decoder", "language_config"):
        nested = getattr(model.config, name, None)
        if nested is not None and nested is not model.config:
            configs.append(nested)
    candidates: list[int] = []
    for config in configs:
        value = _positive_context_limit(
            getattr(config, "max_position_embeddings", None)
        )
        if value is not None:
            candidates.append(value)
    if not candidates:
        for config in configs:
            for name in ("n_positions", "max_sequence_length", "max_seq_len"):
                value = _positive_context_limit(getattr(config, name, None))
                if value is not None:
                    candidates.append(value)
    tokenizer_limit = _positive_context_limit(
        getattr(tokenizer, "model_max_length", None)
    )
    if (
        tokenizer_limit is not None
        and tokenizer_limit < TOKENIZER_SENTINEL_THRESHOLD
    ):
        candidates.append(tokenizer_limit)
    if not candidates:
        raise ServiceError("Checkpoint/tokenizer does not expose a finite context limit")
    return min(candidates)


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise RequestContractError("messages must be a nonempty array")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise RequestContractError(f"messages[{index}] must be an object")
        extra = set(message) - {"role", "content", "name", "tool_call_id"}
        if extra:
            raise RequestContractError(
                f"messages[{index}] has unsupported keys: {sorted(extra)!r}"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise RequestContractError(f"messages[{index}].role is invalid")
        if not isinstance(content, str):
            raise RequestContractError(f"messages[{index}].content must be a string")
        normalized.append({"role": role, "content": content})
    return normalized


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: list[dict[str, str]]
    temperature: float
    top_p: float
    max_tokens: int


def validate_chat_request(payload: Any, *, served_model_name: str) -> ChatRequest:
    if not isinstance(payload, dict):
        raise RequestContractError("request body must be a JSON object")
    supported = {"model", "messages", "temperature", "top_p", "max_tokens", "stream"}
    extra = set(payload) - supported
    if extra:
        raise RequestContractError(f"unsupported request keys: {sorted(extra)!r}")
    if payload.get("model") != served_model_name:
        raise RequestContractError(
            f"model identity mismatch: expected {served_model_name!r}"
        )
    if payload.get("stream", False) is not False:
        raise RequestContractError("streaming is not supported")
    messages = _normalize_messages(payload.get("messages"))
    temperature = payload.get("temperature", 0.0)
    top_p = payload.get("top_p", 1.0)
    max_tokens = payload.get("max_tokens", 8192)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise RequestContractError("temperature must be numeric")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise RequestContractError("top_p must be numeric")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise RequestContractError("max_tokens must be an integer")
    temperature = float(temperature)
    top_p = float(top_p)
    if temperature < 0.0:
        raise RequestContractError("temperature must be nonnegative")
    if not 0.0 < top_p <= 1.0:
        raise RequestContractError("top_p must be in (0, 1]")
    if not 1 <= max_tokens <= 32768:
        raise RequestContractError("max_tokens must be in [1, 32768]")
    return ChatRequest(messages, temperature, top_p, max_tokens)


def _load_device_map(*, loader: str, visible_count: int) -> dict[str, int] | str:
    """Require one fully CUDA-resident checkpoint per service process."""

    if visible_count != 1:
        raise ServiceError(
            "model services require exactly one visible GPU; "
            f"loader={loader!r}, found={visible_count}"
        )
    return {"": 0}


def _validate_resolved_device_map(
    *, loader: str, visible_count: int, resolved_map: Mapping[str, str]
) -> None:
    if not resolved_map or any(
        device.lower() in {"cpu", "disk", "meta"}
        for device in resolved_map.values()
    ):
        raise ServiceError(
            "Model is not fully CUDA resident; refusing offload: "
            f"hf_device_map={dict(resolved_map)!r}"
        )


def _gpt_oss_chunked_eager_attention(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Evaluate GPT-OSS eager attention exactly, without a full QxK matrix.

    Transformers' GPT-OSS eager backend materializes all query/key scores at
    once.  A 27k-token ChatDev reviewer prompt therefore needs tens of GiB for
    a single attention layer.  Query rows are independent, so evaluating them
    in fixed-size chunks preserves the causal/sliding mask and learned sink
    equations while bounding the temporary score tensor.
    """

    import torch
    from transformers.models.gpt_oss.modeling_gpt_oss import repeat_kv

    if module.training or dropout:
        raise ServiceError("Chunked GPT-OSS attention is inference-only")
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    query_length = int(query.shape[-2])
    key_length = int(key_states.shape[-2])
    outputs = []
    for start in range(0, query_length, GPT_OSS_ATTENTION_QUERY_CHUNK_SIZE):
        stop = min(start + GPT_OSS_ATTENTION_QUERY_CHUNK_SIZE, query_length)
        query_chunk = query[:, :, start:stop, :]
        attn_weights = torch.matmul(
            query_chunk, key_states.transpose(2, 3)
        ) * scaling
        if attention_mask is not None:
            if attention_mask.shape[-2] == 1:
                mask_chunk = attention_mask[:, :, :, :key_length]
            else:
                mask_chunk = attention_mask[:, :, start:stop, :key_length]
            attn_weights = attn_weights + mask_chunk
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(
            query_chunk.shape[0], -1, query_chunk.shape[-2], -1
        )
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(
            dim=-1, keepdim=True
        ).values
        probabilities = torch.nn.functional.softmax(
            combined_logits, dim=-1, dtype=combined_logits.dtype
        )
        outputs.append(torch.matmul(probabilities[..., :-1], value_states))
    output = torch.cat(outputs, dim=2).transpose(1, 2).contiguous()
    return output, None


def _register_gpt_oss_chunked_attention() -> None:
    """Register the bounded-memory backend with eager-compatible masks."""

    try:
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import eager_mask
    except Exception as exc:  # pragma: no cover - exercised on HPC
        raise ServiceError(
            "Transformers lacks the attention registration interface"
        ) from exc
    AttentionInterface.register(
        GPT_OSS_ATTENTION_IMPLEMENTATION, _gpt_oss_chunked_eager_attention
    )
    AttentionMaskInterface.register(GPT_OSS_ATTENTION_IMPLEMENTATION, eager_mask)


def _generation_token_budget(
    *, loader: str, prompt_tokens: int, request_max_tokens: int, context_limit: int,
    strict_context_budget: bool = False,
    use_remaining_context: bool = False,
) -> int:
    remaining = context_limit - prompt_tokens
    if remaining <= 0:
        raise RequestContractError(
            "prompt exhausts the checkpoint context window; "
            f"prompt_tokens={prompt_tokens}, context_limit={context_limit}. "
            "Prompt truncation is forbidden."
        )
    if use_remaining_context:
        # Recovery policy: never truncate the prompt and never exceed the
        # checkpoint's physical window. The requested output allowance remains
        # an upper bound; only a binding context window can reduce it.
        return min(request_max_tokens, remaining)
    if loader == "native_mxfp4" and not strict_context_budget:
        # Bound KV-cache growth while still allowing the full configured
        # response budget. Strict JSON stopping normally terminates earlier.
        return min(request_max_tokens, remaining)
    if request_max_tokens > remaining:
        raise RequestContractError(
            "prompt plus max_tokens exceeds the checkpoint context window; "
            f"prompt_tokens={prompt_tokens}, max_tokens={request_max_tokens}, "
            f"context_limit={context_limit}. Prompt truncation is forbidden."
        )
    return request_max_tokens


def _strict_complete_json_contract(content: str) -> bool:
    """Accept one JSON object, optionally after the review decision marker."""

    candidate = content.strip()
    for marker in GPT_OSS_REVIEW_MARKERS:
        if candidate.startswith(marker):
            candidate = candidate[len(marker) :].lstrip()
            break
    if not candidate.startswith("{"):
        return False
    try:
        value, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and not candidate[end:].strip()


def _request_expects_strict_json(messages: Sequence[Mapping[str, str]]) -> bool:
    prompt = "\n".join(message.get("content", "") for message in messages)
    return (
        "Return STRICT JSON ONLY" in prompt
        or (
            "Output format is STRICT" in prompt
            and "JSON object with schema" in prompt
        )
    )


def _native_final_content(
    raw_content: str,
    *,
    final_channel_prefilled: bool = False,
    content_prefix: str = "",
) -> str:
    # Even with a prefilled final channel, GPT-OSS may emit another explicit
    # final-channel marker. Prefer the last generated marker when present;
    # otherwise the generated continuation itself belongs to the prefilled
    # channel. The earlier implementation ignored a generated marker in the
    # prefilled case and consequently rejected valid JSON following it.
    if GPT_OSS_FINAL_CHANNEL_MARKER in raw_content:
        content = raw_content.rsplit(GPT_OSS_FINAL_CHANNEL_MARKER, 1)[1]
    elif final_channel_prefilled:
        content = raw_content
    else:
        raise ServiceError("GPT-OSS completion has no Harmony final channel")
    for terminator in GPT_OSS_TERMINATORS:
        content = content.split(terminator, 1)[0]
    return content_prefix + content


def _native_has_complete_json_contract(
    raw_content: str,
    *,
    final_channel_prefilled: bool = False,
    content_prefix: str = "",
) -> bool:
    try:
        content = _native_final_content(
            raw_content,
            final_channel_prefilled=final_channel_prefilled,
            content_prefix=content_prefix,
        )
    except ServiceError:
        return False
    return _strict_complete_json_contract(content)


def _native_structured_output_shape(
    raw_content: str, *, final_channel_prefilled: bool, content_prefix: str = ""
) -> dict[str, Any]:
    """Describe output structure without retaining any generated text."""

    try:
        content = _native_final_content(
            raw_content,
            final_channel_prefilled=final_channel_prefilled,
            content_prefix=content_prefix,
        )
    except ServiceError:
        content = ""
    candidate = content.strip()
    review_marker = next(
        (marker for marker in GPT_OSS_REVIEW_MARKERS if candidate.startswith(marker)),
        None,
    )
    json_candidate = (
        candidate[len(review_marker) :].lstrip() if review_marker else candidate
    )
    decoded_object = False
    trailing_characters = 0
    json_error_kind = "none"
    if json_candidate.startswith("{"):
        try:
            value, end = json.JSONDecoder().raw_decode(json_candidate)
        except json.JSONDecodeError as exc:
            json_error_kind = (
                "unterminated_string"
                if "Unterminated string" in exc.msg
                else "json_decode_error"
            )
        else:
            decoded_object = isinstance(value, dict)
            trailing_characters = len(json_candidate[end:].strip())
            if not decoded_object:
                json_error_kind = "decoded_non_object"
            elif trailing_characters:
                json_error_kind = "trailing_content"
    elif json_candidate.startswith("```"):
        json_error_kind = "markdown_fence"
    elif json_candidate:
        json_error_kind = "non_json_prefix"
    else:
        json_error_kind = "empty"
    return {
        "raw_character_count": len(raw_content),
        "content_character_count": len(content),
        "raw_sha256": _sha256_text(raw_content),
        "generated_final_channel_marker": GPT_OSS_FINAL_CHANNEL_MARKER in raw_content,
        "generated_analysis_channel_marker": (
            GPT_OSS_ANALYSIS_CHANNEL_MARKER in raw_content
        ),
        "review_marker": review_marker or "none",
        "starts_json_object": json_candidate.startswith("{"),
        "decoded_json_object": decoded_object,
        "trailing_character_count": trailing_characters,
        "json_error_kind": json_error_kind,
    }


class HuggingFaceEngine:
    """One fully CUDA-resident, native-chat-template Hugging Face model."""

    def __init__(
        self,
        *,
        model_key: str,
        model_id: str,
        served_model_name: str,
        endpoint_env: str,
        requested_revision: str | None,
        disable_thinking: bool,
        loader: str,
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover - exercised on HPC
            raise ServiceError(f"Unable to import the CUDA inference stack: {exc}") from exc
        harmony = None
        if loader == "native_mxfp4":
            try:
                import openai_harmony as harmony
            except Exception as exc:  # pragma: no cover - exercised on HPC
                raise ServiceError(
                    "Unable to import the official OpenAI Harmony parser"
                ) from exc
            _register_gpt_oss_chunked_attention()
        if not torch.cuda.is_available():
            raise ServiceError("CUDA is unavailable; refusing silent CPU inference")
        visible_count = torch.cuda.device_count()
        device_map = _load_device_map(loader=loader, visible_count=visible_count)
        initial_memory = [torch.cuda.mem_get_info(index) for index in range(visible_count)]
        for index in range(visible_count):
            torch.cuda.reset_peak_memory_stats(index)
        token = os.environ.get("HF_TOKEN") or None
        revision_args = {"revision": requested_revision} if requested_revision else {}
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=False,
            **revision_args,
        )
        if not getattr(tokenizer, "chat_template", None):
            raise ServiceError(f"{model_id} has no native chat template")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ServiceError(f"{model_id} has neither pad_token_id nor eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token
        if loader in {"bitsandbytes_nf4", "bitsandbytes_nf4_modern"}:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs: dict[str, Any] = {
                "torch_dtype": torch.bfloat16,
                "quantization_config": quantization,
            }
        elif loader == "native_mxfp4":
            load_kwargs = {
                "torch_dtype": "auto",
                "attn_implementation": GPT_OSS_ATTENTION_IMPLEMENTATION,
            }
        else:
            raise ServiceError(f"Unsupported checkpoint loader: {loader!r}")
        placement_kwargs: dict[str, Any] = {"device_map": device_map}
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            **placement_kwargs,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            **load_kwargs,
            **revision_args,
        )
        model.eval()
        # Loading full-precision shards before bitsandbytes conversion can
        # leave tens of GiB in glibc arenas even after Transformers releases
        # its temporary tensors. Return those free arenas to the OS before the
        # next model service starts; the CUDA-resident model itself remains.
        gc.collect()
        malloc_trim_attempted = False
        malloc_trim_succeeded = False
        try:
            malloc_trim = ctypes.CDLL(None).malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            malloc_trim_attempted = True
            malloc_trim_succeeded = bool(malloc_trim(0))
        except (AttributeError, OSError, TypeError):
            pass
        if (
            loader in {"bitsandbytes_nf4", "bitsandbytes_nf4_modern"}
            and getattr(model, "is_loaded_in_4bit", False) is not True
        ):
            raise ServiceError(
                "The loaded checkpoint does not attest bitsandbytes 4-bit quantization"
            )
        if loader == "native_mxfp4":
            quantization_config = getattr(model.config, "quantization_config", {}) or {}
            if not isinstance(quantization_config, Mapping):
                quantization_config = {
                    "quant_method": getattr(quantization_config, "quant_method", None)
                }
            if quantization_config.get("quant_method") != "mxfp4":
                raise ServiceError(
                    "The native checkpoint does not attest MXFP4 quantization"
                )
            if (
                getattr(model.config, "_attn_implementation", None)
                != GPT_OSS_ATTENTION_IMPLEMENTATION
            ):
                raise ServiceError(
                    "GPT-OSS did not resolve the bounded-memory attention backend"
                )
        resolved_map = {
            str(name): str(device)
            for name, device in (getattr(model, "hf_device_map", {}) or {}).items()
        }
        _validate_resolved_device_map(
            loader=loader,
            visible_count=visible_count,
            resolved_map=resolved_map,
        )
        input_device = model.get_input_embeddings().weight.device
        if getattr(input_device, "type", None) != "cuda":
            raise ServiceError(f"Model input embeddings are not on CUDA: {input_device}")
        resolved_revision = getattr(model.config, "_commit_hash", None)
        if not isinstance(resolved_revision, str) or not REVISION_RE.fullmatch(resolved_revision):
            raise ServiceError(
                "Hugging Face did not resolve the checkpoint to an exact 40-hex commit: "
                f"{resolved_revision!r}"
            )
        if requested_revision is not None and resolved_revision != requested_revision:
            raise ServiceError(
                "Resolved checkpoint revision differs from the preflight-pinned revision"
            )
        tokenizer_revision = (
            getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or resolved_revision
        )
        if not isinstance(tokenizer_revision, str) or not REVISION_RE.fullmatch(tokenizer_revision):
            raise ServiceError(
                "Hugging Face did not resolve the tokenizer to an exact 40-hex commit"
            )
        self.model_key = model_key
        self.model_id = model_id
        self.served_model_name = served_model_name
        self.endpoint_env = endpoint_env
        self.requested_revision = requested_revision
        self.resolved_revision = resolved_revision
        self.tokenizer_revision = tokenizer_revision
        self.disable_thinking = disable_thinking
        self.loader = loader
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.transformers = transformers
        self.harmony = harmony
        self.harmony_encoding = (
            harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
            if harmony is not None
            else None
        )
        self.input_device = input_device
        self.context_limit = _context_limit(model, tokenizer)
        self.lock = threading.Lock()
        self.started_monotonic = time.monotonic()
        self.chat_template_sha256 = _sha256_text(tokenizer.chat_template)
        self.library_versions = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": _package_version("accelerate"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "huggingface_hub": _package_version("huggingface-hub"),
            "kernels": _optional_package_version("kernels"),
            "triton": _optional_package_version("triton"),
            "openai_harmony": _optional_package_version("openai-harmony"),
        }
        self.gpu_memory_by_device = []
        for index, (initial_free_bytes, initial_total_bytes) in enumerate(initial_memory):
            free_after_bytes, total_after_bytes = torch.cuda.mem_get_info(index)
            self.gpu_memory_by_device.append(
                {
                    "device_index": index,
                    "total_mib": int(total_after_bytes // (1024 * 1024)),
                    "free_before_load_mib": int(initial_free_bytes // (1024 * 1024)),
                    "free_after_load_mib": int(free_after_bytes // (1024 * 1024)),
                    "allocated_after_load_mib": int(
                        torch.cuda.memory_allocated(index) // (1024 * 1024)
                    ),
                    "reserved_after_load_mib": int(
                        torch.cuda.memory_reserved(index) // (1024 * 1024)
                    ),
                    "peak_allocated_during_load_mib": int(
                        torch.cuda.max_memory_allocated(index) // (1024 * 1024)
                    ),
                    "initial_total_matches_post_load": initial_total_bytes
                    == total_after_bytes,
                }
            )
        # Existing single-GPU Qwen sharing checks read this compatibility field.
        self.gpu_memory = self.gpu_memory_by_device[0]
        current_rss_mib, peak_rss_mib = _linux_process_memory_mib()
        self.cpu_memory = {
            "current_rss_mib_after_trim": current_rss_mib,
            "peak_rss_mib": peak_rss_mib,
            "malloc_trim_attempted": malloc_trim_attempted,
            "malloc_trim_succeeded": malloc_trim_succeeded,
            "current_rss_source": "/proc/self/status:VmRSS",
            "peak_rss_source": "getrusage(RUSAGE_SELF).ru_maxrss_linux_kib",
        }

    def service_record(self, endpoint: str) -> dict[str, Any]:
        return {
            "schema_version": SERVER_SCHEMA_VERSION,
            "model_key": self.model_key,
            "served_model_name": self.served_model_name,
            "hf_model_id": self.model_id,
            "revision": self.resolved_revision,
            "requested_revision": self.requested_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "endpoint_env": self.endpoint_env,
            "endpoint": endpoint,
            "quantization": (
                {
                    "method": "native-mxfp4",
                    "source_weight_format": "safetensors",
                    "use_safetensors": True,
                    "load_in_4bit": False,
                }
                if self.loader == "native_mxfp4"
                else {
                    "method": "bitsandbytes-nf4",
                    "source_weight_format": "safetensors",
                    "use_safetensors": True,
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "bfloat16",
                    "bnb_4bit_use_double_quant": True,
                }
            ),
            "chat_template": {
                "method": "tokenizer.apply_chat_template",
                "add_generation_prompt": True,
                "sha256": self.chat_template_sha256,
                "disable_thinking": self.disable_thinking,
            },
            "library_versions": self.library_versions,
            "server_source_sha256": _sha256_text(
                Path(__file__).read_text(encoding="utf-8")
            ),
            "runtime": {
                "cuda_device_name": self.torch.cuda.get_device_name(0),
                "cuda_device_names": [
                    self.torch.cuda.get_device_name(index)
                    for index in range(self.torch.cuda.device_count())
                ],
                "cuda_visible_device_count": self.torch.cuda.device_count(),
                "cuda_selector": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "process_id": os.getpid(),
                "is_loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False) is True,
                "quantization_runtime": (
                    "native-mxfp4"
                    if self.loader == "native_mxfp4"
                    else "bitsandbytes-nf4"
                ),
                "gpu_memory": self.gpu_memory,
                "gpu_memory_by_device": self.gpu_memory_by_device,
                "cpu_memory": self.cpu_memory,
                "context_limit": self.context_limit,
                "checkpoint_generation_defaults": {
                    name: getattr(self.model.generation_config, name, None)
                    for name in ("top_k", "min_p", "typical_p", "repetition_penalty",
                                 "length_penalty", "num_beams", "no_repeat_ngram_size")
                },
                "sampling_policy": "do_sample iff request.temperature > 0; request.top_p used only when sampling",
                "generation_seed_policy": "no fixed generation seed supplied",
                "generation_budget_policy": (
                    "request_max_tokens_clamped_to_remaining_context"
                    if os.environ.get("OS_MAS_CONTEXT_OVERFLOW_POLICY")
                    == "use_remaining_context"
                    else "request_max_tokens_or_context_error"
                    if os.environ.get("OS_MAS_STRICT_CONTEXT_BUDGET") == "1"
                    else GPT_OSS_GENERATION_BUDGET_POLICY
                    if self.loader == "native_mxfp4"
                    else "request_max_tokens"
                ),
                "structured_output_stopping_policy": (
                    GPT_OSS_STOPPING_POLICY
                    if self.loader == "native_mxfp4"
                    else "checkpoint_eos"
                ),
                "reasoning_effort": (
                    "low" if self.loader == "native_mxfp4" else "not_applicable"
                ),
                "kv_cache_implementation": (
                    "offloaded_all_layers"
                    if self.loader == "native_mxfp4"
                    else "checkpoint_default"
                ),
                "attention_implementation": getattr(
                    self.model.config, "_attn_implementation", "unknown"
                ),
                "attention_query_chunk_size": (
                    GPT_OSS_ATTENTION_QUERY_CHUNK_SIZE
                    if self.loader == "native_mxfp4"
                    else None
                ),
                "hf_device_map": {
                    str(name): str(device)
                    for name, device in (getattr(self.model, "hf_device_map", {}) or {}).items()
                },
            },
        }

    def _eos_ids(self) -> set[int]:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if isinstance(eos, int):
            return {eos}
        if isinstance(eos, (list, tuple)) and all(isinstance(item, int) for item in eos):
            return set(eos)
        raise ServiceError("Generation configuration has no valid EOS token id")

    def generate(self, request: ChatRequest) -> tuple[str, int, int]:
        with self.lock, self.torch.inference_mode():
            template_kwargs: dict[str, Any] = {"add_generation_prompt": True}
            if self.disable_thinking:
                template_kwargs["enable_thinking"] = False
            expects_strict_json = _request_expects_strict_json(request.messages)
            native_final_prefilled = False
            native_content_prefix = ""
            if self.loader == "native_mxfp4":
                if self.harmony is None or self.harmony_encoding is None:
                    raise ServiceError("Official OpenAI Harmony parser is unavailable")
                harmony_messages = [
                    self.harmony.Message.from_role_and_content(
                        self.harmony.Role.SYSTEM,
                        self.harmony.SystemContent.new().with_reasoning_effort(
                            self.harmony.ReasoningEffort.LOW
                        ),
                    )
                ]
                for message in request.messages:
                    role = message.get("role")
                    content = message.get("content", "")
                    if role == "system":
                        harmony_message = self.harmony.Message.from_role_and_content(
                            self.harmony.Role.DEVELOPER,
                            self.harmony.DeveloperContent.new().with_instructions(content),
                        )
                    elif role == "user":
                        harmony_message = self.harmony.Message.from_role_and_content(
                            self.harmony.Role.USER, content
                        )
                    elif role == "assistant":
                        harmony_message = self.harmony.Message.from_role_and_content(
                            self.harmony.Role.ASSISTANT, content
                        ).with_channel("final")
                    else:
                        raise RequestContractError(
                            "GPT-OSS ChatDev requests do not support tool-history messages"
                        )
                    harmony_messages.append(harmony_message)
                conversation = self.harmony.Conversation.from_messages(harmony_messages)
                prompt_ids = self.harmony_encoding.render_conversation_for_completion(
                    conversation, self.harmony.Role.ASSISTANT
                )
                encoded = self.torch.tensor(
                    [prompt_ids], dtype=self.torch.long, device=self.input_device
                )
            else:
                encoded = self.tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    **template_kwargs,
                )
            if isinstance(encoded, Mapping):
                model_inputs = dict(encoded)
            else:
                model_inputs = {
                    "input_ids": encoded,
                    "attention_mask": self.torch.ones_like(encoded),
                }
            model_inputs = {
                key: value.to(self.input_device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }
            input_ids = model_inputs.get("input_ids")
            if input_ids is None or getattr(input_ids, "ndim", None) != 2:
                raise ServiceError("Native chat template did not return batched input_ids")
            prompt_tokens = int(input_ids.shape[-1])
            generation_budget = _generation_token_budget(
                loader=self.loader,
                prompt_tokens=prompt_tokens,
                request_max_tokens=request.max_tokens,
                context_limit=self.context_limit,
                strict_context_budget=os.environ.get("OS_MAS_STRICT_CONTEXT_BUDGET") == "1",
                use_remaining_context=(
                    os.environ.get("OS_MAS_CONTEXT_OVERFLOW_POLICY")
                    == "use_remaining_context"
                ),
            )
            do_sample = request.temperature > 0.0
            generation_kwargs: dict[str, Any] = {
                **model_inputs,
                "max_new_tokens": generation_budget,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": (
                    self.harmony_encoding.stop_tokens_for_assistant_actions()
                    if self.loader == "native_mxfp4"
                    else sorted(self._eos_ids())
                ),
                "use_cache": True,
            }
            if self.loader == "native_mxfp4":
                # GPT-OSS otherwise returns prompt-length logits on the first
                # generation step.  For a 26k-token reviewer prompt and its
                # 200k vocabulary that transient tensor is about 10.8 GiB,
                # even though autoregressive generation needs only the final
                # position.  The pinned GPT-OSS forward API supports this
                # exact last-token-only optimization.
                generation_kwargs["logits_to_keep"] = 1
                # GPT-OSS's long ChatDev review history otherwise fills the
                # remaining 95 GiB device with KV state. Offload every cache
                # layer, including sliding-window layers; model parameters
                # remain fully CUDA resident and are still checked above.
                generation_kwargs["cache_implementation"] = "offloaded"
                generation_kwargs["cache_config"] = {
                    "offload_only_non_sliding": False
                }
            if self.loader == "native_mxfp4" and expects_strict_json:
                tokenizer = self.tokenizer

                class StopAfterStrictJson(self.transformers.StoppingCriteria):
                    def __call__(self, sequences, scores, **kwargs):  # noqa: ANN001
                        raw = tokenizer.decode(
                            sequences[0, prompt_tokens:],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        return _native_has_complete_json_contract(
                            raw,
                            final_channel_prefilled=native_final_prefilled,
                            content_prefix=native_content_prefix,
                        )

                generation_kwargs["stopping_criteria"] = (
                    self.transformers.StoppingCriteriaList([StopAfterStrictJson()])
                )
            if do_sample:
                generation_kwargs.update(
                    {"temperature": request.temperature, "top_p": request.top_p}
                )
            sequences = self.model.generate(**generation_kwargs)
            generated_ids = sequences[0, prompt_tokens:]
            completion_tokens = int(generated_ids.shape[-1])
            if completion_tokens == 0:
                raise ServiceError("Model returned no new tokens")
            if self.loader == "native_mxfp4":
                completion_ids = [int(token_id) for token_id in generated_ids.tolist()]
                raw_content = self.harmony_encoding.decode_utf8(completion_ids)
                try:
                    parsed_messages = self.harmony_encoding.parse_messages_from_completion_tokens(
                        completion_ids, self.harmony.Role.ASSISTANT, strict=True
                    )
                except (RuntimeError, ValueError) as exc:
                    raise ServiceError("Official Harmony parsing failed") from exc
                final_messages = [
                    message for message in parsed_messages if message.channel == "final"
                ]
                if len(final_messages) != 1:
                    raise ServiceError("GPT-OSS returned no unique Harmony final channel")
                content = "".join(
                    part.text
                    for part in final_messages[0].content
                    if isinstance(part, self.harmony.TextContent)
                )
                if not content:
                    raise ServiceError("GPT-OSS Harmony final channel contained no text")
                complete_json = _strict_complete_json_contract(content)
                if expects_strict_json and not complete_json:
                    safe_details = _native_structured_output_shape(
                        raw_content,
                        final_channel_prefilled=native_final_prefilled,
                        content_prefix=native_content_prefix,
                    )
                    safe_details.update(
                        {
                            "completion_tokens": completion_tokens,
                            "generation_budget": generation_budget,
                            "ended_on_eos": int(generated_ids[-1]) in self._eos_ids(),
                        }
                    )
                    if completion_tokens >= generation_budget:
                        raise StructuredOutputError(
                            "GPT-OSS exhausted every remaining declared context "
                            "position before completing the strict JSON contract",
                            safe_details=safe_details,
                        )
                    raise StructuredOutputError(
                        "GPT-OSS stopped without a strict complete JSON contract",
                        safe_details=safe_details,
                    )
                if (
                    not complete_json
                    and completion_tokens >= generation_budget
                    and int(generated_ids[-1]) not in self._eos_ids()
                ):
                    raise ServiceError(
                        "GPT-OSS exhausted every remaining declared context "
                        "position without EOS"
                    )
            else:
                if (
                    completion_tokens >= generation_budget
                    and int(generated_ids[-1]) not in self._eos_ids()
                ):
                    raise ServiceError(
                        "Generation exhausted max_tokens without EOS; refusing a "
                        "truncated completion"
                    )
                content = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
            if not content.strip():
                raise ServiceError("Decoded completion is empty")
            return content, prompt_tokens, completion_tokens


class ModelHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        engine: HuggingFaceEngine,
        api_key: str | None,
    ) -> None:
        self.engine = engine
        self.api_key = api_key
        super().__init__(address, ModelRequestHandler)


class ModelRequestHandler(BaseHTTPRequestHandler):
    server: ModelHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.client_address[0], self.log_date_time_string(), format_string % args)
        )

    def _authorized(self) -> bool:
        expected = self.server.api_key
        if expected is None:
            return True
        return self.headers.get("Authorization") == f"Bearer {expected}"

    def _json(self, status: int, value: Mapping[str, Any]) -> None:
        payload = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _error(
        self,
        status: int,
        message: str,
        error_type: str,
        *,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> None:
        error: dict[str, Any] = {"message": message, "type": error_type}
        if diagnostic is not None:
            error["diagnostic"] = dict(diagnostic)
        self._json(status, {"error": error})

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "invalid bearer token", "authentication_error")
            return
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ready"})
        elif path == "/v1/models":
            self._json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.engine.served_model_name,
                            "object": "model",
                            "owned_by": "huggingface-hpc",
                        }
                    ],
                },
            )
        elif path == "/deployment":
            endpoint = getattr(self.server, "advertised_endpoint", "")
            self._json(HTTPStatus.OK, self.server.engine.service_record(endpoint))
        else:
            self._error(HTTPStatus.NOT_FOUND, "unknown endpoint", "not_found")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "invalid bearer token", "authentication_error")
            return
        if urlsplit(self.path).path.rstrip("/") != "/v1/chat/completions":
            self._error(HTTPStatus.NOT_FOUND, "unknown endpoint", "not_found")
            return
        try:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise RequestContractError("Content-Length is required")
            length = int(raw_length)
            if not 1 <= length <= MAX_REQUEST_BYTES:
                raise RequestContractError("request body size is invalid")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            request = validate_chat_request(
                payload,
                served_model_name=self.server.engine.served_model_name,
            )
            content, prompt_tokens, completion_tokens = self.server.engine.generate(request)
        except (RequestContractError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc), "invalid_request_error")
            return
        except Exception as exc:
            diagnostic = sanitized_generation_diagnostic(exception=exc)
            safe_details = getattr(exc, "safe_details", None)
            # Persist only fixed, content-free metadata.  In particular, never
            # log the request body or the raw exception message here.
            event: dict[str, Any] = {
                "event": "generation_error",
                "model_key": self.server.engine.model_key,
                "diagnostic": diagnostic,
            }
            if isinstance(safe_details, Mapping):
                event["safe_details"] = dict(safe_details)
            sys.stderr.write(
                _canonical_json(event) + "\n"
            )
            sys.stderr.flush()
            if isinstance(safe_details, Mapping):
                diagnostic = {**diagnostic, "safe_details": dict(safe_details)}
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                diagnostic["diagnostic_message"],
                "generation_error",
                diagnostic=diagnostic,
            )
            return
        created = int(time.time())
        self._json(
            HTTPStatus.OK,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": created,
                "model": self.server.engine.served_model_name,
                "system_fingerprint": (
                    "hf-" + self.server.engine.resolved_revision[:16]
                ),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--endpoint-env", required=True)
    parser.add_argument("--revision", "--requested-revision", dest="revision")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--advertise-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--service-record", type=Path, required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--loader",
        choices=("bitsandbytes_nf4", "bitsandbytes_nf4_modern", "native_mxfp4"),
        default="bitsandbytes_nf4",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.model_key):
        parser.error("--model-key contains unsafe characters")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", args.endpoint_env):
        parser.error("--endpoint-env must be an uppercase environment name")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    if args.revision and not REVISION_RE.fullmatch(args.revision):
        parser.error("--revision must be an exact 40-hex Hugging Face commit")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ServiceError(f"{args.api_key_env} is unset or empty")
    engine = HuggingFaceEngine(
        model_key=args.model_key,
        model_id=args.model_id,
        served_model_name=args.served_model_name,
        endpoint_env=args.endpoint_env,
        requested_revision=args.revision,
        disable_thinking=not args.enable_thinking,
        loader=args.loader,
    )
    server = ModelHTTPServer((args.bind_host, args.port), engine, api_key)
    actual_port = int(server.server_address[1])
    endpoint = f"http://{args.advertise_host}:{actual_port}/v1"
    server.advertised_endpoint = endpoint
    record = engine.service_record(endpoint)
    _atomic_write_json(args.service_record.expanduser().resolve(), record)

    def stop_server(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    print(_canonical_json({"status": "ready", **record}), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
