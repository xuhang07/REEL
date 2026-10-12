"""Qwen3.5 online RL loop for the nine-tool Memory Policy protocol.

One parquet row is either one independent shot or a window of consecutive
shots.  Consecutive shots share environment state (registry, committed memory,
and the accepted previous tail) but *never* dialogue history: every shot starts
again from the same system prompt followed by that shot's original user text.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import os
import re
import socket
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jsonschema import Draft202012Validator


ECHO_LONGVIDEO_ROOT = Path(__file__).resolve().parents[2]
while str(ECHO_LONGVIDEO_ROOT) in sys.path:
    sys.path.remove(str(ECHO_LONGVIDEO_ROOT))
sys.path.insert(0, str(ECHO_LONGVIDEO_ROOT))
TOOLS_ROOT = ECHO_LONGVIDEO_ROOT / "tools"
while str(TOOLS_ROOT) in sys.path:
    sys.path.remove(str(TOOLS_ROOT))
# Keep the package root first so this recipe's local ``tools.py`` can never
# shadow the repository's top-level ``tools`` package.
sys.path.insert(1, str(TOOLS_ROOT))

from verl_recipe.storymem_memory_policy_rl.policy_profile import (  # noqa: E402
    DIRECT_PROMPT,
    PROMPT_AUTHORING,
    pe_mode_for_profile,
    TRAINING_SYSTEM_PROMPT,
    training_system_prompt_hash,
    tool_schema_hash,
)
from storymem_adapter.echo15 import (  # noqa: E402
    Echo15BackendAdapter,
    Echo15Capabilities,
)
from storymem_adapter.gemini import IsolatedGeminiTextClient  # noqa: E402
from storymem_policy.contract import (  # noqa: E402
    CANONICAL_TOOL_NAMES,
    TOOL_PROTOCOL_VERSION,
)
from storymem_policy.runtime import MemoryPolicyRuntime  # noqa: E402
from storymem_policy.state import MemoryPolicyState  # noqa: E402
from storymem_policy_collector import (  # noqa: E402
    CollectorConfig,
    Echo15WireTransport,
    FFmpegFinalEvidence,
    ProviderVisualEvidenceCreator,
    StoryCollector,
    assign_pe_mode,
)
from storymem_teacher.providers import GenericImageProvider  # noqa: E402
from verl_recipe.storymem_memory_policy_rl.contracts import (  # noqa: E402
    AGENT_NAME,
    DATASET_SCHEMA_VERSION,
    PROMPT_VALIDATION_SCHEMA_VERSION,
    SHOT_SPEC_SCHEMA_VERSION,
    STATE_SEED_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
)
from verl_recipe.storymem_memory_policy_rl.gemini_reward import (  # noqa: E402
    CrossProcessRateLimiter,
)

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.utils.chat_template import apply_chat_template as render_chat_template
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
    normalize_token_ids,
)

from .provider_config import ProviderConfigError, private_api_key
from .tools import MemoryPolicyToolSession, canonical_json


_RETIRED_ECHO_HOSTS = frozenset(host.strip() for host in os.environ.get("REEL_BLOCKED_ECHO_HOSTS", "").split(",") if host.strip())
_FUNCTION_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*\Z")
_PARAMETER_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*\Z")
_TOOL_CALL_RE = re.compile(
    r"\A(?P<prefix>.*?)<tool_call>\n"
    r"<function=(?P<function>[A-Za-z_][A-Za-z0-9_.:-]*)>\n"
    r"(?P<body>.*?)</function>\n</tool_call>[ \t\r\n]*\Z",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[A-Za-z_][A-Za-z0-9_.:-]*)>\n"
    r"(?P<value>.*?)\n</parameter>\n?",
    re.DOTALL,
)
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
_NO_THINK_GENERATION_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class ToolCallFormatError(ValueError):
    """The assistant did not emit exactly one valid native XML tool call."""

    def __init__(
        self,
        message: str,
        *,
        function_name: str | None = None,
        violations: Sequence[Mapping[str, str]] = (),
    ) -> None:
        super().__init__(message)
        self.function_name = function_name
        self.violations = tuple(dict(row) for row in violations)


class RolloutInfrastructureError(RuntimeError):
    """A provider, media, or reward service failed independently of policy."""


def _strict_json_loads(text: str, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ToolCallFormatError(f"{label} is not strict JSON: {exc}") from exc


def _json_scalar_or_python_keyword(text: str) -> Any:
    """Parse JSON scalars plus Qwen's documented ``None/True/False`` spellings."""

    stripped = text.strip()
    if stripped in {"None", "True", "False"}:
        return {"None": None, "True": True, "False": False}[stripped]
    return _strict_json_loads(stripped, label="parameter")


def _schema_allows_null(schema: Mapping[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "null" or (
        isinstance(schema_type, list) and "null" in schema_type
    ):
        return True
    return any(
        isinstance(branch, Mapping) and _schema_allows_null(branch)
        for key in ("oneOf", "anyOf")
        for branch in (schema.get(key) or [])
    )


def _non_null_schema(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("oneOf", "anyOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            candidates = [
                branch
                for branch in branches
                if isinstance(branch, Mapping) and not _schema_allows_null(branch)
            ]
            if len(candidates) == 1:
                return candidates[0]
    return schema


def _parse_parameter_value(raw: str, schema: Mapping[str, Any]) -> Any:
    value_text = raw
    if value_text.startswith("\n"):
        value_text = value_text[1:]
    if value_text.endswith("\n"):
        value_text = value_text[:-1]
    stripped = value_text.strip()
    if stripped in {"null", "None"} and _schema_allows_null(schema):
        return None

    effective = _non_null_schema(schema)
    schema_type = effective.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        schema_type = non_null[0] if len(non_null) == 1 else None

    if schema_type == "string":
        # XML parameter bodies are already string delimiters.  Do not require
        # another JSON quote layer and do not interpret Python expressions.
        return value_text
    if schema_type in {"object", "array"}:
        return _strict_json_loads(stripped, label=f"{schema_type} parameter")
    if schema_type in {"boolean", "integer", "number", "null"}:
        return _json_scalar_or_python_keyword(stripped)

    # Enum-only schemas in this contract are textual.  Unknown future schema
    # shapes still get strict JSON when they look structured, otherwise text.
    enum = effective.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(item, str) for item in enum):
        return value_text
    if stripped.startswith(("{", "[", '"')):
        return _strict_json_loads(stripped, label="parameter")
    return value_text


def _raw_schema(tool: Any) -> dict[str, Any]:
    if isinstance(tool, Mapping):
        return copy.deepcopy(dict(tool))
    model_dump = getattr(tool, "model_dump", None)
    if callable(model_dump):
        return copy.deepcopy(model_dump(exclude_unset=True, exclude_none=True))
    raise TypeError("tool schema is neither a mapping nor a model_dump wrapper")


def _parse_qwen35_xml_tool_call(
    text: str,
    tools: Sequence[Any],
) -> tuple[str, FunctionCall]:
    """Parse the one wire format emitted by Qwen3.5's official template.

    The XML envelope and newlines deliberately mirror
    ``Qwen3.5-9B/chat_template.jinja``.  This policy additionally requires
    empty assistant content, so both a reasoning/prose prefix and any suffix
    after ``</tool_call>`` are rejected.  Legacy JSON-in-XML and permissive
    partial-call variants are not accepted.
    """

    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        raise ToolCallFormatError("assistant output must contain exactly one tool_call")
    match = _TOOL_CALL_RE.fullmatch(text)
    if match is None:
        raise ToolCallFormatError(
            "assistant output is not the official Qwen3.5 XML tool-call format "
            "or contains content after </tool_call>"
        )
    prefix = match.group("prefix")
    if prefix.strip():
        raise ToolCallFormatError(
            "assistant content must be empty; prose or hidden reasoning before "
            "the tool call is forbidden"
        )
    function_name = match.group("function")
    if _FUNCTION_NAME_RE.fullmatch(function_name) is None:
        raise ToolCallFormatError("function name is malformed")

    schema_by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        raw = _raw_schema(tool)
        function = raw.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            schema_by_name[str(function["name"])] = raw
    if function_name not in schema_by_name:
        raise ToolCallFormatError(
            f"tool {function_name!r} is not exposed", function_name=function_name
        )

    parameters_schema = schema_by_name[function_name]["function"].get("parameters")
    if not isinstance(parameters_schema, Mapping):
        raise ToolCallFormatError(
            "tool parameters schema is malformed", function_name=function_name
        )
    properties = parameters_schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ToolCallFormatError(
            "tool properties schema is malformed", function_name=function_name
        )

    body = match.group("body")
    cursor = 0
    arguments: dict[str, Any] = {}
    for parameter in _PARAMETER_RE.finditer(body):
        if body[cursor : parameter.start()] != "":
            raise ToolCallFormatError(
                "unexpected content between parameters", function_name=function_name
            )
        name = parameter.group("name")
        if _PARAMETER_NAME_RE.fullmatch(name) is None:
            raise ToolCallFormatError(
                "parameter name is malformed", function_name=function_name
            )
        if name in arguments:
            raise ToolCallFormatError(
                f"duplicate parameter {name!r}",
                function_name=function_name,
                violations=({"path": name, "message": "parameter appears more than once"},),
            )
        if name not in properties:
            raise ToolCallFormatError(
                f"unknown parameter {name!r}",
                function_name=function_name,
                violations=({"path": name, "message": "parameter is not allowed"},),
            )
        property_schema = properties[name]
        if not isinstance(property_schema, Mapping):
            raise ToolCallFormatError(
                f"schema for parameter {name!r} is malformed",
                function_name=function_name,
            )
        try:
            arguments[name] = _parse_parameter_value(
                parameter.group("value"), property_schema
            )
        except ToolCallFormatError as exc:
            raise ToolCallFormatError(
                str(exc),
                function_name=function_name,
                violations=({"path": name, "message": str(exc)},),
            ) from exc
        cursor = parameter.end()
    if body[cursor:] != "":
        raise ToolCallFormatError(
            "unexpected content after parameters", function_name=function_name
        )

    errors = sorted(
        Draft202012Validator(parameters_schema).iter_errors(arguments),
        key=lambda error: list(error.path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.path) or "<root>"
        violations = tuple(
            {
                "path": ".".join(str(item) for item in error.path) or "<root>",
                "message": " ".join(str(error.message).split())[:256],
            }
            for error in errors[:8]
        )
        raise ToolCallFormatError(
            f"{function_name} arguments invalid at {location}: {first.message}",
            function_name=function_name,
            violations=violations,
        )
    return prefix, FunctionCall(
        name=function_name, arguments=canonical_json(arguments)
    )


def parse_qwen35_xml_tool_call(
    text: str,
    tools: Sequence[Any],
) -> FunctionCall:
    """Return one schema-validated official Qwen3.5 function call."""

    _, function_call = _parse_qwen35_xml_tool_call(text, tools)
    return function_call


def _required_nested_fields(parameters_schema: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return compact array-item requirements useful for one corrective retry."""

    result: dict[str, list[str]] = {}
    properties = parameters_schema.get("properties")
    if not isinstance(properties, Mapping):
        return result
    for parameter_name in sorted(str(name) for name in properties):
        raw_property = properties.get(parameter_name)
        if not isinstance(raw_property, Mapping):
            continue
        property_schema = _non_null_schema(raw_property)
        if property_schema.get("type") != "array":
            continue
        raw_items = property_schema.get("items")
        if not isinstance(raw_items, Mapping):
            continue
        item_schema = _non_null_schema(raw_items)
        required = item_schema.get("required")
        if isinstance(required, list) and all(isinstance(item, str) for item in required):
            result[f"{parameter_name}[]"] = sorted(required)
    return result


def _format_repair_contract(
    exc: ToolCallFormatError,
    tools: Sequence[Any],
) -> dict[str, Any]:
    """Build a bounded, schema-derived correction hint without repairing the action."""

    contract: dict[str, Any] = {
        "instruction": (
            "Emit one complete replacement native tool call only. Emit every top-level "
            "parameter exactly once in canonical_parameter_order. Object and array "
            "parameter bodies must be strict JSON; use null only where the schema permits it."
        )
    }
    function_name = exc.function_name
    if not function_name:
        return contract
    contract["function_name"] = function_name
    matching_schema: dict[str, Any] | None = None
    for tool in tools:
        raw = _raw_schema(tool)
        function = raw.get("function")
        if isinstance(function, Mapping) and function.get("name") == function_name:
            matching_schema = raw
            break
    if matching_schema is None:
        return contract
    parameters = matching_schema["function"].get("parameters")
    if not isinstance(parameters, Mapping):
        return contract
    properties = parameters.get("properties")
    if isinstance(properties, Mapping):
        contract["canonical_parameter_order"] = sorted(str(name) for name in properties)
    required = parameters.get("required")
    if isinstance(required, list) and all(isinstance(item, str) for item in required):
        contract["required_parameters"] = sorted(required)
    nested = _required_nested_fields(parameters)
    if nested:
        contract["nested_required_fields"] = nested
    if exc.violations:
        contract["violations"] = [dict(row) for row in exc.violations]
    return contract


def _render_qwen35_message_content(content: Any) -> str:
    """Render one Qwen3.5 message body without invoking VERL fallbacks."""

    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, Sequence) or isinstance(
        content, (bytes, bytearray, str)
    ):
        raise TypeError("Qwen3.5 environment message content must be text or an array")
    parts: list[str] = []
    for index, raw_item in enumerate(content):
        if not isinstance(raw_item, Mapping):
            raise TypeError(f"environment content item {index} must be an object")
        item_type = raw_item.get("type")
        if item_type == "image" or "image" in raw_item or "image_url" in raw_item:
            parts.append(_IMAGE_PLACEHOLDER)
        elif item_type == "video" or "video" in raw_item:
            parts.append(_VIDEO_PLACEHOLDER)
        elif item_type == "text" or "text" in raw_item:
            text = raw_item.get("text")
            if not isinstance(text, str):
                raise TypeError(f"environment text item {index} has non-text content")
            parts.append(text)
        else:
            raise TypeError(f"unsupported Qwen3.5 environment content item {index}")
    return "".join(parts).strip()


def render_qwen35_environment_fragment(
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Serialize current Qwen3.5 tool/host observations for incremental decode.

    Qwen3.5's official template refuses an isolated tool response because it
    scans for an earlier user query.  Calling VERL's generic wrapper therefore
    activates its broad dummy-user fallback.  The online loop owns incremental
    history already, so it writes the exact official wire suffix directly:
    close the sampled assistant turn, append tool/user observations, and open a
    new non-thinking assistant turn.  A user observation may be first when the
    sampled assistant turn was malformed and therefore did not invoke a tool.
    """

    if not messages:
        raise ValueError("environment message batch must not be empty")
    if messages[0].get("role") not in {"tool", "user"}:
        raise ValueError(
            "environment message batch must start with a tool response or host event"
        )

    parts = [_IM_END, "\n"]
    seen_host_event = False
    for index, message in enumerate(messages):
        role = message.get("role")
        content = _render_qwen35_message_content(message.get("content"))
        previous_role = messages[index - 1].get("role") if index else None
        next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
        if role == "tool":
            if seen_host_event:
                raise ValueError("tool responses cannot follow a host user event")
            if previous_role != "tool":
                parts.append(f"{_IM_START}user")
            parts.extend(("\n<tool_response>\n", content, "\n</tool_response>"))
            if next_role != "tool":
                parts.extend((_IM_END, "\n"))
        elif role == "user":
            seen_host_event = True
            parts.extend(
                (
                    f"{_IM_START}user\n",
                    content,
                    _IM_END,
                    "\n",
                )
            )
        else:
            raise ValueError(
                "environment messages after a tool call must have role tool or user"
            )
    parts.append(_NO_THINK_GENERATION_PREFIX)
    return "".join(parts)


class StrictQwen35XMLToolParser(ToolParser):
    """VERL parser replacement with deterministic, no-``eval`` semantics."""

    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list[Any] | None = None
    ) -> tuple[str, list[FunctionCall]]:
        text = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                responses_ids, skip_special_tokens=True
            ),
        )
        try:
            prefix, call = _parse_qwen35_xml_tool_call(text, tools or [])
        except ToolCallFormatError:
            return text, []
        return "", [call]


def _media_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def _grid_rows(value: Any, *, name: str) -> list[list[int]]:
    if value is None:
        return []
    rows = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(rows, list):
        raise RuntimeError(f"{name} must be a rank-2 tensor")
    normalized: list[list[int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise RuntimeError(f"{name}[{index}] must contain [T, H, W]")
        dimensions = [int(item) for item in row]
        if any(item <= 0 for item in dimensions):
            raise RuntimeError(f"{name}[{index}] contains a non-positive dimension")
        normalized.append(dimensions)
    return normalized


def _first_dimension(value: Any, *, name: str) -> int:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 1:
        raise RuntimeError(f"{name} must be a tensor with at least one dimension")
    return int(shape[0])


def _visual_token_projection(
    token_ids: Sequence[int],
    *,
    image_token_id: int | None,
    video_token_id: int | None,
) -> list[tuple[str, int]]:
    """Return ordered contiguous image/video pad runs, ignoring text BPE splits."""

    names = {
        int(token_id): name
        for name, token_id in (
            ("image", image_token_id),
            ("video", video_token_id),
        )
        if token_id is not None
    }
    result: list[tuple[str, int]] = []
    active_name: str | None = None
    active_length = 0
    for raw_token in token_ids:
        name = names.get(int(raw_token))
        if name == active_name and name is not None:
            active_length += 1
            continue
        if active_name is not None:
            result.append((active_name, active_length))
        active_name = name
        active_length = 1 if name is not None else 0
    if active_name is not None:
        result.append((active_name, active_length))
    return result


def _validate_raw_inference_media_cardinality(
    worker: AgentLoopWorker,
    agent_data: AgentData,
) -> None:
    """Require one raw Qwen visual placeholder per SGLang media item."""

    image_count = _media_count(agent_data.image_data)
    video_count = _media_count(agent_data.video_data)
    if worker.processor is None:
        if image_count or video_count:
            raise RuntimeError("SGLang received visual media without a processor")
        return

    token_ids = [int(token) for token in agent_data.prompt_ids]
    for name, media_count in (("image", image_count), ("video", video_count)):
        token_id = get_processor_token_id(worker.processor, name)
        if token_id is None:
            if media_count:
                raise RuntimeError(
                    f"Qwen3.5 processor exposes {name} media but no {name} token id"
                )
            continue
        token_count = token_ids.count(int(token_id))
        if token_count != media_count:
            raise RuntimeError(
                f"SGLang raw {name} placeholder/media mismatch: "
                f"tokens={token_count}, media={media_count}"
            )


def _validate_final_training_alignment(
    *,
    initial_prompt_length: int,
    training_prompt_ids: Sequence[int],
    response_mask: Sequence[int],
    response_logprobs: Sequence[float] | None,
    multi_modal_data: Mapping[str, Any] | None,
    routed_experts: Any,
) -> None:
    """Prove that the expanded training stream and token metadata align."""

    response_count = len(response_mask)
    expected_total = int(initial_prompt_length) + response_count
    if len(training_prompt_ids) != expected_total:
        raise AssertionError(
            "expanded training stream length mismatch: "
            f"tokens={len(training_prompt_ids)}, expected={expected_total}, "
            f"initial_prompt={initial_prompt_length}, response={response_count}"
        )
    if response_logprobs and len(response_logprobs) != response_count:
        raise AssertionError(
            "training response logprob/mask length mismatch: "
            f"logprobs={len(response_logprobs)}, mask={response_count}"
        )

    media = multi_modal_data or {}
    has_media = any(
        _media_count(media.get(name)) for name in ("images", "videos", "audios")
    )
    if has_media and routed_experts is not None:
        raise RuntimeError(
            "routed_experts are aligned to the raw SGLang stream and cannot be "
            "returned with processor-expanded multimodal training ids"
        )


def _validate_visual_alignment(
    worker: AgentLoopWorker,
    *,
    token_ids: list[int],
    multi_modal_data: Mapping[str, Any],
    multi_modal_inputs: Mapping[str, Any],
) -> None:
    """Prove that Qwen visual placeholders, grids and patch tensors agree."""

    processor = worker.processor
    image_count = _media_count(multi_modal_data.get("images"))
    video_count = _media_count(multi_modal_data.get("videos"))

    image_rows = _grid_rows(
        multi_modal_inputs.get("image_grid_thw"), name="image_grid_thw"
    )
    video_rows = _grid_rows(
        multi_modal_inputs.get("video_grid_thw"), name="video_grid_thw"
    )
    if len(image_rows) != image_count:
        raise RuntimeError(
            f"image count/grid mismatch: media={image_count}, grids={len(image_rows)}"
        )
    if len(video_rows) != video_count:
        raise RuntimeError(
            f"video count/grid mismatch: media={video_count}, grids={len(video_rows)}"
        )

    image_token_id = get_processor_token_id(processor, "image")
    video_token_id = get_processor_token_id(processor, "video")
    if image_count and image_token_id is None:
        raise RuntimeError("Qwen3.5 processor exposes images but no image token id")
    if video_count and video_token_id is None:
        raise RuntimeError("Qwen3.5 processor exposes videos but no video token id")

    image_merge = int(
        getattr(getattr(processor, "image_processor", None), "merge_size", 0)
    )
    video_merge = int(
        getattr(getattr(processor, "video_processor", None), "merge_size", 0)
    )
    if image_count and image_merge <= 0:
        raise RuntimeError("Qwen3.5 image_processor has no positive merge_size")
    if video_count and video_merge <= 0:
        raise RuntimeError("Qwen3.5 video_processor has no positive merge_size")

    expected_image_tokens = sum(
        (temporal * height * width) // (image_merge**2)
        for temporal, height, width in image_rows
    )
    expected_video_tokens = sum(
        (temporal * height * width) // (video_merge**2)
        for temporal, height, width in video_rows
    )
    actual_image_tokens = (
        token_ids.count(int(image_token_id)) if image_token_id is not None else 0
    )
    actual_video_tokens = (
        token_ids.count(int(video_token_id)) if video_token_id is not None else 0
    )
    if actual_image_tokens != expected_image_tokens:
        raise RuntimeError(
            "image placeholder/grid mismatch: "
            f"tokens={actual_image_tokens}, expected={expected_image_tokens}"
        )
    if actual_video_tokens != expected_video_tokens:
        raise RuntimeError(
            "video placeholder/grid mismatch: "
            f"tokens={actual_video_tokens}, expected={expected_video_tokens}"
        )

    if image_count:
        pixel_values = multi_modal_inputs.get("pixel_values")
        expected_patches = sum(t * h * w for t, h, w in image_rows)
        if pixel_values is None or _first_dimension(
            pixel_values, name="pixel_values"
        ) != expected_patches:
            raise RuntimeError("image patch tensor does not match image_grid_thw")
    if video_count:
        pixel_values = multi_modal_inputs.get("pixel_values_videos")
        expected_patches = sum(t * h * w for t, h, w in video_rows)
        if pixel_values is None or _first_dimension(
            pixel_values, name="pixel_values_videos"
        ) != expected_patches:
            raise RuntimeError("video patch tensor does not match video_grid_thw")


def _split_qwen35_video_grid_for_rope(video_grid_thw: Any) -> Any:
    """Split each Qwen3.5 video grid into its timestamp-delimited spans.

    Qwen3.5 serializes a ``[T, H, W]`` video as T distinct video-token runs,
    separated by timestamp text and vision boundary tokens.  ``get_rope_index``
    consumes one grid row per run, while the visual encoder must retain the
    original one-row-per-video grid.  This is the backport used by current
    Transformers main and is idempotent when every row already has T=1.
    """

    if video_grid_thw is None:
        return None
    import torch

    if not isinstance(video_grid_thw, torch.Tensor):
        raise RuntimeError("video_grid_thw must be a tensor before RoPE construction")
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
        raise RuntimeError("video_grid_thw must have shape [num_videos, 3]")
    temporal = video_grid_thw[:, 0]
    if torch.any(temporal <= 0):
        raise RuntimeError("video_grid_thw contains a non-positive temporal size")
    split = torch.repeat_interleave(video_grid_thw, temporal, dim=0)
    split[:, 0] = 1
    return split


def _video_token_run_lengths(
    mm_token_type_ids: Any,
    attention_mask: Any,
) -> list[int]:
    """Return Qwen3.5's timestamp-delimited video-token group lengths."""

    if mm_token_type_ids.ndim != 2 or attention_mask.shape != mm_token_type_ids.shape:
        raise RuntimeError("mm_token_type_ids and attention_mask must be aligned rank-2 tensors")
    lengths: list[int] = []
    for token_types, mask in zip(mm_token_type_ids, attention_mask, strict=True):
        active_length = 0
        for token_type in token_types[mask.bool()].tolist():
            if int(token_type) == 2:
                active_length += 1
            elif active_length:
                lengths.append(active_length)
                active_length = 0
        if active_length:
            lengths.append(active_length)
    return lengths


def _storymem_compute_position_ids(
    worker: AgentLoopWorker,
    input_ids: Any,
    attention_mask: Any,
    multi_modal_inputs: dict[str, Any],
    mm_processor_kwargs: Mapping[str, Any] | None = None,
    *,
    _ray_trace_ctx: Any = None,
) -> Any:
    """Backport Qwen3.5's per-timestamp video-grid RoPE handling."""

    del _ray_trace_ctx
    position_inputs = dict(multi_modal_inputs)
    video_grid_thw = position_inputs.get("video_grid_thw")
    mm_token_type_ids = position_inputs.get("mm_token_type_ids")
    if video_grid_thw is not None and mm_token_type_ids is not None:
        split_video_grid = _split_qwen35_video_grid_for_rope(video_grid_thw)
        merge_size = int(
            getattr(getattr(worker.processor, "video_processor", None), "merge_size", 0)
        )
        if merge_size <= 0:
            raise RuntimeError("Qwen3.5 video_processor has no positive merge_size")
        expected_run_lengths = [
            (int(height) * int(width)) // (merge_size**2)
            for _, height, width in split_video_grid.tolist()
        ]
        actual_run_lengths = _video_token_run_lengths(
            mm_token_type_ids, attention_mask
        )
        if actual_run_lengths != expected_run_lengths:
            raise RuntimeError(
                "Qwen3.5 video RoPE run/grid mismatch: "
                f"runs={actual_run_lengths!r}, expected={expected_run_lengths!r}"
            )
        position_inputs["video_grid_thw"] = split_video_grid

    # Stock VERL removes mm_token_type_ids after using it only for position-id
    # construction. Preserve that mutation while keeping the original visual
    # encoder video_grid_thw untouched in multi_modal_inputs.
    if "mm_token_type_ids" in multi_modal_inputs:
        multi_modal_inputs.pop("mm_token_type_ids")
    original = getattr(
        _storymem_compute_position_ids, "_storymem_original_position_method", None
    )
    if not callable(original):
        raise RuntimeError("original VERL position-id method was not registered")
    return original(
        worker,
        input_ids,
        attention_mask,
        position_inputs,
        mm_processor_kwargs,
    )


def _storymem_compute_multi_modal_inputs(
    worker: AgentLoopWorker,
    output: AgentLoopOutput,
    input_ids: Any,
    *,
    _ray_trace_ctx: Any = None,
) -> dict[str, Any]:
    """Current-VERL replacement that never deletes Qwen visual special tokens.

    VERL's stock implementation decodes padded ids with
    ``skip_special_tokens=True``.  That removes every ``<|image_pad|>`` and
    ``<|video_pad|>`` before rebuilding processor inputs.  We instead use the
    output's unpadded ids, preserve all specials, rebuild from the loop's exact
    pre-expansion processor text, and assert exact image/video run alignment.
    Ordinary text ids are deliberately excluded from processor reconstruction
    because BPE decode/encode is not guaranteed to preserve a model-sampled
    token segmentation.  The original padded ``input_ids`` are still required
    to construct Qwen3.5's sequence-aligned ``mm_token_type_ids``.
    """

    # Ray injects this private keyword-only argument into every actor method,
    # including this internal helper.  The method is called locally by VERL, so
    # there is no tracing context to consume after we replace Ray's wrapper, but
    # preserving the argument keeps the actor method signature compatible.
    del _ray_trace_ctx
    token_ids = [int(token) for token in (*output.prompt_ids, *output.response_ids)]
    # This decode is intentionally special-token preserving. It also makes a
    # missing visual token immediately inspectable in the raised diagnostics;
    # unlike VERL's stock path it never becomes processor input after specials
    # have been deleted.
    worker.tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    multi_modal_data = output.multi_modal_data or {}
    images = multi_modal_data.get("images")
    videos = multi_modal_data.get("videos")
    audios = multi_modal_data.get("audios")
    if worker.processor is None:
        if images is not None or videos is not None or audios is not None:
            raise RuntimeError("multimodal rollout requires the Qwen3.5 processor")
        return {}

    image_token_id = get_processor_token_id(worker.processor, "image")
    video_token_id = get_processor_token_id(worker.processor, "video")
    # transformers>=5.3 makes this a required argument of Qwen3VLModel's
    # get_rope_index, including for text-only samples.  VERL only forwards the
    # argument when this key exists, then rebuilds it from the full padded
    # input_ids.  Return the correctly aligned tensor ourselves as well so the
    # contract remains explicit and independently testable.
    import torch

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise RuntimeError(
            "Qwen3.5 multimodal postprocessing requires rank-2 tensor input_ids"
        )
    mm_token_type_ids = torch.zeros_like(input_ids)
    if image_token_id is not None:
        mm_token_type_ids.masked_fill_(input_ids == int(image_token_id), 1)
    if video_token_id is not None:
        mm_token_type_ids.masked_fill_(input_ids == int(video_token_id), 2)
    original_visual_projection = _visual_token_projection(
        token_ids,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
    )

    image_count = _media_count(images)
    video_count = _media_count(videos)
    audio_count = _media_count(audios)
    if audio_count:
        raise RuntimeError(
            "Qwen3.5-9B policy rollout cannot consume audio attachments; expose "
            "trusted audio metadata as text instead"
        )

    source_text = output.extra_fields.pop(
        "_storymem_unexpanded_processor_text", None
    )
    if not isinstance(source_text, str) or not source_text:
        raise RuntimeError(
            "v2 rollout output has no unexpanded Qwen3.5 processor text"
        )

    if image_count == 0 and video_count == 0:
        if original_visual_projection:
            raise RuntimeError(
                "rollout contains visual placeholder tokens but no visual media"
            )
        _validate_visual_alignment(
            worker,
            token_ids=token_ids,
            multi_modal_data={},
            multi_modal_inputs={"mm_token_type_ids": mm_token_type_ids},
        )
        return {"mm_token_type_ids": mm_token_type_ids}

    # The decoded rollout contains expanded pad-token runs. Feeding it back to
    # ProcessorMixin would consume one media object per expanded token and fail.
    # The loop therefore carries its exact pre-expansion processor text. This is
    # safer than heuristically collapsing runs (Qwen3VL video expansion may
    # contain one run per temporal patch): reprocessing must reproduce the
    # ordered visual-token projection exactly.
    processor_inputs = build_multimodal_processor_inputs(
        worker.processor,
        text=[source_text],
        images=images,
        videos=videos,
        audio=None,
        mm_processor_kwargs=(
            output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else worker._get_mm_processor_kwargs(None)
        ),
    )
    rebuilt_ids = normalize_token_ids(processor_inputs["input_ids"])
    rebuilt_visual_projection = _visual_token_projection(
        rebuilt_ids,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
    )
    if rebuilt_visual_projection != original_visual_projection:
        raise RuntimeError(
            "Qwen3.5 multimodal visual-token reconstruction differs from the "
            f"rollout: rebuilt={rebuilt_visual_projection!r}, "
            f"original={original_visual_projection!r}"
        )
    processor_inputs.pop("input_ids", None)
    processor_inputs.pop("attention_mask", None)
    multi_modal_inputs = dict(processor_inputs.convert_to_tensors("pt"))
    # Do not depend on a processor default for return_mm_token_type_ids: VERL's
    # next stage uses key presence to decide whether to pass the mandatory
    # Qwen3.5 argument to get_rope_index.
    multi_modal_inputs["mm_token_type_ids"] = mm_token_type_ids
    _validate_visual_alignment(
        worker,
        token_ids=token_ids,
        multi_modal_data=multi_modal_data,
        multi_modal_inputs=multi_modal_inputs,
    )

    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        multi_modal_inputs["images_seqlens"] = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
        )
    return multi_modal_inputs


def _install_current_verl_multimodal_fix() -> None:
    """Install the recipe's required postprocessor on the one supported VERL API."""

    current = AgentLoopWorker._compute_multi_modal_inputs
    if not getattr(current, "_storymem_qwen35_v2", False):
        signature = inspect.signature(current)
        parameters = tuple(signature.parameters)
        supported_parameters = (
            ("self", "output", "input_ids"),
            ("self", "output", "input_ids", "_ray_trace_ctx"),
        )
        if parameters not in supported_parameters:
            raise RuntimeError(
                "unsupported VERL AgentLoopWorker._compute_multi_modal_inputs "
                f"signature: {parameters!r}"
            )
        if "_ray_trace_ctx" in signature.parameters:
            trace_parameter = signature.parameters["_ray_trace_ctx"]
            if (
                trace_parameter.kind is not inspect.Parameter.KEYWORD_ONLY
                or trace_parameter.default is not None
                or not hasattr(current, "__wrapped__")
            ):
                raise RuntimeError(
                    "unsupported VERL AgentLoopWorker._compute_multi_modal_inputs "
                    f"trace parameter: {trace_parameter!r}"
                )
            unwrapped = inspect.unwrap(current)
            code = getattr(unwrapped, "__code__", None)
            if (
                code is None
                or code.co_argcount != 3
                or code.co_kwonlyargcount != 0
                or code.co_varnames[:3] != ("self", "output", "input_ids")
                or code.co_flags
                & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
            ):
                raise RuntimeError(
                    "Ray trace wrapper does not wrap the supported three-argument "
                    "VERL multimodal method"
                )
        if not callable(getattr(AgentLoopWorker, "_get_mm_processor_kwargs", None)):
            raise RuntimeError(
                "current VERL AgentLoopWorker lacks multimodal kwargs support"
            )
        _storymem_compute_multi_modal_inputs._storymem_qwen35_v2 = True
        AgentLoopWorker._compute_multi_modal_inputs = _storymem_compute_multi_modal_inputs

    current_position = AgentLoopWorker._compute_position_ids
    if getattr(current_position, "_storymem_qwen35_video_rope_v1", False):
        return
    position_signature = inspect.signature(current_position)
    position_parameters = tuple(position_signature.parameters)
    supported_position_parameters = (
        (
            "self",
            "input_ids",
            "attention_mask",
            "multi_modal_inputs",
            "mm_processor_kwargs",
        ),
        (
            "self",
            "input_ids",
            "attention_mask",
            "multi_modal_inputs",
            "mm_processor_kwargs",
            "_ray_trace_ctx",
        ),
    )
    if position_parameters not in supported_position_parameters:
        raise RuntimeError(
            "unsupported VERL AgentLoopWorker._compute_position_ids signature: "
            f"{position_parameters!r}"
        )
    original_position = inspect.unwrap(current_position)
    _storymem_compute_position_ids._storymem_original_position_method = (
        original_position
    )
    _storymem_compute_position_ids._storymem_qwen35_video_rope_v1 = True
    AgentLoopWorker._compute_position_ids = _storymem_compute_position_ids


_install_current_verl_multimodal_fix()


class StoryMemAgentLoopManager(AgentLoopManager):
    """Load this recipe before Ray snapshots and traces AgentLoopWorker."""

    @auto_await
    async def generate_sequences(self, prompts):
        if os.getenv("STORYMEM_RL_SINGLE_SHOT_ACCUMULATION", "0") != "1" or prompts.meta_info.get("validate", False):
            single_training = os.getenv("STORYMEM_RL_MODE") == "single" and not prompts.meta_info.get("validate", False)
            if single_training:
                prefix = (f"[storymem-parallel] step={prompts.meta_info.get('global_steps', '?')} "
                          f"groups={len(prompts) // int(self.rollout_config.n)} candidates={len(prompts)}")
                started = time.monotonic()
                print(f"{prefix} status=generating", flush=True)
            output = await super().generate_sequences(prompts)
            if single_training:
                if len(output) != len(prompts):
                    raise ValueError("single-shot parallel collection cannot expand or discard candidate rows")
                print(f"{prefix} status=completed seconds={time.monotonic() - started:.1f}", flush=True)
            return output
        if os.getenv("STORYMEM_RL_MODE") != "single":
            raise ValueError("single-shot accumulation cannot be enabled for multi-shot rollouts")
        from .rollout_accumulation import generate_accumulated_rollouts

        return await generate_accumulated_rollouts(self, prompts)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _env_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _env_nonnegative_float(
    name: str, default: float, *, maximum: float | None = None
) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _split_urls(raw: str) -> list[str]:
    urls = [part.rstrip("/") for part in re.split(r"[,\s]+", raw.strip()) if part]
    if not urls:
        raise ValueError("STORYMEM_RL_ECHO_URLS must contain at least one URL")
    if len(urls) != len(set(urls)):
        raise ValueError("STORYMEM_RL_ECHO_URLS contains duplicate endpoints")
    for url in urls:
        host = (urlparse(url).hostname or "").strip().lower()
        if host in _RETIRED_ECHO_HOSTS:
            raise ValueError(f"retired Echo endpoint is forbidden: {host}")
        if not host:
            raise ValueError(f"invalid Echo endpoint URL: {url!r}")
    return urls


def _safe_name(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return text[:120] or fallback


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_component_sha256(state: Mapping[str, Any]) -> dict[str, str]:
    """Hash only the three compact state components carried between shots."""

    return {
        name: _sha256_text(canonical_json(state[name]))
        for name in ("registry_items", "memory_bank", "previous_evidence")
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


class _RuntimeServices:
    """Provider clients shared by concurrent episodes in one loop worker."""

    def __init__(self, output_root: Path) -> None:
        if PROMPT_AUTHORING:
            # Match the Echo data pipeline's audited whitespace normalization.
            # V7 still checks the exact declarations, counts, order and content.
            from reel_echo15_output_spacing import install_echo15_output_spacing

            install_echo15_output_spacing()
        raw_urls = os.getenv("STORYMEM_RL_ECHO_URLS", "")
        if not raw_urls.strip():
            raise RolloutInfrastructureError(
                "STORYMEM_RL_ECHO_URLS is required; Echo must run outside training nodes"
            )
        echo_urls = _split_urls(raw_urls)
        width = _env_int("STORYMEM_RL_VIDEO_WIDTH", 1280)
        height = _env_int("STORYMEM_RL_VIDEO_HEIGHT", 736)
        num_frames = _env_int("STORYMEM_RL_NUM_FRAMES", 241)
        endpoint_concurrency = _env_int(
            "STORYMEM_RL_ECHO_ENDPOINT_CONCURRENCY", 1
        )
        echo_concurrency = _env_int(
            "STORYMEM_RL_ECHO_CONCURRENCY",
            len(echo_urls) * endpoint_concurrency,
        )
        image_concurrency = _env_int("STORYMEM_RL_IMAGE_CONCURRENCY", 8)
        config = CollectorConfig(
            output_root=output_root,
            prompt_authoring=PROMPT_AUTHORING,
            width=width,
            height=height,
            num_frames=num_frames,
            story_workers=1,
            echo_concurrency=echo_concurrency,
            image_concurrency=image_concurrency,
            max_turns_per_shot=_env_int("STORYMEM_RL_MAX_TOOL_TURNS", 48),
            rewrite_fraction=_env_float("STORYMEM_RL_PE_REWRITE_FRACTION", 0.5),
        )
        image_adapter = os.getenv("STORYMEM_RL_IMAGE_API_STYLE", "seedream").strip()
        if image_adapter != "seedream":
            raise RolloutInfrastructureError(
                "STORYMEM_RL_IMAGE_API_STYLE must be seedream"
            )
        image_url = os.getenv(
            "STORYMEM_RL_IMAGE_URL",
            "https://modelservice.jdcloud.com/v1/imageEdit/generations",
        ).strip()
        image_model = os.getenv(
            "STORYMEM_RL_IMAGE_MODEL", "doubao-seedream-4-5-251128"
        ).strip()
        if not image_url:
            raise RolloutInfrastructureError(
                "the image-generation endpoint is empty"
            )
        if not image_model:
            raise RolloutInfrastructureError(
                "STORYMEM_RL_IMAGE_MODEL must not be empty"
            )
        try:
            image_key = private_api_key("STORYMEM_RL_IMAGE_API_KEY")
        except ProviderConfigError as exc:
            raise RolloutInfrastructureError(str(exc)) from exc
        image_provider = GenericImageProvider(
            api_url=image_url,
            api_key=image_key,
            model=image_model,
            adapter=image_adapter,
        )
        visual_creator = ProviderVisualEvidenceCreator(
            image_provider,
            width=width,
            height=height,
            semaphore=threading.BoundedSemaphore(image_concurrency),
        )
        echo = Echo15WireTransport(
            echo_urls,
            concurrency=echo_concurrency,
            per_endpoint_concurrency=endpoint_concurrency,
            generation_timeout=float(
                os.getenv("STORYMEM_RL_ECHO_TIMEOUT_SECONDS", "7200")
            ),
            request_timeout=float(
                os.getenv("STORYMEM_RL_ECHO_REQUEST_TIMEOUT_SECONDS", "120")
            ),
            poll_seconds=float(
                os.getenv("STORYMEM_RL_ECHO_POLL_SECONDS", "5")
            ),
            http_attempts=_env_int("STORYMEM_RL_ECHO_HTTP_ATTEMPTS", 3),
            allow_seed=PROMPT_AUTHORING,
        )
        if not DIRECT_PROMPT:
            pe_url = os.getenv(
                "STORYMEM_RL_GEMINI_URL",
                "https://modelservice.jdcloud.com/v1/responses",
            ).strip()
            pe_model = os.getenv(
                "STORYMEM_RL_GEMINI_MODEL", "Gemini-3.5-Flash-Third"
            ).strip()
            pe_style = os.getenv("STORYMEM_RL_GEMINI_API_STYLE", "gemini").strip()
            if pe_style != "gemini":
                raise RolloutInfrastructureError(
                    "STORYMEM_RL_GEMINI_API_STYLE must be gemini"
                )
            if not pe_url or not pe_model:
                raise RolloutInfrastructureError(
                    "STORYMEM_RL_GEMINI_URL and STORYMEM_RL_GEMINI_MODEL must not be empty"
                )
            try:
                pe_key = private_api_key("STORYMEM_RL_GEMINI_API_KEY")
            except ProviderConfigError as exc:
                raise RolloutInfrastructureError(str(exc)) from exc
            rate_limit_state_raw = os.getenv(
                "STORYMEM_RL_GEMINI_RATE_LIMIT_STATE_PATH", ""
            ).strip()
            rate_limit_state_path = Path(rate_limit_state_raw).expanduser()
            if not rate_limit_state_raw or not rate_limit_state_path.is_absolute():
                raise RolloutInfrastructureError(
                    "STORYMEM_RL_GEMINI_RATE_LIMIT_STATE_PATH must be an absolute "
                    "path shared by prompt engineering and reward review"
                )
            gemini_qps = _env_positive_float("STORYMEM_RL_GEMINI_QPS", 2.5)
            gemini_cooldown = _env_nonnegative_float(
                "STORYMEM_RL_GEMINI_429_COOLDOWN_SECONDS", 75.0
            )
            pe_options = {
                "api_key": pe_key,
                "timeout": _env_positive_float("STORYMEM_RL_GEMINI_TIMEOUT_SECONDS", 120.0),
                "max_attempts": _env_int("STORYMEM_RL_GEMINI_MAX_RETRIES", 3),
                "rate_limiter": CrossProcessRateLimiter(
                    rate_limit_state_path.resolve(),
                    qps=gemini_qps,
                    cooldown_seconds=gemini_cooldown,
                ),
            }
            from . import gemini_reward as reward_provider
            from .gemini_alias_pool import TextAliasPool, get_selector, load_model_aliases, validate_pool_approval
            from .gemini_transport_recovery import recovery_settings
    
            if load_model_aliases(pe_model) or recovery_settings()[0]:
                pool_config = reward_provider._load_config()
                approval = validate_pool_approval(pool_config) if pool_config.model_aliases else None
                clients = {model: IsolatedGeminiTextClient(pe_url, model=model, **{**pe_options, "max_attempts": 1})
                           for model in (pool_config.model_aliases or (pe_model,))}
                pe_client = TextAliasPool(clients, get_selector(pool_config, approval),
                                          max_attempts=pe_options["max_attempts"])
            else:
                pe_client = IsolatedGeminiTextClient(pe_url, model=pe_model, **pe_options)
        namespace_material = (
            f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}:{uuid.uuid4().hex}"
        )
        run_namespace = "storymem_run_" + hashlib.sha256(
            namespace_material.encode("utf-8")
        ).hexdigest()[:24]
        adapter_factory = (Echo15BackendAdapter.from_policy_direct if DIRECT_PROMPT
                           else Echo15BackendAdapter.from_official_v7)
        adapter = adapter_factory(
            artifact_root=output_root / "adapter_artifacts" / run_namespace,
            **({} if DIRECT_PROMPT else {"gemini": pe_client}),
            capabilities=Echo15Capabilities(max_memory_slots=7, audio_memory=True),
        )
        print(
            "[reel-echo-budget] initial_caption_max_tokens=1024 "
            "tokenizer=Echo-Gemma "
            f"oversize_recovery={'qwen_revision' if DIRECT_PROMPT else 'fresh_original_input'} "
            "pe_compression=disabled",
            flush=True,
        )
        print(f"[reel-prompt-mode] mode={'policy_direct' if DIRECT_PROMPT else 'gemini_rewrite'} "
              f"pe_provider={'disabled' if DIRECT_PROMPT else pe_model}", flush=True)
        self.config = config
        self.collector = StoryCollector(
            config=config,
            teacher_client=None,
            visual_evidence_creator=visual_creator,
            backend_adapter=adapter,
            echo=echo,
            finalizer=FFmpegFinalEvidence(),
            run_namespace=run_namespace,
        )
        self.run_namespace = run_namespace


# AgentLoopManager constructs a loop object for every sampled rollout. Provider
# clients and concurrency semaphores must instead be process-wide, otherwise a
# batch of N rollouts silently creates N independent Echo capacity pools.
_RUNTIME_SERVICES_LOCK = threading.Lock()
_RUNTIME_SERVICES: dict[tuple[str, ...], _RuntimeServices] = {}
_RUNTIME_SERVICE_ENV_KEYS = (
    "STORYMEM_RL_PE_MODE",
    "STORYMEM_RL_ECHO_URLS",
    "STORYMEM_RL_ECHO_CONCURRENCY",
    "STORYMEM_RL_ECHO_ENDPOINT_CONCURRENCY",
    "STORYMEM_RL_IMAGE_CONCURRENCY",
    "STORYMEM_RL_VIDEO_WIDTH",
    "STORYMEM_RL_VIDEO_HEIGHT",
    "STORYMEM_RL_NUM_FRAMES",
    "STORYMEM_RL_MAX_TOOL_TURNS",
    "STORYMEM_RL_PE_REWRITE_FRACTION",
    "STORYMEM_RL_ECHO_TIMEOUT_SECONDS",
    "STORYMEM_RL_ECHO_REQUEST_TIMEOUT_SECONDS",
    "STORYMEM_RL_ECHO_POLL_SECONDS",
    "STORYMEM_RL_ECHO_HTTP_ATTEMPTS",
    "STORYMEM_RL_IMAGE_URL",
    "STORYMEM_RL_IMAGE_MODEL",
    "STORYMEM_RL_IMAGE_API_STYLE",
    "STORYMEM_RL_GEMINI_URL",
    "STORYMEM_RL_GEMINI_MODEL",
    "STORYMEM_RL_GEMINI_API_STYLE",
    "STORYMEM_RL_GEMINI_TIMEOUT_SECONDS",
    "STORYMEM_RL_GEMINI_MAX_RETRIES",
    "STORYMEM_RL_GEMINI_CONCURRENCY",
    "STORYMEM_RL_GEMINI_MODEL_ALIASES",
    "STORYMEM_RL_GEMINI_ALIAS_APPROVAL_PATH",
    "STORYMEM_RL_GEMINI_ALIAS_APPROVAL_SHA256",
    "STORYMEM_RL_GEMINI_RECOVER_TRANSIENT",
    "STORYMEM_RL_GEMINI_RECOVERY_INITIAL_SECONDS",
    "STORYMEM_RL_GEMINI_RECOVERY_MAX_SECONDS",
)


def _runtime_services_key(output_root: Path) -> tuple[str, ...]:
    return (
        str(output_root),
        *(f"{name}={os.getenv(name, '')}" for name in _RUNTIME_SERVICE_ENV_KEYS),
    )


def _decode_state_seed(spec: Mapping[str, Any]) -> dict[str, Any]:
    encoded = spec.get("state_seed")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("shot spec has no canonical state_seed")
    expected = spec.get("state_seed_sha256")
    if expected != _sha256_text(encoded):
        raise ValueError("shot spec state_seed hash mismatch")
    value = _strict_json_loads(encoded, label="state_seed")
    if not isinstance(value, dict):
        raise ValueError("state_seed must decode to an object")
    if value.get("schema_version") != STATE_SEED_SCHEMA_VERSION:
        raise ValueError("state_seed is not the current v2 schema")
    if (value.get("prompt_authoring") is True) != PROMPT_AUTHORING:
        raise ValueError("state_seed policy profile mismatch")
    return value


def _previous_evidence_from_final_bundle(host: Mapping[str, Any]) -> dict[str, Any]:
    bundle = host.get("final_bundle")
    if not isinstance(bundle, Mapping):
        return {}
    rows = [
        row
        for row in (bundle.get("derived_evidence") or [])
        if isinstance(row, Mapping)
        and isinstance(row.get("selector"), Mapping)
        and row["selector"].get("mode") == "tail_frame"
    ]
    if len(rows) != 1:
        return {}
    row = copy.deepcopy(dict(rows[0]))
    representation = row.get("representation")
    evidence_id = str(row.get("evidence_id") or "")
    if not evidence_id or not isinstance(representation, Mapping):
        return {}
    representation_id = str(representation.get("representation_id") or "")
    if not representation_id:
        return {}
    return {
        evidence_id: {
            "evidence_id": evidence_id,
            "source_kind": "sequence_context",
            "modality": "image",
            "description": str(
                row.get("description") or "Accepted previous-shot tail frame."
            ),
            "semantic_roles": copy.deepcopy(
                row.get("semantic_roles") or ["continuity"]
            ),
            "subject_ids": copy.deepcopy(row.get("subject_ids") or []),
            "default_representation_id": representation_id,
            "representations": {
                representation_id: copy.deepcopy(dict(representation))
            },
        }
    }


@register(AGENT_NAME)
class StoryMemMemoryPolicyOnlineAgentLoop(ToolAgentLoop):
    """Single- and multi-shot online rollout with fresh context per shot."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.processor is None or type(self.processor).__name__ != "Qwen3VLProcessor":
            raise ValueError(
                "Memory Policy RL requires the Qwen3.5 Qwen3VLProcessor; "
                "the configured environment/model cannot run this recipe"
            )
        if self.apply_chat_template_kwargs.get("enable_thinking") is not False:
            raise ValueError(
                "Memory Policy RL requires data.apply_chat_template_kwargs."
                "enable_thinking=false so every sampled turn begins at the native "
                "Qwen3.5 XML function-call boundary"
            )
        if self.apply_chat_template_kwargs.get("add_vision_id", False) is not False:
            raise ValueError(
                "Memory Policy RL requires add_vision_id=false; incremental media "
                "placeholders are serialized by the current Qwen3.5 contract"
            )
        if getattr(self.tokenizer, "eos_token", None) != _IM_END:
            raise ValueError(
                f"Qwen3.5 eos_token must be {_IM_END!r}, got "
                f"{getattr(self.tokenizer, 'eos_token', None)!r}"
            )
        if self.tool_parser_name != "qwen3_coder":
            raise ValueError(
                "Qwen3.5 Memory Policy RL requires multi_turn.format=qwen3_coder"
            )
        if tuple(self.tools) != CANONICAL_TOOL_NAMES:
            raise ValueError(
                "tool_config must expose exactly the canonical nine tools in order"
            )
        if [row["function"]["name"] for row in self.tool_schemas] != list(
            CANONICAL_TOOL_NAMES
        ):
            raise ValueError("VERL tool schema order differs from the canonical contract")
        self.tool_parser = StrictQwen35XMLToolParser(self.tokenizer)
        self.max_runtime_rejections = _env_int(
            "STORYMEM_RL_MAX_RUNTIME_REJECTIONS", 6, minimum=0
        )
        self.max_format_rejections = _env_int(
            "STORYMEM_RL_MAX_FORMAT_REJECTIONS", 3, minimum=0
        )
        self.max_infrastructure_attempts = _env_int(
            "STORYMEM_RL_INFRA_MAX_ATTEMPTS", 3
        )
        self.infrastructure_retry_backoff_seconds = _env_nonnegative_float(
            "STORYMEM_RL_INFRA_RETRY_BACKOFF_SECONDS", 1.0, maximum=60.0
        )
        self.output_root = Path(
            os.getenv(
                "STORYMEM_RL_OUTPUT_ROOT",
                str(Path(__file__).resolve().parents[2] / "output" / "rl"),
            )
        ).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _get_services(self) -> _RuntimeServices:
        key = _runtime_services_key(self.output_root)
        with _RUNTIME_SERVICES_LOCK:
            services = _RUNTIME_SERVICES.get(key)
            if services is None:
                services = _RuntimeServices(self.output_root)
                _RUNTIME_SERVICES[key] = services
            return services

    @staticmethod
    def _non_empty(values: Any) -> list[Any] | None:
        if values is None:
            return None
        if isinstance(values, list):
            return values or None
        return [values]

    @staticmethod
    def _multi_modal_data(agent_data: AgentData) -> dict[str, Any] | None:
        result: dict[str, Any] = {}
        if agent_data.image_data:
            result["images"] = agent_data.image_data
        if agent_data.video_data:
            result["videos"] = agent_data.video_data
        if agent_data.audio_data:
            result["audios"] = agent_data.audio_data
        return result or None

    def _set_all_tools(self, agent_data: AgentData) -> None:
        # The SFT template exposes all nine canonical schemas.  Runtime phase
        # rules remain authoritative and return deterministic rejections.
        agent_data._active_tools = self.tools
        agent_data._active_tool_schemas = self.tool_schemas

    async def _encode_initial_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        declared_token_count: int,
    ) -> tuple[list[int], str]:
        """Render the real shot prompt once and verify the v2 length proof."""

        rendered = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: render_chat_template(
                self.processor,
                [copy.deepcopy(dict(message)) for message in messages],
                tools=self.tool_schemas,
                tokenize=False,
                add_generation_prompt=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        model_inputs = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: build_multimodal_processor_inputs(
                self.processor,
                text=[rendered],
                images=None,
                videos=None,
                audio=None,
                mm_processor_kwargs=self._get_mm_processor_kwargs(None),
            ),
        )
        prompt_ids = normalize_token_ids(model_inputs["input_ids"])
        if len(prompt_ids) != declared_token_count:
            raise ValueError(
                "shot initial_prompt_token_count does not match the active Qwen3.5 "
                f"processor/template: declared={declared_token_count}, actual={len(prompt_ids)}"
            )
        if len(prompt_ids) > self.prompt_length:
            raise ValueError(
                f"shot prompt has {len(prompt_ids)} tokens but rollout.prompt_length="
                f"{self.prompt_length}; v2 prompts are never truncated"
            )
        return prompt_ids, rendered

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        _validate_raw_inference_media_cardinality(self, agent_data)
        # Never let a previous turn's parsed call survive a length/turn-limit
        # return from VERL before the current response has been parsed.
        agent_data.tool_calls = []
        state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination=ignore_termination
        )
        # SGLang must receive one unexpanded visual placeholder per media item
        # on every turn.  Training, however, needs the processor-expanded
        # image/video token runs.  Model-produced ids are identical in both
        # streams, so mirror the new assistant span into the training stream.
        agent_data._training_prompt_ids.extend(agent_data.response_ids)
        generated_wire = self.tokenizer.decode(
            agent_data.response_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        agent_data._processor_text_fragments.append(generated_wire)
        text = self.tokenizer.decode(
            agent_data.response_ids, skip_special_tokens=True
        )
        try:
            parse_qwen35_xml_tool_call(text, self.tool_schemas)
        except ToolCallFormatError as exc:
            return await self._reject_malformed_tool_call(
                agent_data, exc, generated_wire=generated_wire
            )
        if state != AgentState.PROCESSING_TOOLS:
            agent_data._protocol_error = (
                "valid tool call could not be processed before a rollout length/turn limit"
            )
            return AgentState.TERMINATED
        if len(agent_data.tool_calls) != 1:
            return await self._reject_malformed_tool_call(
                agent_data,
                ToolCallFormatError("assistant must emit exactly one tool call"),
            )
        return state

    @staticmethod
    def _safe_format_error_message(exc: ToolCallFormatError) -> str:
        """Return a bounded one-line diagnostic safe for model-visible audit."""

        message = " ".join(str(exc).split())
        return (message or "invalid tool call")[:512]

    def _format_retry_has_turn_budget(self, agent_data: AgentData) -> bool:
        # ToolAgentLoop applies both limits immediately after the next sampled
        # assistant response and before parsing it.  Keep one slot strictly
        # below each configured limit so a corrected call can be processed.
        if self.max_assistant_turns and (
            agent_data.assistant_turns + 1 >= self.max_assistant_turns
        ):
            return False
        if self.max_user_turns and agent_data.user_turns + 1 >= self.max_user_turns:
            return False
        return True

    async def _reject_malformed_tool_call(
        self,
        agent_data: AgentData,
        exc: ToolCallFormatError,
        *,
        generated_wire: str | None = None,
    ) -> AgentState:
        """Expose one malformed action as a deterministic host observation.

        The sampled action remains an actor token span.  The trusted correction
        is appended as an environment span, so its tokens are masked out of the
        policy loss.  No canonical tool is executed for this rejected action.
        """

        rejections = getattr(agent_data, "_format_rejections", None)
        if rejections is None:
            rejections = []
            agent_data._format_rejections = rejections
        if generated_wire is None:
            generated_wire = self.tokenizer.decode(
                agent_data.response_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        rejection_index = len(rejections) + 1
        within_retry_budget = rejection_index <= self.max_format_rejections
        retry_allowed = within_retry_budget and self._format_retry_has_turn_budget(
            agent_data
        )
        record = {
            "rejection_index": rejection_index,
            "assistant_turn": int(agent_data.assistant_turns),
            "error_type": type(exc).__name__,
            "message": self._safe_format_error_message(exc),
            "generated_wire_sha256": _sha256_text(generated_wire),
            "generated_wire_excerpt": generated_wire[:2048],
            "retry_allowed": retry_allowed,
            "remaining_retries": max(
                0, self.max_format_rejections - rejection_index
            ),
        }
        rejections.append(record)

        if not retry_allowed:
            if not within_retry_budget:
                reason = "format rejection budget exhausted"
            else:
                reason = "rollout turn budget cannot accommodate a format retry"
            agent_data._protocol_error = f"{reason}: {record['message']}"
            return AgentState.TERMINATED

        envelope = {
            "trusted_runtime_observation": {
                "name": "tool_call_format_rejection",
                "result": {
                    "ok": False,
                    "error": "invalid_tool_call_format",
                    "error_type": record["error_type"],
                    "message": record["message"],
                    "assistant_turn": record["assistant_turn"],
                    "retry_allowed": record["retry_allowed"],
                    "remaining_retries": record["remaining_retries"],
                    "repair_contract": _format_repair_contract(
                        exc, self.tool_schemas
                    ),
                },
            }
        }
        appended = await self._append_environment_messages(
            agent_data,
            messages_and_attachments=(
                (
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": canonical_json(envelope)}
                        ],
                    },
                    (),
                ),
            ),
            require_preceding_tool_call=False,
        )
        if not appended:
            record["retry_allowed"] = False
            if not getattr(agent_data, "_protocol_error", None):
                agent_data._protocol_error = (
                    "tool-call format rejection observation exceeded response_length"
                )
            return AgentState.TERMINATED
        # The strict parser deliberately left this empty.  Clear it explicitly
        # as a guard against future parser implementations retaining partial
        # calls from malformed XML.
        agent_data.tool_calls = []
        return AgentState.GENERATING

    async def _call_tool_exact(
        self, tool_call: FunctionCall, agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        if tool_call.name not in self.tools:
            raise ToolCallFormatError(f"unknown tool {tool_call.name!r}")
        arguments = _strict_json_loads(tool_call.arguments, label="tool arguments")
        if not isinstance(arguments, dict):
            raise ToolCallFormatError("tool arguments must be an object")
        tool = self.tools[tool_call.name]
        instance_id, _ = await tool.create()
        try:
            return await tool.execute(instance_id, arguments, agent_data=agent_data)
        finally:
            await tool.release(instance_id)

    async def _append_environment_messages(
        self,
        agent_data: AgentData,
        *,
        messages_and_attachments: Sequence[
            tuple[dict[str, Any], Sequence[Mapping[str, Any]]]
        ],
        require_preceding_tool_call: bool = True,
    ) -> bool:
        messages: list[dict[str, Any]] = []
        for raw_message, attachments in messages_and_attachments:
            message = copy.deepcopy(raw_message)
            content = message.get("content")
            if not isinstance(content, list):
                raise TypeError("environment message content must be an array")
            for attachment in attachments:
                media_type = str(attachment.get("type") or "")
                path = str(attachment.get("path") or "")
                if media_type not in {"image", "video", "audio"} or not path:
                    raise RolloutInfrastructureError(
                        "runtime returned an invalid media attachment"
                    )
                if not Path(path).is_file():
                    raise RolloutInfrastructureError(
                        f"runtime media attachment is missing: {path}"
                    )
                content.append({"type": media_type, media_type: path})
            messages.append(message)
        if not messages:
            raise ValueError("environment message batch must not be empty")
        new_media = await self.process_multi_modal_info(messages)
        images = self._non_empty(new_media.get("images"))
        videos = self._non_empty(new_media.get("videos"))
        audios = self._non_empty(new_media.get("audios"))
        if audios:
            raise RolloutInfrastructureError(
                "Qwen3.5-9B cannot consume audio attachments; the runtime must "
                "return trusted audio metadata as text"
            )
        if require_preceding_tool_call and len(agent_data.tool_calls) != 1:
            raise ToolCallFormatError(
                "environment responses require exactly one preceding tool call"
            )
        fragment = render_qwen35_environment_fragment(messages)
        eos_token = _IM_END
        generated_wire = self.tokenizer.decode(
            agent_data.response_ids, skip_special_tokens=False
        ).rstrip()
        if eos_token and generated_wire.endswith(eos_token):
            if fragment.startswith(eos_token):
                fragment = fragment[len(eos_token) :]
                if fragment.startswith("\n"):
                    fragment = fragment[1:]
        if self.processor is not None:
            model_inputs = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: build_multimodal_processor_inputs(
                    self.processor,
                    text=[fragment],
                    images=images,
                    videos=videos,
                    audio=audios,
                    mm_processor_kwargs=self._get_mm_processor_kwargs(audios),
                ),
            )
            training_response_ids = normalize_token_ids(
                model_inputs.pop("input_ids")
            )
            # Feeding processor-expanded Qwen3.5 video ids back to SGLang is
            # invalid: one logical video expands into one visual span per
            # sampled frame, while video_data still contains one item.  The
            # server owns expansion, so its history keeps the raw placeholder.
            inference_response_ids = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.tokenizer.encode(fragment, add_special_tokens=False),
            )
            inference_response_ids = normalize_token_ids(inference_response_ids)
        else:
            inference_response_ids = normalize_token_ids(
                self.tokenizer.encode(fragment, add_special_tokens=False)
            )
            training_response_ids = inference_response_ids
        if (
            len(agent_data.response_mask) + len(training_response_ids)
            >= self.response_length
        ):
            agent_data._protocol_error = "tool/host observation exceeded response_length"
            return False
        agent_data._processor_text_fragments.append(fragment)
        agent_data.messages.extend(messages)
        if images:
            agent_data.image_data = list(agent_data.image_data or []) + list(images)
        if videos:
            agent_data.video_data = list(agent_data.video_data or []) + list(videos)
        if audios:
            agent_data.audio_data = list(agent_data.audio_data or []) + list(audios)
        agent_data.prompt_ids += inference_response_ids
        agent_data._training_prompt_ids += training_response_ids
        agent_data.response_mask += [0] * len(training_response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(training_response_ids)
        agent_data.user_turns += 1
        return True

    async def _handle_processing_tools_state(
        self, agent_data: AgentData
    ) -> AgentState:
        if len(agent_data.tool_calls) != 1:
            return await self._reject_malformed_tool_call(
                agent_data,
                ToolCallFormatError("parallel or empty tool calls are forbidden"),
            )
        try:
            response, tool_reward, private = await self._call_tool_exact(
                agent_data.tool_calls[0], agent_data
            )
        except ToolCallFormatError as exc:
            return await self._reject_malformed_tool_call(agent_data, exc)
        attachments = private.get("_media_attachments") or []
        if not isinstance(attachments, list):
            raise RolloutInfrastructureError("tool attachment payload is malformed")
        if tool_reward is not None:
            agent_data.tool_rewards.append(float(tool_reward))
        session = agent_data._memory_policy_session
        message_batch: list[
            tuple[dict[str, Any], Sequence[Mapping[str, Any]]]
        ] = [
            (
                {
                    "role": "tool",
                    "content": [
                        {"type": "text", "text": str(response.text or "")}
                    ],
                },
                attachments,
            )
        ]
        if session.infrastructure_failures:
            await self._append_environment_messages(
                agent_data, messages_and_attachments=message_batch
            )
            return AgentState.TERMINATED
        runtime = session.runtime
        if runtime.phase in {"awaiting_generation", "awaiting_final_evidence"}:
            host_message = await self._drive_host_phase(
                services=agent_data._memory_policy_host_context["services"],
                agent_data=agent_data,
                host=agent_data._memory_policy_host_context["host"],
                shot_dir=agent_data._memory_policy_host_context["shot_dir"],
                story_id=agent_data._memory_policy_host_context["story_id"],
                shot_index=agent_data._memory_policy_host_context["shot_index"],
                sequence_index=agent_data._memory_policy_host_context[
                    "sequence_index"
                ],
                requirement=agent_data._memory_policy_host_context["requirement"],
                pe_mode=agent_data._memory_policy_host_context["pe_mode"],
            )
            if host_message is not None:
                message_batch.append((host_message, ()))
        appended = await self._append_environment_messages(
            agent_data, messages_and_attachments=message_batch
        )
        if not appended or session.infrastructure_failures:
            return AgentState.TERMINATED
        rejected = sum(
            1
            for row in session.call_log
            if isinstance(row.get("result"), Mapping)
            and row["result"].get("ok") is False
        )
        if rejected > self.max_runtime_rejections:
            agent_data._protocol_error = "runtime rejection budget exhausted"
            return AgentState.TERMINATED
        return AgentState.GENERATING

    def _create_runtime(
        self,
        *,
        services: _RuntimeServices,
        episode_dir: Path,
        story_id: str,
        episode_id: str,
        shot_spec: Mapping[str, Any],
        state_seed: Mapping[str, Any],
    ) -> tuple[MemoryPolicyRuntime, dict[str, Any], Path, str]:
        shot_index = int(shot_spec["shot_index"])
        shot_dir = episode_dir / "shots" / f"shot_{shot_index:04d}"
        shot_dir.mkdir(parents=True, exist_ok=False)
        requirement = str(shot_spec["shot_requirement"])
        session_id = (
            f"{services.run_namespace}:{_safe_name(story_id, 'story')}:"
            f"{_safe_name(episode_id, 'episode')}"
        )
        request_id = f"{session_id}:shot:{shot_index:04d}"
        state = MemoryPolicyState.create(
            shot_dir / "policy_state.json",
            session_id=session_id,
            request_id=request_id,
            shot_requirement=requirement,
            condition_budget=copy.deepcopy(dict(state_seed["condition_budget"])),
            registry_items=copy.deepcopy(dict(state_seed["registry_items"])),
            memory_bank=copy.deepcopy(dict(state_seed["memory_bank"])),
            uploaded_evidence=copy.deepcopy(dict(state_seed["uploaded_evidence"])),
            previous_evidence=copy.deepcopy(dict(state_seed["previous_evidence"])),
            visual_creation_budget=int(state_seed["visual_creation_budget"]),
            max_condition_revisions=int(state_seed["max_condition_revisions"]),
            perceivable_modalities=tuple(state_seed["perceivable_modalities"]),
            max_image_reference_inputs=int(state_seed["max_image_reference_inputs"]),
            prompt_authoring=PROMPT_AUTHORING,
        )
        state.data["source_story_id"] = str(story_id)
        state.data["run_namespace"] = services.run_namespace
        floor = int(state_seed.get("memory_counter_floor") or 0)
        if floor:
            state.data["counters"]["memory"] = floor
        state.save()
        if PROMPT_AUTHORING:
            (shot_dir / "initial_policy_state.json").write_text(
                canonical_json(state.data), encoding="utf-8"
            )
        runtime = MemoryPolicyRuntime(
            state=state,
            visual_evidence_creator=services.collector.visual_evidence_creator,
            output_dir=shot_dir / "policy_runtime",
        )
        pe_mode_env = pe_mode_for_profile(os.getenv("STORYMEM_RL_PE_MODE", "auto").strip())
        if pe_mode_env == "auto":
            pe_mode = assign_pe_mode(
                str(story_id),
                rewrite_fraction=services.config.rewrite_fraction,
            )
        elif pe_mode_env in {"gemini_rewrite", "passthrough", "policy_direct"}:
            pe_mode = pe_mode_env
        else:
            raise ValueError(
                "STORYMEM_RL_PE_MODE must be auto, gemini_rewrite, or passthrough"
            )
        host = services.collector._create_or_load_host(
            shot_dir=shot_dir,
            raw_prompt=requirement,
            pe_mode=pe_mode,
        )
        return runtime, host, shot_dir, pe_mode

    async def _drive_host_phase(
        self,
        *,
        services: _RuntimeServices,
        agent_data: AgentData,
        host: dict[str, Any],
        shot_dir: Path,
        story_id: str,
        shot_index: int,
        sequence_index: int,
        requirement: str,
        pe_mode: str,
    ) -> dict[str, Any] | None:
        runtime = agent_data._memory_policy_session.runtime
        before_count = len(runtime.state.data.get("host_events") or [])
        try:
            if runtime.phase == "awaiting_generation":
                await asyncio.to_thread(
                    services.collector._generation_step,
                    runtime=runtime,
                    host=host,
                    shot_dir=shot_dir,
                    story_id=story_id,
                    shot_index=shot_index,
                    sequence_index=sequence_index,
                    raw_prompt=requirement,
                    pe_mode=pe_mode,
                )
            elif runtime.phase == "awaiting_final_evidence":
                await asyncio.to_thread(
                    services.collector._final_evidence_step,
                    runtime=runtime,
                    host=host,
                    shot_dir=shot_dir,
                )
            else:
                return None
        except Exception as exc:
            frames = [
                {
                    "file": str(Path(frame.filename).resolve()),
                    "line": int(frame.lineno),
                    "function": str(frame.name),
                }
                for frame in traceback.extract_tb(exc.__traceback__)
            ]
            failure = {
                "tool": "host_runtime",
                "phase": str(runtime.phase),
                "type": type(exc).__name__,
                "message": str(exc)[:4096],
                "traceback": frames,
            }
            # Retain only bounded diagnostics from PE attempts, never provider
            # request bodies or rejected drafts, so retries are visible live.
            attempts = getattr(exc, "attempts", ()) or ()
            failure["pe_attempts"] = [
                {
                    "attempt": row.get("attempt"),
                    "outcome": row.get("outcome"),
                    "output_character_count": row.get("output_character_count"),
                    "violation_codes": (row.get("validation") or {}).get("violation_codes", []),
                }
                for row in attempts if isinstance(row, Mapping)
            ]
            agent_data._memory_policy_session.infrastructure_failures.append(failure)
            print("[reel-host-failure] " + json.dumps({
                "phase": failure["phase"], "type": failure["type"],
                "shot_dir": str(shot_dir), "pe_attempts": failure["pe_attempts"],
            }, ensure_ascii=False), flush=True)
            return None
        events = runtime.state.data.get("host_events") or []
        if len(events) != before_count + 1:
            raise RolloutInfrastructureError(
                "host transition did not append exactly one runtime event"
            )
        event = events[-1]
        envelope = {
            "trusted_runtime_observation": {
                "name": str(event["name"]),
                "result": copy.deepcopy(event["observation"]),
            }
        }
        return {
            "role": "user",
            "content": [{"type": "text", "text": canonical_json(envelope)}],
        }

    async def _run_one_shot(
        self,
        *,
        services: _RuntimeServices,
        sampling_params: dict[str, Any],
        system_message: Mapping[str, Any],
        story_id: str,
        episode_id: str,
        episode_dir: Path,
        shot_spec: Mapping[str, Any],
        state_seed: Mapping[str, Any],
    ) -> tuple[AgentLoopOutput, dict[str, Any], dict[str, Any]]:
        shot_index = int(shot_spec["shot_index"])
        requirement = str(shot_spec["shot_requirement"])
        messages = [
            copy.deepcopy(dict(system_message)),
            {"role": "user", "content": requirement},
        ]
        declared_token_count = int(shot_spec["initial_prompt_token_count"])
        initial_prompt_ids, initial_processor_text = await self._encode_initial_prompt(
            messages, declared_token_count=declared_token_count
        )
        initial_prompt_length = len(initial_prompt_ids)
        runtime, host, shot_dir, pe_mode = self._create_runtime(
            services=services,
            episode_dir=episode_dir,
            story_id=story_id,
            episode_id=episode_id,
            shot_spec=shot_spec,
            state_seed=state_seed,
        )
        metrics: dict[str, Any] = {}
        request_id = uuid.uuid4().hex
        agent_data = AgentData(
            messages=messages,
            image_data=None,
            video_data=None,
            audio_data=None,
            mm_processor_kwargs=self._get_mm_processor_kwargs(None),
            metrics=metrics,
            request_id=request_id,
            tools_kwargs={},
        )
        agent_data._memory_policy_session = MemoryPolicyToolSession(runtime=runtime)
        agent_data._format_rejections = []
        agent_data._memory_policy_host_context = {
            "services": services,
            "host": host,
            "shot_dir": shot_dir,
            "story_id": story_id,
            "shot_index": shot_index,
            "sequence_index": int(shot_spec["episode_position"]),
            "requirement": requirement,
            "pe_mode": pe_mode,
        }
        self._set_all_tools(agent_data)
        # ToolAgentLoop extends prompt_ids in place after every model/environment
        # turn.  Keep the immutable initial list separate: it is also the length
        # anchor used to prove the expanded training stream is aligned.
        agent_data.prompt_ids = list(initial_prompt_ids)
        # prompt_ids remains the raw-placeholder inference stream used by
        # SGLang.  This parallel stream carries processor-expanded visual ids
        # and is the one returned to VERL for loss/position-id construction.
        agent_data._training_prompt_ids = list(initial_prompt_ids)
        agent_data._processor_text_fragments = [initial_processor_text]

        state = AgentState.GENERATING
        while state != AgentState.TERMINATED:
            if state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
                if runtime.phase == "complete":
                    state = AgentState.TERMINATED
            else:
                agent_data._protocol_error = f"unexpected agent state {state!r}"
                state = AgentState.TERMINATED

        selected = runtime.state.data.get("selected_generation") or {}
        selected_generation_id = str(selected.get("generation_id") or "")
        selected_host = (host.get("generations") or {}).get(selected_generation_id)
        output_video = None
        output_sha256 = None
        if isinstance(selected_host, Mapping):
            candidate_path = Path(str(selected_host.get("output_path") or ""))
            if candidate_path.is_file():
                output_video = str(candidate_path.resolve())
                output_sha256 = str(selected_host.get("output_sha256") or "")

        session = agent_data._memory_policy_session
        carry = {
            "registry_items": copy.deepcopy(runtime.state.data["registry"]["items"]),
            "memory_bank": copy.deepcopy(runtime.state.data["memory_bank"]),
            "previous_evidence": _previous_evidence_from_final_bundle(host),
        }
        shot_record = {
            "shot_index": shot_index,
            "episode_position": int(shot_spec["episode_position"]),
            "shot_requirement": requirement,
            "shot_requirement_sha256": _sha256_text(requirement),
            "fresh_conversation": True,
            "initial_context_sha256": _sha256_text(canonical_json(messages[:2])),
            "initial_state_sha256": _state_component_sha256(state_seed),
            "final_state_sha256": _state_component_sha256(carry),
            "source_reference_output_path": str(
                shot_spec.get("reference_output_path") or ""
            ),
            "source_reference_output_sha256": str(
                shot_spec.get("reference_output_sha256") or ""
            ),
            "source_trajectory_path": str(
                shot_spec.get("source_trajectory_path") or ""
            ),
            "source_trajectory_sha256": str(
                shot_spec.get("source_trajectory_sha256") or ""
            ),
            "source_request_id": str(shot_spec.get("source_request_id") or ""),
            "phase": runtime.phase,
            "completed": runtime.phase == "complete",
            "protocol_error": getattr(agent_data, "_protocol_error", None),
            "format_rejections": copy.deepcopy(agent_data._format_rejections),
            "assistant_turns": agent_data.assistant_turns,
            "environment_turns": agent_data.user_turns,
            "tool_calls": copy.deepcopy(session.call_log),
            "host_events": copy.deepcopy(runtime.state.data.get("host_events") or []),
            "infrastructure_failures": copy.deepcopy(session.infrastructure_failures),
            "selected_generation_id": selected_generation_id or None,
            "output_video_path": output_video,
            "output_video_sha256": output_sha256,
            "policy_state_path": str(runtime.state.path),
            **({"initial_policy_state_path": str(shot_dir / "initial_policy_state.json"),
                "initial_policy_state_sha256": _sha256_text(
                    (shot_dir / "initial_policy_state.json").read_text(encoding="utf-8"))}
               if PROMPT_AUTHORING else {}),
            "initial_memory_ids": sorted(str(key) for key in state_seed["memory_bank"]),
            "final_memory_ids": sorted(
                str(key) for key in runtime.state.data.get("memory_bank", {})
            ),
            "accepted_evidence_plan": copy.deepcopy(
                runtime.state.data.get("accepted_evidence_plan")
            ),
            "condition_review": copy.deepcopy(
                runtime.state.data.get("condition_review")
            ),
            "selected_generation": copy.deepcopy(
                runtime.state.data.get("selected_generation")
            ),
            "final_memory_bank": copy.deepcopy(
                runtime.state.data.get("memory_bank") or {}
            ),
        }
        training_prompt_ids = agent_data._training_prompt_ids
        response_count = len(agent_data.response_mask)
        multi_modal_data = self._multi_modal_data(agent_data)
        _validate_final_training_alignment(
            initial_prompt_length=initial_prompt_length,
            training_prompt_ids=training_prompt_ids,
            response_mask=agent_data.response_mask,
            response_logprobs=agent_data.response_logprobs,
            multi_modal_data=multi_modal_data,
            routed_experts=getattr(agent_data, "routed_experts", None),
        )
        response_ids = (
            training_prompt_ids[-response_count:] if response_count else []
        )
        prompt_ids = (
            training_prompt_ids[:-response_count]
            if response_count
            else training_prompt_ids
        )
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs
                else None
            ),
            routed_experts=getattr(agent_data, "routed_experts", None),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            reward_score=None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=metrics,
            extra_fields={
                **agent_data.extra_fields,
                "raw_prompt": messages[:2],
                "shot_index": shot_index,
                "episode_position": int(shot_spec["episode_position"]),
                "tool_rewards": list(agent_data.tool_rewards),
                "_storymem_unexpanded_processor_text": "".join(
                    agent_data._processor_text_fragments
                ),
            },
        )
        return output, shot_record, carry

    def _validate_episode_input(
        self,
        raw_prompt: Sequence[Mapping[str, Any]],
        extra_info: Mapping[str, Any],
    ) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
        if len(raw_prompt) != 1 or raw_prompt[0] != {
            "role": "system",
            "content": TRAINING_SYSTEM_PROMPT,
        }:
            raise ValueError(
                "RL parquet prompt must contain only the unified Qwen3.5 system message"
            )
        if extra_info.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError("RL dataset row is not the current v2 schema")
        if extra_info.get("system_prompt_sha256") != training_system_prompt_hash():
            raise ValueError("RL dataset system prompt hash mismatch")
        if extra_info.get("tool_protocol_version") != TOOL_PROTOCOL_VERSION:
            raise ValueError("RL dataset tool protocol version mismatch")
        if extra_info.get("tool_schema_sha256") != tool_schema_hash():
            raise ValueError("RL dataset tool schema hash mismatch")
        if (
            extra_info.get("prompt_validation_schema_version")
            != PROMPT_VALIDATION_SCHEMA_VERSION
        ):
            raise ValueError("RL dataset has no current prompt-length proof")
        declared_max_prompt_length = extra_info.get("max_prompt_length")
        if (
            isinstance(declared_max_prompt_length, bool)
            or not isinstance(declared_max_prompt_length, int)
            or declared_max_prompt_length != self.prompt_length
        ):
            raise ValueError(
                "RL dataset max_prompt_length differs from rollout.prompt_length"
            )
        if extra_info.get("fresh_conversation_per_shot") is not True:
            raise ValueError("RL episode does not require fresh context per shot")
        mode = str(extra_info.get("mode") or "")
        if mode not in {"single", "multi"}:
            raise ValueError("RL episode mode must be single or multi")
        story_id = str(extra_info.get("story_id") or "")
        episode_id = str(extra_info.get("episode_id") or "")
        if not story_id or not episode_id:
            raise ValueError("RL episode has no story_id/episode_id")
        raw_specs = extra_info.get("shot_specs")
        if not isinstance(raw_specs, (list, tuple)) or not raw_specs:
            raise ValueError("RL episode has no shot_specs")
        specs = [copy.deepcopy(dict(spec)) for spec in raw_specs]
        raw_prompt_counts = extra_info.get("initial_prompt_token_counts")
        if not isinstance(raw_prompt_counts, (list, tuple)):
            raise ValueError("RL episode has no initial_prompt_token_counts proof")
        prompt_counts = list(raw_prompt_counts)
        if len(prompt_counts) != len(specs):
            raise ValueError("RL episode prompt-count proof has the wrong length")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            or value > self.prompt_length
            for value in prompt_counts
        ):
            raise ValueError("RL episode prompt-count proof is invalid")
        if extra_info.get("window_size") != len(specs):
            raise ValueError("RL episode window_size does not match shot_specs")
        expected_positions = list(range(len(specs)))
        if [spec.get("episode_position") for spec in specs] != expected_positions:
            raise ValueError("shot_specs episode positions are not contiguous")
        for position, spec in enumerate(specs):
            if spec.get("schema_version") != SHOT_SPEC_SCHEMA_VERSION:
                raise ValueError("shot spec is not the current v2 schema")
            if spec.get("fresh_conversation") is not True:
                raise ValueError("shot spec does not require a fresh conversation")
            requirement = spec.get("shot_requirement")
            if not isinstance(requirement, str) or not requirement:
                raise ValueError("shot spec has no user request")
            if spec.get("shot_requirement_sha256") != _sha256_text(requirement):
                raise ValueError("shot requirement hash mismatch")
            prompt_count = spec.get("initial_prompt_token_count")
            if (
                isinstance(prompt_count, bool)
                or not isinstance(prompt_count, int)
                or prompt_count < 1
                or prompt_count > self.prompt_length
            ):
                raise ValueError("shot has an invalid initial_prompt_token_count")
            if prompt_counts[position] != prompt_count:
                raise ValueError("shot prompt count differs from its episode proof")
            inherits = [
                spec.get("inherit_registry_from_previous_rollout_shot") is True,
                spec.get("inherit_memory_bank_from_previous_rollout_shot") is True,
                spec.get("inherit_previous_evidence_from_previous_rollout_shot") is True,
            ]
            if position == 0 and any(inherits):
                raise ValueError("first episode shot cannot inherit online state")
            if position > 0 and mode == "multi" and not all(inherits):
                raise ValueError("later multi-shot specs must inherit all online state")
        if mode == "single" and len(specs) != 1:
            raise ValueError("single-shot row must contain exactly one shot")
        if mode == "multi" and len(specs) < 2:
            raise ValueError("multi-shot row must contain at least two shots")
        return mode, story_id, specs, copy.deepcopy(dict(raw_prompt[0]))

    @staticmethod
    def _inherit_online_state(
        seed: dict[str, Any], carry: Mapping[str, Any]
    ) -> dict[str, Any]:
        inherited = copy.deepcopy(seed)
        inherited["registry_items"] = copy.deepcopy(carry["registry_items"])
        inherited["memory_bank"] = copy.deepcopy(carry["memory_bank"])
        inherited["previous_evidence"] = copy.deepcopy(carry["previous_evidence"])
        memory_ordinals = [
            int(match.group(1))
            for key in inherited["memory_bank"]
            for match in [re.fullmatch(r"memory_(\d+)", str(key))]
            if match is not None
        ]
        inherited["memory_counter_floor"] = max(memory_ordinals, default=0)
        return inherited

    async def _execute_episode_shots(
        self,
        *,
        services: _RuntimeServices,
        sampling_params: Mapping[str, Any],
        system_message: Mapping[str, Any],
        story_id: str,
        rollout_id: str,
        episode_dir: Path,
        shot_specs: Sequence[Mapping[str, Any]],
        inherit_online_state: bool,
    ) -> tuple[
        list[AgentLoopOutput],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Execute every policy shot unless infrastructure makes output invalid.

        A malformed or incomplete policy trajectory is a valid zero-reward RL
        sample, so it never shortens a multi-shot episode.  Infrastructure
        failures are different: no partial tensor batch is returned at all.
        """

        outputs: list[AgentLoopOutput] = []
        shot_records: list[dict[str, Any]] = []
        infrastructure_failures: list[dict[str, Any]] = []
        carry: dict[str, Any] | None = None
        for position, shot_spec in enumerate(shot_specs):
            seed = _decode_state_seed(shot_spec)
            if position:
                if not inherit_online_state:
                    raise AssertionError("single-shot execution received multiple shots")
                if carry is None:
                    raise AssertionError("multi-shot carry state is missing")
                seed = self._inherit_online_state(seed, carry)
            output, shot_record, carry = await self._run_one_shot(
                services=services,
                sampling_params=dict(sampling_params),
                system_message=system_message,
                story_id=story_id,
                episode_id=rollout_id,
                episode_dir=episode_dir,
                shot_spec=shot_spec,
                state_seed=seed,
            )
            expected_state_components = {
                "registry_items",
                "memory_bank",
                "previous_evidence",
            }
            for hash_field in ("initial_state_sha256", "final_state_sha256"):
                hashes = shot_record.get(hash_field)
                if (
                    not isinstance(hashes, Mapping)
                    or set(hashes) != expected_state_components
                    or any(
                        not isinstance(digest, str)
                        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
                        for digest in hashes.values()
                    )
                ):
                    raise AssertionError(
                        f"shot record has invalid {hash_field} carry proof"
                    )
            if position and (
                shot_records[-1].get("final_state_sha256")
                != shot_record.get("initial_state_sha256")
            ):
                raise AssertionError(
                    "multi-shot carry hash differs from the next shot's initial state"
                )
            outputs.append(output)
            shot_records.append(shot_record)
            infrastructure_failures.extend(shot_record["infrastructure_failures"])
            if infrastructure_failures:
                break

        if not infrastructure_failures and len(outputs) != len(shot_specs):
            raise AssertionError(
                "policy execution returned a short episode without an "
                "infrastructure failure"
            )
        return outputs, shot_records, infrastructure_failures

    async def _run_single_episode(
        self, **kwargs: Any
    ) -> tuple[
        list[AgentLoopOutput],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        shot_specs = kwargs["shot_specs"]
        if len(shot_specs) != 1:
            raise AssertionError("single-shot execution requires K=1")
        return await self._execute_episode_shots(
            **kwargs, inherit_online_state=False
        )

    async def _run_multi_episode(
        self, **kwargs: Any
    ) -> tuple[
        list[AgentLoopOutput],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        shot_specs = kwargs["shot_specs"]
        if len(shot_specs) < 2:
            raise AssertionError("multi-shot execution requires K>=2")
        return await self._execute_episode_shots(
            **kwargs, inherit_online_state=True
        )

    async def _run_episode_once(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput | list[AgentLoopOutput]:
        raw_prompt = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        if not isinstance(extra_info, Mapping):
            raise ValueError("extra_info must be an object")
        mode, story_id, shot_specs, system_message = self._validate_episode_input(
            raw_prompt, extra_info
        )
        services = self._get_services()
        source_episode_id = str(extra_info["episode_id"])
        rollout_id = str(
            kwargs.get("_storymem_rollout_id")
            or (
                f"{_safe_name(source_episode_id, 'episode')}_"
                f"rank{os.getenv('RANK', '0')}_{time.time_ns()}_"
                f"{uuid.uuid4().hex[:10]}"
            )
        )
        episode_dir = Path(
            kwargs.get("_storymem_episode_dir")
            or self.output_root / "episodes" / rollout_id
        )
        infrastructure_attempt = int(
            kwargs.get("_storymem_infrastructure_attempt") or 1
        )
        max_infrastructure_attempts = int(
            kwargs.get("_storymem_max_infrastructure_attempts") or 1
        )
        episode_dir.mkdir(parents=True, exist_ok=False)

        episode_kwargs = {
            "services": services,
            "sampling_params": sampling_params,
            "system_message": system_message,
            "story_id": story_id,
            "rollout_id": rollout_id,
            "episode_dir": episode_dir,
            "shot_specs": shot_specs,
        }
        if mode == "single":
            outputs, shot_records, infrastructure_failures = (
                await self._run_single_episode(**episode_kwargs)
            )
        else:
            outputs, shot_records, infrastructure_failures = (
                await self._run_multi_episode(**episode_kwargs)
            )

        expected_shots = len(shot_specs)
        completed_videos = [
            str(row["output_video_path"])
            for row in shot_records
            if row.get("completed") and row.get("output_video_path")
        ]
        episode: dict[str, Any] = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "rollout_id": rollout_id,
            "source_episode_id": source_episode_id,
            "story_id": story_id,
            "mode": mode,
            "expected_shots": expected_shots,
            "completed_shots": sum(bool(row["completed"]) for row in shot_records),
            "fresh_conversation_per_shot": True,
            "system_prompt_sha256": training_system_prompt_hash(),
            "tool_protocol_version": TOOL_PROTOCOL_VERSION,
            "tool_schema_sha256": tool_schema_hash(),
            "shots": shot_records,
            "shot_video_paths": completed_videos,
            "infrastructure_failures": infrastructure_failures,
            "infrastructure_attempt": infrastructure_attempt,
            "max_infrastructure_attempts": max_infrastructure_attempts,
            "created_at_unix": time.time(),
        }
        trajectory_path = episode_dir / "trajectory.json"
        _atomic_json(trajectory_path, episode)

        if infrastructure_failures:
            raise RolloutInfrastructureError(
                "rollout infrastructure failed; trajectory retained at "
                f"{trajectory_path}"
            )

        try:
            from .reward import evaluate_episode

            reward_info = await asyncio.to_thread(
                evaluate_episode, trajectory_path, mode=mode
            )
        except Exception as exc:
            # Import locally so the loop remains testable before reward provider
            # setup.  Any evaluator failure is infrastructure, never reward 0.
            if type(exc).__name__ == "RewardInfrastructureError":
                raise RolloutInfrastructureError(str(exc)) from exc
            raise
        if not isinstance(reward_info, Mapping):
            raise TypeError("evaluate_episode must return an object")
        episode_reward = float(reward_info["reward"])
        if not math.isfinite(episode_reward) or not 0.0 <= episode_reward <= 1.0:
            raise ValueError("evaluate_episode.reward must be finite and in [0, 1]")
        local_rewards = reward_info.get("local_shot_rewards")
        if not isinstance(local_rewards, list) or len(local_rewards) != expected_shots:
            raise TypeError(
                "evaluate_episode.local_shot_rewards must contain exactly K scores"
            )
        local_rewards = [float(value) for value in local_rewards]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in local_rewards):
            raise ValueError("local_shot_rewards must be finite values in [0, 1]")
        all_shots_completed = len(shot_records) == expected_shots and all(
            row["completed"] for row in shot_records
        )
        if not all_shots_completed and (
            episode_reward != 0.0 or any(value != 0.0 for value in local_rewards)
        ):
            raise ValueError(
                "an incomplete episode is atomic and must receive zero episode/local rewards"
            )
        episode["reward"] = episode_reward
        episode["reward_info"] = copy.deepcopy(dict(reward_info))
        _atomic_json(trajectory_path, episode)

        base_uid = str(
            kwargs.get("uid")
            or extra_info.get("index")
            or source_episode_id
        )
        # VERL treats ``reward_extra_info`` as a rectangular dict-of-arrays.
        # Episode audits contain variable-length per-shot structures, so keep
        # the complete audit on disk and carry only its path plus scalar scores
        # through the training batch.
        reward_review_path = str(reward_info.get("review_path") or "")
        if not reward_review_path or not Path(reward_review_path).is_file():
            raise RolloutInfrastructureError(
                "reward evaluator did not persist reward_review.json"
            )
        local_weight = _env_float("STORYMEM_RL_MULTI_LOCAL_MIX", 0.5)
        episode_weight = 1.0 - local_weight
        for position, output in enumerate(outputs):
            if mode == "multi":
                assigned_reward = (
                    episode_weight * episode_reward
                    + local_weight * local_rewards[position]
                )
            else:
                assigned_reward = episode_reward
            output.reward_score = max(0.0, min(1.0, assigned_reward))
            output.extra_fields.update(
                {
                    "uid": f"{base_uid}:shot:{position:03d}",
                    "rollout_id": rollout_id,
                    "trajectory_path": str(trajectory_path),
                    "reward_review_path": reward_review_path,
                    "online_reward": output.reward_score,
                    "episode_reward": episode_reward,
                    "fresh_context_per_shot": True,
                    "rollout_k": expected_shots,
                    "infrastructure_attempt": infrastructure_attempt,
                }
            )
        if len(outputs) != expected_shots:
            raise AssertionError("successful rollout path did not return exactly K outputs")
        return outputs[0] if mode == "single" else outputs

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput | list[AgentLoopOutput]:
        """Retry only infrastructure failures with a completely fresh episode."""

        raw_extra_info = kwargs.get("extra_info")
        source_episode_id = (
            str(raw_extra_info.get("episode_id") or "episode")
            if isinstance(raw_extra_info, Mapping)
            else "episode"
        )
        rollout_base = (
            f"{_safe_name(source_episode_id, 'episode')}_"
            f"rank{os.getenv('RANK', '0')}_{time.time_ns()}_"
            f"{uuid.uuid4().hex[:10]}"
        )
        max_attempts = self.max_infrastructure_attempts
        for attempt in range(1, max_attempts + 1):
            rollout_id = f"{rollout_base}_attempt{attempt:02d}"
            episode_dir = self.output_root / "episodes" / rollout_id
            attempt_kwargs = dict(kwargs)
            if "raw_prompt" in attempt_kwargs:
                attempt_kwargs["raw_prompt"] = copy.deepcopy(
                    attempt_kwargs["raw_prompt"]
                )
            if isinstance(raw_extra_info, Mapping):
                attempt_kwargs["extra_info"] = copy.deepcopy(dict(raw_extra_info))
            attempt_kwargs.update(
                {
                    "_storymem_rollout_id": rollout_id,
                    "_storymem_episode_dir": episode_dir,
                    "_storymem_infrastructure_attempt": attempt,
                    "_storymem_max_infrastructure_attempts": max_attempts,
                }
            )
            try:
                return await self._run_episode_once(
                    copy.deepcopy(sampling_params), **attempt_kwargs
                )
            except RolloutInfrastructureError as exc:
                retrying = attempt < max_attempts
                delay = (
                    min(
                        30.0,
                        self.infrastructure_retry_backoff_seconds
                        * (2 ** (attempt - 1)),
                    )
                    if retrying
                    else 0.0
                )
                marker = {
                    "schema_version": "storymem.memory-policy.infrastructure-attempt.v1",
                    "source_episode_id": source_episode_id,
                    "rollout_id": rollout_id,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error_type": type(exc).__name__,
                    "message": " ".join(str(exc).split())[:2048],
                    "retrying": retrying,
                    "next_retry_delay_seconds": delay,
                    "created_at_unix": time.time(),
                }
                _atomic_json(
                    episode_dir / "infrastructure_attempt_failed.json", marker
                )
                if not retrying:
                    raise RolloutInfrastructureError(
                        f"rollout infrastructure failed after {max_attempts} "
                        f"attempts; final audit marker: "
                        f"{episode_dir / 'infrastructure_attempt_failed.json'}"
                    ) from exc
                if delay:
                    await asyncio.sleep(delay)
        raise AssertionError("infrastructure retry loop exited without a result")


__all__ = [
    "RolloutInfrastructureError",
    "StoryMemMemoryPolicyOnlineAgentLoop",
    "StrictQwen35XMLToolParser",
    "ToolCallFormatError",
    "parse_qwen35_xml_tool_call",
    "render_qwen35_environment_fragment",
]
