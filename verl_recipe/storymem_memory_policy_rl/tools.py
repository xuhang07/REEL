"""VERL tool adapters for the canonical nine-tool Memory Policy runtime.

VERL 0.8's ``OpenAIFunction*Schema`` models only retain shallow parameter
metadata.  The Memory Policy contract deliberately contains nested objects,
arrays and ``oneOf`` branches, so this module keeps each canonical schema
verbatim and exposes only the small attribute surface used by VERL.

The tool objects are intentionally thin.  A fresh :class:`MemoryPolicyRuntime`
is attached to ``AgentData`` by ``online_agent_loop.py`` for every shot; all
state validation and mutation remains owned by that runtime.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from uuid import uuid4

from verl.tools.schemas import ToolResponse

ECHO_LONGVIDEO_ROOT = Path(__file__).resolve().parents[2]
while str(ECHO_LONGVIDEO_ROOT) in sys.path:
    sys.path.remove(str(ECHO_LONGVIDEO_ROOT))
sys.path.insert(0, str(ECHO_LONGVIDEO_ROOT))

from tools.storymem_policy.contract import (
    CANONICAL_TOOL_NAMES,
)
from .policy_profile import DIRECT_PROMPT, get_tool_schemas


def canonical_json(value: Any) -> str:
    """Serialize a model-visible observation without lossy coercion."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class RawOpenAIToolSchema:
    """Minimal VERL-compatible wrapper which preserves the raw JSON Schema.

    ``ToolAgentLoop`` requires ``.function.name`` and ``.model_dump()``.  Its
    type annotations are not a runtime wire contract, so retaining the exact
    canonical dictionary here is both safer and more faithful than coercing it
    through VERL's shallow Pydantic schema.
    """

    def __init__(self, schema: Mapping[str, Any]) -> None:
        # The SFT exporter stores ``row.tools`` as recursively key-sorted JSON.
        # Qwen's chat template renders mappings in their insertion order, so a
        # merely equivalent schema with the contract's construction order is a
        # different autoregressive prompt.  Canonical round-tripping here keeps
        # every nested mapping in the exact order seen during SFT.
        raw = json.loads(canonical_json(dict(schema)))
        function = raw.get("function")
        if raw.get("type") != "function" or not isinstance(function, Mapping):
            raise ValueError("raw tool schema must be an OpenAI function tool")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name:
            raise ValueError("raw tool schema has no function name")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"raw tool schema {name!r} has no parameters object")
        self._raw = raw
        self.type = "function"
        self.function = SimpleNamespace(
            name=name,
            description=str(function.get("description") or ""),
            parameters=copy.deepcopy(dict(parameters)),
            strict=bool(function.get("strict", False)),
        )

    def model_dump(self, **_: Any) -> dict[str, Any]:
        """Match the Pydantic method used by VERL without dropping fields."""

        return copy.deepcopy(self._raw)


_SCHEMA_BY_NAME = {
    row["function"]["name"]: row for row in get_tool_schemas()
}
if tuple(_SCHEMA_BY_NAME) != CANONICAL_TOOL_NAMES:
    raise RuntimeError("canonical Memory Policy tool order changed unexpectedly")


# These rejections are produced after the policy has already supplied a valid
# tool action.  They describe bytes, media executables, or the external image
# provider, so treating them as a bad policy action would put host failures into
# the GRPO signal.  Keep this list deliberately narrow: ordinary FSM/schema
# rejections must remain visible to the policy as learnable failures.
_INFRASTRUCTURE_REJECTION_CODES = frozenset(
    {
        "audio_selector_unavailable",
        "creator_output_not_image",
        "creator_protocol_error",
        "creator_request_mismatch",
        "invalid_audio_memory",
        "media_integrity_mismatch",
        "media_missing",
        "media_perception_failed",
    }
)

# ``MemoryPolicyRuntime`` currently uses ``invalid_time_range`` for both a
# policy-selected out-of-bounds range and a failure while materializing a valid
# range.  Only the latter is infrastructure.  The messages below are emitted by
# the canonical media implementation for missing/corrupt bytes, I/O failures,
# and unavailable/failed media executables.
_SELECTOR_INFRASTRUCTURE_MARKERS = (
    "asset representation file does not exist:",
    "asset representation file is empty:",
    "asset representation hash no longer matches its bytes",
    "binary is unavailable:",
    "cannot determine media duration:",
    "failed:",
    "file does not exist:",
    "file is empty:",
    "media duration is not positive:",
    "media duration is unavailable",
    "no decodable",
    "not a recognized video:",
    "produced no",
)


def _is_infrastructure_rejection(result: Mapping[str, Any]) -> bool:
    """Return whether a canonical runtime rejection is outside policy control."""

    if result.get("ok") is not False:
        return False
    code = str(result.get("error") or "")
    if code in _INFRASTRUCTURE_REJECTION_CODES:
        return True
    if code != "invalid_time_range":
        return False
    message = str(result.get("message") or "").casefold()
    return any(marker in message for marker in _SELECTOR_INFRASTRUCTURE_MARKERS)


def _record_infrastructure_failure(
    session: "MemoryPolicyToolSession",
    *,
    tool_name: str,
    failure_type: str,
    runtime_error: str | None = None,
) -> dict[str, Any]:
    """Record one private retry signal and return a safe model observation."""

    failure = {
        "tool": tool_name,
        "type": failure_type,
        "message": "external tool infrastructure failed; retry this rollout",
    }
    if runtime_error:
        failure["runtime_error"] = runtime_error
    session.infrastructure_failures.append(failure)
    return {
        "ok": False,
        "error": "infrastructure_failure",
        "message": failure["message"],
    }


@dataclass(slots=True)
class MemoryPolicyToolSession:
    """Per-shot bridge installed by the online agent loop."""

    runtime: Any
    call_log: list[dict[str, Any]] = field(default_factory=list)
    infrastructure_failures: list[dict[str, str]] = field(default_factory=list)


def _direct_prompt_rejection(name: str, parameters: dict[str, Any]) -> dict[str, Any] | None:
    if not DIRECT_PROMPT or name != "submit_evidence_plan":
        return None
    draft = parameters.get("generation_prompt")
    if not isinstance(draft, str) or not draft.strip() or draft != draft.strip():
        return {"ok": False, "error": "invalid_generation_prompt",
                "message": "Submit a nonempty, trimmed, complete generation_prompt; no plan was accepted."}
    from storymem_adapter.echo_prompt_budget import get_echo_prompt_budget
    from storymem_adapter.v7_validation import V7PromptValidationError
    try:
        get_echo_prompt_budget().validate(draft)
    except V7PromptValidationError as exc:
        return {"ok": False, "error": "generation_prompt_token_limit",
                "message": "Your final caption exceeds 1024 Echo Gemma tokens. Rewrite it concisely while preserving the original intent and exact dialogue, then resubmit the same pending plan version. No plan was accepted or video generated.",
                "violations": exc.violations}
    return None


class CanonicalMemoryPolicyTool:
    """One configured view of the shared canonical runtime."""

    def __init__(self, config: dict, tool_schema: Any = None) -> None:
        if tool_schema is not None:
            raise ValueError(
                "Memory Policy schemas must come from contract.py; set tool_schema: null"
            )
        self.config = dict(config or {})
        name = str(self.config.get("tool_name") or "")
        if name not in _SCHEMA_BY_NAME:
            raise ValueError(
                f"unknown Memory Policy tool {name!r}; expected one of "
                f"{list(CANONICAL_TOOL_NAMES)!r}"
            )
        self.tool_schema = RawOpenAIToolSchema(_SCHEMA_BY_NAME[name])
        self.name = name

    def get_openai_tool_schema(self) -> RawOpenAIToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: str | None = None, **_: Any
    ) -> tuple[str, ToolResponse]:
        return instance_id or uuid4().hex, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        del instance_id
        agent_data = kwargs.get("agent_data")
        session = getattr(agent_data, "_memory_policy_session", None)
        if not isinstance(session, MemoryPolicyToolSession):
            raise RuntimeError("Memory Policy tool called without a per-shot session")
        if not isinstance(parameters, dict):
            raise TypeError("tool arguments must be a JSON object")

        phase_before = str(session.runtime.phase)
        try:
            raw_result = await asyncio.to_thread(_direct_prompt_rejection, self.name, parameters)
            if raw_result is None:
                raw_result = await asyncio.to_thread(
                    session.runtime.execute, self.name, copy.deepcopy(parameters),
                )
            if not isinstance(raw_result, dict):
                raise TypeError("MemoryPolicyRuntime returned a non-object result")
            observation = copy.deepcopy(raw_result)
            raw_attachments = observation.pop("_media_attachments", [])
            if not isinstance(raw_attachments, list):
                raise TypeError("runtime media attachments must be an array")
            if any(not isinstance(row, Mapping) for row in raw_attachments):
                raise TypeError("runtime media attachment must be an object")
            attachments = [copy.deepcopy(dict(row)) for row in raw_attachments]
            if _is_infrastructure_rejection(observation):
                runtime_error = str(observation.get("error") or "runtime_rejection")
                observation = _record_infrastructure_failure(
                    session,
                    tool_name=self.name,
                    failure_type="RuntimeInfrastructureRejection",
                    runtime_error=runtime_error,
                )
                attachments = []
        except Exception as exc:
            # Runtime/provider exceptions and malformed runtime responses are
            # rollout infrastructure failures, never policy-quality evidence.
            observation = _record_infrastructure_failure(
                session,
                tool_name=self.name,
                failure_type=type(exc).__name__,
            )
            attachments = []

        phase_after = str(session.runtime.phase)
        call_record = {
            "name": self.name,
            "arguments": copy.deepcopy(parameters),
            "result": copy.deepcopy(observation),
            "phase_before": phase_before,
            "phase_after": phase_after,
        }
        session.call_log.append(call_record)
        envelope = {
            "trusted_tool_observation": {
                "name": self.name,
                "result": observation,
            }
        }
        private_result = {
            "call": call_record,
            "observation": observation,
            "_media_attachments": attachments,
        }
        return ToolResponse(text=canonical_json(envelope)), 0.0, private_result

    async def calc_reward(self, instance_id: str, **_: Any) -> float:
        del instance_id
        return 0.0

    async def release(self, instance_id: str, **_: Any) -> None:
        del instance_id


def raw_tool_schemas() -> list[dict[str, Any]]:
    """Return all nine schemas in their canonical order."""

    return [copy.deepcopy(_SCHEMA_BY_NAME[name]) for name in CANONICAL_TOOL_NAMES]


__all__ = [
    "CanonicalMemoryPolicyTool",
    "MemoryPolicyToolSession",
    "RawOpenAIToolSchema",
    "canonical_json",
    "raw_tool_schemas",
]
