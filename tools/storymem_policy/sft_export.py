"""Deterministic Memory Policy trajectory to native-tool SFT projection."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from .contract import CANONICAL_TOOL_NAMES, get_tool_schemas, tool_schema_hash
from .prompts import OPEN_MODEL_SYSTEM_PROMPT, system_prompt_for_mode
from .trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryValidationError,
    _validate_media_row,
    _validate_public_observation,
    build_training_messages,
    fsm_next_phases,
    parse_plain_json_tool_command,
    trajectory_media_manifest,
    validate_trajectory_shard,
)


SFT_SAMPLE_SCHEMA_VERSION = "storymem.memory-policy.sft.sample.v3"
SFT_PROJECTION_VERSION = "plain-json-teacher-to-native-tools.v2"

_SFT_SAMPLE_FIELDS = (
    "schema_version",
    "projection_version",
    "source_trajectory_schema_version",
    "source_trajectory_sha256",
    "source_name",
    "session_id",
    "request_id",
    "raw_request_sha256",
    "tool_schema_sha256",
    "tools",
    "messages",
    "messages_sha256",
    "media_manifest",
    "media_manifest_sha256",
    "projection_audit",
)
_SFT_SAMPLE_FIELD_SET = frozenset(_SFT_SAMPLE_FIELDS)
_AUDIT_FIELDS = frozenset({"events", "host_message_positions"})
_EVENT_AUDIT_FIELDS = frozenset(
    {"event_id", "assistant_text_sha256", "tool_call_sha256"}
)
_HOST_POSITION_FIELDS = frozenset({"message_index", "content_sha256"})


class SFTExportError(ValueError):
    """A trajectory or sample is unsafe or cannot be projected exactly."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 71 or not text.startswith("sha256:") or any(
        character not in "0123456789abcdef" for character in text[7:]
    ):
        raise SFTExportError(f"{label} is not a canonical SHA-256 digest")
    return text


def _require_exact_fields(
    value: Any, expected: frozenset[str], *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SFTExportError(f"{label} must be an object")
    actual = {str(key) for key in value}
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise SFTExportError(
            f"{label} fields are not canonical (missing={missing}, unknown={extra})"
        )
    return value


def _as_export_error(exc: Exception) -> SFTExportError:
    return SFTExportError(str(exc))


def project_plain_teacher_to_native_messages(
    trajectory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project only byte-verified closed-Teacher JSON actions.

    The Teacher's free-form ``reason`` field is intentionally discarded.  No
    call may be repaired, inferred, or regenerated during export.
    """

    try:
        validate_trajectory_shard(trajectory)
        events = trajectory["events"]
        for event in events:
            provenance = event.get("assistant_provenance") or {}
            parsed_call, _ = parse_plain_json_tool_command(
                provenance.get("raw_assistant_text"),
                label=f"event {event.get('event_id')} raw assistant text",
            )
            if parsed_call != event.get("tool_call"):
                raise TrajectoryValidationError(
                    f"event {event.get('event_id')} changed between Teacher and export"
                )
        messages = build_training_messages(
            raw_user_request=str(trajectory["raw_user_request"]),
            events=events,
            host_events=trajectory["host_events"],
            prompt_authoring=(trajectory.get("policy_limits") or {}).get("prompt_authoring") is True,
        )
    except (KeyError, TypeError, ValueError, TrajectoryValidationError) as exc:
        raise _as_export_error(exc) from exc
    for message in messages:
        if message.get("role") == "assistant" and set(message) != {
            "role",
            "tool_calls",
        }:
            raise SFTExportError("assistant projection contains non-tool Teacher text")
    return messages


def build_sft_sample(
    trajectory: Mapping[str, Any],
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Build one self-checking native-tool sample from a validated shard."""

    messages = project_plain_teacher_to_native_messages(trajectory)
    event_audit = []
    for event in trajectory["events"]:
        provenance = event["assistant_provenance"]
        event_audit.append(
            {
                "event_id": event["event_id"],
                "assistant_text_sha256": provenance["assistant_text_sha256"],
                "tool_call_sha256": _sha256_json(event["tool_call"]),
            }
        )
    host_positions = [
        {
            "message_index": index,
            "content_sha256": _sha256_json(message["content"]),
        }
        for index, message in enumerate(messages)
        if index >= 2
        and message.get("role") == "user"
        and isinstance(message.get("content"), list)
    ]
    sample = {
        "schema_version": SFT_SAMPLE_SCHEMA_VERSION,
        "projection_version": SFT_PROJECTION_VERSION,
        "source_trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "source_trajectory_sha256": _sha256_json(trajectory),
        "source_name": source_name,
        "session_id": trajectory["session_id"],
        "request_id": trajectory["request_id"],
        "raw_request_sha256": trajectory["raw_request_sha256"],
        "tool_schema_sha256": trajectory["tool_schema_sha256"],
        "tools": copy.deepcopy(trajectory["tools"]),
        "messages": messages,
        "messages_sha256": _sha256_json(messages),
        "media_manifest": trajectory_media_manifest(messages),
        "projection_audit": {
            "events": event_audit,
            "host_message_positions": host_positions,
        },
    }
    sample["media_manifest_sha256"] = _sha256_json(sample["media_manifest"])
    validate_sft_sample(sample)
    return sample


def validate_sft_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the standalone structure and privacy of an exported sample."""

    if not isinstance(sample, Mapping):
        raise SFTExportError("SFT sample must be an object")
    _require_exact_fields(sample, _SFT_SAMPLE_FIELD_SET, label="SFT sample")
    if sample.get("schema_version") != SFT_SAMPLE_SCHEMA_VERSION:
        raise SFTExportError("SFT sample schema version is incompatible")
    if sample.get("projection_version") != SFT_PROJECTION_VERSION:
        raise SFTExportError("SFT projection version is incompatible")
    if sample.get("source_trajectory_schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise SFTExportError("SFT source trajectory version is incompatible")
    _require_sha256(
        sample.get("source_trajectory_sha256"), "SFT source trajectory hash"
    )
    if sample.get("source_name") is not None and not isinstance(
        sample.get("source_name"), str
    ):
        raise SFTExportError("SFT source_name must be text or null")
    for key in ("session_id", "request_id"):
        if not isinstance(sample.get(key), str) or not sample.get(key):
            raise SFTExportError(f"SFT {key} must be non-empty text")
    prompt_authoring = sample.get("tool_schema_sha256") == tool_schema_hash(prompt_authoring=True)
    if sample.get("tools") != get_tool_schemas(prompt_authoring=prompt_authoring):
        raise SFTExportError("SFT sample tools are not canonical")
    if sample.get("tool_schema_sha256") != tool_schema_hash(prompt_authoring=prompt_authoring):
        raise SFTExportError("SFT tool schema hash mismatch")
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        raise SFTExportError("SFT sample has no complete message trace")
    if messages[0] != {"role": "system", "content": system_prompt_for_mode(prompt_authoring)}:
        raise SFTExportError("SFT sample does not use the complete unified system prompt")
    if not isinstance(messages[1], Mapping) or set(messages[1]) != {
        "role",
        "content",
    }:
        raise SFTExportError("SFT raw user message fields are not canonical")
    if not isinstance(messages[1].get("content"), str) or messages[1].get("role") != "user":
        raise SFTExportError("SFT sample has no raw user request")
    if not messages[1]["content"]:
        raise SFTExportError("SFT raw user request must not be empty")
    if sample.get("raw_request_sha256") != _sha256_text(messages[1]["content"]):
        raise SFTExportError("SFT raw request hash mismatch")
    if sample.get("messages_sha256") != _sha256_json(messages):
        raise SFTExportError("SFT message hash mismatch")

    schema_by_name = {
        row["function"]["name"]: row["function"]["parameters"]
        for row in get_tool_schemas(prompt_authoring=prompt_authoring)
    }

    def parse_observation_block(
        block: Any,
        *,
        envelope_key: str,
        expected_name: str | None,
        label: str,
    ) -> tuple[str, Mapping[str, Any]]:
        _require_exact_fields(block, frozenset({"type", "text"}), label=label)
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise SFTExportError(f"{label} must be one text block")
        try:
            envelope = json.loads(
                block["text"],
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SFTExportError(f"{label} is not strict JSON") from exc
        if block["text"] != _canonical_json(envelope):
            raise SFTExportError(f"{label} is not canonical JSON")
        envelope = _require_exact_fields(
            envelope, frozenset({envelope_key}), label=f"{label} envelope"
        )
        payload = _require_exact_fields(
            envelope[envelope_key],
            frozenset({"name", "result"}),
            label=f"{label} payload",
        )
        name = payload.get("name")
        if not isinstance(name, str) or (
            expected_name is not None and name != expected_name
        ):
            raise SFTExportError(f"{label} observation name mismatch")
        try:
            _validate_public_observation(
                name,
                payload.get("result"),
                envelope_key=envelope_key,
                label=label,
            )
        except TrajectoryValidationError as exc:
            raise SFTExportError(str(exc)) from exc
        return name, payload["result"]

    tool_calls: list[tuple[str, str, dict[str, Any]]] = []
    host_positions: list[dict[str, Any]] = []
    call_ids: set[str] = set()
    phase = "need_registry"
    index = 2
    while index < len(messages):
        assistant = _require_exact_fields(
            messages[index],
            frozenset({"role", "tool_calls"}),
            label=f"SFT message {index}",
        )
        if assistant.get("role") != "assistant":
            raise SFTExportError(
                "SFT trace must alternate an assistant call with its tool result"
            )
        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise SFTExportError("each assistant turn must contain exactly one tool call")
        native = _require_exact_fields(
            calls[0],
            frozenset({"id", "type", "function"}),
            label=f"SFT message {index} native call",
        )
        if native.get("type") != "function":
            raise SFTExportError("native tool call type must be function")
        call_id = native.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in call_ids:
            raise SFTExportError("native tool call IDs must be non-empty and unique")
        call_ids.add(call_id)
        function = _require_exact_fields(
            native.get("function"),
            frozenset({"name", "arguments"}),
            label=f"SFT message {index} native function",
        )
        name = function.get("name")
        if name not in CANONICAL_TOOL_NAMES:
            raise SFTExportError("native call names a non-canonical tool")
        if not tool_calls and name != "view_material_registry":
            raise SFTExportError("view_material_registry must be the first SFT action")
        try:
            arguments = json.loads(
                function.get("arguments"),
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SFTExportError("native call arguments are not strict JSON") from exc
        if not isinstance(arguments, dict) or function["arguments"] != _canonical_json(arguments):
            raise SFTExportError("native call arguments are not canonical JSON")
        errors = sorted(
            Draft202012Validator(schema_by_name[str(name)]).iter_errors(arguments),
            key=lambda error: list(error.path),
        )
        if errors:
            location = ".".join(str(item) for item in errors[0].path) or "<root>"
            raise SFTExportError(
                f"native call arguments are invalid at {location}: {errors[0].message}"
            )
        tool_calls.append((call_id, str(name), arguments))

        index += 1
        if index >= len(messages):
            raise SFTExportError("assistant call has no matching tool result")
        tool_message = _require_exact_fields(
            messages[index],
            frozenset({"role", "tool_call_id", "name", "content"}),
            label=f"SFT message {index}",
        )
        if tool_message.get("role") != "tool":
            raise SFTExportError("assistant call is not followed by a tool result")
        if tool_message.get("tool_call_id") != call_id or tool_message.get("name") != name:
            raise SFTExportError("tool result ID or name does not match its assistant call")
        content = tool_message.get("content")
        if not isinstance(content, list) or not content:
            raise SFTExportError("tool result content must be a non-empty array")
        _, tool_result = parse_observation_block(
            content[0],
            envelope_key="trusted_tool_observation",
            expected_name=str(name),
            label=f"SFT message {index} tool observation",
        )
        try:
            allowed_after = fsm_next_phases("tool", str(name), phase)
        except TrajectoryValidationError as exc:
            raise SFTExportError(str(exc)) from exc
        reported_phase = tool_result.get("phase")
        if reported_phase is None and len(allowed_after) == 1:
            reported_phase = next(iter(allowed_after))
        if reported_phase not in allowed_after:
            raise SFTExportError(
                f"illegal FSM transition for tool:{name}: {phase!r} -> {reported_phase!r}"
            )
        phase = str(reported_phase)
        if len(tool_calls) == 1 and (
            tool_result.get("request_id") != sample.get("request_id")
            or tool_result.get("raw_request_sha256")
            != sample.get("raw_request_sha256")
            or tool_result.get("shot_requirement") != messages[1]["content"]
        ):
            raise SFTExportError(
                "registry observation does not match the SFT request"
            )
        for content_index, block in enumerate(content[1:], start=1):
            try:
                _validate_media_row(
                    block,
                    kind="block",
                    label=f"SFT message {index} media block {content_index}",
                )
            except TrajectoryValidationError as exc:
                raise SFTExportError(str(exc)) from exc

        if name == "finish_memory_session":
            if tool_result.get("done") is not True:
                raise SFTExportError("finish_memory_session result must have done=true")
            if index != len(messages) - 1:
                raise SFTExportError("SFT trace contains messages after its terminal result")
            break

        index += 1
        while index < len(messages):
            candidate = messages[index]
            if not isinstance(candidate, Mapping) or candidate.get("role") != "user":
                break
            host_message = _require_exact_fields(
                candidate,
                frozenset({"role", "content"}),
                label=f"SFT message {index}",
            )
            host_content = host_message.get("content")
            if not isinstance(host_content, list) or len(host_content) != 1:
                raise SFTExportError(
                    "runtime observation must be one structured user text block"
                )
            host_name, host_result = parse_observation_block(
                host_content[0],
                envelope_key="trusted_runtime_observation",
                expected_name=None,
                label=f"SFT message {index} runtime observation",
            )
            try:
                allowed_after = fsm_next_phases("host", host_name, phase)
            except TrajectoryValidationError as exc:
                raise SFTExportError(str(exc)) from exc
            reported_phase = host_result.get("phase")
            if reported_phase is None and len(allowed_after) == 1:
                reported_phase = next(iter(allowed_after))
            if reported_phase not in allowed_after:
                raise SFTExportError(
                    f"illegal FSM transition for host:{host_name}: "
                    f"{phase!r} -> {reported_phase!r}"
                )
            phase = str(reported_phase)
            host_positions.append(
                {
                    "message_index": index,
                    "content_sha256": _sha256_json(host_content),
                }
            )
            index += 1
    else:
        raise SFTExportError("SFT trace does not finish with done=true")

    if not tool_calls or tool_calls[-1][1] != "finish_memory_session":
        raise SFTExportError("SFT trace does not finish with done=true")
    if phase != "complete":
        raise SFTExportError(f"SFT trace terminates in phase {phase!r}")

    public_media = trajectory_media_manifest(messages)
    if sample.get("media_manifest") != public_media:
        raise SFTExportError("SFT public media manifest mismatch")
    if sample.get("media_manifest_sha256") != _sha256_json(public_media):
        raise SFTExportError("SFT media manifest hash mismatch")
    media_ids: set[str] = set()
    for media_index, row in enumerate(public_media):
        try:
            _validate_media_row(
                row, kind="manifest", label=f"SFT public media row {media_index}"
            )
        except TrajectoryValidationError as exc:
            raise SFTExportError(str(exc)) from exc
        media_id = str(row.get("media_id") or "")
        if media_id in media_ids:
            raise SFTExportError("SFT public media IDs must be unique")
        media_ids.add(media_id)

    audit = _require_exact_fields(
        sample.get("projection_audit"), _AUDIT_FIELDS, label="SFT projection audit"
    )
    event_audit = audit.get("events")
    if not isinstance(event_audit, list) or len(event_audit) != len(tool_calls):
        raise SFTExportError("SFT event audit does not cover every tool call")
    for audit_index, (row, call) in enumerate(zip(event_audit, tool_calls)):
        row = _require_exact_fields(
            row, _EVENT_AUDIT_FIELDS, label=f"SFT event audit {audit_index}"
        )
        call_id, name, arguments = call
        if row.get("event_id") != call_id:
            raise SFTExportError("SFT event audit order or event ID mismatch")
        _require_sha256(
            row.get("assistant_text_sha256"),
            f"SFT event audit {audit_index} assistant text hash",
        )
        if row.get("tool_call_sha256") != _sha256_json(
            {"name": name, "arguments": arguments}
        ):
            raise SFTExportError("SFT event audit tool call hash mismatch")
    audited_hosts = audit.get("host_message_positions")
    if not isinstance(audited_hosts, list) or len(audited_hosts) != len(host_positions):
        raise SFTExportError("SFT host-message audit does not cover every host message")
    for audit_index, (row, expected) in enumerate(
        zip(audited_hosts, host_positions)
    ):
        row = _require_exact_fields(
            row,
            _HOST_POSITION_FIELDS,
            label=f"SFT host-message audit {audit_index}",
        )
        if dict(row) != expected:
            raise SFTExportError("SFT host-message audit position or hash mismatch")
    return {
        "ok": True,
        "tool_call_count": len(tool_calls),
        "media_count": len(public_media),
    }


def export_sft_jsonl(
    samples: Iterable[Mapping[str, Any]], output_path: str | Path
) -> dict[str, Any]:
    """Atomically write validated samples as canonical UTF-8 JSONL."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        validate_sft_sample(sample)
        rows.append(
            {
                key: copy.deepcopy(sample[key])
                for key in _SFT_SAMPLE_FIELDS
            }
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(_canonical_json(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "ok": True,
        "sample_count": len(rows),
        "output_path": str(destination),
        "output_sha256": "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


__all__ = [
    "SFT_PROJECTION_VERSION",
    "SFT_SAMPLE_SCHEMA_VERSION",
    "SFTExportError",
    "build_sft_sample",
    "export_sft_jsonl",
    "project_plain_teacher_to_native_messages",
    "validate_sft_sample",
]
