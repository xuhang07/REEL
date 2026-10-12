"""Deterministic integrity helpers for canonical StoryMem trajectory shards.

The collector writes a provider-neutral interchange shard.  This module keeps
that shard self-checking; target-specific SFT exporters still need to map the
generic media blocks through the target model's real chat template.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .contract import (
    CANONICAL_TOOL_NAMES,
    PROMPT_REWRITE_MODES,
    SEMANTIC_MODULES,
    TOOL_PROTOCOL_VERSION,
    build_assessment_contract,
    build_evaluation_contract_authoring,
    canonical_tool_names,
    evaluation_check_evidence_kinds,
    get_tool_schemas,
    tool_schema_hash,
)
from .media import describe_media
from .prompts import (
    OPEN_MODEL_SYSTEM_PROMPT,
    TEACHER_PHASES,
    build_teacher_system_prompt,
)
from .source import validate_source_asset_receipt, validate_story_pe_provenance
from .state import TEACHER_CALL_LEDGER_SCHEMA_VERSION
from .util import canonical_json, sha256_file, sha256_json, sha256_text, utc_now


TRAJECTORY_SCHEMA_VERSION = "storymem.canonical.trajectory.v4"
SEMANTIC_LOCK_SCHEMA_VERSION = "storymem.canonical.semantic-lock.v1"
UPLOAD_PRIVATE_LINEAGE_SCHEMA_VERSION = (
    "storymem.canonical.upload-private-lineage.v1"
)
_FORBIDDEN_OBSERVATION_KEYS = frozenset(
    {
        "assistant_provenance",
        "dynamic_system_prompt",
        "dynamic_system_prompt_sha256",
        "private_retry_feedback",
        "teacher_reason",
    }
)

_SEMANTIC_MODULES = frozenset(SEMANTIC_MODULES)
_ASSESSMENT_FAILURES = frozenset({"fail", "uncertain"})
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_VIDEO_MATERIAL_CATEGORY_MODULE = {
    "person": "identity_wardrobe",
    "entity": "identity_wardrobe",
    "scene": "scene_environment",
    "object": "object_state",
    "style": "style_lighting",
}


def _validate_teacher_provenance(
    provenance: Any,
    *,
    label: str,
    response_status: str,
    expected_input_messages: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Validate the immutable receipt for one paid Teacher HTTP turn."""

    if not isinstance(provenance, Mapping):
        raise TrajectoryValidationError(f"{label} is missing")
    if provenance.get("assistant_source") != "remote_closed_model_api":
        raise TrajectoryValidationError(
            f"{label} does not identify the remote closed-model API"
        )
    if provenance.get("teacher_transport") != "plain_json_tool_dialogue":
        raise TrajectoryValidationError(
            f"{label} did not use the plain-JSON dialogue transport"
        )
    if response_status == "plain_json_tool_command":
        if provenance.get("plain_json_tool_command") is not True:
            raise TrajectoryValidationError(f"{label} is not a plain-JSON tool command")
    elif response_status == "malformed_response":
        if provenance.get("plain_json_tool_command") is True:
            raise TrajectoryValidationError(
                f"{label} marks a malformed response as a parsed plain-JSON command"
            )
    else:
        raise TrajectoryValidationError(f"{label} has an invalid response status")
    model = provenance.get("model")
    api_style = provenance.get("api_style")
    if not isinstance(model, str) or not model.strip():
        raise TrajectoryValidationError(f"{label} has no model identity")
    if api_style not in {"openai_chat", "gemini"}:
        raise TrajectoryValidationError(f"{label} has an invalid API style")
    for key in (
        "endpoint_sha256",
        "request_payload_sha256",
        "response_sha256",
        "dynamic_system_prompt_sha256",
        "teacher_input_messages_sha256",
    ):
        digest = provenance.get(key)
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise TrajectoryValidationError(f"{label} has an invalid {key}")
    raw_assistant_text = provenance.get("raw_assistant_text")
    if not isinstance(raw_assistant_text, str):
        raise TrajectoryValidationError(f"{label} has no raw assistant text")
    if provenance.get("assistant_text_sha256") != sha256_text(raw_assistant_text):
        raise TrajectoryValidationError(f"{label} raw assistant text hash mismatch")
    forbidden_native_receipts = {
        "native_tool_call",
        "native_call_id",
        "provider_tool_schema_sha256",
        "tool_schema_adapter",
    }
    leaked_native_receipts = sorted(forbidden_native_receipts.intersection(provenance))
    if leaked_native_receipts:
        raise TrajectoryValidationError(
            f"{label} contains native-tool transport receipts: {leaked_native_receipts}"
        )
    raw_allowed_tools = provenance.get("teacher_allowed_tools")
    if not isinstance(raw_allowed_tools, list) or not all(
        isinstance(name, str) for name in raw_allowed_tools
    ):
        raise TrajectoryValidationError(f"{label} has no valid Teacher action set")
    try:
        allowed_tools = canonical_tool_names(raw_allowed_tools)
    except ValueError as exc:
        raise TrajectoryValidationError(
            f"{label} has an invalid Teacher action set"
        ) from exc
    if list(allowed_tools) != raw_allowed_tools:
        raise TrajectoryValidationError(
            f"{label} Teacher action set is not in canonical order"
        )
    if provenance.get("tool_schema_sha256") != tool_schema_hash(allowed_tools):
        raise TrajectoryValidationError(
            f"{label} used a different state-scoped tool schema"
        )
    phase = provenance.get("phase")
    if phase not in TEACHER_PHASES:
        raise TrajectoryValidationError(f"{label} has an invalid teacher phase")
    runtime_context = provenance.get("teacher_runtime_context")
    if not isinstance(runtime_context, Mapping) or runtime_context.get("phase") != phase:
        raise TrajectoryValidationError(f"{label} has an invalid Teacher runtime context")
    if any(
        key in runtime_context
        for key in ("remaining_turns", "shot_index", "audio_perception")
    ):
        raise TrajectoryValidationError(f"{label} contains privileged runtime guidance")
    expected_system_hash = sha256_text(
        build_teacher_system_prompt(
            phase,
            allowed_tool_names=allowed_tools,
            runtime_context=runtime_context,
        )
    )
    if provenance.get("dynamic_system_prompt_sha256") != expected_system_hash:
        raise TrajectoryValidationError(f"{label} dynamic system prompt hash mismatch")
    if expected_input_messages is not None:
        if provenance.get("teacher_input_messages_sha256") != sha256_json(
            expected_input_messages
        ):
            raise TrajectoryValidationError(f"{label} Teacher history hash mismatch")
        if provenance.get("teacher_input_message_count") != len(
            expected_input_messages
        ):
            raise TrajectoryValidationError(f"{label} Teacher history count mismatch")
        expected_event_count = sum(
            1 for row in expected_input_messages if row.get("role") == "assistant"
        )
        if provenance.get("teacher_history_event_count") != expected_event_count:
            raise TrajectoryValidationError(
                f"{label} Teacher history event count mismatch"
            )
    private_feedback = runtime_context.get("private_retry_feedback", [])
    if not isinstance(private_feedback, list):
        raise TrajectoryValidationError(f"{label} has invalid private retry feedback")
    if provenance.get("private_retry_count") != len(private_feedback):
        raise TrajectoryValidationError(f"{label} private retry count mismatch")
    attempts = provenance.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise TrajectoryValidationError(f"{label} has no HTTP attempt ledger")
    previous_attempt = 0
    for attempt_index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise TrajectoryValidationError(
                f"{label} HTTP attempt {attempt_index} is invalid"
            )
        sequence = attempt.get("attempt")
        status = attempt.get("http_status")
        trace_id = attempt.get("trace_id")
        if (
            type(sequence) is not int
            or sequence != previous_attempt + 1
            or type(status) is not int
            or not isinstance(trace_id, str)
            or not trace_id.strip()
        ):
            raise TrajectoryValidationError(
                f"{label} HTTP attempt {attempt_index} has invalid ordering or fields"
            )
        if status > 0 and not _SHA256_PATTERN.fullmatch(
            str(attempt.get("response_sha256") or "")
        ):
            raise TrajectoryValidationError(
                f"{label} HTTP attempt {attempt_index} has no response receipt"
            )
        previous_attempt = sequence
    final_status = attempts[-1].get("http_status")
    if type(final_status) is not int or not 200 <= final_status < 300:
        raise TrajectoryValidationError(
            f"{label} has no terminal successful HTTP attempt"
        )
    if attempts[-1].get("response_sha256") != provenance.get("response_sha256"):
        raise TrajectoryValidationError(f"{label} terminal response hash mismatch")


def _validate_teacher_event_provenance(
    event: Mapping[str, Any],
    *,
    event_index: int,
    expected_input_messages: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Require an auditable real closed-model call for every trained action."""

    _validate_teacher_provenance(
        event.get("assistant_provenance"),
        label=f"committed event {event_index} assistant provenance",
        response_status="plain_json_tool_command",
        expected_input_messages=expected_input_messages,
    )
    provenance = event["assistant_provenance"]
    parsed_call, parsed_reason = parse_plain_json_tool_command(
        provenance["raw_assistant_text"],
        label=f"committed event {event_index} raw assistant text",
    )
    if parsed_call != event.get("tool_call"):
        raise TrajectoryValidationError(
            f"committed event {event_index} raw assistant text differs from its tool call"
        )
    if parsed_call["name"] not in provenance["teacher_allowed_tools"]:
        raise TrajectoryValidationError(
            f"committed event {event_index} used a tool outside its Teacher action set"
        )
    if parsed_reason != provenance.get("teacher_reason"):
        raise TrajectoryValidationError(
            f"committed event {event_index} raw assistant reason differs from provenance"
        )


def _source_signature(source: Any) -> tuple[str, str, str] | None:
    """Return the generation-relevant identity of a media source.

    View/selection authorization IDs prove that a source may be used, but a
    second authorization for the same immutable representation is not a
    semantic generation change.  Keeping this helper byte-for-byte aligned
    with the runtime prevents retry trajectories from manufacturing a change
    by merely viewing the same bytes again.
    """

    if source is None:
        return None
    if not isinstance(source, Mapping):
        return ("", "", "")
    return (
        str(source.get("kind") or ""),
        str(source.get("source_id") or ""),
        str(source.get("representation_id") or ""),
    )


def _binding_source_ids(value: Any) -> set[str]:
    """Collect the same authorized binding IDs used by the runtime gate."""

    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"source_id", "representation_id", "authorization_id"} and item:
                result.add(str(item))
            else:
                result.update(_binding_source_ids(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_binding_source_ids(item))
    return result


def _reference_signature(
    references: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str, str] | None, ...]:
    return tuple(_source_signature(item) for item in references)


def _condition_signature(condition: Any) -> Any:
    if condition is None:
        return None
    if not isinstance(condition, Mapping):
        return ("", None)
    return (
        str(condition.get("relation") or ""),
        _source_signature(condition.get("image_source")),
    )


def _memory_slot_signature(
    slots: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            _source_signature(slot.get("image_source")),
            _source_signature(slot.get("audio_source")),
            tuple(sorted(str(item) for item in slot.get("roles") or [])),
            tuple(sorted(str(item) for item in slot.get("subject_ids") or [])),
        )
        for slot in slots
    )


class TrajectoryValidationError(ValueError):
    """A completed shard is not a closed canonical training trajectory."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_object(text: str, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrajectoryValidationError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TrajectoryValidationError(f"{label} must encode one JSON object")
    return value


def parse_plain_json_tool_command(
    text: str, *, label: str = "closed-Teacher assistant text"
) -> tuple[dict[str, Any], str]:
    """Parse the exact text-only command emitted by a closed Teacher.

    Collection deliberately does not rely on provider-native function calls.
    Keeping this envelope strict makes the later native-tool SFT projection a
    deterministic format conversion rather than an inferred or repaired call.
    """

    value = _strict_json_object(text, label=label)
    if set(value) != {"reason", "tool_call"}:
        raise TrajectoryValidationError(
            f"{label} must contain exactly reason and tool_call"
        )
    reason = value.get("reason")
    call = value.get("tool_call")
    if not isinstance(reason, str):
        raise TrajectoryValidationError(f"{label}.reason must be a string")
    if not isinstance(call, Mapping) or set(call) != {"name", "arguments"}:
        raise TrajectoryValidationError(
            f"{label}.tool_call must contain exactly name and arguments"
        )
    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        raise TrajectoryValidationError(
            f"{label}.tool_call requires a non-empty name and object arguments"
        )
    # Re-serialize to reject non-JSON values even if a caller constructed the
    # object programmatically rather than through ``json.loads``.
    try:
        canonical_json(arguments)
    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError(
            f"{label}.tool_call.arguments are not finite JSON: {exc}"
        ) from exc
    return {"name": name, "arguments": copy.deepcopy(arguments)}, reason


def teacher_phase_bundle_hash() -> str:
    """Hash the closed-Teacher renderer across phases and tool subsets."""

    rendered: dict[str, dict[str, str]] = {}
    for phase in TEACHER_PHASES:
        rendered[phase] = {
            "all_tools": build_teacher_system_prompt(
                phase, allowed_tool_names=CANONICAL_TOOL_NAMES
            ),
            "single_tool_renderings": {
                name: build_teacher_system_prompt(
                    phase,
                    allowed_tool_names=(name,),
                    runtime_context={"phase": phase},
                )
                for name in CANONICAL_TOOL_NAMES
            },
        }
    return sha256_json(rendered)


def build_collector_semantic_lock(
    *,
    runtime_version: str,
    pe_manifest_sha256: str,
    tokenizer_profile_sha256: str,
) -> dict[str, Any]:
    """Return the immutable semantics that a checkpoint may resume under."""

    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parents[1]
    source_paths = [
        *package_root.glob("*.py"),
        repo_root / "tools" / "run_storymem_teacher.py",
    ]
    source_hashes = {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in sorted(
            source_paths,
            key=lambda item: item.relative_to(repo_root).as_posix(),
        )
    }
    lock = {
        "schema_version": SEMANTIC_LOCK_SCHEMA_VERSION,
        "runtime_version": str(runtime_version),
        "tool_protocol_version": TOOL_PROTOCOL_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "open_model_system_prompt_sha256": sha256_text(OPEN_MODEL_SYSTEM_PROMPT),
        "teacher_phase_bundle_sha256": teacher_phase_bundle_hash(),
        "tool_schema_sha256": tool_schema_hash(),
        "pe_manifest_sha256": str(pe_manifest_sha256),
        "tokenizer_profile_sha256": str(tokenizer_profile_sha256),
        "collector_source_files_sha256": source_hashes,
        "collector_source_bundle_sha256": sha256_json(source_hashes),
    }
    lock["semantic_lock_sha256"] = sha256_json(lock)
    return lock


def _media_kind(mime_type: str) -> str:
    major = mime_type.split("/", 1)[0]
    if major in {"image", "video", "audio", "text"}:
        return major
    if mime_type in {"application/json", "application/xml"}:
        return "text"
    if mime_type == "application/pdf":
        return "document"
    return major


def training_media_block(item: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one model-visible media reference with its on-disk manifest."""

    source = copy.deepcopy(dict(item))
    raw_type = str(source.pop("type", "") or "")
    declared_type = str(
        source.get("media_type") if raw_type == "media" else raw_type
    ) or str(source.get("media_type") or "")
    path = str(source.get("path") or "")
    representation_id = str(source.get("representation_id") or "")
    if not declared_type or not path or not representation_id:
        raise TrajectoryValidationError(
            "media observation requires type, path, and representation_id"
        )
    descriptor = describe_media(path)
    actual_type = _media_kind(descriptor.mime_type)
    if declared_type != actual_type:
        raise TrajectoryValidationError(
            f"media type mismatch for {path}: declared {declared_type}, detected {actual_type}"
        )
    for key, expected in (
        ("mime_type", descriptor.mime_type),
        ("size_bytes", descriptor.size_bytes),
        ("sha256", descriptor.sha256),
    ):
        if key in source and source[key] != expected:
            raise TrajectoryValidationError(
                f"media observation {key} does not match file for {path}"
            )
    block = {
        "type": "media",
        "media_type": declared_type,
        **source,
        "path": descriptor.path,
        "representation_id": representation_id,
        "mime_type": descriptor.mime_type,
        "size_bytes": descriptor.size_bytes,
        "sha256": descriptor.sha256,
    }
    manifest = {
        key: block[key]
        for key in (
            "media_type",
            "path",
            "representation_id",
            "mime_type",
            "size_bytes",
            "sha256",
        )
    }
    block["manifest_sha256"] = sha256_json(manifest)
    return block


def build_training_messages(
    *, raw_user_request: str, events: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Project committed runtime events into native tool-call messages."""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": OPEN_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": str(raw_user_request)},
    ]
    for raw_event in events:
        if not raw_event.get("committed_to_training"):
            continue
        event = copy.deepcopy(dict(raw_event))
        call_id = str(event.get("event_id") or "")
        call = event.get("tool_call")
        observation = event.get("observation")
        if not call_id or not isinstance(call, Mapping) or not isinstance(observation, Mapping):
            raise TrajectoryValidationError("committed event is missing id, tool call, or observation")
        arguments = copy.deepcopy(call.get("arguments"))
        if not isinstance(arguments, dict):
            raise TrajectoryValidationError(f"event {call_id} tool arguments must be an object")
        try:
            json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TrajectoryValidationError(
                f"event {call_id} tool arguments are not strict JSON: {exc}"
            ) from exc
        name = str(call.get("name") or "")
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": canonical_json(arguments),
                        },
                    }
                ],
            }
        )
        visible_observation = copy.deepcopy(dict(observation))
        media = visible_observation.pop("media", []) or []
        if not isinstance(media, list):
            raise TrajectoryValidationError(f"event {call_id} observation.media must be an array")
        content = [{"type": "text", "text": canonical_json(visible_observation)}]
        content.extend(training_media_block(item) for item in media)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": content,
            }
        )
    return messages


def build_teacher_input_messages(
    *, raw_user_request: str, events: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Reconstruct the exact ordinary dialogue sent to a stateless Teacher.

    Provider-native tool roles are intentionally absent.  Prior model commands
    remain their byte-exact JSON text, while each real tool observation is a
    subsequent USER-role message with any media attached to that same message.
    The exporter independently projects the same events into native tool-call
    messages through :func:`build_training_messages`.
    """

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": str(raw_user_request)}
    ]
    for event_index, raw_event in enumerate(events):
        if not raw_event.get("committed_to_training"):
            continue
        event = copy.deepcopy(dict(raw_event))
        call = event.get("tool_call")
        observation = event.get("observation")
        provenance = event.get("assistant_provenance")
        if (
            not isinstance(call, Mapping)
            or not isinstance(observation, Mapping)
            or not isinstance(provenance, Mapping)
        ):
            raise TrajectoryValidationError(
                f"Teacher history event {event_index} is incomplete"
            )
        raw_text = provenance.get("raw_assistant_text")
        if not isinstance(raw_text, str):
            raise TrajectoryValidationError(
                f"Teacher history event {event_index} has no raw assistant JSON text"
            )
        parsed_call, parsed_reason = parse_plain_json_tool_command(
            raw_text, label=f"Teacher history event {event_index} assistant text"
        )
        expected_call = {
            "name": str(call.get("name") or ""),
            "arguments": copy.deepcopy(call.get("arguments")),
        }
        if parsed_call != expected_call:
            raise TrajectoryValidationError(
                f"Teacher history event {event_index} raw assistant text differs from its tool call"
            )
        if provenance.get("teacher_reason") != parsed_reason:
            raise TrajectoryValidationError(
                f"Teacher history event {event_index} raw assistant reason differs from provenance"
            )
        visible_observation = copy.deepcopy(dict(observation))
        media = visible_observation.pop("media", []) or []
        if not isinstance(media, list):
            raise TrajectoryValidationError(
                f"Teacher history event {event_index} observation.media must be an array"
            )
        observation_envelope = {
            "tool_observation": {
                "name": expected_call["name"],
                "result": visible_observation,
            }
        }
        content: list[dict[str, Any]] = [
            {"type": "text", "text": canonical_json(observation_envelope)}
        ]
        content.extend(training_media_block(item) for item in media)
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({"role": "user", "content": content})
    return messages


def trajectory_media_manifest(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, block in enumerate(content):
            if not isinstance(block, Mapping) or block.get("type") != "media":
                continue
            manifest.append(
                {
                    "message_index": message_index,
                    "content_index": content_index,
                    **{
                        key: block.get(key)
                        for key in (
                            "media_type",
                            "path",
                            "representation_id",
                            "mime_type",
                            "size_bytes",
                            "sha256",
                            "manifest_sha256",
                        )
                    },
                }
            )
    return manifest


def _contains_forbidden_observation_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in _FORBIDDEN_OBSERVATION_KEYS:
                return str(key)
            found = _contains_forbidden_observation_key(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _contains_forbidden_observation_key(child)
            if found:
                return found
    return None


def build_upload_private_lineage(
    *, assets: Mapping[str, Any], source_assets_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal canonical upload bindings outside model-visible observations."""

    rows: list[dict[str, Any]] = []
    for receipt_row in source_assets_receipt.get("assets") or []:
        asset_id = str(receipt_row.get("asset_id") or "")
        asset = assets.get(asset_id)
        if not isinstance(asset, Mapping):
            raise ValueError(f"source receipt asset {asset_id!r} has no runtime asset")
        rows.append(
            {
                "asset_id": asset_id,
                "canonical_material_tags": copy.deepcopy(
                    asset.get("canonical_material_tags") or []
                ),
                "subject_ids": copy.deepcopy(asset.get("subject_ids") or []),
                "speaker_binding": copy.deepcopy(asset.get("speaker_binding")),
            }
        )
    body = {
        "schema_version": UPLOAD_PRIVATE_LINEAGE_SCHEMA_VERSION,
        "source_assets_receipt_sha256": source_assets_receipt.get(
            "receipt_sha256"
        ),
        "assets": rows,
    }
    return {**body, "lineage_sha256": sha256_json(body)}


def _validate_upload_private_lineage(
    value: Any, *, source_assets_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrajectoryValidationError("trajectory has no private upload lineage")
    lineage = copy.deepcopy(dict(value))
    if set(lineage) != {
        "schema_version",
        "source_assets_receipt_sha256",
        "assets",
        "lineage_sha256",
    }:
        raise TrajectoryValidationError("private upload lineage has an invalid shape")
    if lineage.get("schema_version") != UPLOAD_PRIVATE_LINEAGE_SCHEMA_VERSION:
        raise TrajectoryValidationError("private upload lineage schema is incompatible")
    if lineage.get("source_assets_receipt_sha256") != source_assets_receipt.get(
        "receipt_sha256"
    ):
        raise TrajectoryValidationError(
            "private upload lineage is bound to a different source receipt"
        )
    body = {key: child for key, child in lineage.items() if key != "lineage_sha256"}
    if lineage.get("lineage_sha256") != sha256_json(body):
        raise TrajectoryValidationError("private upload lineage hash mismatch")
    receipt_rows = {
        str(row.get("asset_id") or ""): row
        for row in source_assets_receipt.get("assets") or []
        if isinstance(row, Mapping)
    }
    rows = lineage.get("assets")
    if not isinstance(rows, list):
        raise TrajectoryValidationError("private upload lineage assets must be an array")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "asset_id",
            "canonical_material_tags",
            "subject_ids",
            "speaker_binding",
        }:
            raise TrajectoryValidationError("private upload lineage row is malformed")
        asset_id = str(row.get("asset_id") or "")
        if not asset_id or asset_id in seen or asset_id not in receipt_rows:
            raise TrajectoryValidationError(
                "private upload lineage has an unknown or duplicate asset"
            )
        seen.add(asset_id)
        tags = row.get("canonical_material_tags")
        subject_ids = row.get("subject_ids")
        if (
            not isinstance(tags, list)
            or not isinstance(subject_ids, list)
            or any(not isinstance(item, str) or not item for item in [*tags, *subject_ids])
            or len(tags) != len(set(tags))
            or len(subject_ids) != len(set(subject_ids))
        ):
            raise TrajectoryValidationError(
                "private upload lineage has invalid canonical bindings"
            )
        receipt_binding = receipt_rows[asset_id].get("speaker_binding")
        if row.get("speaker_binding") != receipt_binding:
            raise TrajectoryValidationError(
                "private upload lineage speaker binding differs from source receipt"
            )
        if isinstance(receipt_binding, Mapping) and subject_ids != [
            receipt_binding.get("subject_id")
        ]:
            raise TrajectoryValidationError(
                "private upload lineage audio subject differs from source receipt"
            )
    if seen != set(receipt_rows):
        raise TrajectoryValidationError(
            "private upload lineage does not cover the source receipt"
        )
    return lineage


def _validate_causal_ledger(
    value: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> None:
    """Independently verify the committed tool ledger's causal closure.

    Runtime checks are the first line of defence, but a persisted shard is a
    training artifact and must remain auditable without trusting a mutable
    checkpoint.  This verifier intentionally uses only public calls/results
    plus the three immutable lineage tables embedded in the shard.
    """

    if not events:
        raise TrajectoryValidationError("trajectory has no committed tool events")

    raw_events = value.get("events")
    if not isinstance(raw_events, list) or any(
        not isinstance(event, Mapping) for event in raw_events
    ):
        raise TrajectoryValidationError("trajectory events must all be objects")
    event_ids: set[str] = set()
    previous_sequence = -1
    for index, event in enumerate(raw_events):
        event_id = str(event.get("event_id") or "")
        match = re.fullmatch(r"event_(\d+)", event_id)
        if not match:
            raise TrajectoryValidationError(
                f"event {index} has a non-canonical event_id"
            )
        if event_id in event_ids:
            raise TrajectoryValidationError(f"duplicate event_id: {event_id}")
        sequence = int(match.group(1))
        if sequence <= previous_sequence:
            raise TrajectoryValidationError("event IDs are not strictly ordered")
        event_ids.add(event_id)
        previous_sequence = sequence

    prompt_artifacts = value.get("prompt_artifacts")
    pe_receipts = value.get("pe_receipts")
    stored_selections = value.get("selections")
    if not isinstance(prompt_artifacts, Mapping):
        raise TrajectoryValidationError("trajectory prompt_artifacts must be an object")
    if not isinstance(pe_receipts, Mapping):
        raise TrajectoryValidationError("trajectory pe_receipts must be an object")
    if not isinstance(stored_selections, Mapping):
        raise TrajectoryValidationError("trajectory selections must be an object")

    shot_request_id = str(value.get("shot_request_id") or "")
    prompt_rewrite_mode = str(value.get("prompt_rewrite_mode") or "")
    private_upload_rows = {
        str(row.get("asset_id") or ""): copy.deepcopy(dict(row))
        for row in (value.get("upload_private_lineage") or {}).get("assets", [])
        if isinstance(row, Mapping)
    }
    observations: dict[str, str] = {}
    observation_payloads: dict[str, dict[str, Any]] = {}
    observation_sequence: dict[str, int] = {}
    latest_search_id: str | None = None
    search_candidates: dict[str, set[str]] = {}
    view_authorizations: dict[tuple[str, str, str, str], str] = {}
    viewed_source_metadata: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    registry_snapshot_id: str | None = None
    registry_next_cursor: str | None = None
    registry_complete = False
    registry_closed = False
    registry_tags: dict[str, dict[str, Any]] = {}
    registry_shot_plan: dict[str, Any] | None = None
    registry_active_material_tags: list[str] | None = None
    registry_capabilities: dict[str, Any] | None = None
    uploaded_asset_ids: set[str] = set()
    uploaded_assets: dict[str, dict[str, Any]] = {}
    previous_tail_ids: set[str] = set()
    seen_receipts: dict[str, dict[str, Any]] = {}
    seen_artifacts: dict[str, dict[str, Any]] = {}
    used_artifacts: set[str] = set()
    pending_receipt: str | None = None
    pending_artifact: str | None = None
    pending_candidate: tuple[str, str] | None = None
    groups: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    inspections: dict[str, dict[str, Any]] = {}
    assessments: dict[str, dict[str, Any]] = {}
    seen_selections: dict[str, dict[str, Any]] = {}
    selection_order: list[str] = []
    consumed_image_selections: set[str] = set()
    video_selection_id: str | None = None
    memory_update_id: str | None = None
    generation_count = {"image": 0, "video": 0}
    budgets: dict[str, int] = {}

    def event_parts(event: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        call = event.get("tool_call")
        observation = event.get("observation")
        if not isinstance(call, Mapping) or not isinstance(call.get("arguments"), Mapping):
            raise TrajectoryValidationError("causal ledger event has no valid tool call")
        if not isinstance(observation, Mapping) or observation.get("ok") is not True:
            raise TrajectoryValidationError("committed tool event must have ok=true")
        return (
            str(call.get("name") or ""),
            dict(call["arguments"]),
            dict(observation),
        )

    def add_observation(observation: Mapping[str, Any], kind: str) -> str:
        observation_id = str(observation.get("observation_id") or "")
        if not observation_id:
            raise TrajectoryValidationError(f"{kind} result has no observation_id")
        if observation_id in observations:
            raise TrajectoryValidationError(
                f"duplicate observation_id: {observation_id}"
            )
        observations[observation_id] = kind
        observation_payloads[observation_id] = copy.deepcopy(dict(observation))
        observation_sequence[observation_id] = len(observation_sequence)
        return observation_id

    def source_observation_id(source: Any) -> str:
        """Resolve one public source binding to the observation that authorized it."""

        if not isinstance(source, Mapping):
            raise TrajectoryValidationError("generation contains a malformed media source")
        kind, source_id, representation_id = _source_signature(source) or ("", "", "")
        authorization_id = str(source.get("authorization_id") or "")
        if kind == "generated_image":
            selection = seen_selections.get(authorization_id)
            candidate = candidates.get(source_id)
            if (
                not selection
                or selection.get("target") != "image"
                or selection.get("selected_candidate_id") != source_id
                or not candidate
                or candidate.get("target") != "image"
                or candidate.get("representation_id") != representation_id
                or not candidate.get("inspection_id")
            ):
                raise TrajectoryValidationError(
                    "generated image source lacks exact selection authorization"
                )
            inspection = inspections.get(str(candidate["inspection_id"]))
            observation_id = str(inspection.get("observation_id") or "") if inspection else ""
            if not observation_id:
                raise TrajectoryValidationError(
                    "generated image source has no inspection observation"
                )
            return observation_id
        key = (kind, source_id, representation_id, authorization_id)
        observation_id = view_authorizations.get(key)
        if not observation_id:
            raise TrajectoryValidationError(
                f"{kind or 'media'} source lacks exact current-shot view authorization"
            )
        return observation_id

    def source_metadata(source: Any) -> dict[str, Any]:
        """Return source metadata, privately enriching uploaded bindings."""

        if not isinstance(source, Mapping):
            raise TrajectoryValidationError("generation contains a malformed media source")
        kind, source_id, representation_id = _source_signature(source) or ("", "", "")
        authorization_id = str(source.get("authorization_id") or "")
        if kind == "generated_image":
            candidate = candidates.get(source_id)
            if not candidate or candidate.get("representation_id") != representation_id:
                raise TrajectoryValidationError(
                    "generated image source metadata is unavailable"
                )
            return {
                "subject_ids": copy.deepcopy(candidate.get("subject_ids") or []),
                "representation": copy.deepcopy(candidate.get("representation") or {}),
            }
        metadata = viewed_source_metadata.get(
            (kind, source_id, representation_id, authorization_id)
        )
        if metadata is None:
            raise TrajectoryValidationError(
                f"{kind or 'media'} source metadata lacks exact view authorization"
            )
        resolved = copy.deepcopy(metadata)
        if kind == "uploaded_asset":
            private = private_upload_rows.get(source_id)
            if not private:
                raise TrajectoryValidationError(
                    "uploaded source lacks private canonical lineage"
                )
            resolved.update(
                {
                    "canonical_material_tags": copy.deepcopy(
                        private.get("canonical_material_tags") or []
                    ),
                    "subject_ids": copy.deepcopy(private.get("subject_ids") or []),
                    "speaker_binding": copy.deepcopy(private.get("speaker_binding")),
                    "transport_eligible": isinstance(
                        private.get("speaker_binding"), Mapping
                    ),
                    "verification_basis": (
                        private.get("speaker_binding", {}).get("basis")
                        if isinstance(private.get("speaker_binding"), Mapping)
                        else None
                    ),
                }
            )
        return resolved

    def candidate_quality(candidate_id: str) -> str:
        candidate = candidates.get(candidate_id)
        assessment = (
            assessments.get(str(candidate.get("assessment_id") or ""))
            if candidate
            else None
        )
        return str(assessment.get("quality") or "unassessed") if assessment else "unassessed"

    def primary_use(candidate: Mapping[str, Any]) -> str:
        if candidate.get("target") == "video":
            return "final_video"
        return "condition" if candidate.get("purpose") == "first_frame" else "memory_slot"

    def primary_use_authorized(
        candidate_id: str, *, allow_failed_best_effort: bool = False
    ) -> bool:
        candidate = candidates.get(candidate_id)
        assessment = (
            assessments.get(str(candidate.get("assessment_id") or ""))
            if candidate
            else None
        )
        if not candidate or not assessment:
            return False
        wanted = primary_use(candidate)
        accepted = {"pass", "capability_blocked"}
        if allow_failed_best_effort:
            accepted.update(_ASSESSMENT_FAILURES)
        results = assessment.get("use_result_by_id") or {}
        return any(
            contract.get("use") == wanted
            and contract.get("source_ids") == [candidate_id]
            and (results.get(str(contract.get("use_contract_id") or "")) or {}).get(
                "result"
            )
            in accepted
            for contract in assessment.get("use_contracts") or []
        )

    for index, event in enumerate(events):
        name, arguments, observation = event_parts(event)
        if index == 0 and name != "view_material_registry":
            raise TrajectoryValidationError(
                "the first committed tool must be view_material_registry"
            )
        if registry_next_cursor is not None and name != "view_material_registry":
            raise TrajectoryValidationError(
                "registry pagination was not completed before another tool"
            )
        if pending_receipt is not None and name != "register_prompt":
            raise TrajectoryValidationError(
                f"PE receipt {pending_receipt} was not registered immediately"
            )
        if pending_artifact is not None:
            target = seen_artifacts[pending_artifact]["target"]
            required = "generate_image" if target == "image" else "generate_shot"
            if name != required:
                raise TrajectoryValidationError(
                    f"prompt artifact {pending_artifact} was not consumed immediately"
                )
        if pending_candidate is not None:
            candidate_id, stage = pending_candidate
            required = "inspect_candidate" if stage == "inspect" else "submit_candidate_assessment"
            if name != required:
                raise TrajectoryValidationError(
                    f"candidate {candidate_id} did not proceed to {required}"
                )
        if video_selection_id is not None and name not in {
            "update_memory_bank",
            "submit_shot",
        }:
            raise TrajectoryValidationError(
                "only memory update or submit is legal after video selection"
            )

        if name == "view_material_registry":
            if registry_closed:
                raise TrajectoryValidationError("registry snapshot was reopened")
            if registry_snapshot_id is None:
                if arguments.get("snapshot_id") is not None or arguments.get("cursor") is not None:
                    raise TrajectoryValidationError(
                        "the first registry call must start with null snapshot/cursor"
                    )
                registry_snapshot_id = str(observation.get("snapshot_id") or "")
                if not registry_snapshot_id:
                    raise TrajectoryValidationError("registry result has no snapshot_id")
            elif (
                arguments.get("snapshot_id") != registry_snapshot_id
                or arguments.get("cursor") != registry_next_cursor
            ):
                raise TrajectoryValidationError("registry pagination lineage is broken")
            if observation.get("snapshot_id") != registry_snapshot_id:
                raise TrajectoryValidationError("registry result changed snapshot_id")
            if observation.get("shot_request_id") != shot_request_id:
                raise TrajectoryValidationError("registry result has the wrong shot_request_id")
            add_observation(observation, "material_registry")
            next_cursor = observation.get("next_cursor")
            registry_next_cursor = None if next_cursor is None else str(next_cursor)
            registry_complete = registry_next_cursor is None
            if registry_complete:
                registry_closed = True
            phase_budget = observation.get("phase_budget")
            if not isinstance(phase_budget, Mapping) or any(
                not isinstance(phase_budget.get(key), int)
                or isinstance(phase_budget.get(key), bool)
                or int(phase_budget[key]) < 1
                for key in ("image_attempts", "video_attempts")
            ):
                raise TrajectoryValidationError(
                    "registry result has an invalid generation budget"
                )
            for key in ("image_attempts", "video_attempts"):
                budgets[key] = int(phase_budget[key])
            capabilities = observation.get("capabilities")
            if not isinstance(capabilities, Mapping):
                raise TrajectoryValidationError(
                    "registry result has no frozen capabilities"
                )
            audio_perception = capabilities.get("audio_perception")
            audio_transport = capabilities.get("audio_transport")
            max_memory_slots = capabilities.get("max_memory_slots")
            observed_prompt_rewrite_mode = capabilities.get(
                "prompt_rewrite_mode"
            )
            if (
                audio_transport not in {"disabled", "enabled"}
                or audio_perception not in {"disabled", "verified"}
                or (
                not isinstance(max_memory_slots, int)
                or isinstance(max_memory_slots, bool)
                or not 0 <= max_memory_slots <= 7
                )
                or observed_prompt_rewrite_mode != prompt_rewrite_mode
            ):
                raise TrajectoryValidationError(
                    "registry result has invalid capabilities"
                )
            if registry_capabilities is None:
                registry_capabilities = copy.deepcopy(dict(capabilities))
            elif dict(capabilities) != registry_capabilities:
                raise TrajectoryValidationError(
                    "registry capabilities changed during pagination"
                )
            story_prompt_state = observation.get("story_prompt_state")
            if not isinstance(story_prompt_state, Mapping):
                raise TrajectoryValidationError(
                    "registry result has no frozen story_prompt_state"
                )
            current_shot_plan = story_prompt_state.get("current_shot_plan")
            if not isinstance(current_shot_plan, Mapping):
                raise TrajectoryValidationError(
                    "registry story_prompt_state has no current_shot_plan"
                )
            if (
                story_prompt_state.get("outer_shot_index") != value.get("shot_index")
                or current_shot_plan.get("shot_index") != value.get("shot_index")
                or current_shot_plan.get("raw_request") != value.get("raw_user_request")
            ):
                raise TrajectoryValidationError(
                    "registry current_shot_plan differs from the trajectory shot"
                )
            active_material_tags = current_shot_plan.get("active_material_tags")
            if not isinstance(active_material_tags, list) or any(
                not isinstance(tag, str) or not tag or tag != tag.strip()
                for tag in active_material_tags
            ):
                raise TrajectoryValidationError(
                    "registry current_shot_plan has invalid active_material_tags"
                )
            if len(active_material_tags) != len(set(active_material_tags)):
                raise TrajectoryValidationError(
                    "registry current_shot_plan repeats an active material tag"
                )
            if registry_shot_plan is None:
                registry_shot_plan = copy.deepcopy(dict(current_shot_plan))
                registry_active_material_tags = list(active_material_tags)
            elif (
                dict(current_shot_plan) != registry_shot_plan
                or active_material_tags != registry_active_material_tags
            ):
                raise TrajectoryValidationError(
                    "registry current_shot_plan changed during pagination"
                )
            materials = observation.get("materials") or []
            if not isinstance(materials, list):
                raise TrajectoryValidationError(
                    "registry materials must be an array"
                )
            for material in materials:
                if not isinstance(material, Mapping):
                    raise TrajectoryValidationError(
                        "registry contains a malformed material"
                    )
                tag = str(material.get("tag") or "")
                if not tag or tag in registry_tags:
                    raise TrajectoryValidationError(
                        "registry material tag is missing or duplicated"
                    )
                registry_tags[tag] = dict(material)
            if registry_complete:
                unresolved_active_tags = sorted(
                    set(registry_active_material_tags or []) - set(registry_tags)
                )
                if unresolved_active_tags:
                    raise TrajectoryValidationError(
                        "registry current_shot_plan references an unknown active material"
                    )
                unsupported_active_categories = {
                    tag: str(registry_tags[tag].get("category") or "")
                    for tag in registry_active_material_tags or []
                    if str(registry_tags[tag].get("category") or "")
                    not in _VIDEO_MATERIAL_CATEGORY_MODULE
                }
                if unsupported_active_categories:
                    raise TrajectoryValidationError(
                        "registry active material has an unsupported category"
                    )
                for private_upload in private_upload_rows.values():
                    canonical_tags = list(
                        private_upload.get("canonical_material_tags") or []
                    )
                    subject_ids = list(private_upload.get("subject_ids") or [])
                    if any(tag not in registry_tags for tag in canonical_tags):
                        raise TrajectoryValidationError(
                            "private upload lineage references an unknown canonical tag"
                        )
                    expected_subjects = [
                        tag
                        for tag in canonical_tags
                        if registry_tags[tag].get("category") == "person"
                    ]
                    if subject_ids != expected_subjects:
                        raise TrajectoryValidationError(
                            "private upload lineage has inconsistent person subjects"
                        )
            pending_uploads = observation.get("pending_uploads") or []
            if not isinstance(pending_uploads, list):
                raise TrajectoryValidationError(
                    "registry pending_uploads must be an array"
                )
            for pending_upload in pending_uploads:
                if not isinstance(pending_upload, Mapping):
                    continue
                leaked_upload_fields = {
                    "canonical_material_tags",
                    "subject_ids",
                    "tag_binding_provenance",
                    "speaker_binding",
                    "transport_eligible",
                    "audio_quality_status",
                    "transport_description",
                    "transport_authorization_id",
                    "quality_authorization_id",
                }.intersection(pending_upload)
                if leaked_upload_fields:
                    raise TrajectoryValidationError(
                        "registry pending upload leaked backend bindings: "
                        + ", ".join(sorted(leaked_upload_fields))
                    )
                asset_id = str(
                    pending_upload.get("asset_id")
                    or pending_upload.get("upload_id")
                    or pending_upload.get("id")
                    or ""
                )
                if not asset_id:
                    raise TrajectoryValidationError(
                        "registry pending upload has no asset ID"
                    )
                uploaded_asset_ids.add(asset_id)
                uploaded_assets[asset_id] = copy.deepcopy(dict(pending_upload))
            previous_tail = observation.get("previous_tail")
            if isinstance(previous_tail, Mapping):
                tail_id = str(
                    previous_tail.get("source_id")
                    or previous_tail.get("representation_id")
                    or previous_tail.get("id")
                    or ""
                )
                if tail_id:
                    previous_tail_ids.add(tail_id)
            continue

        if not registry_complete:
            raise TrajectoryValidationError("tool used before registry completion")

        if name == "search_memory_bank":
            search_id = str(observation.get("search_id") or "")
            if not search_id or search_id in search_candidates:
                raise TrajectoryValidationError("memory search_id is missing or duplicated")
            rows = observation.get("candidates") or []
            if not isinstance(rows, list):
                raise TrajectoryValidationError("memory search candidates must be an array")
            search_candidates[search_id] = {
                str(row.get("memory_id") or "")
                for row in rows
                if isinstance(row, Mapping)
            }
            latest_search_id = search_id
            add_observation(observation, "memory_search")
            continue

        if name == "view_memories":
            requested_items = list(arguments.get("items") or [])
            returned_items = observation.get("items") or []
            if not isinstance(returned_items, list) or len(returned_items) != len(
                requested_items
            ):
                raise TrajectoryValidationError(
                    "memory view result does not match its requested items"
                )
            observation_id = add_observation(observation, "memory_view")
            authorization_id = str(observation.get("authorization_id") or "")
            if not authorization_id:
                raise TrajectoryValidationError(
                    "memory view result has no authorization_id"
                )
            view_media = observation.get("media") or []
            if not isinstance(view_media, list) or any(
                not isinstance(row, Mapping) for row in view_media
            ):
                raise TrajectoryValidationError("memory view media must be an array")
            for item, returned in zip(requested_items, returned_items):
                if not isinstance(item, Mapping) or not isinstance(returned, Mapping):
                    raise TrajectoryValidationError("memory view item is malformed")
                if item.get("kind") == "memory":
                    search_id = str(item.get("search_id") or "")
                    memory_id = str(item.get("id") or "")
                    if (
                        search_id != latest_search_id
                        or memory_id not in search_candidates.get(search_id, set())
                    ):
                        raise TrajectoryValidationError(
                            "memory view is not authorized by the latest search"
                        )
                kind = str(item.get("kind") or "")
                source_id = str(item.get("id") or "")
                if kind == "uploaded_asset" and source_id not in uploaded_asset_ids:
                    raise TrajectoryValidationError(
                        "memory view requested an upload absent from the registry snapshot"
                    )
                if kind == "previous_tail" and source_id not in previous_tail_ids:
                    raise TrajectoryValidationError(
                        "memory view requested a previous tail absent from the registry snapshot"
                    )
                if returned.get("kind") != kind or returned.get("id") != source_id:
                    raise TrajectoryValidationError(
                        "memory view result changed a requested source"
                    )
                if returned.get("authorization_id") != authorization_id:
                    raise TrajectoryValidationError(
                        "memory view result changed authorization_id"
                    )
                if kind == "uploaded_asset":
                    leaked_upload_fields = {
                        "canonical_material_tags",
                        "subject_ids",
                        "tag_binding_provenance",
                        "speaker_binding",
                        "source_declared_tags",
                        "tags",
                        "source_declared_tags",
                        "tags",
                    }.intersection(returned)
                    if leaked_upload_fields:
                        raise TrajectoryValidationError(
                            "uploaded asset view leaked backend bindings: "
                            + ", ".join(sorted(leaked_upload_fields))
                        )
                    private_upload = private_upload_rows.get(source_id)
                    if not private_upload:
                        raise TrajectoryValidationError(
                            "uploaded asset view lacks private canonical lineage"
                        )
                    declaration = uploaded_assets.get(source_id) or {}
                    if returned.get("declared_tags") != declaration.get(
                        "declared_tags", []
                    ):
                        raise TrajectoryValidationError(
                            "uploaded asset view changed the user's declared tags"
                        )
                    binding = private_upload.get("speaker_binding")
                    if isinstance(binding, Mapping):
                        if returned.get("speaker_name") != binding.get(
                            "declared_name"
                        ):
                            raise TrajectoryValidationError(
                                "uploaded audio view changed the declared speaker name"
                            )
                        if returned.get("transport_eligible") is not True:
                            raise TrajectoryValidationError(
                                "uploaded audio view omitted its transport capability"
                            )
                representations = [returned.get("representation")]
                supplemental = returned.get("supplemental_representations") or []
                if not isinstance(supplemental, list):
                    raise TrajectoryValidationError(
                        "memory view supplemental_representations must be an array"
                    )
                representations.extend(supplemental)
                for representation in representations:
                    if not isinstance(representation, Mapping):
                        raise TrajectoryValidationError(
                            "memory view result has a malformed representation"
                        )
                    representation_id = str(
                        representation.get("representation_id") or ""
                    )
                    if not representation_id:
                        raise TrajectoryValidationError(
                            "memory view representation has no representation_id"
                        )
                    key = (kind, source_id, representation_id, authorization_id)
                    if key in view_authorizations:
                        raise TrajectoryValidationError(
                            "memory view repeated an authorization binding"
                        )
                    view_authorizations[key] = observation_id
                    viewed_source_metadata[key] = copy.deepcopy(dict(returned))
                    if str(representation.get("media_type") or "") == "audio":
                        matching_media = [
                            row
                            for row in view_media
                            if row.get("source_id") == source_id
                            and row.get("representation_id") == representation_id
                        ]
                        perception_enabled = bool(
                            registry_capabilities
                            and registry_capabilities.get("audio_perception")
                            == "verified"
                        )
                        if perception_enabled:
                            if len(matching_media) != 1:
                                raise TrajectoryValidationError(
                                    "verified audio view must expose its exact media bytes"
                                )
                        elif matching_media or returned.get("perception_exposed") is not False:
                            raise TrajectoryValidationError(
                                "audio-disabled view exposed audio bytes or claimed perception"
                            )
            continue

        if name == "get_pe_skill":
            if prompt_rewrite_mode == "passthrough_disabled":
                raise TrajectoryValidationError(
                    "passthrough_disabled trajectory called get_pe_skill"
                )
            if arguments.get("shot_request_id") != shot_request_id:
                raise TrajectoryValidationError("PE call has the wrong shot_request_id")
            evidence_ids = list(arguments.get("context_observation_ids") or [])
            if any(
                observations.get(str(item))
                not in {"material_registry", "memory_view", "candidate_inspection"}
                for item in evidence_ids
            ) or not any(observations.get(str(item)) == "material_registry" for item in evidence_ids):
                raise TrajectoryValidationError("PE call has ungrounded observation evidence")
            if evidence_ids != sorted(
                evidence_ids, key=lambda item: observation_sequence[str(item)]
            ):
                raise TrajectoryValidationError(
                    "PE observation evidence is not in visible order"
                )
            operation = str(arguments.get("operation") or "")
            outstanding = [
                (group_id, group.get("retry"))
                for group_id, group in groups.items()
                if group.get("retry") is not None
            ]
            if operation == "compose" and outstanding:
                raise TrajectoryValidationError("compose bypassed an outstanding retry")
            if operation == "revise":
                assessment_id = str(arguments.get("assessment_id") or "")
                parent_id = str(arguments.get("parent_prompt_artifact_id") or "")
                assessment = assessments.get(assessment_id)
                parent = seen_artifacts.get(parent_id)
                if not assessment or not parent:
                    raise TrajectoryValidationError("revision lineage is missing")
                candidate = candidates.get(str(assessment.get("candidate_id") or ""))
                group = groups.get(str(candidate.get("group_id") or "")) if candidate else None
                retry = group.get("retry") if group else None
                if (
                    not retry
                    or retry.get("assessment_id") != assessment_id
                    or retry.get("candidate_id") != assessment.get("candidate_id")
                    or list(retry.get("axes") or []) != list(arguments.get("axes") or [])
                    or assessment.get("candidate_id") != parent.get("last_candidate_id")
                ):
                    raise TrajectoryValidationError("revision does not serve the outstanding retry")
                inspection = inspections.get(str(assessment.get("inspection_id") or ""))
                if (
                    not inspection
                    or inspection.get("observation_id") not in evidence_ids
                ):
                    raise TrajectoryValidationError(
                        "revision PE omits its candidate inspection evidence"
                    )
                if parent.get("target") != arguments.get("target"):
                    raise TrajectoryValidationError(
                        "revision PE changed its parent target"
                    )
                if parent.get("purpose") != arguments.get("purpose"):
                    raise TrajectoryValidationError(
                        "revision PE changed its parent purpose"
                    )
                if parent.get("generation_scope") != arguments.get(
                    "generation_scope"
                ):
                    raise TrajectoryValidationError(
                        "revision PE changed its generation scope"
                    )
            context = arguments.get("prompt_context") or {}
            condition = context.get("condition") if isinstance(context, Mapping) else None
            if condition is not None:
                condition_observation_id = source_observation_id(
                    condition.get("image_source")
                    if isinstance(condition, Mapping)
                    else None
                )
                if condition_observation_id not in evidence_ids:
                    raise TrajectoryValidationError(
                        "PE call omits its exact condition observation"
                    )
            receipt_id = str(observation.get("skill_application_receipt_id") or "")
            if not receipt_id or receipt_id in seen_receipts:
                raise TrajectoryValidationError("PE receipt ID is missing or duplicated")
            stored = pe_receipts.get(receipt_id)
            if not isinstance(stored, Mapping):
                raise TrajectoryValidationError("PE result has no matching stored receipt")
            for key in (
                "target",
                "operation",
                "purpose",
                "generation_scope",
                "prompt_context",
                "parent_prompt_artifact_id",
                "assessment_id",
                "axes",
            ):
                if stored.get(key) != arguments.get(key):
                    raise TrajectoryValidationError(
                        f"stored PE receipt differs on {key}"
                    )
            if list(stored.get("context_observation_ids") or []) != evidence_ids:
                raise TrajectoryValidationError("stored PE receipt evidence differs")
            application_payload = {
                "target": arguments.get("target"),
                "operation": arguments.get("operation"),
                "purpose": arguments.get("purpose"),
                "context": arguments.get("prompt_context"),
                "generation_scope": arguments.get("generation_scope"),
                "evidence_ids": evidence_ids,
                "parent": arguments.get("parent_prompt_artifact_id"),
                "assessment": arguments.get("assessment_id"),
                "axes": list(arguments.get("axes") or []),
            }
            if stored.get("application_key") != sha256_json(application_payload):
                raise TrajectoryValidationError(
                    "stored PE receipt application key is invalid"
                )
            public_bundle = {
                key: copy.deepcopy(item)
                for key, item in observation.items()
                if key not in {"ok", "skill_application_receipt_id"}
            }
            if stored.get("bundle") != public_bundle:
                raise TrajectoryValidationError(
                    "stored PE receipt bundle differs from its observation"
                )
            seen_receipts[receipt_id] = dict(stored)
            pending_receipt = receipt_id
            continue

        if name == "register_prompt":
            if arguments.get("shot_request_id") != shot_request_id:
                raise TrajectoryValidationError(
                    "register_prompt has the wrong shot_request_id"
                )
            raw_receipt_id = arguments.get("skill_application_receipt_id")
            if raw_receipt_id is None:
                if prompt_rewrite_mode != "passthrough_disabled":
                    raise TrajectoryValidationError(
                        "rewrite_enabled register_prompt omitted its PE receipt"
                    )
                if pending_receipt is not None:
                    raise TrajectoryValidationError(
                        "direct registration bypassed a pending PE receipt"
                    )
                target = str(arguments.get("target") or "")
                purpose = str(arguments.get("purpose") or "")
                if (
                    target not in {"image", "video"}
                    or arguments.get("operation") != "compose"
                    or arguments.get("parent_prompt_artifact_id") is not None
                    or arguments.get("assessment_id") is not None
                    or list(arguments.get("axes") or [])
                    or (target == "video" and purpose != "shot")
                    or (
                        target == "image"
                        and purpose not in {"first_frame", "reference_asset"}
                    )
                ):
                    raise TrajectoryValidationError(
                        "passthrough registration is not compose-only"
                    )
                if arguments.get("text") != value.get("raw_user_request"):
                    raise TrajectoryValidationError(
                        "passthrough prompt differs from the raw USER bytes"
                    )
                evidence_ids = list(arguments.get("evidence_ids") or [])
                if (
                    any(
                        observations.get(str(item))
                        not in {
                            "material_registry",
                            "memory_view",
                            "candidate_inspection",
                        }
                        for item in evidence_ids
                    )
                    or not any(
                        observations.get(str(item)) == "material_registry"
                        for item in evidence_ids
                    )
                    or evidence_ids
                    != sorted(
                        evidence_ids,
                        key=lambda item: observation_sequence[str(item)],
                    )
                ):
                    raise TrajectoryValidationError(
                        "passthrough registration has ungrounded observation evidence"
                    )
                outstanding = [
                    group_id
                    for group_id, group in groups.items()
                    if group.get("retry") is not None
                ]
                if outstanding:
                    raise TrajectoryValidationError(
                        "passthrough compose bypassed an outstanding retry"
                    )
                context = arguments.get("prompt_context") or {}
                condition = (
                    context.get("condition")
                    if target == "video" and isinstance(context, Mapping)
                    else None
                )
                if condition is not None:
                    condition_observation_id = source_observation_id(
                        condition.get("image_source")
                        if isinstance(condition, Mapping)
                        else None
                    )
                    if condition_observation_id not in evidence_ids:
                        raise TrajectoryValidationError(
                            "passthrough registration omits its condition observation"
                        )
                artifact_id = str(observation.get("prompt_artifact_id") or "")
                stored = prompt_artifacts.get(artifact_id)
                if (
                    not artifact_id
                    or artifact_id in seen_artifacts
                    or not isinstance(stored, Mapping)
                ):
                    raise TrajectoryValidationError(
                        "direct prompt artifact lineage is missing"
                    )
                for key in (
                    "target",
                    "operation",
                    "purpose",
                    "generation_scope",
                    "prompt_context",
                    "parent_prompt_artifact_id",
                    "assessment_id",
                    "axes",
                ):
                    if stored.get(key) != arguments.get(key):
                        raise TrajectoryValidationError(
                            f"stored direct prompt artifact differs on {key}"
                        )
                text = str(arguments.get("text") or "")
                expected_validation = {
                    "mode": "passthrough_disabled",
                    "char_count": len(text),
                    "byte_count": len(text.encode("utf-8")),
                    "token_count": None,
                }
                if (
                    stored.get("text") != text
                    or stored.get("text_sha256") != sha256_text(text)
                    or stored.get("skill_application_receipt_id") is not None
                    or list(stored.get("receipt_chain") or [])
                    or stored.get("prompt_rewrite_mode")
                    != "passthrough_disabled"
                    or list(stored.get("evidence_ids") or []) != evidence_ids
                    or stored.get("prompt_context_sha256")
                    != sha256_json(arguments.get("prompt_context"))
                    or stored.get("revision_count") != 0
                    or stored.get("validation") != expected_validation
                ):
                    raise TrajectoryValidationError(
                        "stored passthrough prompt artifact is invalid"
                    )
                if (
                    observation.get("text_sha256") != stored.get("text_sha256")
                    or observation.get("prompt_context_sha256")
                    != stored.get("prompt_context_sha256")
                    or list(observation.get("receipt_chain") or [])
                    or observation.get("revision_count") != 0
                    or any(
                        observation.get(key) != expected_validation[key]
                        for key in expected_validation
                    )
                ):
                    raise TrajectoryValidationError(
                        "direct register_prompt result changed its artifact"
                    )
                expected_authoring = build_evaluation_contract_authoring(
                    shot_request_id=shot_request_id,
                    raw_user_request=str(value.get("raw_user_request") or ""),
                    prompt_artifact_id=artifact_id,
                    receipt_chain=[],
                )
                expected_authoring_sha256 = sha256_json(expected_authoring)
                if (
                    stored.get("evaluation_contract_authoring")
                    != expected_authoring
                    or stored.get("evaluation_contract_authoring_sha256")
                    != expected_authoring_sha256
                    or observation.get("evaluation_contract_authoring")
                    != expected_authoring
                    or observation.get("evaluation_contract_authoring_sha256")
                    != expected_authoring_sha256
                ):
                    raise TrajectoryValidationError(
                        "direct register_prompt authoring guidance is invalid"
                    )
                seen_artifacts[artifact_id] = dict(stored)
                pending_artifact = artifact_id
                continue
            if prompt_rewrite_mode != "rewrite_enabled":
                raise TrajectoryValidationError(
                    "passthrough_disabled register_prompt used a PE receipt"
                )
            receipt_id = str(raw_receipt_id)
            if receipt_id != pending_receipt:
                raise TrajectoryValidationError("register_prompt did not consume the pending receipt")
            receipt = seen_receipts.get(receipt_id)
            artifact_id = str(observation.get("prompt_artifact_id") or "")
            stored = prompt_artifacts.get(artifact_id)
            if not receipt or not artifact_id or artifact_id in seen_artifacts or not isinstance(stored, Mapping):
                raise TrajectoryValidationError("registered prompt artifact lineage is missing")
            for key in (
                "target",
                "operation",
                "purpose",
                "generation_scope",
                "prompt_context",
                "parent_prompt_artifact_id",
                "assessment_id",
                "axes",
            ):
                if receipt.get(key) != arguments.get(key):
                    raise TrajectoryValidationError(
                        f"register_prompt differs from its PE receipt on {key}"
                    )
            if list(receipt.get("context_observation_ids") or []) != list(
                arguments.get("evidence_ids") or []
            ):
                raise TrajectoryValidationError(
                    "register_prompt evidence differs from its PE receipt"
                )
            for key in (
                "target",
                "operation",
                "purpose",
                "generation_scope",
                "prompt_context",
                "parent_prompt_artifact_id",
                "assessment_id",
                "axes",
            ):
                if stored.get(key) != arguments.get(key):
                    raise TrajectoryValidationError(
                        f"stored prompt artifact differs on {key}"
                    )
            if stored.get("skill_application_receipt_id") != receipt_id:
                raise TrajectoryValidationError("prompt artifact points to the wrong PE receipt")
            if stored.get("prompt_rewrite_mode") != "rewrite_enabled":
                raise TrajectoryValidationError(
                    "PE-authored prompt artifact has the wrong rewrite mode"
                )
            if stored.get("text") != arguments.get("text") or stored.get("text_sha256") != sha256_text(
                str(arguments.get("text") or "")
            ):
                raise TrajectoryValidationError("stored prompt text differs from registration")
            if stored.get("target") == "video":
                prompt_text = str(stored.get("text") or "")
                entity_tags = set(
                    re.findall(
                        r"(?<![A-Za-z0-9_])ENTITY_[1-9][0-9]*(?![A-Za-z0-9_])",
                        prompt_text,
                    )
                )
                if entity_tags:
                    raise TrajectoryValidationError(
                        "video generator prompt exposes a non-human ENTITY_N tag"
                    )
                used_person_ids = set(
                    re.findall(
                        r"(?<![A-Za-z0-9_])ID_[A-Z][A-Z0-9_]*(?![A-Za-z0-9_])",
                        prompt_text,
                    )
                )
                active_person_ids = {
                    tag
                    for tag in registry_active_material_tags or []
                    if registry_tags.get(tag, {}).get("category") == "person"
                }
                if len(used_person_ids) > 2:
                    raise TrajectoryValidationError(
                        "video generator prompt contains more than two human IDs"
                    )
                inactive_person_ids = sorted(
                    used_person_ids - active_person_ids
                )
                if inactive_person_ids:
                    raise TrajectoryValidationError(
                        "video generator prompt contains inactive human IDs: "
                        + ", ".join(inactive_person_ids)
                    )
                missing_person_ids = sorted(
                    active_person_ids - used_person_ids
                )
                if missing_person_ids:
                    raise TrajectoryValidationError(
                        "video generator prompt omits active visible human IDs: "
                        + ", ".join(missing_person_ids)
                    )
            if stored.get("prompt_context_sha256") != sha256_json(
                arguments.get("prompt_context")
            ):
                raise TrajectoryValidationError(
                    "stored prompt context hash differs from registration"
                )
            if list(stored.get("evidence_ids") or []) != list(arguments.get("evidence_ids") or []):
                raise TrajectoryValidationError("stored prompt evidence differs from registration")
            if receipt.get("registered_prompt_artifact_id") != artifact_id:
                raise TrajectoryValidationError("PE receipt does not point back to its prompt artifact")
            parent_id = stored.get("parent_prompt_artifact_id")
            expected_chain = (
                list(seen_artifacts[str(parent_id)].get("receipt_chain") or [])
                if parent_id is not None and str(parent_id) in seen_artifacts
                else []
            ) + [receipt_id]
            if list(stored.get("receipt_chain") or []) != expected_chain:
                raise TrajectoryValidationError("prompt receipt_chain is broken")
            expected_revision = (
                int(seen_artifacts[str(parent_id)].get("revision_count", -1)) + 1
                if parent_id is not None and str(parent_id) in seen_artifacts
                else 0
            )
            if stored.get("revision_count") != expected_revision:
                raise TrajectoryValidationError(
                    "prompt artifact revision_count is inconsistent"
                )
            if observation.get("text_sha256") != stored.get("text_sha256"):
                raise TrajectoryValidationError(
                    "register_prompt result changed the prompt hash"
                )
            if observation.get("prompt_context_sha256") != stored.get(
                "prompt_context_sha256"
            ):
                raise TrajectoryValidationError(
                    "register_prompt result changed the context hash"
                )
            if list(observation.get("receipt_chain") or []) != list(
                stored.get("receipt_chain") or []
            ):
                raise TrajectoryValidationError(
                    "register_prompt result changed the receipt chain"
                )
            if observation.get("revision_count") != expected_revision:
                raise TrajectoryValidationError(
                    "register_prompt result changed the revision count"
                )
            expected_authoring = build_evaluation_contract_authoring(
                shot_request_id=shot_request_id,
                raw_user_request=str(value.get("raw_user_request") or ""),
                prompt_artifact_id=artifact_id,
                receipt_chain=stored.get("receipt_chain") or [],
            )
            expected_authoring_sha256 = sha256_json(expected_authoring)
            if (
                stored.get("evaluation_contract_authoring")
                != expected_authoring
                or stored.get("evaluation_contract_authoring_sha256")
                != expected_authoring_sha256
                or observation.get("evaluation_contract_authoring")
                != expected_authoring
                or observation.get("evaluation_contract_authoring_sha256")
                != expected_authoring_sha256
            ):
                raise TrajectoryValidationError(
                    "register_prompt evaluation-contract authoring guidance is invalid"
                )
            seen_artifacts[artifact_id] = dict(stored)
            pending_receipt = None
            pending_artifact = artifact_id
            continue

        if name in {"generate_image", "generate_shot"}:
            target = "image" if name == "generate_image" else "video"
            contract = arguments.get("evaluation_contract")
            if not isinstance(contract, Mapping):
                raise TrajectoryValidationError(
                    "generation has no evaluation contract"
                )
            contract_modules = {
                str(module) for module in contract.get("modules") or []
            }
            required_modules = (
                {"camera_composition", "style_lighting"}
                if target == "image"
                else {"camera_composition", "motion_temporal", "speech_audio"}
            )
            if target == "video":
                if registry_active_material_tags is None:
                    raise TrajectoryValidationError(
                        "video generation has no frozen active-material context"
                    )
                required_modules.update(
                    _VIDEO_MATERIAL_CATEGORY_MODULE[
                        str(registry_tags[tag].get("category") or "")
                    ]
                    for tag in registry_active_material_tags
                )
            missing_modules = sorted(required_modules - contract_modules)
            if missing_modules:
                raise TrajectoryValidationError(
                    "generation evaluation contract omits required semantic modules: "
                    + ", ".join(missing_modules)
                )
            artifact_field = (
                "prompt_artifact_id"
                if target == "image"
                else "execution_prompt_artifact_id"
            )
            artifact_id = str(arguments.get(artifact_field) or "")
            artifact = seen_artifacts.get(artifact_id)
            if not artifact or artifact.get("target") != target:
                raise TrajectoryValidationError("generation used an unregistered prompt artifact")
            passthrough_artifact = (
                artifact.get("prompt_rewrite_mode") == "passthrough_disabled"
            )
            if arguments.get("shot_request_id") != shot_request_id:
                raise TrajectoryValidationError("generation has the wrong shot_request_id")

            if target == "image":
                purpose = str(arguments.get("purpose") or "")
                expected_scope = {
                    "subject_ids": list(arguments.get("subject_ids") or []),
                    "asset_role": arguments.get("asset_role"),
                }
                references = list(arguments.get("input_references") or [])
                if artifact.get("purpose") != purpose:
                    raise TrajectoryValidationError(
                        "image generation purpose differs from its prompt artifact"
                    )
                if artifact.get("generation_scope") != expected_scope:
                    raise TrajectoryValidationError(
                        "image generation scope differs from its prompt artifact"
                    )
                if artifact.get("prompt_context") != {
                    "condition": None,
                    "has_reference_inputs": bool(references),
                } and not (
                    passthrough_artifact
                    and arguments.get("generation_group_id") is not None
                ):
                    raise TrajectoryValidationError(
                        "image references differ from the prompt artifact context"
                    )
                source_observation_ids = {
                    source_observation_id(source) for source in references
                }
                runtime_bindings: Any = {"input_references": references}
            else:
                purpose = "shot"
                condition = copy.deepcopy(arguments.get("condition"))
                slots = list(arguments.get("memory_slots") or [])
                if artifact.get("purpose") != "shot" or artifact.get(
                    "generation_scope"
                ) is not None:
                    raise TrajectoryValidationError(
                        "video generation used a non-shot prompt artifact"
                    )
                if artifact.get("prompt_context") != {
                    "condition": condition,
                    "has_reference_inputs": None,
                } and not (
                    passthrough_artifact
                    and arguments.get("generation_group_id") is not None
                ):
                    raise TrajectoryValidationError(
                        "video condition differs from its prompt artifact context"
                    )
                source_observation_ids: set[str] = set()
                if condition is not None:
                    source = condition.get("image_source")
                    source_observation_ids.add(source_observation_id(source))
                    kind = str(source.get("kind") or "") if isinstance(source, Mapping) else ""
                    relation = str(condition.get("relation") or "")
                    if (kind == "previous_tail") != (
                        relation == "temporal_continuation"
                    ):
                        raise TrajectoryValidationError(
                            "video condition relation is incompatible with its source"
                        )
                    if kind == "generated_image":
                        selection = seen_selections.get(
                            str(source.get("authorization_id") or "")
                        )
                        if not selection or selection.get("purpose") != "first_frame":
                            raise TrajectoryValidationError(
                                "generated condition is not a selected first-frame image"
                            )
                bound_images: list[tuple[str, str, str] | None] = []
                audio_slot_manifests: dict[int, dict[str, Any]] = {}
                if condition is not None:
                    bound_images.append(_source_signature(condition.get("image_source")))
                for slot_index, slot in enumerate(slots):
                    image_source = slot.get("image_source")
                    source_observation_ids.add(source_observation_id(image_source))
                    bound_images.append(_source_signature(image_source))
                    if isinstance(image_source, Mapping) and image_source.get(
                        "kind"
                    ) == "generated_image":
                        selection = seen_selections.get(
                            str(image_source.get("authorization_id") or "")
                        )
                        if not selection or selection.get("purpose") != "reference_asset":
                            raise TrajectoryValidationError(
                                "generated slot image is not a selected reference asset"
                            )
                    audio_source = slot.get("audio_source")
                    if audio_source is not None:
                        source_observation_ids.add(source_observation_id(audio_source))
                        if not registry_capabilities or registry_capabilities.get(
                            "audio_transport"
                        ) != "enabled":
                            raise TrajectoryValidationError(
                                "video audio slot was used without audio transport capability"
                            )
                        roles = [str(role) for role in slot.get("roles") or []]
                        subject_ids = [
                            str(subject_id)
                            for subject_id in slot.get("subject_ids") or []
                        ]
                        if (
                            "voice" not in roles
                            or len(subject_ids) != 1
                            or registry_tags.get(subject_ids[0], {}).get("category")
                            != "person"
                        ):
                            raise TrajectoryValidationError(
                                "video audio slot lacks one canonical voice subject"
                            )
                        image_metadata = source_metadata(image_source)
                        audio_metadata = source_metadata(audio_source)
                        if (
                            list(image_metadata.get("subject_ids") or [])
                            != subject_ids
                            or list(audio_metadata.get("subject_ids") or [])
                            != subject_ids
                        ):
                            raise TrajectoryValidationError(
                                "video audio slot image/audio speaker binding is inconsistent"
                            )
                        audio_representation = audio_metadata.get("representation")
                        if not isinstance(audio_representation, Mapping):
                            raise TrajectoryValidationError(
                                "video audio slot has no exact representation manifest"
                            )
                        selector = audio_representation.get("selector") or {}
                        if (
                            audio_representation.get("media_type") != "audio"
                            or not str(audio_representation.get("mime_type") or "").startswith(
                                "audio/"
                            )
                            or not isinstance(selector, Mapping)
                            or selector.get("mode") != "time_range"
                            or type(selector.get("start_ms")) is not int
                            or type(selector.get("end_ms")) is not int
                            or int(selector["start_ms"]) < 0
                            or int(selector["end_ms"]) <= int(selector["start_ms"])
                        ):
                            raise TrajectoryValidationError(
                                "video audio slot requires an exact audio time range"
                            )
                        binding = audio_metadata.get("speaker_binding")
                        if (
                            audio_metadata.get("transport_eligible") is not True
                            or not isinstance(binding, Mapping)
                            or binding.get("subject_id") != subject_ids[0]
                            or binding.get("matching_method")
                            != "exact_normalized_name"
                            or binding.get("basis") != "trusted_user_declared"
                        ):
                            raise TrajectoryValidationError(
                                "video audio slot lacks a trusted exact speaker transport binding"
                            )
                        if audio_source.get("kind") == "memory":
                            verification_basis = audio_metadata.get(
                                "verification_basis"
                            )
                            if verification_basis == "agent_heard":
                                authorized = bool(
                                    audio_metadata.get("quality_authorization_id")
                                )
                            else:
                                authorized = bool(
                                    verification_basis == "trusted_user_declared"
                                    and audio_metadata.get(
                                        "transport_authorization_id"
                                    )
                                )
                            if not authorized:
                                raise TrajectoryValidationError(
                                    "stored audio slot lacks its transport or quality authorization"
                                )
                        audio_slot_manifests[slot_index] = copy.deepcopy(
                            dict(audio_representation)
                        )
                if len(bound_images) != len(set(bound_images)):
                    raise TrajectoryValidationError(
                        "video condition and slots repeat an image representation"
                    )
                runtime_bindings = {
                    "condition": condition,
                    "memory_slots": slots,
                }
            if not source_observation_ids.issubset(
                set(artifact.get("evidence_ids") or [])
            ) and not (
                passthrough_artifact
                and arguments.get("generation_group_id") is not None
            ):
                raise TrajectoryValidationError(
                    "generation prompt artifact omits a bound source observation"
                )

            authoring = build_evaluation_contract_authoring(
                shot_request_id=shot_request_id,
                raw_user_request=str(value.get("raw_user_request") or ""),
                prompt_artifact_id=artifact_id,
                receipt_chain=artifact.get("receipt_chain") or [],
            )
            requirements = list(contract.get("requirements") or [])
            requirement_ids = [
                str(item.get("requirement_id") or "") for item in requirements
            ]
            if len(requirement_ids) != len(set(requirement_ids)):
                raise TrajectoryValidationError(
                    "generation evaluation contract repeats a requirement ID"
                )
            receipt_chain = [
                str(item) for item in artifact.get("receipt_chain") or []
            ]
            pe_source_ids = (
                {str(artifact_id), *receipt_chain} if receipt_chain else set()
            )
            runtime_source_ids = _binding_source_ids(runtime_bindings)
            artifact_evidence_ids = {
                str(item) for item in artifact.get("evidence_ids") or []
            }
            runtime_source_ids.update(artifact_evidence_ids)
            for observation_id in artifact_evidence_ids:
                runtime_source_ids.update(
                    _binding_source_ids(observation_payloads.get(observation_id))
                )
            prompt_text = str(artifact.get("text") or "")
            for requirement in requirements:
                provenance = str(requirement.get("provenance_kind") or "")
                source_id = str(requirement.get("source_id") or "")
                source_span = requirement.get("source_span")
                if provenance == "user_required":
                    required_user = authoring["user_required"]
                    if (
                        source_id != required_user["source_id"]
                        or source_span != required_user["source_span"]
                    ):
                        raise TrajectoryValidationError(
                            "user_required must copy the full raw-request provenance from register_prompt"
                        )
                elif provenance in {"pe_required", "pe_added"}:
                    if passthrough_artifact:
                        raise TrajectoryValidationError(
                            "passthrough evaluation contract cannot claim PE provenance"
                        )
                    if source_id not in pe_source_ids:
                        raise TrajectoryValidationError(
                            "PE requirement does not bind the current prompt artifact or receipt chain"
                        )
                    if source_span is not None:
                        if source_id != artifact_id or not isinstance(
                            source_span, Mapping
                        ):
                            raise TrajectoryValidationError(
                                "PE requirement span is not bound to the current prompt artifact"
                            )
                        start = int(source_span.get("start", -1))
                        end = int(source_span.get("end", -1))
                        if start < 0 or end <= start or end > len(prompt_text):
                            raise TrajectoryValidationError(
                                "PE requirement span is outside the current prompt artifact"
                            )
                elif provenance == "runtime_bound":
                    if source_id not in runtime_source_ids or source_span is not None:
                        raise TrajectoryValidationError(
                            "runtime_bound requirement has invalid source or source span"
                        )

            requested_group_id = arguments.get("generation_group_id")
            retry = arguments.get("retry")
            if requested_group_id is None:
                if retry is not None or any(group.get("retry") for group in groups.values()):
                    raise TrajectoryValidationError("new generation group bypassed a retry")
                if pending_artifact != artifact_id or artifact.get("operation") != "compose":
                    raise TrajectoryValidationError("new generation group lacks its pending compose artifact")
                group_id = str(observation.get("generation_group_id") or "")
                if not group_id or group_id in groups:
                    raise TrajectoryValidationError("generation group ID is missing or duplicated")
                if target == "video" and any(group.get("target") == "video" for group in groups.values()):
                    raise TrajectoryValidationError("a shot contains multiple video groups")
                groups[group_id] = {
                    "target": target,
                    "purpose": purpose,
                    "candidates": [],
                    "selection_id": None,
                    "retry": None,
                    "evaluation_contract": copy.deepcopy(
                        arguments.get("evaluation_contract")
                    ),
                    "generation_scope": (
                        copy.deepcopy(expected_scope) if target == "image" else None
                    ),
                }
            else:
                group_id = str(requested_group_id)
                group = groups.get(group_id)
                if not group or group.get("target") != target or group.get("selection_id"):
                    raise TrajectoryValidationError("retry used an unknown or closed generation group")
                if not isinstance(retry, Mapping) or group.get("retry") != {
                    "candidate_id": retry.get("candidate_id"),
                    "assessment_id": retry.get("assessment_id"),
                    "axes": list(retry.get("axes") or []),
                }:
                    raise TrajectoryValidationError("generation retry lineage is broken")
                prior = candidates.get(str(retry.get("candidate_id") or ""))
                if not prior or prior.get("group_id") != group_id:
                    raise TrajectoryValidationError("generation retry candidate is unknown")
                axes = list(retry.get("axes") or [])
                if passthrough_artifact and "prompt" in axes:
                    raise TrajectoryValidationError(
                        "passthrough retry attempted to change prompt"
                    )
                changed: set[str] = set()
                prior_artifact = seen_artifacts.get(str(prior.get("artifact_id") or ""))
                if not prior_artifact:
                    raise TrajectoryValidationError(
                        "generation retry parent artifact is unknown"
                    )
                if artifact.get("text") != prior_artifact.get("text"):
                    changed.add("prompt")
                if int(arguments.get("seed")) != int(prior.get("seed")):
                    changed.add("seed")
                if target == "image":
                    previous_references = list(prior.get("input_references") or [])
                    if _reference_signature(references) != _reference_signature(
                        previous_references
                    ):
                        changed.add("references")
                    prompt_affecting = (
                        False
                        if passthrough_artifact
                        else "prompt" in axes
                        or bool(references) != bool(previous_references)
                    )
                else:
                    if _condition_signature(condition) != _condition_signature(
                        prior.get("condition")
                    ):
                        changed.add("condition")
                    if _memory_slot_signature(slots) != _memory_slot_signature(
                        prior.get("memory_slots") or []
                    ):
                        changed.add("memory_slots")
                    prompt_affecting = (
                        False
                        if passthrough_artifact
                        else bool(set(axes) & {"prompt", "condition"})
                    )
                if changed != set(axes):
                    raise TrajectoryValidationError(
                        "generation retry axes do not equal stable input changes"
                    )
                if prompt_affecting and pending_artifact != artifact_id:
                    raise TrajectoryValidationError("prompt-affecting retry lacks a new artifact")
                if prompt_affecting:
                    if (
                        artifact.get("operation") != "revise"
                        or artifact.get("assessment_id") != retry.get("assessment_id")
                        or artifact.get("parent_prompt_artifact_id")
                        != prior.get("artifact_id")
                    ):
                        raise TrajectoryValidationError(
                            "prompt-affecting retry has the wrong revision artifact"
                        )
                elif artifact_id != prior.get("artifact_id"):
                    raise TrajectoryValidationError(
                        "non-prompt retry changed prompt artifact"
                    )
                if group.get("purpose") != purpose:
                    raise TrajectoryValidationError(
                        "generation retry changed group purpose"
                    )
                if target == "image" and group.get(
                    "generation_scope"
                ) != expected_scope:
                    raise TrajectoryValidationError(
                        "image retry changed group generation scope"
                    )
            group = groups[group_id]
            if group.get("evaluation_contract") != arguments.get(
                "evaluation_contract"
            ):
                raise TrajectoryValidationError(
                    "generation changed its frozen evaluation contract"
                )
            if observation.get("generation_group_id") != group_id:
                raise TrajectoryValidationError("generation result changed its group ID")
            if observation.get("prompt_artifact_id") != artifact_id:
                raise TrajectoryValidationError("generation result changed its prompt artifact ID")
            candidate_id = str(observation.get("candidate_id") or "")
            if not candidate_id or candidate_id in candidates:
                raise TrajectoryValidationError("candidate ID is missing or duplicated")
            representation = observation.get("representation")
            representation_id = (
                str(representation.get("representation_id") or "")
                if isinstance(representation, Mapping)
                else ""
            )
            if not representation_id:
                raise TrajectoryValidationError(
                    "generation result has no candidate representation"
                )
            if observation.get("media"):
                raise TrajectoryValidationError(
                    "generation result exposed candidate media before inspect_candidate"
                )
            representation_sha256 = str(representation.get("sha256") or "")
            representation_mime = str(representation.get("mime_type") or "")
            representation_size = representation.get("size_bytes")
            if (
                representation.get("media_type") != target
                or not _SHA256_PATTERN.fullmatch(representation_sha256)
                or not representation_mime.startswith(f"{target}/")
                or type(representation_size) is not int
                or representation_size <= 0
            ):
                raise TrajectoryValidationError(
                    "generation result has an invalid candidate representation manifest"
                )
            if target == "image" and observation.get("purpose") != purpose:
                raise TrajectoryValidationError(
                    "image generation result changed its purpose"
                )
            if observation.get("submitted_prompt_sha256") != artifact.get(
                "text_sha256"
            ):
                raise TrajectoryValidationError(
                    "generation result changed the submitted prompt"
                )
            provider_request = observation.get("provider_request")
            provider_provenance = observation.get("provider_provenance")
            if (
                not isinstance(provider_request, Mapping)
                or provider_request.get("prompt") != artifact.get("text")
                or not isinstance(provider_provenance, Mapping)
                or provider_provenance.get("prompt_sha256")
                != artifact.get("text_sha256")
                or provider_provenance.get("output_sha256")
                != representation_sha256
                or provider_provenance.get("output_mime_type")
                != representation_mime
            ):
                raise TrajectoryValidationError(
                    "generation provider receipt is not bound to the prompt artifact"
                )
            if target == "video" and audio_slot_manifests:
                request_slots = provider_request.get("memory_slots")
                media_inputs = provider_provenance.get("media_inputs")
                provenance_slots = (
                    media_inputs.get("memory_slots")
                    if isinstance(media_inputs, Mapping)
                    else None
                )
                if (
                    not isinstance(request_slots, list)
                    or len(request_slots) != len(slots)
                    or not isinstance(provenance_slots, list)
                    or len(provenance_slots) != len(slots)
                ):
                    raise TrajectoryValidationError(
                        "audio generation provider receipt omits its memory-slot manifest"
                    )
                for slot_index, manifest in audio_slot_manifests.items():
                    request_slot = request_slots[slot_index]
                    provenance_slot = provenance_slots[slot_index]
                    audio_manifest = (
                        provenance_slot.get("audio")
                        if isinstance(provenance_slot, Mapping)
                        else None
                    )
                    if (
                        not isinstance(request_slot, Mapping)
                        or "audio_url" not in request_slot
                        or request_slot.get("audio_mode") is not None
                        or not isinstance(audio_manifest, Mapping)
                        or any(
                            audio_manifest.get(key) != manifest.get(key)
                            for key in ("mime_type", "size_bytes", "sha256")
                        )
                    ):
                        raise TrajectoryValidationError(
                            "audio generation provider receipt does not prove exact audio transport"
                        )
                final_response = provider_provenance.get("final_response")
                final_result = (
                    final_response.get("result")
                    if isinstance(final_response, Mapping)
                    else None
                )
                final_metadata = (
                    final_result.get("metadata")
                    if isinstance(final_result, Mapping)
                    else None
                )
                conditioning = (
                    final_metadata.get("conditioning")
                    if isinstance(final_metadata, Mapping)
                    else None
                )
                if not isinstance(conditioning, Mapping) or conditioning.get(
                    "has_memory_audio"
                ) is not True:
                    raise TrajectoryValidationError(
                        "audio generation provider receipt does not confirm memory audio conditioning"
                    )
            groups[group_id]["candidates"].append(candidate_id)
            groups[group_id]["retry"] = None
            candidates[candidate_id] = {
                "target": target,
                "group_id": group_id,
                "artifact_id": artifact_id,
                "purpose": purpose,
                "representation_id": representation_id,
                "representation": copy.deepcopy(dict(representation)),
                "provider_provenance": copy.deepcopy(dict(provider_provenance)),
                "seed": int(arguments.get("seed")),
                "input_references": copy.deepcopy(
                    arguments.get("input_references") or []
                ),
                "condition": copy.deepcopy(arguments.get("condition")),
                "memory_slots": copy.deepcopy(arguments.get("memory_slots") or []),
                "subject_ids": copy.deepcopy(arguments.get("subject_ids") or []),
                "generation_observation": copy.deepcopy(observation),
                "inspection_id": None,
                "assessment_id": None,
            }
            artifact["last_candidate_id"] = candidate_id
            used_artifacts.add(artifact_id)
            pending_artifact = None
            pending_candidate = (candidate_id, "inspect")
            generation_count[target] += 1
            budget_key = f"{target}_attempts"
            if budget_key in budgets and generation_count[target] > budgets[budget_key]:
                raise TrajectoryValidationError(f"{target} generation exceeded its shot budget")
            if target == "video":
                declared = list(arguments.get("consumed_image_selection_ids") or [])
                expected_consumed = [
                    selection_id
                    for selection_id in selection_order
                    if seen_selections[selection_id].get("target") == "image"
                    and selection_id not in consumed_image_selections
                ]
                if declared != expected_consumed:
                    raise TrajectoryValidationError(
                        "video generation image-selection manifest is incomplete or out of order"
                    )
                condition = arguments.get("condition")
                slots = list(arguments.get("memory_slots") or [])
                for selection_id in declared:
                    selection = seen_selections[selection_id]
                    if selection.get("purpose") == "first_frame":
                        actual = (
                            isinstance(condition, Mapping)
                            and isinstance(condition.get("image_source"), Mapping)
                            and condition["image_source"].get("authorization_id")
                            == selection_id
                        )
                    else:
                        actual = any(
                            isinstance(slot.get("image_source"), Mapping)
                            and slot["image_source"].get("authorization_id")
                            == selection_id
                            for slot in slots
                        )
                    if not actual:
                        raise TrajectoryValidationError(
                            "selected image was not consumed in its declared primary role"
                        )
                    consumed_image_selections.add(selection_id)
            continue

        if name == "inspect_candidate":
            candidate_id = str(arguments.get("candidate_id") or "")
            if pending_candidate != (candidate_id, "inspect"):
                raise TrajectoryValidationError("inspection did not consume the pending candidate")
            if observation.get("candidate_id") != candidate_id:
                raise TrajectoryValidationError("inspection result changed candidate_id")
            inspection_id = str(observation.get("inspection_id") or "")
            if not inspection_id or inspection_id in inspections:
                raise TrajectoryValidationError("inspection ID is missing or duplicated")
            candidate = candidates[candidate_id]
            if observation.get("target") != candidate.get("target"):
                raise TrajectoryValidationError(
                    "inspection result changed candidate target"
                )
            if observation.get("raw_user_request") != value.get("raw_user_request"):
                raise TrajectoryValidationError(
                    "inspection result changed the raw user request"
                )
            requested_uses = list(arguments.get("requested_uses") or [])
            wanted_primary_use = primary_use(candidate)
            compatible_uses = {wanted_primary_use, "memory_writeback"}
            if candidate.get("target") == "video":
                compatible_uses.add("condition")
            if (
                wanted_primary_use not in requested_uses
                or set(requested_uses) - compatible_uses
            ):
                raise TrajectoryValidationError(
                    "inspection requested incompatible or incomplete candidate uses"
                )
            group = groups[str(candidate["group_id"])]
            assessed_peers = [
                other_id
                for other_id in group["candidates"]
                if other_id != candidate_id
                and candidates[other_id].get("assessment_id") is not None
            ]
            if assessed_peers and arguments.get("compare_pool") is not True:
                raise TrajectoryValidationError(
                    "later candidate inspection omitted its comparison pool"
                )
            comparison_pool = observation.get("comparison_pool") or []
            if not isinstance(comparison_pool, list) or {
                str(row.get("candidate_id") or "")
                for row in comparison_pool
                if isinstance(row, Mapping)
            } != set(assessed_peers if arguments.get("compare_pool") else []):
                raise TrajectoryValidationError(
                    "inspection comparison pool differs from assessed peers"
                )

            prompt_snapshot = observation.get("prompt_artifact")
            artifact = seen_artifacts[str(candidate["artifact_id"])]
            if not isinstance(prompt_snapshot, Mapping) or any(
                prompt_snapshot.get(key) != artifact.get(key)
                for key in (
                    "prompt_artifact_id",
                    "target",
                    "purpose",
                    "text",
                    "text_sha256",
                    "prompt_context",
                    "generation_scope",
                )
            ):
                raise TrajectoryValidationError(
                    "inspection prompt artifact snapshot is inconsistent"
                )
            input_manifest = observation.get("input_manifest")
            if not isinstance(input_manifest, Mapping) or input_manifest.get(
                "condition"
            ) != candidate.get("condition") or list(
                input_manifest.get("memory_slots") or []
            ) != list(
                candidate.get("memory_slots") or []
            ):
                raise TrajectoryValidationError(
                    "inspection input manifest differs from generation"
                )
            generation_observation = candidate.get("generation_observation") or {}
            if input_manifest.get("provider_request") != generation_observation.get(
                "provider_request"
            ):
                raise TrajectoryValidationError(
                    "inspection provider request differs from generation"
                )

            media_rows = observation.get("media") or []
            if not isinstance(media_rows, list):
                raise TrajectoryValidationError(
                    "inspection media must be an array"
                )
            media_by_evidence_id: dict[str, dict[str, Any]] = {}
            for media_row in media_rows:
                if not isinstance(media_row, Mapping):
                    raise TrajectoryValidationError(
                        "inspection media contains a malformed item"
                    )
                media_evidence_id = str(media_row.get("evidence_id") or "")
                if (
                    not media_evidence_id
                    or media_evidence_id in media_by_evidence_id
                ):
                    raise TrajectoryValidationError(
                        "inspection media evidence ID is missing or duplicated"
                    )
                media_by_evidence_id[media_evidence_id] = dict(media_row)
            target_evidence_kind = (
                "target_video"
                if candidate["target"] == "video"
                else "target_image"
            )
            target_media = [
                row
                for row in media_rows
                if isinstance(row, Mapping)
                and row.get("source_id") == candidate_id
                and row.get("type") == candidate["target"]
                and row.get("evidence_kind") == target_evidence_kind
            ]
            if len(target_media) != 1:
                raise TrajectoryValidationError(
                    "inspection must expose the generated candidate exactly once"
                )
            visible = target_media[0]
            generated_representation = candidate["representation"]
            provider_provenance = candidate["provider_provenance"]
            generated_representation_id = str(
                generated_representation["representation_id"]
            )
            visible_representation_id = str(
                visible.get("representation_id") or ""
            )
            if visible_representation_id == generated_representation_id:
                if visible.get("parent_sha256") is not None or visible.get(
                    "derivation"
                ) is not None:
                    raise TrajectoryValidationError(
                        "exact candidate representation has unexpected derivation lineage"
                    )
                if any(
                    visible.get(key) != generated_representation.get(key)
                    for key in ("mime_type", "size_bytes", "sha256")
                ):
                    raise TrajectoryValidationError(
                        "inspection media differs from the generated representation"
                    )
                if (
                    visible.get("sha256")
                    != provider_provenance.get("output_sha256")
                    or visible.get("mime_type")
                    != provider_provenance.get("output_mime_type")
                ):
                    raise TrajectoryValidationError(
                        "inspection media differs from the provider output receipt"
                    )
            else:
                expected_derivation = {
                    "operation": "strip_audio_for_visual_perception",
                    "source_representation_id": generated_representation_id,
                }
                if (
                    candidate["target"] != "video"
                    or not visible_representation_id
                    or visible.get("parent_sha256")
                    != generated_representation.get("sha256")
                    or visible.get("parent_sha256")
                    != provider_provenance.get("output_sha256")
                    or visible.get("derivation") != expected_derivation
                ):
                    raise TrajectoryValidationError(
                        "inspection visual-only proxy has invalid provider lineage"
                    )

            allowed_evidence = observation.get("allowed_evidence") or []
            if not isinstance(allowed_evidence, list):
                raise TrajectoryValidationError(
                    "inspection allowed_evidence must be an array"
                )
            evidence_by_id: dict[str, dict[str, Any]] = {}
            for evidence in allowed_evidence:
                if not isinstance(evidence, Mapping):
                    raise TrajectoryValidationError(
                        "inspection contains malformed evidence"
                    )
                evidence_id = str(evidence.get("evidence_id") or "")
                if not evidence_id or evidence_id in evidence_by_id:
                    raise TrajectoryValidationError(
                        "inspection evidence ID is missing or duplicated"
                    )
                if str(evidence.get("source_id") or "") in uploaded_asset_ids:
                    leaked_upload_fields = {
                        "canonical_material_tags",
                        "subject_ids",
                        "tag_binding_provenance",
                        "speaker_binding",
                        "tags",
                    }.intersection(evidence)
                    if leaked_upload_fields:
                        raise TrajectoryValidationError(
                            "uploaded asset inspection leaked backend bindings: "
                            + ", ".join(sorted(leaked_upload_fields))
                        )
                evidence_by_id[evidence_id] = dict(evidence)
            audio_perception_enabled = bool(
                registry_capabilities
                and registry_capabilities.get("audio_perception") == "verified"
            )
            if not audio_perception_enabled and any(
                str(row.get("type") or "") == "audio"
                or str(row.get("mime_type") or "").startswith("audio/")
                for row in media_rows
            ):
                raise TrajectoryValidationError(
                    "audio-disabled inspection leaked an audio media block"
                )
            metadata_only_audio_ids: set[str] = set()
            for evidence_id, evidence in evidence_by_id.items():
                if evidence.get("evidence_kind") != "slot_audio" or evidence.get(
                    "perception_exposed"
                ) is not False:
                    continue
                selector = evidence.get("selector") or {}
                source_id = str(evidence.get("source_id") or "")
                private_upload = private_upload_rows.get(source_id)
                private_binding = (
                    private_upload.get("speaker_binding")
                    if isinstance(private_upload, Mapping)
                    else None
                )
                binding = private_binding or evidence.get("speaker_binding")
                if (
                    audio_perception_enabled
                    or "path" in evidence
                    or "type" in evidence
                    or not str(evidence.get("mime_type") or "").startswith("audio/")
                    or type(evidence.get("size_bytes")) is not int
                    or int(evidence["size_bytes"]) <= 0
                    or not _SHA256_PATTERN.fullmatch(str(evidence.get("sha256") or ""))
                    or not str(evidence.get("representation_id") or "")
                    or not isinstance(selector, Mapping)
                    or selector.get("mode") != "time_range"
                    or type(selector.get("start_ms")) is not int
                    or type(selector.get("end_ms")) is not int
                    or int(selector["start_ms"]) < 0
                    or int(selector["end_ms"]) <= int(selector["start_ms"])
                    or evidence.get("transport_eligible") is not True
                    or not isinstance(binding, Mapping)
                    or binding.get("basis") != "trusted_user_declared"
                    or binding.get("matching_method")
                    != "exact_normalized_name"
                    or (
                        private_upload is not None
                        and (
                            "speaker_binding" in evidence
                            or evidence.get("speaker_name")
                            != binding.get("declared_name")
                        )
                    )
                ):
                    raise TrajectoryValidationError(
                        "transport-only slot audio evidence has invalid metadata lineage"
                    )
                metadata_only_audio_ids.add(evidence_id)
            media_evidence_ids = {
                evidence_id
                for evidence_id, evidence in evidence_by_id.items()
                if evidence.get("evidence_kind") != "provider_manifest"
                and evidence_id not in metadata_only_audio_ids
            }
            if set(media_by_evidence_id) != media_evidence_ids:
                raise TrajectoryValidationError(
                    "inspection media evidence does not match its visible media blocks"
                )
            for evidence_id in media_evidence_ids:
                evidence = evidence_by_id[evidence_id]
                media_row = media_by_evidence_id[evidence_id]
                if any(
                    evidence.get(key) != media_row.get(key)
                    for key in (
                        "representation_id",
                        "mime_type",
                        "size_bytes",
                        "sha256",
                        "source_id",
                        "evidence_kind",
                        "parent_sha256",
                        "derivation",
                        "comparison_candidate_id",
                    )
                ):
                    raise TrajectoryValidationError(
                        f"inspection evidence {evidence_id} differs from its visible media block"
                    )
            if not audio_perception_enabled:
                unexpected_audio_evidence = {
                    evidence_id
                    for evidence_id, evidence in evidence_by_id.items()
                    if evidence.get("evidence_kind") == "slot_audio"
                } - metadata_only_audio_ids
                if unexpected_audio_evidence:
                    raise TrajectoryValidationError(
                        "audio-disabled inspection contains perceptual slot-audio evidence"
                    )
            expected_bound_evidence: dict[str, tuple[str, str, str]] = {}

            def expect_bound_evidence(
                evidence_id: str, evidence_kind: str, source: Any
            ) -> None:
                signature = _source_signature(source)
                if signature is None:
                    raise TrajectoryValidationError(
                        "inspection expected evidence for a missing generation binding"
                    )
                _, source_id, representation_id = signature
                expected_bound_evidence[evidence_id] = (
                    evidence_kind,
                    source_id,
                    representation_id,
                )

            if candidate["target"] == "image":
                for reference_index, source in enumerate(
                    candidate.get("input_references") or []
                ):
                    expect_bound_evidence(
                        f"{candidate_id}.input_reference.{reference_index}",
                        "input_reference",
                        source,
                    )
            else:
                condition = candidate.get("condition")
                if isinstance(condition, Mapping):
                    expect_bound_evidence(
                        f"{candidate_id}.condition",
                        "condition_image",
                        condition.get("image_source"),
                    )
                for slot_index, slot in enumerate(
                    candidate.get("memory_slots") or []
                ):
                    expect_bound_evidence(
                        f"{candidate_id}.slot.{slot_index}.image",
                        "slot_image",
                        slot.get("image_source"),
                    )
                    if slot.get("audio_source") is not None:
                        expect_bound_evidence(
                            f"{candidate_id}.slot.{slot_index}.audio",
                            "slot_audio",
                            slot.get("audio_source"),
                        )
            binding_kinds = {
                "condition_image",
                "input_reference",
                "slot_image",
                "slot_audio",
            }
            actual_bound_ids = {
                evidence_id
                for evidence_id, evidence in evidence_by_id.items()
                if evidence.get("evidence_kind") in binding_kinds
            }
            if actual_bound_ids != set(expected_bound_evidence):
                raise TrajectoryValidationError(
                    "inspection binding evidence differs from generation inputs"
                )
            for evidence_id, expected in expected_bound_evidence.items():
                evidence = evidence_by_id[evidence_id]
                actual = (
                    str(evidence.get("evidence_kind") or ""),
                    str(evidence.get("source_id") or ""),
                    str(evidence.get("representation_id") or ""),
                )
                if actual != expected:
                    raise TrajectoryValidationError(
                        f"inspection binding evidence {evidence_id} has invalid lineage"
                    )
            if candidate["target"] == "video":
                for slot_index, slot in enumerate(
                    candidate.get("memory_slots") or []
                ):
                    audio_source = slot.get("audio_source")
                    if audio_source is None:
                        continue
                    evidence_id = f"{candidate_id}.slot.{slot_index}.audio"
                    audio_evidence = evidence_by_id[evidence_id]
                    viewed_audio = source_metadata(audio_source)
                    viewed_manifest = viewed_audio.get("representation") or {}
                    if any(
                        audio_evidence.get(key) != viewed_manifest.get(key)
                        for key in (
                            "representation_id",
                            "mime_type",
                            "size_bytes",
                            "sha256",
                        )
                    ):
                        raise TrajectoryValidationError(
                            "inspection slot audio differs from its viewed representation"
                        )
                    if not audio_perception_enabled and (
                        audio_evidence.get("selector")
                        != viewed_manifest.get("selector")
                        or (
                            audio_source.get("kind") == "uploaded_asset"
                            and (
                                "speaker_binding" in audio_evidence
                                or audio_evidence.get("speaker_name")
                                != viewed_audio.get("speaker_name")
                            )
                        )
                        or (
                            audio_source.get("kind") == "memory"
                            and audio_evidence.get("speaker_binding")
                            != viewed_audio.get("speaker_binding")
                        )
                        or audio_evidence.get("verification_basis")
                        != viewed_audio.get("verification_basis")
                    ):
                        raise TrajectoryValidationError(
                            "transport-only slot audio evidence rewrites trusted metadata"
                        )
            provider_evidence = evidence_by_id.get(
                f"{candidate_id}.provider_manifest"
            )
            if (
                not isinstance(provider_evidence, Mapping)
                or provider_evidence.get("evidence_kind") != "provider_manifest"
                or provider_evidence.get("source_id") != candidate_id
                or sum(
                    1
                    for evidence in evidence_by_id.values()
                    if evidence.get("evidence_kind") == "provider_manifest"
                )
                != 1
            ):
                raise TrajectoryValidationError(
                    "inspection provider-manifest evidence is invalid"
                )
            target_evidence_id = str(visible.get("evidence_id") or "")
            target_evidence = evidence_by_id.get(target_evidence_id)
            if (
                not target_evidence_id
                or not isinstance(target_evidence, Mapping)
                or any(
                    target_evidence.get(key) != visible.get(key)
                    for key in (
                        "representation_id",
                        "mime_type",
                        "size_bytes",
                        "sha256",
                        "source_id",
                        "evidence_kind",
                        "parent_sha256",
                        "derivation",
                    )
                )
            ):
                raise TrajectoryValidationError(
                    "inspection target media is not bound to its allowed evidence"
                )

            checklist = observation.get("checklist") or []
            if not isinstance(checklist, list) or not checklist:
                raise TrajectoryValidationError(
                    "inspection checklist must be non-empty"
                )
            checklist_by_id: dict[str, dict[str, Any]] = {}
            for check in checklist:
                if not isinstance(check, Mapping):
                    raise TrajectoryValidationError(
                        "inspection checklist contains a malformed item"
                    )
                check_id = str(check.get("check_id") or "")
                if not check_id or check_id in checklist_by_id:
                    raise TrajectoryValidationError(
                        "inspection checklist ID is missing or duplicated"
                    )
                allowed_ids = list(check.get("allowed_evidence_ids") or [])
                if not allowed_ids or any(
                    evidence_id not in evidence_by_id for evidence_id in allowed_ids
                ):
                    raise TrajectoryValidationError(
                        f"inspection check {check_id} has invalid evidence"
                    )
                allowed_kinds = set(check.get("allowed_evidence_kinds") or [])
                if any(
                    evidence_by_id[evidence_id].get("evidence_kind")
                    not in allowed_kinds
                    for evidence_id in allowed_ids
                ):
                    raise TrajectoryValidationError(
                        f"inspection check {check_id} has an invalid evidence kind"
                    )
                expected_allowed_ids = [
                    evidence_id
                    for evidence_id, evidence in evidence_by_id.items()
                    if evidence.get("evidence_kind") in allowed_kinds
                ]
                if allowed_ids != expected_allowed_ids:
                    raise TrajectoryValidationError(
                        f"inspection check {check_id} omits or reorders authorized evidence"
                    )
                checklist_by_id[check_id] = dict(check)

            output_scope = (
                "final_video"
                if candidate["target"] == "video"
                else str(candidate.get("purpose") or "")
            )
            expected_checks: dict[str, dict[str, Any]] = {
                "prompt_content.core_request": {
                    "module": "prompt_content",
                    "scope": output_scope,
                    "severity": "critical",
                    "not_applicable_allowed": False,
                    "provenance_kind": "user_required",
                    "allowed_evidence_kinds": (
                        ["target_video", "target_frame"]
                        if candidate["target"] == "video"
                        else ["target_image"]
                    ),
                },
                "technical_integrity.output": {
                    "module": "technical_integrity",
                    "scope": output_scope,
                    "severity": "critical",
                    "not_applicable_allowed": False,
                    "provenance_kind": "runtime_bound",
                    "allowed_evidence_kinds": [
                        "target_video",
                        "target_frame",
                        "target_image",
                    ],
                },
                "provenance_alignment.inputs": {
                    "module": "provenance_alignment",
                    "scope": "provider_request",
                    "severity": "critical",
                    "not_applicable_allowed": False,
                    "provenance_kind": "runtime_bound",
                    "allowed_evidence_kinds": ["provider_manifest"],
                },
            }
            if candidate["target"] == "image":
                purpose_check_id = (
                    "condition_readiness.first_frame"
                    if candidate.get("purpose") == "first_frame"
                    else "reference_asset_quality.reusable"
                )
                expected_checks[purpose_check_id] = {
                    "module": (
                        "condition_readiness"
                        if candidate.get("purpose") == "first_frame"
                        else "reference_asset_quality"
                    ),
                    "scope": output_scope,
                    "severity": "critical",
                    "not_applicable_allowed": False,
                    "provenance_kind": "runtime_bound",
                    "allowed_evidence_kinds": ["target_image", "input_reference"],
                }
                for reference_index, _ in enumerate(
                    candidate.get("input_references") or [], start=1
                ):
                    expected_checks[
                        f"reference_consistency.image_input_{reference_index}"
                    ] = {
                        "module": "identity_wardrobe",
                        "scope": output_scope,
                        "severity": "critical",
                        "not_applicable_allowed": False,
                        "provenance_kind": "runtime_bound",
                        "allowed_evidence_kinds": [
                            "target_image",
                            "input_reference",
                        ],
                    }
            else:
                condition = candidate.get("condition")
                if condition is not None:
                    condition_check_id = (
                        "tail_to_start_continuity"
                        if condition.get("relation") == "temporal_continuation"
                        else "frame0_condition_alignment"
                    )
                    expected_checks[condition_check_id] = {
                        "module": (
                            "motion_temporal"
                            if condition.get("relation")
                            == "temporal_continuation"
                            else "camera_composition"
                        ),
                        "scope": "final_video",
                        "severity": "critical",
                        "not_applicable_allowed": False,
                        "provenance_kind": "runtime_bound",
                        "allowed_evidence_kinds": [
                            "target_video",
                            "target_frame",
                            "condition_image",
                        ],
                    }
                for slot_index, slot in enumerate(
                    candidate.get("memory_slots") or [], start=1
                ):
                    expected_checks[f"reference_consistency.slot_{slot_index}"] = {
                        "module": "identity_wardrobe",
                        "scope": "final_video",
                        "severity": "critical",
                        "not_applicable_allowed": False,
                        "provenance_kind": "runtime_bound",
                        "allowed_evidence_kinds": [
                            "target_video",
                            "target_frame",
                            "slot_image",
                        ],
                    }
                    if slot.get("audio_source") is not None:
                        expected_checks[
                            f"voice_memory_quality.slot_{slot_index}"
                        ] = {
                            "module": "speech_audio",
                            "scope": "final_video",
                            "severity": "critical",
                            "not_applicable_allowed": False,
                            "provenance_kind": "runtime_bound",
                            "allowed_evidence_kinds": [
                                "slot_audio",
                                "target_audio",
                            ],
                        }
                        expected_checks[
                            f"speaker_image_audio_binding.slot_{slot_index}"
                        ] = {
                            "module": "speech_audio",
                            "scope": "final_video",
                            "severity": "critical",
                            "not_applicable_allowed": False,
                            "provenance_kind": "runtime_bound",
                            "allowed_evidence_kinds": [
                                "target_video",
                                "target_audio",
                                "target_frame",
                                "slot_image",
                                "slot_audio",
                            ],
                        }
            contract = group.get("evaluation_contract") or {}
            for module in contract.get("modules") or []:
                expected_checks[f"{module}.module_contract"] = {
                    "module": module,
                    "scope": output_scope,
                    "severity": "critical",
                    "not_applicable_allowed": False,
                    "provenance_kind": "pe_required",
                    "allowed_evidence_kinds": evaluation_check_evidence_kinds(
                        target=str(candidate["target"]), module=str(module)
                    ),
                }
            for requirement in contract.get("requirements") or []:
                if not isinstance(requirement, Mapping):
                    continue
                expected_checks[str(requirement.get("requirement_id") or "")] = {
                    "module": requirement.get("module"),
                    "scope": requirement.get("scope"),
                    "requirement": str(requirement.get("text") or "").strip(),
                    "severity": requirement.get("severity"),
                    "not_applicable_allowed": bool(
                        requirement.get("not_applicable_allowed", False)
                    ),
                    "provenance_kind": requirement.get("provenance_kind"),
                    "allowed_evidence_kinds": evaluation_check_evidence_kinds(
                        target=str(candidate["target"]),
                        module=str(requirement.get("module") or "prompt_content"),
                    ),
                }
            if set(checklist_by_id) != set(expected_checks):
                raise TrajectoryValidationError(
                    "inspection checklist differs from the frozen generation contract"
                )
            for check_id, expected in expected_checks.items():
                actual = checklist_by_id[check_id]
                if any(actual.get(key) != expected_value for key, expected_value in expected.items()):
                    raise TrajectoryValidationError(
                        f"inspection check {check_id} rewrites its frozen contract"
                    )

            use_contracts = observation.get("use_contracts") or []
            if not isinstance(use_contracts, list) or not use_contracts:
                raise TrajectoryValidationError(
                    "inspection use contracts must be non-empty"
                )
            contracts_by_id: dict[str, dict[str, Any]] = {}
            primary_contracts = 0
            for contract_row in use_contracts:
                if not isinstance(contract_row, Mapping):
                    raise TrajectoryValidationError(
                        "inspection contains a malformed use contract"
                    )
                contract_id = str(contract_row.get("use_contract_id") or "")
                if not contract_id or contract_id in contracts_by_id:
                    raise TrajectoryValidationError(
                        "inspection use-contract ID is missing or duplicated"
                    )
                evidence_ids = list(contract_row.get("allowed_evidence_ids") or [])
                if not evidence_ids or any(
                    evidence_id not in evidence_by_id for evidence_id in evidence_ids
                ):
                    raise TrajectoryValidationError(
                        "inspection use contract has invalid evidence"
                    )
                if (
                    contract_row.get("use") == wanted_primary_use
                    and contract_row.get("source_ids") == [candidate_id]
                    and contract_row.get("modality") == candidate.get("target")
                ):
                    primary_contracts += 1
                if (
                    contract_row.get("use") == "memory_writeback"
                    and "memory_writeback" not in requested_uses
                ):
                    raise TrajectoryValidationError(
                        "inspection returned an unrequested memory-writeback contract"
                    )
                if (
                    contract_row.get("use") == "memory_writeback"
                    and contract_row.get("modality") == "audio"
                ):
                    evidence_kinds = {
                        str(evidence_by_id[evidence_id].get("evidence_kind") or "")
                        for evidence_id in evidence_ids
                    }
                    if evidence_kinds != {"slot_audio"}:
                        raise TrajectoryValidationError(
                            "audio memory writeback must come from an exact viewed slot clip, "
                            "not a generated soundtrack"
                        )
                contracts_by_id[contract_id] = dict(contract_row)
            if primary_contracts != 1:
                raise TrajectoryValidationError(
                    "inspection lacks exactly one primary-use contract"
                )

            capability_mask = observation.get("capability_mask")
            expected_capability_mask = {
                "visual": True,
                "audio": bool(
                    registry_capabilities
                    and registry_capabilities.get("audio_perception") == "verified"
                ),
            }
            if (
                not isinstance(capability_mask, Mapping)
                or dict(capability_mask) != expected_capability_mask
                or any(type(value) is not bool for value in capability_mask.values())
            ):
                raise TrajectoryValidationError(
                    "inspection capability mask differs from the frozen registry capabilities"
                )
            expected_assessment_contract = build_assessment_contract(
                checklist, expected_capability_mask, use_contracts
            )
            if observation.get("assessment_contract") != expected_assessment_contract:
                raise TrajectoryValidationError(
                    "inspection assessment contract differs from its frozen checklist and capabilities"
                )
            expected_revision_hash = sha256_json(
                {
                    "checklist": checklist,
                    "use_contracts": use_contracts,
                    "capability_mask": dict(capability_mask),
                    "assessment_contract": expected_assessment_contract,
                }
            )
            if (
                observation.get("checklist_sha256") != expected_revision_hash
                or observation.get("inspection_revision_hash")
                != expected_revision_hash
            ):
                raise TrajectoryValidationError(
                    "inspection checklist revision hash is invalid"
                )
            remaining = max(
                0,
                int(budgets.get(f"{candidate['target']}_attempts", 0))
                - generation_count[candidate["target"]],
            )
            if (
                f"{candidate['target']}_attempts" in budgets
                and observation.get("remaining_attempts") != remaining
            ):
                raise TrajectoryValidationError(
                    "inspection remaining-attempt budget is inconsistent"
                )
            observation_id = add_observation(observation, "candidate_inspection")
            inspections[inspection_id] = {
                "candidate_id": candidate_id,
                "observation_id": observation_id,
                "requested_uses": requested_uses,
                "checklist": copy.deepcopy(checklist),
                "checklist_by_id": checklist_by_id,
                "revision_hash": expected_revision_hash,
                "allowed_evidence": evidence_by_id,
                "use_contracts": copy.deepcopy(use_contracts),
                "contracts_by_id": contracts_by_id,
                "capability_mask": dict(capability_mask),
                "assessment_contract": copy.deepcopy(
                    expected_assessment_contract
                ),
            }
            candidates[candidate_id]["inspection_id"] = inspection_id
            pending_candidate = (candidate_id, "assessment")
            continue

        if name == "submit_candidate_assessment":
            inspection_id = str(arguments.get("inspection_id") or "")
            inspection = inspections.get(inspection_id)
            if not inspection:
                raise TrajectoryValidationError("assessment used an unknown inspection")
            candidate_id = str(inspection["candidate_id"])
            if pending_candidate != (candidate_id, "assessment"):
                raise TrajectoryValidationError("assessment did not consume the pending inspection")
            if observation.get("candidate_id") != candidate_id:
                raise TrajectoryValidationError("assessment result changed candidate_id")
            assessment_id = str(observation.get("assessment_id") or "")
            if not assessment_id or assessment_id in assessments:
                raise TrajectoryValidationError("assessment ID is missing or duplicated")
            if arguments.get("inspection_revision_hash") != inspection.get(
                "revision_hash"
            ):
                raise TrajectoryValidationError(
                    "assessment used a stale inspection revision hash"
                )
            assessment_contract = inspection["assessment_contract"]
            if (
                arguments.get("module_applicability")
                != assessment_contract["module_applicability"]
            ):
                raise TrajectoryValidationError(
                    "assessment module applicability differs from its frozen assessment contract"
                )

            checklist_by_id = inspection["checklist_by_id"]
            check_results = list(arguments.get("check_results") or [])
            check_result_by_id: dict[str, dict[str, Any]] = {}
            for result in check_results:
                check_id = str(result.get("check_id") or "")
                if check_id in check_result_by_id:
                    raise TrajectoryValidationError(
                        "assessment contains duplicate checklist results"
                    )
                check = checklist_by_id.get(check_id)
                if not check:
                    raise TrajectoryValidationError(
                        "assessment cites an unknown checklist item"
                    )
                evidence_ids = list(result.get("evidence_ids") or [])
                if set(evidence_ids) - set(check.get("allowed_evidence_ids") or []):
                    raise TrajectoryValidationError(
                        "assessment checklist result cites unauthorized evidence"
                    )
                result_value = str(result.get("result") or "")
                if result_value in {
                    "pass",
                    "fail",
                    "uncertain",
                } and not evidence_ids:
                    raise TrajectoryValidationError(
                        "assessment checklist result omits required evidence"
                    )
                if result_value == "not_applicable" and not check.get(
                    "not_applicable_allowed"
                ):
                    raise TrajectoryValidationError(
                        "assessment marks a mandatory check not applicable"
                    )
                if result_value == "capability_blocked" and not (
                    check.get("module") == "speech_audio"
                    and inspection["capability_mask"].get("audio") is False
                ):
                    raise TrajectoryValidationError(
                        "assessment uses capability_blocked for observable evidence"
                    )
                check_result_by_id[check_id] = dict(result)
            if set(check_result_by_id) != set(checklist_by_id):
                raise TrajectoryValidationError(
                    "assessment does not cover the complete checklist"
                )
            for forced_result in assessment_contract["forced_check_results"]:
                if check_result_by_id.get(forced_result["check_id"]) != forced_result:
                    raise TrajectoryValidationError(
                        "assessment forced check result differs from its frozen assessment contract"
                    )

            applicability_rows = list(arguments.get("module_applicability") or [])
            applicability: dict[str, str] = {}
            for row in applicability_rows:
                module = str(row.get("module") or "")
                if module in applicability:
                    raise TrajectoryValidationError(
                        "assessment repeats module applicability"
                    )
                applicability[module] = str(row.get("applicability") or "")
            if set(applicability) != _SEMANTIC_MODULES:
                raise TrajectoryValidationError(
                    "assessment must cover every semantic module"
                )
            applicable_from_checks = {
                str(check.get("module") or "")
                for check in checklist_by_id.values()
                if check.get("module") in _SEMANTIC_MODULES
            }
            if any(
                (applicability[module] == "applicable")
                != (module in applicable_from_checks)
                for module in _SEMANTIC_MODULES
            ):
                raise TrajectoryValidationError(
                    "assessment module applicability differs from its checklist"
                )

            contracts_by_id = inspection["contracts_by_id"]
            use_results = list(arguments.get("use_results") or [])
            use_result_by_id: dict[str, dict[str, Any]] = {}
            for result in use_results:
                contract_id = str(result.get("use_contract_id") or "")
                if contract_id in use_result_by_id:
                    raise TrajectoryValidationError(
                        "assessment contains duplicate use results"
                    )
                contract = contracts_by_id.get(contract_id)
                if not contract:
                    raise TrajectoryValidationError(
                        "assessment cites an unknown use contract"
                    )
                if (
                    result.get("use") != contract.get("use")
                    or result.get("source_ids") != contract.get("source_ids")
                ):
                    raise TrajectoryValidationError(
                        "assessment use result differs from its contract"
                    )
                evidence_ids = list(result.get("evidence_ids") or [])
                if set(evidence_ids) - set(
                    contract.get("allowed_evidence_ids") or []
                ):
                    raise TrajectoryValidationError(
                        "assessment use result cites unauthorized evidence"
                    )
                result_value = str(result.get("result") or "")
                if result_value in {
                    "pass",
                    "fail",
                    "uncertain",
                } and not evidence_ids:
                    raise TrajectoryValidationError(
                        "assessment use result omits required evidence"
                    )
                if result_value == "capability_blocked" and not (
                    contract.get("modality") == "audio"
                    and inspection["capability_mask"].get("audio") is False
                ):
                    raise TrajectoryValidationError(
                        "assessment blocks an observable use contract"
                    )
                use_result_by_id[contract_id] = dict(result)
            if set(use_result_by_id) != set(contracts_by_id):
                raise TrajectoryValidationError(
                    "assessment does not cover every use contract"
                )
            for forced_result in assessment_contract.get(
                "forced_use_results", []
            ):
                if use_result_by_id.get(
                    str(forced_result.get("use_contract_id") or "")
                ) != forced_result:
                    raise TrajectoryValidationError(
                        "assessment forced use result differs from its frozen assessment contract"
                    )

            critical_results = [
                check_result_by_id[check_id]["result"]
                for check_id, check in checklist_by_id.items()
                if check.get("severity") == "critical"
            ]
            quality = (
                "failed"
                if any(item in _ASSESSMENT_FAILURES for item in critical_results)
                else "capability_limited"
                if any(item == "capability_blocked" for item in critical_results)
                else "qualified"
            )
            primary_statuses = [
                use_result_by_id[contract_id]["result"]
                for contract_id, contract in contracts_by_id.items()
                if contract.get("use") == primary_use(candidates[candidate_id])
                and contract.get("source_ids") == [candidate_id]
            ]
            if any(item in _ASSESSMENT_FAILURES for item in primary_statuses):
                quality = "failed"
            elif quality == "qualified" and any(
                item == "capability_blocked" for item in primary_statuses
            ):
                quality = "capability_limited"
            assessment = {
                "candidate_id": candidate_id,
                "inspection_id": inspection_id,
                "assessment_contract": copy.deepcopy(assessment_contract),
                "module_applicability": copy.deepcopy(
                    arguments.get("module_applicability") or []
                ),
                "next_action": arguments.get("next_action"),
                "retry_change": copy.deepcopy(arguments.get("retry_change")),
                "check_results": copy.deepcopy(check_results),
                "check_result_by_id": check_result_by_id,
                "use_contracts": copy.deepcopy(inspection["use_contracts"]),
                "use_results": copy.deepcopy(use_results),
                "use_result_by_id": use_result_by_id,
                "quality": quality,
            }
            assessments[assessment_id] = assessment
            candidates[candidate_id]["assessment_id"] = assessment_id
            group = groups[candidates[candidate_id]["group_id"]]
            expected_remaining = max(
                0,
                int(
                    budgets.get(
                        f"{candidates[candidate_id]['target']}_attempts", 0
                    )
                )
                - generation_count[candidates[candidate_id]["target"]],
            )
            if (
                observation.get("next_action") != arguments.get("next_action")
                or (
                    f"{candidates[candidate_id]['target']}_attempts" in budgets
                    and observation.get("remaining_attempts") != expected_remaining
                )
            ):
                raise TrajectoryValidationError(
                    "assessment result differs from the submitted decision"
                )
            if arguments.get("next_action") == "retry":
                retry_change = arguments.get("retry_change") or {}
                if (
                    prompt_rewrite_mode == "passthrough_disabled"
                    and "prompt" in list(retry_change.get("axes") or [])
                ):
                    raise TrajectoryValidationError(
                        "passthrough retry attempted to change prompt"
                    )
                remaining = max(
                    0,
                    int(
                        budgets.get(
                            f"{candidates[candidate_id]['target']}_attempts", 0
                        )
                    )
                    - generation_count[candidates[candidate_id]["target"]],
                )
                if (
                    f"{candidates[candidate_id]['target']}_attempts" in budgets
                    and remaining <= 0
                ):
                    raise TrajectoryValidationError(
                        "retry was selected after generation budget exhaustion"
                    )
                if quality != "failed" or any(
                    candidate_quality(other_id) == "qualified"
                    for other_id in group["candidates"]
                    if other_id != candidate_id
                ):
                    raise TrajectoryValidationError(
                        "retry was selected without an unresolved primary failure"
                    )
                actual_failed_ids = {
                    check_id
                    for check_id, result in check_result_by_id.items()
                    if result.get("result") in _ASSESSMENT_FAILURES
                }
                actual_failed_ids.update(
                    contract_id
                    for contract_id, result in use_result_by_id.items()
                    if result.get("result") in _ASSESSMENT_FAILURES
                    and contracts_by_id[contract_id].get("use")
                    == primary_use(candidates[candidate_id])
                )
                if set(retry_change.get("failed_check_ids") or []) != actual_failed_ids:
                    raise TrajectoryValidationError(
                        "retry failed_check_ids differ from the failed assessment"
                    )
                if list(observation.get("required_retry_axes") or []) != list(
                    retry_change.get("axes") or []
                ):
                    raise TrajectoryValidationError(
                        "retry result changed the required axes"
                    )
                group["retry"] = {
                    "candidate_id": candidate_id,
                    "assessment_id": assessment_id,
                    "axes": list(retry_change.get("axes") or []),
                }
            else:
                selection = arguments.get("selection") or {}
                ranked = list(selection.get("ranked_candidate_ids") or [])
                if set(ranked) != set(group["candidates"]) or len(ranked) != len(set(ranked)):
                    raise TrajectoryValidationError("selection does not rank the complete candidate group")
                selected_candidate_id = str(selection.get("selected_candidate_id") or "")
                if not ranked or ranked[0] != selected_candidate_id:
                    raise TrajectoryValidationError("selection winner is not ranked first")
                selected_quality = (
                    quality
                    if selected_candidate_id == candidate_id
                    else candidate_quality(selected_candidate_id)
                )
                quality_rank = {
                    "failed": 0,
                    "capability_limited": 1,
                    "qualified": 2,
                }
                if any(
                    quality_rank.get(
                        quality
                        if other_id == candidate_id
                        else candidate_quality(other_id),
                        -1,
                    )
                    > quality_rank.get(selected_quality, -1)
                    for other_id in ranked
                    if other_id != selected_candidate_id
                ):
                    raise TrajectoryValidationError(
                        "selection ranks an inferior candidate above a better candidate"
                    )
                remaining = max(
                    0,
                    int(
                        budgets.get(
                            f"{candidates[candidate_id]['target']}_attempts", 0
                        )
                    )
                    - generation_count[candidates[candidate_id]["target"]],
                )
                if (
                    selected_quality == "failed"
                    and f"{candidates[candidate_id]['target']}_attempts" in budgets
                    and remaining > 0
                ):
                    raise TrajectoryValidationError(
                        "failed candidate was selected while retry budget remained"
                    )
                expected_mode = (
                    "best_effort"
                    if selected_quality == "failed"
                    else "capability_limited"
                    if selected_quality == "capability_limited"
                    else "requirements_met"
                )
                if selection.get("mode") != expected_mode:
                    raise TrajectoryValidationError(
                        "selection mode differs from assessed candidate quality"
                    )
                allow_failed = (
                    selected_quality == "failed" and remaining == 0
                )
                if selected_candidate_id == candidate_id:
                    selected_primary_authorized = any(
                        contract.get("use")
                        == primary_use(candidates[selected_candidate_id])
                        and contract.get("source_ids") == [selected_candidate_id]
                        and use_result_by_id[contract_id].get("result")
                        in (
                            {"pass", "capability_blocked", "fail", "uncertain"}
                            if allow_failed
                            else {"pass", "capability_blocked"}
                        )
                        for contract_id, contract in contracts_by_id.items()
                    )
                else:
                    selected_primary_authorized = primary_use_authorized(
                        selected_candidate_id,
                        allow_failed_best_effort=allow_failed,
                    )
                if not selected_primary_authorized:
                    raise TrajectoryValidationError(
                        "selected candidate lacks primary-use authorization"
                    )
                selection_id = str(observation.get("selection_id") or "")
                if not selection_id or selection_id in seen_selections:
                    raise TrajectoryValidationError("selection ID is missing or duplicated")
                stored = stored_selections.get(selection_id)
                if not isinstance(stored, Mapping):
                    raise TrajectoryValidationError("selection result has no stored record")
                expected_target = candidates[candidate_id]["target"]
                if (
                    stored.get("target") != expected_target
                    or stored.get("purpose") != group.get("purpose")
                    or stored.get("group_id") != candidates[candidate_id]["group_id"]
                    or stored.get("selected_candidate_id") != selected_candidate_id
                    or list(stored.get("ranked_candidate_ids") or []) != ranked
                    or stored.get("mode") != selection.get("mode")
                    or stored.get("reason") != selection.get("reason")
                    or observation.get("selected_candidate_id") != selected_candidate_id
                    or observation.get("mode") != selection.get("mode")
                ):
                    raise TrajectoryValidationError("stored selection lineage is broken")
                group["selection_id"] = selection_id
                group["retry"] = None
                seen_selections[selection_id] = dict(stored)
                selection_order.append(selection_id)
                if expected_target == "video":
                    if video_selection_id is not None:
                        raise TrajectoryValidationError("trajectory has multiple video selections")
                    if observation.get("video_selection_id") != selection_id:
                        raise TrajectoryValidationError("video selection result is inconsistent")
                    video_selection_id = selection_id
                elif observation.get("image_selection_id") != selection_id:
                    raise TrajectoryValidationError(
                        "image selection result is inconsistent"
                    )
            pending_candidate = None
            continue

        if name == "update_memory_bank":
            if arguments.get("video_selection_id") != video_selection_id:
                raise TrajectoryValidationError("memory update used the wrong video selection")
            if memory_update_id is not None:
                raise TrajectoryValidationError("trajectory stages multiple memory updates")
            add_memories = list(arguments.get("add_memories") or [])
            selected_video_candidate_id = str(
                seen_selections[video_selection_id].get("selected_candidate_id")
            )
            client_refs = [str(item.get("client_ref") or "") for item in add_memories]
            if len(client_refs) != len(set(client_refs)):
                raise TrajectoryValidationError(
                    "memory update contains duplicate client_ref values"
                )
            new_tag_refs: set[str] = set()
            for item in add_memories:
                assessment = assessments.get(str(item.get("assessment_id") or ""))
                if not assessment:
                    raise TrajectoryValidationError("memory writeback cites an unknown assessment")
                inspection = inspections.get(str(assessment.get("inspection_id") or ""))
                if not inspection or item.get("observation_id") != inspection.get("observation_id"):
                    raise TrajectoryValidationError("memory writeback observation lineage is broken")
                contract_id = str(item.get("use_contract_id") or "")
                contract = next(
                    (
                        row
                        for row in inspection.get("use_contracts") or []
                        if row.get("use_contract_id") == contract_id
                    ),
                    None,
                )
                use_result = next(
                    (
                        row
                        for row in assessment.get("use_results") or []
                        if row.get("use_contract_id") == contract_id
                    ),
                    None,
                )
                modality = str(item.get("modality") or "")
                transport_only_audio = bool(
                    modality == "audio"
                    and item.get("verification_basis")
                    == "trusted_user_declared"
                    and registry_capabilities
                    and registry_capabilities.get("audio_transport") == "enabled"
                    and registry_capabilities.get("audio_perception") == "disabled"
                )
                expected_use_result = (
                    "capability_blocked" if transport_only_audio else "pass"
                )
                if (
                    not isinstance(contract, Mapping)
                    or contract.get("use") != "memory_writeback"
                    or contract.get("source_ids") != [item.get("source_id")]
                    or not isinstance(use_result, Mapping)
                    or use_result.get("result") != expected_use_result
                    or use_result.get("source_ids") != contract.get("source_ids")
                    or set(use_result.get("evidence_ids") or [])
                    - set(contract.get("allowed_evidence_ids") or [])
                    or item.get("modality") != contract.get("modality")
                ):
                    raise TrajectoryValidationError("memory writeback use-contract lineage is broken")
                evidence_rows = [
                    inspection.get("allowed_evidence", {}).get(evidence_id)
                    for evidence_id in contract.get("allowed_evidence_ids") or []
                ]
                if not any(
                    isinstance(row, Mapping)
                    and row.get("source_id") == item.get("source_id")
                    for row in evidence_rows
                ):
                    raise TrajectoryValidationError(
                        "memory writeback source is absent from its inspected evidence"
                    )
                if modality == "audio":
                    subject_ids = [
                        str(subject_id) for subject_id in item.get("subject_ids") or []
                    ]
                    audio_evidence = [
                        row
                        for row in evidence_rows
                        if isinstance(row, Mapping)
                        and row.get("source_id") == item.get("source_id")
                        and row.get("evidence_kind") == "slot_audio"
                    ]
                    if len(subject_ids) != 1 or item.get("roles") is None or "voice" not in set(
                        item.get("roles") or []
                    ):
                        raise TrajectoryValidationError(
                            "audio memory writeback lacks one canonical voice subject"
                        )
                    if transport_only_audio:
                        source_upload = private_upload_rows.get(
                            str(item.get("source_id") or "")
                        )
                        binding = (
                            source_upload.get("speaker_binding")
                            if isinstance(source_upload, Mapping)
                            else None
                        )
                        if (
                            not isinstance(source_upload, Mapping)
                            or not isinstance(binding, Mapping)
                            or binding.get("basis") != "trusted_user_declared"
                            or binding.get("matching_method")
                            != "exact_normalized_name"
                            or binding.get("subject_id") != subject_ids[0]
                            or list(source_upload.get("subject_ids") or [])
                            != subject_ids
                            or len(audio_evidence) != 1
                            or audio_evidence[0].get("perception_exposed") is not False
                            or use_result.get("evidence_ids")
                        ):
                            raise TrajectoryValidationError(
                                "transport-only audio writeback lacks trusted metadata lineage"
                            )
                    elif (
                        not registry_capabilities
                        or registry_capabilities.get("audio_perception") != "verified"
                        or item.get("verification_basis") != "agent_heard"
                    ):
                        raise TrajectoryValidationError(
                            "audio memory writeback lacks verified perception"
                        )
                source_candidate = candidates.get(str(item.get("source_id") or ""))
                if source_candidate and source_candidate.get("target") == "image":
                    if not any(
                        selection.get("target") == "image"
                        and selection.get("selected_candidate_id")
                        == item.get("source_id")
                        for selection in seen_selections.values()
                    ):
                        raise TrajectoryValidationError(
                            "memory writeback used an unselected generated image"
                        )
                elif inspection.get("candidate_id") != selected_video_candidate_id:
                    raise TrajectoryValidationError(
                        "memory writeback source was not inspected with the selected video"
                    )
                for tag_ref in item.get("tag_refs") or []:
                    if tag_ref.get("kind") == "new_tag":
                        new_tag_refs.add(str(tag_ref.get("client_ref") or ""))
                    elif str(tag_ref.get("tag") or "") not in registry_tags:
                        raise TrajectoryValidationError(
                            "memory update references an unknown registry tag"
                        )
            registry_updates = list(arguments.get("registry_updates") or [])
            transport_only_client_refs = {
                str(item.get("client_ref") or "")
                for item in add_memories
                if item.get("modality") == "audio"
                and item.get("verification_basis") == "trusted_user_declared"
            }
            declared_new_tags: set[str] = set()
            for update in registry_updates:
                evidence_refs = set(update.get("evidence_client_refs") or [])
                if not evidence_refs or not evidence_refs.issubset(set(client_refs)):
                    raise TrajectoryValidationError(
                        "registry update cites an unknown memory client_ref"
                    )
                if evidence_refs & transport_only_client_refs:
                    raise TrajectoryValidationError(
                        "transport-only audio cannot support registry refinement"
                    )
                tag_ref = update.get("tag_ref") or {}
                if tag_ref.get("kind") == "new_tag":
                    if update.get("operation") != "append" or update.get(
                        "category"
                    ) == "person":
                        raise TrajectoryValidationError(
                            "new registry tag has an invalid append operation"
                        )
                    client_ref = str(tag_ref.get("client_ref") or "")
                    if client_ref in declared_new_tags:
                        raise TrajectoryValidationError(
                            "memory update declares a duplicate new tag"
                        )
                    declared_new_tags.add(client_ref)
                else:
                    tag = str(tag_ref.get("tag") or "")
                    prior = registry_tags.get(tag)
                    if (
                        update.get("operation") != "refine"
                        or not prior
                        or prior.get("category") != update.get("category")
                    ):
                        raise TrajectoryValidationError(
                            "registry refinement changed an unknown tag or category"
                        )
            if not new_tag_refs.issubset(declared_new_tags):
                raise TrajectoryValidationError(
                    "new memory tag references do not match registry updates"
                )
            memory_update_id = str(observation.get("memory_update_id") or "")
            if not memory_update_id:
                raise TrajectoryValidationError("memory update result has no transaction ID")
            continue

        if name == "submit_shot":
            if index != len(events) - 1:
                raise TrajectoryValidationError("submit_shot must be the unique final tool")
            if arguments.get("video_selection_id") != video_selection_id:
                raise TrajectoryValidationError("submit_shot used the wrong video selection")
            selection = seen_selections.get(str(video_selection_id or ""))
            if not selection or selection.get("target") != "video":
                raise TrajectoryValidationError(
                    "submit_shot has no selected final video"
                )
            if pending_receipt or pending_artifact or pending_candidate:
                raise TrajectoryValidationError("submit_shot has a pending causal obligation")
            if any(not group.get("selection_id") or group.get("retry") for group in groups.values()):
                raise TrajectoryValidationError("submit_shot has an open generation group")
            if {
                selection_id
                for selection_id, selected in seen_selections.items()
                if selected.get("target") == "image"
            } != consumed_image_selections:
                raise TrajectoryValidationError("submit_shot has unconsumed image selections")
            memory = arguments.get("memory_update") or {}
            if memory.get("status") == "updated":
                if memory.get("memory_update_id") != memory_update_id:
                    raise TrajectoryValidationError("submit_shot changed the staged memory transaction")
            elif memory_update_id is not None:
                raise TrajectoryValidationError("submit_shot skipped a staged memory transaction")
            if (
                observation.get("done") is not True
                or observation.get("video_selection_id") != video_selection_id
                or observation.get("shot_index") != value.get("shot_index")
            ):
                raise TrajectoryValidationError(
                    "submit_shot result is inconsistent with the trajectory"
                )
            continue

        raise TrajectoryValidationError(f"causal ledger contains unknown tool {name!r}")

    last_name, _, last_observation = event_parts(events[-1])
    if last_name != "submit_shot" or last_observation.get("done") is not True:
        raise TrajectoryValidationError("trajectory must end with submit_shot(done=true)")
    if pending_receipt or pending_artifact or pending_candidate:
        raise TrajectoryValidationError("trajectory ends with a pending causal obligation")
    if set(seen_receipts) != set(pe_receipts):
        raise TrajectoryValidationError("trajectory PE receipt table is not event-closed")
    if set(seen_artifacts) != set(prompt_artifacts) or set(used_artifacts) != set(prompt_artifacts):
        raise TrajectoryValidationError("trajectory prompt artifact table is not consumed and event-closed")
    if set(seen_selections) != set(stored_selections):
        raise TrajectoryValidationError("trajectory selection table is not event-closed")
    for receipt_id, receipt in seen_receipts.items():
        artifact_id = receipt.get("registered_prompt_artifact_id")
        if not artifact_id or seen_artifacts.get(str(artifact_id), {}).get(
            "skill_application_receipt_id"
        ) != receipt_id:
            raise TrajectoryValidationError(
                "trajectory contains an unconsumed PE receipt"
            )
    for artifact_id, artifact in seen_artifacts.items():
        generated = [
            candidate_id
            for candidate_id, candidate in candidates.items()
            if candidate.get("artifact_id") == artifact_id
        ]
        if (
            artifact.get("consumed") is not True
            or not generated
            or artifact.get("generation_group_id")
            != candidates[generated[0]].get("group_id")
            or artifact.get("last_candidate_id") != generated[-1]
        ):
            raise TrajectoryValidationError(
                "stored prompt artifact consumption state is inconsistent"
            )


def validate_trajectory_shard(
    shard: Mapping[str, Any],
    *,
    expected_semantic_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a complete canonical shard and every referenced media byte."""

    value = copy.deepcopy(dict(shard))
    if value.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise TrajectoryValidationError("trajectory schema_version is incompatible")
    prompt_rewrite_mode = value.get("prompt_rewrite_mode")
    if prompt_rewrite_mode not in PROMPT_REWRITE_MODES:
        raise TrajectoryValidationError("trajectory has an invalid prompt_rewrite_mode")

    source_sha256 = value.get("source_sha256")
    if not isinstance(source_sha256, str) or not _SHA256_PATTERN.fullmatch(
        source_sha256
    ):
        raise TrajectoryValidationError("trajectory source hash is invalid")
    try:
        source_assets_receipt = validate_source_asset_receipt(
            value.get("source_assets_receipt"),
            expected_source_sha256=source_sha256,
            expected_receipt_sha256=value.get("source_assets_receipt_sha256"),
        )
    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError(
            f"invalid source asset receipt: {exc}"
        ) from exc
    if value.get("source_assets_receipt") != source_assets_receipt:
        raise TrajectoryValidationError("trajectory source asset receipt is not canonical")
    upload_private_lineage = _validate_upload_private_lineage(
        value.get("upload_private_lineage"),
        source_assets_receipt=source_assets_receipt,
    )
    if value.get("upload_private_lineage") != upload_private_lineage:
        raise TrajectoryValidationError("trajectory private upload lineage is not canonical")

    semantic_lock = value.get("collector_semantic_lock")
    if not isinstance(semantic_lock, Mapping):
        raise TrajectoryValidationError("trajectory has no collector_semantic_lock")
    semantic_lock = dict(semantic_lock)
    recorded_lock_hash = semantic_lock.pop("semantic_lock_sha256", None)
    if recorded_lock_hash != sha256_json(semantic_lock):
        raise TrajectoryValidationError("collector semantic lock hash mismatch")
    semantic_lock["semantic_lock_sha256"] = recorded_lock_hash
    if expected_semantic_lock is not None and semantic_lock != dict(expected_semantic_lock):
        raise TrajectoryValidationError("trajectory collector semantic lock changed")
    if value.get("collector_semantic_lock_sha256") != recorded_lock_hash:
        raise TrajectoryValidationError("trajectory semantic lock reference mismatch")

    expected_tools = get_tool_schemas()
    if value.get("tools") != expected_tools or len(expected_tools) != 11:
        raise TrajectoryValidationError("trajectory must expose the exact eleven tools")
    if value.get("tool_schema_sha256") != tool_schema_hash():
        raise TrajectoryValidationError("trajectory tool schema hash mismatch")
    if semantic_lock.get("tool_schema_sha256") != value.get("tool_schema_sha256"):
        raise TrajectoryValidationError("semantic lock and trajectory tool schema differ")
    if value.get("tool_protocol_version") != TOOL_PROTOCOL_VERSION:
        raise TrajectoryValidationError("trajectory tool protocol version mismatch")
    if semantic_lock.get("tool_protocol_version") != TOOL_PROTOCOL_VERSION:
        raise TrajectoryValidationError("semantic lock tool protocol version mismatch")
    if value.get("runtime_version") != semantic_lock.get("runtime_version"):
        raise TrajectoryValidationError("trajectory runtime version mismatch")
    if value.get("pe_manifest_sha256") != semantic_lock.get("pe_manifest_sha256"):
        raise TrajectoryValidationError("trajectory PE manifest hash mismatch")
    story_plan_sha256 = value.get("story_plan_sha256")
    if not isinstance(story_plan_sha256, str) or not _SHA256_PATTERN.fullmatch(
        story_plan_sha256
    ):
        raise TrajectoryValidationError("trajectory story plan hash is invalid")
    story_pe_provenance = value.get("story_pe_provenance")
    if value.get("story_pe_provenance_sha256") != sha256_json(
        story_pe_provenance
    ):
        raise TrajectoryValidationError("trajectory story PE provenance hash mismatch")
    try:
        validate_story_pe_provenance(
            story_pe_provenance,
            expected_story_plan_sha256=story_plan_sha256,
            expected_pe_manifest_sha256=str(value.get("pe_manifest_sha256") or ""),
        )
    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError(f"invalid story PE provenance: {exc}") from exc
    if semantic_lock.get("teacher_phase_bundle_sha256") != teacher_phase_bundle_hash():
        raise TrajectoryValidationError("teacher phase-guidance bundle hash mismatch")

    messages = value.get("messages")
    if not isinstance(messages, list) or len(messages) < 4 or len(messages) % 2:
        raise TrajectoryValidationError("trajectory messages must be system, user, and tool-call pairs")
    system_rows = [row for row in messages if isinstance(row, Mapping) and row.get("role") == "system"]
    user_rows = [row for row in messages if isinstance(row, Mapping) and row.get("role") == "user"]
    if len(system_rows) != 1 or messages[0] != {
        "role": "system",
        "content": OPEN_MODEL_SYSTEM_PROMPT,
    }:
        raise TrajectoryValidationError("trajectory must contain one byte-identical canonical system")
    if len(user_rows) != 1 or messages[1] != {
        "role": "user",
        "content": value.get("raw_user_request"),
    }:
        raise TrajectoryValidationError("trajectory must contain one exact raw USER message")
    expected_system_hash = sha256_text(OPEN_MODEL_SYSTEM_PROMPT)
    if value.get("system_prompt_sha256") != expected_system_hash:
        raise TrajectoryValidationError("trajectory system prompt hash mismatch")
    if semantic_lock.get("open_model_system_prompt_sha256") != expected_system_hash:
        raise TrajectoryValidationError("semantic lock system prompt hash mismatch")

    events = value.get("events")
    if not isinstance(events, list):
        raise TrajectoryValidationError("trajectory events must be an array")
    committed_events = [row for row in events if isinstance(row, Mapping) and row.get("committed_to_training")]
    committed_event_ids = [str(row.get("event_id") or "") for row in committed_events]
    if len(committed_event_ids) != len(set(committed_event_ids)):
        raise TrajectoryValidationError("trajectory contains duplicate event_id values")
    teacher_calls = value.get("teacher_calls")
    if not isinstance(teacher_calls, list):
        raise TrajectoryValidationError("trajectory teacher_calls must be an audit array")
    if value.get("teacher_calls_sha256") != sha256_json(teacher_calls):
        raise TrajectoryValidationError("trajectory teacher call ledger hash mismatch")
    teacher_calls_by_id: dict[str, Mapping[str, Any]] = {}
    for record_index, record in enumerate(teacher_calls):
        if not isinstance(record, Mapping):
            raise TrajectoryValidationError(
                f"teacher call audit record {record_index} is invalid"
            )
        audit_id = str(record.get("teacher_call_audit_id") or "")
        if (
            record.get("schema_version") != TEACHER_CALL_LEDGER_SCHEMA_VERSION
            or not re.fullmatch(r"teachercall_[0-9]{6}", audit_id)
            or audit_id in teacher_calls_by_id
        ):
            raise TrajectoryValidationError(
                f"teacher call audit record {record_index} has invalid identity"
            )
        teacher_calls_by_id[audit_id] = record
    if len(messages) != 2 + 2 * len(committed_events):
        raise TrajectoryValidationError("training messages do not match committed event count")
    schemas = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in expected_tools
    }
    media_blocks: list[dict[str, Any]] = []
    last_name = ""
    last_observation: dict[str, Any] = {}
    for index, event in enumerate(committed_events):
        _validate_teacher_event_provenance(event, event_index=index)
        event_provenance = event["assistant_provenance"]
        if (
            event_provenance.get("teacher_runtime_context", {}).get(
                "prompt_rewrite_mode"
            )
            != prompt_rewrite_mode
        ):
            raise TrajectoryValidationError(
                f"committed event {index} has the wrong prompt rewrite mode"
            )
        audit_id = str(event_provenance.get("teacher_call_audit_id") or "")
        audit_record = teacher_calls_by_id.get(audit_id)
        if (
            audit_record is None
            or audit_record.get("response_status") != "plain_json_tool_command"
            or audit_record.get("execution_status") != "committed"
            or audit_record.get("event_id") != event.get("event_id")
            or audit_record.get("tool_call") != event.get("tool_call")
        ):
            raise TrajectoryValidationError(
                f"committed event {index} is not linked to its Teacher call audit"
            )
        comparable_provenance = copy.deepcopy(dict(event_provenance))
        comparable_provenance.pop("teacher_call_audit_id", None)
        if audit_record.get("provenance") != comparable_provenance:
            raise TrajectoryValidationError(
                f"committed event {index} provenance differs from its Teacher audit"
            )
        assistant = messages[2 + 2 * index]
        tool = messages[3 + 2 * index]
        if not isinstance(assistant, Mapping) or set(assistant) != {"role", "tool_calls"}:
            raise TrajectoryValidationError(f"assistant message {index} has an invalid shape")
        calls = assistant.get("tool_calls")
        if assistant.get("role") != "assistant" or not isinstance(calls, list) or len(calls) != 1:
            raise TrajectoryValidationError(f"assistant message {index} must contain one tool call")
        native_call = calls[0]
        if not isinstance(native_call, Mapping) or native_call.get("type") != "function":
            raise TrajectoryValidationError(f"assistant tool call {index} is invalid")
        function = native_call.get("function")
        if not isinstance(function, Mapping):
            raise TrajectoryValidationError(f"assistant tool call {index} has no function")
        event_id = str(event.get("event_id") or "")
        name = str(function.get("name") or "")
        if native_call.get("id") != event_id or name not in schemas:
            raise TrajectoryValidationError(f"assistant tool call {index} is not paired to its event")
        arguments = _strict_json_object(
            function.get("arguments"), label=f"assistant tool call {index} arguments"
        )
        event_call = event.get("tool_call")
        if not isinstance(event_call, Mapping) or event_call.get("name") != name or event_call.get("arguments") != arguments:
            raise TrajectoryValidationError(f"assistant tool call {index} differs from audit event")
        errors = sorted(
            Draft202012Validator(schemas[name]).iter_errors(arguments),
            key=lambda error: list(error.path),
        )
        if errors:
            raise TrajectoryValidationError(
                f"assistant tool call {index} fails {name} schema: {errors[0].message}"
            )
        if not isinstance(tool, Mapping) or tool.get("role") != "tool":
            raise TrajectoryValidationError(f"tool result {index} has an invalid role")
        if tool.get("tool_call_id") != event_id or tool.get("name") != name:
            raise TrajectoryValidationError(f"tool result {index} is not paired to its assistant call")
        content = tool.get("content")
        if not isinstance(content, list) or not content:
            raise TrajectoryValidationError(f"tool result {index} has no content blocks")
        text_block = content[0]
        if not isinstance(text_block, Mapping) or set(text_block) != {"type", "text"} or text_block.get("type") != "text":
            raise TrajectoryValidationError(f"tool result {index} must start with one text block")
        text_observation = _strict_json_object(
            text_block.get("text"), label=f"tool result {index} observation"
        )
        forbidden = _contains_forbidden_observation_key(text_observation)
        if forbidden:
            raise TrajectoryValidationError(
                f"training observation contains private teacher field {forbidden!r}"
            )
        event_observation = event.get("observation")
        if not isinstance(event_observation, Mapping):
            raise TrajectoryValidationError(f"event {event_id} has no observation")
        expected_text = copy.deepcopy(dict(event_observation))
        expected_media = expected_text.pop("media", []) or []
        if text_observation != expected_text:
            raise TrajectoryValidationError(f"tool result {index} differs from audit observation")
        blocks = content[1:]
        if len(blocks) != len(expected_media):
            raise TrajectoryValidationError(f"tool result {index} media count differs from audit event")
        for media_index, (block, original) in enumerate(zip(blocks, expected_media)):
            if not isinstance(block, Mapping) or block.get("type") != "media":
                raise TrajectoryValidationError(
                    f"tool result {index} media block {media_index} is invalid"
                )
            for key, original_value in dict(original).items():
                block_key = "media_type" if key == "type" else key
                expected_value = (
                    str(Path(original_value).expanduser().resolve())
                    if key == "path"
                    else original_value
                )
                if block.get(block_key) != expected_value:
                    raise TrajectoryValidationError(
                        f"tool result {index} media block {media_index} differs from audit event"
                    )
            resealed = training_media_block(block)
            if dict(block) != resealed:
                raise TrajectoryValidationError(
                    f"tool result {index} media block {media_index} manifest mismatch"
                )
            media_blocks.append(dict(block))
        last_name = name
        last_observation = text_observation

    _validate_causal_ledger(value, committed_events)
    committed_audit_ids = {
        str(event["assistant_provenance"]["teacher_call_audit_id"])
        for event in committed_events
    }
    committed_events_by_id = {
        str(event.get("event_id") or ""): event for event in committed_events
    }
    audit_history: list[Mapping[str, Any]] = []
    all_event_audit_ids = {
        str((event.get("assistant_provenance") or {}).get("teacher_call_audit_id") or "")
        for event in events
        if isinstance(event, Mapping)
    }
    for record_index, record in enumerate(teacher_calls):
        audit_id = str(record.get("teacher_call_audit_id") or "")
        if record.get("sequence") != record_index + 1:
            raise TrajectoryValidationError("Teacher call audit sequence is not contiguous")
        response_status = record.get("response_status")
        status = record.get("execution_status")
        if status == "received":
            raise TrajectoryValidationError("trajectory has an unfinished Teacher call")
        if (status == "committed") != (audit_id in committed_audit_ids):
            raise TrajectoryValidationError(
                "Teacher call audit committed status differs from trajectory events"
            )
        if response_status == "plain_json_tool_command":
            if status not in {"committed", "rejected"}:
                raise TrajectoryValidationError(
                    "plain-JSON Teacher command has an invalid final execution status"
                )
            if not isinstance(record.get("tool_call"), Mapping):
                raise TrajectoryValidationError(
                    "plain-JSON Teacher audit has no exact tool call"
                )
        elif response_status == "malformed_response":
            if (
                status != "not_applicable"
                or record.get("tool_call") is not None
                or not isinstance(record.get("error"), Mapping)
            ):
                raise TrajectoryValidationError(
                    "malformed Teacher response has an invalid audit-only record"
                )
        else:
            raise TrajectoryValidationError("Teacher audit has an invalid response status")
        _validate_teacher_provenance(
            record.get("provenance"),
            label=f"Teacher call audit record {record_index}",
            response_status=str(response_status),
            expected_input_messages=build_teacher_input_messages(
                raw_user_request=str(value.get("raw_user_request") or ""),
                events=audit_history,
            ),
        )
        provenance = record["provenance"]
        if (
            provenance.get("teacher_runtime_context", {}).get(
                "prompt_rewrite_mode"
            )
            != prompt_rewrite_mode
        ):
            raise TrajectoryValidationError(
                f"Teacher call audit record {record_index} has the wrong prompt rewrite mode"
            )
        if response_status == "plain_json_tool_command":
            parsed_call, parsed_reason = parse_plain_json_tool_command(
                provenance["raw_assistant_text"],
                label=f"Teacher call audit record {record_index} raw assistant text",
            )
            if parsed_call != record.get("tool_call"):
                raise TrajectoryValidationError(
                    "Teacher audit raw assistant text differs from its parsed tool call"
                )
            if parsed_reason != record.get("teacher_reason"):
                raise TrajectoryValidationError(
                    "Teacher audit raw assistant reason differs from its reason receipt"
                )
        if record.get("teacher_reason") != provenance.get("teacher_reason"):
            raise TrajectoryValidationError("Teacher audit reason receipt mismatch")
        if status == "committed":
            event = committed_events_by_id.get(str(record.get("event_id") or ""))
            if event is None:
                raise TrajectoryValidationError("committed Teacher audit has no event")
            audit_history.append(event)
        else:
            if audit_id in all_event_audit_ids:
                raise TrajectoryValidationError(
                    "rejected or malformed Teacher call leaked into the event ledger"
                )
            if record.get("event_id") is not None:
                raise TrajectoryValidationError("audit-only Teacher call has an event ID")
            if status == "rejected" and not isinstance(
                record.get("runtime_rejection"), Mapping
            ):
                raise TrajectoryValidationError(
                    "rejected Teacher call has no runtime rejection"
                )
    if [str(event.get("event_id") or "") for event in audit_history] != committed_event_ids:
        raise TrajectoryValidationError(
            "Teacher call audit order differs from committed event order"
        )
    if last_name != "submit_shot" or last_observation.get("done") is not True:
        raise TrajectoryValidationError("trajectory must end with submit_shot(done=true)")
    if value.get("messages_sha256") != sha256_json(messages):
        raise TrajectoryValidationError("trajectory messages hash mismatch")
    media_manifest = trajectory_media_manifest(messages)
    if value.get("media_manifest") != media_manifest:
        raise TrajectoryValidationError("trajectory media manifest differs from messages")
    if value.get("media_manifest_sha256") != sha256_json(media_manifest):
        raise TrajectoryValidationError("trajectory media manifest hash mismatch")
    return {
        "ok": True,
        "message_count": len(messages),
        "tool_call_count": len(committed_events),
        "media_count": len(media_blocks),
        "messages_sha256": value["messages_sha256"],
        "media_manifest_sha256": value["media_manifest_sha256"],
    }


def build_trajectory_shard(
    *,
    runtime_version: str,
    story_id: str,
    shot_index: int,
    shot_request_id: str,
    raw_user_request: str,
    source_sha256: str,
    source_assets_receipt: Mapping[str, Any],
    story_plan_sha256: str,
    story_pe_provenance: Mapping[str, Any],
    pe_manifest_sha256: str,
    collector_semantic_lock: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    teacher_calls: Sequence[Mapping[str, Any]],
    prompt_artifacts: Mapping[str, Any],
    pe_receipts: Mapping[str, Any],
    selections: Mapping[str, Any],
    upload_private_lineage: Mapping[str, Any] | None = None,
    prompt_rewrite_mode: str = "rewrite_enabled",
) -> dict[str, Any]:
    """Build and validate the exact envelope written by ``submit_shot``."""

    messages = build_training_messages(
        raw_user_request=raw_user_request,
        events=events,
    )
    media_manifest = trajectory_media_manifest(messages)
    semantic_lock = copy.deepcopy(dict(collector_semantic_lock))
    validated_source_assets_receipt = validate_source_asset_receipt(
        source_assets_receipt,
        expected_source_sha256=source_sha256,
    )
    if upload_private_lineage is None:
        fallback_assets: dict[str, dict[str, Any]] = {}
        for row in validated_source_assets_receipt.get("assets") or []:
            binding = row.get("speaker_binding")
            subject_ids = (
                [str(binding.get("subject_id") or "")]
                if isinstance(binding, Mapping)
                else []
            )
            fallback_assets[str(row.get("asset_id") or "")] = {
                "canonical_material_tags": list(subject_ids),
                "subject_ids": list(subject_ids),
                "speaker_binding": copy.deepcopy(binding),
            }
        upload_private_lineage = build_upload_private_lineage(
            assets=fallback_assets,
            source_assets_receipt=validated_source_assets_receipt,
        )
    output = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "runtime_version": str(runtime_version),
        "tool_protocol_version": TOOL_PROTOCOL_VERSION,
        "story_id": str(story_id),
        "shot_index": int(shot_index),
        "shot_request_id": str(shot_request_id),
        "raw_user_request": str(raw_user_request),
        "prompt_rewrite_mode": str(prompt_rewrite_mode),
        "source_sha256": str(source_sha256),
        "source_assets_receipt": validated_source_assets_receipt,
        "source_assets_receipt_sha256": validated_source_assets_receipt[
            "receipt_sha256"
        ],
        "upload_private_lineage": copy.deepcopy(dict(upload_private_lineage)),
        "story_plan_sha256": str(story_plan_sha256),
        "story_pe_provenance": copy.deepcopy(dict(story_pe_provenance)),
        "story_pe_provenance_sha256": sha256_json(story_pe_provenance),
        "system_prompt_sha256": sha256_text(OPEN_MODEL_SYSTEM_PROMPT),
        "teacher_phase_bundle_sha256": teacher_phase_bundle_hash(),
        "tool_schema_sha256": tool_schema_hash(),
        "pe_manifest_sha256": str(pe_manifest_sha256),
        "collector_semantic_lock": semantic_lock,
        "collector_semantic_lock_sha256": semantic_lock.get("semantic_lock_sha256"),
        "messages_sha256": sha256_json(messages),
        "media_manifest": media_manifest,
        "media_manifest_sha256": sha256_json(media_manifest),
        "messages": messages,
        "tools": get_tool_schemas(),
        "events": copy.deepcopy(list(events)),
        "teacher_calls": copy.deepcopy(list(teacher_calls)),
        "teacher_calls_sha256": sha256_json(teacher_calls),
        "prompt_artifacts": copy.deepcopy(dict(prompt_artifacts)),
        "pe_receipts": copy.deepcopy(dict(pe_receipts)),
        "selections": copy.deepcopy(dict(selections)),
        "created_at": utc_now(),
    }
    validate_trajectory_shard(output, expected_semantic_lock=semantic_lock)
    return output


__all__ = [
    "SEMANTIC_LOCK_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "TrajectoryValidationError",
    "build_collector_semantic_lock",
    "build_teacher_input_messages",
    "build_training_messages",
    "build_trajectory_shard",
    "build_upload_private_lineage",
    "parse_plain_json_tool_command",
    "teacher_phase_bundle_hash",
    "training_media_block",
    "trajectory_media_manifest",
    "validate_trajectory_shard",
]
