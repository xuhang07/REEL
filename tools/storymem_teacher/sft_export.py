"""Strict canonical StoryMem V3 -> provider-neutral native-tool SFT export.

The closed Teacher is called with ordinary dialogue and emits one strict JSON
command per turn.  Native function calls are materialized only here, from the
byte-exact ``raw_assistant_text`` retained in each committed event.  This
module deliberately accepts no legacy transcript or best-effort repair path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

from .contract import get_tool_schemas, tool_schema_hash
from .prompts import OPEN_MODEL_SYSTEM_PROMPT
from .trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryValidationError,
    parse_plain_json_tool_command,
    trajectory_media_manifest,
    validate_trajectory_shard,
)
from .util import canonical_json, sha256_bytes, sha256_json, sha256_text, utc_now


SFT_SAMPLE_SCHEMA_VERSION = "storymem.canonical.sft.sample.v1"
SFT_MANIFEST_SCHEMA_VERSION = "storymem.canonical.sft.manifest.v1"
SFT_SOURCE_INDEX_SCHEMA_VERSION = "storymem.canonical.sft.source-index.v1"
SFT_PROJECTION_VERSION = "plain-json-teacher-to-native-tools.v1"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SHOT_FILE_RE = re.compile(r"shot_([0-9]+)\.json")
_MOCK_TOKEN_RE = re.compile(r"(?:^|[._:/\\-])(?:mock|dummy|fake)(?:$|[._:/\\-])", re.I)


class CanonicalSFTExportError(ValueError):
    """Input cannot be represented as a trustworthy canonical SFT sample."""


@dataclass(frozen=True)
class SourceShard:
    path: Path
    file_sha256: str
    size_bytes: int
    story_id: str
    shot_index: int
    shot_request_id: str
    source_sha256: str
    story_plan_sha256: str
    story_pe_provenance_sha256: str
    semantic_lock_sha256: str
    messages_sha256: str
    media_manifest_sha256: str
    tool_call_count: int
    media_count: int


@dataclass(frozen=True)
class MediaSnapshot:
    path: Path
    size_bytes: int
    mtime_ns: int
    sha256: str


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_stable_json(path: Path) -> tuple[dict[str, Any], bytes]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise CanonicalSFTExportError(f"source shard changed while reading: {path}")
    if not payload:
        raise CanonicalSFTExportError(f"source shard is empty: {path}")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise CanonicalSFTExportError(f"invalid strict JSON shard {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CanonicalSFTExportError(f"source shard is not a JSON object: {path}")
    return value, payload


def discover_shards(inputs: Sequence[str | Path]) -> list[Path]:
    """Resolve only canonical per-shot files, never legacy transcript JSONL."""

    if not inputs:
        raise CanonicalSFTExportError("at least one input path is required")
    found: dict[str, Path] = {}
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("shots/shot_*.json"))
            if path.name == "shots":
                candidates.extend(sorted(path.glob("shot_*.json")))
        else:
            raise CanonicalSFTExportError(f"input does not exist: {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            found[str(resolved)] = resolved
    paths = [found[key] for key in sorted(found)]
    if not paths:
        raise CanonicalSFTExportError(
            "no canonical shots/shot_*.json shards found; legacy JSONL/transcripts are unsupported"
        )
    return paths


def _committed_events(shard: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = shard.get("events")
    if not isinstance(events, list):
        raise CanonicalSFTExportError("trajectory events must be an array")
    return [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("committed_to_training") is True
    ]


def _event_media_blocks(shard: Mapping[str, Any], event_index: int) -> list[dict[str, Any]]:
    messages = shard.get("messages")
    message_index = 3 + 2 * event_index
    if not isinstance(messages, list) or message_index >= len(messages):
        raise CanonicalSFTExportError("trajectory is missing its sealed tool observation")
    message = messages[message_index]
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, list) or not content:
        raise CanonicalSFTExportError("trajectory tool observation has no content blocks")
    return [
        copy.deepcopy(dict(block))
        for block in content[1:]
        if isinstance(block, Mapping) and block.get("type") == "media"
    ]


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if not _SHA256_RE.fullmatch(digest):
        raise CanonicalSFTExportError(f"{label} is not a canonical SHA-256 digest")
    return digest


def _reject_mock_marker(value: Any, *, label: str) -> None:
    text = str(value or "")
    if _MOCK_TOKEN_RE.search(text):
        raise CanonicalSFTExportError(f"{label} contains a mock/fake marker: {text!r}")


def _has_successful_http_attempt(attempts: Any) -> bool:
    return isinstance(attempts, list) and any(
        isinstance(item, Mapping)
        and type(item.get("http_status")) is int
        and 200 <= int(item["http_status"]) < 300
        and _SHA256_RE.fullmatch(str(item.get("response_sha256") or "")) is not None
        for item in attempts
    )


def _validate_real_generation(
    *,
    event_index: int,
    name: str,
    observation: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Validate a real provider receipt without exposing candidate bytes yet.

    The canonical runtime deliberately makes generation return only an opaque
    candidate handle plus an audit-safe representation manifest.  The actual
    bytes become model-visible in the immediately following
    ``inspect_candidate`` observation, where they can be bound to this receipt
    and checked independently.
    """

    target = "image" if name == "generate_image" else "video"
    candidate_id = str(observation.get("candidate_id") or "")
    representation = observation.get("representation")
    if not candidate_id or not isinstance(representation, Mapping):
        raise CanonicalSFTExportError(
            f"generation event {event_index} has no candidate/representation"
        )
    representation_id = str(representation.get("representation_id") or "")
    if not representation_id:
        raise CanonicalSFTExportError(
            f"generation event {event_index} has no representation_id"
        )
    if representation.get("media_type") != target:
        raise CanonicalSFTExportError(
            f"generation event {event_index} representation has the wrong media type"
        )
    mime_type = str(representation.get("mime_type") or "")
    size_bytes = representation.get("size_bytes")
    if not mime_type.startswith(f"{target}/"):
        raise CanonicalSFTExportError(
            f"generation event {event_index} representation MIME is not {target}"
        )
    if type(size_bytes) is not int or size_bytes <= 0:
        raise CanonicalSFTExportError(
            f"generation event {event_index} representation size is invalid"
        )
    _require_sha256(
        representation.get("sha256"),
        label=f"generation event {event_index} representation sha256",
    )

    provenance = observation.get("provider_provenance")
    if not isinstance(provenance, Mapping):
        raise CanonicalSFTExportError(
            f"generation event {event_index} has no provider provenance"
        )
    if provenance.get("provider_kind") != target:
        raise CanonicalSFTExportError(
            f"generation event {event_index} provider kind is not {target}"
        )
    adapter = str(provenance.get("adapter") or "")
    request_id = str(provenance.get("request_id") or "")
    if not adapter or not request_id:
        raise CanonicalSFTExportError(
            f"generation event {event_index} lacks provider adapter/request identity"
        )
    _reject_mock_marker(adapter, label=f"generation event {event_index} provider adapter")
    if provenance.get("prompt_transport") != "verbatim":
        raise CanonicalSFTExportError(
            f"generation event {event_index} did not use verbatim prompt transport"
        )
    for key in (
        "endpoint_sha256",
        "request_payload_sha256",
        "prompt_sha256",
        "output_sha256",
    ):
        _require_sha256(provenance.get(key), label=f"generation event {event_index} {key}")
    if provenance.get("output_sha256") != representation.get("sha256"):
        raise CanonicalSFTExportError(
            f"generation event {event_index} provider output hash differs from its representation"
        )
    if provenance.get("output_mime_type") != representation.get("mime_type"):
        raise CanonicalSFTExportError(
            f"generation event {event_index} provider output MIME differs from its representation"
        )
    attempts = (
        provenance.get("http_attempts")
        if target == "image"
        else provenance.get("submit_attempts")
    )
    if not _has_successful_http_attempt(attempts):
        raise CanonicalSFTExportError(
            f"generation event {event_index} has no successful real provider HTTP receipt"
        )
    return candidate_id, copy.deepcopy(dict(representation)), copy.deepcopy(dict(provenance))


def _validate_inspected_generation_media(
    *,
    event_index: int,
    candidate_id: str,
    generation: Mapping[str, Any],
    media_blocks: Sequence[Mapping[str, Any]],
) -> None:
    """Bind the sole visible candidate representation to its provider receipt."""

    target = str(generation["target"])
    representation = generation["representation"]
    provenance = generation["provenance"]
    representation_id = str(representation["representation_id"])
    target_evidence_kind = "target_image" if target == "image" else "target_video"
    matches = [
        block
        for block in media_blocks
        if block.get("source_id") == candidate_id
        and block.get("media_type") == target
        and block.get("evidence_kind") == target_evidence_kind
    ]
    if len(matches) != 1:
        raise CanonicalSFTExportError(
            f"inspection event {event_index} must expose the exact generated "
            f"{target} bytes once"
        )
    media = matches[0]
    media_path = Path(str(media.get("path") or ""))
    if not media_path.is_file() or media_path.stat().st_size <= 0:
        raise CanonicalSFTExportError(
            f"inspection event {event_index} media is missing or empty: {media_path}"
        )
    _reject_mock_marker(media_path, label=f"inspection event {event_index} media path")
    visible_representation_id = str(media.get("representation_id") or "")
    if visible_representation_id == representation_id:
        if media.get("parent_sha256") is not None or media.get("derivation") is not None:
            raise CanonicalSFTExportError(
                f"inspection event {event_index} exact representation has unexpected derivation"
            )
        for key in ("mime_type", "size_bytes", "sha256"):
            if representation.get(key) != media.get(key):
                raise CanonicalSFTExportError(
                    f"inspection event {event_index} generation representation differs "
                    f"from visible media on {key}"
                )
        if provenance.get("output_sha256") != media.get("sha256"):
            raise CanonicalSFTExportError(
                f"inspection event {event_index} provider output hash differs from visible media"
            )
        if provenance.get("output_mime_type") != media.get("mime_type"):
            raise CanonicalSFTExportError(
                f"inspection event {event_index} provider output MIME differs from visible media"
            )
        return

    # In an audio-disabled run, the exact provider video may not be exposed to
    # the Teacher because its soundtrack would cross the capability boundary.
    # The only accepted substitute is a byte-sealed visual-only derivation
    # explicitly linked to the original representation and provider hash.
    expected_derivation = {
        "operation": "strip_audio_for_visual_perception",
        "source_representation_id": representation_id,
    }
    if (
        target != "video"
        or media.get("parent_sha256") != representation.get("sha256")
        or media.get("parent_sha256") != provenance.get("output_sha256")
        or media.get("derivation") != expected_derivation
        or visible_representation_id == ""
    ):
        raise CanonicalSFTExportError(
            f"inspection event {event_index} visual-only proxy has invalid provider lineage"
        )


def validate_exportable_shard(
    shard: Mapping[str, Any],
    *,
    expected_semantic_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply V3 validation plus SFT-specific real-media/projection gates."""

    if shard.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise CanonicalSFTExportError(
            "only canonical StoryMem trajectory v3 shards are exportable"
        )
    try:
        base = validate_trajectory_shard(
            shard, expected_semantic_lock=expected_semantic_lock
        )
    except (TrajectoryValidationError, OSError, ValueError) as exc:
        raise CanonicalSFTExportError(f"canonical shard validation failed: {exc}") from exc

    events = _committed_events(shard)
    generated: dict[str, dict[str, Any]] = {}
    inspected: set[str] = set()
    generation_count = {"image": 0, "video": 0}
    all_media: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        call = event.get("tool_call") or {}
        observation = event.get("observation") or {}
        name = str(call.get("name") or "")
        blocks = _event_media_blocks(shard, index)
        all_media.extend(blocks)
        for block in blocks:
            _reject_mock_marker(
                Path(str(block.get("path") or "")),
                label=f"event {index} media path",
            )
        if name in {"generate_image", "generate_shot"}:
            if blocks:
                raise CanonicalSFTExportError(
                    f"generation event {index} exposes media before inspect_candidate"
                )
            candidate_id, representation, provenance = _validate_real_generation(
                event_index=index,
                name=name,
                observation=observation,
            )
            if candidate_id in generated:
                raise CanonicalSFTExportError(f"duplicate generated candidate {candidate_id}")
            target = "image" if name == "generate_image" else "video"
            generated[candidate_id] = {
                "target": target,
                "representation": representation,
                "provenance": provenance,
            }
            generation_count[target] += 1
        elif name == "inspect_candidate":
            candidate_id = str(observation.get("candidate_id") or "")
            generation = generated.get(candidate_id)
            if generation is None:
                raise CanonicalSFTExportError(
                    f"inspection event {index} has no prior generated candidate"
                )
            if candidate_id in inspected:
                raise CanonicalSFTExportError(
                    f"generated candidate {candidate_id} was inspected more than once"
                )
            _validate_inspected_generation_media(
                event_index=index,
                candidate_id=candidate_id,
                generation=generation,
                media_blocks=blocks,
            )
            inspected.add(candidate_id)

    if generation_count["video"] < 1:
        raise CanonicalSFTExportError("completed shot contains no real video generation")
    if set(generated) != inspected:
        missing = sorted(set(generated) - inspected)
        raise CanonicalSFTExportError(
            f"generated candidates were not inspected with visible media: {missing}"
        )
    final_event = events[-1]
    final_call = final_event.get("tool_call") or {}
    final_observation = final_event.get("observation") or {}
    if final_call.get("name") != "submit_shot" or final_observation.get("done") is not True:
        raise CanonicalSFTExportError("shot is unfinished: expected submit_shot(done=true)")
    selection_id = str((final_call.get("arguments") or {}).get("video_selection_id") or "")
    selection = (shard.get("selections") or {}).get(selection_id)
    candidate_id = (
        str(selection.get("selected_candidate_id") or "")
        if isinstance(selection, Mapping)
        else ""
    )
    if (generated.get(candidate_id) or {}).get("target") != "video" or candidate_id not in inspected:
        raise CanonicalSFTExportError(
            "submitted video was not both generated and inspected with visible media"
        )
    if not all_media:
        raise CanonicalSFTExportError("trajectory contains no model-visible media")
    return {
        **base,
        "image_generation_count": generation_count["image"],
        "video_generation_count": generation_count["video"],
    }


def project_plain_teacher_to_native_messages(
    shard: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project audited plain JSON Teacher turns to native function-call turns."""

    raw_user_request = shard.get("raw_user_request")
    if not isinstance(raw_user_request, str) or not raw_user_request:
        raise CanonicalSFTExportError("raw_user_request must be a non-empty string")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": OPEN_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": raw_user_request},
    ]
    source_messages = shard.get("messages")
    if not isinstance(source_messages, list):
        raise CanonicalSFTExportError("validated shard has no sealed observations")
    for event_index, event in enumerate(_committed_events(shard)):
        provenance = event.get("assistant_provenance")
        if not isinstance(provenance, Mapping):
            raise CanonicalSFTExportError(
                f"event {event_index} lacks closed-Teacher provenance"
            )
        raw_text = provenance.get("raw_assistant_text")
        if not isinstance(raw_text, str):
            raise CanonicalSFTExportError(
                f"event {event_index} lacks raw plain-JSON assistant text"
            )
        try:
            parsed_call, _ = parse_plain_json_tool_command(
                raw_text, label=f"event {event_index} raw assistant text"
            )
        except TrajectoryValidationError as exc:
            raise CanonicalSFTExportError(str(exc)) from exc
        if parsed_call != event.get("tool_call"):
            raise CanonicalSFTExportError(
                f"event {event_index} plain JSON differs from committed call"
            )
        event_id = str(event.get("event_id") or "")
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": event_id,
                        "type": "function",
                        "function": {
                            "name": parsed_call["name"],
                            "arguments": canonical_json(parsed_call["arguments"]),
                        },
                    }
                ],
            }
        )
        source_tool = source_messages[3 + 2 * event_index]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": event_id,
                "name": parsed_call["name"],
                "content": copy.deepcopy(source_tool["content"]),
            }
        )
    return messages


def build_sft_sample(
    shard: Mapping[str, Any],
    *,
    source_path: str | Path,
    source_file_sha256: str,
) -> dict[str, Any]:
    """Build one self-auditing, provider-neutral SFT row from one valid shot."""

    messages = project_plain_teacher_to_native_messages(shard)
    media_manifest = trajectory_media_manifest(messages)
    story_id = str(shard.get("story_id") or "")
    shot_index = int(shard.get("shot_index"))
    body = {
        "schema_version": SFT_SAMPLE_SCHEMA_VERSION,
        "id": f"{story_id}:shot:{shot_index:06d}",
        "story_id": story_id,
        "shot_index": shot_index,
        "shot_request_id": str(shard.get("shot_request_id") or ""),
        "source_sha256": str(shard.get("source_sha256") or ""),
        "story_plan_sha256": str(shard.get("story_plan_sha256") or ""),
        "system_prompt_sha256": sha256_text(OPEN_MODEL_SYSTEM_PROMPT),
        "tool_schema_sha256": tool_schema_hash(),
        "projection": {
            "version": SFT_PROJECTION_VERSION,
            "source_transport": "plain_json_tool_dialogue",
            "target_transport": "native_function_calls",
            "teacher_reason_exported": False,
        },
        "source_shard": {
            "path": str(Path(source_path).expanduser().resolve()),
            "file_sha256": source_file_sha256,
            "trajectory_messages_sha256": str(shard.get("messages_sha256") or ""),
            "teacher_calls_sha256": str(shard.get("teacher_calls_sha256") or ""),
            "collector_semantic_lock_sha256": str(
                shard.get("collector_semantic_lock_sha256") or ""
            ),
        },
        "messages": messages,
        "tools": get_tool_schemas(),
        "media_manifest": media_manifest,
        "messages_sha256": sha256_json(messages),
        "media_manifest_sha256": sha256_json(media_manifest),
    }
    return {**body, "sample_sha256": sha256_json(body)}


def validate_sft_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the portable output row without consulting collector internals."""

    value = copy.deepcopy(dict(sample))
    digest = value.pop("sample_sha256", None)
    if digest != sha256_json(value):
        raise CanonicalSFTExportError("SFT sample hash mismatch")
    if value.get("schema_version") != SFT_SAMPLE_SCHEMA_VERSION:
        raise CanonicalSFTExportError("SFT sample schema is incompatible")
    if value.get("tools") != get_tool_schemas():
        raise CanonicalSFTExportError("SFT sample tool schemas are not canonical")
    if value.get("tool_schema_sha256") != tool_schema_hash():
        raise CanonicalSFTExportError("SFT sample tool schema hash mismatch")
    messages = value.get("messages")
    if not isinstance(messages, list) or len(messages) < 4 or len(messages) % 2:
        raise CanonicalSFTExportError("SFT sample has an invalid message count")
    if messages[0] != {"role": "system", "content": OPEN_MODEL_SYSTEM_PROMPT}:
        raise CanonicalSFTExportError("SFT sample does not use the one canonical system prompt")
    if not isinstance(messages[1], Mapping) or set(messages[1]) != {"role", "content"}:
        raise CanonicalSFTExportError("SFT sample raw USER turn has an invalid shape")
    if messages[1].get("role") != "user" or not isinstance(messages[1].get("content"), str):
        raise CanonicalSFTExportError("SFT sample raw USER turn is invalid")
    for index in range(2, len(messages), 2):
        assistant = messages[index]
        tool = messages[index + 1]
        if not isinstance(assistant, Mapping) or set(assistant) != {"role", "tool_calls"}:
            raise CanonicalSFTExportError("assistant turn is not a native tool call")
        calls = assistant.get("tool_calls")
        if assistant.get("role") != "assistant" or not isinstance(calls, list) or len(calls) != 1:
            raise CanonicalSFTExportError("assistant turn must contain exactly one native tool call")
        call = calls[0]
        function = call.get("function") if isinstance(call, Mapping) else None
        if (
            not isinstance(function, Mapping)
            or call.get("type") != "function"
            or not isinstance(function.get("name"), str)
        ):
            raise CanonicalSFTExportError("assistant native tool call is malformed")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise CanonicalSFTExportError("native tool arguments must be canonical JSON text")
        try:
            parsed_arguments = json.loads(arguments, parse_constant=_reject_constant)
        except (ValueError, json.JSONDecodeError) as exc:
            raise CanonicalSFTExportError("native tool arguments are invalid JSON") from exc
        if not isinstance(parsed_arguments, dict) or arguments != canonical_json(parsed_arguments):
            raise CanonicalSFTExportError("native tool arguments are not canonicalized")
        if (
            not isinstance(tool, Mapping)
            or tool.get("role") != "tool"
            or tool.get("tool_call_id") != call.get("id")
            or tool.get("name") != function.get("name")
            or not isinstance(tool.get("content"), list)
            or not tool.get("content")
        ):
            raise CanonicalSFTExportError("native tool result is not paired to its call")
    if messages[-2]["tool_calls"][0]["function"]["name"] != "submit_shot":
        raise CanonicalSFTExportError("SFT sample does not end at submit_shot")
    final_text = messages[-1]["content"][0]
    if not isinstance(final_text, Mapping) or final_text.get("type") != "text":
        raise CanonicalSFTExportError("submit_shot result has no text observation")
    try:
        final_observation = json.loads(
            str(final_text.get("text") or ""), parse_constant=_reject_constant
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise CanonicalSFTExportError("submit_shot result is invalid JSON") from exc
    if not isinstance(final_observation, Mapping) or final_observation.get("done") is not True:
        raise CanonicalSFTExportError("SFT sample does not end with done=true")
    if value.get("messages_sha256") != sha256_json(messages):
        raise CanonicalSFTExportError("SFT sample messages hash mismatch")
    media_manifest = trajectory_media_manifest(messages)
    if value.get("media_manifest") != media_manifest:
        raise CanonicalSFTExportError("SFT sample media manifest mismatch")
    if value.get("media_manifest_sha256") != sha256_json(media_manifest):
        raise CanonicalSFTExportError("SFT sample media manifest hash mismatch")
    return {
        "ok": True,
        "tool_call_count": (len(messages) - 2) // 2,
        "media_count": len(media_manifest),
    }


def _source_record(
    path: Path,
    shard: Mapping[str, Any],
    payload: bytes,
    validation: Mapping[str, Any],
) -> SourceShard:
    match = _SHOT_FILE_RE.fullmatch(path.name)
    shot_index = shard.get("shot_index")
    if type(shot_index) is not int or shot_index < 1:
        raise CanonicalSFTExportError(f"invalid positive shot_index in {path}")
    if not match or int(match.group(1)) != shot_index:
        raise CanonicalSFTExportError(
            f"canonical shard filename does not match shot_index in {path}"
        )
    fields = {
        key: str(shard.get(key) or "")
        for key in (
            "story_id",
            "shot_request_id",
            "source_sha256",
            "story_plan_sha256",
            "story_pe_provenance_sha256",
            "collector_semantic_lock_sha256",
            "messages_sha256",
            "media_manifest_sha256",
        )
    }
    if not fields["story_id"] or not fields["shot_request_id"]:
        raise CanonicalSFTExportError(f"shard identity is incomplete: {path}")
    for key in (
        "source_sha256",
        "story_plan_sha256",
        "story_pe_provenance_sha256",
        "collector_semantic_lock_sha256",
        "messages_sha256",
        "media_manifest_sha256",
    ):
        _require_sha256(fields[key], label=f"{path} {key}")
    return SourceShard(
        path=path,
        file_sha256=sha256_bytes(payload),
        size_bytes=len(payload),
        story_id=fields["story_id"],
        shot_index=shot_index,
        shot_request_id=fields["shot_request_id"],
        source_sha256=fields["source_sha256"],
        story_plan_sha256=fields["story_plan_sha256"],
        story_pe_provenance_sha256=fields["story_pe_provenance_sha256"],
        semantic_lock_sha256=fields["collector_semantic_lock_sha256"],
        messages_sha256=fields["messages_sha256"],
        media_manifest_sha256=fields["media_manifest_sha256"],
        tool_call_count=int(validation["tool_call_count"]),
        media_count=int(validation["media_count"]),
    )


def _snapshot_media(
    shard: Mapping[str, Any], snapshots: dict[str, MediaSnapshot]
) -> None:
    for row in shard.get("media_manifest") or []:
        if not isinstance(row, Mapping):
            raise CanonicalSFTExportError("media manifest contains a non-object row")
        path = Path(str(row.get("path") or "")).resolve()
        stat = path.stat()
        current = MediaSnapshot(
            path=path,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            sha256=str(row.get("sha256") or ""),
        )
        prior = snapshots.get(str(path))
        if prior is not None and (
            prior.size_bytes != current.size_bytes or prior.sha256 != current.sha256
        ):
            raise CanonicalSFTExportError(
                f"one media path has conflicting immutable manifests: {path}"
            )
        snapshots[str(path)] = current


def _validate_story_catalog(records: Sequence[SourceShard]) -> dict[str, list[SourceShard]]:
    stories: dict[str, list[SourceShard]] = {}
    # StateStore allocates request IDs independently inside each story.  The
    # stable public identity is therefore the story-scoped pair, not the bare
    # request ID (which legitimately restarts at shotreq_000001).
    request_ids: set[tuple[str, str]] = set()
    for record in records:
        request_key = (record.story_id, record.shot_request_id)
        if request_key in request_ids:
            raise CanonicalSFTExportError(
                "duplicate story-local shot_request_id: "
                f"{record.story_id}:{record.shot_request_id}"
            )
        request_ids.add(request_key)
        stories.setdefault(record.story_id, []).append(record)

    source_owner: dict[str, str] = {}
    for story_id, rows in stories.items():
        rows.sort(key=lambda row: row.shot_index)
        indices = [row.shot_index for row in rows]
        if indices != list(range(1, len(rows) + 1)):
            raise CanonicalSFTExportError(
                f"story {story_id!r} is not a contiguous committed prefix: {indices}"
            )
        for field in (
            "source_sha256",
            "story_plan_sha256",
            "story_pe_provenance_sha256",
            "semantic_lock_sha256",
        ):
            if len({getattr(row, field) for row in rows}) != 1:
                raise CanonicalSFTExportError(
                    f"story {story_id!r} changes {field} between shots"
                )
        source_sha256 = rows[0].source_sha256
        owner = source_owner.setdefault(source_sha256, story_id)
        if owner != story_id:
            raise CanonicalSFTExportError(
                f"duplicate source story appears under IDs {owner!r} and {story_id!r}"
            )
    return stories


def choose_validation_stories(
    story_ids: Sequence[str], *, seed: str, count: int
) -> set[str]:
    if count < 0 or count > len(story_ids):
        raise CanonicalSFTExportError("validation story count is out of range")
    ranked = sorted(
        set(story_ids),
        key=lambda story_id: (
            hashlib.sha256(f"{seed}\0{story_id}".encode("utf-8")).hexdigest(),
            story_id,
        ),
    )
    return set(ranked[:count])


class _JSONLWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: BinaryIO = path.open("xb")
        self.digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0

    def write(self, value: Mapping[str, Any]) -> None:
        payload = (canonical_json(value) + "\n").encode("utf-8")
        self.handle.write(payload)
        self.digest.update(payload)
        self.rows += 1
        self.bytes += len(payload)

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": "sha256:" + self.digest.hexdigest(),
        }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _media_still_unchanged(snapshots: Iterable[MediaSnapshot]) -> None:
    for snapshot in snapshots:
        try:
            stat = snapshot.path.stat()
        except FileNotFoundError as exc:
            raise CanonicalSFTExportError(
                f"media disappeared during export: {snapshot.path}"
            ) from exc
        if stat.st_size != snapshot.size_bytes or stat.st_mtime_ns != snapshot.mtime_ns:
            raise CanonicalSFTExportError(
                f"media changed during export: {snapshot.path}"
            )


def export_canonical_sft(
    *,
    inputs: Sequence[str | Path],
    output_dir: str | Path,
    val_fraction: float = 0.05,
    val_stories: int | None = None,
    split_seed: str = "storymem-canonical-v3-split-v1",
    expected_stories: int | None = None,
    expected_shots: int | None = None,
) -> dict[str, Any]:
    """Validate a point-in-time shard set and atomically publish JSONL files."""

    if not 0.0 <= val_fraction < 1.0:
        raise CanonicalSFTExportError("val_fraction must be in [0, 1)")
    if val_stories is not None and val_stories < 0:
        raise CanonicalSFTExportError("val_stories must be non-negative")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise CanonicalSFTExportError(f"refusing to overwrite output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    paths = discover_shards(inputs)
    records: list[SourceShard] = []
    media_snapshots: dict[str, MediaSnapshot] = {}
    semantic_lock: Mapping[str, Any] | None = None
    seen_shots: dict[tuple[str, int], SourceShard] = {}
    duplicate_sources: list[dict[str, str]] = []
    for path in paths:
        shard, payload = _read_stable_json(path)
        if semantic_lock is None:
            candidate_lock = shard.get("collector_semantic_lock")
            if not isinstance(candidate_lock, Mapping):
                raise CanonicalSFTExportError(f"shard has no semantic lock: {path}")
            semantic_lock = copy.deepcopy(dict(candidate_lock))
        validation = validate_exportable_shard(
            shard, expected_semantic_lock=semantic_lock
        )
        record = _source_record(path, shard, payload, validation)
        key = (record.story_id, record.shot_index)
        prior = seen_shots.get(key)
        if prior is not None:
            if prior.file_sha256 != record.file_sha256:
                raise CanonicalSFTExportError(
                    f"conflicting shards for story/shot {key}: {prior.path} and {record.path}"
                )
            duplicate_sources.append(
                {"kept": str(prior.path), "ignored_identical": str(record.path)}
            )
            continue
        seen_shots[key] = record
        records.append(record)
        _snapshot_media(shard, media_snapshots)

    stories = _validate_story_catalog(records)
    story_count = len(stories)
    shot_count = len(records)
    if expected_stories is not None and story_count != expected_stories:
        raise CanonicalSFTExportError(
            f"expected {expected_stories} stories, found {story_count}"
        )
    if expected_shots is not None and shot_count != expected_shots:
        raise CanonicalSFTExportError(
            f"expected {expected_shots} shots, found {shot_count}"
        )
    if val_stories is None:
        validation_count = int(round(story_count * val_fraction))
        if val_fraction > 0 and story_count > 1:
            validation_count = max(1, min(validation_count, story_count - 1))
    else:
        validation_count = val_stories
    validation_story_ids = choose_validation_stories(
        list(stories), seed=split_seed, count=validation_count
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    train_writer: _JSONLWriter | None = None
    validation_writer: _JSONLWriter | None = None
    index_writer: _JSONLWriter | None = None
    try:
        train_writer = _JSONLWriter(staging / "train.jsonl")
        validation_writer = _JSONLWriter(staging / "validation.jsonl")
        index_writer = _JSONLWriter(staging / "source_index.jsonl")
        output_indices = {"train": 0, "validation": 0}
        for record in sorted(records, key=lambda row: (row.story_id, row.shot_index)):
            shard, payload = _read_stable_json(record.path)
            if sha256_bytes(payload) != record.file_sha256:
                raise CanonicalSFTExportError(
                    f"source shard changed after validation: {record.path}"
                )
            sample = build_sft_sample(
                shard,
                source_path=record.path,
                source_file_sha256=record.file_sha256,
            )
            validate_sft_sample(sample)
            split = "validation" if record.story_id in validation_story_ids else "train"
            writer = validation_writer if split == "validation" else train_writer
            writer.write(sample)
            index_writer.write(
                {
                    "schema_version": SFT_SOURCE_INDEX_SCHEMA_VERSION,
                    "story_id": record.story_id,
                    "shot_index": record.shot_index,
                    "shot_request_id": record.shot_request_id,
                    "split": split,
                    "output_row_index": output_indices[split],
                    "source_path": str(record.path),
                    "source_file_sha256": record.file_sha256,
                    "source_size_bytes": record.size_bytes,
                    "source_sha256": record.source_sha256,
                    "story_plan_sha256": record.story_plan_sha256,
                    "messages_sha256": sample["messages_sha256"],
                    "media_manifest_sha256": sample["media_manifest_sha256"],
                    "sample_sha256": sample["sample_sha256"],
                    "tool_call_count": record.tool_call_count,
                    "media_count": record.media_count,
                }
            )
            output_indices[split] += 1
        train_writer.close()
        validation_writer.close()
        index_writer.close()
        _media_still_unchanged(media_snapshots.values())

        files = {
            "train": train_writer.summary(),
            "validation": validation_writer.summary(),
            "source_index": index_writer.summary(),
        }
        manifest_body = {
            "schema_version": SFT_MANIFEST_SCHEMA_VERSION,
            "created_at": utc_now(),
            "projection_version": SFT_PROJECTION_VERSION,
            "source_trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
            "system_prompt_sha256": sha256_text(OPEN_MODEL_SYSTEM_PROMPT),
            "tool_schema_sha256": tool_schema_hash(),
            "collector_semantic_lock_sha256": records[0].semantic_lock_sha256,
            "split": {
                "unit": "story",
                "method": "sha256_rank",
                "seed": split_seed,
                "requested_val_fraction": val_fraction,
                "requested_val_stories": val_stories,
                "validation_story_ids_sha256": sha256_json(
                    sorted(validation_story_ids)
                ),
            },
            "counts": {
                "stories": story_count,
                "shots": shot_count,
                "train_stories": story_count - len(validation_story_ids),
                "validation_stories": len(validation_story_ids),
                "train_shots": train_writer.rows,
                "validation_shots": validation_writer.rows,
                "unique_media_files": len(media_snapshots),
                "identical_duplicate_shards_ignored": len(duplicate_sources),
            },
            "files": files,
            "identical_duplicate_shards": duplicate_sources,
            "invariants": {
                "one_sample_per_completed_shot": True,
                "one_complete_system_prompt_per_sample": True,
                "raw_user_request_only": True,
                "assistant_native_tool_calls": True,
                "teacher_collection_transport": "plain_json_tool_dialogue",
                "real_generation_media_required": True,
                "story_disjoint_split": True,
                "legacy_and_mock_inputs_rejected": True,
            },
        }
        manifest = {
            **manifest_body,
            "manifest_sha256": sha256_json(manifest_body),
        }
        _atomic_json(staging / "manifest.json", manifest)
        _fsync_directory(staging)
        if destination.exists():
            raise CanonicalSFTExportError(
                f"output directory appeared during export: {destination}"
            )
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return manifest
    except BaseException:
        for writer in (train_writer, validation_writer, index_writer):
            if writer is not None:
                writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "CanonicalSFTExportError",
    "SFT_MANIFEST_SCHEMA_VERSION",
    "SFT_PROJECTION_VERSION",
    "SFT_SAMPLE_SCHEMA_VERSION",
    "build_sft_sample",
    "choose_validation_stories",
    "discover_shards",
    "export_canonical_sft",
    "project_plain_teacher_to_native_messages",
    "validate_exportable_shard",
    "validate_sft_sample",
]
