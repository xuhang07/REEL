"""Auditable trajectory format for the model-independent Memory Policy.

The closed teacher is intentionally used as an ordinary text chat model.  It
emits one strict JSON object per turn.  This module verifies that byte-retained
text against the call committed by :mod:`storymem_policy.runtime`, then builds
the native function-call conversation used for open-model training.

Provider receipts and filesystem paths remain in private top-level manifests;
they are never copied into model-visible messages.
"""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .contract import (
    CANONICAL_TOOL_NAMES,
    TOOL_PROTOCOL_VERSION,
    get_tool_schemas,
    tool_schema_hash,
)
from .prompts import OPEN_MODEL_SYSTEM_PROMPT, system_prompt_for_mode
from .runtime import RUNTIME_VERSION
from .safety import (
    validate_neutral_binding_outcome,
    validate_neutral_reason_code,
    validate_neutral_semantic_effect,
    validate_public_representation,
)


TRAJECTORY_SCHEMA_VERSION = "storymem.memory-policy.trajectory.v2"
_HOST_EVENT_NAMES = frozenset(
    {"attach_generation_evidence", "attach_compilation_feedback"}
)
_FSM_TRANSITIONS = {
    ("tool", "view_material_registry", "need_registry"): frozenset({"planning"}),
    ("tool", "view_material_registry", "writeback"): frozenset({"writeback"}),
    ("tool", "search_memory_bank", "planning"): frozenset({"planning"}),
    ("tool", "view_memories", "planning"): frozenset({"planning"}),
    ("tool", "view_memories", "generation_review"): frozenset({"generation_review"}),
    ("tool", "view_memories", "writeback"): frozenset({"writeback"}),
    ("tool", "generate_image", "planning"): frozenset({"planning"}),
    ("tool", "inspect_candidate", "planning"): frozenset({"planning"}),
    ("tool", "inspect_candidate", "generation_review"): frozenset({"generation_review"}),
    ("tool", "inspect_candidate", "writeback"): frozenset({"writeback"}),
    ("tool", "submit_candidate_assessment", "planning"): frozenset({"planning"}),
    ("tool", "submit_candidate_assessment", "generation_review"): frozenset(
        {"planning", "awaiting_final_evidence"}
    ),
    ("tool", "submit_candidate_assessment", "writeback"): frozenset({"writeback"}),
    ("tool", "submit_evidence_plan", "planning"): frozenset({"awaiting_generation"}),
    ("tool", "update_memory_bank", "writeback"): frozenset({"writeback"}),
    ("tool", "finish_memory_session", "writeback"): frozenset({"complete"}),
    ("host", "attach_compilation_feedback", "awaiting_generation"): frozenset(
        {"planning"}
    ),
    ("host", "attach_generation_evidence", "awaiting_generation"): frozenset(
        {"generation_review"}
    ),
    ("host", "attach_generation_evidence", "awaiting_final_evidence"): frozenset(
        {"writeback"}
    ),
}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "private_creation_receipt",
        "private_receipt",
        "generation_bundle_private",
        "generation_bundle_history_private",
        "compilation_receipt",
        "provider",
        "provider_request",
        "provider_payload",
        "native_payload",
        "native_request",
        "backend_payload",
        "generator_prompt",
        "pe_subdialog",
        "content_path",
        "provider_response",
        "path",
        "output_path",
        "condition_img",
        "memory_slots",
        "fl2va",
        "ref2va",
    }
)
_NORMALIZED_FORBIDDEN_MODEL_KEYS = frozenset(
    re.sub(r"[^a-z0-9]", "", key.casefold()) for key in _FORBIDDEN_MODEL_KEYS
)

_TRAJECTORY_FIELDS = frozenset(
    {
        "schema_version",
        "runtime_version",
        "tool_protocol_version",
        "session_id",
        "request_id",
        "raw_user_request",
        "raw_request_sha256",
        "condition_budget",
        "condition_budget_sha256",
        "policy_limits",
        "policy_limits_sha256",
        "system_prompt_sha256",
        "tool_schema_sha256",
        "tools",
        "events",
        "host_events",
        "plans",
        "messages",
        "messages_sha256",
        "media_manifest",
        "media_manifest_sha256",
        "private_media_manifest",
        "private_generation_bundles",
        "private_generation_reservations",
        "private_compilation_feedback",
        "final_phase",
    }
)
_TOOL_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "tool_call",
        "observation",
        "phase_before",
        "phase_after",
        "media_attachments_private",
        "assistant_provenance",
    }
)
_ASSISTANT_PROVENANCE_REQUIRED_FIELDS = frozenset(
    {
        "assistant_source",
        "teacher_transport",
        "model",
        "raw_assistant_text",
        "assistant_text_sha256",
        "teacher_input_messages_sha256",
        "teacher_input_message_count",
        "endpoint_sha256",
        "request_payload_sha256",
        "response_sha256",
    }
)
_ASSISTANT_PROVENANCE_FIELDS = _ASSISTANT_PROVENANCE_REQUIRED_FIELDS | frozenset(
    {
        "api_style",
        "request_sha256",
        "response_id",
        "response_request_id",
        "returned_model",
        "usage",
        "attempts",
        "tool_schema_sha256",
        "plain_json_tool_command",
        "dynamic_system_prompt_sha256",
        "teacher_provider_input_manifest_sha256",
        "teacher_provider_media_manifest",
        "teacher_history_event_count",
        "private_retry_count",
        "teacher_allowed_tools",
        "teacher_tool_schema_sha256",
        "phase",
        "teacher_reason",
    }
)
_HOST_EVENT_FIELDS = {
    "attach_generation_evidence": frozenset(
        {
            "event_id",
            "name",
            "phase_before",
            "phase_after",
            "bundle_sha256",
            "observation",
        }
    ),
    "attach_compilation_feedback": frozenset(
        {
            "event_id",
            "name",
            "phase_before",
            "phase_after",
            "feedback_id",
            "feedback_sha256",
            "observation",
        }
    ),
}
_PRIVATE_ATTACHMENT_REQUIRED_FIELDS = frozenset(
    {"type", "path", "representation_id", "evidence_id"}
)
_PRIVATE_ATTACHMENT_OPTIONAL_FIELDS = frozenset(
    {"candidate_id", "parent_sha256", "derivation"}
)
_MEDIA_REQUIRED_FIELDS = frozenset(
    {
        "type",
        "media_id",
        "media_type",
        "representation_id",
        "mime_type",
        "size_bytes",
        "sha256",
    }
)
_MEDIA_OPTIONAL_FIELDS = frozenset(
    {"evidence_id", "candidate_id", "parent_sha256", "derivation"}
)

# A shape value of ``None`` means a JSON scalar.  Mapping/list values are
# recursively closed, while scalar types and optionality remain the runtime's
# responsibility.  This keeps the model-visible boundary fail-closed without
# duplicating the much richer semantic validation below.
_SELECTOR_SHAPE = {
    "mode": None,
    "page_index": None,
    "time_ms": None,
    "start_ms": None,
    "end_ms": None,
}
_PUBLIC_REPRESENTATION_SHAPE = {
    "representation_id": None,
    "modality": None,
    "mime_type": None,
    "size_bytes": None,
    "sha256": None,
}
_SPEAKER_BINDING_SHAPE = {
    "basis": None,
    "declared_name": None,
    "subject_id": None,
    "matching_method": None,
}
_DERIVATION_SHAPE = {
    "version": None,
    "operation": None,
    "source_representation_id": None,
}
_PERCEPTION_SHAPE = {
    "representation": _PUBLIC_REPRESENTATION_SHAPE,
    "parent_sha256": None,
    "derivation": _DERIVATION_SHAPE,
}
_REQUIREMENT_SHAPE = {
    "requirement_id": None,
    "description": None,
    "semantic_roles": [None],
    "subject_ids": [None],
    "acceptable_modalities": [None],
    "priority": None,
    "required": None,
}
_BINDING_SHAPE = {
    "binding_id": None,
    "requirement_ids": [None],
    "source_kind": None,
    "evidence_id": None,
    "representation_id": None,
    "authorization_id": None,
    "modality": None,
    "semantic_roles": [None],
    "subject_ids": [None],
    "intended_effect": None,
    "timeline_role": None,
    "rank": None,
    "priority": None,
    "required": None,
    "fallback": None,
}
_PLAN_SHAPE = {
    "plan_id": None,
    "parent_plan_id": None,
    "version": None,
    "request_id": None,
    "raw_request_sha256": None,
    "condition_budget_id": None,
    "requirements": [_REQUIREMENT_SHAPE],
    "ranked_bindings": [_BINDING_SHAPE],
    "unfulfilled_requirement_ids": [None],
    "selection_mode": None,
    "rationale": None,
    "plan_sha256": None,
    "generation_prompt": None,
    "attempt_kind": None,
}
_WRITEBACK_TEMPLATE_SHAPE = {
    "record_id": None,
    "source_kind": None,
    "source_evidence_id": None,
    "representation_id": None,
    "authorization_id": None,
    "modality": None,
    "semantic_roles": [None],
    "timeline_role": None,
    "tags": [None],
    "description": None,
    "stable_facts": [None],
    "reuse_scope": None,
    "source_selector": _SELECTOR_SHAPE,
    "writeback_evidence_ids": [None],
}
_REGISTRY_MATERIAL_SHAPE = {
    "subject_id": None,
    "tag": None,
    "category": None,
    "description": None,
    "stable_description": None,
    "aliases": [None],
    "semantic_roles": [None],
}
_UPLOADED_MATERIAL_SHAPE = {
    "evidence_id": None,
    "description": None,
    "declared_tags": [None],
    "modality": None,
    "selectors": [_SELECTOR_SHAPE],
    "one_shot": None,
}
_SEQUENCE_CONTEXT_SHAPE = {
    "source_id": None,
    "evidence_id": None,
    "modality": None,
    "description": None,
    "semantic_roles": [None],
    "subject_ids": [None],
    "available_representations": [_PUBLIC_REPRESENTATION_SHAPE],
}
_SEARCH_CANDIDATE_SHAPE = {
    "evidence_id": None,
    "source_kind": None,
    "modality": None,
    "description": None,
    "tags": [None],
    "semantic_roles": [None],
    "timeline_role": None,
    "subject_ids": [None],
    "score": None,
    "speaker_binding": _SPEAKER_BINDING_SHAPE,
    "selectors": [_SELECTOR_SHAPE],
}
_VIEW_ITEM_SHAPE = {
    "source_kind": None,
    "evidence_id": None,
    "authorization_id": None,
    "description": None,
    "modality": None,
    "semantic_roles": [None],
    "subject_ids": [None],
    "representation": _PUBLIC_REPRESENTATION_SHAPE,
    "perception_status": None,
    "selector": _SELECTOR_SHAPE,
    "speaker_binding": _SPEAKER_BINDING_SHAPE,
    "declared_tags": [None],
    "trusted_stable_facts": [None],
    "writeback_record_template": _WRITEBACK_TEMPLATE_SHAPE,
    "speaker_name": None,
    "one_shot": None,
    "perception": _PERCEPTION_SHAPE,
}
_CHECKLIST_ITEM_SHAPE = {
    "check_id": None,
    "requirement": None,
    "severity": None,
    "binding_id": None,
    "comparison_evidence_ids": [None],
    "compilation_outcome": None,
    "applied_effect": None,
    "deterministic_result": None,
    "representation_id": None,
    "representation_sha256": None,
    "selector": _SELECTOR_SHAPE,
    "speaker_binding": _SPEAKER_BINDING_SHAPE,
}
_METADATA_CONTEXT_SHAPE = {
    "description": None,
    "semantic_roles": [None],
    "trusted_stable_facts": [None],
    "reuse_scope": None,
    "one_shot": None,
    "speaker_binding": _SPEAKER_BINDING_SHAPE,
}
_BINDING_OUTCOME_SHAPE = {
    "binding_id": None,
    "outcome": None,
    "applied_effect": None,
    "reason_code": None,
}
_OBSERVATION_SHAPES: Mapping[str, Mapping[str, Any]] = {
    "view_material_registry": {
        "ok": None,
        "request_id": None,
        "phase": None,
        "snapshot_id": None,
        "registry_version": None,
        "materials": [_REGISTRY_MATERIAL_SHAPE],
        "next_cursor": None,
        "uploaded_materials": [_UPLOADED_MATERIAL_SHAPE],
        "sequence_context": [_SEQUENCE_CONTEXT_SHAPE],
        "condition_budget": {
            "budget_id": None,
            "max_total_conditions": None,
            "max_image_conditions": None,
            "max_video_conditions": None,
            "max_audio_conditions": None,
            "max_text_conditions": None,
        },
        "policy_limits": {
            "max_condition_revisions": None,
            "remaining_condition_revisions": None,
            "visual_creation_budget": None,
            "remaining_visual_creations": None,
            "max_image_reference_inputs": None,
            "perceivable_modalities": [None],
            "prompt_authoring": None,
        },
        "shot_requirement": None,
        "raw_request_sha256": None,
    },
    "search_memory_bank": {
        "ok": None,
        "search_id": None,
        "candidates": [_SEARCH_CANDIDATE_SHAPE],
    },
    "view_memories": {
        "ok": None,
        "authorization_id": None,
        "items": [_VIEW_ITEM_SHAPE],
    },
    "generate_image": {
        "ok": None,
        "candidate_id": None,
        "generation_group_id": None,
        "purpose": None,
        "representation": _PUBLIC_REPRESENTATION_SHAPE,
        "inspection_required": None,
        "remaining_attempts": None,
    },
    "inspect_candidate": {
        "ok": None,
        "inspection_id": None,
        "candidate_id": None,
        "representation_id": None,
        "checklist": [_CHECKLIST_ITEM_SHAPE],
        "requested_uses": [{"use_id": None, "use": None}],
        "comparison_evidence_ids": [None],
        "stage": None,
        "evidence_ids": [None],
        "writeback_evidence_ids": [None],
        "remaining_condition_revisions": None,
        "source_kind": None,
        "representation": _PUBLIC_REPRESENTATION_SHAPE,
        "selector": _SELECTOR_SHAPE,
        "authorization_id": None,
        "metadata_context": _METADATA_CONTEXT_SHAPE,
        "perception": _PERCEPTION_SHAPE,
    },
    "submit_candidate_assessment": {
        "ok": None,
        "assessment_id": None,
        "candidate_id": None,
        "quality": None,
        "next_action": None,
        "selected_candidate_id": None,
        "authorization_id": None,
        "phase": None,
        "remaining_condition_revisions": None,
    },
    "update_memory_bank": {
        "ok": None,
        "transaction_id": None,
        "staged_memory_ids": [None],
        "registry_update_count": None,
    },
    "submit_evidence_plan": {
        "ok": None,
        "phase": None,
        "evidence_plan": _PLAN_SHAPE,
    },
    "finish_memory_session": {
        "ok": None,
        "done": None,
        "writeback_disposition": None,
        "memory_transaction_id": None,
    },
    "attach_compilation_feedback": {
        "ok": None,
        "phase": None,
        "feedback_kind": None,
        "binding_id": None,
        "reason_code": None,
        "revision_required": None,
        "remaining_condition_revisions": None,
    },
    "attach_generation_evidence": {
        "ok": None,
        "phase": None,
        "bundle_id": None,
        "bundle_kind": None,
        "generation_id": None,
        "generation_output_sha256": None,
        "condition_signature_sha256": None,
        "review_target_evidence_id": None,
        "binding_outcomes": [_BINDING_OUTCOME_SHAPE],
        "available_evidence": [
            {"evidence_id": None, "modality": None, "writeback_candidate": None}
        ],
    },
}
_OBSERVATION_REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    name: frozenset(shape)
    for name, shape in _OBSERVATION_SHAPES.items()
}
_OBSERVATION_REQUIRED_FIELDS = {
    **_OBSERVATION_REQUIRED_FIELDS,
    "inspect_candidate": frozenset(
        {
            "ok",
            "inspection_id",
            "candidate_id",
            "representation_id",
            "checklist",
            "requested_uses",
            "comparison_evidence_ids",
            "stage",
            "evidence_ids",
            "writeback_evidence_ids",
            "remaining_condition_revisions",
        }
    ),
    "finish_memory_session": frozenset(
        {"ok", "done", "writeback_disposition"}
    ),
}


class TrajectoryValidationError(ValueError):
    """A shard cannot be trusted as a complete Memory Policy trajectory."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "")
    if _SHA256_RE.fullmatch(digest) is None:
        raise TrajectoryValidationError(f"{label} is not a canonical SHA-256 digest")
    return digest


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    *,
    label: str,
) -> None:
    actual = {str(key) for key in value}
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"unknown={extra}")
        raise TrajectoryValidationError(
            f"{label} fields are not canonical ({', '.join(detail)})"
        )


def _require_allowed_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str] | set[str],
    *,
    required: frozenset[str] | set[str] = frozenset(),
    label: str,
) -> None:
    actual = {str(key) for key in value}
    missing = set(required) - actual
    extra = actual - set(allowed)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing={sorted(missing)}")
        if extra:
            detail.append(f"unknown={sorted(extra)}")
        raise TrajectoryValidationError(
            f"{label} fields are not canonical ({', '.join(detail)})"
        )


def _assert_closed_shape(value: Any, shape: Any, *, label: str) -> None:
    """Reject objects/arrays outside a recursively closed public shape."""

    if isinstance(value, Mapping):
        if not isinstance(shape, Mapping):
            raise TrajectoryValidationError(f"{label} must be a scalar")
        unknown = {str(key) for key in value} - set(shape)
        if unknown:
            raise TrajectoryValidationError(
                f"{label} contains unknown fields: {sorted(unknown)}"
            )
        for key, child in value.items():
            _assert_closed_shape(child, shape[str(key)], label=f"{label}.{key}")
        return
    if isinstance(value, list):
        if not isinstance(shape, list) or len(shape) != 1:
            raise TrajectoryValidationError(f"{label} must not be an array")
        for index, child in enumerate(value):
            _assert_closed_shape(child, shape[0], label=f"{label}[{index}]")
        return
    if isinstance(shape, list):
        raise TrajectoryValidationError(f"{label} must be an array")
    if isinstance(shape, Mapping) and value is not None:
        raise TrajectoryValidationError(f"{label} must be an object or null")


def _validate_public_observation(
    name: str,
    observation: Any,
    *,
    envelope_key: str,
    label: str,
) -> None:
    expected_envelope = (
        "trusted_runtime_observation"
        if name in _HOST_EVENT_NAMES
        else "trusted_tool_observation"
    )
    if envelope_key != expected_envelope:
        raise TrajectoryValidationError(
            f"{label} uses the wrong trusted observation envelope"
        )
    shape = _OBSERVATION_SHAPES.get(name)
    if shape is None:
        raise TrajectoryValidationError(f"{label} names an unknown observation source")
    if not isinstance(observation, Mapping):
        raise TrajectoryValidationError(f"{label} must be an object")
    _require_allowed_fields(
        observation,
        set(shape),
        required=_OBSERVATION_REQUIRED_FIELDS[name],
        label=label,
    )
    if observation.get("ok") is not True:
        raise TrajectoryValidationError(f"{label}.ok must be true")
    _assert_closed_shape(observation, shape, label=label)
    if name == "search_memory_bank":
        candidates = observation.get("candidates")
        if not isinstance(candidates, list):
            raise TrajectoryValidationError(f"{label}.candidates must be an array")
        for index, candidate in enumerate(candidates):
            candidate_label = f"{label}.candidates[{index}]"
            if not isinstance(candidate, Mapping):
                raise TrajectoryValidationError(
                    f"{candidate_label} must be an object"
                )
            selectors_present = "selectors" in candidate
            if candidate.get("modality") != "audio":
                if selectors_present:
                    raise TrajectoryValidationError(
                        f"{candidate_label}.selectors is reserved for audio"
                    )
                continue
            selectors = candidate.get("selectors")
            if not isinstance(selectors, list) or len(selectors) != 1:
                raise TrajectoryValidationError(
                    f"{candidate_label}.selectors must contain one exact audio time_range"
                )
            selector = selectors[0]
            if not isinstance(selector, Mapping) or set(selector) != {
                "mode",
                "start_ms",
                "end_ms",
            }:
                raise TrajectoryValidationError(
                    f"{candidate_label}.selectors[0] must be an exact audio time_range"
                )
            start_ms = selector.get("start_ms")
            end_ms = selector.get("end_ms")
            if (
                selector.get("mode") != "time_range"
                or type(start_ms) is not int
                or type(end_ms) is not int
                or start_ms < 0
                or end_ms <= start_ms
            ):
                raise TrajectoryValidationError(
                    f"{candidate_label}.selectors[0] must satisfy "
                    "0 <= start_ms < end_ms"
                )
    _assert_model_safe(observation, label=label)


def _validate_private_attachment(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise TrajectoryValidationError(f"{label} must be an object")
    _require_allowed_fields(
        value,
        _PRIVATE_ATTACHMENT_REQUIRED_FIELDS | _PRIVATE_ATTACHMENT_OPTIONAL_FIELDS,
        required=_PRIVATE_ATTACHMENT_REQUIRED_FIELDS,
        label=label,
    )
    if value.get("type") not in {"image", "video", "audio", "text"}:
        raise TrajectoryValidationError(f"{label}.type is invalid")
    for key in ("path", "representation_id", "evidence_id"):
        if not isinstance(value.get(key), str) or not value.get(key):
            raise TrajectoryValidationError(f"{label}.{key} must be non-empty text")
    if "candidate_id" in value and (
        not isinstance(value.get("candidate_id"), str) or not value.get("candidate_id")
    ):
        raise TrajectoryValidationError(f"{label}.candidate_id must be non-empty text")
    has_parent = "parent_sha256" in value
    has_derivation = "derivation" in value
    if has_parent != has_derivation:
        raise TrajectoryValidationError(
            f"{label} must carry parent_sha256 and derivation together"
        )
    if has_parent:
        _require_sha256(value.get("parent_sha256"), f"{label}.parent_sha256")
        derivation = value.get("derivation")
        if not isinstance(derivation, Mapping):
            raise TrajectoryValidationError(f"{label}.derivation must be an object")
        _require_exact_fields(
            derivation, frozenset(_DERIVATION_SHAPE), label=f"{label}.derivation"
        )
        if (
            derivation.get("version") != "storymem.visual-only-perception.v1"
            or derivation.get("operation")
            != "strip_audio_for_visual_perception"
            or not isinstance(derivation.get("source_representation_id"), str)
            or not derivation.get("source_representation_id")
        ):
            raise TrajectoryValidationError(f"{label}.derivation is invalid")


def _validate_media_row(value: Any, *, kind: str, label: str) -> None:
    if not isinstance(value, Mapping):
        raise TrajectoryValidationError(f"{label} must be an object")
    required = set(_MEDIA_REQUIRED_FIELDS)
    allowed = set(_MEDIA_REQUIRED_FIELDS | _MEDIA_OPTIONAL_FIELDS)
    if kind == "manifest":
        required.update({"message_index", "content_index"})
        allowed.update({"message_index", "content_index"})
    elif kind == "private":
        required.update({"event_id", "path"})
        allowed.update({"event_id", "path"})
    elif kind != "block":
        raise TrajectoryValidationError(f"unknown media row kind: {kind}")
    _require_allowed_fields(value, allowed, required=required, label=label)
    if value.get("type") != "media":
        raise TrajectoryValidationError(f"{label}.type must be 'media'")
    for key in ("media_id", "media_type", "representation_id", "mime_type"):
        if not isinstance(value.get(key), str) or not value.get(key):
            raise TrajectoryValidationError(f"{label}.{key} must be non-empty text")
    if value.get("media_type") not in {"image", "video", "audio", "text"}:
        raise TrajectoryValidationError(f"{label}.media_type is invalid")
    if type(value.get("size_bytes")) is not int or value["size_bytes"] < 1:
        raise TrajectoryValidationError(f"{label}.size_bytes is invalid")
    _require_sha256(value.get("sha256"), f"{label}.sha256")
    if kind == "manifest":
        for key in ("message_index", "content_index"):
            if type(value.get(key)) is not int or value[key] < 0:
                raise TrajectoryValidationError(f"{label}.{key} is invalid")
    if kind == "private":
        for key in ("event_id", "path"):
            if not isinstance(value.get(key), str) or not value.get(key):
                raise TrajectoryValidationError(f"{label}.{key} must be non-empty text")
    has_parent = "parent_sha256" in value
    has_derivation = "derivation" in value
    if has_parent != has_derivation:
        raise TrajectoryValidationError(
            f"{label} must carry parent_sha256 and derivation together"
        )
    if has_parent:
        _require_sha256(value.get("parent_sha256"), f"{label}.parent_sha256")
        derivation = value.get("derivation")
        if not isinstance(derivation, Mapping):
            raise TrajectoryValidationError(f"{label}.derivation must be an object")
        _require_exact_fields(
            derivation, frozenset(_DERIVATION_SHAPE), label=f"{label}.derivation"
        )
        if (
            derivation.get("version") != "storymem.visual-only-perception.v1"
            or derivation.get("operation")
            != "strip_audio_for_visual_perception"
            or not isinstance(derivation.get("source_representation_id"), str)
            or not derivation.get("source_representation_id")
        ):
            raise TrajectoryValidationError(f"{label}.derivation is invalid")


def parse_plain_json_tool_command(
    text: str, *, label: str = "closed-Teacher assistant text"
) -> tuple[dict[str, Any], str]:
    """Parse the exact plain-JSON action used during closed-model collection."""

    if not isinstance(text, str):
        raise TrajectoryValidationError(f"{label} must be text")
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrajectoryValidationError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"reason", "tool_call"}:
        raise TrajectoryValidationError(
            f"{label} must contain exactly reason and tool_call"
        )
    reason = value["reason"]
    call = value["tool_call"]
    if not isinstance(reason, str):
        raise TrajectoryValidationError(f"{label}.reason must be a string")
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise TrajectoryValidationError(
            f"{label}.tool_call must contain exactly name and arguments"
        )
    if call.get("name") not in CANONICAL_TOOL_NAMES:
        raise TrajectoryValidationError(f"{label} names a non-canonical tool")
    if not isinstance(call.get("arguments"), dict):
        raise TrajectoryValidationError(f"{label}.tool_call.arguments must be an object")
    return copy.deepcopy(call), reason


def _assert_model_safe(value: Any, *, label: str) -> None:
    """Reject private/backend keys from protocol-owned structured values."""

    if isinstance(value, Mapping):
        leaked = {
            str(key)
            for key in value
            if re.sub(r"[^a-z0-9]", "", str(key).casefold())
            in _NORMALIZED_FORBIDDEN_MODEL_KEYS
        }
        if leaked:
            raise TrajectoryValidationError(
                f"{label} leaks private/backend keys: {sorted(leaked)}"
            )
        if "representation" in value:
            try:
                validate_public_representation(value["representation"])
            except ValueError as exc:
                raise TrajectoryValidationError(f"{label}: {exc}") from exc
        if "derivation" in value:
            derivation = value.get("derivation")
            if (
                not isinstance(derivation, Mapping)
                or set(derivation) != set(_DERIVATION_SHAPE)
                or derivation.get("version")
                != "storymem.visual-only-perception.v1"
                or derivation.get("operation")
                != "strip_audio_for_visual_perception"
                or not isinstance(derivation.get("source_representation_id"), str)
                or not derivation.get("source_representation_id")
            ):
                raise TrajectoryValidationError(
                    f"{label} contains invalid public derivation metadata"
                )
        if "reason_code" in value:
            try:
                validate_neutral_reason_code(value["reason_code"])
            except ValueError as exc:
                raise TrajectoryValidationError(f"{label}: {exc}") from exc
        for key in ("applied_effect", "intended_effect"):
            if value.get(key) is not None:
                try:
                    validate_neutral_semantic_effect(value[key])
                except ValueError as exc:
                    raise TrajectoryValidationError(f"{label}: {exc}") from exc
        for child in value.values():
            _assert_model_safe(child, label=label)
        return
    if isinstance(value, list):
        for child in value:
            _assert_model_safe(child, label=label)


def _validate_tool_call(call: Any, *, label: str, prompt_authoring: bool = False) -> dict[str, Any]:
    if not isinstance(call, Mapping) or set(call) != {"name", "arguments"}:
        raise TrajectoryValidationError(f"{label} must contain exactly name and arguments")
    name = call.get("name")
    arguments = call.get("arguments")
    if name not in CANONICAL_TOOL_NAMES or not isinstance(arguments, Mapping):
        raise TrajectoryValidationError(f"{label} is not a canonical tool call")
    schema_by_name = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in get_tool_schemas(prompt_authoring=prompt_authoring)
    }
    errors = sorted(
        Draft202012Validator(schema_by_name[str(name)]).iter_errors(arguments),
        key=lambda error: list(error.path),
    )
    if errors:
        location = ".".join(str(item) for item in errors[0].path) or "<root>"
        raise TrajectoryValidationError(
            f"{label} arguments are invalid at {location}: {errors[0].message}"
        )
    return {"name": str(name), "arguments": copy.deepcopy(dict(arguments))}


def fsm_next_phases(kind: str, name: str, phase_before: str) -> frozenset[str]:
    """Return the sole canonical phase relation shared by trajectory and SFT."""

    phases = _FSM_TRANSITIONS.get((kind, name, phase_before))
    if phases is None:
        raise TrajectoryValidationError(
            f"illegal FSM action {kind}:{name} in phase {phase_before!r}"
        )
    return phases


def _validate_fsm_transition(
    kind: str, name: str, phase_before: str, phase_after: str
) -> None:
    allowed = fsm_next_phases(kind, name, phase_before)
    if phase_after not in allowed:
        raise TrajectoryValidationError(
            f"illegal FSM transition for {kind}:{name}: "
            f"{phase_before!r} -> {phase_after!r}"
        )


def _merged_events(
    events: Sequence[Mapping[str, Any]],
    host_events: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    """Merge the two append-only ledgers by their explicit FSM transitions."""

    tools = [copy.deepcopy(dict(row)) for row in events]
    hosts = [copy.deepcopy(dict(row)) for row in host_events]
    tool_index = host_index = 0
    phase = "need_registry"
    merged: list[tuple[str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    while tool_index < len(tools) or host_index < len(hosts):
        candidates: list[tuple[str, dict[str, Any]]] = []
        if tool_index < len(tools) and tools[tool_index].get("phase_before") == phase:
            candidates.append(("tool", tools[tool_index]))
        if host_index < len(hosts) and hosts[host_index].get("phase_before") == phase:
            candidates.append(("host", hosts[host_index]))
        if len(candidates) != 1:
            raise TrajectoryValidationError(
                f"event ledgers cannot be merged unambiguously from phase {phase!r}"
            )
        kind, event = candidates[0]
        event_id = str(event.get("event_id") or "")
        if not event_id or event_id in seen_ids:
            raise TrajectoryValidationError("event IDs must be non-empty and globally unique")
        seen_ids.add(event_id)
        phase_after = str(event.get("phase_after") or "")
        if not phase_after:
            raise TrajectoryValidationError(f"event {event_id} has no phase_after")
        name = (
            str((event.get("tool_call") or {}).get("name") or "")
            if kind == "tool"
            else str(event.get("name") or "")
        )
        _validate_fsm_transition(kind, name, phase, phase_after)
        merged.append((kind, event))
        phase = phase_after
        if kind == "tool":
            tool_index += 1
        else:
            host_index += 1
    if require_complete and phase != "complete":
        raise TrajectoryValidationError(
            f"trajectory terminates in phase {phase!r}, not 'complete'"
        )
    return merged


def _attachment_media(
    event: Mapping[str, Any], attachment_index: int, attachment: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"event {event.get('event_id')} media attachment {attachment_index}"
    _validate_private_attachment(attachment, label=label)
    path = Path(str(attachment.get("path") or "")).expanduser().resolve()
    if not path.is_file() or path.stat().st_size < 1:
        raise TrajectoryValidationError(f"private media attachment is missing: {path}")
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    media_type = str(attachment.get("type") or mime_type.split("/", 1)[0])
    representation_id = str(attachment.get("representation_id") or "")
    if not media_type or not representation_id:
        raise TrajectoryValidationError(
            "private media attachment needs type and representation_id"
        )
    event_id = str(event["event_id"])
    media_id = "media_" + hashlib.sha256(
        f"{event_id}\0{attachment_index}\0{representation_id}\0{digest}".encode("utf-8")
    ).hexdigest()[:24]
    public = {
        "type": "media",
        "media_id": media_id,
        "media_type": media_type,
        "representation_id": representation_id,
        "mime_type": mime_type,
        "size_bytes": len(payload),
        "sha256": digest,
    }
    for key in ("evidence_id", "candidate_id"):
        if attachment.get(key) is not None:
            public[key] = str(attachment[key])
    if attachment.get("parent_sha256") is not None:
        public["parent_sha256"] = _require_sha256(
            attachment.get("parent_sha256"), "perception proxy parent_sha256"
        )
    if attachment.get("derivation") is not None:
        derivation = attachment.get("derivation")
        if (
            not isinstance(derivation, Mapping)
            or set(derivation) != set(_DERIVATION_SHAPE)
            or derivation.get("version")
            != "storymem.visual-only-perception.v1"
            or derivation.get("operation") != "strip_audio_for_visual_perception"
            or not isinstance(derivation.get("source_representation_id"), str)
            or not derivation.get("source_representation_id")
        ):
            raise TrajectoryValidationError(
                "visual-only perception proxy has invalid derivation metadata"
            )
        public["derivation"] = copy.deepcopy(dict(derivation))
    private = {
        **copy.deepcopy(public),
        "event_id": event_id,
        "path": str(path),
    }
    return public, private


def _prepare_media(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_event: dict[str, list[dict[str, Any]]] = {}
    private_manifest: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("event_id") or "")
        attachments = event.get("media_attachments_private") or []
        if not isinstance(attachments, list):
            raise TrajectoryValidationError(
                f"event {event_id} media_attachments_private must be an array"
            )
        blocks: list[dict[str, Any]] = []
        for index, attachment in enumerate(attachments):
            if not isinstance(attachment, Mapping):
                raise TrajectoryValidationError(f"event {event_id} has malformed media")
            public, private = _attachment_media(event, index, attachment)
            blocks.append(public)
            private_manifest.append(private)
        by_event[event_id] = blocks
    return by_event, private_manifest


def _observation_content(
    envelope_key: str,
    name: str,
    observation: Mapping[str, Any],
    media: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    _validate_public_observation(
        name,
        observation,
        envelope_key=envelope_key,
        label=f"{name} observation",
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _canonical_json(
                {envelope_key: {"name": name, "result": copy.deepcopy(dict(observation))}}
            ),
        }
    ]
    content.extend(copy.deepcopy(dict(item)) for item in media)
    return content


def _append_closed_teacher_observation(
    messages: list[dict[str, Any]], content: Sequence[Mapping[str, Any]]
) -> None:
    """Append one environment observation to an ordinary chat transcript.

    The closed Teacher transport has no native ``tool`` role, so both tool
    results and host/runtime events are represented as USER content.  A host
    event may immediately follow a tool result without an intervening model
    turn.  In that case both observations belong to the same USER turn: keep
    their independently serialized blocks (and any media) in ledger order
    while satisfying providers that require strict USER/ASSISTANT alternation.

    The raw request is a string, whereas environment turns are content lists;
    consequently this helper cannot accidentally fold a host event into the
    original request.
    """

    blocks = [copy.deepcopy(dict(block)) for block in content]
    if not blocks:
        raise TrajectoryValidationError("Teacher observation content is empty")
    if messages and messages[-1].get("role") == "user":
        previous = messages[-1].get("content")
        if not isinstance(previous, list):
            raise TrajectoryValidationError(
                "a Teacher environment observation cannot precede the first assistant turn"
            )
        previous.extend(blocks)
        return
    messages.append({"role": "user", "content": blocks})


def _teacher_provenance(event: Mapping[str, Any], expected_history: Sequence[Any], *, prompt_authoring: bool = False) -> None:
    event_id = str(event.get("event_id") or "")
    provenance = event.get("assistant_provenance")
    if not isinstance(provenance, Mapping):
        raise TrajectoryValidationError(f"event {event_id} has no Teacher provenance")
    _require_allowed_fields(
        provenance,
        _ASSISTANT_PROVENANCE_FIELDS,
        required=_ASSISTANT_PROVENANCE_REQUIRED_FIELDS,
        label=f"event {event_id} Teacher provenance",
    )
    if provenance.get("assistant_source") != "remote_closed_model_api":
        raise TrajectoryValidationError(f"event {event_id} is not from a closed-model API")
    if provenance.get("teacher_transport") != "plain_json_tool_dialogue":
        raise TrajectoryValidationError(f"event {event_id} used the wrong Teacher transport")
    model = provenance.get("model")
    if not isinstance(model, str) or not model.strip():
        raise TrajectoryValidationError(f"event {event_id} has no Teacher model identity")
    raw_text = provenance.get("raw_assistant_text")
    parsed_call, parsed_reason = parse_plain_json_tool_command(
        raw_text, label=f"event {event_id} raw assistant text"
    )
    expected_call = _validate_tool_call(event.get("tool_call"), label=f"event {event_id}", prompt_authoring=prompt_authoring)
    if parsed_call != expected_call:
        raise TrajectoryValidationError(
            f"event {event_id} raw assistant text differs from the committed call"
        )
    if provenance.get("assistant_text_sha256") != _sha256_text(str(raw_text)):
        raise TrajectoryValidationError(f"event {event_id} assistant text hash mismatch")
    if provenance.get("teacher_input_messages_sha256") != _sha256_json(expected_history):
        raise TrajectoryValidationError(f"event {event_id} Teacher history hash mismatch")
    if provenance.get("teacher_input_message_count") != len(expected_history):
        raise TrajectoryValidationError(f"event {event_id} Teacher history count mismatch")
    for key in ("endpoint_sha256", "request_payload_sha256", "response_sha256"):
        _require_sha256(provenance.get(key), f"event {event_id} {key}")
    for key in (
        "request_sha256",
        "tool_schema_sha256",
        "dynamic_system_prompt_sha256",
        "teacher_provider_input_manifest_sha256",
        "teacher_tool_schema_sha256",
    ):
        if key in provenance:
            _require_sha256(provenance.get(key), f"event {event_id} {key}")
    if "teacher_reason" in provenance and provenance.get("teacher_reason") != parsed_reason:
        raise TrajectoryValidationError(f"event {event_id} Teacher reason mismatch")
    if "teacher_allowed_tools" in provenance:
        allowed = provenance.get("teacher_allowed_tools")
        if (
            not isinstance(allowed, list)
            or not all(isinstance(item, str) for item in allowed)
            or list(dict.fromkeys(allowed)) != allowed
            or any(item not in CANONICAL_TOOL_NAMES for item in allowed)
            or expected_call["name"] not in allowed
        ):
            raise TrajectoryValidationError(
                f"event {event_id} has an invalid Teacher action set"
            )


def build_teacher_input_messages(
    *,
    raw_user_request: str,
    events: Sequence[Mapping[str, Any]],
    host_events: Sequence[Mapping[str, Any]] = (),
    stop_before_event_id: str | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct the ordinary multimodal dialogue seen by the closed teacher."""

    media_by_event, _ = _prepare_media(events)
    messages: list[dict[str, Any]] = [{"role": "user", "content": raw_user_request}]
    for kind, event in _merged_events(events, host_events, require_complete=False):
        if event["event_id"] == stop_before_event_id:
            break
        if kind == "host":
            if event.get("name") not in _HOST_EVENT_NAMES:
                raise TrajectoryValidationError("unknown host event")
            _append_closed_teacher_observation(
                messages,
                _observation_content(
                    "trusted_runtime_observation",
                    str(event["name"]),
                    event.get("observation") or {},
                ),
            )
            continue
        provenance = event.get("assistant_provenance") or {}
        raw_text = provenance.get("raw_assistant_text")
        parse_plain_json_tool_command(
            raw_text, label=f"event {event['event_id']} raw assistant text"
        )
        messages.append({"role": "assistant", "content": raw_text})
        _append_closed_teacher_observation(
            messages,
            _observation_content(
                "trusted_tool_observation",
                str(event["tool_call"]["name"]),
                event.get("observation") or {},
                media_by_event.get(str(event["event_id"]), ()),
            ),
        )
    return messages


def build_training_messages(
    *,
    raw_user_request: str,
    events: Sequence[Mapping[str, Any]],
    host_events: Sequence[Mapping[str, Any]] = (),
    prompt_authoring: bool = False,
) -> list[dict[str, Any]]:
    """Project verified plain-JSON Teacher actions to native function calls."""

    media_by_event, _ = _prepare_media(events)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt_for_mode(prompt_authoring)},
        {"role": "user", "content": raw_user_request},
    ]
    for kind, event in _merged_events(events, host_events):
        event_id = str(event["event_id"])
        if kind == "host":
            if event.get("name") not in _HOST_EVENT_NAMES:
                raise TrajectoryValidationError("unknown host event")
            messages.append(
                {
                    "role": "user",
                    "content": _observation_content(
                        "trusted_runtime_observation",
                        str(event["name"]),
                        event.get("observation") or {},
                    ),
                }
            )
            continue
        call = _validate_tool_call(event.get("tool_call"), label=f"event {event_id}", prompt_authoring=prompt_authoring)
        raw_text = (event.get("assistant_provenance") or {}).get("raw_assistant_text")
        parsed_call, _ = parse_plain_json_tool_command(
            raw_text, label=f"event {event_id} raw assistant text"
        )
        if parsed_call != call:
            raise TrajectoryValidationError(
                f"event {event_id} raw assistant text differs from the committed call"
            )
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": event_id,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": _canonical_json(call["arguments"]),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": event_id,
                "name": call["name"],
                "content": _observation_content(
                    "trusted_tool_observation",
                    call["name"],
                    event.get("observation") or {},
                    media_by_event.get(event_id, ()),
                ),
            }
        )
    # The first two messages are trusted prompt/request text.  In particular,
    # an arbitrary user request may itself contain JSON-looking examples with
    # keys such as "path"; that is user data, not an infrastructure leak.
    _assert_model_safe(messages[2:], label="training messages")
    return messages


def trajectory_media_manifest(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the public, pathless media references present in messages."""

    result: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, block in enumerate(content):
            if not isinstance(block, Mapping) or block.get("type") != "media":
                continue
            if "path" in block:
                raise TrajectoryValidationError("model-visible media block contains a path")
            result.append(
                {
                    "message_index": message_index,
                    "content_index": content_index,
                    **copy.deepcopy(dict(block)),
                }
            )
    return result


def _host_bundle_by_id(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bundles = list(state.get("generation_bundle_history_private") or [])
    current = state.get("generation_bundle_private")
    if isinstance(current, Mapping):
        bundles.append(current)
    result: dict[str, dict[str, Any]] = {}
    for row in bundles:
        if not isinstance(row, Mapping):
            raise TrajectoryValidationError("private generation bundle is malformed")
        bundle_id = str(row.get("bundle_id") or "")
        if not bundle_id or bundle_id in result:
            raise TrajectoryValidationError("private generation bundle IDs must be unique")
        result[bundle_id] = copy.deepcopy(dict(row))
    return result


def build_trajectory_shard(
    *,
    runtime_state: Mapping[str, Any] | Any,
    assistant_provenance_by_event_id: Mapping[str, Mapping[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Seal one completed runtime state as an auditable training shard."""

    raw_state = getattr(runtime_state, "data", runtime_state)
    if not isinstance(raw_state, Mapping):
        raise TrajectoryValidationError("runtime_state must be a mapping or state object")
    state = copy.deepcopy(dict(raw_state))
    events = copy.deepcopy(list(state.get("events") or []))
    provenance_map = assistant_provenance_by_event_id or {}
    for event in events:
        event_id = str(event.get("event_id") or "")
        supplied = provenance_map.get(event_id)
        if supplied is not None:
            if event.get("assistant_provenance") not in (None, supplied):
                raise TrajectoryValidationError(f"event {event_id} has conflicting provenance")
            event["assistant_provenance"] = copy.deepcopy(dict(supplied))
    host_events = copy.deepcopy(list(state.get("host_events") or []))
    raw_request = str(state.get("shot_requirement") or "")
    raw_hash = str(state.get("shot_requirement_sha256") or "")
    if not raw_request or raw_hash != _sha256_text(raw_request):
        raise TrajectoryValidationError("runtime state raw request/hash mismatch")
    _, private_media = _prepare_media(events)
    prompt_authoring = (state.get("policy_limits") or {}).get("prompt_authoring") is True
    messages = build_training_messages(
        raw_user_request=raw_request,
        events=events,
        host_events=host_events,
        prompt_authoring=prompt_authoring,
    )
    private_bundles = list(_host_bundle_by_id(state).values())
    private_reservations = copy.deepcopy(
        list((state.get("generation_reservations") or {}).values())
    )
    private_compilation_feedback = copy.deepcopy(
        list(state.get("compilation_feedback_history_private") or [])
    )
    plans = [
        *copy.deepcopy(list(state.get("plan_history") or [])),
        copy.deepcopy(state.get("accepted_evidence_plan")),
    ]
    if not plans[-1]:
        raise TrajectoryValidationError("runtime state has no accepted EvidencePlan")
    shard = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "tool_protocol_version": TOOL_PROTOCOL_VERSION,
        "session_id": str(state.get("session_id") or ""),
        "request_id": str(state.get("request_id") or ""),
        "raw_user_request": raw_request,
        "raw_request_sha256": raw_hash,
        "condition_budget": copy.deepcopy(state.get("condition_budget")),
        "condition_budget_sha256": _sha256_json(state.get("condition_budget")),
        "policy_limits": copy.deepcopy(state.get("policy_limits")),
        "policy_limits_sha256": _sha256_json(state.get("policy_limits")),
        "system_prompt_sha256": _sha256_text(system_prompt_for_mode(prompt_authoring)),
        "tool_schema_sha256": tool_schema_hash(prompt_authoring=prompt_authoring),
        "tools": get_tool_schemas(prompt_authoring=prompt_authoring),
        "events": events,
        "host_events": host_events,
        "plans": plans,
        "messages": messages,
        "messages_sha256": _sha256_json(messages),
        "media_manifest": trajectory_media_manifest(messages),
        "private_media_manifest": private_media,
        "private_generation_bundles": private_bundles,
        "private_generation_reservations": private_reservations,
        "private_compilation_feedback": private_compilation_feedback,
        "final_phase": str(state.get("phase") or ""),
    }
    shard["media_manifest_sha256"] = _sha256_json(shard["media_manifest"])
    if created_at is not None:
        shard["created_at"] = str(created_at)
    validate_trajectory_shard(shard)
    return shard


def validate_trajectory_shard(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate transport, causality, hashes, terminal state, and privacy."""

    if not isinstance(value, Mapping):
        raise TrajectoryValidationError("trajectory shard must be an object")
    _require_allowed_fields(
        value,
        _TRAJECTORY_FIELDS | {"created_at"},
        required=_TRAJECTORY_FIELDS,
        label="trajectory",
    )
    if "created_at" in value and not isinstance(value.get("created_at"), str):
        raise TrajectoryValidationError("trajectory created_at must be text")
    if value.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise TrajectoryValidationError("trajectory schema version is incompatible")
    if value.get("runtime_version") != RUNTIME_VERSION:
        raise TrajectoryValidationError("memory-policy runtime version is incompatible")
    raw_request = value.get("raw_user_request")
    if not isinstance(raw_request, str) or not raw_request:
        raise TrajectoryValidationError("trajectory has no raw user request")
    raw_hash = _require_sha256(value.get("raw_request_sha256"), "raw_request_sha256")
    if raw_hash != _sha256_text(raw_request):
        raise TrajectoryValidationError("raw request hash mismatch")
    if value.get("tool_protocol_version") != TOOL_PROTOCOL_VERSION:
        raise TrajectoryValidationError("tool protocol version mismatch")
    prompt_authoring = (value.get("policy_limits") or {}).get("prompt_authoring") is True
    if value.get("tools") != get_tool_schemas(prompt_authoring=prompt_authoring):
        raise TrajectoryValidationError("trajectory tool schemas are not canonical")
    if value.get("tool_schema_sha256") != tool_schema_hash(prompt_authoring=prompt_authoring):
        raise TrajectoryValidationError("trajectory tool schema hash mismatch")
    if value.get("system_prompt_sha256") != _sha256_text(system_prompt_for_mode(prompt_authoring)):
        raise TrajectoryValidationError("open-model system prompt hash mismatch")
    condition_budget = value.get("condition_budget")
    if not isinstance(condition_budget, Mapping):
        raise TrajectoryValidationError("trajectory has no condition budget")
    if value.get("condition_budget_sha256") != _sha256_json(condition_budget):
        raise TrajectoryValidationError("condition budget hash mismatch")
    policy_limits = value.get("policy_limits")
    if not isinstance(policy_limits, Mapping):
        raise TrajectoryValidationError("trajectory has no policy limits")
    if value.get("policy_limits_sha256") != _sha256_json(policy_limits):
        raise TrajectoryValidationError("policy limits hash mismatch")

    events = value.get("events")
    host_events = value.get("host_events")
    if not isinstance(events, list) or not isinstance(host_events, list):
        raise TrajectoryValidationError("trajectory event ledgers must be arrays")
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise TrajectoryValidationError(
                f"trajectory tool event {index} must be an object"
            )
        _require_exact_fields(
            event, _TOOL_EVENT_FIELDS, label=f"trajectory tool event {index}"
        )
        _validate_tool_call(
            event.get("tool_call"), label=f"trajectory tool event {index}",
            prompt_authoring=prompt_authoring,
        )
        if not isinstance(event.get("phase_before"), str) or not isinstance(
            event.get("phase_after"), str
        ):
            raise TrajectoryValidationError(
                f"trajectory tool event {index} has invalid phases"
            )
        attachments = event.get("media_attachments_private")
        if not isinstance(attachments, list):
            raise TrajectoryValidationError(
                f"trajectory tool event {index} media attachments must be an array"
            )
        for attachment_index, attachment in enumerate(attachments):
            _validate_private_attachment(
                attachment,
                label=(
                    f"trajectory tool event {index} media attachment "
                    f"{attachment_index}"
                ),
            )
        provenance = event.get("assistant_provenance")
        if not isinstance(provenance, Mapping):
            raise TrajectoryValidationError(
                f"trajectory tool event {index} has no Teacher provenance"
            )
        _require_allowed_fields(
            provenance,
            _ASSISTANT_PROVENANCE_FIELDS,
            required=_ASSISTANT_PROVENANCE_REQUIRED_FIELDS,
            label=f"trajectory tool event {index} Teacher provenance",
        )
    for index, event in enumerate(host_events):
        if not isinstance(event, Mapping):
            raise TrajectoryValidationError(
                f"trajectory host event {index} must be an object"
            )
        name = str(event.get("name") or "")
        expected_fields = _HOST_EVENT_FIELDS.get(name)
        if expected_fields is None:
            raise TrajectoryValidationError(
                f"trajectory host event {index} has an unknown name"
            )
        _require_exact_fields(
            event, expected_fields, label=f"trajectory host event {index}"
        )
        if not isinstance(event.get("phase_before"), str) or not isinstance(
            event.get("phase_after"), str
        ):
            raise TrajectoryValidationError(
                f"trajectory host event {index} has invalid phases"
            )
        digest_field = (
            "bundle_sha256"
            if name == "attach_generation_evidence"
            else "feedback_sha256"
        )
        _require_sha256(
            event.get(digest_field), f"trajectory host event {index} {digest_field}"
        )
    merged = _merged_events(events, host_events)
    if not merged or merged[0][0] != "tool":
        raise TrajectoryValidationError("trajectory has no initial tool action")
    first_call = _validate_tool_call(merged[0][1].get("tool_call"), label="first event", prompt_authoring=prompt_authoring)
    if first_call["name"] != "view_material_registry":
        raise TrajectoryValidationError("view_material_registry must be the first action")

    teacher_history: list[dict[str, Any]] = [{"role": "user", "content": raw_request}]
    media_by_event, expected_private_media = _prepare_media(events)
    last_tool_name = ""
    last_tool_observation: Mapping[str, Any] = {}
    for kind, event in merged:
        if kind == "host":
            if event.get("name") not in _HOST_EVENT_NAMES:
                raise TrajectoryValidationError("trajectory contains an unknown host event")
            observation = event.get("observation")
            if not isinstance(observation, Mapping):
                raise TrajectoryValidationError("host event has no public observation")
            _assert_model_safe(observation, label="host observation")
            _append_closed_teacher_observation(
                teacher_history,
                _observation_content(
                    "trusted_runtime_observation",
                    str(event["name"]),
                    observation,
                ),
            )
            continue
        event_id = str(event["event_id"])
        _teacher_provenance(event, teacher_history, prompt_authoring=prompt_authoring)
        call = _validate_tool_call(event.get("tool_call"), label=f"event {event_id}", prompt_authoring=prompt_authoring)
        observation = event.get("observation")
        if not isinstance(observation, Mapping):
            raise TrajectoryValidationError(f"event {event_id} has no observation")
        _assert_model_safe(observation, label=f"event {event_id} observation")
        if call["name"] == "view_material_registry" and (
            observation.get("request_id") != value.get("request_id")
            or observation.get("raw_request_sha256") != raw_hash
            or observation.get("shot_requirement") != raw_request
        ):
            raise TrajectoryValidationError(
                "registry observation does not match the trajectory request"
            )
        raw_text = event["assistant_provenance"]["raw_assistant_text"]
        teacher_history.append({"role": "assistant", "content": raw_text})
        _append_closed_teacher_observation(
            teacher_history,
            _observation_content(
                "trusted_tool_observation",
                call["name"],
                observation,
                media_by_event.get(event_id, ()),
            ),
        )
        last_tool_name = call["name"]
        last_tool_observation = observation
    if last_tool_name != "finish_memory_session" or last_tool_observation.get("done") is not True:
        raise TrajectoryValidationError(
            "trajectory must end with finish_memory_session(done=true)"
        )
    if value.get("final_phase") != "complete":
        raise TrajectoryValidationError("trajectory final phase is not complete")

    plans = value.get("plans")
    if not isinstance(plans, list) or not plans:
        raise TrajectoryValidationError("trajectory contains no EvidencePlan")
    previous: Mapping[str, Any] | None = None
    for index, plan in enumerate(plans):
        if not isinstance(plan, Mapping):
            raise TrajectoryValidationError("trajectory plan is malformed")
        if plan.get("request_id") != value.get("request_id"):
            raise TrajectoryValidationError("EvidencePlan request ID mismatch")
        if plan.get("raw_request_sha256") != raw_hash:
            raise TrajectoryValidationError("EvidencePlan raw request hash mismatch")
        if plan.get("condition_budget_id") != condition_budget.get("budget_id"):
            raise TrajectoryValidationError("EvidencePlan condition budget mismatch")
        if plan.get("version") != index + 1:
            raise TrajectoryValidationError("EvidencePlan versions must be contiguous")
        if previous is None:
            if plan.get("parent_plan_id") is not None:
                raise TrajectoryValidationError("initial EvidencePlan has a parent")
        else:
            if plan.get("parent_plan_id") != previous.get("plan_id"):
                raise TrajectoryValidationError("EvidencePlan parent chain is broken")
            for key in ("request_id", "raw_request_sha256", "condition_budget_id", "requirements"):
                if plan.get(key) != previous.get(key):
                    raise TrajectoryValidationError(
                        f"EvidencePlan revision changed immutable {key}"
                    )
        expected_plan_hash = _sha256_json(
            {key: copy.deepcopy(child) for key, child in plan.items() if key != "plan_sha256"}
        )
        if plan.get("plan_sha256") != expected_plan_hash:
            raise TrajectoryValidationError("EvidencePlan hash mismatch")
        previous = plan
    max_revisions = int(policy_limits.get("max_condition_revisions", 0))
    if len(plans) - 1 > max_revisions:
        raise TrajectoryValidationError("EvidencePlan revision budget exceeded")

    revision_decisions = [
        event["tool_call"]["arguments"]
        for event in events
        if (event.get("tool_call") or {}).get("name")
        == "submit_candidate_assessment"
        and (event.get("tool_call") or {}).get("arguments", {}).get("decision")
        == "revise_evidence_plan"
    ]
    compilation_revision_events = [
        event
        for event in host_events
        if event.get("name") == "attach_compilation_feedback"
    ]
    if len(revision_decisions) + len(compilation_revision_events) != len(plans) - 1:
        raise TrajectoryValidationError(
            "EvidencePlan revisions do not match condition reviews or compilation feedback"
        )
    if any(row.get("assessment_scope") != "condition_adherence" for row in revision_decisions):
        raise TrajectoryValidationError(
            "EvidencePlan revision was not caused by condition adherence"
        )

    plan_by_id = {str(plan.get("plan_id") or ""): plan for plan in plans}
    reservations = value.get("private_generation_reservations")
    if not isinstance(reservations, list):
        raise TrajectoryValidationError(
            "private generation reservations must be an array"
        )
    reservation_by_id: dict[str, Mapping[str, Any]] = {}
    reserved_signatures: set[str] = set()
    reserved_generation_ids: set[str] = set()
    for reservation in reservations:
        if not isinstance(reservation, Mapping):
            raise TrajectoryValidationError("generation reservation is malformed")
        reservation_id = str(reservation.get("generation_reservation_id") or "")
        generation_id = str(reservation.get("generation_id") or "")
        if not reservation_id or reservation_id in reservation_by_id:
            raise TrajectoryValidationError("generation reservation IDs must be unique")
        if not generation_id or generation_id in reserved_generation_ids:
            raise TrajectoryValidationError("reserved generation IDs must be unique")
        if reservation.get("status") != "completed":
            raise TrajectoryValidationError(
                "a completed trajectory contains an unfinished generation reservation"
            )
        if reservation.get("request_id") != value.get("request_id") or reservation.get(
            "raw_request_sha256"
        ) != raw_hash:
            raise TrajectoryValidationError("generation reservation request mismatch")
        plan = plan_by_id.get(str(reservation.get("plan_id") or ""))
        if plan is None or reservation.get("plan_sha256") != plan.get("plan_sha256"):
            raise TrajectoryValidationError("generation reservation plan mismatch")
        canonical_conditions = reservation.get("canonical_conditions")
        private_conditions = reservation.get("compiled_conditions_private")
        if not isinstance(canonical_conditions, list) or not isinstance(
            private_conditions, list
        ) or len(canonical_conditions) != len(private_conditions):
            raise TrajectoryValidationError(
                "generation reservation condition manifests are malformed"
            )
        for index, (canonical, private) in enumerate(
            zip(canonical_conditions, private_conditions), start=1
        ):
            if not isinstance(canonical, Mapping) or set(canonical) != {
                "ordinal",
                "modality",
                "content_sha256",
                "applied_effect",
            }:
                raise TrajectoryValidationError(
                    "canonical compiled condition has non-canonical fields"
                )
            if canonical.get("ordinal") != index:
                raise TrajectoryValidationError(
                    "canonical compiled condition order is invalid"
                )
            _require_sha256(
                canonical.get("content_sha256"), "compiled condition content_sha256"
            )
            try:
                validate_neutral_semantic_effect(canonical.get("applied_effect"))
            except ValueError as exc:
                raise TrajectoryValidationError(str(exc)) from exc
            if not isinstance(private, Mapping):
                raise TrajectoryValidationError("private compiled condition is malformed")
            path = Path(str(private.get("content_path") or ""))
            if not path.is_file() or _sha256_bytes(path.read_bytes()) != canonical.get(
                "content_sha256"
            ):
                raise TrajectoryValidationError(
                    "compiled condition bytes differ from the reserved manifest"
                )
            for key in canonical:
                if private.get(key) != canonical.get(key):
                    raise TrajectoryValidationError(
                        "private and canonical compiled conditions differ"
                    )
        effective_input = {}
        if prompt_authoring:
            effective_input["generator_prompt_sha256"] = _require_sha256(reservation.get("generator_prompt_sha256"), "generator_prompt_sha256")
            seed = reservation.get("generator_seed")
            if type(seed) is not int or not 0 <= seed < 2 ** 32:
                raise TrajectoryValidationError("generation reservation has an invalid seed")
            effective_input["generator_seed"] = seed
        expected_signature = _sha256_json({
            "schema_version": "storymem.effective-generation-input.v2" if prompt_authoring else "storymem.compiled-reference-conditions.v1",
            "conditions": canonical_conditions,
            **effective_input,
        })
        if reservation.get("condition_signature_sha256") != expected_signature:
            raise TrajectoryValidationError("generation reservation signature mismatch")
        if expected_signature in reserved_signatures:
            raise TrajectoryValidationError(
                "the same compiled reference conditions were reserved twice"
            )
        normalized_input = {
            key: copy.deepcopy(reservation.get(key))
            for key in (
                "request_id",
                "raw_request_sha256",
                "plan_id",
                "plan_sha256",
                "base_prompt_artifact_id",
                "base_prompt_artifact_sha256",
                "wire_prompt_artifact_id",
                "wire_prompt_artifact_sha256",
                "compilation_receipt_id",
                "compilation_receipt_sha256",
                "binding_outcomes",
                "canonical_conditions",
            )
        }
        if reservation.get("reservation_input_sha256") != _sha256_json(
            {**normalized_input, **effective_input}
        ):
            raise TrajectoryValidationError("generation reservation input hash mismatch")
        reservation_by_id[reservation_id] = reservation
        reserved_signatures.add(expected_signature)
        reserved_generation_ids.add(generation_id)

    bundles = value.get("private_generation_bundles")
    if not isinstance(bundles, list):
        raise TrajectoryValidationError("private generation bundles must be an array")
    bundle_by_id: dict[str, Mapping[str, Any]] = {}
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            raise TrajectoryValidationError("private generation bundle is malformed")
        bundle_id = str(bundle.get("bundle_id") or "")
        if not bundle_id or bundle_id in bundle_by_id:
            raise TrajectoryValidationError("private generation bundle IDs must be unique")
        if bundle.get("request_id", bundle.get("shot_request_id")) != value.get("request_id"):
            raise TrajectoryValidationError("generation bundle request ID mismatch")
        if bundle.get("raw_request_sha256") != raw_hash:
            raise TrajectoryValidationError("generation bundle raw request hash mismatch")
        bundle_by_id[bundle_id] = bundle
    private_feedback = value.get("private_compilation_feedback")
    if not isinstance(private_feedback, list):
        raise TrajectoryValidationError(
            "private compilation feedback must be an array"
        )
    feedback_by_id: dict[str, Mapping[str, Any]] = {}
    for feedback in private_feedback:
        if not isinstance(feedback, Mapping) or set(feedback) != {
            "feedback_id",
            "request_id",
            "raw_request_sha256",
            "plan_id",
            "plan_sha256",
            "binding_id",
            "reason_code",
        }:
            raise TrajectoryValidationError(
                "private compilation feedback is malformed"
            )
        feedback_id = str(feedback.get("feedback_id") or "")
        if not feedback_id or feedback_id in feedback_by_id:
            raise TrajectoryValidationError(
                "compilation feedback IDs must be non-empty and unique"
            )
        if (
            feedback.get("request_id") != value.get("request_id")
            or feedback.get("raw_request_sha256") != raw_hash
        ):
            raise TrajectoryValidationError(
                "compilation feedback request lineage mismatch"
            )
        try:
            reason_code = validate_neutral_reason_code(feedback.get("reason_code"))
        except ValueError as exc:
            raise TrajectoryValidationError(str(exc)) from exc
        if reason_code in {"condition.applied", "condition.fallback"}:
            raise TrajectoryValidationError(
                "compilation failure uses a non-failure reason code"
            )
        feedback_by_id[feedback_id] = feedback
    host_kinds: list[str] = []
    candidate_plan_ids: list[str] = []
    compiled_failure_plan_ids: list[str] = []
    resolved_plan_ids: list[str] = []
    candidate_bundle_by_id: dict[str, Mapping[str, Any]] = {}
    used_condition_signatures: set[str] = set()
    used_generation_ids: set[str] = set()
    base_prompt_artifact_lock: tuple[str, str] | None = None
    for event in host_events:
        observation = event.get("observation") or {}
        if event.get("name") == "attach_compilation_feedback":
            feedback_id = str(event.get("feedback_id") or "")
            feedback = feedback_by_id.get(feedback_id)
            if feedback is None:
                raise TrajectoryValidationError(
                    "host compilation feedback has no matching private record"
                )
            if event.get("feedback_sha256") != _sha256_json(feedback):
                raise TrajectoryValidationError(
                    "host compilation feedback hash mismatch"
                )
            if (
                event.get("phase_before") != "awaiting_generation"
                or event.get("phase_after") != "planning"
            ):
                raise TrajectoryValidationError(
                    "compilation feedback has an invalid FSM transition"
                )
            plan = plan_by_id.get(str(feedback.get("plan_id") or ""))
            if plan is None or feedback.get("plan_sha256") != plan.get(
                "plan_sha256"
            ):
                raise TrajectoryValidationError(
                    "compilation feedback EvidencePlan lineage mismatch"
                )
            binding_id = feedback.get("binding_id")
            if binding_id is not None and binding_id not in {
                row.get("binding_id") for row in plan.get("ranked_bindings") or []
            }:
                raise TrajectoryValidationError(
                    "compilation feedback names an unknown binding"
                )
            expected_observation = {
                "ok": True,
                "phase": "planning",
                "feedback_kind": "condition_compilation_failure",
                "binding_id": binding_id,
                "reason_code": feedback.get("reason_code"),
                "revision_required": True,
                "remaining_condition_revisions": observation.get(
                    "remaining_condition_revisions"
                ),
            }
            remaining = observation.get("remaining_condition_revisions")
            if (
                observation != expected_observation
                or type(remaining) is not int
                or not 0 <= remaining <= max_revisions
            ):
                raise TrajectoryValidationError(
                    "public compilation feedback is malformed or differs from its private record"
                )
            plan_id = str(plan.get("plan_id") or "")
            if plan_id in compiled_failure_plan_ids:
                raise TrajectoryValidationError(
                    "an EvidencePlan has duplicate compilation feedback"
                )
            compiled_failure_plan_ids.append(plan_id)
            resolved_plan_ids.append(plan_id)
            continue
        if event.get("name") != "attach_generation_evidence":
            raise TrajectoryValidationError("trajectory contains an unknown host event")
        bundle_id = str(observation.get("bundle_id") or "")
        bundle = bundle_by_id.get(bundle_id)
        if bundle is None:
            raise TrajectoryValidationError("host event has no matching private bundle")
        if event.get("bundle_sha256") != _sha256_json(bundle):
            raise TrajectoryValidationError("host generation bundle hash mismatch")
        if observation.get("bundle_kind") != bundle.get("bundle_kind"):
            raise TrajectoryValidationError("host generation bundle kind mismatch")
        bundle_kind = str(bundle.get("bundle_kind") or "")
        host_kinds.append(bundle_kind)
        plan = plan_by_id.get(str(bundle.get("plan_id") or ""))
        if plan is None or bundle.get("plan_sha256") != plan.get("plan_sha256"):
            raise TrajectoryValidationError("generation bundle EvidencePlan lineage mismatch")
        reservation_id = str(bundle.get("generation_reservation_id") or "")
        reservation = reservation_by_id.get(reservation_id)
        if reservation is None:
            raise TrajectoryValidationError(
                "generation bundle has no durable pre-generation reservation"
            )
        for field_name in (
            "request_id",
            "raw_request_sha256",
            "plan_id",
            "plan_sha256",
            "base_prompt_artifact_id",
            "base_prompt_artifact_sha256",
            "wire_prompt_artifact_id",
            "wire_prompt_artifact_sha256",
            "compilation_receipt_id",
            "compilation_receipt_sha256",
            "condition_signature_sha256",
            "generation_id",
            "generation_output_sha256",
        ):
            if bundle.get(field_name) != reservation.get(field_name):
                raise TrajectoryValidationError(
                    f"generation bundle differs from reservation {field_name}"
                )
        base_prompt_artifact = (
            str(bundle.get("base_prompt_artifact_id") or ""),
            _require_sha256(
                bundle.get("base_prompt_artifact_sha256"),
                "base_prompt_artifact_sha256",
            ),
        )
        wire_prompt_artifact = (
            str(bundle.get("wire_prompt_artifact_id") or ""),
            _require_sha256(
                bundle.get("wire_prompt_artifact_sha256"),
                "wire_prompt_artifact_sha256",
            ),
        )
        if not base_prompt_artifact[0] or not wire_prompt_artifact[0]:
            raise TrajectoryValidationError(
                "generation bundle has incomplete prompt artifact lineage"
            )
        if base_prompt_artifact_lock is None:
            base_prompt_artifact_lock = base_prompt_artifact
        elif not prompt_authoring and base_prompt_artifact != base_prompt_artifact_lock:
            raise TrajectoryValidationError(
                "shot-level base prompt artifact changed across condition plans"
            )
        for field_name in (
            "compilation_receipt_sha256",
            "condition_signature_sha256",
            "generation_output_sha256",
        ):
            _require_sha256(bundle.get(field_name), field_name)
        if not bundle.get("compilation_receipt_id") or not bundle.get("generation_id"):
            raise TrajectoryValidationError("generation bundle has incomplete lineage IDs")
        if observation.get("generation_id") != bundle.get("generation_id"):
            raise TrajectoryValidationError("public generation ID differs from private bundle")
        if observation.get("generation_output_sha256") != bundle.get(
            "generation_output_sha256"
        ):
            raise TrajectoryValidationError("public generation hash differs from private bundle")
        if observation.get("condition_signature_sha256") != bundle.get(
            "condition_signature_sha256"
        ):
            raise TrajectoryValidationError("public condition signature differs from private bundle")
        derived = bundle.get("derived_evidence")
        if not isinstance(derived, list):
            raise TrajectoryValidationError("generation bundle derived_evidence is malformed")
        expected_available = [
            {
                "evidence_id": row.get("evidence_id", row.get("evidence_ref")),
                "modality": row.get("modality")
                or (row.get("representation") or {}).get("modality"),
                "writeback_candidate": bool(row.get("writeback_candidate", False)),
            }
            for row in derived
        ]
        if observation.get("available_evidence") != expected_available:
            raise TrajectoryValidationError(
                "host observation differs from private derived evidence"
            )
        output_hash = str(bundle.get("generation_output_sha256") or "")
        for row in derived:
            representation_hash = str((row.get("representation") or {}).get("sha256") or "")
            if representation_hash == output_hash:
                continue
            if row.get("parent_output_sha256") != output_hash:
                raise TrajectoryValidationError(
                    "derived evidence parent differs from generation output"
                )
            if not row.get("derivation") or row.get("selector") is None:
                raise TrajectoryValidationError(
                    "derived evidence lacks deterministic derivation metadata"
                )
        if bundle_kind == "candidate_feedback":
            candidate_plan_ids.append(str(bundle.get("plan_id") or ""))
            resolved_plan_ids.append(str(bundle.get("plan_id") or ""))
            signature = str(bundle.get("condition_signature_sha256") or "")
            generation_id = str(bundle.get("generation_id") or "")
            if signature in used_condition_signatures:
                raise TrajectoryValidationError("condition signature was generated twice")
            if generation_id in used_generation_ids:
                raise TrajectoryValidationError("generation ID was reviewed twice")
            used_condition_signatures.add(signature)
            used_generation_ids.add(generation_id)
            candidate_bundle_by_id[bundle_id] = bundle
            review_target_id = str(bundle.get("review_target_evidence_id") or "")
            if review_target_id not in {
                str(row.get("evidence_id", row.get("evidence_ref")) or "")
                for row in derived
            }:
                raise TrajectoryValidationError(
                    "candidate review target is absent from derived evidence"
                )
            if observation.get("review_target_evidence_id") != review_target_id:
                raise TrajectoryValidationError(
                    "public candidate review target differs from private bundle"
                )
            if not any(
                str((row.get("representation") or {}).get("sha256") or "")
                == output_hash
                for row in derived
            ):
                raise TrajectoryValidationError(
                    "candidate feedback omits the generated output representation"
                )
            outcomes = bundle.get("binding_outcomes")
            if not isinstance(outcomes, list):
                raise TrajectoryValidationError("candidate feedback has no binding outcomes")
            if outcomes != reservation.get("binding_outcomes"):
                raise TrajectoryValidationError(
                    "candidate binding outcomes differ from generation reservation"
                )
            binding_by_id = {
                str(row.get("binding_id") or ""): row
                for row in plan.get("ranked_bindings") or []
            }
            seen: set[str] = set()
            public_outcomes: list[dict[str, Any]] = []
            for outcome in outcomes:
                if not isinstance(outcome, Mapping):
                    raise TrajectoryValidationError("candidate binding outcome is malformed")
                binding_id = str(outcome.get("binding_id") or "")
                status = str(outcome.get("outcome") or "")
                if binding_id not in binding_by_id or binding_id in seen:
                    raise TrajectoryValidationError(
                        "candidate binding outcomes contain an unknown or duplicate binding"
                    )
                if status not in {
                    "applied_as_requested",
                    "allowed_fallback",
                    "not_applied",
                }:
                    raise TrajectoryValidationError("candidate binding outcome is invalid")
                if (
                    status == "allowed_fallback"
                    and binding_by_id[binding_id].get("fallback") == "forbidden"
                ):
                    raise TrajectoryValidationError(
                        "candidate feedback used a forbidden fallback"
                    )
                reason_code = outcome.get("reason_code")
                applied_effect = outcome.get("applied_effect")
                requested_effect = binding_by_id[binding_id].get("intended_effect")
                try:
                    validate_neutral_binding_outcome(
                        outcome=status,
                        reason_code=reason_code,
                        applied_effect=applied_effect,
                        requested_effect=requested_effect,
                    )
                except ValueError as exc:
                    raise TrajectoryValidationError(
                        f"candidate binding outcome is not model-neutral: {exc}"
                    ) from exc
                seen.add(binding_id)
                public_outcomes.append(
                    {
                        "binding_id": binding_id,
                        "outcome": status,
                        "applied_effect": applied_effect,
                        "reason_code": reason_code,
                    }
                )
            if seen != set(binding_by_id):
                raise TrajectoryValidationError(
                    "candidate feedback must report each plan binding exactly once"
                )
            if observation.get("binding_outcomes") != public_outcomes:
                raise TrajectoryValidationError(
                    "public binding outcomes differ from the private bundle"
                )
        elif bundle_kind == "final_accepted":
            if bundle.get("selection_disposition") not in {
                "condition_pass",
                "budget_best_effort",
            } or not bundle.get("finalization_receipt_id"):
                raise TrajectoryValidationError(
                    "final bundle lacks a valid selection disposition or receipt"
                )
            if observation.get("binding_outcomes") != []:
                raise TrajectoryValidationError(
                    "final bundle observation must not report binding outcomes"
                )
            if observation.get("review_target_evidence_id") is not None:
                raise TrajectoryValidationError(
                    "final bundle must not expose a condition review target"
                )
            accepted_bundle_id = str(bundle.get("accepted_candidate_bundle_id") or "")
            accepted_candidate = candidate_bundle_by_id.get(accepted_bundle_id)
            if accepted_candidate is None:
                raise TrajectoryValidationError(
                    "final bundle does not identify a reviewed candidate"
                )
            for field_name in (
                "plan_id",
                "plan_sha256",
                "generation_reservation_id",
                "base_prompt_artifact_id",
                "base_prompt_artifact_sha256",
                "wire_prompt_artifact_id",
                "wire_prompt_artifact_sha256",
                "compilation_receipt_id",
                "compilation_receipt_sha256",
                "condition_signature_sha256",
                "generation_id",
                "generation_output_sha256",
            ):
                if bundle.get(field_name) != accepted_candidate.get(field_name):
                    raise TrajectoryValidationError(
                        f"final bundle changed reviewed candidate {field_name}"
                    )
        else:
            raise TrajectoryValidationError("generation bundle kind is invalid")
    if not host_kinds or host_kinds[-1] != "final_accepted":
        raise TrajectoryValidationError("trajectory has no final_accepted bundle")
    if len(feedback_by_id) != len(compiled_failure_plan_ids):
        raise TrajectoryValidationError(
            "private compilation feedback is not represented exactly once"
        )
    expected_plan_ids = [str(plan.get("plan_id") or "") for plan in plans]
    if resolved_plan_ids != expected_plan_ids:
        raise TrajectoryValidationError(
            "EvidencePlans were not resolved once each in revision order"
        )
    generated_plan_count = len(plans) - len(compiled_failure_plan_ids)
    if host_kinds.count("candidate_feedback") != generated_plan_count:
        raise TrajectoryValidationError(
            "each compilable EvidencePlan must receive exactly one candidate_feedback bundle"
        )
    if len(reservation_by_id) != host_kinds.count("candidate_feedback"):
        raise TrajectoryValidationError(
            "each generated candidate must have exactly one durable reservation"
        )
    if candidate_plan_ids != [
        plan_id for plan_id in expected_plan_ids if plan_id not in compiled_failure_plan_ids
    ]:
        raise TrajectoryValidationError(
            "candidate feedback bundles do not follow compilable EvidencePlan order"
        )
    final_bundle_id = str((host_events[-1].get("observation") or {}).get("bundle_id") or "")
    final_bundle = bundle_by_id[final_bundle_id]
    selected_candidate_bundle = candidate_bundle_by_id.get(
        str(final_bundle.get("accepted_candidate_bundle_id") or "")
    )
    terminal_condition_decisions = [
        event["tool_call"]["arguments"]
        for event in events
        if (event.get("tool_call") or {}).get("name")
        == "submit_candidate_assessment"
        and (event.get("tool_call") or {}).get("arguments", {}).get("decision")
        in {"accept_conditions", "cannot_fix_with_memory"}
    ]
    if len(terminal_condition_decisions) != 1 or selected_candidate_bundle is None:
        raise TrajectoryValidationError(
            "trajectory must contain one terminal condition candidate selection"
        )
    if terminal_condition_decisions[0].get("selected_candidate_id") != (
        selected_candidate_bundle.get("review_target_evidence_id")
    ):
        raise TrajectoryValidationError(
            "final bundle differs from the agent-selected assessed candidate"
        )
    expected_disposition = (
        "condition_pass"
        if terminal_condition_decisions[0].get("decision") == "accept_conditions"
        else "budget_best_effort"
    )
    if final_bundle.get("selection_disposition") != expected_disposition:
        raise TrajectoryValidationError(
            "final selection disposition differs from terminal condition decision"
        )
    finish_arguments = events[-1]["tool_call"]["arguments"]
    if finish_arguments.get("accepted_plan_id") != final_bundle.get("plan_id"):
        raise TrajectoryValidationError(
            "finish_memory_session does not name the selected candidate's plan"
        )

    messages = value.get("messages")
    if not isinstance(messages, list):
        raise TrajectoryValidationError("trajectory messages must be an array")
    expected_messages = build_training_messages(
        raw_user_request=raw_request,
        events=events,
        host_events=host_events,
        prompt_authoring=prompt_authoring,
    )
    if messages != expected_messages:
        raise TrajectoryValidationError("trajectory messages are not deterministic")
    if value.get("messages_sha256") != _sha256_json(messages):
        raise TrajectoryValidationError("trajectory messages hash mismatch")
    _assert_model_safe(messages[2:], label="trajectory messages")
    public_media = trajectory_media_manifest(messages)
    if value.get("media_manifest") != public_media:
        raise TrajectoryValidationError("trajectory public media manifest mismatch")
    if value.get("media_manifest_sha256") != _sha256_json(public_media):
        raise TrajectoryValidationError("trajectory media manifest hash mismatch")
    stored_public_media = value.get("media_manifest")
    private_media = value.get("private_media_manifest")
    if not isinstance(stored_public_media, list) or not isinstance(private_media, list):
        raise TrajectoryValidationError("trajectory media manifests must be arrays")
    public_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(stored_public_media):
        _validate_media_row(row, kind="manifest", label=f"public media row {index}")
        media_id = str(row.get("media_id") or "")
        if media_id in public_by_id:
            raise TrajectoryValidationError("trajectory public media IDs must be unique")
        public_by_id[media_id] = row
    private_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(private_media):
        _validate_media_row(row, kind="private", label=f"private media row {index}")
        media_id = str(row.get("media_id") or "")
        if media_id in private_by_id:
            raise TrajectoryValidationError("trajectory private media IDs must be unique")
        private_by_id[media_id] = row
    if set(public_by_id) != set(private_by_id):
        raise TrajectoryValidationError(
            "trajectory public/private media IDs must match exactly"
        )
    for media_id, public in public_by_id.items():
        private = private_by_id[media_id]
        public_payload = {
            key: copy.deepcopy(child)
            for key, child in public.items()
            if key not in {"message_index", "content_index"}
        }
        private_payload = {
            key: copy.deepcopy(child)
            for key, child in private.items()
            if key not in {"event_id", "path"}
        }
        if public_payload != private_payload:
            raise TrajectoryValidationError(
                f"trajectory media {media_id} public/private metadata mismatch"
            )
        path = Path(str(private.get("path") or ""))
        if not path.is_file():
            raise TrajectoryValidationError(
                f"trajectory private media file is missing: {path}"
            )
        if (
            _sha256_bytes(path.read_bytes()) != public.get("sha256")
            or path.stat().st_size != public.get("size_bytes")
        ):
            raise TrajectoryValidationError(
                f"trajectory media {media_id} bytes differ from the manifest"
            )
    if private_media != expected_private_media:
        raise TrajectoryValidationError("trajectory private media manifest mismatch")
    return {
        "ok": True,
        "tool_call_count": len(events),
        "host_event_count": len(host_events),
        "plan_count": len(plans),
        "media_count": len(public_media),
    }


__all__ = [
    "TRAJECTORY_SCHEMA_VERSION",
    "TrajectoryValidationError",
    "build_teacher_input_messages",
    "build_training_messages",
    "build_trajectory_shard",
    "parse_plain_json_tool_command",
    "trajectory_media_manifest",
    "validate_trajectory_shard",
]
