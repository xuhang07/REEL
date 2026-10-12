"""Canonical, dependency-free StoryMem V3 tool contract.

This module contains the model-visible action space only.  It deliberately
does not import a provider SDK, execute a model, or read/write media.  Runtime
implementations, teacher collection, SFT export, RL rollout, and inference
must all obtain their tool definitions from :func:`get_tool_schemas`.

Every ``function.parameters`` value is an independent JSON Schema Draft
2020-12 document.  Object inputs are closed so misspelled or legacy fields are
rejected instead of being silently ignored.  Cross-call and stateful rules
(for example registry-first and search-before-view) remain runtime FSM checks;
the schemas encode every useful within-call invariant.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


TOOL_PROTOCOL_VERSION = "storymem.canonical.tools.v4"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
PROMPT_REWRITE_MODES: tuple[str, ...] = (
    "rewrite_enabled",
    "passthrough_disabled",
)

CANONICAL_TOOL_NAMES: tuple[str, ...] = (
    "view_material_registry",
    "search_memory_bank",
    "view_memories",
    "get_pe_skill",
    "register_prompt",
    "generate_image",
    "generate_shot",
    "inspect_candidate",
    "submit_candidate_assessment",
    "update_memory_bank",
    "submit_shot",
)


def _id(description: str = "Opaque runtime identifier.") -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "description": description}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"oneOf": [copy.deepcopy(schema), {"type": "null"}]}


def _closed_object(
    properties: dict[str, Any],
    required: tuple[str, ...] | list[str],
    *,
    description: str | None = None,
    one_of: list[dict[str, Any]] | None = None,
    all_of: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    if one_of:
        schema["oneOf"] = one_of
    if all_of:
        schema["allOf"] = all_of
    return schema


_FRAME_SOURCE = _closed_object(
    {
        "kind": {"type": "string", "enum": ["memory", "uploaded_asset", "generated_image"]},
        "source_id": _id("ID of the immutable source asset or candidate."),
        "representation_id": _id("Exact provider-ready image representation ID."),
        "authorization_id": _id("Current-shot view or selection authorization ID."),
    },
    ("kind", "source_id", "representation_id", "authorization_id"),
    description="An image authorized for use as a complete frame-zero condition.",
)

_TAIL_SOURCE = _closed_object(
    {
        "kind": {"const": "previous_tail"},
        "source_id": _id("Previous accepted shot tail handle."),
        "representation_id": _id("Exact viewed tail-frame representation ID."),
        "authorization_id": _id("Current-shot view authorization ID."),
    },
    ("kind", "source_id", "representation_id", "authorization_id"),
    description="The viewed tail frame of the previous accepted shot.",
)

_CONDITION_BINDING: dict[str, Any] = {
    "description": "The complete public video condition binding.",
    "oneOf": [
        _closed_object(
            {
                "relation": {"const": "frame0_match"},
                "image_source": copy.deepcopy(_FRAME_SOURCE),
            },
            ("relation", "image_source"),
        ),
        _closed_object(
            {
                "relation": {"const": "temporal_continuation"},
                "image_source": copy.deepcopy(_TAIL_SOURCE),
            },
            ("relation", "image_source"),
        ),
    ],
}

_CONDITION_OR_NULL: dict[str, Any] = {
    "oneOf": [{"type": "null"}, copy.deepcopy(_CONDITION_BINDING)]
}

_IMAGE_PROMPT_CONTEXT = _closed_object(
    {
        "condition": {"type": "null"},
        "has_reference_inputs": {"type": "boolean"},
    },
    ("condition", "has_reference_inputs"),
    description="Image prompt context; condition is never used for image generation.",
)

_VIDEO_PROMPT_CONTEXT = _closed_object(
    {
        "condition": copy.deepcopy(_CONDITION_OR_NULL),
        "has_reference_inputs": {"type": "null"},
    },
    ("condition", "has_reference_inputs"),
    description="Video prompt context; ordinary memory slots are not prompt-context fields.",
)

_PROMPT_CONTEXT: dict[str, Any] = {
    "description": "Closed target-tagged prompt context.",
    "oneOf": [copy.deepcopy(_IMAGE_PROMPT_CONTEXT), copy.deepcopy(_VIDEO_PROMPT_CONTEXT)],
}

_IMAGE_GENERATION_SCOPE = _closed_object(
    {
        "subject_ids": {
            "type": "array",
            "items": _id("Stable subject ID that this image group must preserve."),
            "uniqueItems": True,
        },
        "asset_role": _nullable(
            _id("Stable role that distinguishes reusable reference-asset groups.")
        ),
    },
    ("subject_ids", "asset_role"),
    description=(
        "Immutable semantic scope for one image prompt/group. It distinguishes "
        "multiple reference assets that use the same PE route and evidence."
    ),
)

_GENERATION_SCOPE: dict[str, Any] = {
    "description": "Image group scope; null for video prompts.",
    "oneOf": [copy.deepcopy(_IMAGE_GENERATION_SCOPE), {"type": "null"}],
}

_SOURCE_SPAN = {
    "oneOf": [
        {"type": "null"},
        _closed_object(
            {
                "unit": {"const": "unicode_codepoint"},
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 1},
            },
            ("unit", "start", "end"),
            description="Half-open source span; runtime additionally verifies end > start.",
        ),
    ]
}

SEMANTIC_MODULES = (
    "identity_wardrobe",
    "scene_environment",
    "object_state",
    "interaction_spatial",
    "camera_composition",
    "motion_temporal",
    "style_lighting",
    "speech_audio",
)

ASSESSMENT_CONTRACT_SCHEMA_VERSION = (
    "storymem.canonical.assessment-contract.v1"
)
EVALUATION_CONTRACT_AUTHORING_SCHEMA_VERSION = (
    "storymem.canonical.evaluation-contract-authoring.v1"
)
APPLICABLE_MODULE_REASON = (
    "The frozen inspection checklist contains a check for this semantic module."
)
NOT_APPLICABLE_MODULE_REASON = (
    "The frozen inspection checklist contains no check for this semantic module."
)
AUDIO_CAPABILITY_BLOCKED_RATIONALE = (
    "Audio perception is disabled for this inspection."
)
AUDIO_TRANSPORT_ONLY_USE_RATIONALE = (
    "Audio is authorized for transport from trusted metadata but cannot be "
    "acoustically evaluated by this agent."
)


def build_assessment_contract(
    checklist: Iterable[Mapping[str, Any]],
    capability_mask: Mapping[str, Any],
    use_contracts: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the model-visible deterministic portion of an assessment.

    The agent judges observable content, but it must not improvise bookkeeping
    or claims for a disabled modality. Keeping this builder in the canonical
    contract package gives collection, replay validation, SFT, and inference a
    single byte-stable source of truth.
    """

    rows = [copy.deepcopy(dict(item)) for item in checklist]
    applicable_modules = {
        str(item.get("module") or "")
        for item in rows
        if item.get("module") in SEMANTIC_MODULES
    }
    module_applicability = [
        {
            "module": module,
            "applicability": (
                "applicable" if module in applicable_modules else "not_applicable"
            ),
            "reason": (
                APPLICABLE_MODULE_REASON
                if module in applicable_modules
                else NOT_APPLICABLE_MODULE_REASON
            ),
        }
        for module in SEMANTIC_MODULES
    ]
    forced_check_results = []
    if capability_mask.get("audio") is False:
        forced_check_results = [
            {
                "check_id": str(item.get("check_id") or ""),
                "result": "capability_blocked",
                "evidence_ids": [],
                "rationale": AUDIO_CAPABILITY_BLOCKED_RATIONALE,
            }
            for item in rows
            if item.get("module") == "speech_audio"
        ]
    forced_use_results = []
    if capability_mask.get("audio") is False:
        forced_use_results = [
            {
                "use_contract_id": str(item.get("use_contract_id") or ""),
                "use": str(item.get("use") or ""),
                "result": "capability_blocked",
                "source_ids": [str(source_id) for source_id in item.get("source_ids") or []],
                "evidence_ids": [],
                "rationale": AUDIO_TRANSPORT_ONLY_USE_RATIONALE,
            }
            for item in use_contracts
            if item.get("modality") == "audio"
        ]
    all_observable_pass_mode = (
        "capability_limited"
        if forced_check_results or forced_use_results
        else "requirements_met"
    )
    return {
        "schema_version": ASSESSMENT_CONTRACT_SCHEMA_VERSION,
        "module_applicability": module_applicability,
        "forced_check_results": forced_check_results,
        "forced_use_results": forced_use_results,
        "selection_mode_guidance": {
            "result_scope": (
                "selected_candidate_critical_checks_and_primary_use_contracts"
            ),
            "candidate_quality_precedence": [
                "qualified",
                "capability_limited",
                "failed",
            ],
            "selection_mode_by_selected_quality": {
                "qualified": "requirements_met",
                "capability_limited": "capability_limited",
                "failed": "best_effort",
            },
            "if_selected_any_fail_or_uncertain": "best_effort",
            "if_selected_none_fail_or_uncertain_and_any_capability_blocked": (
                "capability_limited"
            ),
            "if_selected_all_pass": "requirements_met",
            "failed_selection_requires_zero_remaining_attempts": True,
            "selected_candidate_must_have_highest_available_quality": True,
            "all_non_forced_required_results_pass_mode": all_observable_pass_mode,
        },
    }


def build_evaluation_contract_authoring(
    *,
    shot_request_id: str,
    raw_user_request: str,
    prompt_artifact_id: str,
    receipt_chain: Iterable[str],
) -> dict[str, Any]:
    """Return deterministic provenance fields and span rules for generation."""

    # A prompt artifact is PE-authored only when it has a real receipt chain.
    # Passthrough artifacts deliberately have an empty chain, so advertising the
    # artifact itself as a PE provenance source would let the model relabel raw
    # USER text as ``pe_required``/``pe_added``.
    receipts = [str(item) for item in receipt_chain]
    pe_source_ids = (
        list(dict.fromkeys([str(prompt_artifact_id), *receipts]))
        if receipts
        else []
    )
    return {
        "schema_version": EVALUATION_CONTRACT_AUTHORING_SCHEMA_VERSION,
        "raw_user_request_sha256": "sha256:"
        + hashlib.sha256(raw_user_request.encode("utf-8")).hexdigest(),
        "user_required": {
            "provenance_kind": "user_required",
            "source_id": str(shot_request_id),
            "source_span": {
                "unit": "unicode_codepoint",
                "start": 0,
                "end": len(raw_user_request),
            },
            "copy_policy": "copy_source_id_and_source_span_exactly",
        },
        "provenance_span_rules": [
            {
                "provenance_kind": "user_required",
                "allowed_source_ids": [str(shot_request_id)],
                "source_span_policy": "required_full_raw_user_request",
            },
            {
                "provenance_kind": "pe_required",
                "allowed_source_ids": pe_source_ids,
                "source_span_policy": "null_or_unicode_subspan_only_for_current_prompt_artifact",
            },
            {
                "provenance_kind": "pe_added",
                "allowed_source_ids": pe_source_ids,
                "source_span_policy": "null_or_unicode_subspan_only_for_current_prompt_artifact",
            },
            {
                "provenance_kind": "runtime_bound",
                "allowed_source_ids": [],
                "source_id_policy": "authorized_runtime_binding_or_context_observation",
                "source_span_policy": "must_be_null",
            },
        ],
    }


def evaluation_check_evidence_kinds(*, target: str, module: str) -> list[str]:
    """Return the deterministic evidence kinds appropriate for one check."""

    if target != "video":
        return ["target_image", "input_reference"]
    visual_target = ["target_video", "target_frame"]
    if module in {
        "identity_wardrobe",
        "scene_environment",
        "object_state",
        "interaction_spatial",
        "style_lighting",
        "reference_consistency",
    }:
        return [*visual_target, "condition_image", "slot_image"]
    if module in {
        "camera_composition",
        "motion_temporal",
        "frame0_condition_alignment",
        "tail_to_start_continuity",
        "condition_readiness",
    }:
        return [*visual_target, "condition_image"]
    if module == "speech_audio":
        return ["target_video", "target_audio", "slot_audio"]
    return visual_target


# Private compatibility alias used throughout the schema declarations below.
_SEMANTIC_MODULES = SEMANTIC_MODULES

# Runtime-bound modules above are instantiated deterministically from the
# actual condition/reference/slot/writeback bindings.  Letting the agent add
# lookalike rows would bypass their specialized evidence and capability rules.
_DECLARED_REQUIREMENT_MODULES = ("prompt_content", *_SEMANTIC_MODULES)

_REQUIREMENT = _closed_object(
    {
        "requirement_id": _id("Stable requirement ID within the generation group."),
        "module": {
            "type": "string",
            "enum": list(_DECLARED_REQUIREMENT_MODULES),
            "description": (
                "Agent-authored semantic module. Binding-specific checks such as "
                "condition, reference, voice, and writeback quality are added by "
                "the runtime and cannot be declared here."
            ),
        },
        "scope": {
            "type": "string",
            "enum": ["first_frame", "reference_asset", "final_video", "memory_writeback"],
        },
        "text": {"type": "string", "minLength": 1},
        "severity": {"type": "string", "enum": ["critical", "advisory"]},
        "provenance_kind": {
            "type": "string",
            "enum": ["user_required", "pe_required", "pe_added", "runtime_bound"],
        },
        "source_id": _id(
            "Raw request, PE receipt, prompt artifact, or runtime binding ID; obey the "
            "evaluation_contract_authoring guidance returned by register_prompt."
        ),
        "source_span": {
            **copy.deepcopy(_SOURCE_SPAN),
            "description": (
                "Copy the full raw-request Unicode span for user_required; use null or a "
                "valid current-prompt subspan for PE provenance; use null for runtime_bound."
            ),
        },
        "not_applicable_allowed": {"type": "boolean"},
    },
    (
        "requirement_id",
        "module",
        "scope",
        "text",
        "severity",
        "provenance_kind",
        "source_id",
        "source_span",
        "not_applicable_allowed",
    ),
)

_EVALUATION_CONTRACT = _closed_object(
    {
        "modules": {
            "type": "array",
            "items": {"type": "string", "enum": list(_SEMANTIC_MODULES)},
            "minItems": 1,
            "uniqueItems": True,
            "description": (
                "Relevant semantic checks selected before generation. Image groups "
                "must include camera_composition and style_lighting; video groups "
                "must include camera_composition, motion_temporal, and speech_audio. "
                "A video must additionally include identity_wardrobe for each active "
                "person/entity, scene_environment for each active scene, object_state "
                "for each active object, and style_lighting for each active style. Add "
                "every other module required by the request, prompt, or bound evidence."
            ),
        },
        "requirements": {"type": "array", "items": copy.deepcopy(_REQUIREMENT)},
    },
    ("modules", "requirements"),
    description=(
        "Semantic modules and immutable pre-generation requirements for this group. "
        "Construct provenance using the evaluation_contract_authoring object returned "
        "with the consumed prompt artifact; copy its user_required source_id and full "
        "Unicode source_span exactly."
    ),
)

_IMAGE_AXES = {
    "type": "array",
    "items": {"type": "string", "enum": ["prompt", "references", "seed"]},
    "minItems": 1,
    "uniqueItems": True,
}

_VIDEO_AXES = {
    "type": "array",
    "items": {"type": "string", "enum": ["prompt", "condition", "memory_slots", "seed"]},
    "minItems": 1,
    "uniqueItems": True,
}

_IMAGE_REVISE_AXES = {
    **copy.deepcopy(_IMAGE_AXES),
    "contains": {"type": "string", "enum": ["prompt", "references"]},
    "minContains": 1,
}

_VIDEO_REVISE_AXES = {
    **copy.deepcopy(_VIDEO_AXES),
    "contains": {"type": "string", "enum": ["prompt", "condition"]},
    "minContains": 1,
}

_EMPTY_AXES = {"type": "array", "maxItems": 0}

_IMAGE_RETRY = _closed_object(
    {
        "candidate_id": _id("Candidate whose assessment authorized this retry."),
        "assessment_id": _id("Immutable assessment authorizing this retry."),
        "axes": copy.deepcopy(_IMAGE_AXES),
        "reason": {"type": "string", "minLength": 1},
    },
    ("candidate_id", "assessment_id", "axes", "reason"),
)

_VIDEO_RETRY = _closed_object(
    {
        "candidate_id": _id("Candidate whose assessment authorized this retry."),
        "assessment_id": _id("Immutable assessment authorizing this retry."),
        "axes": copy.deepcopy(_VIDEO_AXES),
        "reason": {"type": "string", "minLength": 1},
    },
    ("candidate_id", "assessment_id", "axes", "reason"),
)

_AUTHORIZED_IMAGE_SOURCE = _closed_object(
    {
        "kind": {
            "type": "string",
            "enum": ["memory", "uploaded_asset", "generated_image", "previous_tail"],
        },
        "source_id": _id(),
        "representation_id": _id("Exact viewed provider-ready representation ID."),
        "authorization_id": _id("Search/view or selection authorization ID."),
    },
    ("kind", "source_id", "representation_id", "authorization_id"),
)

_SLOT_IMAGE_SOURCE = _closed_object(
    {
        "kind": {"type": "string", "enum": ["memory", "uploaded_asset", "generated_image"]},
        "source_id": _id(),
        "representation_id": _id("Exact viewed Echo-ready image representation ID."),
        "authorization_id": _id("Search/view or image-selection authorization ID."),
    },
    ("kind", "source_id", "representation_id", "authorization_id"),
)

_SLOT_AUDIO_SOURCE = _closed_object(
    {
        "kind": {"type": "string", "enum": ["memory", "uploaded_asset"]},
        "source_id": _id(),
        "representation_id": _id("Exact authorized audio representation ID."),
        "authorization_id": _id(
            "Current-shot view authorization; transport and quality authority remain source metadata."
        ),
    },
    ("kind", "source_id", "representation_id", "authorization_id"),
)

_MEMORY_SLOT = _closed_object(
    {
        "image_source": copy.deepcopy(_SLOT_IMAGE_SOURCE),
        "audio_source": {
            "oneOf": [{"type": "null"}, copy.deepcopy(_SLOT_AUDIO_SOURCE)]
        },
        "roles": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Use character_identity/wardrobe/voice only for human ID_X subjects; "
                    "use entity_identity/entity_appearance with empty subject_ids for "
                    "animals, robots, and other ENTITY_N subjects."
                ),
            },
            "minItems": 1,
            "uniqueItems": True,
        },
        "subject_ids": {
            "type": "array",
            "items": _id("Stable subject ID such as ID_A."),
            "uniqueItems": True,
        },
        "reason": {"type": "string", "minLength": 1},
    },
    ("image_source", "audio_source", "roles", "subject_ids", "reason"),
)


def _parameters(
    properties: dict[str, Any],
    required: tuple[str, ...] | list[str],
    *,
    one_of: list[dict[str, Any]] | None = None,
    all_of: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    schema = _closed_object(properties, required, one_of=one_of, all_of=all_of)
    schema["$schema"] = JSON_SCHEMA_DIALECT
    return schema


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _prompt_route_variants() -> list[dict[str, Any]]:
    """Return the four exact target/operation route constraints."""

    return [
        {
            "properties": {
                "target": {"const": "image"},
                "operation": {"const": "compose"},
                "purpose": {"enum": ["first_frame", "reference_asset"]},
                "prompt_context": copy.deepcopy(_IMAGE_PROMPT_CONTEXT),
                "generation_scope": copy.deepcopy(_IMAGE_GENERATION_SCOPE),
                "parent_prompt_artifact_id": {"type": "null"},
                "assessment_id": {"type": "null"},
                "axes": copy.deepcopy(_EMPTY_AXES),
            }
        },
        {
            "properties": {
                "target": {"const": "image"},
                "operation": {"const": "revise"},
                "purpose": {"enum": ["first_frame", "reference_asset"]},
                "prompt_context": copy.deepcopy(_IMAGE_PROMPT_CONTEXT),
                "generation_scope": copy.deepcopy(_IMAGE_GENERATION_SCOPE),
                "parent_prompt_artifact_id": _id(),
                "assessment_id": _id(),
                "axes": copy.deepcopy(_IMAGE_REVISE_AXES),
            }
        },
        {
            "properties": {
                "target": {"const": "video"},
                "operation": {"const": "compose"},
                "purpose": {"const": "shot"},
                "prompt_context": copy.deepcopy(_VIDEO_PROMPT_CONTEXT),
                "generation_scope": {"type": "null"},
                "parent_prompt_artifact_id": {"type": "null"},
                "assessment_id": {"type": "null"},
                "axes": copy.deepcopy(_EMPTY_AXES),
            }
        },
        {
            "properties": {
                "target": {"const": "video"},
                "operation": {"const": "revise"},
                "purpose": {"const": "shot"},
                "prompt_context": copy.deepcopy(_VIDEO_PROMPT_CONTEXT),
                "generation_scope": {"type": "null"},
                "parent_prompt_artifact_id": _id(),
                "assessment_id": _id(),
                "axes": copy.deepcopy(_VIDEO_REVISE_AXES),
            }
        },
    ]


_PROMPT_ROUTE_PROPERTIES: dict[str, Any] = {
    "shot_request_id": _id("Opaque shot request ID returned by view_material_registry."),
    "target": {"type": "string", "enum": ["image", "video"]},
    "operation": {"type": "string", "enum": ["compose", "revise"]},
    "purpose": {"type": "string", "enum": ["first_frame", "reference_asset", "shot"]},
    "prompt_context": copy.deepcopy(_PROMPT_CONTEXT),
    "generation_scope": copy.deepcopy(_GENERATION_SCOPE),
    "parent_prompt_artifact_id": _nullable(_id()),
    "assessment_id": _nullable(_id()),
    "axes": {
        "type": "array",
        "items": {
            "type": "string",
            "enum": ["prompt", "references", "condition", "memory_slots", "seed"],
        },
        "uniqueItems": True,
    },
}

_PROMPT_ROUTE_REQUIRED = (
    "shot_request_id",
    "target",
    "operation",
    "purpose",
    "prompt_context",
    "generation_scope",
    "parent_prompt_artifact_id",
    "assessment_id",
    "axes",
)


_MATERIAL_SELECTOR = {
    "oneOf": [
        _closed_object({"mode": {"const": "default_preview"}}, ("mode",)),
        _closed_object({"mode": {"const": "original"}}, ("mode",)),
        _closed_object(
            {
                "mode": {"const": "page"},
                "page_index": {"type": "integer", "minimum": 0},
            },
            ("mode", "page_index"),
        ),
        _closed_object(
            {
                "mode": {"const": "frame"},
                "time_ms": {"type": "integer", "minimum": 0},
            },
            ("mode", "time_ms"),
        ),
        _closed_object(
            {
                "mode": {"const": "time_range"},
                "start_ms": {"type": "integer", "minimum": 0},
                "end_ms": {"type": "integer", "minimum": 1},
            },
            ("mode", "start_ms", "end_ms"),
        ),
    ]
}

_VIEW_ITEM = {
    "oneOf": [
        _closed_object(
            {
                "kind": {"const": "memory"},
                "id": _id("Memory ID returned by search_memory_bank."),
                "search_id": _id("Search authorization containing this memory."),
                "representation_id": _nullable(_id("Optional exact representation; null selects the deterministic default.")),
            },
            ("kind", "id", "search_id", "representation_id"),
        ),
        _closed_object(
            {
                "kind": {"const": "uploaded_asset"},
                "id": _id("Pending upload handle from the registry snapshot."),
                "selector": copy.deepcopy(_MATERIAL_SELECTOR),
            },
            ("kind", "id", "selector"),
        ),
        _closed_object(
            {
                "kind": {"const": "previous_tail"},
                "id": _id("Previous-tail handle from the registry snapshot."),
            },
            ("kind", "id"),
        ),
    ]
}


_TAG_REF = {
    "oneOf": [
        _closed_object(
            {"kind": {"const": "existing_tag"}, "tag": _id("Existing stable registry tag.")},
            ("kind", "tag"),
        ),
        _closed_object(
            {
                "kind": {"const": "new_tag"},
                "client_ref": _id("Transaction-local reference for a newly allocated tag."),
            },
            ("kind", "client_ref"),
        ),
    ]
}

_MEMORY_COMMON_PROPERTIES: dict[str, Any] = {
    "client_ref": _id("Transaction-local unique memory reference."),
    "source_id": _id("Exact frame, generated reference, upload derivative, or audio clip ID."),
    "observation_id": _id("Inspection/view observation that exposed the exact source."),
    "assessment_id": _id("Assessment containing the memory-writeback decision."),
    "use_contract_id": _id("Passed source-specific memory-writeback use contract."),
    "tag_refs": {"type": "array", "items": copy.deepcopy(_TAG_REF), "minItems": 1},
    "roles": {
        "type": "array",
        "items": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Human identity uses character_identity/wardrobe/voice with ID_X; "
                "non-human identity uses entity_identity/entity_appearance with empty subject_ids."
            ),
        },
        "minItems": 1,
        "uniqueItems": True,
    },
    "subject_ids": {
        "type": "array",
        "items": _id("Stable subject ID."),
        "uniqueItems": True,
    },
    "description": {"type": "string", "minLength": 1},
}

_ADD_MEMORY = {
    "oneOf": [
        _closed_object(
            {
                **copy.deepcopy(_MEMORY_COMMON_PROPERTIES),
                "modality": {"const": "image"},
                "condition_eligible": {
                    "type": "boolean",
                    "description": (
                        "Request future frame-zero eligibility. Runtime grants it only to "
                        "an authorized complete frame and records the deterministic basis."
                    ),
                },
            },
            (
                "client_ref",
                "modality",
                "source_id",
                "observation_id",
                "assessment_id",
                "use_contract_id",
                "tag_refs",
                "roles",
                "subject_ids",
                "description",
            ),
        ),
        _closed_object(
            {
                **copy.deepcopy(_MEMORY_COMMON_PROPERTIES),
                "modality": {"const": "audio"},
                "verification_basis": {
                    "type": "string",
                    "enum": ["agent_heard", "trusted_user_declared"],
                },
            },
            (
                "client_ref",
                "modality",
                "source_id",
                "observation_id",
                "assessment_id",
                "use_contract_id",
                "tag_refs",
                "roles",
                "subject_ids",
                "description",
                "verification_basis",
            ),
        ),
    ]
}

_REGISTRY_UPDATE = _closed_object(
    {
        "operation": {"type": "string", "enum": ["append", "refine"]},
        "tag_ref": copy.deepcopy(_TAG_REF),
        "category": {
            "type": "string",
            "enum": ["person", "entity", "scene", "object", "style"],
        },
        "stable_description": {"type": "string", "minLength": 1},
        "aliases": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "evidence_client_refs": {
            "type": "array",
            "items": _id("client_ref of an add_memories entry in this transaction."),
            "minItems": 1,
            "uniqueItems": True,
        },
    },
    ("operation", "tag_ref", "category", "stable_description", "aliases", "evidence_client_refs"),
    all_of=[
        {
            "if": {"properties": {"operation": {"const": "append"}}, "required": ["operation"]},
            "then": {
                "properties": {
                    "tag_ref": {
                        "properties": {"kind": {"const": "new_tag"}},
                        "required": ["kind"],
                    }
                }
            },
            "else": {
                "properties": {
                    "tag_ref": {
                        "properties": {"kind": {"const": "existing_tag"}},
                        "required": ["kind"],
                    }
                }
            },
        }
    ],
)


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "view_material_registry": _tool(
        "view_material_registry",
        "Read the authoritative immutable material-registry snapshot for this shot. This must be the first tool action; it returns shot scope, story locks, canonical material rows, pending-upload handles with only natural user-declared labels/descriptions, budgets, capabilities, and frozen profiles, but no media bytes. Match natural upload declarations to registry aliases yourself; uploads do not reveal a private canonical binding.",
        _parameters(
            {
                "snapshot_id": _nullable(_id("Snapshot ID from an earlier page; null starts the shot.")),
                "cursor": _nullable(_id("Pagination cursor; null requests the first or final page as applicable.")),
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            ("snapshot_id", "cursor", "limit"),
        ),
    ),
    "search_memory_bank": _tool(
        "search_memory_bank",
        "Deterministically search committed registry and memory text. Results are summaries and handles only; any selected medium must subsequently be viewed.",
        _parameters(
            {
                "query": {"type": "string", "minLength": 1},
                "filters": _closed_object(
                    {
                        "tags": {"type": "array", "items": _id(), "uniqueItems": True},
                        "modalities": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["image", "audio", "video"]},
                            "uniqueItems": True,
                        },
                        "roles": {"type": "array", "items": _id(), "uniqueItems": True},
                        "subject_ids": {"type": "array", "items": _id(), "uniqueItems": True},
                    },
                    (),
                ),
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            ("query", "filters", "top_k"),
        ),
    ),
    "view_memories": _tool(
        "view_memories",
        "Load exact current-shot-authorized media for search results, uploads, or the previous tail. The observation carries real content blocks and immutable representation hashes; text descriptions never substitute for this call. Uploaded media retains natural declarations and never exposes its private backend canonical-subject binding, so derive any public ID_X from the registry and observed evidence.",
        _parameters(
            {
                "items": {
                    "type": "array",
                    "items": copy.deepcopy(_VIEW_ITEM),
                    "minItems": 1,
                    "maxItems": 32,
                }
            },
            ("items",),
        ),
    ),
    "get_pe_skill": _tool(
        "get_pe_skill",
        "In rewrite_enabled mode, resolve and return the exact frozen prompt-engineering skill bundle for image/video compose or revise. Image calls bind an immutable subject/asset generation scope; video calls use null. It never invokes a hidden model and never writes the prompt. This tool is unavailable in passthrough_disabled mode.",
        _parameters(
            {
                **copy.deepcopy(_PROMPT_ROUTE_PROPERTIES),
                "context_observation_ids": {
                    "type": "array",
                    "items": _id("Registry/view observation used by the prompt."),
                    "minItems": 1,
                    "uniqueItems": True,
                },
            },
            (*_PROMPT_ROUTE_REQUIRED[:4], "context_observation_ids", *_PROMPT_ROUTE_REQUIRED[4:]),
            one_of=_prompt_route_variants(),
        ),
    ),
    "register_prompt": _tool(
        "register_prompt",
        "Register an immutable context-and-generation-scope-bound prompt artifact. In rewrite_enabled mode, consume the matching get_pe_skill receipt and submit the complete agent-written prompt. In passthrough_disabled mode, image and video compose both skip get_pe_skill: submit the raw USER request byte-for-byte and set skill_application_receipt_id=null. This tool never rewrites, trims, normalizes, or truncates text. Its observation returns deterministic evaluation_contract_authoring guidance and a hash; use that guidance verbatim when authoring the immediately following generation evaluation contract.",
        _parameters(
            {
                **copy.deepcopy(_PROMPT_ROUTE_PROPERTIES),
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Exact bytes to be submitted to the generator. Runtime applies the "
                        "selected rewrite mode: official PE validation when enabled, or exact "
                        "raw-USER byte equality for passthrough image or video compose."
                    ),
                },
                "skill_application_receipt_id": _nullable(
                    _id("Receipt returned by the matching get_pe_skill call.")
                ),
                "evidence_ids": {
                    "type": "array",
                    "items": _id("Observation bound by the skill receipt."),
                    "minItems": 1,
                    "uniqueItems": True,
                },
            },
            (
                *_PROMPT_ROUTE_REQUIRED[:4],
                "text",
                "skill_application_receipt_id",
                "evidence_ids",
                *_PROMPT_ROUTE_REQUIRED[4:],
            ),
            one_of=_prompt_route_variants(),
        ),
    ),
    "generate_image": _tool(
        "generate_image",
        "Generate one real image candidate from a registered image prompt artifact. Its evaluation modules must include camera_composition and style_lighting plus every other applicable module. Starting a group creates a mandatory inspect/assess/select/consume obligation; retries must follow the prior assessment and declared changed axes.",
        _parameters(
            {
                "shot_request_id": _id(),
                "generation_group_id": _nullable(_id("Null starts a group; a prior group ID performs a semantic retry.")),
                "purpose": {"type": "string", "enum": ["first_frame", "reference_asset"]},
                "prompt_artifact_id": _id("Accepted image compose/revise artifact consumed by this request."),
                "subject_ids": {
                    "type": "array",
                    "items": _id("Stable subject ID."),
                    "uniqueItems": True,
                },
                "asset_role": _nullable(_id("Declared role of a reusable reference asset.")),
                "input_references": {
                    "type": "array",
                    "items": copy.deepcopy(_AUTHORIZED_IMAGE_SOURCE),
                    "uniqueItems": True,
                },
                "seed": {"type": "integer", "minimum": 0, "maximum": 9223372036854775807},
                "evaluation_contract": copy.deepcopy(_EVALUATION_CONTRACT),
                "retry": {"oneOf": [{"type": "null"}, copy.deepcopy(_IMAGE_RETRY)]},
            },
            (
                "shot_request_id",
                "generation_group_id",
                "purpose",
                "prompt_artifact_id",
                "subject_ids",
                "asset_role",
                "input_references",
                "seed",
                "evaluation_contract",
                "retry",
            ),
            all_of=[
                {
                    "if": {"properties": {"generation_group_id": {"type": "null"}}, "required": ["generation_group_id"]},
                    "then": {"properties": {"retry": {"type": "null"}}},
                    "else": {"properties": {"retry": copy.deepcopy(_IMAGE_RETRY)}},
                },
                {
                    "if": {"properties": {"purpose": {"const": "first_frame"}}, "required": ["purpose"]},
                    "then": {"properties": {"asset_role": {"type": "null"}}},
                    "else": {"properties": {"asset_role": _id("Role the reusable reference asset must depict.")}},
                },
            ],
        ),
    ),
    "generate_shot": _tool(
        "generate_shot",
        "Generate one real Echo shot candidate from an immutable video prompt artifact, optional typed frame-zero condition, and zero to seven ordered memory slots. Its evaluation modules must include camera_composition, motion_temporal, speech_audio, and the category module for every active material: identity_wardrobe for person/entity, scene_environment for scene, object_state for object, and style_lighting for style. The tool does not judge semantic quality.",
        _parameters(
            {
                "shot_request_id": _id(),
                "generation_group_id": _nullable(_id("Null starts the video group; a prior group ID retries it.")),
                "execution_prompt_artifact_id": _id("Accepted video compose/revise artifact consumed by this request."),
                "condition": copy.deepcopy(_CONDITION_OR_NULL),
                "memory_slots": {
                    "type": "array",
                    "items": copy.deepcopy(_MEMORY_SLOT),
                    "maxItems": 7,
                },
                "consumed_image_selection_ids": {
                    "type": "array",
                    "items": _id("Image selection consumed in its frozen primary use."),
                    "uniqueItems": True,
                },
                "seed": {"type": "integer", "minimum": 0, "maximum": 9223372036854775807},
                "evaluation_contract": copy.deepcopy(_EVALUATION_CONTRACT),
                "retry": {"oneOf": [{"type": "null"}, copy.deepcopy(_VIDEO_RETRY)]},
            },
            (
                "shot_request_id",
                "generation_group_id",
                "execution_prompt_artifact_id",
                "condition",
                "memory_slots",
                "consumed_image_selection_ids",
                "seed",
                "evaluation_contract",
                "retry",
            ),
            all_of=[
                {
                    "if": {"properties": {"generation_group_id": {"type": "null"}}, "required": ["generation_group_id"]},
                    "then": {"properties": {"retry": {"type": "null"}}},
                    "else": {"properties": {"retry": copy.deepcopy(_VIDEO_RETRY)}},
                }
            ],
        ),
    ),
    "inspect_candidate": _tool(
        "inspect_candidate",
        "Return the real candidate media, comparison pool, exact generation inputs, frozen checklist, deterministic assessment contract, capability mask, and remaining semantic-attempt budget. Copy the assessment contract's module applicability and forced check results exactly; the tool performs no visual or acoustic judgment.",
        _parameters(
            {
                "candidate_id": _id("Decodable image or video candidate ID."),
                "requested_uses": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["condition", "memory_slot", "final_video", "memory_writeback"],
                    },
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "compare_pool": {"type": "boolean"},
            },
            ("candidate_id", "requested_uses", "compare_pool"),
        ),
    ),
    "submit_candidate_assessment": _tool(
        "submit_candidate_assessment",
        "Submit the agent's complete immutable assessment for one inspection, copying deterministic module applicability and forced check results exactly from its assessment contract, then either authorize a concrete retry or select the best assessed candidate. Follow assessment_contract.selection_mode_guidance for the selected candidate: fail/uncertain maps to best_effort and is selectable only at zero remaining attempts, blocked without failure maps to capability_limited, and all-pass maps to requirements_met; never rank a lower-quality candidate above an available higher-quality one. The tool checks coverage, lineage, budget, and state but makes no perceptual judgment.",
        _parameters(
            {
                "inspection_id": _id(),
                "inspection_revision_hash": {
                    "type": "string",
                    "pattern": r"^sha256:[0-9a-f]{64}$",
                },
                "module_applicability": {
                    "type": "array",
                    "description": "Exact ordered module_applicability array returned by inspect_candidate.assessment_contract.",
                    "items": _closed_object(
                        {
                            "module": {"type": "string", "enum": list(_SEMANTIC_MODULES)},
                            "applicability": {"type": "string", "enum": ["applicable", "not_applicable"]},
                            "reason": {"type": "string", "minLength": 1},
                        },
                        ("module", "applicability", "reason"),
                    ),
                    "uniqueItems": True,
                },
                "check_results": {
                    "type": "array",
                    "description": "One result per frozen check. Rows named by assessment_contract.forced_check_results must be copied exactly; judge all remaining rows from allowed evidence.",
                    "items": _closed_object(
                        {
                            "check_id": _id(),
                            "result": {
                                "type": "string",
                                "enum": ["pass", "fail", "uncertain", "not_applicable", "capability_blocked"],
                            },
                            "evidence_ids": {
                                "type": "array",
                                "items": _id("Evidence ID allowed by the frozen checklist."),
                                "uniqueItems": True,
                            },
                            "rationale": {"type": "string", "minLength": 1},
                        },
                        ("check_id", "result", "evidence_ids", "rationale"),
                    ),
                },
                "use_results": {
                    "type": "array",
                    "items": _closed_object(
                        {
                            "use_contract_id": _id(),
                            "use": {
                                "type": "string",
                                "enum": ["condition", "memory_slot", "final_video", "memory_writeback"],
                            },
                            "result": {
                                "type": "string",
                                "enum": ["pass", "fail", "uncertain", "capability_blocked"],
                            },
                            "source_ids": {"type": "array", "items": _id(), "minItems": 1, "uniqueItems": True},
                            "evidence_ids": {"type": "array", "items": _id(), "uniqueItems": True},
                            "rationale": {"type": "string", "minLength": 1},
                        },
                        ("use_contract_id", "use", "result", "source_ids", "evidence_ids", "rationale"),
                    ),
                },
                "next_action": {"type": "string", "enum": ["retry", "select"]},
                "retry_change": {
                    "oneOf": [
                        {"type": "null"},
                        _closed_object(
                            {
                                "axes": {
                                    "anyOf": [copy.deepcopy(_IMAGE_AXES), copy.deepcopy(_VIDEO_AXES)]
                                },
                                "reason": {"type": "string", "minLength": 1},
                                "failed_check_ids": {
                                    "type": "array",
                                    "items": _id(),
                                    "minItems": 1,
                                    "uniqueItems": True,
                                },
                            },
                            ("axes", "reason", "failed_check_ids"),
                        ),
                    ]
                },
                "selection": {
                    "oneOf": [
                        {"type": "null"},
                        _closed_object(
                            {
                                "mode": {
                                    "type": "string",
                                    "enum": ["requirements_met", "capability_limited", "best_effort"],
                                },
                                "selected_candidate_id": _id(),
                                "ranked_candidate_ids": {
                                    "type": "array",
                                    "items": _id(),
                                    "minItems": 1,
                                    "uniqueItems": True,
                                },
                                "reason": {"type": "string", "minLength": 1},
                            },
                            ("mode", "selected_candidate_id", "ranked_candidate_ids", "reason"),
                        ),
                    ]
                },
            },
            (
                "inspection_id",
                "inspection_revision_hash",
                "module_applicability",
                "check_results",
                "use_results",
                "next_action",
                "retry_change",
                "selection",
            ),
            one_of=[
                {
                    "properties": {
                        "next_action": {"const": "retry"},
                        "retry_change": _closed_object(
                            {
                                "axes": {"anyOf": [copy.deepcopy(_IMAGE_AXES), copy.deepcopy(_VIDEO_AXES)]},
                                "reason": {"type": "string", "minLength": 1},
                                "failed_check_ids": {
                                    "type": "array",
                                    "items": _id(),
                                    "minItems": 1,
                                    "uniqueItems": True,
                                },
                            },
                            ("axes", "reason", "failed_check_ids"),
                        ),
                        "selection": {"type": "null"},
                    }
                },
                {
                    "properties": {
                        "next_action": {"const": "select"},
                        "retry_change": {"type": "null"},
                        "selection": _closed_object(
                            {
                                "mode": {
                                    "type": "string",
                                    "enum": ["requirements_met", "capability_limited", "best_effort"],
                                },
                                "selected_candidate_id": _id(),
                                "ranked_candidate_ids": {
                                    "type": "array",
                                    "items": _id(),
                                    "minItems": 1,
                                    "uniqueItems": True,
                                },
                                "reason": {"type": "string", "minLength": 1},
                            },
                            ("mode", "selected_candidate_id", "ranked_candidate_ids", "reason"),
                        ),
                    }
                },
            ],
        ),
    ),
    "update_memory_bank": _tool(
        "update_memory_bank",
        "Stage an evidence-backed, typed long-term memory and text-registry transaction for the selected video. Image and audio are separate authorizations; this call does not finish the shot.",
        _parameters(
            {
                "video_selection_id": _id("Selected final-video authorization."),
                "add_memories": {
                    "type": "array",
                    "items": copy.deepcopy(_ADD_MEMORY),
                    "minItems": 1,
                },
                "registry_updates": {
                    "type": "array",
                    "items": copy.deepcopy(_REGISTRY_UPDATE),
                },
            },
            ("video_selection_id", "add_memories", "registry_updates"),
        ),
    ),
    "submit_shot": _tool(
        "submit_shot",
        "Sole terminal action. Atomically commit the selected video, either commit its staged memory transaction or explicitly skip memory update, save the tail, and persist the complete canonical trajectory.",
        _parameters(
            {
                "video_selection_id": _id("Selection returned by submit_candidate_assessment."),
                "memory_update": {
                    "oneOf": [
                        _closed_object(
                            {
                                "status": {"const": "updated"},
                                "memory_update_id": _id("Staged transaction returned by update_memory_bank."),
                                "reason": {"type": "string", "minLength": 1},
                            },
                            ("status", "memory_update_id", "reason"),
                        ),
                        _closed_object(
                            {
                                "status": {"const": "not_needed"},
                                "memory_update_id": {"type": "null"},
                                "reason": {"type": "string", "minLength": 1},
                            },
                            ("status", "memory_update_id", "reason"),
                        ),
                    ]
                },
                "summary": {"type": "string", "minLength": 1},
            },
            ("video_selection_id", "memory_update", "summary"),
        ),
    ),
}


def _validate_static_contract() -> None:
    if tuple(TOOL_SCHEMAS) != CANONICAL_TOOL_NAMES:
        raise RuntimeError("TOOL_SCHEMAS order must match CANONICAL_TOOL_NAMES")
    for expected_name, tool in TOOL_SCHEMAS.items():
        if tool.get("type") != "function" or tool.get("function", {}).get("name") != expected_name:
            raise RuntimeError(f"Malformed tool schema for {expected_name}")


_validate_static_contract()


def canonical_tool_names(names: Iterable[str] | None = None) -> tuple[str, ...]:
    """Validate a tool subset and return it in canonical catalog order."""

    if names is None:
        return CANONICAL_TOOL_NAMES
    requested = tuple(str(name) for name in names)
    if not requested:
        raise ValueError("at least one canonical tool name is required")
    if len(requested) != len(set(requested)):
        raise ValueError("canonical tool names must not contain duplicates")
    unknown = sorted(set(requested) - set(CANONICAL_TOOL_NAMES))
    if unknown:
        raise ValueError(f"unknown canonical tool names: {unknown}")
    return tuple(name for name in CANONICAL_TOOL_NAMES if name in requested)


def get_tool_schemas(names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Return a validated canonical tool subset as independent deep copies."""

    return [copy.deepcopy(TOOL_SCHEMAS[name]) for name in canonical_tool_names(names)]


def tool_schema_hash(names: Iterable[str] | None = None) -> str:
    """Return the stable digest of a canonical ``tools[]`` subset."""

    payload = json.dumps(
        get_tool_schemas(names),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


TOOL_SCHEMA_SHA256 = tool_schema_hash()


__all__ = [
    "ASSESSMENT_CONTRACT_SCHEMA_VERSION",
    "APPLICABLE_MODULE_REASON",
    "AUDIO_CAPABILITY_BLOCKED_RATIONALE",
    "AUDIO_TRANSPORT_ONLY_USE_RATIONALE",
    "CANONICAL_TOOL_NAMES",
    "EVALUATION_CONTRACT_AUTHORING_SCHEMA_VERSION",
    "JSON_SCHEMA_DIALECT",
    "PROMPT_REWRITE_MODES",
    "NOT_APPLICABLE_MODULE_REASON",
    "SEMANTIC_MODULES",
    "TOOL_PROTOCOL_VERSION",
    "TOOL_SCHEMAS",
    "TOOL_SCHEMA_SHA256",
    "build_assessment_contract",
    "build_evaluation_contract_authoring",
    "evaluation_check_evidence_kinds",
    "canonical_tool_names",
    "get_tool_schemas",
    "tool_schema_hash",
]
