"""Canonical tool contract for the model-independent StoryMem memory policy.

This module is dependency-free and side-effect-free.  It defines only the
model-visible memory action space; media I/O, model clients, and compilation
into an implementation's concrete generation inputs live outside this
package.

Each ``function.parameters`` value is an independent JSON Schema Draft
2020-12 document.  Every object is closed, and every property is required;
optional values are represented explicitly with ``null``.  Stateful rules
such as registry-first and view-before-use are enforced by the policy runtime.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from typing import Any

from .models import (
    FALLBACK_POLICIES,
    IMAGE_GENERATION_PURPOSES,
    MODALITIES,
    SELECTION_MODES,
    SOURCE_KINDS,
    TIMELINE_ROLES,
)
from .safety import NEUTRAL_SEMANTIC_EFFECTS


TOOL_PROTOCOL_VERSION = "storymem.memory-policy.tools.v2"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

CANONICAL_TOOL_NAMES: tuple[str, ...] = (
    "view_material_registry",
    "search_memory_bank",
    "view_memories",
    "generate_image",
    "inspect_candidate",
    "submit_candidate_assessment",
    "update_memory_bank",
    "submit_evidence_plan",
    "finish_memory_session",
)


def _closed_object(
    properties: dict[str, Any],
    required: Iterable[str] | None = None,
    *,
    description: str | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties if required is None else required),
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    return schema


def _id(description: str) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "description": description}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"oneOf": [copy.deepcopy(schema), {"type": "null"}]}


def _token(description: str) -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": "^[a-z][a-z0-9_.:-]*$",
        "description": description,
    }


def _sha256(description: str) -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": "^sha256:[0-9a-f]{64}$",
        "description": description,
    }


def _unique_string_array(
    description: str,
    *,
    min_items: int = 0,
    token: bool = False,
) -> dict[str, Any]:
    item = _token("One semantic token.") if token else _id("One exact value.")
    return {
        "type": "array",
        "items": item,
        "minItems": min_items,
        "uniqueItems": True,
        "description": description,
    }


def _evidence_reference(
    *,
    description: str,
    modalities: Iterable[str] = MODALITIES,
) -> dict[str, Any]:
    allowed_modalities = tuple(modalities)
    return _closed_object(
        {
            "source_kind": {
                "type": "string",
                "enum": list(SOURCE_KINDS),
                "description": "Origin category of the evidence.",
            },
            "evidence_id": _id("Opaque evidence source ID."),
            "representation_id": _id("Exact representation selected for use."),
            "authorization_id": _id(
                "Current-session authorization returned by a view or selection."
            ),
            "modality": {"type": "string", "enum": list(allowed_modalities)},
            "semantic_roles": _unique_string_array(
                "Roles this representation is intended to support.",
                min_items=1,
                token=True,
            ),
            "subject_ids": _unique_string_array(
                "Opaque stable subject IDs supported by this representation."
            ),
        },
        description=description,
    )


def _requirement_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "requirement_id": _id("Unique requirement ID within this plan."),
            "description": _id("Observable semantic need from the user request."),
            "semantic_roles": _unique_string_array(
                "Relevant roles such as identity, wardrobe, scene, object_state, "
                "style, interaction, motion, or voice.",
                min_items=1,
                token=True,
            ),
            "subject_ids": _unique_string_array(
                "Opaque stable subjects covered by this requirement."
            ),
            "acceptable_modalities": {
                "type": "array",
                "items": {"type": "string", "enum": list(MODALITIES)},
                "minItems": 1,
                "uniqueItems": True,
            },
            "priority": {"type": "integer", "minimum": 0, "maximum": 100},
            "required": {"type": "boolean"},
        },
        description="One evidence requirement, independent of any concrete input layout.",
    )


def _binding_schema() -> dict[str, Any]:
    properties = _evidence_reference(
        description="One exact, authorized evidence representation."
    )["properties"]
    properties["subject_ids"]["description"] = (
        "Complete canonical subject_ids array for this exact evidence, copied "
        "verbatim and in order; subject-subset bindings are invalid."
    )
    return _closed_object(
        {
            "binding_id": _id("Unique binding ID within this plan."),
            "requirement_ids": _unique_string_array(
                "All compatible requirements covered by this one physical evidence "
                "representation. The same source_kind/evidence_id/representation_id "
                "must not appear in another binding.",
                min_items=1,
            ),
            **copy.deepcopy(properties),
            "intended_effect": {
                "type": "string",
                "enum": sorted(NEUTRAL_SEMANTIC_EFFECTS),
                "description": (
                    "Implementation-neutral semantic effect requested from the "
                    "downstream compiler. Adapter-native model, channel, slot, and "
                    "workflow labels are forbidden."
                ),
            },
            "timeline_role": {
                "type": "string",
                "enum": list(TIMELINE_ROLES),
            },
            "rank": {"type": "integer", "minimum": 1},
            "priority": {"type": "integer", "minimum": 0, "maximum": 100},
            "required": {"type": "boolean"},
            "fallback": {"type": "string", "enum": list(FALLBACK_POLICIES)},
        },
        description=(
            "A priority-ranked semantic binding. Rank 1 is considered first; "
            "the runtime validates capacity and fallback consistency."
        ),
    )


def _parameters(title: str, schema: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(schema)
    value["$schema"] = JSON_SCHEMA_DIALECT
    value["title"] = title
    return value


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": _parameters(f"{name} arguments", parameters),
        },
    }


_VIEW_MATERIAL_REGISTRY = _tool(
    "view_material_registry",
    "Read the authoritative request ID and raw-request hash, text registry, uploaded-material handles, discoverable prior sequence-context representations, and current condition budget. This reveals safe metadata, never media paths or media content.",
    _closed_object(
        {
            "snapshot_id": _nullable(_id("Snapshot ID returned by the first page.")),
            "cursor": _nullable(_id("Exact next-page cursor returned by the prior page.")),
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        }
    ),
)

_SEARCH_MEMORY_BANK = _tool(
    "search_memory_bank",
    "Retrieve memory summaries using a concrete query derived from the request and registry. Search results are discovery metadata and must be viewed before use.",
    _closed_object(
        {
            "snapshot_id": _id("Completed registry snapshot ID for this session."),
            "query": {"type": "string", "minLength": 1, "maxLength": 2000},
            "filters": _closed_object(
                {
                    "tags": _unique_string_array(
                        "All listed tags must be present on a result."
                    ),
                    "modalities": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(MODALITIES)},
                        "uniqueItems": True,
                    },
                    "semantic_roles": _unique_string_array(
                        "All listed semantic roles must be present on a result.",
                        token=True,
                    ),
                    "timeline_roles": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(TIMELINE_ROLES),
                        },
                        "uniqueItems": True,
                        "description": (
                            "A result's single timeline role must be one of these values."
                        ),
                    },
                }
            ),
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
        }
    ),
)

_VIEW_MEMORIES = _tool(
    "view_memories",
    "Expose exact representations for discovered memory, uploaded material, or sequence context. Returned authorizations are required for later use.",
    _closed_object(
        {
            "snapshot_id": _id("Completed registry snapshot ID for this session."),
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 64,
                "items": _closed_object(
                    {
                        "source_kind": {
                            "type": "string",
                            "enum": list(SOURCE_KINDS),
                        },
                        "source_id": _id("Exact ID returned by registry or search."),
                        "discovery_id": _id(
                            "Search receipt or registry snapshot that exposed this ID."
                        ),
                        "representation_id": _nullable(
                            _id("Exact representation to view, or null for the default.")
                        ),
                        "time_range": _nullable(
                            _closed_object(
                                {
                                    "start_ms": {"type": "integer", "minimum": 0},
                                    "end_ms": {
                                        "type": "integer",
                                        "minimum": 1,
                                    },
                                },
                                description=(
                                    "Optional half-open audio/video range in integer "
                                    "milliseconds. The runtime "
                                    "materializes and authorizes exact derived bytes."
                                ),
                            )
                        ),
                    }
                ),
            },
        }
    ),
)

_GENERATE_IMAGE = _tool(
    "generate_image",
    "Create optional visual evidence from a neutral visual brief and only previously authorized input evidence. The tool does not create moving media.",
    _closed_object(
        {
            "purpose": {
                "type": "string",
                "enum": list(IMAGE_GENERATION_PURPOSES),
            },
            "visual_brief": {
                "type": "string",
                "minLength": 1,
                "maxLength": 12000,
                "description": "Implementation-neutral description of the image to create.",
            },
            "input_evidence": {
                "type": "array",
                "items": _evidence_reference(
                    description="Viewed image evidence authorized as an image-generation input.",
                    modalities=("image",),
                ),
                "uniqueItems": True,
                "description": (
                    "Previously viewed image representations selected as inputs for "
                    "creating this evidence."
                ),
            },
            "requirement_ids": _unique_string_array(
                "Provisional requirements this image is intended to cover.", min_items=1
            ),
            "subject_ids": _unique_string_array(
                "Opaque registry subject IDs the created evidence should preserve."
            ),
            "semantic_roles": _unique_string_array(
                "Roles intended for the created evidence; reference evidence normally "
                "declares at least one role.",
                token=True,
            ),
        }
    ),
)

_INSPECT_CANDIDATE = _tool(
    "inspect_candidate",
    "Return a policy-created candidate or host-attached generation-derived candidate, its lineage, a frozen checklist, and allowed comparison evidence. This supports both evidence selection and post-generation memory writeback.",
    _closed_object(
        {
            "candidate_id": _id(
                "Policy-created or host-attached generation-derived candidate ID."
            ),
            "requested_uses": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "opening_frame_evidence",
                        "reference_evidence",
                        "condition_adherence",
                        "memory_writeback",
                    ],
                },
                "minItems": 1,
                "uniqueItems": True,
            },
            "comparison_evidence_ids": _unique_string_array(
                "Viewed evidence IDs needed for direct comparison."
            ),
        }
    ),
)

_ASSESSMENT_RESULT = {
    "type": "string",
    "enum": ["pass", "fail", "uncertain", "capability_blocked"],
}

_SUBMIT_CANDIDATE_ASSESSMENT = _tool(
    "submit_candidate_assessment",
    "Record the policy's evidence-grounded judgment of one inspected candidate. Policy-created images may be selected or recreated from changed inputs. A host-attached generation candidate may cause a bounded evidence-plan revision only for a memory-condition failure; an unchanged condition plan can never trigger another moving-media generation.",
    _closed_object(
        {
            "inspection_id": _id("Inspection receipt that supplied the checklist."),
            "candidate_id": _id("Candidate assessed in this call."),
            "check_results": {
                "type": "array",
                "minItems": 0,
                "items": _closed_object(
                    {
                        "check_id": _id("Exact checklist item ID."),
                        "result": copy.deepcopy(_ASSESSMENT_RESULT),
                        "evidence_ids": _unique_string_array(
                            "Viewed or inspected evidence supporting the result."
                        ),
                        "rationale": _id("Concise observable justification."),
                    }
                ),
            },
            "use_results": {
                "type": "array",
                "minItems": 1,
                "items": _closed_object(
                    {
                        "use_id": _id("Exact requested-use ID from inspection."),
                        "result": copy.deepcopy(_ASSESSMENT_RESULT),
                        "evidence_ids": _unique_string_array(
                            "Evidence supporting suitability for this use."
                        ),
                        "rationale": _id("Concise observable justification."),
                    }
                ),
            },
            "assessment_scope": {
                "type": "string",
                "enum": [
                    "created_evidence_suitability",
                    "condition_adherence",
                    "memory_writeback",
                ],
            },
            "assessment_result": {
                "type": "string",
                "enum": ["pass", "capability_limited", "fail"],
            },
            "decision": {
                "type": "string",
                "enum": [
                    "select",
                    "retry",
                    "accept_conditions",
                    "revise_evidence_plan",
                    "cannot_fix_with_memory",
                ],
            },
            "candidate_ranking": _unique_string_array(
                "All assessed candidates in the relevant image group or generation history, best first."
            ),
            "selected_candidate_id": _nullable(
                _id(
                    "Selected assessed candidate. Required for image select, "
                    "accept_conditions, and cannot_fix_with_memory."
                )
            ),
            "retry_instruction": _nullable(
                _id(
                    "Concrete correction for the next image attempt or evidence-plan "
                    "revision; null when no policy action can help."
                )
            ),
            "changed_axes": _unique_string_array(
                "Concrete image-input or reference-condition axes that must change.",
                token=True,
            ),
        }
    ),
)

_MEMORY_RECORD = _closed_object(
    {
        "record_id": _nullable(_id("Existing record ID, or null to create one.")),
        "source_kind": {
            "type": "string",
            "enum": [
                "uploaded_material",
                "generated_evidence",
                "generation_derived",
            ],
        },
        "source_evidence_id": _id("Viewed or selected evidence source ID."),
        "representation_id": _id("Exact reusable representation ID."),
        "authorization_id": _id("Authorization for this writeback."),
        "modality": {"type": "string", "enum": list(MODALITIES)},
        "source_selector": _nullable(
            _closed_object(
                {
                    "mode": {"type": "string", "const": "time_range"},
                    "start_ms": {"type": "integer", "minimum": 0},
                    "end_ms": {"type": "integer", "minimum": 1},
                },
                description=(
                    "Exact source time range. Required for audio writeback and "
                    "null for non-audio records."
                ),
            )
        ),
        "semantic_roles": _unique_string_array(
            "Stable roles supported by the evidence.", min_items=1, token=True
        ),
        "timeline_role": {"type": "string", "enum": list(TIMELINE_ROLES)},
        "tags": _unique_string_array("Existing registry tags linked to the record."),
        "description": _id(
            "Stable searchable description; exclude transient events. For uploaded "
            "material this must exactly copy the description exposed by view_memories."
        ),
        "stable_facts": _unique_string_array(
            "Evidence-backed facts safe to reuse in later requests. For uploaded "
            "material every value must come from trusted_stable_facts exposed by "
            "view_memories; use [] when no such fact was exposed."
        ),
        "reuse_scope": {
            "type": "string",
            "enum": ["session", "story", "project"],
        },
        "writeback_evidence_ids": _unique_string_array(
            "Exact provenance IDs authorizing stable storage. For uploaded material "
            "this is exactly [source_evidence_id]. For selected generated evidence "
            "copy exactly the writeback_evidence_ids returned by its candidate "
            "inspection after the required assessment passes.",
            min_items=1,
        ),
    },
    description=(
        "One durable record. Uploaded records copy trusted view metadata exactly; "
        "selected generated records copy the passing assessment's writeback "
        "authorization."
    ),
)

_REGISTRY_UPDATE = _closed_object(
    {
        "subject_id": _id("Exact existing opaque registry subject ID to update."),
        "description": _id("Stable evidence-backed tag description."),
        "aliases": _unique_string_array("Stable user-facing aliases."),
        "supporting_evidence_ids": _unique_string_array(
            "Viewed evidence supporting this text update.", min_items=1
        ),
    }
)

_UPDATE_MEMORY_BANK = _tool(
    "update_memory_bank",
    "Stage the single immutable transaction containing every stable, reusable, "
    "evidence-backed record and registry update to save. Writeback is optional "
    "except when an accepted reusable trusted uploaded voice has no equivalent "
    "committed audio memory. The host already carries the accepted tail frame as "
    "next-shot sequence context, so do not duplicate it merely for adjacent-shot "
    "continuity. This action does not finish the session.",
    _closed_object(
        {
            "records": {
                "type": "array",
                "items": copy.deepcopy(_MEMORY_RECORD),
                "uniqueItems": True,
            },
            "registry_updates": {
                "type": "array",
                "items": copy.deepcopy(_REGISTRY_UPDATE),
                "uniqueItems": True,
            },
            "rationale": _id("Why these durable updates are justified."),
        }
    ),
)

_SUBMIT_EVIDENCE_PLAN = _tool(
    "submit_evidence_plan",
    "Submit the complete semantic evidence plan for budget and causal validation. The plan ranks meaning and intended effects, never concrete implementation channels.",
    _closed_object(
        {
            "request_id": _id("Current immutable request ID."),
            "raw_request_sha256": _sha256(
                "Exact raw-request digest returned by the registry."
            ),
            "condition_budget_id": _id(
                "Exact condition budget observed in the registry."
            ),
            "parent_plan_id": _nullable(
                _id("Accepted parent plan ID for a revision, or null for version 1.")
            ),
            "version": {"type": "integer", "minimum": 1},
            "requirements": {
                "type": "array",
                "items": _requirement_schema(),
                "uniqueItems": True,
            },
            "ranked_bindings": {
                "type": "array",
                "items": _binding_schema(),
                "uniqueItems": True,
                "description": (
                    "Bindings ordered by ascending rank. Each exact physical evidence "
                    "representation may occur once only; aggregate all compatible "
                    "requirement IDs and semantic roles into that binding. Capacity is "
                    "validated against the observed condition budget at runtime."
                ),
            },
            "unfulfilled_requirement_ids": _unique_string_array(
                "Required needs that cannot be covered by current evidence and budget."
            ),
            "selection_mode": {"type": "string", "enum": list(SELECTION_MODES)},
            "rationale": _id("Concise explanation of coverage and tradeoffs."),
        }
    ),
)

_FINISH_MEMORY_SESSION = _tool(
    "finish_memory_session",
    "Sole terminal action. Commit one accepted evidence plan and either its staged memory transaction or an explicit no-update decision. An adopted reusable trusted uploaded voice must already have equivalent committed or staged audio memory.",
    _closed_object(
        {
            "accepted_plan_id": _id("Plan ID returned by submit_evidence_plan."),
            "memory_transaction_id": _nullable(
                _id("Staged memory transaction ID, or null when no update is needed.")
            ),
            "writeback_disposition": {
                "type": "string",
                "enum": ["commit_staged", "not_needed"],
            },
            "rationale": _id("Concise terminal audit summary."),
        }
    ),
)


TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    _VIEW_MATERIAL_REGISTRY,
    _SEARCH_MEMORY_BANK,
    _VIEW_MEMORIES,
    _GENERATE_IMAGE,
    _INSPECT_CANDIDATE,
    _SUBMIT_CANDIDATE_ASSESSMENT,
    _UPDATE_MEMORY_BANK,
    _SUBMIT_EVIDENCE_PLAN,
    _FINISH_MEMORY_SESSION,
)

_TOOL_BY_NAME = {item["function"]["name"]: item for item in TOOL_SCHEMAS}
if tuple(_TOOL_BY_NAME) != CANONICAL_TOOL_NAMES or len(_TOOL_BY_NAME) != len(
    TOOL_SCHEMAS
):
    raise RuntimeError("canonical memory-policy tool order is inconsistent")


def canonical_tool_names(tool_names: Iterable[str] | None = None) -> tuple[str, ...]:
    """Validate a requested subset and return it in canonical protocol order."""

    if tool_names is None:
        return CANONICAL_TOOL_NAMES
    requested = tuple(tool_names)
    if not requested:
        raise ValueError("at least one tool name is required")
    if len(set(requested)) != len(requested):
        raise ValueError("tool names must not contain duplicates")
    unknown = set(requested) - set(CANONICAL_TOOL_NAMES)
    if unknown:
        raise ValueError(f"unknown memory-policy tools: {sorted(unknown)}")
    requested_set = set(requested)
    return tuple(name for name in CANONICAL_TOOL_NAMES if name in requested_set)


def get_tool_schemas(
    tool_names: Iterable[str] | None = None,
    *,
    prompt_authoring: bool = False,
) -> list[dict[str, Any]]:
    """Return a defensive copy of all tools or a validated canonical subset."""

    names = canonical_tool_names(tool_names)
    schemas = copy.deepcopy([_TOOL_BY_NAME[name] for name in names])
    if prompt_authoring:
        for schema in schemas:
            function = schema["function"]
            if function["name"] != "submit_evidence_plan":
                continue
            function["description"] += " Also author a complete generation prompt; the adapter still performs prompt enhancement."
            parameters = function["parameters"]
            parameters["properties"]["generation_prompt"] = {
                "type": "string", "minLength": 10, "maxLength": 24000,
                "description": "Complete natural video prompt preserving the immutable user's intent. No tool IDs or backend layout. Not a patch instruction.",
            }
            parameters["properties"]["attempt_kind"] = {
                "type": "string", "enum": ["initial", "repair", "resample"],
                "description": "initial for version 1; repair changes evidence and/or prompt after observed failure; resample keeps both unchanged and requests a new host-owned random seed.",
            }
            parameters["required"].extend(["generation_prompt", "attempt_kind"])
    return schemas


def tool_schema_hash(tool_names: Iterable[str] | None = None, *, prompt_authoring: bool = False) -> str:
    """Return the stable SHA-256 identity of a canonical tool-set snapshot."""

    encoded = json.dumps(
        get_tool_schemas(tool_names, prompt_authoring=prompt_authoring),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


TOOL_SCHEMA_SHA256 = tool_schema_hash()


__all__ = [
    "CANONICAL_TOOL_NAMES",
    "JSON_SCHEMA_DIALECT",
    "TOOL_PROTOCOL_VERSION",
    "TOOL_SCHEMAS",
    "TOOL_SCHEMA_SHA256",
    "canonical_tool_names",
    "get_tool_schemas",
    "tool_schema_hash",
]
