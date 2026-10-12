from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.storymem_policy import (
    CANONICAL_TOOL_NAMES,
    JSON_SCHEMA_DIALECT,
    OPEN_MODEL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TOOL_SCHEMA_SHA256,
    ConditionBudget,
    EvidenceBinding,
    EvidenceRequirement,
    EvidencePlan,
    get_tool_schemas,
    tool_schema_hash,
)


EXPECTED_TOOLS = (
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


def _parameters(name: str) -> dict:
    tools = {item["function"]["name"]: item for item in get_tool_schemas()}
    return tools[name]["function"]["parameters"]


def _assert_valid(name: str, instance: dict) -> None:
    Draft202012Validator(_parameters(name)).validate(instance)


def _assert_invalid(name: str, instance: dict) -> None:
    with pytest.raises(Exception):
        Draft202012Validator(_parameters(name)).validate(instance)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _budget(**overrides) -> ConditionBudget:
    values = {
        "budget_id": "budget_portable_v1",
        "max_total_conditions": 4,
        "max_image_conditions": 2,
        "max_video_conditions": 1,
        "max_audio_conditions": 1,
        "max_text_conditions": 1,
    }
    values.update(overrides)
    return ConditionBudget(**values)


def _requirement(
    requirement_id: str = "req_identity",
    *,
    role: str = "identity",
    modality: str = "image",
    required: bool = True,
    subject_ids=("subject_main",),
) -> EvidenceRequirement:
    return EvidenceRequirement(
        requirement_id=requirement_id,
        description=f"Preserve {role}",
        semantic_roles=[role],
        subject_ids=subject_ids,
        acceptable_modalities=[modality],
        priority=90,
        required=required,
    )


def _binding(
    binding_id: str = "bind_identity",
    *,
    requirement_id: str = "req_identity",
    role: str = "identity",
    modality: str = "image",
    effect: str = "identity_reference",
    representation_id: str = "repr_identity",
    rank: int = 1,
    required: bool = True,
    fallback: str = "forbidden",
    source_kind: str = "retrieved_memory",
    subject_ids=("subject_main",),
    timeline_role: str | None = None,
) -> EvidenceBinding:
    return EvidenceBinding(
        binding_id=binding_id,
        requirement_ids=[requirement_id],
        source_kind=source_kind,
        evidence_id=f"evidence_{binding_id}",
        representation_id=representation_id,
        authorization_id=f"authorization_{binding_id}",
        modality=modality,
        semantic_roles=[role],
        subject_ids=subject_ids,
        intended_effect=effect,
        timeline_role=(
            timeline_role
            if timeline_role is not None
            else "opening" if effect == "exact_opening_state" else "timeless"
        ),
        rank=rank,
        priority=90,
        required=required,
        fallback=fallback,
    )


def _plan(
    *,
    requirements=None,
    bindings=None,
    unfulfilled=(),
    selection_mode: str = "requirements_met",
) -> EvidencePlan:
    return EvidencePlan(
        plan_id="plan_001",
        parent_plan_id=None,
        version=1,
        request_id="request_001",
        raw_request_sha256="sha256:" + "a" * 64,
        condition_budget_id="budget_portable_v1",
        requirements=[_requirement()] if requirements is None else requirements,
        ranked_bindings=[_binding()] if bindings is None else bindings,
        unfulfilled_requirement_ids=unfulfilled,
        selection_mode=selection_mode,
        rationale="Identity evidence is the highest-value binding.",
    )


def test_exact_nine_tool_contract_and_schema_dialect() -> None:
    assert CANONICAL_TOOL_NAMES == EXPECTED_TOOLS
    schemas = get_tool_schemas()
    assert len(schemas) == 9
    assert tuple(item["function"]["name"] for item in schemas) == EXPECTED_TOOLS
    for tool in schemas:
        assert tool["type"] == "function"
        assert tool["function"]["strict"] is True
        parameters = tool["function"]["parameters"]
        assert parameters["$schema"] == JSON_SCHEMA_DIALECT
        Draft202012Validator.check_schema(parameters)


def test_every_object_schema_is_closed_and_strictly_required() -> None:
    for tool in get_tool_schemas():
        for node in _walk(tool["function"]["parameters"]):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", ())) == set(node.get("properties", ()))


def test_schema_copy_subset_and_hash_are_stable() -> None:
    first = get_tool_schemas()
    second = get_tool_schemas()
    assert first == second
    assert first is not second
    first[0]["function"]["name"] = "mutated"
    assert get_tool_schemas()[0]["function"]["name"] == EXPECTED_TOOLS[0]

    assert tool_schema_hash() == TOOL_SCHEMA_SHA256
    assert tool_schema_hash().startswith("sha256:")
    assert len(tool_schema_hash()) == 71
    int(tool_schema_hash().removeprefix("sha256:"), 16)

    requested = ("finish_memory_session", "search_memory_bank")
    assert [item["function"]["name"] for item in get_tool_schemas(requested)] == [
        "search_memory_bank",
        "finish_memory_session",
    ]
    assert tool_schema_hash(requested) == tool_schema_hash(tuple(reversed(requested)))
    assert tool_schema_hash(requested) != TOOL_SCHEMA_SHA256
    with pytest.raises(ValueError, match="at least one"):
        get_tool_schemas(())
    with pytest.raises(ValueError, match="duplicates"):
        get_tool_schemas(("view_memories", "view_memories"))
    with pytest.raises(ValueError, match="unknown"):
        get_tool_schemas(("invented",))


def test_registry_search_and_view_are_closed_and_model_independent() -> None:
    _assert_valid(
        "view_material_registry",
        {"snapshot_id": None, "cursor": None, "limit": 100},
    )
    _assert_invalid(
        "view_material_registry",
        {"snapshot_id": None, "cursor": None, "limit": 100, "query": "hidden"},
    )
    _assert_valid(
        "search_memory_bank",
        {
            "snapshot_id": "snapshot_001",
            "query": "the protagonist's current coat and speaking voice",
            "filters": {
                "tags": ["protagonist"],
                "modalities": ["image", "audio"],
                "semantic_roles": ["identity", "voice"],
                "timeline_roles": ["timeless"],
            },
            "top_k": 8,
        },
    )
    _assert_valid(
        "view_memories",
        {
            "snapshot_id": "snapshot_001",
            "items": [
                {
                    "source_kind": "retrieved_memory",
                    "source_id": "memory_001",
                    "discovery_id": "search_001",
                    "representation_id": "repr_001",
                    "time_range": None,
                },
                    {
                        "source_kind": "generation_derived",
                        "source_id": "result_001",
                        "discovery_id": "attached_bundle_001",
                        "representation_id": None,
                        "time_range": None,
                    },
            ],
        },
    )
    valid_range = {
        "snapshot_id": "snapshot_001",
        "items": [
            {
                "source_kind": "generation_derived",
                "source_id": "result_001",
                "discovery_id": "attached_bundle_001",
                "representation_id": None,
                "time_range": {"start_ms": 0, "end_ms": 2500},
            }
        ],
    }
    _assert_valid("view_memories", valid_range)
    invalid_range = json.loads(json.dumps(valid_range))
    invalid_range["items"][0]["time_range"]["end_ms"] = 2.5
    _assert_invalid("view_memories", invalid_range)


def test_generate_image_requires_neutral_brief_and_authorized_inputs() -> None:
    valid = {
        "purpose": "opening_frame_evidence",
        "visual_brief": "A complete opening view preserving the red coat and room layout.",
        "input_evidence": [
            {
                "source_kind": "uploaded_material",
                "evidence_id": "upload_001",
                "representation_id": "repr_upload_001",
                "authorization_id": "view_001",
                "modality": "image",
                "semantic_roles": ["identity", "wardrobe"],
                "subject_ids": ["subject_main"],
            }
        ],
        "requirement_ids": ["req_identity", "req_scene"],
        "subject_ids": ["subject_main"],
        "semantic_roles": ["identity", "wardrobe"],
    }
    _assert_valid("generate_image", valid)
    with_seed = {**valid, "seed": 7}
    _assert_invalid("generate_image", with_seed)
    missing_brief = dict(valid)
    missing_brief.pop("visual_brief")
    _assert_invalid("generate_image", missing_brief)
    missing_authorization = json.loads(json.dumps(valid))
    missing_authorization["input_evidence"][0].pop("authorization_id")
    _assert_invalid("generate_image", missing_authorization)
    wrong_modality = json.loads(json.dumps(valid))
    wrong_modality["input_evidence"][0]["modality"] = "audio"
    _assert_invalid("generate_image", wrong_modality)
    empty_scope = json.loads(json.dumps(valid))
    empty_scope["subject_ids"] = []
    empty_scope["semantic_roles"] = []
    _assert_valid("generate_image", empty_scope)
    missing_scope = json.loads(json.dumps(valid))
    missing_scope.pop("subject_ids")
    _assert_invalid("generate_image", missing_scope)
    implementation_field = dict(valid)
    implementation_field["input_channel"] = 0
    _assert_invalid("generate_image", implementation_field)
    output_spec = dict(valid)
    output_spec["output_spec"] = {"aspect_ratio": "16:9", "size": "1024x576"}
    _assert_invalid("generate_image", output_spec)


def test_candidate_inspection_and_assessment_are_explicit() -> None:
    _assert_valid(
        "inspect_candidate",
        {
            "candidate_id": "candidate_001",
            "requested_uses": ["opening_frame_evidence", "memory_writeback"],
            "comparison_evidence_ids": ["upload_001"],
        },
    )
    assessment = {
        "inspection_id": "inspection_001",
        "candidate_id": "candidate_001",
        "check_results": [
            {
                "check_id": "identity.001",
                "result": "pass",
                "evidence_ids": ["inspection_frame_001", "upload_001"],
                "rationale": "Visible facial and wardrobe features agree.",
            }
        ],
        "use_results": [
            {
                "use_id": "opening_frame_evidence",
                "result": "pass",
                "evidence_ids": ["inspection_frame_001"],
                "rationale": "The complete opening composition is visible.",
            }
        ],
        "assessment_scope": "created_evidence_suitability",
        "assessment_result": "pass",
        "decision": "select",
        "candidate_ranking": ["candidate_001"],
        "selected_candidate_id": "candidate_001",
        "retry_instruction": None,
        "changed_axes": [],
    }
    _assert_valid("submit_candidate_assessment", assessment)
    bad = json.loads(json.dumps(assessment))
    bad["check_results"][0]["result"] = "looks_good"
    _assert_invalid("submit_candidate_assessment", bad)

    condition_review = json.loads(json.dumps(assessment))
    condition_review["assessment_scope"] = "condition_adherence"
    condition_review["decision"] = "accept_conditions"
    condition_review["selected_candidate_id"] = None
    condition_review["candidate_ranking"] = []
    _assert_valid("submit_candidate_assessment", condition_review)
    empty_condition_review = json.loads(json.dumps(condition_review))
    empty_condition_review["check_results"] = []
    _assert_valid("submit_candidate_assessment", empty_condition_review)
    for decision in (
        "revise_evidence_plan",
        "cannot_fix_with_memory",
    ):
        condition_review["decision"] = decision
        _assert_valid("submit_candidate_assessment", condition_review)

    invalid_decision = json.loads(json.dumps(condition_review))
    invalid_decision["decision"] = "retry_video"
    _assert_invalid("submit_candidate_assessment", invalid_decision)


def test_memory_writeback_schema_matches_runtime_sources_and_opaque_subjects() -> None:
    parameters = _parameters("update_memory_bank")
    record = parameters["properties"]["records"]["items"]
    assert record["properties"]["source_kind"]["enum"] == [
        "uploaded_material",
        "generated_evidence",
        "generation_derived",
    ]
    evidence_description = record["properties"]["writeback_evidence_ids"][
        "description"
    ]
    assert "exactly [source_evidence_id]" in evidence_description
    assert "candidate inspection" in evidence_description
    assert "source_selector" in record["required"]
    selector = record["properties"]["source_selector"]
    exact_range = selector["oneOf"][0]
    assert exact_range["properties"]["mode"]["const"] == "time_range"
    assert set(exact_range["required"]) == {"mode", "start_ms", "end_ms"}
    assert "Required for audio writeback" in exact_range["description"]
    assert selector["oneOf"][1] == {"type": "null"}
    update = parameters["properties"]["registry_updates"]["items"]
    assert "subject_id" in update["properties"]
    assert "tag" not in update["properties"]


def test_submit_evidence_plan_contains_complete_semantic_bindings() -> None:
    value = {
        "request_id": "request_001",
        "raw_request_sha256": "sha256:" + "a" * 64,
        "condition_budget_id": "budget_portable_v1",
        "parent_plan_id": None,
        "version": 1,
        "requirements": [_requirement().to_dict()],
        "ranked_bindings": [_binding().to_dict()],
        "unfulfilled_requirement_ids": [],
        "selection_mode": "requirements_met",
        "rationale": "One viewed identity representation covers the required need.",
    }
    _assert_valid("submit_evidence_plan", value)
    for required_field in (
        "semantic_roles",
        "intended_effect",
        "timeline_role",
        "priority",
        "required",
        "fallback",
    ):
        invalid = json.loads(json.dumps(value))
        invalid["ranked_bindings"][0].pop(required_field)
        _assert_invalid("submit_evidence_plan", invalid)

    invalid = json.loads(json.dumps(value))
    invalid["ranked_bindings"][0]["concrete_input_index"] = 1
    _assert_invalid("submit_evidence_plan", invalid)

    old_profile = json.loads(json.dumps(value))
    old_profile["capability_profile_id"] = old_profile.pop("condition_budget_id")
    _assert_invalid("submit_evidence_plan", old_profile)

    paired = json.loads(json.dumps(value))
    paired["ranked_bindings"][0]["paired_binding_id"] = None
    _assert_invalid("submit_evidence_plan", paired)

    backend_native_effect = json.loads(json.dumps(value))
    backend_native_effect["ranked_bindings"][0]["intended_effect"] = (
        "echo15_first_frame_slot"
    )
    _assert_invalid("submit_evidence_plan", backend_native_effect)

    binding_schema = _parameters("submit_evidence_plan")["properties"][
        "ranked_bindings"
    ]["items"]
    subject_description = binding_schema["properties"]["subject_ids"][
        "description"
    ]
    assert "Complete canonical subject_ids array" in subject_description
    assert "subject-subset bindings are invalid" in subject_description
    binding_list_description = _parameters("submit_evidence_plan")["properties"][
        "ranked_bindings"
    ]["description"]
    assert "Each exact physical evidence representation may occur once only" in (
        binding_list_description
    )


def test_evidence_plan_rejects_repeated_physical_representation() -> None:
    first = _binding()
    second = dataclasses.replace(
        _binding(
            binding_id="bind_wardrobe",
            requirement_id="req_wardrobe",
            role="wardrobe",
            representation_id=first.representation_id,
            rank=2,
        ),
        source_kind=first.source_kind,
        evidence_id=first.evidence_id,
        authorization_id="authorization_reauthorized",
    )
    plan = _plan(
        requirements=[
            _requirement(),
            _requirement("req_wardrobe", role="wardrobe"),
        ],
        bindings=[first, second],
    )
    with pytest.raises(ValueError, match="at most one binding"):
        plan.validate(_budget())


def test_finish_is_the_only_terminal_tool() -> None:
    _assert_valid(
        "finish_memory_session",
        {
            "accepted_plan_id": "plan_001",
            "memory_transaction_id": None,
            "writeback_disposition": "not_needed",
            "rationale": "No new stable reusable evidence was produced.",
        },
    )
    descriptions = {
        item["function"]["name"]: item["function"]["description"].lower()
        for item in get_tool_schemas()
    }
    assert "sole terminal" in descriptions["finish_memory_session"]
    assert all(
        "sole terminal" not in description
        for name, description in descriptions.items()
        if name != "finish_memory_session"
    )
    assert "accepted reusable trusted uploaded voice" in descriptions[
        "update_memory_bank"
    ]
    assert "equivalent committed or staged audio memory" in descriptions[
        "finish_memory_session"
    ]


def test_models_are_frozen_and_normalize_nested_sequences() -> None:
    requirement = _requirement()
    binding = _binding()
    budget = _budget()
    plan = _plan(requirements=[requirement], bindings=[binding])

    assert isinstance(requirement.semantic_roles, tuple)
    assert isinstance(binding.requirement_ids, tuple)
    assert isinstance(plan.requirements, tuple)
    assert isinstance(plan.ranked_bindings, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.selection_mode = "best_effort"
    with pytest.raises(dataclasses.FrozenInstanceError):
        budget.max_image_conditions = 99


def test_condition_budget_is_minimal_and_validated() -> None:
    budget = _budget()
    budget.validate()
    assert tuple(ConditionBudget.__dataclass_fields__) == (
        "budget_id",
        "max_total_conditions",
        "max_image_conditions",
        "max_video_conditions",
        "max_audio_conditions",
        "max_text_conditions",
    )
    assert budget.to_dict() == {
        "budget_id": "budget_portable_v1",
        "max_total_conditions": 4,
        "max_image_conditions": 2,
        "max_video_conditions": 1,
        "max_audio_conditions": 1,
        "max_text_conditions": 1,
    }
    with pytest.raises(ValueError, match="max_image_conditions"):
        _budget(max_total_conditions=1, max_image_conditions=2).validate()
    with pytest.raises(ValueError, match="non-negative"):
        _budget(max_audio_conditions=-1).validate()


def test_memory_plan_validates_references_ranks_and_condition_budget() -> None:
    budget = _budget()
    valid = _plan()
    valid.validate(budget)
    assert json.loads(json.dumps(valid.to_dict(budget))) == valid.to_dict(
        budget
    )

    with pytest.raises(ValueError, match="unknown requirements"):
        _plan(bindings=[_binding(requirement_id="missing")]).validate(budget)

    with pytest.raises(ValueError, match="ranks"):
        _plan(bindings=[_binding(rank=2)]).validate(budget)

    two_image_plan = _plan(
        requirements=[
            _requirement("req_1"),
            _requirement("req_2", role="scene"),
        ],
        bindings=[
            _binding("bind_1", requirement_id="req_1", rank=1),
            _binding(
                "bind_2",
                requirement_id="req_2",
                role="scene",
                representation_id="repr_2",
                rank=2,
            ),
        ],
    )
    with pytest.raises(ValueError, match="max_image_conditions"):
        two_image_plan.validate(_budget(max_image_conditions=1))
    with pytest.raises(ValueError, match="max_total_conditions"):
        two_image_plan.validate(
            _budget(
                max_total_conditions=1,
                max_image_conditions=1,
                max_video_conditions=1,
                max_audio_conditions=1,
                max_text_conditions=1,
            )
        )

    # Effects and representation reuse are semantic policy choices, not backend
    # capabilities.  Only the portable count budget constrains this plan.
    arbitrary_effect = _plan(bindings=[_binding(effect="style_reference")])
    arbitrary_effect.validate(budget)
    reused_representation = _plan(
        requirements=[
            _requirement("req_1"),
            _requirement("req_2", role="wardrobe"),
        ],
        bindings=[
            _binding("bind_1", requirement_id="req_1", rank=1),
            _binding(
                "bind_2",
                requirement_id="req_2",
                role="wardrobe",
                representation_id="repr_identity",
                rank=2,
            ),
        ],
    )
    reused_representation.validate(budget)

    with pytest.raises(ValueError, match="condition budget ID"):
        valid.validate(dataclasses.replace(budget, budget_id="budget_other"))


def test_required_coverage_and_fallback_rules_are_fail_closed() -> None:
    requirement = _requirement()
    with pytest.raises(ValueError, match="declared unfulfilled"):
        _plan(requirements=[requirement], bindings=[]).validate(_budget())

    best_effort = _plan(
        requirements=[requirement],
        bindings=[],
        unfulfilled=["req_identity"],
        selection_mode="capability_limited",
    )
    best_effort.validate(_budget())

    with pytest.raises(ValueError, match="requirements_met"):
        _plan(
            requirements=[requirement],
            bindings=[],
            unfulfilled=["req_identity"],
        ).validate(_budget())

    with pytest.raises(ValueError, match="unconditionally omitted"):
        _binding(fallback="allow_drop").validate()


def test_timeline_and_voice_invariants_are_backend_independent() -> None:
    video_requirement = _requirement(
        "req_scene_video", role="scene", modality="video", subject_ids=()
    )
    video_plan = _plan(
        requirements=[video_requirement],
        bindings=[
            _binding(
                "bind_scene_video",
                requirement_id="req_scene_video",
                role="scene",
                modality="video",
                representation_id="repr_scene_video",
                subject_ids=(),
            )
        ],
    )
    with pytest.raises(ValueError, match="max_video_conditions"):
        video_plan.validate(_budget(max_video_conditions=0))

    ending_requirement = _requirement(
        "req_ending", role="object_state", subject_ids=()
    )
    ending_binding = _binding(
        "bind_ending",
        requirement_id="req_ending",
        role="object_state",
        effect="exact_ending_state",
        representation_id="repr_ending",
        subject_ids=(),
        timeline_role="ending",
    )
    _plan(
        requirements=[ending_requirement], bindings=[ending_binding]
    ).validate(_budget())

    with pytest.raises(ValueError, match="timeline_role=ending"):
        _plan(
            requirements=[ending_requirement],
            bindings=[dataclasses.replace(ending_binding, timeline_role="timeless")],
        ).validate()

    with pytest.raises(ValueError, match="timeline_role=opening"):
        _plan(
            requirements=[_requirement("req_opening", role="scene", subject_ids=())],
            bindings=[
                _binding(
                    "bind_opening",
                    requirement_id="req_opening",
                    role="scene",
                    effect="exact_opening_state",
                    subject_ids=(),
                    timeline_role="middle",
                ),
            ],
        ).validate()

    with pytest.raises(ValueError, match="timeline_role=opening"):
        _plan(
            requirements=[
                _requirement("req_continuation", role="motion", subject_ids=())
            ],
            bindings=[
                _binding(
                    "bind_continuation",
                    requirement_id="req_continuation",
                    role="motion",
                    effect="temporal_continuation",
                    subject_ids=(),
                    timeline_role="ending",
                )
            ],
        ).validate()

    with pytest.raises(ValueError, match="exactly one subject"):
        _binding(
            role="voice",
            modality="audio",
            effect="voice_reference",
            subject_ids=(),
        ).validate()
    _binding(
        role="voice",
        modality="audio",
        effect="voice_reference",
        subject_ids=("subject_main",),
    ).validate()


def test_system_prompt_preserves_trusted_voice_binding_for_writeback() -> None:
    assert "exact trusted speaker_binding" in SYSTEM_PROMPT
    assert "basis=trusted_user_declared" in SYSTEM_PROMPT
    assert "matching_method=exact_normalized_name" in SYSTEM_PROMPT
    assert "stable cross-shot evidence" in SYSTEM_PROMPT
    assert "trusted source metadata explicitly carries one_shot=true" in SYSTEM_PROMPT
    assert "no committed or staged audio memory has the same exact-range" in SYSTEM_PROMPT
    assert "representation SHA-256, source_selector, and speaker_binding" in SYSTEM_PROMPT
    assert "independent metadata-level memory_writeback inspection" in SYSTEM_PROMPT
    assert "then stage it with update_memory_bank before finish_memory_session" in SYSTEM_PROMPT
    assert "Merely viewing a voice reference without adopting it" in SYSTEM_PROMPT
    assert "leaves writeback discretionary" in SYSTEM_PROMPT
    assert "copy its writeback_record_template from view_memories" in SYSTEM_PROMPT
    assert "description, source_selector, and writeback_evidence_ids must remain exact" in (
        SYSTEM_PROMPT
    )
    assert "tail frame into the next shot as sequence_context" in SYSTEM_PROMPT
    assert "do not duplicate an opening or tail frame" in SYSTEM_PROMPT
    assert "A capability_blocked result limits auditory and acoustic claims only" in (
        SYSTEM_PROMPT
    )
    assert "does not invalidate, erase, or weaken that trusted speaker binding" in (
        SYSTEM_PROMPT
    )


def test_plan_revision_locks_requirements_and_requires_semantic_binding_change() -> None:
    budget = _budget()
    initial = _plan()
    initial.validate(
        budget,
        expected_raw_request_sha256="sha256:" + "a" * 64,
    )
    revision = dataclasses.replace(
        initial,
        plan_id="plan_002",
        parent_plan_id="plan_001",
        version=2,
        ranked_bindings=(
            dataclasses.replace(
                initial.ranked_bindings[0],
                intended_effect="exact_opening_state",
                timeline_role="opening",
            ),
        ),
        rationale="Use the same evidence as an exact opening state.",
    )
    revision.validate(budget, parent_plan=initial)
    with pytest.raises(ValueError, match="requires its parent_plan"):
        revision.validate(budget)

    with pytest.raises(ValueError, match="raw request hash"):
        initial.validate(
            budget,
            expected_raw_request_sha256="sha256:" + "b" * 64,
        )

    with pytest.raises(ValueError, match="exactly match"):
        dataclasses.replace(
            revision,
            requirements=(
                dataclasses.replace(
                    initial.requirements[0], description="A different target"
                ),
            ),
        ).validate(budget, parent_plan=initial)

    metadata_only_changes = (
        {"binding_id": "bind_renamed"},
        {"authorization_id": "authorization_refreshed"},
        {"source_kind": "uploaded_material"},
        {"priority": 10},
        {"required": False},
        {"fallback": "allow_as_reference"},
        {"subject_ids": ("subject_main", "subject_secondary")},
    )
    for index, changes in enumerate(metadata_only_changes):
        metadata_only_revision = dataclasses.replace(
            initial,
            plan_id=f"plan_metadata_only_{index}",
            parent_plan_id=initial.plan_id,
            version=2,
            ranked_bindings=(
                dataclasses.replace(initial.ranked_bindings[0], **changes),
            ),
            rationale="Only planning or provenance metadata changed.",
        )
        with pytest.raises(ValueError, match="materially change"):
            metadata_only_revision.validate(budget, parent_plan=initial)

    representation_revision = dataclasses.replace(
        initial,
        plan_id="plan_repr_changed",
        parent_plan_id=initial.plan_id,
        version=2,
        ranked_bindings=(
            dataclasses.replace(
                initial.ranked_bindings[0],
                evidence_id="evidence_alternative",
                representation_id="repr_alternative",
            ),
        ),
    )
    representation_revision.validate(budget, parent_plan=initial)

    # Revision depth is workflow policy, not a generator capability field.
    third = dataclasses.replace(
        representation_revision,
        plan_id="plan_003",
        parent_plan_id=representation_revision.plan_id,
        version=3,
        ranked_bindings=(
            dataclasses.replace(
                representation_revision.ranked_bindings[0],
                intended_effect="soft_visual_reference",
            ),
        ),
    )
    third.validate(budget, parent_plan=representation_revision)


def test_revision_metadata_cannot_masquerade_as_a_condition_change() -> None:
    budget = _budget()
    requirements = (
        _requirement("req_a"),
        _requirement("req_b"),
    )
    bindings = (
        _binding("bind_a", requirement_id="req_a", rank=1),
        _binding(
            "bind_b",
            requirement_id="req_b",
            representation_id="repr_b",
            rank=2,
        ),
    )
    parent = _plan(requirements=requirements, bindings=bindings)
    coverage_only_revision = dataclasses.replace(
        parent,
        plan_id="plan_coverage_only",
        parent_plan_id=parent.plan_id,
        version=2,
        ranked_bindings=(
            dataclasses.replace(bindings[0], requirement_ids=("req_b",)),
            dataclasses.replace(bindings[1], requirement_ids=("req_a",)),
        ),
    )
    with pytest.raises(ValueError, match="materially change"):
        coverage_only_revision.validate(budget, parent_plan=parent)

    multimodal_requirement = dataclasses.replace(
        _requirement(), acceptable_modalities=("image", "video")
    )
    modality_parent = _plan(requirements=(multimodal_requirement,))
    modality_only_revision = dataclasses.replace(
        modality_parent,
        plan_id="plan_modality_only",
        parent_plan_id=modality_parent.plan_id,
        version=2,
        ranked_bindings=(
            dataclasses.replace(modality_parent.ranked_bindings[0], modality="video"),
        ),
    )
    with pytest.raises(ValueError, match="materially change"):
        modality_only_revision.validate(budget, parent_plan=modality_parent)


def test_revision_accepts_only_real_reference_condition_changes() -> None:
    budget = _budget()
    parent = _plan()
    condition_changes = (
        {"evidence_id": "evidence_alternative"},
        {"representation_id": "repr_alternative"},
        {"semantic_roles": ("identity", "wardrobe")},
        {"intended_effect": "style_reference"},
        {"timeline_role": "middle"},
    )
    for index, changes in enumerate(condition_changes):
        revision = dataclasses.replace(
            parent,
            plan_id=f"plan_condition_change_{index}",
            parent_plan_id=parent.plan_id,
            version=2,
            ranked_bindings=(
                dataclasses.replace(parent.ranked_bindings[0], **changes),
            ),
        )
        revision.validate(budget, parent_plan=parent)

    requirements = (
        _requirement("req_required"),
        _requirement("req_optional", required=False),
    )
    base = _plan(
        requirements=requirements,
        bindings=(
            _binding("bind_required", requirement_id="req_required", rank=1),
        ),
    )
    added = dataclasses.replace(
        base,
        plan_id="plan_binding_added",
        parent_plan_id=base.plan_id,
        version=2,
        ranked_bindings=(
            base.ranked_bindings[0],
            _binding(
                "bind_optional",
                requirement_id="req_optional",
                representation_id="repr_optional",
                rank=2,
                required=False,
                fallback="allow_drop",
            ),
        ),
    )
    added.validate(budget, parent_plan=base)

    removed = dataclasses.replace(
        added,
        plan_id="plan_binding_removed",
        parent_plan_id=added.plan_id,
        version=3,
        ranked_bindings=(added.ranked_bindings[0],),
    )
    removed.validate(budget, parent_plan=added)

    reordered = dataclasses.replace(
        added,
        plan_id="plan_bindings_reordered",
        parent_plan_id=added.plan_id,
        version=3,
        ranked_bindings=(
            dataclasses.replace(added.ranked_bindings[1], rank=1),
            dataclasses.replace(added.ranked_bindings[0], rank=2),
        ),
    )
    reordered.validate(budget, parent_plan=added)


def test_public_contract_and_prompt_contain_no_implementation_leakage() -> None:
    public_text = json.dumps(get_tool_schemas(), ensure_ascii=False) + SYSTEM_PROMPT
    portable_contract_text = "\n".join(
        (Path(__file__).parent / "storymem_policy" / filename).read_text(
            encoding="utf-8"
        )
        for filename in ("__init__.py", "contract.py", "models.py")
    )
    source_text = "\n".join(
        (Path(__file__).parent / "storymem_policy" / filename).read_text(
            encoding="utf-8"
        )
        for filename in ("__init__.py", "contract.py", "models.py", "prompts.py")
    )
    forbidden = (
        "get_pe_skill",
        "register_prompt",
        "generate_shot",
        "submit_shot",
        "condition_img",
        "memory_slots",
        "ID_A",
        "Echo",
        "MiniMax",
    )
    for value in forbidden:
        assert value not in public_text
        assert value not in source_text

    removed_backend_fields = (
        "PolicyCapabilities",
        "capability_profile_id",
        "supported_intended_effects",
        "opening_anchor_support",
        "ending_anchor_support",
        "continuation_anchor_support",
        "audio_binding_mode",
        "paired_binding_id",
        "evidence_order_sensitive",
        "allow_representation_reuse",
        "perceivable_modalities",
        "max_image_reference_inputs",
        "supported_image_aspect_ratios",
        "supported_image_sizes",
        "output_spec",
    )
    for value in removed_backend_fields:
        assert value not in portable_contract_text

    assert OPEN_MODEL_SYSTEM_PROMPT == SYSTEM_PROMPT
    assert "model-independent semantic evidence plan" in SYSTEM_PROMPT
    assert "Do not rewrite or author a media-generator prompt" in SYSTEM_PROMPT
    assert "finish_memory_session is the sole terminal action" in SYSTEM_PROMPT
    assert "complete subject_ids array verbatim, including order" in SYSTEM_PROMPT
    assert "never split one composite evidence item" in SYSTEM_PROMPT
    assert "may cover multiple compatible requirements" in SYSTEM_PROMPT


def test_public_import_has_no_runtime_or_provider_side_effects() -> None:
    script = """
import sys
import tools.storymem_policy
assert 'tools.storymem_policy.runtime' not in sys.modules
assert 'tools.storymem_teacher' not in sys.modules
assert not any(name.startswith('torch') for name in sys.modules if name != 'torchgen')
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
