"""Immutable domain models for the model-independent memory policy.

The classes in this module contain semantic intent only.  They deliberately
do not describe a concrete generator input layout.  A downstream compiler can
therefore consume the same :class:`EvidencePlan` for implementations with
different evidence channels and limits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .safety import validate_neutral_semantic_effect


MODALITIES: tuple[str, ...] = ("image", "video", "audio", "text")
VISUAL_MODALITIES: frozenset[str] = frozenset(("image", "video"))
TIMELINE_ROLES: tuple[str, ...] = ("opening", "middle", "ending", "timeless")
IMAGE_GENERATION_PURPOSES: tuple[str, ...] = (
    "opening_frame_evidence",
    "reference_evidence",
)
SOURCE_KINDS: tuple[str, ...] = (
    "retrieved_memory",
    "uploaded_material",
    "sequence_context",
    "generated_evidence",
    "generation_derived",
)
FALLBACK_POLICIES: tuple[str, ...] = (
    "forbidden",
    "allow_as_reference",
    "allow_drop",
)
SELECTION_MODES: tuple[str, ...] = (
    "requirements_met",
    "capability_limited",
    "best_effort",
)
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_.:-]*$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _tuple(value: Iterable[Any], field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of values, not text")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be iterable") from exc


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_token(value: Any, field_name: str) -> str:
    text = _require_text(value, field_name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise ValueError(
            f"{field_name} must start with a lowercase letter and contain only "
            "lowercase letters, digits, '_', '.', ':', or '-'"
        )
    return text


def _require_bool(value: Any, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _require_non_negative_int(value: Any, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_unique_strings(
    values: tuple[Any, ...],
    field_name: str,
    *,
    allow_empty: bool,
    tokens: bool = False,
) -> None:
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must not be empty")
    validator = _require_token if tokens else _require_text
    for index, value in enumerate(values):
        validator(value, f"{field_name}[{index}]")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")


def _ensure_json_safe(value: dict[str, Any]) -> dict[str, Any]:
    """Fail fast if a future field accidentally stops being JSON-safe."""

    json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


@dataclass(frozen=True, slots=True)
class ConditionBudget:
    """Portable upper bounds for a plan's semantic evidence conditions.

    The budget says only how many conditions may cross the Memory Policy
    boundary.  It does not describe generator slots, supported effects,
    ordering, pairing, perception, or any other backend behavior.
    """

    budget_id: str
    max_total_conditions: int
    max_image_conditions: int
    max_video_conditions: int
    max_audio_conditions: int
    max_text_conditions: int

    def validate(self) -> None:
        _require_text(self.budget_id, "budget_id")
        for field_name in (
            "max_total_conditions",
            "max_image_conditions",
            "max_video_conditions",
            "max_audio_conditions",
            "max_text_conditions",
        ):
            _require_non_negative_int(getattr(self, field_name), field_name)
        for field_name in (
            "max_image_conditions",
            "max_video_conditions",
            "max_audio_conditions",
            "max_text_conditions",
        ):
            if getattr(self, field_name) > self.max_total_conditions:
                raise ValueError(
                    f"{field_name} cannot exceed max_total_conditions"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _ensure_json_safe(
            {
                "budget_id": self.budget_id,
                "max_total_conditions": self.max_total_conditions,
                "max_image_conditions": self.max_image_conditions,
                "max_video_conditions": self.max_video_conditions,
                "max_audio_conditions": self.max_audio_conditions,
                "max_text_conditions": self.max_text_conditions,
            }
        )


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    """One semantic continuity or composition need extracted from the request."""

    requirement_id: str
    description: str
    semantic_roles: tuple[str, ...]
    subject_ids: tuple[str, ...]
    acceptable_modalities: tuple[str, ...]
    priority: int
    required: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "semantic_roles", _tuple(self.semantic_roles, "semantic_roles")
        )
        object.__setattr__(self, "subject_ids", _tuple(self.subject_ids, "subject_ids"))
        object.__setattr__(
            self,
            "acceptable_modalities",
            _tuple(self.acceptable_modalities, "acceptable_modalities"),
        )

    def validate(self) -> None:
        _require_text(self.requirement_id, "requirement_id")
        _require_text(self.description, "description")
        _validate_unique_strings(
            self.semantic_roles, "semantic_roles", allow_empty=False, tokens=True
        )
        _validate_unique_strings(self.subject_ids, "subject_ids", allow_empty=True)
        _validate_unique_strings(
            self.acceptable_modalities, "acceptable_modalities", allow_empty=False
        )
        unknown_modalities = set(self.acceptable_modalities) - set(MODALITIES)
        if unknown_modalities:
            raise ValueError(
                f"unsupported acceptable modalities: {sorted(unknown_modalities)}"
            )
        if type(self.priority) is not int or not 0 <= self.priority <= 100:
            raise ValueError("priority must be an integer from 0 through 100")
        _require_bool(self.required, "required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _ensure_json_safe(
            {
                "requirement_id": self.requirement_id,
                "description": self.description,
                "semantic_roles": list(self.semantic_roles),
                "subject_ids": list(self.subject_ids),
                "acceptable_modalities": list(self.acceptable_modalities),
                "priority": self.priority,
                "required": self.required,
            }
        )


@dataclass(frozen=True, slots=True)
class EvidenceBinding:
    """A ranked, authorized representation chosen for one or more needs."""

    binding_id: str
    requirement_ids: tuple[str, ...]
    source_kind: str
    evidence_id: str
    representation_id: str
    authorization_id: str
    modality: str
    semantic_roles: tuple[str, ...]
    subject_ids: tuple[str, ...]
    intended_effect: str
    timeline_role: str
    rank: int
    priority: int
    required: bool
    fallback: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requirement_ids", _tuple(self.requirement_ids, "requirement_ids")
        )
        object.__setattr__(
            self, "semantic_roles", _tuple(self.semantic_roles, "semantic_roles")
        )
        object.__setattr__(self, "subject_ids", _tuple(self.subject_ids, "subject_ids"))

    def validate(self) -> None:
        for field_name in (
            "binding_id",
            "evidence_id",
            "representation_id",
            "authorization_id",
        ):
            _require_text(getattr(self, field_name), field_name)
        _validate_unique_strings(
            self.requirement_ids, "requirement_ids", allow_empty=False
        )
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"unsupported source_kind: {self.source_kind!r}")
        if self.modality not in MODALITIES:
            raise ValueError(f"unsupported modality: {self.modality!r}")
        _validate_unique_strings(
            self.semantic_roles, "semantic_roles", allow_empty=False, tokens=True
        )
        _validate_unique_strings(self.subject_ids, "subject_ids", allow_empty=True)
        if (
            self.modality == "audio"
            and "voice" in self.semantic_roles
            and len(self.subject_ids) != 1
        ):
            raise ValueError("voice audio evidence must name exactly one subject ID")
        validate_neutral_semantic_effect(self.intended_effect)
        if self.timeline_role not in TIMELINE_ROLES:
            raise ValueError(f"unsupported timeline_role: {self.timeline_role!r}")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        if type(self.priority) is not int or not 0 <= self.priority <= 100:
            raise ValueError("priority must be an integer from 0 through 100")
        _require_bool(self.required, "required")
        if self.fallback not in FALLBACK_POLICIES:
            raise ValueError(f"unsupported fallback: {self.fallback!r}")
        if self.required and self.fallback == "allow_drop":
            raise ValueError("a required binding cannot be unconditionally omitted")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _ensure_json_safe(
            {
                "binding_id": self.binding_id,
                "requirement_ids": list(self.requirement_ids),
                "source_kind": self.source_kind,
                "evidence_id": self.evidence_id,
                "representation_id": self.representation_id,
                "authorization_id": self.authorization_id,
                "modality": self.modality,
                "semantic_roles": list(self.semantic_roles),
                "subject_ids": list(self.subject_ids),
                "intended_effect": self.intended_effect,
                "timeline_role": self.timeline_role,
                "rank": self.rank,
                "priority": self.priority,
                "required": self.required,
                "fallback": self.fallback,
            }
        )


def _reference_condition_signature(binding: EvidenceBinding) -> tuple[Any, ...]:
    """Return only the generator-relevant meaning of one reference condition.

    Evidence identity, representation identity, semantic use, and timeline
    placement can change the compiled generator condition.  Planning and
    provenance metadata such as IDs, source category, requirement coverage,
    priority, requiredness, and fallback cannot make a revision substantive.
    Binding addition, removal, and order remain visible in the enclosing tuple.
    """

    return (
        binding.evidence_id,
        binding.representation_id,
        tuple(sorted(binding.semantic_roles)),
        binding.intended_effect,
        binding.timeline_role,
    )


@dataclass(frozen=True, slots=True)
class EvidencePlan:
    """Final semantic evidence plan handed to a downstream compiler."""

    plan_id: str
    parent_plan_id: str | None
    version: int
    request_id: str
    raw_request_sha256: str
    condition_budget_id: str
    requirements: tuple[EvidenceRequirement, ...]
    ranked_bindings: tuple[EvidenceBinding, ...]
    unfulfilled_requirement_ids: tuple[str, ...]
    selection_mode: str
    rationale: str
    generation_prompt: str | None = None
    attempt_kind: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirements", _tuple(self.requirements, "requirements"))
        object.__setattr__(
            self, "ranked_bindings", _tuple(self.ranked_bindings, "ranked_bindings")
        )
        object.__setattr__(
            self,
            "unfulfilled_requirement_ids",
            _tuple(self.unfulfilled_requirement_ids, "unfulfilled_requirement_ids"),
        )

    def validate(
        self,
        condition_budget: ConditionBudget | None = None,
        *,
        expected_raw_request_sha256: str | None = None,
        parent_plan: "EvidencePlan" | None = None,
    ) -> None:
        for field_name in (
            "plan_id",
            "request_id",
            "condition_budget_id",
            "rationale",
        ):
            _require_text(getattr(self, field_name), field_name)
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be a positive integer")
        if self.version == 1 and self.parent_plan_id is not None:
            raise ValueError("version 1 must not have a parent plan")
        if self.version > 1:
            _require_text(self.parent_plan_id, "parent_plan_id")
            if parent_plan is None:
                raise ValueError("a revision requires its parent_plan for validation")
        if self.parent_plan_id == self.plan_id:
            raise ValueError("a plan cannot be its own parent")
        if not isinstance(self.raw_request_sha256, str) or _SHA256_RE.fullmatch(
            self.raw_request_sha256
        ) is None:
            raise ValueError("raw_request_sha256 must be a lowercase sha256 digest")
        if (
            expected_raw_request_sha256 is not None
            and self.raw_request_sha256 != expected_raw_request_sha256
        ):
            raise ValueError("raw request hash does not match the immutable request")
        if parent_plan is not None:
            if not isinstance(parent_plan, EvidencePlan):
                raise ValueError("parent_plan must be an EvidencePlan value")
            if self.parent_plan_id != parent_plan.plan_id:
                raise ValueError("parent_plan_id does not match the supplied parent")
            if self.version != parent_plan.version + 1:
                raise ValueError("revision version must increment its parent by one")
            for field_name in (
                "request_id",
                "raw_request_sha256",
                "condition_budget_id",
            ):
                if getattr(self, field_name) != getattr(parent_plan, field_name):
                    raise ValueError(
                        f"revision cannot change immutable {field_name}"
                    )
        if self.selection_mode not in SELECTION_MODES:
            raise ValueError(f"unsupported selection_mode: {self.selection_mode!r}")
        if self.generation_prompt is not None:
            _require_text(self.generation_prompt, "generation_prompt")
            if not 10 <= len(self.generation_prompt) <= 24000:
                raise ValueError("generation_prompt must be a complete bounded prompt")
            expected_kinds = {"initial"} if parent_plan is None else {"repair", "resample"}
            if self.attempt_kind not in expected_kinds:
                raise ValueError("attempt_kind does not match the plan version")
            if parent_plan is not None and parent_plan.generation_prompt is None:
                raise ValueError("cannot enable prompt authoring during a session")
        elif self.attempt_kind is not None or (parent_plan is not None and parent_plan.generation_prompt is not None):
            raise ValueError("generation_prompt and attempt_kind must remain paired")

        for requirement in self.requirements:
            if not isinstance(requirement, EvidenceRequirement):
                raise ValueError("requirements must contain EvidenceRequirement values")
            requirement.validate()
        requirement_ids = [item.requirement_id for item in self.requirements]
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("requirement IDs must be unique")
        requirement_by_id = {item.requirement_id: item for item in self.requirements}
        if parent_plan is not None and self.requirements != parent_plan.requirements:
            raise ValueError("revision requirements must exactly match parent requirements")

        for binding in self.ranked_bindings:
            if not isinstance(binding, EvidenceBinding):
                raise ValueError("ranked_bindings must contain EvidenceBinding values")
            binding.validate()
        binding_ids = [item.binding_id for item in self.ranked_bindings]
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("binding IDs must be unique")
        representation_keys = [
            (item.source_kind, item.evidence_id, item.representation_id)
            for item in self.ranked_bindings
        ]
        if len(set(representation_keys)) != len(representation_keys):
            raise ValueError(
                "each evidence representation may appear in at most one binding; "
                "merge all compatible requirement_ids and semantic_roles into that binding"
            )
        expected_ranks = list(range(1, len(self.ranked_bindings) + 1))
        actual_ranks = [item.rank for item in self.ranked_bindings]
        if actual_ranks != expected_ranks:
            raise ValueError("binding ranks must be unique, contiguous, and list-ordered")
        if parent_plan is not None:
            current_signature = tuple(
                _reference_condition_signature(item) for item in self.ranked_bindings
            )
            parent_signature = tuple(
                _reference_condition_signature(item)
                for item in parent_plan.ranked_bindings
            )
            prompt_changed = self.generation_prompt != parent_plan.generation_prompt
            if self.attempt_kind == "resample" and (prompt_changed or current_signature != parent_signature):
                raise ValueError("resample must preserve prompt and evidence conditions")
            if current_signature == parent_signature and not prompt_changed and self.attempt_kind != "resample":
                raise ValueError(
                    "revision must materially change ranked_bindings; changing only "
                    "planning or provenance metadata is not substantive"
                )

        covered_ids: set[str] = set()
        for binding in self.ranked_bindings:
            unknown_ids = set(binding.requirement_ids) - set(requirement_ids)
            if unknown_ids:
                raise ValueError(
                    f"binding {binding.binding_id!r} references unknown requirements: "
                    f"{sorted(unknown_ids)}"
                )
            for requirement_id in binding.requirement_ids:
                requirement = requirement_by_id[requirement_id]
                if binding.modality not in requirement.acceptable_modalities:
                    raise ValueError(
                        f"binding {binding.binding_id!r} uses modality {binding.modality!r} "
                        f"outside requirement {requirement_id!r}"
                    )
                if not set(binding.semantic_roles).intersection(
                    requirement.semantic_roles
                ):
                    raise ValueError(
                        f"binding {binding.binding_id!r} has no semantic role in common "
                        f"with requirement {requirement_id!r}"
                    )
                if not set(requirement.subject_ids).issubset(binding.subject_ids):
                    raise ValueError(
                        f"binding {binding.binding_id!r} does not cover all subjects "
                        f"for requirement {requirement_id!r}"
                    )
            covered_ids.update(binding.requirement_ids)

        _validate_unique_strings(
            self.unfulfilled_requirement_ids,
            "unfulfilled_requirement_ids",
            allow_empty=True,
        )
        unknown_unfulfilled = set(self.unfulfilled_requirement_ids) - set(requirement_ids)
        if unknown_unfulfilled:
            raise ValueError(
                f"unknown unfulfilled requirements: {sorted(unknown_unfulfilled)}"
            )
        overlap = covered_ids.intersection(self.unfulfilled_requirement_ids)
        if overlap:
            raise ValueError(
                f"requirements cannot be both bound and unfulfilled: {sorted(overlap)}"
            )
        missing_required = {
            item.requirement_id
            for item in self.requirements
            if item.required and item.requirement_id not in covered_ids
        }
        if missing_required != set(self.unfulfilled_requirement_ids).intersection(
            missing_required
        ):
            omitted = missing_required - set(self.unfulfilled_requirement_ids)
            if omitted:
                raise ValueError(
                    f"uncovered required requirements must be declared unfulfilled: "
                    f"{sorted(omitted)}"
                )
        if self.selection_mode == "requirements_met" and self.unfulfilled_requirement_ids:
            raise ValueError(
                "requirements_met cannot contain unfulfilled requirement IDs"
            )

        for binding in self.ranked_bindings:
            if (
                binding.intended_effect == "exact_opening_state"
                and binding.timeline_role != "opening"
            ):
                raise ValueError("exact_opening_state requires timeline_role=opening")
            if binding.intended_effect == "exact_ending_state" and (
                binding.timeline_role != "ending"
            ):
                raise ValueError("exact_ending_state requires timeline_role=ending")
            if (
                binding.intended_effect == "temporal_continuation"
                and binding.timeline_role != "opening"
            ):
                raise ValueError("temporal_continuation requires timeline_role=opening")

        if condition_budget is None:
            return
        if not isinstance(condition_budget, ConditionBudget):
            raise ValueError("condition_budget must be a ConditionBudget value")
        condition_budget.validate()
        if self.condition_budget_id != condition_budget.budget_id:
            raise ValueError("condition budget ID does not match the plan")
        if len(self.ranked_bindings) > condition_budget.max_total_conditions:
            raise ValueError("plan exceeds max_total_conditions")
        modality_limits = {
            "image": condition_budget.max_image_conditions,
            "video": condition_budget.max_video_conditions,
            "audio": condition_budget.max_audio_conditions,
            "text": condition_budget.max_text_conditions,
        }
        for modality, maximum in modality_limits.items():
            count = sum(item.modality == modality for item in self.ranked_bindings)
            if count > maximum:
                raise ValueError(f"plan exceeds max_{modality}_conditions")

    def to_dict(
        self,
        condition_budget: ConditionBudget | None = None,
        *,
        expected_raw_request_sha256: str | None = None,
        parent_plan: "EvidencePlan" | None = None,
    ) -> dict[str, Any]:
        self.validate(
            condition_budget,
            expected_raw_request_sha256=expected_raw_request_sha256,
            parent_plan=parent_plan,
        )
        return _ensure_json_safe(
            {
                "plan_id": self.plan_id,
                "parent_plan_id": self.parent_plan_id,
                "version": self.version,
                "request_id": self.request_id,
                "raw_request_sha256": self.raw_request_sha256,
                "condition_budget_id": self.condition_budget_id,
                "requirements": [item.to_dict() for item in self.requirements],
                "ranked_bindings": [item.to_dict() for item in self.ranked_bindings],
                "unfulfilled_requirement_ids": list(
                    self.unfulfilled_requirement_ids
                ),
                "selection_mode": self.selection_mode,
                "rationale": self.rationale,
                **({"generation_prompt": self.generation_prompt, "attempt_kind": self.attempt_kind}
                   if self.generation_prompt is not None else {}),
            }
        )


__all__ = [
    "ConditionBudget",
    "FALLBACK_POLICIES",
    "IMAGE_GENERATION_PURPOSES",
    "MODALITIES",
    "SELECTION_MODES",
    "SOURCE_KINDS",
    "TIMELINE_ROLES",
    "VISUAL_MODALITIES",
    "EvidenceBinding",
    "EvidenceRequirement",
    "EvidencePlan",
]
