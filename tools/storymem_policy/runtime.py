"""Provider-neutral runtime for the StoryMem Memory Policy V1 FSM."""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from storymem_teacher.media import (
    content_sha256 as media_content_sha256,
    materialize_asset_selector,
    selector_catalog,
    strip_video_audio,
)

from .contract import CANONICAL_TOOL_NAMES, get_tool_schemas
from .models import (
    ConditionBudget,
    EvidenceBinding,
    EvidencePlan,
    EvidenceRequirement,
)
from .ports import (
    VisualEvidenceCreationRequest,
    VisualEvidenceCreationResult,
    VisualEvidenceCreator,
)
from .safety import (
    public_representation,
    validate_neutral_binding_outcome,
    validate_neutral_reason_code,
    validate_neutral_semantic_effect,
)
from .state import MemoryPolicyState, StateConflictError


RUNTIME_VERSION = "storymem.memory-policy.runtime.v4"
TOOL_NAMES = CANONICAL_TOOL_NAMES
_MODEL_FORBIDDEN_KEYS = frozenset(
    {
        "provider",
        "provider_payload",
        "provider_request",
        "native_payload",
        "native_request",
        "private_receipt",
        "backend_payload",
    }
)

_TRUSTED_AUDIO_BINDING_BASIS = "trusted_user_declared"
_TRUSTED_AUDIO_MATCHING_METHOD = "exact_normalized_name"


class PolicyRejected(RuntimeError):
    def __init__(self, code: str, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _require_sha256(value: Any, field_name: str) -> str:
    text = str(value or "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise PolicyRejected(
            "invalid_generation_lineage", f"{field_name} must be a lowercase SHA-256 digest"
        )
    return text


def _require_identifier(value: Any, field_name: str) -> str:
    text = str(value or "")
    if not text:
        raise PolicyRejected(
            "invalid_generation_lineage", f"{field_name} must be non-empty"
        )
    return text


def _file_manifest(path: str | Path, representation_id: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size < 1:
        raise PolicyRejected("media_missing", f"evidence file is missing: {resolved}")
    mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    modality = mime_type.split("/", 1)[0]
    if modality not in {"image", "video", "audio", "text"}:
        modality = "text" if mime_type in {"application/json", "application/xml"} else "binary"
    return {
        "representation_id": representation_id,
        "modality": modality,
        "mime_type": mime_type,
        "size_bytes": resolved.stat().st_size,
        "sha256": media_content_sha256(resolved),
        "path": str(resolved),
    }


def _assert_model_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        leaked = _MODEL_FORBIDDEN_KEYS.intersection(str(key) for key in value)
        if leaked:
            raise RuntimeError(
                "model-visible observation leaked private infrastructure fields: "
                + ", ".join(sorted(leaked))
            )
        for child in value.values():
            _assert_model_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_model_safe(child)


def _as_condition_budget(value: Mapping[str, Any]) -> ConditionBudget:
    allowed = ConditionBudget.__dataclass_fields__
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(
            "condition budget contains non-portable fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    data = {key: child for key, child in value.items() if key in allowed}
    budget = ConditionBudget(**data)
    budget.validate()
    return budget


def _normalize_binding_outcomes(
    plan: Mapping[str, Any], outcomes: Any
) -> list[dict[str, Any]]:
    """Validate the public, implementation-neutral result for every binding."""

    if not isinstance(outcomes, list):
        raise PolicyRejected(
            "invalid_binding_outcomes", "compilation must report every plan binding"
        )
    binding_by_id = {
        str(row.get("binding_id") or ""): row
        for row in plan.get("ranked_bindings") or []
        if isinstance(row, Mapping)
    }
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in outcomes:
        if not isinstance(row, Mapping):
            raise PolicyRejected(
                "invalid_binding_outcomes", "binding outcome is malformed"
            )
        if set(row) != {"binding_id", "outcome", "applied_effect", "reason_code"}:
            raise PolicyRejected(
                "invalid_binding_outcomes",
                "binding outcome must use only the neutral public fields",
            )
        binding_id = str(row.get("binding_id") or "")
        if binding_id not in binding_by_id or binding_id in seen:
            raise PolicyRejected(
                "invalid_binding_outcomes", "binding outcome is unknown or duplicated"
            )
        binding = binding_by_id[binding_id]
        outcome = str(row.get("outcome") or "")
        applied_effect = row.get("applied_effect")
        reason_code = row.get("reason_code")
        try:
            validate_neutral_binding_outcome(
                outcome=outcome,
                reason_code=reason_code,
                applied_effect=applied_effect,
                requested_effect=binding.get("intended_effect"),
            )
        except ValueError as exc:
            raise PolicyRejected("invalid_binding_outcomes", str(exc)) from exc
        if outcome == "allowed_fallback" and binding.get("fallback") == "forbidden":
            raise PolicyRejected(
                "forbidden_fallback_reported",
                "compilation applied a fallback forbidden by the plan",
            )
        if (
            outcome == "not_applied"
            and binding.get("required") is True
            and binding.get("fallback") == "forbidden"
        ):
            raise PolicyRejected(
                "required_condition_not_compiled",
                "a required forbidden-fallback condition must fail before generation",
            )
        seen.add(binding_id)
        normalized.append(
            {
                "binding_id": binding_id,
                "outcome": outcome,
                "applied_effect": applied_effect,
                "reason_code": str(reason_code),
            }
        )
    if seen != set(binding_by_id):
        raise PolicyRejected(
            "incomplete_binding_outcomes",
            "compilation must report every plan binding exactly once",
        )
    return normalized


def _normalize_compiled_conditions(
    conditions: Any,
    *,
    plan: Mapping[str, Any],
    binding_outcomes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve exact condition bytes and derive their canonical signature input.

    This host-only manifest describes the actual reference payloads handed to
    the black-box generator.  Logical IDs and adapter receipts are deliberately
    excluded from the signature, so changing bookkeeping cannot manufacture a
    fresh generation input.
    """

    if not isinstance(conditions, list):
        raise PolicyRejected(
            "invalid_compiled_conditions", "compiled_conditions must be an array"
        )
    outcome_by_id = {
        str(row["binding_id"]): row for row in binding_outcomes
    }
    plan_binding_by_id = {
        str(row.get("binding_id") or ""): row
        for row in plan.get("ranked_bindings") or []
        if isinstance(row, Mapping)
    }
    applied_binding_ids = {
        binding_id
        for binding_id, row in outcome_by_id.items()
        if row.get("outcome") != "not_applied"
    }
    seen_binding_ids: set[str] = set()
    private_rows: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    allowed_keys = {
        "ordinal",
        "binding_ids",
        "modality",
        "content_path",
        "applied_effect",
    }
    for index, row in enumerate(conditions, start=1):
        if not isinstance(row, Mapping) or set(row) != allowed_keys:
            raise PolicyRejected(
                "invalid_compiled_conditions",
                "each compiled condition must contain exactly the neutral manifest fields",
            )
        ordinal = row.get("ordinal")
        modality = str(row.get("modality") or "")
        binding_ids = row.get("binding_ids")
        applied_effect = row.get("applied_effect")
        if ordinal != index or modality not in {"image", "video", "audio", "text"}:
            raise PolicyRejected(
                "invalid_compiled_conditions",
                "condition ordinals must be contiguous and modalities must be supported",
            )
        if (
            not isinstance(binding_ids, list)
            or not binding_ids
            or any(not isinstance(item, str) or not item for item in binding_ids)
            or len(set(binding_ids)) != len(binding_ids)
        ):
            raise PolicyRejected(
                "invalid_compiled_conditions",
                "each compiled condition must name unique applied binding IDs",
            )
        try:
            effect = validate_neutral_semantic_effect(applied_effect)
        except ValueError as exc:
            raise PolicyRejected("invalid_compiled_conditions", str(exc)) from exc
        for binding_id in binding_ids:
            outcome = outcome_by_id.get(binding_id)
            if (
                binding_id in seen_binding_ids
                or binding_id not in applied_binding_ids
                or outcome is None
                or outcome.get("applied_effect") != effect
                or (plan_binding_by_id.get(binding_id) or {}).get("modality")
                != modality
            ):
                raise PolicyRejected(
                    "invalid_compiled_conditions",
                    "compiled condition does not match one unique applied binding",
                )
            seen_binding_ids.add(binding_id)
        content_path = Path(str(row.get("content_path") or "")).expanduser().resolve()
        if not content_path.is_file() or content_path.stat().st_size < 1:
            raise PolicyRejected(
                "invalid_compiled_conditions",
                "compiled condition must point to the exact non-empty payload bytes",
            )
        content_sha256 = media_content_sha256(content_path)
        canonical = {
            "ordinal": index,
            "modality": modality,
            "content_sha256": content_sha256,
            "applied_effect": effect,
        }
        canonical_rows.append(canonical)
        private_rows.append(
            {
                **canonical,
                "binding_ids": list(binding_ids),
                "content_path": str(content_path),
            }
        )
    if seen_binding_ids != applied_binding_ids:
        raise PolicyRejected(
            "invalid_compiled_conditions",
            "compiled conditions must cover every and only applied binding",
        )
    return private_rows, canonical_rows


class MemoryPolicyRuntime:
    """Runtime with no PE resolver and no video-generation dependency."""

    def __init__(
        self,
        *,
        state: MemoryPolicyState,
        visual_evidence_creator: VisualEvidenceCreator,
        output_dir: str | Path,
    ) -> None:
        self.state = state
        self.visual_evidence_creator = visual_evidence_creator
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.condition_budget = _as_condition_budget(
            self.state.data["condition_budget"]
        )
        policy_limits = self.state.data.get("policy_limits") or {}
        if int(self.state.data.get("max_condition_revisions", -1)) != int(
            policy_limits.get("max_condition_revisions", -2)
        ):
            raise ValueError(
                "state max_condition_revisions differs from policy limits"
            )
        self.perceivable_modalities = frozenset(
            str(item) for item in policy_limits.get("perceivable_modalities") or ()
        )
        self.max_image_reference_inputs = int(
            policy_limits.get("max_image_reference_inputs", 0)
        )
        self.prompt_authoring = policy_limits.get("prompt_authoring") is True
        self._schemas = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in get_tool_schemas(prompt_authoring=self.prompt_authoring)
        }

    @property
    def phase(self) -> str:
        return self.state.phase

    def _condition_review_target_is_viewed(self) -> bool:
        """Return whether the exact host-designated review bytes were viewed.

        ``generation_review`` has one mandatory causal prefix: the attached
        generation representation must be viewed before it can be inspected.
        Keeping both tools visible before that prerequisite is met caused the
        closed Teacher to spend paid, hidden turns on calls which the runtime
        could already prove were illegal.  This predicate mirrors the exact
        check in :meth:`tool_inspect_candidate`; it does not choose any
        semantic policy action for the Teacher.
        """

        candidate_id = self.state.data.get("condition_review_target_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            return False
        candidate = self.state.data.get("generation_evidence", {}).get(candidate_id)
        if not isinstance(candidate, Mapping):
            return False
        representation = self._record_representation(candidate)
        representation_id = representation.get("representation_id")
        return any(
            row.get("evidence_id") == candidate_id
            and row.get("representation_id") == representation_id
            for row in self.state.data.get("views", {}).values()
            if isinstance(row, Mapping)
        )

    def allowed_tool_names(self) -> tuple[str, ...]:
        inflight = self.state.data.get("inflight")
        if isinstance(inflight, Mapping):
            return (str(inflight["tool_name"]),)
        if self.phase == "need_registry":
            return ("view_material_registry",)
        if self.phase == "planning":
            if self.state.data.get("pending_candidate_id"):
                return ("inspect_candidate",)
            if self.state.data.get("pending_assessment_id"):
                return ("submit_candidate_assessment",)
            if self.state.data.get("pending_retry_group_id"):
                return ("generate_image",)
            return (
                "search_memory_bank",
                "view_memories",
                "generate_image",
                "inspect_candidate",
                "submit_candidate_assessment",
                "submit_evidence_plan",
            )
        if self.phase == "generation_review":
            if self.state.data.get("pending_candidate_id"):
                return ("inspect_candidate",)
            if self.state.data.get("pending_assessment_id"):
                return ("submit_candidate_assessment",)
            # A host-attached generation has one deterministic causal prefix.
            # Hide actions that the runtime would necessarily reject while
            # retaining the Teacher's full autonomy over the actual condition
            # assessment and its accept/revise/cannot-fix decision.
            if not self._condition_review_target_is_viewed():
                return ("view_memories",)
            return ("inspect_candidate",)
        if self.phase == "writeback":
            if self.state.data.get("pending_candidate_id"):
                return ("inspect_candidate",)
            if self.state.data.get("pending_assessment_id"):
                return ("submit_candidate_assessment",)
            return (
                "view_material_registry",
                "view_memories",
                "inspect_candidate",
                "submit_candidate_assessment",
                "update_memory_bank",
                "finish_memory_session",
            )
        return ()

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        assistant_provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if name not in TOOL_NAMES:
            return {"ok": False, "error": "unknown_tool", "message": name}
        before = copy.deepcopy(self.state.data)
        persisted_sha256_before = self.state.persisted_sha256
        phase_before = self.phase
        commit_started = False
        try:
            if name not in self.allowed_tool_names():
                raise PolicyRejected(
                    "tool_not_allowed",
                    f"{name} is unavailable in phase {self.phase}",
                )
            errors = sorted(
                Draft202012Validator(self._schemas[name]).iter_errors(arguments),
                key=lambda error: list(error.path),
            )
            if errors:
                first = errors[0]
                location = ".".join(str(item) for item in first.path) or "<root>"
                raise PolicyRejected(
                    "invalid_arguments",
                    f"{name} arguments are invalid at {location}: {first.message}",
                )
            inflight = self.state.data.get("inflight")
            if isinstance(inflight, Mapping) and (
                inflight.get("tool_name") != name
                or inflight.get("arguments_sha256") != _sha256_json(arguments)
                or inflight.get("arguments") != dict(arguments)
            ):
                raise PolicyRejected(
                    "inflight_call_mismatch",
                    "resume requires the exact persisted generate_image call",
                )
            validated_provenance = None
            if assistant_provenance is not None:
                validated_provenance = copy.deepcopy(dict(assistant_provenance))
                raw_text = validated_provenance.get("raw_assistant_text")
                if not isinstance(raw_text, str) or validated_provenance.get(
                    "assistant_text_sha256"
                ) != _sha256_bytes(raw_text.encode("utf-8")):
                    raise PolicyRejected(
                        "invalid_assistant_provenance",
                        "assistant raw text or hash is invalid",
                    )
                try:
                    parsed = json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    raise PolicyRejected(
                        "invalid_assistant_provenance",
                        "assistant raw text is not strict JSON",
                    ) from exc
                if not isinstance(parsed, Mapping) or parsed.get("tool_call") != {
                    "name": name,
                    "arguments": dict(arguments),
                }:
                    raise PolicyRejected(
                        "invalid_assistant_provenance",
                        "assistant raw action differs from the executed tool call",
                    )
            handler = getattr(self, f"tool_{name}")
            result = handler(copy.deepcopy(dict(arguments)))
            attachments = copy.deepcopy(result.pop("_media_attachments", []))
            _assert_model_safe(result)
            event_id = self.state.next_id("event")
            self.state.data["events"].append(
                {
                    "event_id": event_id,
                    "tool_call": {
                        "name": name,
                        "arguments": copy.deepcopy(dict(arguments)),
                    },
                    "observation": copy.deepcopy(result),
                    "phase_before": phase_before,
                    "phase_after": self.phase,
                    "media_attachments_private": attachments,
                    "assistant_provenance": validated_provenance,
                }
            )
            commit_started = True
            self.state.save()
            return {**result, "_media_attachments": attachments}
        except PolicyRejected as exc:
            if commit_started or self.state.persisted_sha256 != persisted_sha256_before:
                self.state.reload()
            else:
                self.state.data = before
            result = {"ok": False, "error": exc.code, "message": str(exc)}
            if exc.details is not None:
                result["details"] = copy.deepcopy(exc.details)
            return result
        except Exception as exc:
            if (
                isinstance(exc, StateConflictError)
                or commit_started
                or self.state.persisted_sha256 != persisted_sha256_before
            ):
                self.state.reload()
            else:
                self.state.data = before
            raise

    def _assert_request(self, arguments: Mapping[str, Any]) -> None:
        request_id = arguments.get("request_id") or arguments.get("shot_request_id")
        if request_id is not None and request_id != self.state.data["request_id"]:
            raise PolicyRejected("wrong_request", "request_id does not match this session")

    @staticmethod
    def _record_representation(
        record: Mapping[str, Any], representation_id: str | None = None
    ) -> dict[str, Any]:
        representations = record.get("representations") or {}
        wanted = str(
            representation_id or record.get("default_representation_id") or ""
        )
        representation = (
            representations.get(wanted) if isinstance(representations, Mapping) else None
        )
        if not isinstance(representation, Mapping):
            direct = record.get("representation")
            representation = direct if isinstance(direct, Mapping) else None
        if not isinstance(representation, Mapping):
            raise PolicyRejected("unknown_representation", "evidence has no representation")
        result = copy.deepcopy(dict(representation))
        if not result.get("representation_id"):
            result["representation_id"] = wanted
        return result

    def _evidence_record(self, source_kind: str, evidence_id: str) -> dict[str, Any]:
        table_name = {
            "retrieved_memory": "memory_bank",
            "memory": "memory_bank",
            "uploaded_material": "uploaded_evidence",
            "uploaded_asset": "uploaded_evidence",
            "sequence_context": "previous_evidence",
            "previous_tail": "previous_evidence",
            "generated_evidence": "created_evidence",
            "created": "created_evidence",
            "generation_derived": "generation_evidence",
            "derived": "generation_evidence",
        }.get(source_kind)
        record = self.state.data.get(table_name or "", {}).get(evidence_id)
        if source_kind == "generated_evidence" and not isinstance(record, dict):
            record = self.state.data.get("generation_evidence", {}).get(evidence_id)
        if not isinstance(record, dict):
            raise PolicyRejected(
                "unknown_evidence", f"unknown evidence {source_kind}:{evidence_id}"
            )
        return record

    def _evidence_plan_by_id(self, plan_id: Any) -> Mapping[str, Any]:
        """Return one immutable plan revision from the current session.

        Generation candidates can outlive the plan revision which produced
        them, so best-available ranking must resolve each assessment against
        that candidate's own plan rather than the currently accepted plan.
        """

        matches = [
            row
            for row in [
                *(self.state.data.get("plan_history") or []),
                self.state.data.get("accepted_evidence_plan"),
            ]
            if isinstance(row, Mapping) and row.get("plan_id") == plan_id
        ]
        if len(matches) != 1:
            raise PolicyRejected(
                "generation_plan_missing",
                "an assessed generation has no unique EvidencePlan lineage",
            )
        return matches[0]

    @staticmethod
    def _aggregate_check_result(
        results: Sequence[str], *, unfulfilled_required_count: int = 0
    ) -> str:
        """Collapse checklist results using a conservative deterministic order."""

        if unfulfilled_required_count or "fail" in results:
            return "fail"
        if "uncertain" in results:
            return "uncertain"
        if "capability_blocked" in results:
            return "capability_blocked"
        return "pass"

    @staticmethod
    def _overall_assessment_result(results: Sequence[str]) -> str:
        """Derive the public assessment result from checks and requested uses."""

        if any(result in {"fail", "uncertain"} for result in results):
            return "fail"
        if "capability_blocked" in results:
            return "capability_limited"
        return "pass"

    def _unfulfilled_required_count(self, plan: Mapping[str, Any]) -> int:
        required_ids = {
            str(row.get("requirement_id") or "")
            for row in plan.get("requirements") or []
            if isinstance(row, Mapping) and row.get("required") is True
        }
        return len(
            required_ids.intersection(
                str(value) for value in plan.get("unfulfilled_requirement_ids") or []
            )
        )

    def _canonical_condition_candidate_ranking(
        self,
    ) -> tuple[list[str], dict[str, tuple[int, int, int, int]]]:
        """Rank all reviewed, decodable candidates from frozen runtime facts.

        The Teacher's ``candidate_ranking`` is an assertion, never the source
        of truth.  Required failures (including explicitly unfulfilled
        required requirements), required uncertainty, and optional failures
        are counted from each candidate's own frozen checklist and plan.  The
        append order in ``reviewed_generations`` is the stable generation-order
        tiebreaker required by the protocol.
        """

        scored: list[tuple[tuple[int, int, int, int], str]] = []
        seen_candidate_ids: set[str] = set()
        for generation_order, generation in enumerate(
            self.state.data.get("reviewed_generations") or []
        ):
            if not isinstance(generation, Mapping):
                raise PolicyRejected(
                    "invalid_generation_history",
                    "reviewed generation history is malformed",
                )
            candidate_id = str(generation.get("review_target_evidence_id") or "")
            if not candidate_id or candidate_id in seen_candidate_ids:
                raise PolicyRejected(
                    "invalid_generation_history",
                    "reviewed generation candidate IDs must be non-empty and unique",
                )
            seen_candidate_ids.add(candidate_id)
            assessment = self.state.data.get("assessments", {}).get(
                generation.get("assessment_id")
            )
            if not isinstance(assessment, Mapping):
                raise PolicyRejected(
                    "invalid_generation_history",
                    "reviewed generation is missing its immutable assessment",
                )
            inspection = self.state.data.get("inspections", {}).get(
                assessment.get("inspection_id")
            )
            if not isinstance(inspection, Mapping):
                raise PolicyRejected(
                    "invalid_generation_history",
                    "reviewed generation is missing its frozen checklist",
                )
            plan = self._evidence_plan_by_id(generation.get("plan_id"))
            requirement_by_id = {
                str(row.get("requirement_id") or ""): row
                for row in plan.get("requirements") or []
                if isinstance(row, Mapping)
            }
            binding_by_id = {
                str(row.get("binding_id") or ""): row
                for row in plan.get("ranked_bindings") or []
                if isinstance(row, Mapping)
            }
            check_results = assessment.get("check_results") or []
            result_by_id = {
                str(row.get("check_id") or ""): str(row.get("result") or "")
                for row in check_results
                if isinstance(row, Mapping)
            }
            checklist = inspection.get("checklist") or []
            if (
                len(result_by_id) != len(check_results)
                or {str(row.get("check_id") or "") for row in checklist}
                != set(result_by_id)
            ):
                raise PolicyRejected(
                    "invalid_generation_history",
                    "reviewed generation assessment does not cover its checklist exactly once",
                )

            required_failures = self._unfulfilled_required_count(plan)
            required_uncertain = 0
            optional_failures = 0
            for item in checklist:
                check_id = str(item.get("check_id") or "")
                result = result_by_id.get(check_id)
                if result not in {"pass", "fail", "uncertain", "capability_blocked"}:
                    raise PolicyRejected(
                        "invalid_generation_history",
                        "reviewed generation contains an invalid checklist result",
                    )
                binding = binding_by_id.get(str(item.get("binding_id") or ""))
                generation_check = self.prompt_authoring and check_id in {"generation.request_adherence", "generation.visual_quality"}
                if not isinstance(binding, Mapping) and not generation_check:
                    raise PolicyRejected(
                        "invalid_generation_history",
                        "condition checklist item has no binding in its generation plan",
                    )
                binding_requirement_ids = {
                    str(value) for value in (binding or {}).get("requirement_ids") or []
                }
                is_required = generation_check or bool((binding or {}).get("required")) or any(
                    bool(requirement_by_id.get(requirement_id, {}).get("required"))
                    for requirement_id in binding_requirement_ids
                )
                if is_required:
                    if result in {"fail", "capability_blocked"}:
                        required_failures += 1
                    elif result == "uncertain":
                        required_uncertain += 1
                elif result != "pass":
                    optional_failures += 1
            scored.append(
                (
                    (
                        required_failures,
                        required_uncertain,
                        optional_failures,
                        generation_order,
                    ),
                    candidate_id,
                )
            )

        scored.sort(key=lambda row: row[0])
        ranking = [candidate_id for _, candidate_id in scored]
        scores = {candidate_id: score for score, candidate_id in scored}
        return ranking, scores

    @staticmethod
    def _trusted_audio_speaker_binding(record: Mapping[str, Any]) -> dict[str, str]:
        """Return the canonical trusted speaker binding for an audio record.

        Audio bytes may be transported without being exposed to the policy
        model, so the opaque subject association has to survive as trusted
        runtime lineage. A bare ``subject_ids`` list is not sufficient: it
        does not prove that a user supplied a speaker name which resolved to
        exactly one registry subject.
        """

        binding = record.get("speaker_binding")
        subjects = list(record.get("subject_ids") or [])
        if not isinstance(binding, Mapping):
            raise PolicyRejected(
                "untrusted_audio_speaker_binding",
                "audio evidence has no trusted speaker binding",
            )
        declared_name = str(binding.get("declared_name") or "").strip()
        subject_id = str(binding.get("subject_id") or "")
        if (
            binding.get("basis") != _TRUSTED_AUDIO_BINDING_BASIS
            or binding.get("matching_method") != _TRUSTED_AUDIO_MATCHING_METHOD
            or not declared_name
            or len(subjects) != 1
            or subject_id != subjects[0]
        ):
            raise PolicyRejected(
                "untrusted_audio_speaker_binding",
                "audio evidence must retain one exact trusted named-speaker binding",
            )
        speaker_name = record.get("speaker_name")
        if speaker_name is not None and str(speaker_name).strip() != declared_name:
            raise PolicyRejected(
                "untrusted_audio_speaker_binding",
                "audio speaker name differs from its trusted binding",
            )
        return {
            "basis": _TRUSTED_AUDIO_BINDING_BASIS,
            "declared_name": declared_name,
            "subject_id": subject_id,
            "matching_method": _TRUSTED_AUDIO_MATCHING_METHOD,
        }

    @classmethod
    def _exact_audio_selector(
        cls, value: Any, *, field_name: str = "source_selector"
    ) -> dict[str, int | str]:
        if not isinstance(value, Mapping) or set(value) != {
            "mode",
            "start_ms",
            "end_ms",
        }:
            raise PolicyRejected(
                "audio_time_range_required",
                f"{field_name} must be an exact audio time_range selector",
            )
        start_ms = value.get("start_ms")
        end_ms = value.get("end_ms")
        if (
            value.get("mode") != "time_range"
            or type(start_ms) is not int
            or type(end_ms) is not int
            or start_ms < 0
            or end_ms <= start_ms
        ):
            raise PolicyRejected(
                "audio_time_range_required",
                f"{field_name} must satisfy 0 <= start_ms < end_ms",
            )
        return {"mode": "time_range", "start_ms": start_ms, "end_ms": end_ms}

    @classmethod
    def _audio_memory_matches(
        cls,
        record: Any,
        *,
        content_sha256: str,
        speaker_binding: Mapping[str, str],
        source_selector: Mapping[str, Any],
    ) -> bool:
        """Return whether one durable/staged record is the same named voice."""

        if not isinstance(record, Mapping) or record.get("modality") != "audio":
            return False
        try:
            candidate_binding = cls._trusted_audio_speaker_binding(record)
        except PolicyRejected:
            return False
        if candidate_binding != dict(speaker_binding):
            return False
        try:
            candidate_selector = cls._exact_audio_selector(
                record.get("source_selector"), field_name="memory.source_selector"
            )
        except PolicyRejected:
            return False
        if candidate_selector != dict(source_selector):
            return False
        representations = record.get("representations")
        if not isinstance(representations, Mapping):
            return False
        return any(
            isinstance(representation, Mapping)
            and representation.get("modality") == "audio"
            and representation.get("sha256") == content_sha256
            and representation.get("selector") == candidate_selector
            for representation in representations.values()
        )

    def _missing_required_uploaded_voice_memories(
        self,
        plan: Mapping[str, Any],
        *,
        staged_memories: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find adopted reusable uploaded voices absent from durable writeback.

        This is a terminal-state invariant, not a prescribed action order.  A
        matching record may already be committed or may be part of the single
        transaction staged for this finish call.
        """

        memory_bank = self.state.data.get("memory_bank")
        if not isinstance(memory_bank, Mapping):
            raise PolicyRejected(
                "invalid_memory_bank", "durable memory_bank is not an object"
            )
        committed = list(memory_bank.values())
        if staged_memories is None:
            staged = self.state.data.get("staged_transaction")
            staged_records = (
                list(staged.get("memories") or [])
                if isinstance(staged, Mapping)
                else []
            )
        else:
            staged_records = list(staged_memories)
        available = [*committed, *staged_records]
        missing: list[dict[str, Any]] = []
        seen_equivalence: set[tuple[Any, ...]] = set()
        uploads = self.state.data.get("uploaded_evidence") or {}
        for raw_binding in plan.get("ranked_bindings") or []:
            if (
                not isinstance(raw_binding, Mapping)
                or raw_binding.get("source_kind") != "uploaded_material"
                or raw_binding.get("modality") != "audio"
            ):
                continue
            evidence_id = str(raw_binding.get("evidence_id") or "")
            source = uploads.get(evidence_id) if isinstance(uploads, Mapping) else None
            if not isinstance(source, Mapping) or source.get("modality") != "audio":
                raise PolicyRejected(
                    "invalid_required_voice_evidence",
                    "accepted uploaded voice binding no longer resolves to audio evidence",
                )
            speaker_binding = self._trusted_audio_speaker_binding(source)
            if source.get("one_shot") is True:
                continue
            representation = self._record_representation(
                source, str(raw_binding.get("representation_id") or "")
            )
            content_sha256 = str(representation.get("sha256") or "")
            if (
                representation.get("modality") != "audio"
                or re.fullmatch(r"sha256:[0-9a-f]{64}", content_sha256) is None
            ):
                raise PolicyRejected(
                    "invalid_required_voice_evidence",
                    "accepted uploaded voice has no canonical audio content hash",
                )
            source_selector = self._exact_audio_selector(
                representation.get("selector"),
                field_name="accepted audio representation selector",
            )
            equivalence = (
                content_sha256,
                tuple(sorted(speaker_binding.items())),
                tuple(sorted(source_selector.items())),
            )
            if equivalence in seen_equivalence:
                continue
            seen_equivalence.add(equivalence)
            if any(
                self._audio_memory_matches(
                    record,
                    content_sha256=content_sha256,
                    speaker_binding=speaker_binding,
                    source_selector=source_selector,
                )
                for record in available
            ):
                continue
            missing.append(
                {
                    "evidence_id": evidence_id,
                    "representation_id": representation["representation_id"],
                    "content_sha256": content_sha256,
                    "speaker_binding": speaker_binding,
                    "source_selector": source_selector,
                }
            )
        return missing

    def _selected_generation_plan(self) -> dict[str, Any]:
        """Resolve the exact plan used by the selected historical generation."""

        selected = self.state.data.get("selected_generation")
        if not isinstance(selected, Mapping):
            raise PolicyRejected(
                "selected_generation_missing",
                "writeback requires one selected reviewed generation",
            )
        plan_id = str(selected.get("plan_id") or "")
        plans = [
            *(self.state.data.get("plan_history") or []),
            self.state.data.get("accepted_evidence_plan"),
        ]
        matches = [
            copy.deepcopy(dict(plan))
            for plan in plans
            if isinstance(plan, Mapping) and plan.get("plan_id") == plan_id
        ]
        if len(matches) != 1:
            raise PolicyRejected(
                "selected_generation_plan_missing",
                "the selected generation does not resolve to exactly one frozen evidence plan",
            )
        return matches[0]

    def _teacher_perception_attachment(
        self,
        representation: Mapping[str, Any],
        *,
        evidence_id: str,
        candidate_id: str | None = None,
        observation_evidence_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Return bytes that expose no modality unavailable to the policy model.

        A video container may carry an audio stream even though its declared
        policy modality is ``video``.  When the policy model cannot perceive
        audio, remux such a video into a deterministic visual-only proxy before
        it is attached to the Teacher conversation.  The source artifact is
        never modified, and the public observation records the exact proxy and
        its derivation so provider-wire provenance remains auditable.
        """

        source = copy.deepcopy(dict(representation))
        source_path = Path(str(source.get("path") or "")).expanduser().resolve()
        if not source_path.is_file() or source_path.stat().st_size < 1:
            raise PolicyRejected(
                "media_missing", f"missing evidence bytes: {evidence_id}"
            )
        source_modality = str(source.get("modality") or "")
        perception = None
        perception_path = source_path
        perception_representation = source
        if (
            source_modality == "video"
            and "video" in self.perceivable_modalities
            and "audio" not in self.perceivable_modalities
        ):
            source_sha256 = str(source.get("sha256") or "")
            if source_sha256 != media_content_sha256(source_path):
                raise PolicyRejected(
                    "media_integrity_mismatch",
                    f"evidence bytes changed after registration: {evidence_id}",
                )
            proxy_key = _sha256_json(
                {
                    "operation": "strip_audio_for_visual_perception",
                    "source_representation_id": source.get("representation_id"),
                    "source_sha256": source_sha256,
                }
            ).removeprefix("sha256:")
            suffix = source_path.suffix.lower() or ".mp4"
            try:
                perception_path, audio_removed = strip_video_audio(
                    source_path,
                    self.output_dir
                    / "perception_proxies"
                    / f"{proxy_key}{suffix}",
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise PolicyRejected(
                    "media_perception_failed",
                    f"video could not be prepared for visual-only perception: {evidence_id}",
                ) from exc
            if audio_removed:
                derivation = {
                    "version": "storymem.visual-only-perception.v1",
                    "operation": "strip_audio_for_visual_perception",
                    "source_representation_id": str(
                        source.get("representation_id") or ""
                    ),
                }
                perception_representation = _file_manifest(
                    perception_path,
                    f"repr_perception_{proxy_key[:24]}",
                )
                perception_representation.update(
                    {
                        "parent_sha256": source_sha256,
                        "derivation": derivation,
                    }
                )
                perception = {
                    "representation": public_representation(
                        perception_representation
                    ),
                    "parent_sha256": source_sha256,
                    "derivation": copy.deepcopy(derivation),
                }

        attachment = {
            "type": str(perception_representation.get("modality") or source_modality),
            "path": str(perception_path.resolve()),
            "representation_id": str(
                perception_representation.get("representation_id") or ""
            ),
            "evidence_id": observation_evidence_id or evidence_id,
        }
        if candidate_id is not None:
            attachment["candidate_id"] = candidate_id
        if perception is not None:
            attachment.update(
                {
                    "parent_sha256": perception["parent_sha256"],
                    "derivation": copy.deepcopy(perception["derivation"]),
                }
            )
        return attachment, perception

    def tool_view_material_registry(self, arguments: dict[str, Any]) -> dict[str, Any]:
        limit = int(arguments["limit"])
        if not 1 <= limit <= 200:
            raise PolicyRejected("invalid_limit", "registry limit must be 1..200")
        requested_snapshot_id = arguments.get("snapshot_id")
        snapshot_id = self.state.data.get("registry_snapshot_id")
        if snapshot_id is None:
            if requested_snapshot_id is not None or arguments.get("cursor") is not None:
                raise PolicyRejected(
                    "invalid_registry_start", "first registry page requires null snapshot and cursor"
                )
            snapshot_id = self.state.next_id("registrysnapshot")
            self.state.data["registry_snapshot_id"] = snapshot_id
            self.state.data["registry_next_cursor"] = None
        elif requested_snapshot_id != snapshot_id:
            raise PolicyRejected("snapshot_mismatch", "registry snapshot_id changed")
        cursor = arguments.get("cursor")
        if cursor != self.state.data.get("registry_next_cursor"):
            raise PolicyRejected("cursor_mismatch", "registry cursor is out of order")
        offset = int(cursor or 0)
        items = sorted(
            self.state.data["registry"]["items"].values(),
            key=lambda row: str(row.get("subject_id") or row.get("tag") or ""),
        )
        page = [
            {
                key: copy.deepcopy(row.get(key))
                for key in (
                    "subject_id",
                    "tag",
                    "category",
                    "description",
                    "stable_description",
                    "aliases",
                    "semantic_roles",
                )
                if row.get(key) is not None
            }
            for row in items[offset : offset + limit]
        ]
        next_offset = offset + len(page)
        next_cursor = str(next_offset) if next_offset < len(items) else None
        self.state.data["registry_next_cursor"] = next_cursor
        if self.phase == "need_registry" and next_cursor is None:
            self.state.transition("need_registry", "planning")
        for record in self.state.data["uploaded_evidence"].values():
            if (
                isinstance(record, dict)
                and record.get("modality") in {"audio", "video"}
                and not record.get("selectors")
            ):
                representation = self._record_representation(record)
                catalog = selector_catalog(str(representation.get("path") or ""))
                record["selectors"] = copy.deepcopy(catalog["selectors"])
                record["selector_limits"] = copy.deepcopy(catalog["limits"])
                if record.get("modality") == "audio":
                    full_ranges = [
                        selector
                        for selector in catalog["selectors"]
                        if selector.get("mode") == "time_range"
                    ]
                    if len(full_ranges) != 1:
                        raise PolicyRejected(
                            "audio_selector_unavailable",
                            "audio has no unique canonical full-range selector",
                        )
                    representation["selector"] = copy.deepcopy(full_ranges[0])
                    record["representations"][
                        str(representation["representation_id"])
                    ] = copy.deepcopy(representation)
        uploads = [
            {
                "evidence_id": evidence_id,
                "description": record.get("description") or "",
                "declared_tags": copy.deepcopy(record.get("declared_tags") or []),
                "modality": record.get("modality"),
                "selectors": copy.deepcopy(record.get("selectors") or []),
                **({"one_shot": True} if record.get("one_shot") is True else {}),
            }
            for evidence_id, record in sorted(
                self.state.data["uploaded_evidence"].items()
            )
        ]
        sequence_context = []
        for evidence_id, record in sorted(
            self.state.data["previous_evidence"].items()
        ):
            representations = record.get("representations") or {}
            if not isinstance(representations, Mapping):
                representations = {}
            available_representations = []
            for representation_id, representation in sorted(
                representations.items(), key=lambda item: str(item[0])
            ):
                if not isinstance(representation, Mapping):
                    continue
                # The registry is discovery metadata only.  In particular, it
                # must never expose the private path that view_memories later
                # uses to attach the actual bytes.
                public = public_representation(
                    {**copy.deepcopy(dict(representation)),
                     "representation_id": str(
                         representation.get("representation_id")
                         or representation_id
                     )}
                )
                available_representations.append(public)
            sequence_context.append(
                {
                    "source_id": evidence_id,
                    "evidence_id": evidence_id,
                    "modality": record.get("modality"),
                    "description": record.get("description") or "",
                    "semantic_roles": copy.deepcopy(
                        record.get("semantic_roles") or []
                    ),
                    "subject_ids": copy.deepcopy(record.get("subject_ids") or []),
                    "available_representations": available_representations,
                }
            )
        return {
            "ok": True,
            "request_id": self.state.data["request_id"],
            "phase": self.phase,
            "snapshot_id": snapshot_id,
            "registry_version": self.state.data["registry"]["version"],
            "materials": page,
            "next_cursor": next_cursor,
            "uploaded_materials": uploads,
            "sequence_context": sequence_context,
            "condition_budget": self.condition_budget.to_dict(),
            "policy_limits": {
                "max_condition_revisions": int(
                    self.state.data["max_condition_revisions"]
                ),
                "remaining_condition_revisions": max(
                    0,
                    int(self.state.data["max_condition_revisions"])
                    - int(self.state.data.get("revision_count", 0)),
                ),
                "visual_creation_budget": int(
                    self.state.data["visual_creation_budget"]
                ),
                "remaining_visual_creations": max(
                    0,
                    int(self.state.data["visual_creation_budget"])
                    - int(self.state.data["visual_creation_count"]),
                ),
                "max_image_reference_inputs": self.max_image_reference_inputs,
                "perceivable_modalities": sorted(self.perceivable_modalities),
                **({"prompt_authoring": True} if self.prompt_authoring else {}),
            },
            "shot_requirement": self.state.data["shot_requirement"],
            "raw_request_sha256": self.state.data["shot_requirement_sha256"],
        }

    def tool_search_memory_bank(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments.get("snapshot_id") != self.state.data.get("registry_snapshot_id"):
            raise PolicyRejected("snapshot_mismatch", "search requires the completed registry snapshot")
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise PolicyRejected("empty_query", "query must not be empty")
        top_k = int(arguments["top_k"])
        if not 1 <= top_k <= 50:
            raise PolicyRejected("invalid_top_k", "top_k must be 1..50")
        query_tokens = set(re.findall(r"[\w.-]+", query.lower()))
        filters = arguments.get("filters") or {}
        required_tags = set(filters.get("tags") or [])
        required_semantic_roles = set(filters.get("semantic_roles") or [])
        allowed_modalities = set(filters.get("modalities") or [])
        allowed_timeline_roles = set(filters.get("timeline_roles") or [])
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for evidence_id, record in self.state.data["memory_bank"].items():
            if record.get("modality") == "audio":
                try:
                    self._trusted_audio_speaker_binding(record)
                    source_selector = self._exact_audio_selector(
                        record.get("source_selector"),
                        field_name="memory.source_selector",
                    )
                    representation = self._record_representation(record)
                    if representation.get("selector") != source_selector:
                        raise PolicyRejected(
                            "invalid_audio_memory",
                            "audio memory representation does not match its source selector",
                        )
                except PolicyRejected:
                    # Legacy or corrupted audio may remain in durable state for
                    # audit, but it is not discoverable as reusable evidence.
                    continue
            if allowed_modalities and record.get("modality") not in allowed_modalities:
                continue
            if required_tags and not required_tags.issubset(
                set(record.get("tags") or [])
            ):
                continue
            if required_semantic_roles and not required_semantic_roles.issubset(
                set(record.get("semantic_roles") or [])
            ):
                continue
            if allowed_timeline_roles and record.get(
                "timeline_role"
            ) not in allowed_timeline_roles:
                continue
            text = " ".join(
                [
                    str(record.get("description") or ""),
                    *(str(item) for item in record.get("tags") or []),
                    *(str(item) for item in record.get("semantic_roles") or []),
                    *(str(item) for item in record.get("subject_ids") or []),
                ]
            ).lower()
            tokens = set(re.findall(r"[\w.-]+", text))
            score = len(query_tokens.intersection(tokens)) / max(1, len(query_tokens))
            if score:
                rows.append((score, evidence_id, record))
        rows.sort(key=lambda item: (-item[0], item[1]))
        candidates = []
        for score, evidence_id, record in rows[:top_k]:
            candidate = {
                "evidence_id": evidence_id,
                "source_kind": "retrieved_memory",
                "modality": record.get("modality"),
                "description": record.get("description") or "",
                "tags": copy.deepcopy(record.get("tags") or []),
                "semantic_roles": copy.deepcopy(record.get("semantic_roles") or []),
                "timeline_role": record.get("timeline_role"),
                "subject_ids": copy.deepcopy(record.get("subject_ids") or []),
                "score": round(score, 6),
            }
            if record.get("modality") == "audio":
                candidate["speaker_binding"] = copy.deepcopy(
                    self._trusted_audio_speaker_binding(record)
                )
                candidate["selectors"] = [
                    self._exact_audio_selector(
                        record.get("source_selector"),
                        field_name="memory.source_selector",
                    )
                ]
            candidates.append(candidate)
        search_id = self.state.next_id("search")
        self.state.data["searches"][search_id] = {
            "query": query,
            "filters": copy.deepcopy(filters),
            "candidate_ids": [row["evidence_id"] for row in candidates],
        }
        return {"ok": True, "search_id": search_id, "candidates": candidates}

    def tool_view_memories(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments.get("snapshot_id") != self.state.data.get("registry_snapshot_id"):
            raise PolicyRejected("snapshot_mismatch", "view requires the completed registry snapshot")
        requests = arguments.get("items") or []
        if not isinstance(requests, list) or not requests:
            raise PolicyRejected("empty_view", "items must contain evidence to view")
        authorization_id = self.state.next_id("viewauth")
        returned: list[dict[str, Any]] = []
        media: list[dict[str, Any]] = []
        for request in requests:
            source_kind = str(request.get("source_kind") or request.get("kind") or "")
            evidence_id = str(
                request.get("source_id")
                or request.get("evidence_id")
                or request.get("id")
                or ""
            )
            record = self._evidence_record(source_kind, evidence_id)
            if source_kind in {"retrieved_memory", "memory"}:
                search_id = str(request.get("discovery_id") or request.get("search_id") or "")
                search = self.state.data["searches"].get(search_id)
                if not search or evidence_id not in search["candidate_ids"]:
                    raise PolicyRejected(
                        "search_authorization_required",
                        "retrieved memory must come from a current search result",
                    )
            elif source_kind in {"uploaded_material", "sequence_context"}:
                if request.get("discovery_id") != self.state.data.get(
                    "registry_snapshot_id"
                ):
                    raise PolicyRejected(
                        "discovery_authorization_required",
                        "registry evidence must cite the current registry snapshot",
                    )
            elif source_kind in {"generated_evidence", "generation_derived"}:
                bundle = self.state.data.get("generation_bundle_private") or {}
                if source_kind == "generation_derived" and request.get(
                    "discovery_id"
                ) != bundle.get("bundle_id"):
                    raise PolicyRejected(
                        "discovery_authorization_required",
                        "generation-derived evidence must cite its attached bundle",
                    )
            representation = self._record_representation(
                record, request.get("representation_id")
            )
            requested_range = request.get("time_range")
            source_modality = str(
                record.get("modality") or representation.get("modality") or ""
            )
            if source_modality == "audio" and requested_range is None:
                ranges = [
                    copy.deepcopy(selector)
                    for selector in record.get("selectors") or []
                    if isinstance(selector, Mapping)
                    and selector.get("mode") == "time_range"
                ]
                if not ranges:
                    catalog = selector_catalog(str(representation.get("path") or ""))
                    ranges = [
                        copy.deepcopy(selector)
                        for selector in catalog["selectors"]
                        if selector.get("mode") == "time_range"
                    ]
                    record["selectors"] = copy.deepcopy(catalog["selectors"])
                    record["selector_limits"] = copy.deepcopy(catalog["limits"])
                raise PolicyRejected(
                    "audio_time_range_required",
                    "audio must be viewed through an explicit exact time range",
                    details={"available_time_ranges": ranges},
                )
            if requested_range is not None:
                if source_modality not in {"audio", "video"}:
                    raise PolicyRejected(
                        "unsupported_time_range",
                        "time_range is supported only for audio or video evidence",
                    )
                start_ms = requested_range.get("start_ms")
                end_ms = requested_range.get("end_ms")
                if (
                    type(start_ms) is not int
                    or type(end_ms) is not int
                    or start_ms < 0
                    or end_ms <= start_ms
                ):
                    raise PolicyRejected(
                        "invalid_time_range", "time range must satisfy 0 <= start < end"
                    )
                source_representation_id = str(representation["representation_id"])
                requested_selector = {
                    "mode": "time_range",
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
                if representation.get("selector") != requested_selector:
                    try:
                        materialized = materialize_asset_selector(
                            {
                                "asset_id": f"{evidence_id}_{source_representation_id}",
                                "default_representation_id": source_representation_id,
                                "representations": {
                                    source_representation_id: representation
                                },
                            },
                            requested_selector,
                            self.output_dir / "view_materializations",
                        )
                    except (OSError, RuntimeError, ValueError) as exc:
                        raise PolicyRejected("invalid_time_range", str(exc)) from exc
                    representation = copy.deepcopy(materialized["representation"])
                    # ``materialize_asset_selector`` is shared with the legacy
                    # runtime and names this field ``media_type``.  Normalize it at
                    # the provider-neutral boundary before exposing or persisting
                    # the derived representation.
                    representation["modality"] = str(
                        representation.get("modality")
                        or representation.get("media_type")
                        or source_modality
                    )
                    record.setdefault("representations", {})[
                        representation["representation_id"]
                    ] = copy.deepcopy(representation)
            path = Path(str(representation.get("path") or ""))
            if not path.is_file():
                raise PolicyRejected("media_missing", f"missing evidence bytes: {evidence_id}")
            public_manifest = public_representation(representation)
            item_modality = str(
                record.get("modality") or representation.get("modality") or ""
            )
            item = {
                "source_kind": source_kind,
                "evidence_id": evidence_id,
                "authorization_id": authorization_id,
                "description": record.get("description") or "",
                "modality": item_modality,
                "semantic_roles": copy.deepcopy(record.get("semantic_roles") or []),
                "subject_ids": copy.deepcopy(record.get("subject_ids") or []),
                "representation": public_manifest,
                "perception_status": (
                    "available"
                    if item_modality in self.perceivable_modalities
                    else "capability_blocked"
                ),
                "selector": copy.deepcopy(
                    representation.get("selector")
                    if requested_range is not None
                    else None
                ),
            }
            if item_modality == "audio":
                item["speaker_binding"] = copy.deepcopy(
                    self._trusted_audio_speaker_binding(record)
                )
            if source_kind in {"uploaded_material", "uploaded_asset"}:
                item.pop("subject_ids", None)
                item["declared_tags"] = copy.deepcopy(
                    record.get("declared_tags") or []
                )
                item["trusted_stable_facts"] = copy.deepcopy(
                    record.get("trusted_stable_facts") or []
                )
                # Uploaded bytes are authorized by trusted provenance rather
                # than by a perceptual assessment.  Give the policy an exact,
                # schema-valid record it can copy instead of asking it to
                # reverse-engineer private allowlists (especially when audio
                # perception is capability-blocked).
                if item["semantic_roles"]:
                    item["writeback_record_template"] = {
                        "record_id": None,
                        "source_kind": "uploaded_material",
                        "source_evidence_id": evidence_id,
                        "representation_id": representation[
                            "representation_id"
                        ],
                        "authorization_id": authorization_id,
                        "modality": item_modality,
                        "semantic_roles": copy.deepcopy(item["semantic_roles"]),
                        "timeline_role": str(
                            record.get("timeline_role") or "timeless"
                        ),
                        "tags": copy.deepcopy(item["declared_tags"]),
                        "description": str(item["description"]),
                        "stable_facts": copy.deepcopy(
                            item["trusted_stable_facts"]
                        ),
                        "reuse_scope": str(record.get("reuse_scope") or "story"),
                        "source_selector": copy.deepcopy(item["selector"]),
                        "writeback_evidence_ids": [evidence_id],
                    }
                if record.get("speaker_name"):
                    item["speaker_name"] = record["speaker_name"]
                if record.get("one_shot") is True:
                    item["one_shot"] = True
            returned.append(item)
            if item_modality in self.perceivable_modalities:
                attachment, perception = self._teacher_perception_attachment(
                    representation,
                    evidence_id=evidence_id,
                )
                media.append(attachment)
                if perception is not None:
                    item["perception"] = perception
            key = "|".join(
                [
                    source_kind,
                    evidence_id,
                    str(representation.get("representation_id") or ""),
                    authorization_id,
                ]
            )
            self.state.data["views"][key] = {
                "source_kind": source_kind,
                "evidence_id": evidence_id,
                "representation_id": representation.get("representation_id"),
                "authorization_id": authorization_id,
                "selector": copy.deepcopy(
                    representation.get("selector")
                    if requested_range is not None
                    else None
                ),
            }
        return {
            "ok": True,
            "authorization_id": authorization_id,
            "items": returned,
            "_media_attachments": media,
        }

    def _resolve_view_binding(self, value: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
        source_kind = str(value.get("source_kind") or value.get("kind") or "")
        evidence_id = str(value.get("evidence_id") or value.get("id") or "")
        representation_id = str(value.get("representation_id") or "")
        authorization_id = str(value.get("authorization_id") or "")
        key = "|".join(
            [source_kind, evidence_id, representation_id, authorization_id]
        )
        if key not in self.state.data["views"]:
            raise PolicyRejected(
                "view_authorization_required",
                f"evidence {evidence_id} lacks exact current-session view authorization",
            )
        record = self._evidence_record(source_kind, evidence_id)
        representation = self._record_representation(record, representation_id)
        modality = str(record.get("modality") or representation.get("modality") or "")
        if modality == "audio":
            self._trusted_audio_speaker_binding(record)
        if "subject_ids" in value and list(value.get("subject_ids") or []) != list(
            record.get("subject_ids") or []
        ):
            raise PolicyRejected(
                "subject_binding_mismatch",
                "submitted subject_ids differ from the source's private canonical binding",
            )
        if value.get("modality") is not None and value.get("modality") != modality:
            raise PolicyRejected(
                "modality_mismatch", "submitted modality differs from the evidence"
            )
        return representation, Path(str(representation["path"]))

    def tool_generate_image(self, arguments: dict[str, Any]) -> dict[str, Any]:
        purpose = str(arguments.get("purpose") or "")
        count = int(self.state.data["visual_creation_count"])
        if count >= int(self.state.data["visual_creation_budget"]):
            raise PolicyRejected("creation_budget_exhausted", "image creation budget is exhausted")
        input_bindings = arguments.get("input_evidence") or arguments.get("references") or []
        if len(input_bindings) > self.max_image_reference_inputs:
            raise PolicyRejected(
                "too_many_inputs", "image creation exceeds the neutral reference-input limit"
            )
        known_subject_ids = set(self.state.data["registry"]["items"])
        known_subject_ids.update(
            str(row.get("subject_id") or row.get("tag") or "")
            for row in self.state.data["registry"]["items"].values()
        )
        unknown_subject_ids = sorted(
            set(arguments["subject_ids"]) - known_subject_ids
        )
        if unknown_subject_ids:
            raise PolicyRejected(
                "unknown_subject_ids",
                "generated evidence requested unknown registry subjects",
                details={"subject_ids": unknown_subject_ids},
            )
        if purpose == "reference_evidence" and not arguments["semantic_roles"]:
            raise PolicyRejected(
                "semantic_roles_required",
                "reference evidence requires at least one semantic role",
            )
        input_paths = [self._resolve_view_binding(item)[1] for item in input_bindings]
        creation_signature = _sha256_json(
            {
                "purpose": purpose,
                "visual_brief": arguments["visual_brief"],
                "ordered_input_sha256": [
                    media_content_sha256(path) for path in input_paths
                ],
            }
        )
        inflight = self.state.data.get("inflight")
        if isinstance(inflight, Mapping):
            request_id = str(inflight["request_id"])
            group_id = str(inflight["generation_group_id"])
        else:
            group_id = self.state.data.get("pending_retry_group_id")
            if group_id is None:
                group_id = self.state.next_id("imagegroup")
                self.state.data["creation_groups"][group_id] = {
                    "purpose": purpose,
                    "candidate_ids": [],
                    "selected_candidate_id": None,
                    "requirement_ids": copy.deepcopy(arguments["requirement_ids"]),
                    "attempt_signatures": [],
                }
            group = self.state.data["creation_groups"].get(str(group_id))
            if not group or group.get("purpose") != purpose or group.get(
                "selected_candidate_id"
            ):
                raise PolicyRejected(
                    "invalid_creation_group", "image creation group is unavailable"
                )
            attempt_signatures = group.setdefault("attempt_signatures", [])
            if creation_signature in attempt_signatures:
                raise PolicyRejected(
                    "duplicate_image_creation_request",
                    "recreating image evidence requires changed creator inputs",
                )
            attempt_signatures.append(creation_signature)
            request_id = self.state.next_id("imagecreate")
            self.state.data["inflight"] = {
                "tool_name": "generate_image",
                "arguments": copy.deepcopy(arguments),
                "arguments_sha256": _sha256_json(arguments),
                "request_id": request_id,
                "generation_group_id": str(group_id),
                "creation_signature_sha256": creation_signature,
            }
            # This checkpoint precedes the remote call. A restarted runtime
            # can only replay these exact arguments with the same request ID.
            self.state.save()
        group = self.state.data["creation_groups"].get(str(group_id))
        if not group or group.get("purpose") != purpose or group.get("selected_candidate_id"):
            raise PolicyRejected("invalid_creation_group", "image creation group is unavailable")
        request = VisualEvidenceCreationRequest(
            request_id=request_id,
            purpose=purpose,
            creation_spec={
                "visual_brief": arguments["visual_brief"],
            },
            input_paths=tuple(input_paths),
            output_dir=self.output_dir / "visual_evidence" / request_id,
        )
        raw_result = self.visual_evidence_creator.create(request)
        if isinstance(raw_result, VisualEvidenceCreationResult):
            result_request_id = raw_result.request_id
            output_path = raw_result.output_path
            private_receipt = raw_result.private_receipt
        elif isinstance(raw_result, Mapping):
            result_request_id = str(raw_result.get("request_id") or "")
            output_path = Path(str(raw_result.get("output_path") or ""))
            private_receipt = raw_result.get("private_receipt") or {}
        else:
            raise PolicyRejected("creator_protocol_error", "visual creator returned an invalid result")
        if result_request_id != request_id:
            raise PolicyRejected("creator_request_mismatch", "visual creator changed request_id")
        candidate_id = self.state.next_id("evidence")
        representation_id = self.state.next_id("repr")
        representation = _file_manifest(output_path, representation_id)
        if representation["modality"] != "image":
            raise PolicyRejected("creator_output_not_image", "visual creator did not return an image")
        record = {
            "evidence_id": candidate_id,
            "source_kind": "generated_evidence",
            "modality": "image",
            "purpose": purpose,
            "description": str(arguments["visual_brief"]),
            "semantic_roles": copy.deepcopy(arguments["semantic_roles"]),
            "subject_ids": copy.deepcopy(arguments["subject_ids"]),
            "requirement_ids": copy.deepcopy(arguments["requirement_ids"]),
            "input_evidence": copy.deepcopy(input_bindings),
            "default_representation_id": representation_id,
            "representations": {representation_id: representation},
            "creation_group_id": str(group_id),
            "inspection_id": None,
            "assessment_id": None,
            "private_creation_receipt": copy.deepcopy(dict(private_receipt)),
        }
        self.state.data["created_evidence"][candidate_id] = record
        group["candidate_ids"].append(candidate_id)
        self.state.data["visual_creation_count"] = count + 1
        self.state.data["pending_candidate_id"] = candidate_id
        self.state.data["pending_retry_group_id"] = None
        self.state.data["inflight"] = None
        return {
            "ok": True,
            "candidate_id": candidate_id,
            "generation_group_id": str(group_id),
            "purpose": purpose,
            "representation": public_representation(representation),
            "inspection_required": True,
            "remaining_attempts": int(self.state.data["visual_creation_budget"])
            - count
            - 1,
        }

    def tool_inspect_candidate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(arguments.get("candidate_id") or "")
        candidate = self.state.data["created_evidence"].get(candidate_id)
        post_generation_candidate = False
        uploaded_audio_candidate = False
        if not candidate and self.phase in {"generation_review", "writeback"}:
            candidate = self.state.data["generation_evidence"].get(candidate_id)
            post_generation_candidate = candidate is not None
        if not candidate and self.phase == "writeback":
            uploaded = self.state.data["uploaded_evidence"].get(candidate_id)
            if isinstance(uploaded, Mapping) and uploaded.get("modality") == "audio":
                candidate = uploaded
                uploaded_audio_candidate = True
        if not candidate:
            raise PolicyRejected("unknown_candidate", f"unknown candidate: {candidate_id}")
        pending_candidate_id = self.state.data.get("pending_candidate_id")
        if pending_candidate_id is not None and pending_candidate_id != candidate_id:
            raise PolicyRejected("pending_candidate_mismatch", "inspect the pending candidate next")
        if candidate.get("inspection_id"):
            raise PolicyRejected("already_inspected", "candidate inspection is immutable")
        audio_view: Mapping[str, Any] | None = None
        audio_selector: dict[str, int | str] | None = None
        if uploaded_audio_candidate:
            selected_plan: Mapping[str, Any] = {}
            if isinstance(self.state.data.get("selected_generation"), Mapping):
                selected_plan = self._selected_generation_plan()
            elif isinstance(
                self.state.data.get("accepted_evidence_plan"), Mapping
            ):
                selected_plan = self.state.data["accepted_evidence_plan"]
            adopted_bindings = [
                row
                for row in selected_plan.get("ranked_bindings") or []
                if isinstance(row, Mapping)
                and row.get("source_kind") == "uploaded_material"
                and row.get("evidence_id") == candidate_id
                and row.get("modality") == "audio"
            ]
            if len(adopted_bindings) > 1:
                raise PolicyRejected(
                    "ambiguous_audio_writeback_source",
                    "selected plan contains duplicate uploaded-audio bindings",
                )
            adopted_representation_id = (
                str(adopted_bindings[0].get("representation_id") or "")
                if adopted_bindings
                else None
            )
            matching_views = [
                row
                for row in self.state.data.get("views", {}).values()
                if isinstance(row, Mapping)
                and row.get("source_kind") in {"uploaded_material", "uploaded_asset"}
                and row.get("evidence_id") == candidate_id
                and (
                    adopted_representation_id is None
                    or row.get("representation_id") == adopted_representation_id
                )
            ]
            if not matching_views:
                raise PolicyRejected(
                    "view_required",
                    "the exact selected uploaded-audio representation must be viewed "
                    "before metadata-level writeback inspection",
                )
            # A view call is an explicit selection action.  Use its latest
            # authorization so a subsequently selected time range cannot be
            # silently replaced by an earlier full-file view.
            audio_view = matching_views[-1]
            representation = self._record_representation(
                candidate, str(audio_view.get("representation_id") or "")
            )
            audio_selector = self._exact_audio_selector(
                audio_view.get("selector") or representation.get("selector"),
                field_name="viewed audio selector",
            )
        else:
            representation = self._record_representation(candidate)
        path = Path(str(representation["path"]))
        if post_generation_candidate and not any(
            row.get("evidence_id") == candidate_id
            and row.get("representation_id") == representation.get("representation_id")
            for row in self.state.data["views"].values()
        ):
            raise PolicyRejected(
                "view_required", "generation-derived evidence must be viewed before inspection"
            )
        inspection_id = self.state.next_id("inspection")
        purpose = candidate.get("purpose") or (
            "condition_adherence"
            if self.phase == "generation_review"
            else "memory_writeback"
        )
        required_use = (
            "condition_adherence"
            if self.phase == "generation_review"
            else "memory_writeback"
            if self.phase == "writeback"
            else None
        )
        if required_use and required_use not in arguments["requested_uses"]:
            raise PolicyRejected(
                "required_use_missing", f"inspection must request {required_use}"
            )
        if uploaded_audio_candidate and (
            arguments["requested_uses"] != ["memory_writeback"]
            or arguments["comparison_evidence_ids"]
        ):
            raise PolicyRejected(
                "invalid_audio_writeback_inspection",
                "uploaded audio inspection is metadata-only and must request only "
                "memory_writeback without perceptual comparisons",
            )
        if purpose == "condition_adherence":
            expected_target_id = self.state.data.get("condition_review_target_id")
            if candidate_id != expected_target_id:
                raise PolicyRejected(
                    "wrong_condition_review_target",
                    "inspect the host-designated representation of this generation",
                )
            plan = self.state.data.get("accepted_evidence_plan") or {}
            expected_comparisons = {
                binding["evidence_id"]
                for binding in plan.get("ranked_bindings") or []
                if binding["modality"] in self.perceivable_modalities
            }
            if set(arguments["comparison_evidence_ids"]) != expected_comparisons:
                raise PolicyRejected(
                    "incomplete_condition_comparison",
                    "condition review must compare every perceivable plan binding exactly once",
                    details={"expected_evidence_ids": sorted(expected_comparisons)},
                )
        requested_uses = [
            {"use_id": f"{inspection_id}.use.{index}", "use": use}
            for index, use in enumerate(arguments["requested_uses"], start=1)
        ]
        viewed_evidence_ids = {
            row["evidence_id"] for row in self.state.data["views"].values()
        }
        if not set(arguments["comparison_evidence_ids"]).issubset(
            viewed_evidence_ids
        ):
            raise PolicyRejected(
                "comparison_not_viewed", "comparison evidence must already be viewed"
            )
        if purpose == "condition_adherence":
            plan = self.state.data.get("accepted_evidence_plan") or {}
            requirement_by_id = {
                row["requirement_id"]: row
                for row in plan.get("requirements") or []
            }
            outcome_by_id = {
                row["binding_id"]: row
                for row in (
                    self.state.data.get("generation_bundle_private") or {}
                ).get("binding_outcomes")
                or []
            }
            checklist = []
            for binding in plan.get("ranked_bindings") or []:
                outcome = outcome_by_id.get(binding["binding_id"]) or {}
                applied_effect = outcome.get("applied_effect")
                is_perceivable = binding["modality"] in self.perceivable_modalities
                item = {
                    "check_id": f"condition.binding.{binding['binding_id']}",
                    "requirement": (
                        "The generation preserves the applied "
                        f"{applied_effect or binding['intended_effect']} effect for: "
                        + "; ".join(
                            requirement_by_id[requirement_id]["description"]
                            for requirement_id in binding["requirement_ids"]
                        )
                    ),
                    "severity": "critical" if binding["required"] else "major",
                    "binding_id": binding["binding_id"],
                    "comparison_evidence_ids": [binding["evidence_id"]],
                    "compilation_outcome": outcome.get("outcome"),
                    "applied_effect": applied_effect,
                }
                if not is_perceivable or outcome.get("outcome") == "not_applied":
                    item["deterministic_result"] = "capability_blocked"
                checklist.append(item)
            if self.prompt_authoring:
                for check_id, requirement in (
                    ("request_adherence", "The visible video preserves the original user's requested entities, action, relations, setting, camera intent and style: " + self.state.data["shot_requirement"]),
                    ("visual_quality", "The visible subject, interaction and motion are coherent and usable, without severe identity drift, malformed anatomy, object discontinuity or rendering collapse. Judge only actually observed frames."),
                ):
                    checklist.append({"check_id": "generation." + check_id,
                                      "requirement": requirement, "severity": "critical"})
        elif purpose == "memory_writeback" and uploaded_audio_candidate:
            speaker_binding = self._trusted_audio_speaker_binding(candidate)
            assert audio_selector is not None
            checklist = [
                {
                    "check_id": "audio.exact_selection",
                    "requirement": (
                        "The writeback is bound to the exact viewed audio "
                        "representation and selector recorded in metadata."
                    ),
                    "severity": "critical",
                    "deterministic_result": "pass",
                    "representation_id": representation["representation_id"],
                    "representation_sha256": representation["sha256"],
                    "selector": copy.deepcopy(audio_selector),
                },
                {
                    "check_id": "audio.speaker_lineage",
                    "requirement": (
                        "Trusted metadata binds this audio to exactly one known speaker."
                    ),
                    "severity": "critical",
                    "deterministic_result": "pass",
                    "speaker_binding": copy.deepcopy(speaker_binding),
                },
                {
                    "check_id": "audio.future_reuse_metadata",
                    "requirement": (
                        "Trusted description, stable facts, semantic roles, and reuse "
                        "scope make this exact selection useful for future continuity."
                    ),
                    "severity": "critical",
                },
            ]
        elif purpose == "memory_writeback":
            checklist = [
                {
                    "check_id": "memory.stability",
                    "requirement": "The evidence depicts a stable fact rather than a transient event.",
                    "severity": "critical",
                },
                {
                    "check_id": "memory.reuse_value",
                    "requirement": "The evidence is clear and useful for future continuity.",
                    "severity": "critical",
                },
            ]
        else:
            checklist = [
                {
                    "check_id": "technical_integrity",
                    "requirement": "The image is decodable and free of obvious corruption.",
                    "severity": "critical",
                },
                {
                    "check_id": "evidence_relevance",
                    "requirement": "The visible evidence supports its declared semantic roles.",
                    "severity": "critical",
                },
                {
                    "check_id": "opening_structure"
                    if purpose == "opening_frame_evidence"
                    else "reference_clarity",
                    "requirement": (
                        "The complete frame is suitable as an opening-state structural intent."
                        if purpose == "opening_frame_evidence"
                        else "The reference is clear, stable, and minimally ambiguous."
                    ),
                    "severity": "critical",
                },
            ]
        inspection = {
            "inspection_id": inspection_id,
            "candidate_id": candidate_id,
            "representation_id": representation["representation_id"],
            "checklist": checklist,
            "requested_uses": requested_uses,
            "comparison_evidence_ids": copy.deepcopy(
                arguments["comparison_evidence_ids"]
            ),
            "stage": (
                "post_generation"
                if post_generation_candidate
                else "writeback_metadata"
                if uploaded_audio_candidate
                else "planning"
            ),
            "evidence_ids": (
                [candidate_id]
                if uploaded_audio_candidate
                else [f"{inspection_id}.candidate"]
            ),
        }
        if uploaded_audio_candidate:
            inspection["source_kind"] = "uploaded_material"
            inspection["representation"] = public_representation(representation)
            assert audio_selector is not None
            inspection["selector"] = copy.deepcopy(audio_selector)
            inspection["authorization_id"] = str(
                (audio_view or {}).get("authorization_id") or ""
            )
            inspection["metadata_context"] = {
                "description": str(candidate.get("description") or ""),
                "semantic_roles": copy.deepcopy(
                    candidate.get("semantic_roles") or []
                ),
                "trusted_stable_facts": copy.deepcopy(
                    candidate.get("trusted_stable_facts") or []
                ),
                "reuse_scope": str(candidate.get("reuse_scope") or "story"),
                "one_shot": candidate.get("one_shot") is True,
                "speaker_binding": copy.deepcopy(
                    self._trusted_audio_speaker_binding(candidate)
                ),
            }
        # If the candidate is later selected and otherwise eligible, these are
        # the exact IDs update_memory_bank accepts as perceptual support.
        inspection["writeback_evidence_ids"] = copy.deepcopy(
            inspection["evidence_ids"]
        )
        self.state.data["inspections"][inspection_id] = inspection
        candidate["inspection_id"] = inspection_id
        self.state.data["pending_candidate_id"] = None
        self.state.data["pending_assessment_id"] = inspection_id
        attachments: list[dict[str, Any]] = []
        perception = None
        candidate_modality = str(
            representation.get("modality") or candidate.get("modality") or ""
        )
        if (
            not uploaded_audio_candidate
            and candidate_modality in self.perceivable_modalities
        ):
            attachment, perception = self._teacher_perception_attachment(
                representation,
                evidence_id=candidate_id,
                candidate_id=candidate_id,
                observation_evidence_id=f"{inspection_id}.candidate",
            )
            attachments.append(attachment)
        result = {
            "ok": True,
            **copy.deepcopy(inspection),
            "remaining_condition_revisions": max(
                0,
                int(self.state.data["max_condition_revisions"])
                - int(self.state.data.get("revision_count", 0)),
            ),
            "_media_attachments": attachments,
        }
        if perception is not None:
            result["perception"] = perception
        return result

    def tool_submit_candidate_assessment(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        candidate_id = str(arguments.get("candidate_id") or "")
        candidate = self.state.data["created_evidence"].get(candidate_id)
        post_generation_candidate = False
        uploaded_audio_candidate = False
        if not candidate and self.phase in {"generation_review", "writeback"}:
            candidate = self.state.data["generation_evidence"].get(candidate_id)
            post_generation_candidate = candidate is not None
        if not candidate and self.phase == "writeback":
            uploaded = self.state.data["uploaded_evidence"].get(candidate_id)
            if isinstance(uploaded, Mapping) and uploaded.get("modality") == "audio":
                candidate = uploaded
                uploaded_audio_candidate = True
        if not candidate or not candidate.get("inspection_id"):
            raise PolicyRejected("inspection_required", "candidate must be inspected first")
        if candidate.get("assessment_id"):
            raise PolicyRejected("already_assessed", "candidate assessment is immutable")
        inspection_id = str(arguments["inspection_id"])
        if self.state.data.get("pending_assessment_id") != inspection_id:
            raise PolicyRejected("pending_assessment_mismatch", "assess the pending inspection next")
        inspection = self.state.data["inspections"].get(inspection_id)
        if not inspection or inspection_id != candidate.get("inspection_id"):
            raise PolicyRejected("inspection_mismatch", "assessment changed inspection lineage")
        expected_checks = {row["check_id"] for row in inspection["checklist"]}
        actual_check_ids = [row["check_id"] for row in arguments["check_results"]]
        if len(actual_check_ids) != len(expected_checks) or set(
            actual_check_ids
        ) != expected_checks:
            raise PolicyRejected(
                "incomplete_checklist",
                "assessment must cover every frozen checklist item exactly once",
            )
        expected_uses = {row["use_id"] for row in inspection["requested_uses"]}
        actual_use_ids = [row["use_id"] for row in arguments["use_results"]]
        if len(actual_use_ids) != len(expected_uses) or set(actual_use_ids) != expected_uses:
            raise PolicyRejected(
                "incomplete_use_results",
                "assessment must cover every requested use exactly once",
            )
        scope = str(arguments["assessment_scope"])
        quality = str(arguments["assessment_result"])
        action = str(arguments["decision"])
        expected_scope = (
            "condition_adherence"
            if self.phase == "generation_review"
            else "memory_writeback"
            if self.phase == "writeback"
            else "created_evidence_suitability"
        )
        if scope != expected_scope:
            raise PolicyRejected("assessment_scope_mismatch", "assessment scope changed FSM stage")
        allowed_decisions = {
            "created_evidence_suitability": {"select", "retry"},
            "condition_adherence": {
                "accept_conditions",
                "revise_evidence_plan",
                "cannot_fix_with_memory",
            },
            "memory_writeback": {"select"},
        }[scope]
        if action not in allowed_decisions:
            raise PolicyRejected("invalid_next_action", "assessment decision is invalid for its scope")
        use_results = [
            str(row.get("result") or "") for row in arguments["use_results"]
        ]
        use_result_values = set(use_results)
        check_result_by_id = {
            str(row.get("check_id") or ""): str(row.get("result") or "")
            for row in arguments["check_results"]
        }
        check_results = list(check_result_by_id.values())
        check_row_by_id = {
            str(row.get("check_id") or ""): row
            for row in arguments["check_results"]
        }
        deterministic_capability_blocked = False
        for item in inspection["checklist"]:
            deterministic = item.get("deterministic_result")
            if deterministic is not None and check_result_by_id.get(
                item["check_id"]
            ) != deterministic:
                raise PolicyRejected(
                    "deterministic_check_changed",
                    "assessment changed a deterministic compilation result",
                )
            if deterministic == "capability_blocked":
                deterministic_capability_blocked = True
                if (check_row_by_id.get(item["check_id"]) or {}).get(
                    "evidence_ids"
                ):
                    raise PolicyRejected(
                        "capability_blocked_claims_evidence",
                        "a non-perceivable condition check cannot cite perceived evidence",
                    )
        critical_check_failures = {
            row["check_id"]
            for row in inspection["checklist"]
            if row.get("severity") == "critical"
            and check_result_by_id.get(row["check_id"])
            in {"fail", "uncertain", "capability_blocked"}
        }
        unfulfilled_required_count = 0
        if scope == "condition_adherence":
            bundle = self.state.data.get("generation_bundle_private") or {}
            current_plan = self._evidence_plan_by_id(bundle.get("plan_id"))
            unfulfilled_required_count = self._unfulfilled_required_count(current_plan)
        condition_checks_pass = (
            all(result == "pass" for result in check_results)
            and unfulfilled_required_count == 0
        )
        if action == "accept_conditions" and (
            quality != "pass"
            or not condition_checks_pass
            or any(result != "pass" for result in use_results)
        ):
            raise PolicyRejected(
                "failed_condition_cannot_be_accepted",
                "accept_conditions requires every required and optional condition to pass",
            )

        # Condition adherence and memory writeback each request one holistic
        # use.  Its result must agree with the frozen checklist; otherwise a
        # caller could label failed condition/stability/reuse checks as a
        # passing use and smuggle them through a terminal decision.
        if scope in {"condition_adherence", "memory_writeback"}:
            expected_use_result = (
                "capability_blocked"
                if scope == "condition_adherence"
                and deterministic_capability_blocked
                else self._aggregate_check_result(
                    check_results,
                    unfulfilled_required_count=unfulfilled_required_count,
                )
            )
            if any(result != expected_use_result for result in use_results):
                raise PolicyRejected(
                    "assessment_use_result_mismatch",
                    "requested-use result contradicts the frozen checklist",
                    details={"expected_use_result": expected_use_result},
                )

        derived_quality = (
            "capability_limited"
            if scope == "condition_adherence" and deterministic_capability_blocked
            else self._overall_assessment_result(
                [*check_results, *use_results]
                + (["fail"] if unfulfilled_required_count else [])
            )
        )
        if quality != derived_quality:
            raise PolicyRejected(
                "assessment_result_mismatch",
                "assessment_result contradicts checklist or requested-use results",
                details={"expected_assessment_result": derived_quality},
            )
        if scope == "condition_adherence" and deterministic_capability_blocked:
            if (
                action != "cannot_fix_with_memory"
                or "capability_blocked" not in use_result_values
            ):
                raise PolicyRejected(
                    "nonperceivable_condition_requires_limited_selection",
                    "a non-perceivable condition requires cannot_fix_with_memory "
                    "and a capability_blocked use result",
                )
        if action == "revise_evidence_plan" and (
            not (
                use_result_values.intersection({"fail", "uncertain"})
                or {
                    check_id
                    for check_id in critical_check_failures
                    if check_result_by_id.get(check_id) in {"fail", "uncertain"}
                }
            )
            or not arguments.get("retry_instruction")
            or not arguments.get("changed_axes")
            or quality != "fail"
        ):
            raise PolicyRejected(
                "revision_not_justified",
                "plan revision requires failed or uncertain condition evidence and a concrete change",
            )
        if action == "cannot_fix_with_memory" and (
            not (
                use_result_values.intersection(
                    {"fail", "uncertain", "capability_blocked"}
                )
                or critical_check_failures
            )
            or quality not in {"fail", "capability_limited"}
        ):
            raise PolicyRejected(
                "unfixable_condition_not_justified",
                "cannot_fix_with_memory requires failed or blocked condition evidence",
            )
        if action == "cannot_fix_with_memory" and (
            arguments.get("retry_instruction") is not None
            or bool(arguments.get("changed_axes"))
        ):
            raise PolicyRejected(
                "unfixable_condition_has_retry",
                "cannot_fix_with_memory must not prescribe another policy retry",
            )
        if scope == "memory_writeback" and action == "select" and quality != "pass":
            raise PolicyRejected(
                "failed_writeback_cannot_be_selected",
                "memory writeback selection requires every stability and reuse check to pass",
            )
        if scope == "created_evidence_suitability":
            if action == "select" and quality == "fail":
                raise PolicyRejected(
                    "failed_candidate_cannot_be_selected",
                    "failed created evidence cannot be selected",
                )
            if action == "retry" and quality != "fail":
                raise PolicyRejected(
                    "passing_candidate_cannot_be_retried",
                    "retry requires failed or uncertain created-evidence suitability",
                )
        assessment_id = self.state.next_id("assessment")
        assessment = {
            "assessment_id": assessment_id,
            "candidate_id": candidate_id,
            "inspection_id": inspection_id,
            "quality": quality,
            "assessment_scope": scope,
            "check_results": copy.deepcopy(arguments["check_results"]),
            "use_results": copy.deepcopy(arguments["use_results"]),
            "next_action": action,
            "rationale": " ".join(
                str(row.get("rationale") or "")
                for row in arguments["use_results"]
            ),
        }
        self.state.data["assessments"][assessment_id] = assessment
        candidate["assessment_id"] = assessment_id
        self.state.data["pending_assessment_id"] = None
        generation_summary = None
        if scope == "condition_adherence":
            bundle = self.state.data.get("generation_bundle_private") or {}
            generation_summary = {
                "candidate_bundle_id": bundle.get("bundle_id"),
                "plan_id": bundle.get("plan_id"),
                "plan_sha256": bundle.get("plan_sha256"),
                "generation_reservation_id": bundle.get(
                    "generation_reservation_id"
                ),
                "base_prompt_artifact_id": bundle.get(
                    "base_prompt_artifact_id"
                ),
                "base_prompt_artifact_sha256": bundle.get(
                    "base_prompt_artifact_sha256"
                ),
                "wire_prompt_artifact_id": bundle.get(
                    "wire_prompt_artifact_id"
                ),
                "wire_prompt_artifact_sha256": bundle.get(
                    "wire_prompt_artifact_sha256"
                ),
                "compilation_receipt_id": bundle.get("compilation_receipt_id"),
                "compilation_receipt_sha256": bundle.get(
                    "compilation_receipt_sha256"
                ),
                "condition_signature_sha256": bundle.get(
                    "condition_signature_sha256"
                ),
                "generation_id": bundle.get("generation_id"),
                "generation_output_sha256": bundle.get(
                    "generation_output_sha256"
                ),
                "review_target_evidence_id": bundle.get(
                    "review_target_evidence_id"
                ),
                "assessment_id": assessment_id,
                "assessment_result": quality,
                "decision": action,
            }
            self.state.data["reviewed_generations"].append(
                copy.deepcopy(generation_summary)
            )
        group = (
            {
                "candidate_ids": [candidate_id],
                "selected_candidate_id": None,
            }
            if post_generation_candidate or uploaded_audio_candidate
            else self.state.data["creation_groups"][candidate["creation_group_id"]]
        )
        assessed_ids = [
            evidence_id
            for evidence_id in group["candidate_ids"]
            if (self.state.data["created_evidence"].get(evidence_id) or {}).get(
                "assessment_id"
            )
        ]
        if post_generation_candidate and scope == "condition_adherence":
            assessed_ids = [
                str(row["review_target_evidence_id"])
                for row in self.state.data["reviewed_generations"]
            ]
        elif post_generation_candidate or uploaded_audio_candidate:
            assessed_ids = [candidate_id]
        canonical_ranking: list[str] | None = None
        if scope == "condition_adherence":
            canonical_ranking, canonical_scores = (
                self._canonical_condition_candidate_ranking()
            )
            if arguments["candidate_ranking"] != canonical_ranking:
                raise PolicyRejected(
                    "noncanonical_candidate_ranking",
                    "candidate ranking differs from deterministic best-available order",
                    details={
                        "expected_candidate_ranking": canonical_ranking,
                        "ranking_scores": {
                            key: list(value) for key, value in canonical_scores.items()
                        },
                    },
                )
        elif set(arguments["candidate_ranking"]) != set(assessed_ids):
            raise PolicyRejected(
                "incomplete_candidate_ranking",
                "candidate ranking must contain every assessed candidate in the pool",
            )
        terminal_selected_generation = None
        if action in {"accept_conditions", "cannot_fix_with_memory"}:
            selected_id = str(arguments.get("selected_candidate_id") or "")
            if selected_id not in assessed_ids:
                raise PolicyRejected(
                    "invalid_generation_selection",
                    "terminal condition decision must select an assessed generation candidate",
                )
            if action == "accept_conditions" and selected_id != candidate_id:
                raise PolicyRejected(
                    "passing_candidate_not_selected",
                    "accept_conditions must select the currently passing candidate",
                )
            if not canonical_ranking or selected_id != canonical_ranking[0]:
                raise PolicyRejected(
                    "noncanonical_generation_selection",
                    "terminal condition decision must select the deterministic best candidate",
                    details={
                        "expected_selected_candidate_id": (
                            canonical_ranking[0] if canonical_ranking else None
                        )
                    },
                )
            terminal_selected_generation = next(
                row
                for row in self.state.data["reviewed_generations"]
                if row["review_target_evidence_id"] == selected_id
            )
        elif scope == "condition_adherence" and arguments.get(
            "selected_candidate_id"
        ) is not None:
            raise PolicyRejected(
                "unexpected_generation_selection",
                "a revising condition decision cannot select a final generation",
            )
        if action == "select":
            selected_id = str(arguments.get("selected_candidate_id") or "")
            if selected_id not in group["candidate_ids"]:
                raise PolicyRejected("invalid_selection", "selection is outside the creation group")
            if uploaded_audio_candidate:
                if selected_id != candidate_id:
                    raise PolicyRejected(
                        "invalid_selection",
                        "uploaded audio assessment can select only its exact candidate",
                    )
                candidate["selection_authorization_id"] = inspection.get(
                    "authorization_id"
                )
                group["selected_candidate_id"] = selected_id
            else:
                selected = self.state.data["created_evidence"].get(selected_id) or {}
                if not selected.get("assessment_id"):
                    if selected_id != candidate_id:
                        raise PolicyRejected("unassessed_selection", "selected candidate was not assessed")
                group["selected_candidate_id"] = selected_id
                representation = self._record_representation(
                    self.state.data["generation_evidence"].get(selected_id)
                    or self.state.data["created_evidence"][selected_id]
                )
                authorization_id = self.state.next_id("selectionauth")
                key = "|".join(
                    [
                        "generation_derived" if post_generation_candidate else "generated_evidence",
                        selected_id,
                        str(representation["representation_id"]),
                        authorization_id,
                    ]
                )
                self.state.data["views"][key] = {
                    "source_kind": "generation_derived"
                    if post_generation_candidate
                    else "generated_evidence",
                    "evidence_id": selected_id,
                    "representation_id": representation["representation_id"],
                    "authorization_id": authorization_id,
                }
                candidate["selection_authorization_id"] = authorization_id
        elif action == "retry":
            if arguments.get("selected_candidate_id") is not None:
                raise PolicyRejected("retry_has_selection", "retry cannot select a candidate")
            self.state.data["pending_retry_group_id"] = candidate["creation_group_id"]
        elif action == "revise_evidence_plan":
            max_revisions = int(self.state.data["max_condition_revisions"])
            revision_count = int(self.state.data.get("revision_count", 0))
            if revision_count >= max_revisions:
                raise PolicyRejected(
                    "condition_revision_budget_exhausted",
                    "condition revision budget is exhausted; accept or mark unfixable by memory",
                )
            self.state.data["revision_count"] = revision_count + 1
            self.state.data["condition_review"] = copy.deepcopy(assessment)
            self.state.transition("generation_review", "planning")
        elif action in {"accept_conditions", "cannot_fix_with_memory"}:
            self.state.data["condition_review"] = copy.deepcopy(assessment)
            self.state.data["selected_generation"] = copy.deepcopy(
                terminal_selected_generation
            )
            self.state.transition("generation_review", "awaiting_final_evidence")
        return {
            "ok": True,
            "assessment_id": assessment_id,
            "candidate_id": candidate_id,
            "quality": quality,
            "next_action": action,
            "selected_candidate_id": (
                arguments.get("selected_candidate_id")
                if scope == "condition_adherence"
                else group.get("selected_candidate_id")
            ),
            "authorization_id": candidate.get("selection_authorization_id"),
            "phase": self.phase,
            "remaining_condition_revisions": max(
                0,
                int(self.state.data["max_condition_revisions"])
                - int(self.state.data.get("revision_count", 0)),
            ),
        }

    def tool_submit_evidence_plan(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._assert_request(arguments)
        if arguments["raw_request_sha256"] != self.state.data["shot_requirement_sha256"]:
            raise PolicyRejected("raw_request_hash_mismatch", "EvidencePlan changed the raw request")
        if arguments["condition_budget_id"] != self.condition_budget.budget_id:
            raise PolicyRejected(
                "condition_budget_mismatch", "EvidencePlan changed condition budget"
            )
        unfinished = [
            group_id
            for group_id, group in self.state.data["creation_groups"].items()
            if not group.get("selected_candidate_id")
        ]
        if unfinished:
            raise PolicyRejected(
                "unfinished_creation_groups",
                "every visual creation group must select an assessed candidate",
                details={"group_ids": unfinished},
            )
        try:
            def parse_requirements(rows: Sequence[Mapping[str, Any]]) -> tuple[EvidenceRequirement, ...]:
                return tuple(
                EvidenceRequirement(
                    **{
                        **row,
                        "semantic_roles": tuple(row["semantic_roles"]),
                        "acceptable_modalities": tuple(row["acceptable_modalities"]),
                    }
                )
                for row in rows
                )

            def parse_bindings(rows: Sequence[Mapping[str, Any]]) -> tuple[EvidenceBinding, ...]:
                return tuple(
                EvidenceBinding(
                    **{
                        **row,
                        "requirement_ids": tuple(row["requirement_ids"]),
                        "semantic_roles": tuple(row["semantic_roles"]),
                    }
                )
                for row in rows
                )

            requirements = parse_requirements(arguments["requirements"])
            bindings = parse_bindings(arguments["ranked_bindings"])
            parent_dict = self.state.data.get("accepted_evidence_plan")
            parent_plan = None
            if parent_dict is not None:
                parent_plan = EvidencePlan(
                    plan_id=parent_dict["plan_id"],
                    parent_plan_id=parent_dict["parent_plan_id"],
                    version=parent_dict["version"],
                    request_id=parent_dict["request_id"],
                    raw_request_sha256=parent_dict["raw_request_sha256"],
                    condition_budget_id=parent_dict["condition_budget_id"],
                    requirements=parse_requirements(parent_dict["requirements"]),
                    ranked_bindings=parse_bindings(parent_dict["ranked_bindings"]),
                    unfulfilled_requirement_ids=tuple(
                        parent_dict["unfulfilled_requirement_ids"]
                    ),
                    selection_mode=parent_dict["selection_mode"],
                    rationale=parent_dict["rationale"],
                    generation_prompt=parent_dict.get("generation_prompt"),
                    attempt_kind=parent_dict.get("attempt_kind"),
                )
            plan = EvidencePlan(
                plan_id=self.state.next_id("plan"),
                parent_plan_id=arguments["parent_plan_id"],
                version=int(arguments["version"]),
                request_id=self.state.data["request_id"],
                raw_request_sha256=self.state.data["shot_requirement_sha256"],
                condition_budget_id=self.condition_budget.budget_id,
                requirements=requirements,
                ranked_bindings=bindings,
                unfulfilled_requirement_ids=tuple(
                    arguments.get("unfulfilled_requirement_ids") or []
                ),
                selection_mode=str(arguments.get("selection_mode") or ""),
                rationale=str(arguments.get("rationale") or ""),
                generation_prompt=arguments.get("generation_prompt") if self.prompt_authoring else None,
                attempt_kind=arguments.get("attempt_kind") if self.prompt_authoring else None,
            )
            plan_dict = plan.to_dict(
                self.condition_budget,
                expected_raw_request_sha256=self.state.data[
                    "shot_requirement_sha256"
                ],
                parent_plan=parent_plan,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyRejected("invalid_evidence_plan", str(exc)) from exc
        for binding in plan_dict["ranked_bindings"]:
            self._resolve_view_binding(binding)
            record = self._evidence_record(
                binding["source_kind"], binding["evidence_id"]
            )
            if record.get("creation_group_id"):
                group = self.state.data["creation_groups"][
                    record["creation_group_id"]
                ]
                if group.get("selected_candidate_id") != binding["evidence_id"]:
                    raise PolicyRejected(
                        "unselected_created_evidence",
                        "plan references an unselected generated candidate",
                    )
                assessment = self.state.data["assessments"].get(
                    record.get("assessment_id")
                ) or {}
                if assessment.get("quality") not in {"pass", "capability_limited"}:
                    raise PolicyRejected(
                        "unapproved_created_evidence",
                        "plan references generated evidence without usable assessment",
                    )
        plan_dict["plan_sha256"] = _sha256_json(plan_dict)
        if self.state.data.get("accepted_evidence_plan") is not None:
            self.state.data["plan_history"].append(
                copy.deepcopy(self.state.data["accepted_evidence_plan"])
            )
        self.state.data["accepted_evidence_plan"] = copy.deepcopy(plan_dict)
        self.state.transition("planning", "awaiting_generation")
        return {"ok": True, "phase": self.phase, "evidence_plan": plan_dict}

    def reserve_generation(self, compilation: Mapping[str, Any]) -> dict[str, Any]:
        """Durably authorize exactly one black-box call for compiled conditions.

        This is a host-only preflight API, not an agent tool.  The host must
        reserve before invoking the moving-media generator and must use the
        returned ``generation_id`` as the provider idempotency key.  Replaying
        the exact same preflight returns the existing reservation; it never
        authorizes a second call.
        """

        if self.phase != "awaiting_generation":
            raise PolicyRejected(
                "wrong_phase", "generation can be reserved only for an accepted plan"
            )
        before = copy.deepcopy(self.state.data)
        value = copy.deepcopy(dict(compilation))
        allowed_fields = {
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
            "compiled_conditions",
        }
        if self.prompt_authoring:
            allowed_fields.update({"generator_prompt_sha256", "generator_seed"})
        if set(value) != allowed_fields:
            raise PolicyRejected(
                "invalid_generation_preflight",
                "generation preflight must contain exactly the portable reservation fields",
            )
        plan = self.state.data.get("accepted_evidence_plan") or {}
        if (
            value.get("request_id") != self.state.data["request_id"]
            or value.get("raw_request_sha256")
            != self.state.data["shot_requirement_sha256"]
            or value.get("plan_id") != plan.get("plan_id")
            or value.get("plan_sha256") != plan.get("plan_sha256")
        ):
            raise PolicyRejected(
                "generation_lineage_mismatch",
                "compiled conditions are not bound to this request and plan",
            )
        base_prompt_artifact = {
            "base_prompt_artifact_id": _require_identifier(
                value.get("base_prompt_artifact_id"),
                "base_prompt_artifact_id",
            ),
            "base_prompt_artifact_sha256": _require_sha256(
                value.get("base_prompt_artifact_sha256"),
                "base_prompt_artifact_sha256",
            ),
        }
        base_prompt_lock = self.state.data.get("base_prompt_artifact_lock")
        if not self.prompt_authoring and base_prompt_lock is not None and base_prompt_artifact != base_prompt_lock:
            raise PolicyRejected(
                "base_prompt_artifact_changed",
                "condition revision cannot change the shot-level base prompt artifact",
            )
        wire_prompt_artifact = {
            "wire_prompt_artifact_id": _require_identifier(
                value.get("wire_prompt_artifact_id"),
                "wire_prompt_artifact_id",
            ),
            "wire_prompt_artifact_sha256": _require_sha256(
                value.get("wire_prompt_artifact_sha256"),
                "wire_prompt_artifact_sha256",
            ),
        }
        compilation_receipt = {
            "compilation_receipt_id": _require_identifier(
                value.get("compilation_receipt_id"), "compilation_receipt_id"
            ),
            "compilation_receipt_sha256": _require_sha256(
                value.get("compilation_receipt_sha256"),
                "compilation_receipt_sha256",
            ),
        }
        binding_outcomes = _normalize_binding_outcomes(
            plan, value.get("binding_outcomes")
        )
        private_conditions, canonical_conditions = _normalize_compiled_conditions(
            value.get("compiled_conditions"),
            plan=plan,
            binding_outcomes=binding_outcomes,
        )
        effective_input = {}
        if self.prompt_authoring:
            effective_input["generator_prompt_sha256"] = _require_sha256(value.get("generator_prompt_sha256"), "generator_prompt_sha256")
            seed = value.get("generator_seed")
            if type(seed) is not int or not 0 <= seed < 2 ** 32:
                raise PolicyRejected("invalid_generator_seed", "seed must be an actual host-owned uint32")
            effective_input["generator_seed"] = seed
        condition_signature = _sha256_json({
            "schema_version": "storymem.effective-generation-input.v2" if self.prompt_authoring else "storymem.compiled-reference-conditions.v1",
            "conditions": canonical_conditions,
            **effective_input,
        })
        normalized_input = {
            "request_id": self.state.data["request_id"],
            "raw_request_sha256": self.state.data["shot_requirement_sha256"],
            "plan_id": plan.get("plan_id"),
            "plan_sha256": plan.get("plan_sha256"),
            **base_prompt_artifact,
            **wire_prompt_artifact,
            **compilation_receipt,
            "binding_outcomes": binding_outcomes,
            "canonical_conditions": canonical_conditions,
            **effective_input,
        }
        reservation_input_sha256 = _sha256_json(normalized_input)
        active_id = self.state.data.get("active_generation_reservation_id")
        reservations = self.state.data["generation_reservations"]
        if active_id is not None:
            active = reservations.get(str(active_id)) or {}
            if active.get("reservation_input_sha256") != reservation_input_sha256:
                raise PolicyRejected(
                    "generation_already_reserved",
                    "a different black-box generation is already durably reserved",
                )
            return {
                "ok": True,
                "replayed": True,
                "invoke_allowed": False,
                "generation_reservation_id": active["generation_reservation_id"],
                "generation_id": active["generation_id"],
                "condition_signature_sha256": active[
                    "condition_signature_sha256"
                ],
            }
        if condition_signature in self.state.data["used_condition_signatures"]:
            raise PolicyRejected(
                "duplicate_condition_signature",
                "the same compiled reference conditions cannot authorize another call",
            )
        reservation_id = self.state.next_id("generationreservation")
        local_generation_id = self.state.next_id("generation")
        scope_digest = hashlib.sha256(
            (
                str(self.state.data["session_id"])
                + "\x00"
                + str(self.state.data["request_id"])
            ).encode("utf-8")
        ).hexdigest()[:24]
        generation_id = (
            f"generation_{scope_digest}_{local_generation_id.rsplit('_', 1)[-1]}"
        )
        reservation = {
            "generation_reservation_id": reservation_id,
            "generation_id": generation_id,
            "status": "reserved",
            "reservation_input_sha256": reservation_input_sha256,
            **normalized_input,
            "condition_signature_sha256": condition_signature,
            "compiled_conditions_private": private_conditions,
            "candidate_bundle_id": None,
            "generation_output_sha256": None,
        }
        reservations[reservation_id] = reservation
        self.state.data["active_generation_reservation_id"] = reservation_id
        self.state.data["used_condition_signatures"].append(condition_signature)
        if base_prompt_lock is None:
            self.state.data["base_prompt_artifact_lock"] = copy.deepcopy(
                base_prompt_artifact
            )
        try:
            # This durable checkpoint must complete before the host is allowed
            # to invoke the black-box generator.
            self.state.save()
        except StateConflictError:
            self.state.reload()
            active_id = self.state.data.get("active_generation_reservation_id")
            active = self.state.data["generation_reservations"].get(str(active_id))
            if isinstance(active, Mapping) and active.get(
                "reservation_input_sha256"
            ) == reservation_input_sha256:
                return {
                    "ok": True,
                    "replayed": True,
                    "invoke_allowed": False,
                    "generation_reservation_id": active[
                        "generation_reservation_id"
                    ],
                    "generation_id": active["generation_id"],
                    "condition_signature_sha256": active[
                        "condition_signature_sha256"
                    ],
                }
            raise
        except Exception:
            self.state.reload()
            raise
        return {
            "ok": True,
            "replayed": False,
            "invoke_allowed": True,
            "generation_reservation_id": reservation_id,
            "generation_id": generation_id,
            "condition_signature_sha256": condition_signature,
        }

    def attach_compilation_feedback(
        self, feedback: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return a deterministic adapter rejection to policy planning.

        This is a host-only boundary, not an agent tool.  It is legal only
        before a generation reservation exists and consumes the same bounded
        condition-revision budget as a post-generation condition assessment.
        Backend names, layouts, and diagnostics never enter the public
        observation; the model sees only a binding-scoped neutral reason.
        """

        if self.phase != "awaiting_generation":
            raise PolicyRejected(
                "wrong_phase",
                "compilation feedback is legal only for an accepted plan awaiting generation",
            )
        if self.state.data.get("active_generation_reservation_id") is not None:
            raise PolicyRejected(
                "generation_already_reserved",
                "compilation feedback cannot replace a durably reserved generation",
            )
        value = copy.deepcopy(dict(feedback))
        allowed_fields = {
            "request_id",
            "raw_request_sha256",
            "plan_id",
            "plan_sha256",
            "binding_id",
            "reason_code",
        }
        if set(value) != allowed_fields:
            raise PolicyRejected(
                "invalid_compilation_feedback",
                "compilation feedback must contain exactly the portable lineage and reason fields",
            )
        plan = self.state.data.get("accepted_evidence_plan") or {}
        if (
            value.get("request_id") != self.state.data["request_id"]
            or value.get("raw_request_sha256")
            != self.state.data["shot_requirement_sha256"]
            or value.get("plan_id") != plan.get("plan_id")
            or value.get("plan_sha256") != plan.get("plan_sha256")
        ):
            raise PolicyRejected(
                "compilation_feedback_lineage_mismatch",
                "compilation feedback is not bound to the active request and plan",
            )
        try:
            reason_code = validate_neutral_reason_code(value.get("reason_code"))
        except ValueError as exc:
            raise PolicyRejected("invalid_compilation_feedback", str(exc)) from exc
        if reason_code == "condition.applied" or reason_code == "condition.fallback":
            raise PolicyRejected(
                "invalid_compilation_feedback",
                "compilation failure requires a non-applied neutral reason",
            )
        binding_id = value.get("binding_id")
        binding_ids = {
            str(row.get("binding_id") or "")
            for row in plan.get("ranked_bindings") or []
            if isinstance(row, Mapping)
        }
        if binding_id is not None and binding_id not in binding_ids:
            raise PolicyRejected(
                "invalid_compilation_feedback",
                "compilation feedback names an unknown binding",
            )
        revision_count = int(self.state.data.get("revision_count", 0))
        max_revisions = int(self.state.data["max_condition_revisions"])
        if revision_count >= max_revisions:
            raise PolicyRejected(
                "condition_revision_budget_exhausted",
                "condition compilation failed after the revision budget was exhausted",
            )

        before = copy.deepcopy(self.state.data)
        phase_before = self.phase
        feedback_id = self.state.next_id("compilationfeedback")
        private_record = {
            "feedback_id": feedback_id,
            **value,
        }
        feedback_sha256 = _sha256_json(private_record)
        self.state.data.setdefault(
            "compilation_feedback_history_private", []
        ).append(copy.deepcopy(private_record))
        self.state.data["revision_count"] = revision_count + 1
        observation = {
            "ok": True,
            "phase": "planning",
            "feedback_kind": "condition_compilation_failure",
            "binding_id": binding_id,
            "reason_code": reason_code,
            "revision_required": True,
            "remaining_condition_revisions": max_revisions - revision_count - 1,
        }
        self.state.data["condition_review"] = copy.deepcopy(observation)
        self.state.transition("awaiting_generation", "planning")
        self.state.data["host_events"].append(
            {
                "event_id": self.state.next_id("hostevent"),
                "name": "attach_compilation_feedback",
                "phase_before": phase_before,
                "phase_after": self.phase,
                "feedback_id": feedback_id,
                "feedback_sha256": feedback_sha256,
                "observation": copy.deepcopy(observation),
            }
        )
        try:
            self.state.save()
        except Exception:
            self.state.reload()
            raise
        return observation

    def attach_generation_evidence(self, bundle: Mapping[str, Any]) -> dict[str, Any]:
        """Host-only injection point; it is intentionally absent from TOOL_NAMES."""

        before = copy.deepcopy(self.state.data)
        phase_before = self.phase
        value = copy.deepcopy(dict(bundle))
        bundle_kind = str(value.get("bundle_kind") or "")
        expected_phase = {
            "candidate_feedback": "awaiting_generation",
            "final_accepted": "awaiting_final_evidence",
        }.get(bundle_kind)
        if expected_phase is None or self.phase != expected_phase:
            raise PolicyRejected(
                "wrong_phase",
                "bundle_kind must match the pending candidate-review or final-evidence phase",
            )
        plan = self.state.data.get("accepted_evidence_plan") or {}
        if bundle_kind == "final_accepted":
            selected_generation = self.state.data.get("selected_generation") or {}
            selected_plan_id = selected_generation.get("plan_id")
            matching_plans = [
                row
                for row in [
                    *(self.state.data.get("plan_history") or []),
                    self.state.data.get("accepted_evidence_plan"),
                ]
                if isinstance(row, Mapping) and row.get("plan_id") == selected_plan_id
            ]
            if len(matching_plans) != 1:
                raise PolicyRejected(
                    "selected_plan_missing",
                    "the selected generation has no unique EvidencePlan lineage",
                )
            plan = matching_plans[0]
        if (
            value.get("request_id", value.get("shot_request_id"))
            != self.state.data["request_id"]
            or value.get("raw_request_sha256")
            != self.state.data["shot_requirement_sha256"]
            or value.get("plan_id") != plan.get("plan_id")
            or value.get("plan_sha256") != plan.get("plan_sha256")
        ):
            raise PolicyRejected(
                "generation_lineage_mismatch",
                "generation evidence is not bound to this request and plan",
            )
        base_prompt_artifact = {
            "base_prompt_artifact_id": _require_identifier(
                value.get("base_prompt_artifact_id"),
                "base_prompt_artifact_id",
            ),
            "base_prompt_artifact_sha256": _require_sha256(
                value.get("base_prompt_artifact_sha256"),
                "base_prompt_artifact_sha256",
            ),
        }
        wire_prompt_artifact = {
            "wire_prompt_artifact_id": _require_identifier(
                value.get("wire_prompt_artifact_id"),
                "wire_prompt_artifact_id",
            ),
            "wire_prompt_artifact_sha256": _require_sha256(
                value.get("wire_prompt_artifact_sha256"),
                "wire_prompt_artifact_sha256",
            ),
        }
        generation_lineage = {
            "generation_reservation_id": _require_identifier(
                value.get("generation_reservation_id"),
                "generation_reservation_id",
            ),
            "compilation_receipt_id": _require_identifier(
                value.get("compilation_receipt_id"), "compilation_receipt_id"
            ),
            "compilation_receipt_sha256": _require_sha256(
                value.get("compilation_receipt_sha256"),
                "compilation_receipt_sha256",
            ),
            "condition_signature_sha256": _require_sha256(
                value.get("condition_signature_sha256"),
                "condition_signature_sha256",
            ),
            "generation_id": _require_identifier(
                value.get("generation_id"), "generation_id"
            ),
            "generation_output_sha256": _require_sha256(
                value.get("generation_output_sha256"),
                "generation_output_sha256",
            ),
        }
        if bundle_kind == "candidate_feedback":
            reservation_id = generation_lineage["generation_reservation_id"]
            active_id = self.state.data.get("active_generation_reservation_id")
            reservation = self.state.data["generation_reservations"].get(
                reservation_id
            )
            if (
                active_id != reservation_id
                or not isinstance(reservation, Mapping)
                or reservation.get("status") != "reserved"
            ):
                raise PolicyRejected(
                    "missing_generation_reservation",
                    "candidate feedback has no matching active generation reservation",
                )
            reserved_lineage_keys = (
                "generation_reservation_id",
                "compilation_receipt_id",
                "compilation_receipt_sha256",
                "condition_signature_sha256",
                "generation_id",
            )
            if (
                any(
                    generation_lineage[key] != reservation.get(key)
                    for key in reserved_lineage_keys
                )
                or base_prompt_artifact
                != {
                    "base_prompt_artifact_id": reservation.get(
                        "base_prompt_artifact_id"
                    ),
                    "base_prompt_artifact_sha256": reservation.get(
                        "base_prompt_artifact_sha256"
                    ),
                }
                or wire_prompt_artifact
                != {
                    "wire_prompt_artifact_id": reservation.get(
                        "wire_prompt_artifact_id"
                    ),
                    "wire_prompt_artifact_sha256": reservation.get(
                        "wire_prompt_artifact_sha256"
                    ),
                }
            ):
                raise PolicyRejected(
                    "generation_reservation_mismatch",
                    "candidate feedback differs from its reserved compiled inputs",
                )
            if any(
                row.get("generation_id") == generation_lineage["generation_id"]
                for row in self.state.data["reviewed_generations"]
            ):
                raise PolicyRejected(
                    "duplicate_generation_id", "generation_id has already been reviewed"
                )
        else:
            selected_generation = self.state.data.get("selected_generation")
            accepted_candidate_bundle_id = _require_identifier(
                value.get("accepted_candidate_bundle_id"),
                "accepted_candidate_bundle_id",
            )
            expected_selected = {
                key: selected_generation.get(key) if isinstance(selected_generation, Mapping) else None
                for key in generation_lineage
            }
            expected_base_prompt_artifact = {
                key: selected_generation.get(key)
                if isinstance(selected_generation, Mapping)
                else None
                for key in base_prompt_artifact
            }
            expected_wire_prompt_artifact = {
                key: selected_generation.get(key)
                if isinstance(selected_generation, Mapping)
                else None
                for key in wire_prompt_artifact
            }
            if (
                not isinstance(selected_generation, Mapping)
                or accepted_candidate_bundle_id
                != selected_generation.get("candidate_bundle_id")
                or generation_lineage != expected_selected
                or base_prompt_artifact != expected_base_prompt_artifact
                or wire_prompt_artifact != expected_wire_prompt_artifact
            ):
                raise PolicyRejected(
                    "final_candidate_mismatch",
                    "final_accepted must identify the exact condition-reviewed generation",
                )
        public_binding_outcomes: list[dict[str, Any]] = []
        if bundle_kind == "candidate_feedback":
            public_binding_outcomes = _normalize_binding_outcomes(
                plan, value.get("binding_outcomes")
            )
            reservation = self.state.data["generation_reservations"][
                generation_lineage["generation_reservation_id"]
            ]
            if public_binding_outcomes != reservation.get("binding_outcomes"):
                raise PolicyRejected(
                    "generation_reservation_mismatch",
                    "candidate binding outcomes differ from the reserved compilation",
                )
        if bundle_kind == "final_accepted":
            selected_generation = self.state.data.get("selected_generation") or {}
            expected_disposition = (
                "condition_pass"
                if selected_generation.get("decision") == "accept_conditions"
                else "budget_best_effort"
            )
            if value.get("selection_disposition") != expected_disposition:
                raise PolicyRejected(
                    "selection_disposition_mismatch",
                    "final selection disposition differs from the condition decision",
                )
            _require_identifier(
                value.get("finalization_receipt_id"), "finalization_receipt_id"
            )
        derived = value.get("derived_evidence") or []
        if not isinstance(derived, list):
            raise PolicyRejected("invalid_generation_bundle", "derived_evidence must be an array")
        records: dict[str, dict[str, Any]] = {}
        output_representation_found = False
        for row in derived:
            evidence_id = str(row.get("evidence_id") or row.get("evidence_ref") or "")
            representation = copy.deepcopy(row.get("representation") or {})
            if not representation and row.get("path"):
                representation = _file_manifest(
                    row["path"], str(row.get("representation_id") or self.state.next_id("repr"))
                )
            path = Path(str(representation.get("path") or ""))
            if not evidence_id or not path.is_file():
                raise PolicyRejected(
                    "invalid_generation_evidence", "derived evidence lacks immutable bytes"
                )
            actual = media_content_sha256(path)
            if representation.get("sha256") != actual:
                raise PolicyRejected(
                    "generation_evidence_hash_mismatch", "derived evidence hash does not match bytes"
                )
            if actual == generation_lineage["generation_output_sha256"]:
                output_representation_found = True
            else:
                parent_output_sha256 = _require_sha256(
                    row.get("parent_output_sha256"), "derived_evidence.parent_output_sha256"
                )
                if parent_output_sha256 != generation_lineage["generation_output_sha256"]:
                    raise PolicyRejected(
                        "derived_parent_mismatch",
                        "derived evidence is not bound to the reviewed generation output",
                    )
                if not row.get("derivation") or row.get("selector") is None:
                    raise PolicyRejected(
                        "incomplete_derivation",
                        "derived evidence requires a deterministic selector and derivation",
                    )
            representation_id = str(representation.get("representation_id") or "")
            records[evidence_id] = {
                "evidence_id": evidence_id,
                "source_kind": "generation_derived",
                "modality": row.get("modality") or representation.get("modality"),
                "description": row.get("description") or "Derived evidence from accepted generation.",
                "semantic_roles": copy.deepcopy(row.get("semantic_roles") or []),
                "subject_ids": copy.deepcopy(row.get("subject_ids") or []),
                "writeback_candidate": bool(row.get("writeback_candidate", False)),
                "default_representation_id": representation_id,
                "representations": {representation_id: representation},
                "parent_output_sha256": row.get("parent_output_sha256"),
                "selector": copy.deepcopy(row.get("selector")),
                "derivation": row.get("derivation"),
                "generation_id": generation_lineage["generation_id"],
                "condition_signature_sha256": generation_lineage[
                    "condition_signature_sha256"
                ],
            }
        if bundle_kind == "candidate_feedback" and not output_representation_found:
            raise PolicyRejected(
                "generation_output_missing",
                "candidate feedback must include the exact generated output representation",
            )
        review_target_id = None
        if bundle_kind == "candidate_feedback":
            review_target_id = _require_identifier(
                value.get("review_target_evidence_id"),
                "review_target_evidence_id",
            )
            if review_target_id not in records:
                raise PolicyRejected(
                    "invalid_review_target",
                    "condition review target is absent from generation evidence",
                )
        previous_bundle = self.state.data.get("generation_bundle_private")
        if isinstance(previous_bundle, Mapping):
            self.state.data["generation_bundle_history_private"].append(
                copy.deepcopy(dict(previous_bundle))
            )
        self.state.data["generation_bundle_private"] = value
        self.state.data["generation_evidence"] = records
        if bundle_kind == "candidate_feedback":
            reservation = self.state.data["generation_reservations"][
                generation_lineage["generation_reservation_id"]
            ]
            reservation["status"] = "completed"
            reservation["candidate_bundle_id"] = value.get("bundle_id")
            reservation["generation_output_sha256"] = generation_lineage[
                "generation_output_sha256"
            ]
            self.state.data["active_generation_reservation_id"] = None
            self.state.data["condition_review_target_id"] = review_target_id
        else:
            self.state.data["condition_review_target_id"] = None
        self.state.transition(
            expected_phase,
            "generation_review" if bundle_kind == "candidate_feedback" else "writeback",
        )
        observation = {
            "ok": True,
            "phase": self.phase,
            "bundle_id": value.get("bundle_id"),
            "bundle_kind": bundle_kind,
            "generation_id": generation_lineage["generation_id"],
            "generation_output_sha256": generation_lineage[
                "generation_output_sha256"
            ],
            "condition_signature_sha256": generation_lineage[
                "condition_signature_sha256"
            ],
            "review_target_evidence_id": review_target_id,
            "binding_outcomes": public_binding_outcomes,
            "available_evidence": [
                {
                    "evidence_id": evidence_id,
                    "modality": record["modality"],
                    "writeback_candidate": record["writeback_candidate"],
                }
                for evidence_id, record in records.items()
            ],
        }
        _assert_model_safe(observation)
        self.state.data["host_events"].append(
            {
                "event_id": self.state.next_id("hostevent"),
                "name": "attach_generation_evidence",
                "phase_before": phase_before,
                "phase_after": self.phase,
                "bundle_sha256": _sha256_json(value),
                "observation": copy.deepcopy(observation),
            }
        )
        try:
            self.state.save()
        except Exception:
            self.state.reload()
            raise
        return observation

    def tool_update_memory_bank(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.state.data.get("staged_transaction") is not None:
            raise PolicyRejected("transaction_already_staged", "memory transaction is immutable")
        additions = copy.deepcopy(arguments["records"])
        registry_updates = copy.deepcopy(arguments["registry_updates"])
        if not additions and not registry_updates:
            raise PolicyRejected("empty_transaction", "memory update has no changes")
        staged_memories: list[dict[str, Any]] = []
        for item in additions:
            source_kind = str(item["source_kind"])
            evidence_id = str(item["source_evidence_id"])
            source_table = {
                "generation_derived": "generation_evidence",
                "generated_evidence": "created_evidence",
                "uploaded_material": "uploaded_evidence",
            }.get(source_kind)
            if source_table is None:
                raise PolicyRejected(
                    "invalid_writeback_source",
                    "writeback currently supports generation-derived, selected "
                    "policy-created, or trusted uploaded evidence",
                )
            evidence = self.state.data[source_table].get(evidence_id)
            if not isinstance(evidence, Mapping):
                raise PolicyRejected(
                    "invalid_writeback_source", "writeback evidence does not exist"
                )
            representation_id = str(item["representation_id"])
            authorization_id = str(item["authorization_id"])
            representation, _ = self._resolve_view_binding(
                {
                    "source_kind": source_kind,
                    "evidence_id": evidence_id,
                    "representation_id": representation_id,
                    "authorization_id": authorization_id,
                }
            )
            source_selector = copy.deepcopy(item.get("source_selector"))
            if evidence.get("modality") == "audio":
                source_selector = self._exact_audio_selector(source_selector)
                if representation.get("selector") != source_selector:
                    raise PolicyRejected(
                        "audio_selector_mismatch",
                        "audio writeback selector must exactly match the viewed representation",
                    )
                view_key = "|".join(
                    [source_kind, evidence_id, representation_id, authorization_id]
                )
                view = self.state.data["views"].get(view_key) or {}
                if view.get("selector") != source_selector:
                    raise PolicyRejected(
                        "audio_selector_mismatch",
                        "audio writeback must retain the exact authorized view time range",
                    )
            elif source_selector is not None:
                raise PolicyRejected(
                    "unexpected_source_selector",
                    "source_selector is reserved for exact audio time-range writeback",
                )
            inspection: Mapping[str, Any] = {}
            assessment: Mapping[str, Any] = {}
            speaker_binding: dict[str, str] | None = None
            if evidence.get("modality") == "audio":
                speaker_binding = self._trusted_audio_speaker_binding(evidence)
            if source_kind == "generation_derived":
                if not evidence.get("writeback_candidate"):
                    raise PolicyRejected(
                        "invalid_writeback_source",
                        "generation-derived evidence is not eligible for writeback",
                    )
                inspection = self.state.data["inspections"].get(
                    evidence.get("inspection_id")
                ) or {}
                assessment = self.state.data["assessments"].get(
                    evidence.get("assessment_id")
                ) or {}
                memory_use_ids = {
                    row["use_id"]
                    for row in inspection.get("requested_uses") or []
                    if row.get("use") == "memory_writeback"
                }
                passed_use_ids = {
                    row["use_id"]
                    for row in assessment.get("use_results") or []
                    if row.get("result") == "pass"
                }
                if not memory_use_ids.intersection(passed_use_ids):
                    raise PolicyRejected(
                        "writeback_assessment_required",
                        "generation-derived writeback requires a passed "
                        "post-generation memory-use assessment",
                    )
                if not set(item["writeback_evidence_ids"]).issubset(
                    set(inspection.get("evidence_ids") or [])
                ):
                    raise PolicyRejected(
                        "invalid_writeback_evidence",
                        "writeback evidence is absent from the candidate inspection",
                        details={
                            "record_source_evidence_id": evidence_id,
                            "expected_writeback_evidence_ids": copy.deepcopy(
                                inspection.get("evidence_ids") or []
                            ),
                        },
                    )
            elif source_kind == "generated_evidence":
                group = self.state.data["creation_groups"].get(
                    evidence.get("creation_group_id")
                ) or {}
                assessment = self.state.data["assessments"].get(
                    evidence.get("assessment_id")
                ) or {}
                inspection = self.state.data["inspections"].get(
                    evidence.get("inspection_id")
                ) or {}
                if (
                    group.get("selected_candidate_id") != evidence_id
                    or assessment.get("assessment_scope")
                    != "created_evidence_suitability"
                    or assessment.get("quality") not in {"pass", "capability_limited"}
                    or assessment.get("next_action") != "select"
                ):
                    raise PolicyRejected(
                        "writeback_assessment_required",
                        "policy-created evidence must be the selected assessed candidate",
                    )
                if not set(item["writeback_evidence_ids"]).issubset(
                    set(inspection.get("evidence_ids") or [])
                ):
                    raise PolicyRejected(
                        "invalid_writeback_evidence",
                        "writeback evidence is absent from the creation inspection",
                        details={
                            "record_source_evidence_id": evidence_id,
                            "expected_writeback_evidence_ids": copy.deepcopy(
                                inspection.get("evidence_ids") or []
                            ),
                        },
                    )
            else:
                # Uploaded bytes can be persisted without pretending that the
                # policy perceived facts which came only from a trusted user
                # declaration.  Free-form assistant-authored facts are not an
                # admissible substitute for registry metadata.
                trusted_description = str(evidence.get("description") or "")
                trusted_facts = {
                    str(value)
                    for value in evidence.get("trusted_stable_facts") or []
                    if isinstance(value, str) and value
                }
                declared_tags = {
                    str(value)
                    for value in evidence.get("declared_tags") or []
                    if isinstance(value, str) and value
                }
                trusted_roles = {
                    str(value)
                    for value in evidence.get("semantic_roles") or []
                    if isinstance(value, str) and value
                }
                if str(item["description"]) != trusted_description:
                    raise PolicyRejected(
                        "untrusted_uploaded_description",
                        "uploaded-memory description must copy trusted registry metadata exactly",
                        details={"expected_description": trusted_description},
                    )
                if not set(item["stable_facts"]).issubset(trusted_facts):
                    raise PolicyRejected(
                        "untrusted_uploaded_fact",
                        "uploaded-memory stable facts must come from trusted registry metadata",
                        details={"allowed_stable_facts": sorted(trusted_facts)},
                    )
                if not set(item["tags"]).issubset(declared_tags) or not set(
                    item["semantic_roles"]
                ).issubset(trusted_roles):
                    raise PolicyRejected(
                        "untrusted_uploaded_metadata",
                        "uploaded-memory tags and roles must copy trusted registry metadata",
                        details={
                            "allowed_tags": sorted(declared_tags),
                            "allowed_semantic_roles": sorted(trusted_roles),
                        },
                    )
                if set(item["writeback_evidence_ids"]) != {evidence_id}:
                    raise PolicyRejected(
                        "unsupported_uploaded_perception_claim",
                        "trusted uploaded writeback must cite only the viewed upload itself",
                        details={
                            "expected_writeback_evidence_ids": [evidence_id]
                        },
                    )
                if evidence.get("modality") == "audio":
                    inspection = self.state.data["inspections"].get(
                        evidence.get("inspection_id")
                    ) or {}
                    assessment = self.state.data["assessments"].get(
                        evidence.get("assessment_id")
                    ) or {}
                    check_results = {
                        str(row.get("result") or "")
                        for row in assessment.get("check_results") or []
                    }
                    memory_use_results = {
                        str(row.get("result") or "")
                        for row in assessment.get("use_results") or []
                        if row.get("use_id")
                        in {
                            request.get("use_id")
                            for request in inspection.get("requested_uses") or []
                            if request.get("use") == "memory_writeback"
                        }
                    }
                    if (
                        inspection.get("candidate_id") != evidence_id
                        or inspection.get("representation_id") != representation_id
                        or inspection.get("authorization_id") != authorization_id
                        or inspection.get("selector") != source_selector
                        or inspection.get("source_kind") != "uploaded_material"
                        or assessment.get("assessment_scope") != "memory_writeback"
                        or assessment.get("next_action") != "select"
                        or assessment.get("quality") != "pass"
                        or check_results != {"pass"}
                        or memory_use_results != {"pass"}
                    ):
                        raise PolicyRejected(
                            "audio_writeback_assessment_required",
                            "uploaded audio requires an independent passing metadata-level reuse assessment",
                        )
                if evidence.get("modality") == "audio" and (
                    "voice" in set(item["semantic_roles"])
                    or "voice" in trusted_roles
                ):
                    # The exact named-speaker lineage was normalized above and
                    # is copied into the durable memory record below.
                    assert speaker_binding is not None
            if item["modality"] != evidence.get("modality"):
                raise PolicyRejected(
                    "writeback_modality_mismatch",
                    "memory record modality differs from its source evidence",
                )
            memory_id = str(item.get("record_id") or self.state.next_id("memory"))
            if any(row["memory_id"] == memory_id for row in staged_memories):
                raise PolicyRejected(
                    "duplicate_memory_id", "memory transaction contains a duplicate record ID"
                )
            staged_memory = {
                "memory_id": memory_id,
                "modality": evidence["modality"],
                "description": str(item["description"]),
                "tags": copy.deepcopy(item["tags"]),
                "semantic_roles": copy.deepcopy(item["semantic_roles"]),
                "timeline_role": item["timeline_role"],
                "stable_facts": copy.deepcopy(item["stable_facts"]),
                "reuse_scope": item["reuse_scope"],
                "source_selector": copy.deepcopy(source_selector),
                "subject_ids": copy.deepcopy(evidence.get("subject_ids") or []),
                "default_representation_id": representation_id,
                "representations": {
                    representation_id: copy.deepcopy(
                        evidence["representations"][representation_id]
                    )
                },
                "provenance": {
                    "source_kind": source_kind,
                    "bundle_id": (
                        (self.state.data.get("generation_bundle_private") or {}).get(
                            "bundle_id"
                        )
                        if source_kind == "generation_derived"
                        else None
                    ),
                    "evidence_id": evidence_id,
                    "authorization_id": authorization_id,
                },
            }
            if speaker_binding is not None:
                staged_memory["speaker_binding"] = copy.deepcopy(speaker_binding)
            staged_memories.append(staged_memory)
        registry_items = self.state.data["registry"]["items"]
        staged_registry_updates: list[dict[str, Any]] = []
        for update in registry_updates:
            subject_id = str(update["subject_id"])
            if subject_id not in registry_items:
                raise PolicyRejected(
                    "unknown_registry_subject",
                    "registry update must name an existing opaque subject_id",
                )
            support_records: list[tuple[str, Mapping[str, Any]]] = []
            for support_id in update["supporting_evidence_ids"]:
                matched: dict[tuple[str, str], Mapping[str, Any]] = {}
                for view in self.state.data["views"].values():
                    if view.get("evidence_id") != support_id:
                        continue
                    source_kind = str(view.get("source_kind") or "")
                    if source_kind not in {
                        "uploaded_material",
                        "generated_evidence",
                        "generation_derived",
                    }:
                        continue
                    evidence = self._evidence_record(source_kind, str(support_id))
                    matched[(source_kind, str(support_id))] = evidence
                for inspection in self.state.data["inspections"].values():
                    if support_id not in set(inspection.get("evidence_ids") or []):
                        continue
                    candidate_id = str(inspection.get("candidate_id") or "")
                    if candidate_id in self.state.data["created_evidence"]:
                        source_kind = "generated_evidence"
                        evidence = self.state.data["created_evidence"][candidate_id]
                    elif candidate_id in self.state.data["generation_evidence"]:
                        source_kind = "generation_derived"
                        evidence = self.state.data["generation_evidence"][candidate_id]
                    else:
                        continue
                    matched[(source_kind, candidate_id)] = evidence
                if len(matched) != 1:
                    raise PolicyRejected(
                        "invalid_registry_support",
                        "each registry supporting evidence ID must resolve uniquely to "
                        "a current-session view or inspection",
                    )
                (source_kind, _), evidence = next(iter(matched.items()))
                if subject_id not in set(evidence.get("subject_ids") or []):
                    raise PolicyRejected(
                        "registry_support_subject_mismatch",
                        "registry supporting evidence is not bound to the updated subject",
                    )
                if source_kind == "generated_evidence":
                    group = self.state.data["creation_groups"].get(
                        evidence.get("creation_group_id")
                    ) or {}
                    assessment = self.state.data["assessments"].get(
                        evidence.get("assessment_id")
                    ) or {}
                    if (
                        group.get("selected_candidate_id")
                        != evidence.get("evidence_id")
                        or assessment.get("assessment_scope")
                        != "created_evidence_suitability"
                        or assessment.get("quality")
                        not in {"pass", "capability_limited"}
                        or assessment.get("next_action") != "select"
                    ):
                        raise PolicyRejected(
                            "registry_support_not_writeback_eligible",
                            "policy-created registry support must be the selected assessed candidate",
                        )
                elif source_kind == "generation_derived":
                    inspection = self.state.data["inspections"].get(
                        evidence.get("inspection_id")
                    ) or {}
                    assessment = self.state.data["assessments"].get(
                        evidence.get("assessment_id")
                    ) or {}
                    memory_use_ids = {
                        row["use_id"]
                        for row in inspection.get("requested_uses") or []
                        if row.get("use") == "memory_writeback"
                    }
                    passed_use_ids = {
                        row["use_id"]
                        for row in assessment.get("use_results") or []
                        if row.get("result") == "pass"
                    }
                    if not evidence.get("writeback_candidate") or not (
                        memory_use_ids & passed_use_ids
                    ):
                        raise PolicyRejected(
                            "registry_support_not_writeback_eligible",
                            "generation-derived registry support requires a passed "
                            "memory-writeback assessment",
                        )
                support_records.append((source_kind, evidence))
            normalized = copy.deepcopy(update)
            normalized["subject_id"] = subject_id
            staged_registry_updates.append(normalized)
        missing_voice_memories = self._missing_required_uploaded_voice_memories(
            self._selected_generation_plan(),
            staged_memories=staged_memories,
        )
        if missing_voice_memories:
            raise PolicyRejected(
                "required_voice_memory_writeback",
                "this transaction must include every accepted reusable uploaded "
                "voice that lacks equivalent committed audio memory",
                details={
                    "missing_voice_memories": missing_voice_memories,
                    "equivalence": "audio_content_sha256_exact_selector_and_speaker_binding",
                    "recoverable": True,
                },
            )
        transaction_id = self.state.next_id("memtxn")
        self.state.data["staged_transaction"] = {
            "transaction_id": transaction_id,
            "memories": staged_memories,
            "registry_updates": staged_registry_updates,
            "rationale": arguments["rationale"],
        }
        return {
            "ok": True,
            "transaction_id": transaction_id,
            "staged_memory_ids": [row["memory_id"] for row in staged_memories],
            "registry_update_count": len(registry_updates),
        }

    def tool_finish_memory_session(self, arguments: dict[str, Any]) -> dict[str, Any]:
        plan = self._selected_generation_plan()
        selected_generation = self.state.data.get("selected_generation") or {}
        final_plan_id = selected_generation.get("plan_id")
        if arguments["accepted_plan_id"] != final_plan_id:
            raise PolicyRejected("plan_mismatch", "finish changed the accepted evidence plan")
        missing_voice_memories = self._missing_required_uploaded_voice_memories(plan)
        if missing_voice_memories:
            raise PolicyRejected(
                "required_voice_memory_writeback",
                "an accepted reusable uploaded voice must be persisted with "
                "update_memory_bank before finish_memory_session",
                details={
                    "missing_voice_memories": missing_voice_memories,
                    "equivalence": "audio_content_sha256_exact_selector_and_speaker_binding",
                    "recoverable": True,
                },
            )
        decision = str(arguments["writeback_disposition"])
        staged = self.state.data.get("staged_transaction")
        if decision == "commit_staged":
            transaction_id = str(arguments.get("memory_transaction_id") or "")
            if not staged or transaction_id != staged.get("transaction_id"):
                raise PolicyRejected("transaction_mismatch", "finish must name the staged transaction")
            before = copy.deepcopy(self.state.data)
            try:
                for memory in staged["memories"]:
                    self.state.data["memory_bank"][memory["memory_id"]] = memory
                for update in staged["registry_updates"]:
                    subject_id = str(update["subject_id"])
                    if subject_id not in self.state.data["registry"]["items"]:
                        raise PolicyRejected(
                            "invalid_registry_update",
                            "staged registry subject no longer exists",
                        )
                    current = copy.deepcopy(
                        self.state.data["registry"]["items"].get(subject_id) or {}
                    )
                    current.update(copy.deepcopy(update))
                    current["subject_id"] = subject_id
                    self.state.data["registry"]["items"][subject_id] = current
                self.state.data["registry"]["version"] += 1
                self.state.data["staged_transaction"] = None
                self.state.transition("writeback", "complete")
                self.state.save()
            except Exception:
                self.state.reload()
                raise
            return {
                "ok": True,
                "done": True,
                "writeback_disposition": "commit_staged",
                "memory_transaction_id": transaction_id,
            }
        if arguments.get("memory_transaction_id") is not None:
            raise PolicyRejected(
                "unexpected_transaction", "not_needed requires memory_transaction_id=null"
            )
        if staged:
            raise PolicyRejected(
                "staged_transaction_cannot_be_skipped",
                "a staged transaction must be committed",
            )
        self.state.transition("writeback", "complete")
        return {
            "ok": True,
            "done": True,
            "writeback_disposition": "not_needed",
        }


__all__ = [
    "MemoryPolicyRuntime",
    "PolicyRejected",
    "RUNTIME_VERSION",
    "TOOL_NAMES",
]
