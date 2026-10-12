from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .contract import (
    PROMPT_REWRITE_MODES,
    SEMANTIC_MODULES,
    build_assessment_contract,
    build_evaluation_contract_authoring,
    evaluation_check_evidence_kinds,
)
from .pe import PESkillResolver
from .state import StateStore
from .util import atomic_write_json, canonical_json, sha256_json, sha256_text, utc_now
from .validation import PromptValidationError, PromptValidator


RUNTIME_VERSION = "storymem.canonical.runtime.v2"
_IMAGE_AXES = {"prompt", "references", "seed"}
_VIDEO_AXES = {"prompt", "condition", "memory_slots", "seed"}
_PERSON_MEMORY_ROLES = {"character_identity", "wardrobe", "voice"}
_ENTITY_MEMORY_ROLES = {"entity_identity", "entity_appearance"}
_ENTITY_ASSET_ROLE_RE = re.compile(
    r"^(?:entity_identity|entity_appearance):(ENTITY_[1-9][0-9]*)$"
)
_SEMANTIC_MODULES = frozenset(SEMANTIC_MODULES)
_ASSESSMENT_STATUSES = {
    "pass",
    "fail",
    "uncertain",
    "not_applicable",
    "capability_blocked",
}
_VIDEO_MATERIAL_CATEGORY_MODULE = {
    "person": "identity_wardrobe",
    "entity": "identity_wardrobe",
    "scene": "scene_environment",
    "object": "object_state",
    "style": "style_lighting",
}


class ToolRejected(ValueError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def observation(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": self.code,
            "message": str(self),
            "details": copy.deepcopy(self.details),
        }


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: Path
    width: int = 1280
    height: int = 736
    num_frames: int = 241
    image_attempts: int = 2
    video_attempts: int = 2
    max_memory_slots: int = 7
    audio_transport: str = "disabled"
    audio_perception: str = "disabled"
    prompt_rewrite_mode: str = "rewrite_enabled"

    def __post_init__(self) -> None:
        if self.image_attempts < 1 or self.video_attempts < 1:
            raise ValueError("generation attempt budgets must be positive")
        if not 0 <= self.max_memory_slots <= 7:
            raise ValueError("max_memory_slots must be in [0, 7]")
        if self.audio_perception not in {"disabled", "verified"}:
            raise ValueError("audio_perception must be disabled or verified")
        if self.audio_transport not in {"disabled", "enabled"}:
            raise ValueError("audio_transport must be disabled or enabled")
        if self.prompt_rewrite_mode not in PROMPT_REWRITE_MODES:
            raise ValueError(
                "prompt_rewrite_mode must be one of: "
                + ", ".join(PROMPT_REWRITE_MODES)
            )


def _plain(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if is_dataclass(value):
        return asdict(value)
    method = getattr(value, "to_dict", None)
    if callable(method):
        result = method()
        if isinstance(result, dict):
            return copy.deepcopy(result)
    if hasattr(value, "__dict__"):
        return copy.deepcopy(vars(value))
    raise TypeError(f"provider returned unsupported result type: {type(value)!r}")


def _tool_schema_map() -> dict[str, dict[str, Any]]:
    from .contract import get_tool_schemas

    result: dict[str, dict[str, Any]] = {}
    for entry in get_tool_schemas():
        function = entry.get("function", entry)
        name = str(function.get("name") or "")
        if name:
            result[name] = function
    return result


def _tokenize_search(text: str) -> list[str]:
    lowered = text.casefold()
    latin = re.findall(r"[a-z0-9_<>-]+", lowered)
    han = re.findall(r"[\u3400-\u9fff]", lowered)
    return latin + han


def _bm25_like_score(query: str, document: str) -> float:
    terms = _tokenize_search(query)
    if not terms:
        return 0.0
    tokens = _tokenize_search(document)
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    score = 0.0
    length_norm = 1.0 + 0.25 * math.log1p(len(tokens))
    for term in terms:
        frequency = counts.get(term, 0)
        if frequency:
            score += (1.0 + math.log1p(frequency)) / length_norm
    return score


class StoryMemRuntime:
    """Deterministic executor for the eleven public StoryMem tools.

    The runtime validates and persists decisions.  It never invokes a semantic
    evaluator, prompt writer, captioner, or hidden policy.  All semantic text,
    media choices, and assessments must arrive in the teacher's tool call.
    """

    def __init__(
        self,
        *,
        state: StateStore,
        config: RuntimeConfig,
        pe_resolver: PESkillResolver,
        prompt_validator: PromptValidator,
        image_provider: Any,
        video_provider: Any,
    ) -> None:
        self.state = state
        self.config = config
        self.pe_resolver = pe_resolver
        self.prompt_validator = prompt_validator
        self.image_provider = image_provider
        self.video_provider = video_provider
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._schemas = _tool_schema_map()
        self._active_tool_call: dict[str, Any] | None = None
        self._active_assistant_provenance: dict[str, Any] | None = None

    @property
    def shot(self) -> dict[str, Any]:
        return self.state.shot

    def teacher_allowed_tool_names(self) -> tuple[str, ...]:
        """Return the state-admitted action catalog for the next Teacher turn.

        Argument-level validation remains in the individual handlers.  This
        method mirrors the global causal gates so the closed Teacher is not
        shown eleven large schemas when only a small subset can advance the
        current checkpoint.
        """

        from .contract import CANONICAL_TOOL_NAMES

        shot = self.shot
        inflight = shot.get("inflight")
        if isinstance(inflight, Mapping):
            call = inflight.get("tool_call")
            name = str(call.get("name") or "") if isinstance(call, Mapping) else ""
            return (name,) if name in CANONICAL_TOOL_NAMES else ()
        if shot.get("phase") == "committed":
            return ()
        if (
            shot.get("phase") == "need_registry"
            or not shot.get("registry_pagination_complete", False)
        ):
            return ("view_material_registry",)

        receipt_id = shot.get("pending_pe_receipt_id")
        if receipt_id:
            return ("register_prompt",)
        artifact_id = shot.get("pending_prompt_artifact_id")
        if artifact_id:
            artifact = shot.get("prompt_artifacts", {}).get(artifact_id) or {}
            target = artifact.get("target")
            return (
                ("generate_image",)
                if target == "image"
                else ("generate_shot",)
                if target == "video"
                else ()
            )
        candidate_id = shot.get("pending_candidate_id")
        if candidate_id:
            candidate = shot.get("candidates", {}).get(candidate_id) or {}
            if not candidate.get("inspection_id"):
                return ("inspect_candidate",)
            if not candidate.get("assessment_id"):
                return ("submit_candidate_assessment",)
            return ()
        if shot.get("video_selection_id"):
            names = {"submit_shot"}
            if not shot.get("staged_memory_update"):
                names.add("update_memory_bank")
            return tuple(name for name in CANONICAL_TOOL_NAMES if name in names)

        names = {"search_memory_bank"}
        latest_search_id = shot.get("latest_search_id")
        latest_search = shot.get("searches", {}).get(latest_search_id) or {}
        committed = self.state.data.get("committed_shots") or []
        has_previous_tail = bool(
            committed
            and (committed[-1].get("commit") or {}).get("tail_handle")
        )
        if (
            bool(shot.get("upload_ids"))
            or bool(latest_search.get("candidate_ids"))
            or has_previous_tail
        ):
            names.add("view_memories")

        obligations = self._retry_obligations()
        if not obligations:
            if self.config.prompt_rewrite_mode == "rewrite_enabled":
                names.add("get_pe_skill")
            else:
                names.add("register_prompt")
        for group_id, obligation in obligations:
            group = shot.get("generation_groups", {}).get(group_id) or {}
            target = str(group.get("target") or "")
            axes = set(str(axis) for axis in obligation.get("axes") or [])
            if target == "video":
                if axes.intersection({"prompt", "condition"}):
                    if self.config.prompt_rewrite_mode == "rewrite_enabled":
                        names.add("get_pe_skill")
                    else:
                        names.add("generate_shot")
                else:
                    names.add("generate_shot")
            elif target == "image":
                candidate = shot.get("candidates", {}).get(
                    str(obligation.get("candidate_id") or "")
                ) or {}
                prior_has_references = bool(candidate.get("input_references") or [])
                if (
                    self.config.prompt_rewrite_mode == "rewrite_enabled"
                    and axes.intersection({"prompt", "references"})
                ):
                    names.add("get_pe_skill")
                if self.config.prompt_rewrite_mode == "passthrough_disabled" or (
                    "prompt" not in axes
                    and ("references" not in axes or prior_has_references)
                ):
                    names.add("generate_image")

        return tuple(name for name in CANONICAL_TOOL_NAMES if name in names)

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        assistant_provenance: dict[str, Any] | None = None,
        commit_rejection: bool = False,
    ) -> dict[str, Any]:
        call = {"name": str(name), "arguments": copy.deepcopy(dict(arguments))}
        event_provenance = copy.deepcopy(assistant_provenance or {})
        before_call: dict[str, Any] | None = None
        persisted_inflight = self.shot.get("inflight")
        if isinstance(persisted_inflight, Mapping) and call == persisted_inflight.get("tool_call"):
            saved_provenance = persisted_inflight.get("assistant_provenance")
            if not isinstance(saved_provenance, Mapping):
                raise RuntimeError("provider inflight record has invalid assistant provenance")
            event_provenance = copy.deepcopy(dict(saved_provenance))
        try:
            self._validate_schema(name, call["arguments"])
            self._validate_global_fsm(name, call["arguments"])
            handler = getattr(self, f"tool_{name}", None)
            if not callable(handler):
                raise ToolRejected("unknown_tool", f"unknown tool: {name}")
            before_call = copy.deepcopy(self.state.data)
            self._active_tool_call = copy.deepcopy(call)
            self._active_assistant_provenance = copy.deepcopy(event_provenance)
            try:
                result = handler(call["arguments"])
            finally:
                self._active_tool_call = None
                self._active_assistant_provenance = None
            if not isinstance(result, dict):
                raise RuntimeError(f"tool {name} returned a non-object")
            result.setdefault("ok", True)
        except ToolRejected as exc:
            if before_call is not None:
                self.state.data = before_call
            result = exc.observation()
            if commit_rejection:
                self.state.record_event(
                    tool_call=call,
                    observation=result,
                    assistant_provenance=event_provenance,
                    committed=False,
                )
            return result

        commit_payload = result.pop("_commit_payload", None)
        memory_transaction = result.pop("_memory_transaction", None)
        if commit_payload is None:
            event = self.state.record_event(
                tool_call=call,
                observation=result,
                assistant_provenance=event_provenance,
                committed=True,
            )
            result["event_id"] = event["event_id"]
            return result

        # submit_shot spans the trajectory shard and the durable state file.
        # Keep the persisted state unchanged until all validation and shard
        # serialization succeed, then atomically replace state.json once with
        # the selected shot and optional staged memory transaction together.
        before = copy.deepcopy(self.state.data)
        shard_path = self._committed_shot_path()
        previous_shard = shard_path.read_bytes() if shard_path.is_file() else None
        try:
            event = self.state.record_event(
                tool_call=call,
                observation=result,
                assistant_provenance=event_provenance,
                committed=True,
                save=False,
            )
            result["event_id"] = event["event_id"]
            self._write_committed_shot()
            if memory_transaction is not None:
                self._commit_memory_update(memory_transaction)
            self.state.commit_shot(commit_payload)
        except Exception:
            self.state.data = before
            # A later state commit may fail after the trajectory shard was
            # written. Restore the pre-call shard state so an uncommitted shot
            # can never masquerade as a completed training example.
            try:
                if previous_shard is None:
                    shard_path.unlink(missing_ok=True)
                else:
                    from .media import atomic_write_bytes

                    atomic_write_bytes(shard_path, previous_shard)
            except Exception as rollback_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"submit_shot failed and shard rollback also failed: {rollback_exc}"
                ) from rollback_exc
            raise
        return result

    def _validate_schema(self, name: str, arguments: dict[str, Any]) -> None:
        schema = self._schemas.get(name)
        if schema is None:
            raise ToolRejected("unknown_tool", f"unknown tool: {name}")
        parameters = schema.get("parameters") or {}
        errors = sorted(Draft202012Validator(parameters).iter_errors(arguments), key=lambda e: list(e.path))
        if errors:
            first = errors[0]
            path = ".".join(str(item) for item in first.path)
            raise ToolRejected(
                "invalid_arguments",
                f"{name} arguments failed schema validation at {path or '<root>'}: {first.message}",
            )

    def _validate_global_fsm(self, name: str, arguments: dict[str, Any]) -> None:
        shot = self.shot
        inflight = shot.get("inflight")
        if inflight:
            attempted = {"name": name, "arguments": copy.deepcopy(arguments)}
            if attempted != inflight.get("tool_call"):
                raise ToolRejected(
                    "provider_call_inflight",
                    "resume the exact persisted provider tool call before taking another action",
                    details={"request_id": inflight.get("request_id")},
                )
        if shot["phase"] == "need_registry" and name != "view_material_registry":
            raise ToolRejected("registry_first", "the first action of every shot must be view_material_registry")
        if shot["phase"] != "need_registry" and not shot.get("registry_pagination_complete", False) and name != "view_material_registry":
            raise ToolRejected("registry_pagination_incomplete", "finish the current registry snapshot before using another tool")
        if shot["phase"] == "committed":
            raise ToolRejected("shot_already_committed", "the current shot is already committed")
        pending_receipt = shot.get("pending_pe_receipt_id")
        if pending_receipt and name != "register_prompt":
            raise ToolRejected(
                "pe_receipt_must_be_registered",
                f"the loaded PE receipt {pending_receipt} must be used by register_prompt next",
            )
        pending = shot.get("pending_prompt_artifact_id")
        if pending:
            artifact = shot["prompt_artifacts"][pending]
            required = "generate_image" if artifact["target"] == "image" else "generate_shot"
            if name != required:
                raise ToolRejected(
                    "registered_prompt_must_be_consumed",
                    f"the accepted prompt artifact {pending} must be consumed by {required} next",
                )
        pending_candidate_id = shot.get("pending_candidate_id")
        if pending_candidate_id:
            candidate = shot["candidates"][pending_candidate_id]
            if not candidate.get("inspection_id") and name != "inspect_candidate":
                raise ToolRejected(
                    "candidate_must_be_inspected",
                    f"candidate {pending_candidate_id} must be inspected next",
                )
            if candidate.get("inspection_id") and not candidate.get("assessment_id") and name != "submit_candidate_assessment":
                raise ToolRejected(
                    "candidate_must_be_assessed",
                    f"candidate {pending_candidate_id} must be assessed next",
                )
        if shot.get("video_selection_id") and name not in {"update_memory_bank", "submit_shot"}:
            raise ToolRejected(
                "video_already_selected",
                "after video selection only update_memory_bank or submit_shot is allowed",
            )

    def _assert_request(self, arguments: Mapping[str, Any]) -> None:
        if arguments.get("shot_request_id") != self.shot["shot_request_id"]:
            raise ToolRejected("wrong_shot_request", "shot_request_id does not match the active shot")

    def _new_observation(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        observation_id = self.state.add_observation(kind, payload)
        payload["observation_id"] = observation_id
        return payload

    def _representation_for_asset(
        self,
        asset: Mapping[str, Any],
        representation_id: str | None = None,
    ) -> dict[str, Any]:
        representation_id = str(
            representation_id or asset.get("default_representation_id") or ""
        )
        representations = asset.get("representations") or {}
        representation = representations.get(representation_id) if isinstance(representations, dict) else None
        if not isinstance(representation, dict):
            direct = asset.get("representation")
            representation = direct if isinstance(direct, dict) else None
        if not isinstance(representation, dict):
            raise ToolRejected("missing_representation", "asset has no usable representation")
        return representation

    def _story_prompt_state(self) -> dict[str, Any]:
        """Build the current text-only PE state from committed canonical data."""

        plan = self.state.data.get("story_plan") or {}
        base = copy.deepcopy(plan.get("prompt_state") or {})
        shot_index = int(self.shot["shot_index"])
        shot_plan = next(
            (
                copy.deepcopy(row)
                for row in plan.get("shots") or []
                if int(row.get("shot_index", -1)) == shot_index
            ),
            {"shot_index": shot_index, "raw_request": self.shot["raw_user_request"]},
        )
        registry_rows = sorted(
            self.state.data.get("registry", {}).get("items", {}).values(),
            key=lambda row: str(row.get("tag") or ""),
        )
        stable_facts = [
            {
                "tag": row.get("tag"),
                "category": row.get("category"),
                "stable_description": row.get("stable_description")
                or row.get("visual_description")
                or row.get("description")
                or "",
                "voice_description": row.get("voice_description") or "",
                "aliases": list(row.get("aliases") or []),
                "version": row.get("version"),
            }
            for row in registry_rows
        ]
        committed = self.state.data.get("committed_shots") or []
        prior_text = "\n".join(
            str(row.get("raw_user_request") or "")
            + "\n"
            + str(
                (
                    row.get("prompt_artifacts", {}).get(
                        row.get("candidates", {})
                        .get(row.get("commit", {}).get("selected_candidate_id"), {})
                        .get("prompt_artifact_id"),
                        {},
                    )
                    or {}
                ).get("text", "")
            )
            for row in committed
        )
        first_seen: dict[str, dict[str, bool]] = {}
        current_text = self.shot["raw_user_request"]
        for row in registry_rows:
            tag = str(row.get("tag") or "")
            if not tag:
                continue
            aliases = [tag, *(str(value) for value in row.get("aliases") or [])]
            first_seen[tag] = {
                "seen_before": any(value and value in prior_text for value in aliases),
                "mentioned_in_current_request": any(
                    value and value in current_text for value in aliases
                ),
            }

        previous_prompt = None
        previous_continuity = None
        if committed:
            previous = committed[-1]
            selected_candidate_id = previous.get("commit", {}).get("selected_candidate_id")
            selected_candidate = previous.get("candidates", {}).get(selected_candidate_id) or {}
            artifact_id = selected_candidate.get("prompt_artifact_id")
            artifact = previous.get("prompt_artifacts", {}).get(artifact_id) or {}
            if artifact:
                previous_prompt = {
                    "prompt_artifact_id": artifact_id,
                    "text": artifact.get("text"),
                    "text_sha256": artifact.get("text_sha256"),
                    "operation": artifact.get("operation"),
                    "parent_prompt_artifact_id": artifact.get("parent_prompt_artifact_id"),
                    "receipt_chain": copy.deepcopy(artifact.get("receipt_chain") or []),
                    "revision_count": artifact.get("revision_count"),
                }
            previous_continuity = {
                "shot_index": previous.get("shot_index"),
                "summary": previous.get("commit", {}).get("summary"),
                "tail_handle": copy.deepcopy(previous.get("commit", {}).get("tail_handle")),
            }
        base.update(
            {
                "caption_language": plan.get("caption_language")
                or base.get("caption_language"),
                "dialogue_language": plan.get("dialogue_language")
                or base.get("dialogue_language"),
                "outer_shot_index": shot_index,
                "current_shot_plan": shot_plan,
                "allocated_character_ids": [
                    str(row.get("tag"))
                    for row in registry_rows
                    if row.get("category") == "person" and row.get("tag")
                ],
                "first_seen": first_seen,
                "stable_facts": stable_facts,
                "scene_continuity": previous_continuity,
                "previous_accepted_prompt": previous_prompt,
            }
        )
        return base

    def _resolve_source(self, source: Mapping[str, Any], *, require_view: bool = True) -> dict[str, Any]:
        kind = str(source.get("kind") or "")
        source_id = str(source.get("source_id") or "")
        representation_id = str(source.get("representation_id") or "")
        authorization_id = str(source.get("authorization_id") or "")
        record: dict[str, Any] | None = None
        if kind == "memory":
            record = self.state.data["memory_bank"].get(source_id)
        elif kind == "uploaded_asset":
            record = self.state.data["assets"].get(source_id)
        elif kind == "previous_tail":
            committed = self.state.data.get("committed_shots") or []
            tail = committed[-1].get("commit", {}).get("tail_representation") if committed else None
            if isinstance(tail, dict) and source_id == str(tail.get("source_id") or tail.get("representation_id") or ""):
                record = {"representation": tail}
        elif kind == "generated_image":
            candidate = self.shot["candidates"].get(source_id)
            if candidate and candidate.get("target") == "image":
                record = {
                    "representation": candidate.get("representation"),
                    "subject_ids": candidate.get("subject_ids") or [],
                    "asset_role": candidate.get("asset_role"),
                }
        if not isinstance(record, dict):
            raise ToolRejected("unknown_media_source", f"unknown {kind} source: {source_id}")
        representation = self._representation_for_asset(record, representation_id)
        if representation_id != str(representation.get("representation_id") or ""):
            raise ToolRejected("representation_mismatch", f"representation does not match source {source_id}")
        if require_view:
            if kind == "generated_image":
                selection = self.shot["selections"].get(authorization_id)
                if not selection or selection.get("selected_candidate_id") != source_id:
                    raise ToolRejected("not_authorized", f"generated image {source_id} lacks selection authorization")
            else:
                viewed = self.shot["view_authorizations"].get(source_id) or []
                exact_authorization = any(
                    (
                        isinstance(row, Mapping)
                        and row.get("authorization_id") == authorization_id
                        and row.get("representation_id") == representation_id
                    )
                    or row == authorization_id
                    for row in viewed
                )
                if not exact_authorization:
                    raise ToolRejected("not_viewed", f"source {source_id} was not viewed in this shot")
        path = Path(str(representation.get("path") or ""))
        if not path.is_file():
            raise ToolRejected("media_missing", f"representation file is missing for {source_id}")
        return {"record": record, "representation": representation, "path": path}

    def tool_view_material_registry(self, arguments: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = arguments.get("snapshot_id")
        cursor = arguments.get("cursor")
        limit = int(arguments["limit"])
        if not self.shot["observation_order"] and (snapshot_id is not None or cursor is not None):
            raise ToolRejected("full_snapshot_required", "the first registry read requires snapshot_id=null")
        registry = self.state.data["registry"]
        snapshots = self.shot.setdefault("registry_snapshots", {})
        if snapshot_id is None:
            if snapshots:
                raise ToolRejected("snapshot_id_required", "continue the existing immutable registry snapshot")
            snapshot_id = self.state.next_id("registry")
            snapshots[snapshot_id] = {
                "registry_version": registry["version"],
                "items": copy.deepcopy(sorted(registry["items"].values(), key=lambda item: str(item.get("tag") or ""))),
                "expected_next_cursor": None,
                "started": False,
                "complete": False,
            }
            if cursor is not None:
                raise ToolRejected(
                    "invalid_registry_cursor", "the first registry page requires cursor=null"
                )
            offset = 0
        else:
            snapshot = snapshots.get(snapshot_id)
            if not snapshot:
                raise ToolRejected("unknown_registry_snapshot", "snapshot_id is not active for this shot")
            if snapshot.get("complete"):
                raise ToolRejected(
                    "registry_snapshot_complete",
                    "the immutable registry snapshot was already read completely",
                )
            expected_cursor = snapshot.get("expected_next_cursor")
            if not snapshot.get("started") or cursor != expected_cursor:
                raise ToolRejected(
                    "registry_cursor_out_of_order",
                    "registry pages must use the exact next_cursor returned by the prior page",
                    details={"expected_cursor": expected_cursor},
                )
            try:
                offset = int(cursor)
            except (TypeError, ValueError) as exc:
                raise ToolRejected("invalid_registry_cursor", "registry cursor must be a decimal offset") from exc
        frozen = snapshots[snapshot_id]
        page = frozen["items"][offset : offset + limit]
        next_offset = offset + len(page)
        next_cursor = str(next_offset) if next_offset < len(frozen["items"]) else None
        frozen["started"] = True
        frozen["expected_next_cursor"] = next_cursor
        frozen["complete"] = next_cursor is None
        assets = []
        for asset_id in self.shot.get("upload_ids") or []:
            asset = self.state.data["assets"].get(asset_id)
            if asset:
                assets.append(
                    {
                        "asset_id": asset_id,
                        "media_type": asset.get("media_type"),
                        "user_description": asset.get("user_description"),
                        "declared_tags": copy.deepcopy(
                            asset.get("source_declared_tags")
                            if isinstance(asset.get("source_declared_tags"), list)
                            else asset.get("declared_tags") or []
                        ),
                        "declared_asset_type": asset.get("declared_asset_type"),
                        "condition_eligible": bool(asset.get("condition_eligible", False)),
                        "memory_role": asset.get("memory_role"),
                        "selectors": asset.get("selectors") or [{"mode": "default_preview"}],
                        **(
                            {"speaker_name": asset.get("speaker_name")}
                            if asset.get("media_type") == "audio"
                            and isinstance(asset.get("speaker_name"), str)
                            and asset.get("speaker_name")
                            else {}
                        ),
                    }
                )
        committed = self.state.data.get("committed_shots") or []
        previous_tail = committed[-1].get("commit", {}).get("tail_handle") if committed else None
        snapshot = {
            "ok": True,
            "shot_request_id": self.shot["shot_request_id"],
            "raw_user_prompt_sha256": self.shot["raw_user_prompt_sha256"],
            "snapshot_id": snapshot_id,
            "registry_version": frozen["registry_version"],
            "materials": page,
            "next_cursor": next_cursor,
            "pending_uploads": assets,
            "previous_tail": previous_tail,
            "phase_budget": copy.deepcopy(self.shot["budgets"]),
            "capabilities": {
                "audio_transport": self.config.audio_transport,
                "audio_perception": self.config.audio_perception,
                "max_memory_slots": self.config.max_memory_slots,
                "prompt_rewrite_mode": self.config.prompt_rewrite_mode,
            },
            "profile_locks": self.pe_resolver.profile_locks(),
            "story_prompt_state": self._story_prompt_state(),
        }
        self.shot["phase"] = "ready"
        self.shot["registry_pagination_complete"] = next_cursor is None
        return self._new_observation("material_registry", snapshot)

    def tool_search_memory_bank(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"])
        top_k = int(arguments.get("top_k", 8))
        filters = arguments.get("filters") or {}
        registry_items = self.state.data.get("registry", {}).get("items", {})
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for memory_id, memory in self.state.data["memory_bank"].items():
            memory_tags = [str(item) for item in memory.get("tags") or [] if isinstance(item, str)]
            modalities = set(str(item) for item in filters.get("modalities") or [])
            if modalities and memory.get("modality") not in modalities:
                continue
            tags = set(str(item) for item in filters.get("tags") or [])
            if tags and not tags.intersection(memory_tags):
                continue
            roles = set(str(item) for item in filters.get("roles") or [])
            if roles and not roles.intersection(set(memory.get("roles") or [])):
                continue
            subject_ids = set(str(item) for item in filters.get("subject_ids") or [])
            if subject_ids and not subject_ids.issubset(set(memory.get("subject_ids") or [])):
                continue
            registry_text: list[str] = []
            matched_registry_tags: list[str] = []
            for tag in memory_tags:
                registry_row = registry_items.get(tag)
                if not isinstance(registry_row, Mapping):
                    continue
                matched_registry_tags.append(tag)
                registry_text.extend(
                    [
                        str(registry_row.get("stable_description") or ""),
                        str(registry_row.get("visual_description") or ""),
                        str(registry_row.get("voice_description") or ""),
                        *(str(item) for item in registry_row.get("aliases") or []),
                    ]
                )
            text = " ".join(
                [memory_id, str(memory.get("description") or "")]
                + memory_tags
                + [str(item) for item in memory.get("roles") or []]
                + [str(item) for item in memory.get("subject_ids") or []]
                + registry_text
            )
            score = _bm25_like_score(query, text)
            if score > 0:
                enriched = copy.deepcopy(memory)
                enriched["matched_registry_tags"] = matched_registry_tags
                rows.append((score, memory_id, enriched))
        rows.sort(key=lambda item: (-item[0], item[1]))
        candidates = [
            {
                "memory_id": memory_id,
                "modality": memory.get("modality"),
                "tags": memory.get("tags") or [],
                "roles": memory.get("roles") or [],
                "subject_ids": memory.get("subject_ids") or [],
                "description": memory.get("description") or "",
                "matched_registry_tags": memory.get("matched_registry_tags") or [],
                "speaker_binding": copy.deepcopy(memory.get("speaker_binding")),
                "transport_eligible": bool(memory.get("transport_eligible", False)),
                "transport_authorization_id": memory.get(
                    "transport_authorization_id"
                ),
                "audio_quality_status": memory.get("audio_quality_status"),
                "quality_authorization_id": memory.get("quality_authorization_id"),
                "verification_basis": memory.get("verification_basis"),
                "score": round(score, 6),
            }
            for score, memory_id, memory in rows[:top_k]
        ]
        search_id = self.state.next_id("search")
        self.shot["searches"][search_id] = {
            "query": query,
            "candidate_ids": [item["memory_id"] for item in candidates],
            "index_version": "storymem.lexical.v1",
        }
        self.shot["latest_search_id"] = search_id
        return self._new_observation(
            "memory_search",
            {
                "ok": True,
                "search_id": search_id,
                "query": query,
                "index_version": "storymem.lexical.v1",
                "candidates": candidates,
            },
        )

    def tool_view_memories(self, arguments: dict[str, Any]) -> dict[str, Any]:
        items = arguments.get("items") or []
        returned: list[dict[str, Any]] = []
        media: list[dict[str, Any]] = []
        authorized_representations: list[tuple[str, str]] = []
        authorization_id = self.state.next_id("view")
        for request_index, request in enumerate(items):
            kind = str(request["kind"])
            source_id = str(request["id"])
            record: dict[str, Any] | None = None
            supplemental_representations: list[dict[str, Any]] = []
            if kind == "memory":
                latest_search = self.shot.get("latest_search_id")
                search_id = request.get("search_id")
                if not latest_search or search_id != latest_search:
                    raise ToolRejected("stale_search", "memory view must use the single latest search_id")
                if source_id not in self.shot["searches"][latest_search]["candidate_ids"]:
                    raise ToolRejected("memory_not_in_search", f"memory {source_id} was not returned by the latest search")
                record = self.state.data["memory_bank"].get(source_id)
            elif kind == "uploaded_asset":
                if source_id not in self.shot.get("upload_ids", []):
                    raise ToolRejected("upload_not_in_shot", f"upload {source_id} is not available in this shot")
                record = self.state.data["assets"].get(source_id)
            elif kind == "previous_tail":
                committed = self.state.data.get("committed_shots") or []
                tail = committed[-1].get("commit", {}).get("tail_representation") if committed else None
                if isinstance(tail, dict) and source_id == str(tail.get("source_id") or tail.get("representation_id") or ""):
                    record = {"representation": tail, "description": "previous accepted shot tail"}
            if not isinstance(record, dict):
                raise ToolRejected("unknown_view_item", f"cannot view {kind}:{source_id}")
            if kind == "uploaded_asset":
                from .media import materialize_asset_selector

                try:
                    materialized = materialize_asset_selector(
                        record,
                        request["selector"],
                        self.config.output_dir / "materialized_uploads" / source_id,
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    raise ToolRejected(
                        "asset_selector_failed",
                        f"could not materialize selector for {source_id}: {exc}",
                    ) from exc
                representation = copy.deepcopy(materialized["representation"])
                supplemental_representations = copy.deepcopy(
                    materialized.get("supplemental_representations") or []
                )
                representations = record.setdefault("representations", {})
                for row in [representation, *supplemental_representations]:
                    representations[str(row["representation_id"])] = copy.deepcopy(row)
            else:
                requested_representation_id = request.get("representation_id")
                representation = self._representation_for_asset(record, requested_representation_id)
            path = Path(str(representation.get("path") or ""))
            if not path.is_file():
                raise ToolRejected("media_missing", f"media file is missing for {source_id}")
            item = {
                "kind": kind,
                "id": source_id,
                "authorization_id": authorization_id,
                "description": record.get("description") or record.get("user_description") or "",
                "representation": {key: value for key, value in representation.items() if key != "path"},
                "supplemental_representations": [
                    {key: value for key, value in row.items() if key != "path"}
                    for row in supplemental_representations
                ],
            }
            if kind == "uploaded_asset":
                # Upload-to-canonical bindings are trusted runtime lineage, not
                # model observations.  The model sees exactly the user's labels
                # and must discover canonical tags through the material registry.
                item.update(
                    {
                        "declared_tags": copy.deepcopy(
                            record.get("source_declared_tags")
                            if isinstance(record.get("source_declared_tags"), list)
                            else record.get("declared_tags") or []
                        ),
                        "declared_asset_type": record.get("declared_asset_type"),
                        "condition_eligible": bool(
                            record.get("condition_eligible", False)
                        ),
                    }
                )
            else:
                item.update(
                    {
                        "tags": record.get("tags")
                        or record.get("declared_tags")
                        or [],
                        "subject_ids": record.get("subject_ids") or [],
                    }
                )
            if str(representation.get("media_type") or "").startswith("audio"):
                source_basis = str(record.get("verification_basis") or "")
                speaker_binding = copy.deepcopy(record.get("speaker_binding"))
                if not source_basis and isinstance(speaker_binding, Mapping):
                    source_basis = str(speaker_binding.get("basis") or "")
                item.update(
                    {
                        "transported_as_audio": False,
                        "transport_eligible": bool(
                            record.get("transport_eligible", False)
                        ),
                        "perception_exposed": self.config.audio_perception
                        == "verified",
                        "transport_authorization_id": record.get(
                            "transport_authorization_id"
                        ),
                    }
                )
                if kind == "uploaded_asset":
                    item["speaker_name"] = record.get("speaker_name")
                else:
                    item.update(
                        {
                            "verification_basis": (
                                "agent_heard"
                                if self.config.audio_perception == "verified"
                                else source_basis or "unverified"
                            ),
                            "speaker_binding": speaker_binding,
                            "audio_quality_status": record.get(
                                "audio_quality_status", "unassessed"
                            ),
                            "quality_authorization_id": record.get(
                                "quality_authorization_id"
                            ),
                            "transport_description": record.get(
                                "transport_description"
                            )
                            or record.get("description")
                            or record.get("user_description")
                            or "",
                        }
                    )
            returned.append(item)
            perception_path = path
            perception_representation = representation
            if (
                str(representation.get("media_type") or "") == "video"
                and self.config.audio_perception != "verified"
            ):
                from .media import strip_video_audio

                suffix = path.suffix.lower() or ".mp4"
                perception_path, audio_removed = strip_video_audio(
                    path,
                    self.config.output_dir
                    / "perception_proxies"
                    / authorization_id
                    / f"item_{request_index:03d}_visual_only{suffix}",
                )
                if audio_removed:
                    perception_representation = self._media_manifest(
                        perception_path, self.state.next_id("repr_perception")
                    )
                    perception_representation.update(
                        {
                            "parent_sha256": representation.get("sha256"),
                            "derivation": {
                                "operation": "strip_audio_for_visual_perception",
                                "source_representation_id": representation.get(
                                    "representation_id"
                                ),
                            },
                        }
                    )
                    item["perception_representation"] = {
                        key: value
                        for key, value in perception_representation.items()
                        if key != "path"
                    }
            if not (
                str(representation.get("media_type") or "") == "audio"
                and self.config.audio_perception != "verified"
            ):
                media.append(
                    {
                        "type": str(
                            perception_representation.get("media_type")
                            or "application/octet-stream"
                        ),
                        "path": str(perception_path.resolve()),
                        "representation_id": perception_representation.get(
                            "representation_id"
                        ),
                        "source_id": source_id,
                    }
                )
            for supplemental in supplemental_representations:
                supplemental_path = Path(str(supplemental.get("path") or ""))
                if supplemental_path.is_file() and not (
                    str(supplemental.get("media_type") or "") == "audio"
                    and self.config.audio_perception != "verified"
                ):
                    media.append(
                        {
                            "type": str(supplemental.get("media_type") or "application/octet-stream"),
                            "path": str(supplemental_path.resolve()),
                            "representation_id": supplemental.get("representation_id"),
                            "source_id": source_id,
                            "supplemental": True,
                        }
                    )
            authorized_representations.append(
                (source_id, str(representation.get("representation_id") or ""))
            )
            authorized_representations.extend(
                (source_id, str(row.get("representation_id") or ""))
                for row in supplemental_representations
            )
        observation = self._new_observation(
            "memory_view",
            {"ok": True, "authorization_id": authorization_id, "items": returned, "media": media},
        )
        for source_id, representation_id in authorized_representations:
            self.shot["view_authorizations"].setdefault(source_id, []).append(
                {
                    "authorization_id": authorization_id,
                    "representation_id": representation_id,
                    "observation_id": observation["observation_id"],
                }
            )
        return observation

    @staticmethod
    def _validate_prompt_context(target: str, context: Mapping[str, Any]) -> None:
        if set(context) != {"condition", "has_reference_inputs"}:
            raise ToolRejected(
                "invalid_prompt_context",
                "prompt_context must contain exactly condition and has_reference_inputs",
            )
        if target == "image":
            if context.get("condition") is not None or not isinstance(context.get("has_reference_inputs"), bool):
                raise ToolRejected(
                    "invalid_prompt_context",
                    "image prompt_context requires condition=null and boolean has_reference_inputs",
                )
        elif target == "video":
            if context.get("has_reference_inputs") is not None:
                raise ToolRejected(
                    "invalid_prompt_context", "video prompt_context requires has_reference_inputs=null"
                )
            condition = context.get("condition")
            if condition is not None:
                if set(condition) != {"relation", "image_source"}:
                    raise ToolRejected("invalid_condition", "condition requires relation and image_source")
                if condition.get("relation") not in {"frame0_match", "temporal_continuation"}:
                    raise ToolRejected("invalid_condition", "invalid condition relation")
                image_source = condition.get("image_source")
                if not isinstance(image_source, Mapping):
                    raise ToolRejected("invalid_condition", "condition.image_source must be an object")
        else:
            raise ToolRejected("invalid_target", f"invalid prompt target: {target}")

    def _validate_generation_scope(
        self, target: str, purpose: str, scope: Any
    ) -> dict[str, Any] | None:
        """Validate the immutable semantic scope of one prompt/group.

        PE resources may be identical for two image groups, while the groups
        still depict different subjects or serve different reusable roles.
        Keeping that distinction in the receipt prevents false duplicate-PE
        rejection without weakening the receipt-to-generation binding.
        """

        if target == "video":
            if scope is not None:
                raise ToolRejected(
                    "invalid_generation_scope",
                    "video prompts require generation_scope=null",
                )
            return None
        if not isinstance(scope, Mapping) or set(scope) != {
            "subject_ids",
            "asset_role",
        }:
            raise ToolRejected(
                "invalid_generation_scope",
                "image generation_scope requires exactly subject_ids and asset_role",
            )
        subject_ids = self._validate_subject_ids(
            list(scope.get("subject_ids") or []),
            label="generation_scope.subject_ids",
        )
        asset_role = scope.get("asset_role")
        if purpose == "first_frame" and asset_role is not None:
            raise ToolRejected(
                "invalid_generation_scope",
                "first_frame generation_scope requires asset_role=null",
            )
        if purpose == "reference_asset" and not str(asset_role or "").strip():
            raise ToolRejected(
                "invalid_generation_scope",
                "reference_asset generation_scope requires a non-empty asset_role",
            )
        if purpose == "reference_asset":
            role_text = str(asset_role or "")
            entity_match = _ENTITY_ASSET_ROLE_RE.fullmatch(role_text)
            if "ENTITY_" in role_text and entity_match is None:
                raise ToolRejected(
                    "invalid_entity_asset_role",
                    "an entity reference asset_role must be entity_identity:ENTITY_N "
                    "or entity_appearance:ENTITY_N",
                )
            if entity_match is not None:
                entity_tag = entity_match.group(1)
                if entity_tag not in self._known_entity_tags():
                    raise ToolRejected(
                        "unknown_entity_tag",
                        f"image generation_scope references unknown entity tag {entity_tag}",
                    )
                if subject_ids:
                    raise ToolRejected(
                        "entity_subject_ids_forbidden",
                        "non-human entity reference assets require subject_ids=[]",
                    )
        return {"subject_ids": subject_ids, "asset_role": asset_role}

    def _ordered_evidence(self, ids: list[str]) -> list[str]:
        if len(ids) != len(set(ids)):
            raise ToolRejected("duplicate_evidence", "context observation IDs must be unique")
        observations = self.shot["observations"]
        for observation_id in ids:
            observation = observations.get(observation_id)
            if not observation or observation.get("kind") not in {
                "material_registry",
                "memory_view",
                "candidate_inspection",
            }:
                raise ToolRejected(
                    "invalid_evidence",
                    f"{observation_id} is not a visible registry/view/inspection observation",
                )
        sorted_ids = sorted(ids, key=lambda item: observations[item]["sequence"])
        if ids != sorted_ids:
            raise ToolRejected("evidence_order", "context observation IDs must follow observation order")
        if not any(observations[item]["kind"] == "material_registry" for item in ids):
            raise ToolRejected("registry_evidence_required", "PE requires the current registry observation")
        return ids

    def _validate_pe_evidence_binding(
        self,
        *,
        target: str,
        operation: str,
        context: Mapping[str, Any],
        assessment_id: str | None,
        evidence_ids: list[str],
    ) -> None:
        """Require exact visible observations for condition/revision context."""

        required: set[str] = set()
        if operation == "revise":
            assessment = self.shot["assessments"].get(assessment_id)
            inspection = (
                self.shot["inspections"].get(assessment.get("inspection_id"))
                if isinstance(assessment, Mapping)
                else None
            )
            observation_id = (
                str(inspection.get("observation_id") or "")
                if isinstance(inspection, Mapping)
                else ""
            )
            if not observation_id:
                raise ToolRejected(
                    "revision_inspection_missing",
                    "revision assessment has no visible candidate-inspection observation",
                )
            required.add(observation_id)

        condition = context.get("condition") if target == "video" else None
        if target == "video":
            # ``direct_continuation`` is a story-plan commitment, not merely
            # permission to use the previous tail.  Validate the complete
            # condition (including a required non-null tail binding) before a
            # video PE receipt can be issued.
            self._validate_condition(condition)
        if condition is not None:
            source = condition["image_source"]
            source_id = str(source["source_id"])
            representation_id = str(source["representation_id"])
            authorization_id = str(source["authorization_id"])
            if source.get("kind") == "generated_image":
                candidate = self.shot["candidates"].get(source_id)
                inspection = (
                    self.shot["inspections"].get(candidate.get("inspection_id"))
                    if isinstance(candidate, Mapping)
                    else None
                )
                observation_id = (
                    str(inspection.get("observation_id") or "")
                    if isinstance(inspection, Mapping)
                    else ""
                )
            else:
                authorization = next(
                    (
                        row
                        for row in self.shot.get("view_authorizations", {}).get(source_id, [])
                        if isinstance(row, Mapping)
                        and row.get("authorization_id") == authorization_id
                        and row.get("representation_id") == representation_id
                    ),
                    None,
                )
                observation_id = (
                    str(authorization.get("observation_id") or "")
                    if isinstance(authorization, Mapping)
                    else ""
                )
            if not observation_id:
                raise ToolRejected(
                    "condition_evidence_missing",
                    "the exact condition has no visible observation authorization",
                )
            required.add(observation_id)

        missing = sorted(required - set(evidence_ids))
        if missing:
            raise ToolRejected(
                "required_pe_evidence_missing",
                "PE evidence must include the exact condition and revision observations",
                details={"missing_observation_ids": missing},
            )

    def _assert_new_image_group_available(self, purpose: str) -> None:
        if self._remaining_generation_attempts("image") <= 0:
            raise ToolRejected(
                "generation_budget_exhausted",
                "the shot-wide image candidate budget cannot fund another group",
            )
        image_groups = [
            group
            for group in self.shot.get("generation_groups", {}).values()
            if group.get("target") == "image"
        ]
        if purpose == "first_frame" and any(
            group.get("purpose") == "first_frame" for group in image_groups
        ):
            raise ToolRejected(
                "first_frame_group_already_started",
                "a shot may start at most one first_frame image group",
            )
        if purpose == "reference_asset" and sum(
            group.get("purpose") == "reference_asset" for group in image_groups
        ) >= self.config.max_memory_slots:
            raise ToolRejected(
                "reference_asset_capacity_exhausted",
                "another generated reference asset could not fit in the Echo memory slots",
            )

    def tool_get_pe_skill(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._assert_request(arguments)
        target = str(arguments["target"])
        if self.config.prompt_rewrite_mode == "passthrough_disabled":
            raise ToolRejected(
                "pe_disabled",
                "PE is disabled for this story; register the raw USER request "
                "directly with skill_application_receipt_id=null",
            )
        operation = str(arguments["operation"])
        purpose = str(arguments["purpose"])
        context = copy.deepcopy(arguments["prompt_context"])
        self._validate_prompt_context(target, context)
        generation_scope = self._validate_generation_scope(
            target, purpose, copy.deepcopy(arguments["generation_scope"])
        )
        evidence_ids = self._ordered_evidence(list(arguments.get("context_observation_ids") or []))
        parent_id = arguments.get("parent_prompt_artifact_id")
        assessment_id = arguments.get("assessment_id")
        axes = list(arguments.get("axes") or [])
        obligations = self._retry_obligations()
        if operation == "compose":
            if parent_id is not None or assessment_id is not None or axes:
                raise ToolRejected("invalid_compose_lineage", "compose cannot have parent, assessment, or axes")
            if obligations:
                raise ToolRejected(
                    "retry_obligation_pending",
                    "a fresh compose cannot bypass an assessment-authorized retry",
                    details={"generation_group_ids": [item[0] for item in obligations]},
                )
            if target == "image":
                self._assert_new_image_group_available(purpose)
        else:
            parent = self.shot["prompt_artifacts"].get(parent_id)
            assessment = self.shot["assessments"].get(assessment_id)
            if not parent or parent.get("target") != target:
                raise ToolRejected("invalid_revision_parent", "revise requires a same-target parent artifact")
            if not assessment or assessment.get("candidate_id") != parent.get("last_candidate_id"):
                raise ToolRejected("invalid_revision_assessment", "revise assessment does not match parent candidate")
            expected_axes = assessment.get("retry_change", {}).get("axes") or []
            if axes != expected_axes:
                raise ToolRejected("retry_axes_mismatch", "PE axes must exactly match the assessment")
            if purpose != parent.get("purpose"):
                raise ToolRejected("purpose_changed", "revision purpose must equal parent purpose")
            if generation_scope != parent.get("generation_scope"):
                raise ToolRejected(
                    "generation_scope_changed",
                    "revision generation_scope must equal its parent artifact",
                )
            candidate = self.shot["candidates"].get(assessment.get("candidate_id"))
            group_id = str(candidate.get("group_id") or "") if candidate else ""
            group = self.shot["generation_groups"].get(group_id)
            obligation = group.get("retry_obligation") if isinstance(group, Mapping) else None
            if (
                not isinstance(obligation, Mapping)
                or obligation.get("assessment_id") != assessment_id
                or obligation.get("candidate_id") != assessment.get("candidate_id")
                or any(other_group_id != group_id for other_group_id, _ in obligations)
            ):
                raise ToolRejected(
                    "retry_obligation_mismatch",
                    "revision must serve the exact outstanding generation-group retry",
                )
        self._validate_pe_evidence_binding(
            target=target,
            operation=operation,
            context=context,
            assessment_id=str(assessment_id) if assessment_id is not None else None,
            evidence_ids=evidence_ids,
        )
        allowed_axes = _IMAGE_AXES if target == "image" else _VIDEO_AXES
        if set(axes) - allowed_axes or len(axes) != len(set(axes)):
            raise ToolRejected("invalid_retry_axes", f"invalid or duplicate {target} retry axes")
        if operation == "revise":
            prompt_affecting = (
                "prompt" in axes
                or (target == "video" and "condition" in axes)
                or (
                    target == "image"
                    and "references" in axes
                    and context.get("has_reference_inputs")
                    != self.shot["prompt_artifacts"][parent_id]["prompt_context"].get("has_reference_inputs")
                )
            )
            if not prompt_affecting:
                raise ToolRejected("unnecessary_revision", "these retry axes must reuse the existing prompt artifact")
        application_key = sha256_json(
            {
                "target": target,
                "operation": operation,
                "purpose": purpose,
                "context": context,
                "generation_scope": generation_scope,
                "evidence_ids": evidence_ids,
                "parent": parent_id,
                "assessment": assessment_id,
                "axes": axes,
            }
        )
        if any(item.get("application_key") == application_key for item in self.shot["pe_receipts"].values()):
            raise ToolRejected("duplicate_pe_application", "identical PE application context was already loaded")
        bundle = self.pe_resolver.resolve(
            target=target, operation=operation, purpose=purpose, prompt_context=context
        )
        receipt_id = self.state.next_id("peskill")
        receipt = {
            "skill_application_receipt_id": receipt_id,
            "application_key": application_key,
            "target": target,
            "operation": operation,
            "purpose": purpose,
            "context_observation_ids": evidence_ids,
            "prompt_context": context,
            "generation_scope": generation_scope,
            "parent_prompt_artifact_id": parent_id,
            "assessment_id": assessment_id,
            "axes": axes,
            "bundle": bundle.as_observation(),
            "registered_prompt_artifact_id": None,
        }
        self.shot["pe_receipts"][receipt_id] = receipt
        self.shot["pending_pe_receipt_id"] = receipt_id
        return {"ok": True, "skill_application_receipt_id": receipt_id, **bundle.as_observation()}

    def tool_register_prompt(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._assert_request(arguments)
        receipt_value = arguments.get("skill_application_receipt_id")
        if receipt_value is None:
            return self._register_passthrough_prompt(arguments)
        receipt_id = str(receipt_value)
        receipt = self.shot["pe_receipts"].get(receipt_id)
        if not receipt:
            raise ToolRejected("unknown_pe_receipt", f"unknown PE receipt: {receipt_id}")
        if self.config.prompt_rewrite_mode == "passthrough_disabled":
            raise ToolRejected(
                "pe_disabled",
                "passthrough_disabled stories cannot register a PE-authored prompt",
            )
        if receipt.get("registered_prompt_artifact_id") is not None:
            raise ToolRejected(
                "pe_receipt_already_consumed",
                "a PE application receipt can register exactly one accepted prompt artifact",
                details={"prompt_artifact_id": receipt["registered_prompt_artifact_id"]},
            )
        if self.shot.get("pending_pe_receipt_id") != receipt_id:
            raise ToolRejected(
                "pe_receipt_not_pending",
                "register_prompt must consume the exact pending PE receipt",
                details={"pending_pe_receipt_id": self.shot.get("pending_pe_receipt_id")},
            )
        compared = {
            key: arguments.get(key)
            for key in (
                "target",
                "operation",
                "purpose",
                "generation_scope",
                "parent_prompt_artifact_id",
                "assessment_id",
            )
        }
        expected = {key: receipt.get(key) for key in compared}
        if compared != expected:
            raise ToolRejected("receipt_context_mismatch", "register_prompt does not match its PE receipt")
        obligations = self._retry_obligations()
        if receipt.get("operation") == "compose" and obligations:
            raise ToolRejected(
                "retry_obligation_pending",
                "a compose artifact cannot bypass an assessment-authorized retry",
            )
        if receipt.get("operation") == "revise":
            assessment = self.shot["assessments"].get(receipt.get("assessment_id"))
            candidate = (
                self.shot["candidates"].get(assessment.get("candidate_id"))
                if isinstance(assessment, Mapping)
                else None
            )
            group_id = str(candidate.get("group_id") or "") if candidate else ""
            group = self.shot["generation_groups"].get(group_id)
            obligation = group.get("retry_obligation") if isinstance(group, Mapping) else None
            if (
                not isinstance(obligation, Mapping)
                or obligation.get("assessment_id") != receipt.get("assessment_id")
                or any(other_group_id != group_id for other_group_id, _ in obligations)
            ):
                raise ToolRejected(
                    "retry_obligation_mismatch",
                    "revision artifact must serve the exact outstanding retry",
                )
        evidence_ids = list(arguments.get("evidence_ids") or [])
        if evidence_ids != receipt["context_observation_ids"]:
            raise ToolRejected("receipt_evidence_mismatch", "evidence_ids must exactly equal the PE receipt evidence")
        axes = list(arguments.get("axes") or [])
        if axes != receipt["axes"]:
            raise ToolRejected("receipt_axes_mismatch", "register_prompt axes must exactly equal PE receipt axes")
        context = copy.deepcopy(arguments["prompt_context"])
        if context != receipt["prompt_context"]:
            raise ToolRejected("receipt_prompt_context_mismatch", "prompt_context must exactly equal the PE receipt")
        text = arguments["text"]
        story_prompt_state = self._story_prompt_state()
        try:
            counts = self.prompt_validator.validate(
                text,
                target=receipt["target"],
                contract=receipt["bundle"]["validation_contract"],
                raw_user_request=self.shot["raw_user_request"],
                allowed_person_ids=(
                    self._active_person_subject_ids()
                    if receipt["target"] == "video"
                    else self._known_person_subject_ids()
                ),
                prompt_context=context,
                expected_caption_language=story_prompt_state.get(
                    "caption_language"
                ),
            )
        except PromptValidationError as exc:
            raise ToolRejected("prompt_validation_failed", str(exc)) from exc
        artifact_id = self.state.next_id("prompt")
        parent = self.shot["prompt_artifacts"].get(receipt.get("parent_prompt_artifact_id"))
        artifact = {
            "prompt_artifact_id": artifact_id,
            "target": receipt["target"],
            "operation": receipt["operation"],
            "purpose": receipt["purpose"],
            "text": text,
            "text_sha256": sha256_text(text),
            "skill_application_receipt_id": receipt_id,
            "receipt_chain": (parent.get("receipt_chain") if parent else []) + [receipt_id],
            "prompt_rewrite_mode": self.config.prompt_rewrite_mode,
            "parent_prompt_artifact_id": receipt.get("parent_prompt_artifact_id"),
            "assessment_id": receipt.get("assessment_id"),
            "axes": axes,
            "evidence_ids": evidence_ids,
            "prompt_context": context,
            "generation_scope": copy.deepcopy(receipt.get("generation_scope")),
            "prompt_context_sha256": sha256_json(context),
            "revision_count": int(parent.get("revision_count", -1)) + 1 if parent else 0,
            "validation": counts,
            "consumed": False,
        }
        evaluation_contract_authoring = build_evaluation_contract_authoring(
            shot_request_id=self.shot["shot_request_id"],
            raw_user_request=self.shot["raw_user_request"],
            prompt_artifact_id=artifact_id,
            receipt_chain=artifact["receipt_chain"],
        )
        evaluation_contract_authoring_sha256 = sha256_json(
            evaluation_contract_authoring
        )
        artifact["evaluation_contract_authoring"] = (
            evaluation_contract_authoring
        )
        artifact["evaluation_contract_authoring_sha256"] = (
            evaluation_contract_authoring_sha256
        )
        self.shot["prompt_artifacts"][artifact_id] = artifact
        receipt["registered_prompt_artifact_id"] = artifact_id
        self.shot["pending_pe_receipt_id"] = None
        self.shot["pending_prompt_artifact_id"] = artifact_id
        return {
            "ok": True,
            "prompt_artifact_id": artifact_id,
            "text_sha256": artifact["text_sha256"],
            "prompt_context_sha256": artifact["prompt_context_sha256"],
            "receipt_chain": artifact["receipt_chain"],
            "revision_count": artifact["revision_count"],
            "evaluation_contract_authoring": copy.deepcopy(
                evaluation_contract_authoring
            ),
            "evaluation_contract_authoring_sha256": (
                evaluation_contract_authoring_sha256
            ),
            **counts,
        }

    def _register_passthrough_prompt(
        self, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Register one raw, already-authored video prompt without loading PE.

        This route is intentionally narrow: PE and revision are unavailable,
        and the exact raw USER string is the only prompt that can be persisted
        or sent to either media provider.
        """

        if self.config.prompt_rewrite_mode != "passthrough_disabled":
            raise ToolRejected(
                "pe_receipt_required",
                "rewrite_enabled mode requires a get_pe_skill receipt",
            )
        target = str(arguments.get("target") or "")
        purpose = str(arguments.get("purpose") or "")
        expected_route = {
            "target": target,
            "operation": "compose",
            "parent_prompt_artifact_id": None,
            "assessment_id": None,
            "axes": [],
        }
        actual_route = {
            key: copy.deepcopy(arguments.get(key)) for key in expected_route
        }
        if actual_route != expected_route:
            raise ToolRejected(
                "invalid_passthrough_route",
                "passthrough_disabled accepts only compose with null lineage and axes=[]",
                details={"expected": expected_route},
            )
        if target not in {"image", "video"}:
            raise ToolRejected("invalid_target", f"invalid prompt target: {target}")
        if (target == "video" and purpose != "shot") or (
            target == "image" and purpose not in {"first_frame", "reference_asset"}
        ):
            raise ToolRejected(
                "invalid_passthrough_route",
                "passthrough prompt purpose is incompatible with its target",
            )
        if self.shot.get("pending_pe_receipt_id") is not None:
            raise ToolRejected(
                "pe_receipt_pending",
                "the pending PE receipt must be consumed before direct registration",
            )
        obligations = self._retry_obligations()
        if obligations:
            raise ToolRejected(
                "retry_obligation_pending",
                "a direct compose artifact cannot bypass an assessment-authorized retry",
            )
        context = copy.deepcopy(arguments.get("prompt_context"))
        if not isinstance(context, Mapping):
            raise ToolRejected(
                "invalid_prompt_context", "prompt_context must be an object"
            )
        self._validate_prompt_context(target, context)
        generation_scope = self._validate_generation_scope(
            target, purpose, arguments.get("generation_scope")
        )
        if target == "image":
            self._assert_new_image_group_available(purpose)
        evidence_ids = self._ordered_evidence(
            list(arguments.get("evidence_ids") or [])
        )
        self._validate_pe_evidence_binding(
            target=target,
            operation="compose",
            context=context,
            assessment_id=None,
            evidence_ids=evidence_ids,
        )
        text = str(arguments.get("text"))
        raw_user_request = str(self.shot["raw_user_request"])
        if text.encode("utf-8") != raw_user_request.encode("utf-8"):
            raise ToolRejected(
                "passthrough_text_mismatch",
            "passthrough prompt text must equal the raw USER request byte-for-byte",
            )

        artifact_id = self.state.next_id("prompt")
        validation: dict[str, Any] = {
            "mode": "passthrough_disabled",
            "char_count": len(text),
            "byte_count": len(text.encode("utf-8")),
            "token_count": None,
        }
        artifact = {
            "prompt_artifact_id": artifact_id,
            "target": target,
            "operation": "compose",
            "purpose": purpose,
            "text": text,
            "text_sha256": sha256_text(text),
            "skill_application_receipt_id": None,
            "receipt_chain": [],
            "prompt_rewrite_mode": "passthrough_disabled",
            "parent_prompt_artifact_id": None,
            "assessment_id": None,
            "axes": [],
            "evidence_ids": evidence_ids,
            "prompt_context": context,
            "generation_scope": copy.deepcopy(generation_scope),
            "prompt_context_sha256": sha256_json(context),
            "revision_count": 0,
            "validation": validation,
            "consumed": False,
        }
        evaluation_contract_authoring = build_evaluation_contract_authoring(
            shot_request_id=self.shot["shot_request_id"],
            raw_user_request=raw_user_request,
            prompt_artifact_id=artifact_id,
            receipt_chain=[],
        )
        artifact["evaluation_contract_authoring"] = evaluation_contract_authoring
        artifact["evaluation_contract_authoring_sha256"] = sha256_json(
            evaluation_contract_authoring
        )
        self.shot["prompt_artifacts"][artifact_id] = artifact
        self.shot["pending_prompt_artifact_id"] = artifact_id
        return {
            "ok": True,
            "prompt_artifact_id": artifact_id,
            "text_sha256": artifact["text_sha256"],
            "prompt_context_sha256": artifact["prompt_context_sha256"],
            "receipt_chain": [],
            "revision_count": 0,
            "evaluation_contract_authoring": copy.deepcopy(
                evaluation_contract_authoring
            ),
            "evaluation_contract_authoring_sha256": artifact[
                "evaluation_contract_authoring_sha256"
            ],
            **validation,
        }

    def _assert_passthrough_artifact_integrity(
        self, artifact: Mapping[str, Any], *, target: str
    ) -> None:
        text = str(artifact.get("text") or "")
        expected_validation = {
            "mode": "passthrough_disabled",
            "char_count": len(text),
            "byte_count": len(text.encode("utf-8")),
            "token_count": None,
        }
        if (
            self.config.prompt_rewrite_mode != "passthrough_disabled"
            or artifact.get("prompt_rewrite_mode") != "passthrough_disabled"
            or artifact.get("target") != target
            or artifact.get("skill_application_receipt_id") is not None
            or list(artifact.get("receipt_chain") or [])
            or artifact.get("text") != self.shot["raw_user_request"]
            or artifact.get("operation") != "compose"
            or artifact.get("parent_prompt_artifact_id") is not None
            or artifact.get("assessment_id") is not None
            or list(artifact.get("axes") or [])
            or artifact.get("validation") != expected_validation
        ):
            raise ToolRejected(
                "passthrough_prompt_invalid",
                "direct prompt artifact no longer matches its immutable raw USER registration",
            )

    def _generation_retry(
        self, *, target: str, group_id: str | None, retry: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any] | None, list[str]]:
        obligations = self._retry_obligations()
        if group_id is None:
            if retry is not None:
                raise ToolRejected("retry_without_group", "first generation call requires retry=null")
            if obligations:
                raise ToolRejected(
                    "retry_obligation_pending",
                    "the assessment-authorized retry must be continued before starting another group",
                    details={"generation_group_ids": [item[0] for item in obligations]},
                )
            if target == "video" and any(
                item.get("target") == "video"
                for item in self.shot.get("generation_groups", {}).values()
            ):
                raise ToolRejected(
                    "video_group_already_started",
                    "a shot may start exactly one video generation group",
                )
            if self._remaining_generation_attempts(target) <= 0:
                raise ToolRejected(
                    "generation_budget_exhausted",
                    f"the shot-wide {target} candidate budget is exhausted",
                )
            return None, []
        group = self.shot["generation_groups"].get(group_id)
        if not group or group.get("target") != target:
            raise ToolRejected("unknown_generation_group", f"unknown {target} generation group")
        if group.get("selection_id"):
            raise ToolRejected("generation_group_closed", "selected generation group cannot be retried")
        if not isinstance(retry, Mapping):
            raise ToolRejected("retry_required", "an existing generation group requires retry metadata")
        candidate_id = str(retry.get("candidate_id") or "")
        assessment_id = str(retry.get("assessment_id") or "")
        candidate = self.shot["candidates"].get(candidate_id)
        if not candidate or candidate.get("group_id") != group_id or candidate.get("target") != target:
            raise ToolRejected(
                "retry_candidate_group_mismatch",
                "retry candidate must belong to the requested generation group",
            )
        assessment = self.shot["assessments"].get(assessment_id)
        if not assessment or assessment.get("candidate_id") != candidate_id:
            raise ToolRejected("invalid_retry_assessment", "retry does not reference a matching assessment")
        if assessment.get("next_action") != "retry":
            raise ToolRejected("retry_not_authorized", "assessment did not authorize retry")
        obligation = group.get("retry_obligation")
        if not isinstance(obligation, Mapping):
            raise ToolRejected(
                "retry_obligation_missing",
                "the generation group has no outstanding retry obligation",
            )
        if any(other_group_id != group_id for other_group_id, _ in obligations):
            raise ToolRejected(
                "different_retry_obligation_pending",
                "another generation group has an outstanding retry obligation",
            )
        if (
            obligation.get("candidate_id") != candidate_id
            or obligation.get("assessment_id") != assessment_id
        ):
            raise ToolRejected(
                "retry_obligation_mismatch",
                "retry must consume the group's exact latest assessment obligation",
            )
        axes = list(retry.get("axes") or [])
        expected = assessment.get("retry_change", {}).get("axes") or []
        if axes != expected or axes != list(obligation.get("axes") or []):
            raise ToolRejected("retry_axes_mismatch", "generation retry axes must exactly match assessment")
        if self._remaining_generation_attempts(target) <= 0:
            raise ToolRejected("generation_budget_exhausted", "candidate budget is exhausted; select best available")
        return group, axes

    def _remaining_generation_attempts(self, target: str) -> int:
        budget_key = "image_attempts" if target == "image" else "video_attempts"
        configured = int(
            self.shot.get("budgets", {}).get(
                budget_key,
                self.config.image_attempts if target == "image" else self.config.video_attempts,
            )
        )
        used = sum(
            1
            for candidate in self.shot.get("candidates", {}).values()
            if candidate.get("target") == target
        )
        return max(0, configured - used)

    def _assert_prompt_artifact_ready_for_generation(
        self,
        *,
        target: str,
        artifact_id: str,
        generation_group_id: str | None,
        retry: Mapping[str, Any] | None,
        retry_axes: list[str],
        input_references: list[Mapping[str, Any]] | None = None,
    ) -> None:
        """Bind registration to the one generation call it authorized.

        A newly composed prompt, and a revision needed by a prompt-affecting
        retry, must be the runtime's currently pending artifact.  The only
        legal generation without a pending artifact is a retry that reuses the
        assessed candidate's exact prompt (for example a seed-only retry).
        """

        pending = self.shot.get("pending_prompt_artifact_id")
        artifact = self.shot.get("prompt_artifacts", {}).get(artifact_id) or {}
        passthrough = artifact.get("prompt_rewrite_mode") == "passthrough_disabled"
        if pending is not None and pending != artifact_id:
            raise ToolRejected(
                "pending_prompt_artifact_mismatch",
                f"generation must consume pending prompt artifact {pending}",
                details={"pending_prompt_artifact_id": pending},
            )

        requires_pending = generation_group_id is None
        if retry is not None:
            if passthrough:
                if "prompt" in retry_axes:
                    raise ToolRejected(
                        "passthrough_retry_axes_forbidden",
                        "passthrough_disabled retries cannot change prompt text",
                    )
                requires_pending = False
            elif target == "video":
                requires_pending = bool(set(retry_axes) & {"prompt", "condition"})
            else:
                prior = self.shot["candidates"].get(str(retry.get("candidate_id") or ""))
                prior_references = (
                    list(prior.get("input_references") or [])
                    if isinstance(prior, Mapping)
                    else []
                )
                requires_pending = (
                    "prompt" in retry_axes
                    or bool(input_references or []) != bool(prior_references)
                )
        if requires_pending and pending != artifact_id:
            raise ToolRejected(
                "prompt_artifact_not_pending",
                "a new group or prompt-affecting retry requires its newly registered pending artifact",
            )

    def _validate_retry_changes(
        self,
        *,
        target: str,
        artifact: Mapping[str, Any],
        retry: Mapping[str, Any] | None,
        axes: list[str],
        condition: Any = None,
        memory_slots: list[Mapping[str, Any]] | None = None,
        input_references: list[Mapping[str, Any]] | None = None,
        seed: int,
    ) -> None:
        if retry is None:
            return
        prior = self.shot["candidates"][str(retry["candidate_id"])]
        changed: set[str] = set()
        prior_artifact = self.shot["prompt_artifacts"][prior["prompt_artifact_id"]]
        passthrough = (
            artifact.get("prompt_rewrite_mode") == "passthrough_disabled"
            or prior_artifact.get("prompt_rewrite_mode") == "passthrough_disabled"
        )
        if passthrough and "prompt" in axes:
            raise ToolRejected(
                "passthrough_retry_axes_forbidden",
                "passthrough_disabled retries cannot change prompt text",
            )
        if artifact.get("text") != prior_artifact.get("text"):
            changed.add("prompt")
        if int(seed) != int(prior.get("seed")):
            changed.add("seed")
        if target == "image":
            previous_references = prior.get("input_references") or []
            current_references = input_references or []
            if self._reference_signature(current_references) != self._reference_signature(
                previous_references
            ):
                changed.add("references")
            prompt_affecting = (
                False
                if passthrough
                else "prompt" in axes
                or bool(current_references) != bool(previous_references)
            )
        else:
            if self._condition_signature(condition) != self._condition_signature(
                prior.get("condition")
            ):
                changed.add("condition")
            if self._memory_slot_signature(
                memory_slots or []
            ) != self._memory_slot_signature(prior.get("memory_slots") or []):
                changed.add("memory_slots")
            prompt_affecting = (
                False
                if passthrough
                else bool(set(axes) & {"prompt", "condition"})
            )
        if changed != set(axes):
            raise ToolRejected(
                "retry_change_mismatch",
                "declared retry axes must exactly match prompt/media/seed changes",
                details={"declared": axes, "actual": sorted(changed)},
            )
        if prompt_affecting:
            if (
                artifact.get("operation") != "revise"
                or artifact.get("assessment_id") != retry.get("assessment_id")
                or artifact.get("parent_prompt_artifact_id") != prior.get("prompt_artifact_id")
            ):
                raise ToolRejected(
                    "retry_revision_lineage_mismatch",
                    "prompt-affecting retry requires the assessment-bound child artifact",
                )
        elif artifact.get("prompt_artifact_id") != prior.get("prompt_artifact_id"):
            raise ToolRejected(
                "retry_prompt_not_reused",
                "seed-only or compatible-reference/slot retry must reuse the assessed candidate's exact prompt artifact",
            )

    @staticmethod
    def _source_signature(source: Any) -> tuple[str, str, str] | None:
        if source is None:
            return None
        if not isinstance(source, Mapping):
            return ("", "", "")
        return (
            str(source.get("kind") or ""),
            str(source.get("source_id") or ""),
            str(source.get("representation_id") or ""),
        )

    @classmethod
    def _reference_signature(
        cls, references: list[Mapping[str, Any]]
    ) -> tuple[tuple[str, str, str] | None, ...]:
        return tuple(cls._source_signature(item) for item in references)

    @classmethod
    def _condition_signature(cls, condition: Any) -> Any:
        if condition is None:
            return None
        if not isinstance(condition, Mapping):
            return ("", None)
        return (
            str(condition.get("relation") or ""),
            cls._source_signature(condition.get("image_source")),
        )

    @classmethod
    def _memory_slot_signature(
        cls, slots: list[Mapping[str, Any]]
    ) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                cls._source_signature(slot.get("image_source")),
                cls._source_signature(slot.get("audio_source")),
                tuple(sorted(str(item) for item in slot.get("roles") or [])),
                tuple(sorted(str(item) for item in slot.get("subject_ids") or [])),
            )
            for slot in slots
        )

    def _artifact(self, artifact_id: str, target: str) -> dict[str, Any]:
        artifact = self.shot["prompt_artifacts"].get(artifact_id)
        if not artifact or artifact.get("target") != target:
            raise ToolRejected("invalid_prompt_artifact", f"invalid {target} prompt artifact")
        return artifact

    def _known_person_subject_ids(self) -> set[str]:
        return {
            str(tag)
            for tag, row in self.state.data.get("registry", {}).get("items", {}).items()
            if isinstance(row, Mapping)
            and row.get("category") == "person"
            and re.fullmatch(r"ID_[A-Z][A-Z0-9_]*", str(tag))
        }

    def _active_person_subject_ids(self) -> set[str]:
        prompt_state = self._story_prompt_state()
        shot_plan = prompt_state.get("current_shot_plan")
        active_tags = (
            shot_plan.get("active_material_tags")
            if isinstance(shot_plan, Mapping)
            else []
        )
        return self._known_person_subject_ids().intersection(
            str(tag) for tag in active_tags or []
        )

    def _known_entity_tags(self) -> set[str]:
        return {
            str(tag)
            for tag, row in self.state.data.get("registry", {}).get("items", {}).items()
            if isinstance(row, Mapping)
            and row.get("category") == "entity"
            and re.fullmatch(r"ENTITY_[1-9][0-9]*", str(tag))
        }

    def _entity_tags_for_record(self, record: Mapping[str, Any]) -> set[str]:
        known = self._known_entity_tags()
        tags: set[str] = set()
        for key in ("tags", "declared_tags", "canonical_material_tags"):
            values = record.get(key) or []
            if isinstance(values, list):
                tags.update(str(value) for value in values if str(value) in known)
        asset_role = str(record.get("asset_role") or "")
        match = _ENTITY_ASSET_ROLE_RE.fullmatch(asset_role)
        if match and match.group(1) in known:
            tags.add(match.group(1))
        return tags

    @staticmethod
    def _audio_time_range_is_exact(representation: Mapping[str, Any]) -> bool:
        selector = representation.get("selector") or {}
        return bool(
            isinstance(selector, Mapping)
            and selector.get("mode") == "time_range"
            and type(selector.get("start_ms")) is int
            and type(selector.get("end_ms")) is int
            and int(selector["start_ms"]) >= 0
            and int(selector["end_ms"]) > int(selector["start_ms"])
        )

    @staticmethod
    def _audio_binding_subject(record: Mapping[str, Any]) -> str:
        binding = record.get("speaker_binding")
        return (
            str(binding.get("subject_id") or "")
            if isinstance(binding, Mapping)
            else ""
        )

    @classmethod
    def _is_trusted_transport_audio(
        cls, record: Mapping[str, Any], subject_id: str
    ) -> bool:
        binding = record.get("speaker_binding")
        return bool(
            record.get("transport_eligible") is True
            and isinstance(binding, Mapping)
            and binding.get("basis") == "trusted_user_declared"
            and binding.get("matching_method") == "exact_normalized_name"
            and cls._audio_binding_subject(record) == subject_id
            and list(record.get("subject_ids") or []) == [subject_id]
        )

    def _validate_subject_ids(
        self,
        subject_ids: list[Any],
        *,
        label: str,
        required: bool = False,
        exactly_one: bool = False,
    ) -> list[str]:
        values = [str(item) for item in subject_ids]
        if len(values) != len(set(values)):
            raise ToolRejected("duplicate_subject_ids", f"{label} contains duplicate subject IDs")
        if required and not values:
            raise ToolRejected("subject_ids_required", f"{label} requires a subject ID")
        if exactly_one and len(values) != 1:
            raise ToolRejected(
                "exactly_one_subject_required", f"{label} requires exactly one subject ID"
            )
        unknown = sorted(set(values) - self._known_person_subject_ids())
        if unknown:
            raise ToolRejected(
                "unknown_subject_ids",
                f"{label} may use only current person IDs from the material registry",
                details={"subject_ids": unknown},
            )
        return values

    @staticmethod
    def _source_ids_from_binding(value: Any) -> set[str]:
        result: set[str] = set()
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"source_id", "representation_id", "authorization_id"} and item:
                    result.add(str(item))
                else:
                    result.update(StoryMemRuntime._source_ids_from_binding(item))
        elif isinstance(value, list):
            for item in value:
                result.update(StoryMemRuntime._source_ids_from_binding(item))
        return result

    def _validate_evaluation_contract(
        self,
        contract: Mapping[str, Any],
        *,
        artifact: Mapping[str, Any],
        runtime_bindings: Any,
    ) -> None:
        """Fail closed on requirement provenance before generation starts."""

        target = str(artifact.get("target") or "")
        modules = {str(item) for item in contract.get("modules") or []}
        required_modules = {"camera_composition", "style_lighting"} if target == "image" else set()
        if target == "video":
            # Speech/audio is mandatory even for an explicitly silent shot: in
            # that case the agent must verify the absence of speech.  With
            # audio perception disabled, inspection deterministically turns
            # this into capability_blocked supervision instead of silently
            # dropping the requirement.
            required_modules.update(
                {"camera_composition", "motion_temporal", "speech_audio"}
            )
            prompt_state = self._story_prompt_state()
            current_shot_plan = prompt_state.get("current_shot_plan")
            active_tags = (
                current_shot_plan.get("active_material_tags")
                if isinstance(current_shot_plan, Mapping)
                else None
            )
            if not isinstance(active_tags, list) or any(
                not isinstance(tag, str) or not tag for tag in active_tags
            ):
                raise ToolRejected(
                    "invalid_evaluation_context",
                    "video evaluation requires the frozen current-shot active material tags",
                )
            registry_items = self.state.data.get("registry", {}).get("items", {})
            missing_tags: list[str] = []
            unsupported_categories: dict[str, str] = {}
            for tag in active_tags:
                row = registry_items.get(tag) if isinstance(registry_items, Mapping) else None
                if not isinstance(row, Mapping):
                    missing_tags.append(tag)
                    continue
                category = str(row.get("category") or "")
                module = _VIDEO_MATERIAL_CATEGORY_MODULE.get(category)
                if module is None:
                    unsupported_categories[tag] = category
                    continue
                required_modules.add(module)
            if missing_tags or unsupported_categories:
                raise ToolRejected(
                    "invalid_evaluation_context",
                    "video evaluation active materials do not resolve to supported registry categories",
                    details={
                        "missing_tags": sorted(set(missing_tags)),
                        "unsupported_categories": unsupported_categories,
                    },
                )
        missing_modules = sorted(required_modules - modules)
        if missing_modules:
            raise ToolRejected(
                "required_evaluation_modules_missing",
                f"{target} evaluation must include its required semantic modules",
                details={
                    "missing_modules": missing_modules,
                    "required_modules": sorted(required_modules),
                },
            )

        requirements = list(contract.get("requirements") or [])
        requirement_ids = [str(item.get("requirement_id") or "") for item in requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ToolRejected(
                "duplicate_requirement_id",
                "evaluation-contract requirement IDs must be unique",
            )
        reserved_ids = {
            "prompt_content.core_request",
            "technical_integrity.output",
            "provenance_alignment.inputs",
            "frame0_condition_alignment",
            "tail_to_start_continuity",
            "condition_readiness.first_frame",
            "reference_asset_quality.reusable",
            *(f"{module}.module_contract" for module in _SEMANTIC_MODULES),
        }
        collisions = sorted(
            requirement_id
            for requirement_id in requirement_ids
            if requirement_id in reserved_ids
            or requirement_id.startswith("reference_consistency.slot_")
            or requirement_id.startswith("reference_consistency.image_input_")
            or requirement_id.startswith("voice_memory_quality.slot_")
            or requirement_id.startswith("speaker_image_audio_binding.slot_")
        )
        if collisions:
            raise ToolRejected(
                "reserved_requirement_id",
                "evaluation-contract requirement IDs may not collide with runtime checks",
                details={"requirement_ids": collisions},
            )
        raw_request = self.shot["raw_user_request"]
        artifact_id = str(artifact.get("prompt_artifact_id") or "")
        authoring = build_evaluation_contract_authoring(
            shot_request_id=self.shot["shot_request_id"],
            raw_user_request=raw_request,
            prompt_artifact_id=artifact_id,
            receipt_chain=artifact.get("receipt_chain") or [],
        )
        pe_source_ids = {
            artifact_id,
            str(artifact.get("skill_application_receipt_id") or ""),
            *(str(item) for item in artifact.get("receipt_chain") or []),
        }
        pe_source_ids.discard("")
        runtime_source_ids = self._source_ids_from_binding(runtime_bindings)
        evidence_ids = set(str(item) for item in artifact.get("evidence_ids") or [])
        runtime_source_ids.update(evidence_ids)
        for observation_id in evidence_ids:
            observation = self.shot.get("observations", {}).get(observation_id)
            if isinstance(observation, Mapping):
                runtime_source_ids.update(
                    self._source_ids_from_binding(observation.get("payload") or {})
                )

        for requirement in requirements:
            provenance = str(requirement.get("provenance_kind") or "")
            source_id = str(requirement.get("source_id") or "")
            span = requirement.get("source_span")
            if provenance == "user_required":
                required_user = authoring["user_required"]
                if source_id != required_user["source_id"]:
                    raise ToolRejected(
                        "ungrounded_user_requirement",
                        "user_required must bind the active shot_request_id",
                    )
                if span != required_user["source_span"]:
                    raise ToolRejected(
                        "user_requirement_span_mismatch",
                        "user_required must copy the full raw-request Unicode source span from register_prompt",
                    )
            elif provenance in {"pe_required", "pe_added"}:
                if source_id not in pe_source_ids:
                    raise ToolRejected(
                        "ungrounded_pe_requirement",
                        "PE requirements must bind the current prompt artifact or receipt chain",
                    )
                if span is not None:
                    if source_id != artifact_id:
                        raise ToolRejected(
                            "invalid_pe_requirement_span",
                            "only the current prompt artifact supports a text source span",
                        )
                    start = int(span.get("start", -1))
                    end = int(span.get("end", -1))
                    text = str(artifact.get("text") or "")
                    if start < 0 or end <= start or end > len(text):
                        raise ToolRejected(
                            "invalid_pe_requirement_span",
                            "PE requirement source_span is outside the current prompt artifact",
                        )
            elif provenance == "runtime_bound":
                if source_id not in runtime_source_ids:
                    raise ToolRejected(
                        "ungrounded_runtime_requirement",
                        "runtime_bound must bind an authorized observation or media source",
                    )
                if span is not None:
                    raise ToolRejected(
                        "runtime_requirement_span_forbidden",
                        "runtime media/observation bindings do not accept text source spans",
                    )

    def _retry_obligations(self) -> list[tuple[str, dict[str, Any]]]:
        obligations: list[tuple[str, dict[str, Any]]] = []
        for group_id, group in self.shot.get("generation_groups", {}).items():
            obligation = group.get("retry_obligation")
            if isinstance(obligation, Mapping):
                obligations.append((str(group_id), dict(obligation)))
        return obligations

    @staticmethod
    def _provider_output_path(result: dict[str, Any]) -> Path:
        path = result.get("output_path") or result.get("image_path") or result.get("video_path") or result.get("video")
        output = Path(str(path or ""))
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(f"provider returned no decodable output path: {path!r}")
        return output.resolve()

    @staticmethod
    def _validate_provider_receipt(
        result: Mapping[str, Any], *, request_id: str, prompt: str
    ) -> None:
        provenance = result.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError("provider result has no provenance object")
        returned_request_id = result.get("request_id") or provenance.get("request_id")
        if returned_request_id != request_id:
            raise RuntimeError("provider result request_id differs from the durable request")
        if provenance.get("prompt_sha256") != sha256_text(prompt):
            raise RuntimeError("provider result prompt hash differs from the registered artifact")

    @staticmethod
    def _probe_video_output(path: Path) -> dict[str, Any]:
        from .media import ffprobe

        probe = ffprobe(path)
        video_streams = [
            stream
            for stream in probe.get("streams") or []
            if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
        ]
        if not video_streams:
            raise RuntimeError("provider video has no decodable video stream")
        duration_values: list[Any] = []
        if isinstance(probe.get("format"), Mapping):
            duration_values.append(probe["format"].get("duration"))
        duration_values.extend(stream.get("duration") for stream in video_streams)
        positive_duration = False
        for value in duration_values:
            try:
                positive_duration = float(value) > 0
            except (TypeError, ValueError):
                continue
            if positive_duration:
                break
        if not positive_duration:
            raise RuntimeError("provider video has no positive duration")
        return copy.deepcopy(probe)

    def _media_manifest(self, path: Path, representation_id: str) -> dict[str, Any]:
        from .media import media_manifest

        result = media_manifest(path, representation_id=representation_id)
        return _plain(result)

    @staticmethod
    def _provider_call_hash(tool_name: str, arguments: Mapping[str, Any]) -> str:
        return sha256_json({"name": tool_name, "arguments": dict(arguments)})

    def pending_provider_call(self) -> dict[str, Any] | None:
        """Return a validated crash-resume record, if one is pending.

        The stored call is the exact schema-valid assistant action that first
        opened the provider request.  A runner must replay it directly rather
        than asking the teacher to reconstruct it.
        """

        inflight = self.shot.get("inflight")
        if inflight is None:
            return None
        if not isinstance(inflight, Mapping):
            raise RuntimeError("corrupt provider inflight record")
        call = inflight.get("tool_call")
        provenance = inflight.get("assistant_provenance")
        if not isinstance(call, Mapping) or not isinstance(call.get("name"), str):
            raise RuntimeError("provider inflight record has no complete tool call")
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            raise RuntimeError("provider inflight tool call has invalid arguments")
        if call["name"] not in {"generate_image", "generate_shot"}:
            raise RuntimeError("provider inflight record names a non-provider tool")
        expected_hash = self._provider_call_hash(str(call["name"]), arguments)
        if inflight.get("tool_call_sha256") != expected_hash:
            raise RuntimeError("provider inflight tool call hash mismatch")
        self._validate_schema(str(call["name"]), dict(arguments))
        if not isinstance(provenance, Mapping):
            raise RuntimeError("provider inflight record has no assistant provenance")
        return copy.deepcopy(dict(inflight))

    def _matching_inflight(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        inflight = self.pending_provider_call()
        if inflight is None:
            return None
        call = {"name": tool_name, "arguments": copy.deepcopy(dict(arguments))}
        if call != inflight["tool_call"]:
            raise ToolRejected(
                "different_call_inflight",
                "a different provider request is already in flight",
            )
        return inflight

    def _start_or_get_inflight(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        generation_group_id: str,
        retry_axes: list[str],
    ) -> str:
        call_hash = self._provider_call_hash(tool_name, arguments)
        inflight = self._matching_inflight(tool_name, arguments)
        if inflight is not None:
            if inflight.get("generation_group_id") != generation_group_id:
                raise RuntimeError("provider inflight generation group mismatch")
            inflight["resume_attempts"] = int(inflight.get("resume_attempts", 0)) + 1
            inflight["last_resumed_at"] = utc_now()
            self.shot["inflight"] = inflight
            self.state.save()
            return str(inflight["request_id"])
        active_call = self._active_tool_call
        if active_call != {"name": tool_name, "arguments": arguments}:
            raise RuntimeError("provider request was not opened by the active validated tool call")
        namespace = str(self.state.data.get("provider_request_namespace") or "")
        if not namespace:
            raise RuntimeError(
                "state has no persistent provider request namespace; refusing "
                "a non-globally-unique provider call"
            )
        local_request_id = self.state.next_id("providerreq")
        request_id = (
            f"{namespace}-shot{int(self.shot['shot_index']):04d}-{local_request_id}"
        )
        self.shot["inflight"] = {
            "tool_call": copy.deepcopy(active_call),
            "tool_call_sha256": call_hash,
            "assistant_provenance": copy.deepcopy(self._active_assistant_provenance or {}),
            "request_id": request_id,
            "idempotency_key": request_id,
            "generation_group_id": generation_group_id,
            "retry_axes": list(retry_axes),
            "resume_attempts": 0,
            "started_at": utc_now(),
        }
        self.state.save()
        return request_id

    def _finish_inflight(self) -> None:
        self.shot["inflight"] = None

    def tool_generate_image(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._assert_request(arguments)
        artifact_id = str(arguments["prompt_artifact_id"])
        artifact = self._artifact(artifact_id, "image")
        passthrough = artifact.get("prompt_rewrite_mode") == "passthrough_disabled"
        if passthrough:
            self._assert_passthrough_artifact_integrity(artifact, target="image")
        purpose = str(arguments["purpose"])
        if artifact["purpose"] != purpose:
            raise ToolRejected("purpose_mismatch", "image artifact purpose differs from generation purpose")
        references = list(arguments.get("input_references") or [])
        subject_ids = self._validate_subject_ids(
            list(arguments.get("subject_ids") or []), label="generate_image.subject_ids"
        )
        expected_scope = {
            "subject_ids": subject_ids,
            "asset_role": arguments.get("asset_role"),
        }
        if artifact.get("generation_scope") != expected_scope:
            raise ToolRejected(
                "generation_scope_mismatch",
                "image generation arguments differ from the prompt artifact generation_scope",
            )
        has_references = bool(references)
        if (
            artifact["prompt_context"]["has_reference_inputs"] != has_references
            and not (
                passthrough and arguments.get("generation_group_id") is not None
            )
        ):
            raise ToolRejected("reference_context_mismatch", "prompt artifact reference mode differs from input_references")
        group_id_arg = arguments.get("generation_group_id")
        retry = arguments.get("retry")
        inflight = self._matching_inflight("generate_image", arguments)
        if inflight is not None:
            group_id = str(inflight["generation_group_id"])
            group = self.shot["generation_groups"].get(group_id)
            if not group or group.get("target") != "image":
                raise RuntimeError("persisted image inflight group is missing or corrupt")
            axes = list(inflight.get("retry_axes") or [])
            if group_id_arg is not None:
                checked_group, checked_axes = self._generation_retry(
                    target="image", group_id=str(group_id_arg), retry=retry
                )
                if checked_group is not group or checked_axes != axes:
                    raise RuntimeError("persisted image retry state changed")
        else:
            group, axes = self._generation_retry(
                target="image", group_id=group_id_arg, retry=retry
            )
        self._assert_prompt_artifact_ready_for_generation(
            target="image",
            artifact_id=artifact_id,
            generation_group_id=(
                str(group_id_arg) if group_id_arg is not None else None
            ),
            retry=retry,
            retry_axes=axes,
            input_references=references,
        )
        if group is None:
            self._assert_new_image_group_available(purpose)
            if artifact.get("operation") != "compose":
                raise ToolRejected(
                    "first_image_requires_compose",
                    "a new image group requires an unconsumed compose artifact",
                )
            if artifact.get("consumed"):
                raise ToolRejected(
                    "prompt_artifact_already_consumed",
                    "a consumed prompt artifact cannot start another generation group",
                )
            group_id = self.state.next_id("imagegroup")
            group = {
                "generation_group_id": group_id,
                "target": "image",
                "purpose": purpose,
                "subject_ids": subject_ids,
                "asset_role": arguments.get("asset_role"),
                "candidate_ids": [],
                "selection_id": None,
                "max_attempts": int(self.shot["budgets"].get("image_attempts", self.config.image_attempts)),
                "evaluation_contract": copy.deepcopy(arguments["evaluation_contract"]),
            }
            self.shot["generation_groups"][group_id] = group
        elif inflight is None:
            group_id = str(group_id_arg)
            if group["purpose"] != purpose:
                raise ToolRejected("group_purpose_mismatch", "image group purpose is immutable")
        elif group["purpose"] != purpose:
            raise RuntimeError("persisted image inflight purpose changed")
        bound_group_id = artifact.get("generation_group_id")
        if artifact.get("consumed") and bound_group_id != group_id:
            raise ToolRejected(
                "prompt_artifact_group_mismatch",
                "a consumed prompt artifact is bound to its original generation group",
            )
        if (
            group.get("evaluation_contract") != arguments["evaluation_contract"]
            or group.get("subject_ids", []) != subject_ids
            or group.get("asset_role") != arguments.get("asset_role")
        ):
            raise ToolRejected(
                "image_group_contract_changed",
                "image group evaluation_contract, subject_ids, and asset_role are immutable",
            )
        self._validate_retry_changes(
            target="image",
            artifact=artifact,
            retry=retry,
            axes=axes,
            input_references=references,
            seed=int(arguments["seed"]),
        )
        reference_paths: list[Path] = []
        required_reference_observations: set[str] = set()
        for source in references:
            resolved_reference = self._resolve_source(source)
            if not str(
                resolved_reference["representation"].get("media_type") or ""
            ).startswith("image"):
                raise ToolRejected(
                    "image_reference_required",
                    "every image generation reference must use an image representation",
                )
            reference_paths.append(resolved_reference["path"])
            source_id = str(source["source_id"])
            if source.get("kind") == "generated_image":
                candidate = self.shot["candidates"].get(source_id)
                inspection = (
                    self.shot["inspections"].get(candidate.get("inspection_id"))
                    if isinstance(candidate, Mapping)
                    else None
                )
                observation_id = (
                    str(inspection.get("observation_id") or "")
                    if isinstance(inspection, Mapping)
                    else ""
                )
            else:
                authorization = next(
                    (
                        row
                        for row in self.shot.get("view_authorizations", {}).get(source_id, [])
                        if isinstance(row, Mapping)
                        and row.get("authorization_id") == source.get("authorization_id")
                        and row.get("representation_id") == source.get("representation_id")
                    ),
                    None,
                )
                observation_id = (
                    str(authorization.get("observation_id") or "")
                    if isinstance(authorization, Mapping)
                    else ""
                )
            if not observation_id:
                raise ToolRejected(
                    "reference_evidence_missing",
                    "an image reference has no exact visible observation",
                )
            required_reference_observations.add(observation_id)
        missing_reference_evidence = sorted(
            required_reference_observations - set(artifact.get("evidence_ids") or [])
        )
        if missing_reference_evidence and not (
            passthrough and arguments.get("generation_group_id") is not None
        ):
            raise ToolRejected(
                "required_reference_evidence_missing",
                "the prompt artifact must bind every viewed image reference observation",
                details={"missing_observation_ids": missing_reference_evidence},
            )
        self._validate_evaluation_contract(
            arguments["evaluation_contract"],
            artifact=artifact,
            runtime_bindings={"input_references": references},
        )
        request_id = self._start_or_get_inflight(
            "generate_image",
            arguments,
            generation_group_id=group_id,
            retry_axes=axes,
        )
        output_dir = self.config.output_dir / "provider" / request_id
        from .providers import ImageGenerationRequest

        provider_result = self.image_provider.generate(
            ImageGenerationRequest(
                request_id=request_id,
                prompt=artifact["text"],
                reference_paths=reference_paths,
                seed=int(arguments["seed"]),
                width=self.config.width,
                height=self.config.height,
                output_dir=output_dir,
            )
        )
        result = _plain(provider_result)
        self._validate_provider_receipt(
            result, request_id=request_id, prompt=str(artifact["text"])
        )
        path = self._provider_output_path(result)
        candidate_id = self.state.next_id("cand_image")
        representation = self._media_manifest(path, self.state.next_id("repr_image"))
        candidate = {
            "candidate_id": candidate_id,
            "target": "image",
            "group_id": group_id,
            "purpose": purpose,
            "prompt_artifact_id": artifact_id,
            "representation": representation,
            "provider": result,
            "seed": int(arguments["seed"]),
            "input_references": copy.deepcopy(references),
            "subject_ids": subject_ids,
            "asset_role": arguments.get("asset_role"),
            "inspection_id": None,
            "assessment_id": None,
            "created_at": utc_now(),
        }
        self.shot["candidates"][candidate_id] = candidate
        group["candidate_ids"].append(candidate_id)
        artifact["consumed"] = True
        artifact["generation_group_id"] = group_id
        artifact["last_candidate_id"] = candidate_id
        group["retry_obligation"] = None
        self.shot["pending_prompt_artifact_id"] = None
        self.shot["pending_candidate_id"] = candidate_id
        self._finish_inflight()
        remaining = self._remaining_generation_attempts("image")
        return {
            "ok": True,
            "candidate_id": candidate_id,
            "generation_group_id": group_id,
            "purpose": purpose,
            "representation": {key: value for key, value in representation.items() if key != "path"},
            "provider_request": result.get("request_payload"),
            "provider_provenance": result.get("provenance"),
            "prompt_artifact_id": artifact_id,
            "submitted_prompt_sha256": sha256_text(artifact["text"]),
            "remaining_attempts": remaining,
            "inspection_required": True,
        }

    def _validate_condition(self, condition: Any) -> dict[str, Any] | None:
        current_index = int(self.shot["shot_index"])
        current_plan = next(
            (
                item
                for item in self.state.data.get("story_plan", {}).get("shots", [])
                if int(item.get("shot_index", -1)) == current_index
            ),
            None,
        )
        direct_continuation = (
            isinstance(current_plan, Mapping)
            and current_plan.get("direct_continuation") is True
        )
        if current_index == 1 and direct_continuation:
            raise ToolRejected(
                "invalid_first_shot_continuation",
                "the first shot cannot be a direct continuation",
            )
        if condition is None:
            if direct_continuation:
                raise ToolRejected(
                    "direct_continuation_requires_previous_tail",
                    "direct_continuation=true requires the exact viewed previous tail "
                    "with relation=temporal_continuation",
                )
            return None
        if not isinstance(condition, Mapping) or set(condition) != {"relation", "image_source"}:
            raise ToolRejected("invalid_condition", "condition must contain exactly relation and image_source")
        relation = condition.get("relation")
        source = condition.get("image_source")
        kind = source.get("kind") if isinstance(source, Mapping) else None
        if direct_continuation and (
            kind != "previous_tail" or relation != "temporal_continuation"
        ):
            raise ToolRejected(
                "direct_continuation_requires_previous_tail",
                "direct_continuation=true requires the exact viewed previous tail "
                "with relation=temporal_continuation",
            )
        if kind == "previous_tail":
            if not isinstance(current_plan, Mapping) or current_plan.get(
                "direct_continuation"
            ) is not True:
                raise ToolRejected(
                    "previous_tail_not_direct_continuation",
                    "previous_tail may condition only a shot explicitly marked direct_continuation=true",
                )
        resolved = self._resolve_source(source)
        if not str(resolved["representation"].get("media_type") or "").startswith(
            "image"
        ):
            raise ToolRejected(
                "condition_image_required",
                "a video condition must use an image representation",
            )
        if kind == "previous_tail" and relation != "temporal_continuation":
            raise ToolRejected("condition_relation_mismatch", "previous_tail requires temporal_continuation")
        if kind != "previous_tail" and relation != "frame0_match":
            raise ToolRejected("condition_relation_mismatch", "non-tail condition requires frame0_match")
        if kind in {"memory", "uploaded_asset"} and not bool(resolved["record"].get("condition_eligible", False)):
            raise ToolRejected("condition_not_authorized", f"{kind} source is not eligible as a complete frame-zero condition")
        if kind == "generated_image":
            selection = self.shot["selections"].get(source.get("authorization_id"))
            if not selection or selection.get("purpose") != "first_frame":
                raise ToolRejected("condition_primary_use_mismatch", "generated condition must come from a first_frame selection")
        return {"binding": copy.deepcopy(dict(condition)), **resolved}

    def _validate_slot(self, slot: Mapping[str, Any]) -> dict[str, Any]:
        image = self._resolve_source(slot["image_source"])
        if not str(image["representation"].get("media_type") or "").startswith("image"):
            raise ToolRejected("slot_image_required", "every memory slot requires an image representation")
        image_source = slot["image_source"]
        if image_source.get("kind") == "generated_image":
            selection = self.shot["selections"].get(image_source.get("authorization_id"))
            if not selection or selection.get("purpose") != "reference_asset":
                raise ToolRejected("slot_primary_use_mismatch", "generated slot image must come from a reference_asset selection")
        audio_source = slot.get("audio_source")
        audio = None
        roles = [str(item) for item in slot.get("roles") or []]
        character_role = bool(set(roles) & _PERSON_MEMORY_ROLES)
        entity_role = bool(set(roles) & _ENTITY_MEMORY_ROLES)
        if character_role and entity_role:
            raise ToolRejected(
                "mixed_person_entity_roles",
                "one memory slot cannot mix human character and non-human entity roles",
            )
        if audio_source is not None:
            if self.config.audio_transport != "enabled":
                raise ToolRejected(
                    "audio_transport_disabled",
                    "audio_source requires audio_transport=enabled",
                )
            audio = self._resolve_source(audio_source)
            if not str(audio["representation"].get("media_type") or "").startswith(
                "audio"
            ):
                raise ToolRejected(
                    "slot_audio_required",
                    "audio_source must use an audio representation",
                )
        subject_ids = self._validate_subject_ids(
            list(slot.get("subject_ids") or []),
            label="memory_slots.subject_ids",
            required=character_role,
            exactly_one=audio_source is not None,
        )
        if entity_role and subject_ids:
            raise ToolRejected(
                "entity_subject_ids_forbidden",
                "entity_identity/entity_appearance slots require subject_ids=[]",
            )
        entity_tags = self._entity_tags_for_record(image["record"])
        if entity_role and not entity_tags:
            raise ToolRejected(
                "entity_tag_evidence_required",
                "an entity memory slot requires an image bound to a current ENTITY_N tag",
            )
        image_subjects = set(image["record"].get("subject_ids") or [])
        if (
            character_role and image_subjects != set(subject_ids)
        ) or (image_subjects and image_subjects != set(subject_ids)):
            raise ToolRejected(
                "slot_image_subject_mismatch",
                "slot subject_ids must exactly match the authorized image source subjects",
            )
        if audio_source is not None:
            assert audio is not None
            if len(subject_ids) != 1:
                raise ToolRejected(
                    "audio_requires_one_speaker",
                    "a slot with audio must bind exactly one explicit speaking subject",
                )
            audio_subjects = set(audio["record"].get("subject_ids") or [])
            if image_subjects != set(subject_ids) or audio_subjects != set(subject_ids):
                raise ToolRejected("audio_subject_mismatch", "slot image/audio subject_ids must match")
            if "voice" not in roles:
                raise ToolRejected("audio_voice_role_required", "audio source requires the voice role")
            subject_id = subject_ids[0]
            record = audio["record"]
            trusted_transport = self._is_trusted_transport_audio(record, subject_id)
            heard_and_authorized = bool(
                record.get("transport_eligible") is True
                and record.get("verification_basis") == "agent_heard"
                and record.get("quality_authorization_id")
                and self._audio_binding_subject(record) == subject_id
            )
            if audio_source.get("kind") == "uploaded_asset":
                if not trusted_transport:
                    raise ToolRejected(
                        "audio_transport_not_authorized",
                        "uploaded audio requires an exact trusted user-declared speaker binding",
                    )
            elif not (trusted_transport or heard_and_authorized):
                raise ToolRejected(
                    "audio_transport_not_authorized",
                    "stored audio lacks a trusted transport or independently heard quality authorization",
                )
            if audio_source.get("kind") == "memory" and trusted_transport and not record.get(
                "transport_authorization_id"
            ):
                raise ToolRejected(
                    "audio_transport_not_authorized",
                    "stored trusted audio lacks its transport authorization",
                )
            if not self._audio_time_range_is_exact(audio["representation"]):
                raise ToolRejected(
                    "audio_time_range_required",
                    "audio memory slots require an exact nonzero time-range representation",
                )
        elif "voice" in roles:
            raise ToolRejected("voice_role_without_audio", "voice role requires an audio source")
        return {
            "image_path": str(image["path"]),
            "audio_path": str(audio["path"]) if audio else None,
            "metadata": {
                "roles": roles,
                "subject_ids": subject_ids,
                "reason": str(slot.get("reason") or ""),
                **(
                    {
                        "audio_transport": "trusted_metadata_bound",
                        "audio_source_id": str(audio_source.get("source_id") or ""),
                        "audio_representation_id": str(
                            audio_source.get("representation_id") or ""
                        ),
                        "speaker_name": (
                            audio["record"].get("speaker_name")
                            or (audio["record"].get("speaker_binding") or {}).get(
                                "declared_name"
                            )
                        ),
                    }
                    if audio is not None
                    else {}
                ),
            },
        }

    def _assert_selected_images_consumed(
        self, condition: Mapping[str, Any] | None, slots: list[Mapping[str, Any]]
    ) -> list[str]:
        authorization_ids: set[str] = set()
        if condition:
            authorization_ids.add(str(condition["image_source"].get("authorization_id") or ""))
        for slot in slots:
            authorization_ids.add(str(slot["image_source"].get("authorization_id") or ""))
        unconsumed = [
            selection_id
            for selection_id, selection in self.shot["selections"].items()
            if selection.get("target") == "image"
            and selection_id not in self.shot["consumed_image_selection_ids"]
        ]
        missing = [item for item in unconsumed if item not in authorization_ids]
        if missing:
            raise ToolRejected(
                "selected_image_not_consumed",
                "all selected image groups must be consumed by their declared primary use",
                details={"missing_selection_ids": missing},
            )
        for selection_id in unconsumed:
            selection = self.shot["selections"][selection_id]
            if selection["purpose"] == "first_frame":
                actual = condition and condition["image_source"].get("authorization_id") == selection_id
            else:
                actual = any(slot["image_source"].get("authorization_id") == selection_id for slot in slots)
            if not actual:
                raise ToolRejected("image_primary_use_mismatch", f"image selection {selection_id} used in wrong role")
        return unconsumed

    def tool_generate_shot(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._assert_request(arguments)
        artifact_id = str(arguments["execution_prompt_artifact_id"])
        artifact = self._artifact(artifact_id, "video")
        passthrough = artifact.get("prompt_rewrite_mode") == "passthrough_disabled"
        if passthrough:
            self._assert_passthrough_artifact_integrity(artifact, target="video")
        condition = copy.deepcopy(arguments.get("condition"))
        if artifact["prompt_context"]["condition"] != condition and not (
            passthrough and arguments.get("generation_group_id") is not None
        ):
            raise ToolRejected("condition_artifact_mismatch", "generation condition differs from prompt artifact")
        resolved_condition = self._validate_condition(condition)
        slots = list(arguments.get("memory_slots") or [])
        if len(slots) > self.config.max_memory_slots:
            raise ToolRejected("too_many_memory_slots", "Echo accepts at most seven memory slots")
        resolved_slots = [self._validate_slot(slot) for slot in slots]
        required_slot_observations: set[str] = set()
        for slot in slots:
            for source in [slot["image_source"], slot.get("audio_source")]:
                if source is None:
                    continue
                source_id = str(source["source_id"])
                if source.get("kind") == "generated_image":
                    source_candidate = self.shot["candidates"].get(source_id)
                    source_inspection = (
                        self.shot["inspections"].get(source_candidate.get("inspection_id"))
                        if isinstance(source_candidate, Mapping)
                        else None
                    )
                    observation_id = (
                        str(source_inspection.get("observation_id") or "")
                        if isinstance(source_inspection, Mapping)
                        else ""
                    )
                else:
                    authorization = next(
                        (
                            row
                            for row in self.shot.get("view_authorizations", {}).get(source_id, [])
                            if isinstance(row, Mapping)
                            and row.get("authorization_id") == source.get("authorization_id")
                            and row.get("representation_id") == source.get("representation_id")
                        ),
                        None,
                    )
                    observation_id = (
                        str(authorization.get("observation_id") or "")
                        if isinstance(authorization, Mapping)
                        else ""
                    )
                if not observation_id:
                    raise ToolRejected(
                        "slot_evidence_missing",
                        "a memory-slot source has no exact visible observation",
                    )
                required_slot_observations.add(observation_id)
        missing_slot_evidence = sorted(
            required_slot_observations - set(artifact.get("evidence_ids") or [])
        )
        if missing_slot_evidence and not (
            passthrough and arguments.get("generation_group_id") is not None
        ):
            raise ToolRejected(
                "required_slot_evidence_missing",
                "the video prompt artifact must bind every memory-slot observation",
                details={"missing_observation_ids": missing_slot_evidence},
            )
        bound_images: list[tuple[str, str, str]] = []
        if condition is not None:
            source = condition["image_source"]
            bound_images.append(
                (
                    str(source["kind"]),
                    str(source["source_id"]),
                    str(source["representation_id"]),
                )
            )
        bound_images.extend(
            (
                str(slot["image_source"]["kind"]),
                str(slot["image_source"]["source_id"]),
                str(slot["image_source"]["representation_id"]),
            )
            for slot in slots
        )
        if len(bound_images) != len(set(bound_images)):
            raise ToolRejected(
                "duplicate_image_binding",
                "condition and memory slots must use distinct image source representations",
            )
        consumed_selections = self._assert_selected_images_consumed(condition, slots)
        declared_consumed = list(arguments.get("consumed_image_selection_ids") or [])
        if declared_consumed != consumed_selections:
            raise ToolRejected(
                "consumed_selection_manifest_mismatch",
                "consumed_image_selection_ids must exactly list newly consumed selections in selection order",
                details={"expected": consumed_selections},
            )
        group_id_arg = arguments.get("generation_group_id")
        retry = arguments.get("retry")
        inflight = self._matching_inflight("generate_shot", arguments)
        if inflight is not None:
            group_id = str(inflight["generation_group_id"])
            group = self.shot["generation_groups"].get(group_id)
            if not group or group.get("target") != "video":
                raise RuntimeError("persisted video inflight group is missing or corrupt")
            axes = list(inflight.get("retry_axes") or [])
            if group_id_arg is not None:
                checked_group, checked_axes = self._generation_retry(
                    target="video", group_id=str(group_id_arg), retry=retry
                )
                if checked_group is not group or checked_axes != axes:
                    raise RuntimeError("persisted video retry state changed")
        else:
            group, axes = self._generation_retry(
                target="video", group_id=group_id_arg, retry=retry
            )
        self._assert_prompt_artifact_ready_for_generation(
            target="video",
            artifact_id=artifact_id,
            generation_group_id=(
                str(group_id_arg) if group_id_arg is not None else None
            ),
            retry=retry,
            retry_axes=axes,
        )
        if group is None:
            if artifact["operation"] != "compose":
                raise ToolRejected("first_video_requires_compose", "first video call requires a compose artifact")
            if artifact.get("consumed"):
                raise ToolRejected(
                    "prompt_artifact_already_consumed",
                    "a consumed prompt artifact cannot start another generation group",
                )
            group_id = self.state.next_id("videogroup")
            group = {
                "generation_group_id": group_id,
                "target": "video",
                "purpose": "shot",
                "candidate_ids": [],
                "selection_id": None,
                "max_attempts": int(self.shot["budgets"].get("video_attempts", self.config.video_attempts)),
                "evaluation_contract": copy.deepcopy(arguments["evaluation_contract"]),
            }
            self.shot["generation_groups"][group_id] = group
        elif inflight is None:
            group_id = str(group_id_arg)
        bound_group_id = artifact.get("generation_group_id")
        if artifact.get("consumed") and bound_group_id != group_id:
            raise ToolRejected(
                "prompt_artifact_group_mismatch",
                "a consumed prompt artifact is bound to its original generation group",
            )
        if group.get("evaluation_contract") != arguments["evaluation_contract"]:
            raise ToolRejected(
                "video_group_contract_changed",
                "video group evaluation_contract is immutable across retries",
            )
        self._validate_retry_changes(
            target="video",
            artifact=artifact,
            retry=retry,
            axes=axes,
            condition=condition,
            memory_slots=slots,
            seed=int(arguments["seed"]),
        )
        self._validate_evaluation_contract(
            arguments["evaluation_contract"],
            artifact=artifact,
            runtime_bindings={"condition": condition, "memory_slots": slots},
        )
        if artifact.get("text_sha256") != sha256_text(str(artifact.get("text") or "")):
            raise ToolRejected(
                "prompt_artifact_hash_mismatch",
                "video prompt artifact text no longer matches its registered hash",
            )
        artifact_mode = artifact.get("prompt_rewrite_mode", "rewrite_enabled")
        if artifact_mode == "passthrough_disabled":
            self._assert_passthrough_artifact_integrity(artifact, target="video")
        else:
            receipt = self.shot["pe_receipts"].get(
                artifact.get("skill_application_receipt_id")
            )
            if (
                not isinstance(receipt, Mapping)
                or receipt.get("registered_prompt_artifact_id") != artifact_id
            ):
                raise ToolRejected(
                    "prompt_receipt_missing",
                    "video generation requires the exact live PE receipt for its prompt artifact",
                )
            try:
                current_validation = self.prompt_validator.validate(
                    artifact["text"],
                    target="video",
                    contract=receipt.get("bundle", {}).get("validation_contract") or {},
                    raw_user_request=self.shot["raw_user_request"],
                    allowed_person_ids=self._active_person_subject_ids(),
                    prompt_context=artifact.get("prompt_context"),
                    expected_caption_language=self._story_prompt_state().get(
                        "caption_language"
                    ),
                )
            except PromptValidationError as exc:
                raise ToolRejected("prompt_preflight_failed", str(exc)) from exc
            if current_validation != artifact.get("validation"):
                raise ToolRejected(
                    "prompt_validation_profile_drift",
                    "video prompt preflight differs from its registration receipt",
                )
        request_id = self._start_or_get_inflight(
            "generate_shot",
            arguments,
            generation_group_id=group_id,
            retry_axes=axes,
        )
        output_dir = self.config.output_dir / "provider" / request_id
        from .providers import VideoGenerationRequest

        provider_result = self.video_provider.generate(
            VideoGenerationRequest(
                request_id=request_id,
                work_id=str(self.state.data["story_id"]),
                shot_id=f"shot_{int(self.shot['shot_index']):03d}",
                prompt=artifact["text"],
                condition_path=str(resolved_condition["path"]) if resolved_condition else None,
                memory_slots=resolved_slots,
                seed=int(arguments["seed"]),
                width=self.config.width,
                height=self.config.height,
                num_frames=self.config.num_frames,
                output_dir=output_dir,
            )
        )
        result = _plain(provider_result)
        self._validate_provider_receipt(
            result, request_id=request_id, prompt=str(artifact["text"])
        )
        path = self._provider_output_path(result)
        probe = self._probe_video_output(path)
        candidate_id = self.state.next_id("cand_video")
        representation = self._media_manifest(path, self.state.next_id("repr_video"))
        representation["probe"] = probe
        candidate = {
            "candidate_id": candidate_id,
            "target": "video",
            "group_id": group_id,
            "purpose": "shot",
            "prompt_artifact_id": artifact_id,
            "representation": representation,
            "provider": result,
            "condition": condition,
            "memory_slots": copy.deepcopy(slots),
            "seed": int(arguments["seed"]),
            "inspection_id": None,
            "assessment_id": None,
            "created_at": utc_now(),
        }
        self.shot["candidates"][candidate_id] = candidate
        group["candidate_ids"].append(candidate_id)
        artifact["consumed"] = True
        artifact["generation_group_id"] = group_id
        artifact["last_candidate_id"] = candidate_id
        group["retry_obligation"] = None
        self.shot["pending_prompt_artifact_id"] = None
        self.shot["pending_candidate_id"] = candidate_id
        self.shot["consumed_image_selection_ids"].extend(consumed_selections)
        self.shot["consumed_image_selection_ids"] = list(dict.fromkeys(self.shot["consumed_image_selection_ids"]))
        self._finish_inflight()
        remaining = self._remaining_generation_attempts("video")
        return {
            "ok": True,
            "candidate_id": candidate_id,
            "generation_group_id": group_id,
            "representation": {key: value for key, value in representation.items() if key != "path"},
            "provider_request": result.get("request_payload"),
            "provider_provenance": result.get("provenance"),
            "prompt_artifact_id": artifact_id,
            "submitted_prompt_sha256": sha256_text(artifact["text"]),
            "remaining_attempts": remaining,
            "inspection_required": True,
        }

    def _checklist(self, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        group = self.shot["generation_groups"][candidate["group_id"]]
        target = candidate["target"]
        if target == "video":
            core_requirement = self.shot["raw_user_request"]
        elif candidate.get("purpose") == "first_frame":
            core_requirement = (
                "Judge only the frame-zero-visible subjects, state, scene, composition, and "
                "continuity implied by this request; do not require the still image to complete "
                f"future motion or dialogue: {self.shot['raw_user_request']}"
            )
        else:
            core_requirement = (
                "Judge the reusable reference asset for the visual role needed by this request, "
                f"not completion of its full action: {self.shot['raw_user_request']}"
            )
        checks: list[dict[str, Any]] = [
            {
                "check_id": "prompt_content.core_request",
                "module": "prompt_content",
                "scope": "final_video" if target == "video" else candidate["purpose"],
                "requirement": core_requirement,
                "severity": "critical",
                "allowed_evidence_kinds": ["target_video", "target_frame"] if target == "video" else ["target_image"],
                "not_applicable_allowed": False,
                "provenance_kind": "user_required",
            },
            {
                "check_id": "technical_integrity.output",
                "module": "technical_integrity",
                "scope": "final_video" if target == "video" else candidate["purpose"],
                "requirement": "The generated media is decodable and has no severe corruption or generation artifacts.",
                "severity": "critical",
                "allowed_evidence_kinds": ["target_video", "target_frame", "target_image"],
                "not_applicable_allowed": False,
                "provenance_kind": "runtime_bound",
            },
            {
                "check_id": "provenance_alignment.inputs",
                "module": "provenance_alignment",
                "scope": "provider_request",
                "requirement": "Displayed media and the provider request use the recorded prompt and exact bound representations.",
                "severity": "critical",
                "allowed_evidence_kinds": ["provider_manifest"],
                "not_applicable_allowed": False,
                "provenance_kind": "runtime_bound",
            },
        ]
        if target == "image":
            if candidate.get("purpose") == "first_frame":
                checks.append(
                    {
                        "check_id": "condition_readiness.first_frame",
                        "module": "condition_readiness",
                        "scope": "first_frame",
                        "requirement": (
                            "The still is a coherent complete frame-zero condition with no "
                            "contradiction that would prevent the requested motion from beginning."
                        ),
                        "severity": "critical",
                        "allowed_evidence_kinds": ["target_image", "input_reference"],
                        "not_applicable_allowed": False,
                        "provenance_kind": "runtime_bound",
                    }
                )
            else:
                checks.append(
                    {
                        "check_id": "reference_asset_quality.reusable",
                        "module": "reference_asset_quality",
                        "scope": "reference_asset",
                        "requirement": (
                            "The image is clear, internally consistent, and reusable for its "
                            "declared reference role without transient or contradictory content."
                        ),
                        "severity": "critical",
                        "allowed_evidence_kinds": ["target_image", "input_reference"],
                        "not_applicable_allowed": False,
                        "provenance_kind": "runtime_bound",
                    }
                )
        contract = group.get("evaluation_contract") or {}
        module_requirements = {
            "identity_wardrobe": "All depicted subjects preserve the intended identity and wardrobe evidence.",
            "scene_environment": "The location, environment, and scene continuity match the intended shot.",
            "object_state": "Required objects, attributes, and persistent object state are correct.",
            "interaction_spatial": "Subject interactions and spatial relationships are physically coherent.",
            "camera_composition": "Framing, camera placement, and composition satisfy the intended shot design.",
            "motion_temporal": "Motion, event order, timing, and temporal continuity satisfy the intended shot.",
            "style_lighting": "Visual style and lighting are coherent with the intended shot and references.",
            "speech_audio": "Required speech, speaker behavior, and audio are correct where observable.",
        }
        for module in contract.get("modules") or []:
            module = str(module)
            requirement = module_requirements.get(module)
            if requirement:
                checks.append(
                    {
                        "check_id": f"{module}.module_contract",
                        "module": module,
                        "scope": "final_video" if target == "video" else candidate["purpose"],
                        "requirement": requirement,
                        "severity": "critical",
                        "allowed_evidence_kinds": (
                            evaluation_check_evidence_kinds(
                                target=target, module=module
                            )
                        ),
                        "not_applicable_allowed": False,
                        "provenance_kind": "pe_required",
                    }
                )
        for index, requirement in enumerate(contract.get("requirements") or [], start=1):
            if not isinstance(requirement, Mapping):
                continue
            module = str(requirement.get("module") or "prompt_content")
            checks.append(
                {
                    "check_id": str(requirement.get("requirement_id") or f"{module}.declared_{index:03d}"),
                    "module": module,
                    "scope": str(requirement.get("scope") or ("final_video" if target == "video" else candidate["purpose"])),
                    "requirement": str(requirement.get("text") or "").strip(),
                    "severity": str(requirement.get("severity") or "critical"),
                    "allowed_evidence_kinds": evaluation_check_evidence_kinds(
                        target=target, module=module
                    ),
                    "not_applicable_allowed": bool(requirement.get("not_applicable_allowed", False)),
                    "provenance_kind": str(requirement.get("provenance_kind") or "pe_added"),
                }
            )
        if target == "video" and candidate.get("condition"):
            relation = candidate["condition"]["relation"]
            checks.append(
                {
                    "check_id": "tail_to_start_continuity" if relation == "temporal_continuation" else "frame0_condition_alignment",
                    "module": "motion_temporal" if relation == "temporal_continuation" else "camera_composition",
                    "scope": "final_video",
                    "requirement": "The first video frame matches the bound condition and then follows the declared relation.",
                    "severity": "critical",
                    "allowed_evidence_kinds": ["target_video", "target_frame", "condition_image"],
                    "not_applicable_allowed": False,
                    "provenance_kind": "runtime_bound",
                }
            )
        for index, slot in enumerate(candidate.get("memory_slots") or [], start=1):
            checks.append(
                {
                    "check_id": f"reference_consistency.slot_{index}",
                    "module": "identity_wardrobe",
                    "scope": "final_video",
                    "requirement": str(slot.get("reason") or "The generated subject matches the viewed reference."),
                    "severity": "critical",
                    "allowed_evidence_kinds": ["target_video", "target_frame", "slot_image"],
                    "not_applicable_allowed": False,
                    "provenance_kind": "runtime_bound",
                }
            )
            if slot.get("audio_source") is not None:
                checks.append(
                    {
                        "check_id": f"voice_memory_quality.slot_{index}",
                        "module": "speech_audio",
                        "scope": "final_video",
                        "requirement": (
                            "The exact bound voice clip is a clean, reusable, single-speaker "
                            "voice reference and the generated audio remains compatible with it."
                        ),
                        "severity": "critical",
                        "allowed_evidence_kinds": ["slot_audio", "target_audio"],
                        "not_applicable_allowed": False,
                        "provenance_kind": "runtime_bound",
                    }
                )
                checks.append(
                    {
                        "check_id": f"speaker_image_audio_binding.slot_{index}",
                        "module": "speech_audio",
                        "scope": "final_video",
                        "requirement": (
                            "The generated speaking subject matches the exact bound image and "
                            "voice clip, with intelligible single-speaker voice where observable."
                        ),
                        "severity": "critical",
                        "allowed_evidence_kinds": [
                            "target_video",
                            "target_audio",
                            "target_frame",
                            "slot_image",
                            "slot_audio",
                        ],
                        "not_applicable_allowed": False,
                        "provenance_kind": "runtime_bound",
                    }
                )
        if target == "image":
            for index, _ in enumerate(candidate.get("input_references") or [], start=1):
                checks.append(
                    {
                        "check_id": f"reference_consistency.image_input_{index}",
                        "module": "identity_wardrobe",
                        "scope": candidate["purpose"],
                        "requirement": (
                            "The generated image preserves the relevant visible identity, "
                            "appearance, object, scene, or style evidence from this exact input reference."
                        ),
                        "severity": "critical",
                        "allowed_evidence_kinds": ["target_image", "input_reference"],
                        "not_applicable_allowed": False,
                        "provenance_kind": "runtime_bound",
                    }
                )
        unique: dict[str, dict[str, Any]] = {}
        for check in checks:
            if not check["requirement"]:
                continue
            if check["check_id"] in unique:
                raise ToolRejected(
                    "duplicate_check_id",
                    f"evaluation checklist contains duplicate check ID: {check['check_id']}",
                )
            unique[check["check_id"]] = check
        return list(unique.values())

    def tool_inspect_candidate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(arguments["candidate_id"])
        candidate = self.shot["candidates"].get(candidate_id)
        if not candidate:
            raise ToolRejected("unknown_candidate", f"unknown candidate: {candidate_id}")
        if candidate.get("inspection_id"):
            raise ToolRejected("candidate_already_inspected", "candidate inspection is immutable")
        representation = candidate["representation"]
        path = Path(representation["path"])
        media: list[dict[str, Any]] = []
        evidence: dict[str, dict[str, Any]] = {}
        derived: list[dict[str, Any]] = []

        def add_media_evidence(
            *,
            evidence_id: str,
            evidence_kind: str,
            media_path: str | Path,
            manifest: Mapping[str, Any],
            source_id: str,
            comparison_candidate_id: str | None = None,
        ) -> None:
            entry = {
                "type": manifest.get("media_type"),
                "path": str(Path(media_path).resolve()),
                "representation_id": manifest.get("representation_id"),
                "mime_type": manifest.get("mime_type"),
                "size_bytes": manifest.get("size_bytes"),
                "sha256": manifest.get("sha256"),
                "source_id": source_id,
                "evidence_id": evidence_id,
                "evidence_kind": evidence_kind,
            }
            # When audio perception is disabled, a generated video with an
            # audio stream is shown through a byte-distinct visual-only proxy.
            # Preserve a public, hash-bound derivation edge back to the exact
            # provider output instead of pretending the proxy is that output.
            if manifest.get("parent_sha256") is not None:
                entry["parent_sha256"] = manifest.get("parent_sha256")
            if manifest.get("derivation") is not None:
                entry["derivation"] = copy.deepcopy(manifest.get("derivation"))
            if comparison_candidate_id is not None:
                entry["comparison_candidate_id"] = comparison_candidate_id
            media.append(entry)
            evidence[evidence_id] = {
                key: value for key, value in entry.items() if key not in {"path", "type"}
            }

        def add_transport_only_audio_evidence(
            *,
            evidence_id: str,
            source_id: str,
            source_kind: str,
            resolved_audio: Mapping[str, Any],
        ) -> None:
            """Bind exact audio bytes without exposing them to the Teacher."""

            manifest = resolved_audio["representation"]
            record = resolved_audio["record"]
            row = {
                "evidence_id": evidence_id,
                "evidence_kind": "slot_audio",
                "source_id": source_id,
                "representation_id": manifest.get("representation_id"),
                "mime_type": manifest.get("mime_type"),
                "size_bytes": manifest.get("size_bytes"),
                "sha256": manifest.get("sha256"),
                "selector": copy.deepcopy(manifest.get("selector")),
                "transport_eligible": bool(record.get("transport_eligible", False)),
                "transport_authorization_id": record.get(
                    "transport_authorization_id"
                ),
                "audio_quality_status": record.get(
                    "audio_quality_status", "unassessed"
                ),
                "verification_basis": record.get("verification_basis")
                or (
                    record.get("speaker_binding", {}).get("basis")
                    if isinstance(record.get("speaker_binding"), Mapping)
                    else None
                ),
                "perception_exposed": False,
            }
            if source_kind == "uploaded_asset":
                row["speaker_name"] = record.get("speaker_name")
            else:
                row["speaker_binding"] = copy.deepcopy(
                    record.get("speaker_binding")
                )
            evidence[evidence_id] = row

        target_kind = "target_video" if candidate["target"] == "video" else "target_image"
        target_evidence_id = f"{candidate_id}.target"
        target_path = path
        target_manifest = representation
        if (
            candidate["target"] == "video"
            and self.config.audio_perception != "verified"
        ):
            from .media import strip_video_audio

            inspect_dir = self.config.output_dir / "inspections" / candidate_id
            suffix = path.suffix.lower() or ".mp4"
            target_path, audio_removed = strip_video_audio(
                path, inspect_dir / f"visual_only{suffix}"
            )
            if audio_removed:
                target_manifest = self._media_manifest(
                    target_path, self.state.next_id("repr_visual_only")
                )
                target_manifest.update(
                    {
                        "source_id": candidate_id,
                        "parent_sha256": representation.get("sha256"),
                        "derivation": {
                            "operation": "strip_audio_for_visual_perception",
                            "source_representation_id": representation.get(
                                "representation_id"
                            ),
                        },
                    }
                )
                derived.append(target_manifest)
        add_media_evidence(
            evidence_id=target_evidence_id,
            evidence_kind=target_kind,
            media_path=target_path,
            manifest=target_manifest,
            source_id=candidate_id,
        )
        if candidate["target"] == "image":
            for index, source in enumerate(candidate.get("input_references") or []):
                resolved_reference = self._resolve_source(source)
                add_media_evidence(
                    evidence_id=f"{candidate_id}.input_reference.{index}",
                    evidence_kind="input_reference",
                    media_path=resolved_reference["path"],
                    manifest=resolved_reference["representation"],
                    source_id=str(source["source_id"]),
                )
        provider_evidence_id = f"{candidate_id}.provider_manifest"
        evidence[provider_evidence_id] = {
            "evidence_id": provider_evidence_id,
            "evidence_kind": "provider_manifest",
            "source_id": candidate_id,
        }
        if candidate["target"] == "video":
            from .media import (
                extract_audio,
                extract_video_frames,
                extract_video_tail_frame,
                ffprobe,
            )

            inspect_dir = self.config.output_dir / "inspections" / candidate_id
            probe = ffprobe(path)
            duration_values: list[Any] = []
            if isinstance(probe.get("format"), Mapping):
                duration_values.append(probe["format"].get("duration"))
            duration_values.extend(
                row.get("duration")
                for row in probe.get("streams") or []
                if isinstance(row, Mapping) and row.get("codec_type") == "video"
            )
            duration = None
            for value in duration_values:
                try:
                    candidate_duration = float(value)
                except (TypeError, ValueError):
                    continue
                if candidate_duration > 0:
                    duration = candidate_duration
                    break
            if duration is None:
                raise RuntimeError("cannot inspect video without a positive duration")
            frame_paths = extract_video_frames(
                path,
                inspect_dir / "frames",
                timestamps=[0.0, duration / 2.0],
            )
            tail_path = extract_video_tail_frame(path, inspect_dir / "frames" / "tail.png")
            labeled_frames = [
                ("frame0", frame_paths[0]),
                ("mid", frame_paths[1]),
                ("tail", tail_path),
            ]
            for index, (frame_role, frame_path) in enumerate(labeled_frames):
                manifest = self._media_manifest(frame_path, self.state.next_id("repr_frame"))
                manifest["source_id"] = self.state.next_id("frame")
                manifest["frame_role"] = frame_role
                derived.append(manifest)
                add_media_evidence(
                    evidence_id=f"{candidate_id}.frame.{index}",
                    evidence_kind="target_frame",
                    media_path=manifest["path"],
                    manifest=manifest,
                    source_id=str(manifest["source_id"]),
                )
                media[-1]["frame_role"] = frame_role
            audio_path = (
                extract_audio(path, inspect_dir / "audio.wav", required=False)
                if self.config.audio_perception == "verified"
                else None
            )
            if audio_path is not None:
                manifest = self._media_manifest(audio_path, self.state.next_id("repr_audio"))
                manifest["source_id"] = self.state.next_id("audio_clip")
                manifest["selector"] = {
                    "mode": "time_range",
                    "start_ms": 0,
                    "end_ms": max(1, int(round(duration * 1000))),
                }
                derived.append(manifest)
                if self.config.audio_perception == "verified":
                    add_media_evidence(
                        evidence_id=f"{candidate_id}.audio",
                        evidence_kind="target_audio",
                        media_path=manifest["path"],
                        manifest=manifest,
                        source_id=str(manifest["source_id"]),
                    )

        bound_writeback_sources: list[dict[str, Any]] = []
        if candidate["target"] == "video":
            condition = candidate.get("condition")
            if condition:
                source = condition["image_source"]
                resolved = self._resolve_source(source)
                evidence_id = f"{candidate_id}.condition"
                add_media_evidence(
                    evidence_id=evidence_id,
                    evidence_kind="condition_image",
                    media_path=resolved["path"],
                    manifest=resolved["representation"],
                    source_id=str(source["source_id"]),
                )
                if source.get("kind") == "uploaded_asset":
                    bound_writeback_sources.append({
                        "source_id": str(source["source_id"]),
                        "modality": "image",
                        "evidence_ids": [evidence_id],
                    })
            for index, slot in enumerate(candidate.get("memory_slots") or []):
                image_source = slot["image_source"]
                resolved_image = self._resolve_source(image_source)
                image_evidence_id = f"{candidate_id}.slot.{index}.image"
                add_media_evidence(
                    evidence_id=image_evidence_id,
                    evidence_kind="slot_image",
                    media_path=resolved_image["path"],
                    manifest=resolved_image["representation"],
                    source_id=str(image_source["source_id"]),
                )
                if image_source.get("kind") == "uploaded_asset":
                    bound_writeback_sources.append({
                        "source_id": str(image_source["source_id"]),
                        "modality": "image",
                        "evidence_ids": [image_evidence_id],
                    })
                audio_source = slot.get("audio_source")
                if audio_source is not None:
                    resolved_audio = self._resolve_source(audio_source)
                    audio_evidence_id = f"{candidate_id}.slot.{index}.audio"
                    if self.config.audio_perception == "verified":
                        add_media_evidence(
                            evidence_id=audio_evidence_id,
                            evidence_kind="slot_audio",
                            media_path=resolved_audio["path"],
                            manifest=resolved_audio["representation"],
                            source_id=str(audio_source["source_id"]),
                        )
                        evidence[audio_evidence_id]["perception_exposed"] = True
                        media[-1]["perception_exposed"] = True
                    else:
                        add_transport_only_audio_evidence(
                            evidence_id=audio_evidence_id,
                            source_id=str(audio_source["source_id"]),
                            source_kind=str(audio_source["kind"]),
                            resolved_audio=resolved_audio,
                        )
                    if audio_source.get("kind") == "uploaded_asset":
                        bound_writeback_sources.append(
                            {
                                "source_id": str(audio_source["source_id"]),
                                "modality": "audio",
                                "evidence_ids": [audio_evidence_id],
                            }
                        )

        comparison_pool: list[dict[str, Any]] = []
        group = self.shot["generation_groups"][candidate["group_id"]]
        has_assessed_peer = any(
            other_id != candidate_id
            and bool(self.shot["candidates"][other_id].get("assessment_id"))
            for other_id in group["candidate_ids"]
        )
        if has_assessed_peer and not bool(arguments.get("compare_pool")):
            raise ToolRejected(
                "comparison_pool_required",
                "later candidates must expose assessed peers for comparison",
            )
        if bool(arguments.get("compare_pool")):
            for other_id in group["candidate_ids"]:
                if other_id == candidate_id:
                    continue
                other = self.shot["candidates"][other_id]
                assessment_id = other.get("assessment_id")
                if not assessment_id:
                    continue
                preview_evidence_ids: list[str] = []
                other_inspection = self.shot["inspections"].get(other.get("inspection_id")) or {}
                preview_rows = [
                    row
                    for row in other_inspection.get("derived_representations") or []
                    if str(row.get("media_type") or "").startswith("image")
                ]
                if not preview_rows:
                    preview_rows = [other["representation"]]
                for index, preview in enumerate(preview_rows):
                    evidence_id = f"{candidate_id}.comparison.{other_id}.{index}"
                    add_media_evidence(
                        evidence_id=evidence_id,
                        evidence_kind="comparison_frame" if other["target"] == "video" else "comparison_candidate",
                        media_path=preview["path"],
                        manifest=preview,
                        source_id=other_id,
                        comparison_candidate_id=other_id,
                    )
                    preview_evidence_ids.append(evidence_id)
                comparison_pool.append(
                    {
                        "candidate_id": other_id,
                        "assessment_id": assessment_id,
                        "quality": self._candidate_quality(other_id),
                        "preview_evidence_ids": preview_evidence_ids,
                    }
                )

        checklist = self._checklist(candidate)
        for check in checklist:
            allowed_kinds = set(check.get("allowed_evidence_kinds") or [])
            check["allowed_evidence_ids"] = [
                evidence_id
                for evidence_id, row in evidence.items()
                if row["evidence_kind"] in allowed_kinds
            ]

        requested_uses = list(arguments.get("requested_uses") or [])
        primary_use = (
            "final_video"
            if candidate["target"] == "video"
            else "condition"
            if candidate["purpose"] == "first_frame"
            else "memory_slot"
        )
        if primary_use not in requested_uses:
            raise ToolRejected(
                "primary_use_inspection_required",
                f"{candidate['target']} candidate inspection must request {primary_use}",
            )
        compatible_uses = {primary_use, "memory_writeback"}
        if candidate["target"] == "video":
            # A selected video's exact derived full frames may be assessed for
            # future frame-zero use.  The resulting per-frame contract is
            # required before such a memory can become condition eligible.
            compatible_uses.add("condition")
        incompatible = set(requested_uses) - compatible_uses
        if incompatible:
            raise ToolRejected(
                "incompatible_requested_use",
                f"candidate cannot be inspected for uses: {sorted(incompatible)}",
            )
        use_contracts: list[dict[str, Any]] = []

        def add_use_contract(
            use: str,
            source_id: str,
            modality: str,
            evidence_ids: list[str],
        ) -> None:
            use_contracts.append(
                {
                    "use_contract_id": self.state.next_id("use"),
                    "use": use,
                    "source_ids": [source_id],
                    "modality": modality,
                    "allowed_evidence_ids": list(dict.fromkeys(evidence_ids)),
                }
            )

        primary_evidence_ids = [
            evidence_id
            for evidence_id, row in evidence.items()
            if row["evidence_kind"] in {target_kind, "target_frame", "target_audio"}
        ]
        add_use_contract(primary_use, candidate_id, candidate["target"], primary_evidence_ids)
        if candidate["target"] == "video" and "condition" in requested_uses:
            for row in derived:
                if not str(row.get("media_type") or "").startswith("image"):
                    continue
                matching_ids = [
                    evidence_id
                    for evidence_id, item in evidence.items()
                    if item.get("source_id") == row.get("source_id")
                ]
                if matching_ids:
                    add_use_contract(
                        "condition",
                        str(row["source_id"]),
                        "image",
                        matching_ids,
                    )
        if "memory_writeback" in requested_uses:
            if candidate["target"] == "image":
                add_use_contract("memory_writeback", candidate_id, "image", [target_evidence_id])
            else:
                for row in derived:
                    modality = str(row.get("media_type") or "").split("/", 1)[0]
                    selector = row.get("selector") or {}
                    has_exact_audio_range = (
                        modality == "audio"
                        and isinstance(selector.get("start_ms"), int)
                        and isinstance(selector.get("end_ms"), int)
                        and selector["end_ms"] > selector["start_ms"]
                    )
                    if modality == "audio" and (
                        self.config.audio_perception != "verified" or not has_exact_audio_range
                    ):
                        continue
                    matching_ids = [
                        evidence_id
                        for evidence_id, item in evidence.items()
                        if item.get("source_id") == row.get("source_id")
                    ]
                    # A whole generated soundtrack is not a trustworthy voice
                    # memory: it may contain music, effects, or multiple
                    # speakers.  Keep exact uploaded/bound audio writeback
                    # below, but require a future clip-selection contract
                    # before generated audio can enter the bank.
                    if modality == "image" and matching_ids:
                        add_use_contract(
                            "memory_writeback",
                            str(row["source_id"]),
                            modality,
                            matching_ids,
                        )
                for row in bound_writeback_sources:
                    add_use_contract(
                        "memory_writeback",
                        str(row["source_id"]),
                        str(row["modality"]),
                        list(row["evidence_ids"]),
                    )

        inspection_id = self.state.next_id("inspect")
        capability_mask = {
            "visual": True,
            "audio": self.config.audio_perception == "verified",
        }
        assessment_contract = build_assessment_contract(
            checklist, capability_mask, use_contracts
        )
        inspection_revision_hash = sha256_json(
            {
                "checklist": checklist,
                "use_contracts": use_contracts,
                "capability_mask": capability_mask,
                "assessment_contract": assessment_contract,
            }
        )
        inspection = {
            "inspection_id": inspection_id,
            "candidate_id": candidate_id,
            "target": candidate["target"],
            "requested_uses": requested_uses,
            "checklist": checklist,
            "checklist_sha256": inspection_revision_hash,
            "media": media,
            "derived_representations": derived,
            "evidence": evidence,
            "use_contracts": use_contracts,
            "comparison_pool": comparison_pool,
            "capability_mask": capability_mask,
            "assessment_contract": assessment_contract,
        }
        self.shot["inspections"][inspection_id] = inspection
        candidate["inspection_id"] = inspection_id
        group = self.shot["generation_groups"][candidate["group_id"]]
        remaining = self._remaining_generation_attempts(candidate["target"])
        observation = self._new_observation(
            "candidate_inspection",
            {
                "ok": True,
                "inspection_id": inspection_id,
                "candidate_id": candidate_id,
                "target": candidate["target"],
                "raw_user_request": self.shot["raw_user_request"],
                "prompt_artifact": copy.deepcopy(self.shot["prompt_artifacts"][candidate["prompt_artifact_id"]]),
                "input_manifest": {
                    "condition": candidate.get("condition"),
                    "memory_slots": candidate.get("memory_slots") or [],
                    "provider_request": candidate.get("provider", {}).get("request_payload"),
                },
                "checklist": checklist,
                "checklist_sha256": inspection["checklist_sha256"],
                "inspection_revision_hash": inspection["checklist_sha256"],
                "allowed_evidence": list(evidence.values()),
                "use_contracts": use_contracts,
                "comparison_pool": comparison_pool,
                "remaining_attempts": remaining,
                "capability_mask": inspection["capability_mask"],
                "assessment_contract": copy.deepcopy(
                    inspection["assessment_contract"]
                ),
                "media": media,
            },
        )
        inspection["observation_id"] = observation["observation_id"]
        return observation

    @staticmethod
    def _result_status_map(check_results: list[Mapping[str, Any]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for item in check_results:
            check_id = str(item.get("check_id") or "")
            status = str(item.get("result") or "")
            if not check_id or status not in _ASSESSMENT_STATUSES:
                raise ToolRejected("invalid_check_result", "every check result needs a valid check_id/status")
            if check_id in result:
                raise ToolRejected("duplicate_check_result", f"duplicate check result: {check_id}")
            result[check_id] = status
        return result

    @staticmethod
    def _assessment_quality(
        inspection: Mapping[str, Any], status_by_check: Mapping[str, str]
    ) -> str:
        critical_ids = {
            str(item["check_id"])
            for item in inspection.get("checklist") or []
            if item.get("severity") == "critical"
        }
        statuses = [status_by_check[item] for item in critical_ids]
        if any(value in {"fail", "uncertain"} for value in statuses):
            return "failed"
        if any(value == "capability_blocked" for value in statuses):
            return "capability_limited"
        return "qualified"

    def _candidate_quality(self, candidate_id: str) -> str:
        candidate = self.shot["candidates"][candidate_id]
        assessment_id = candidate.get("assessment_id")
        if not assessment_id:
            return "unassessed"
        assessment = self.shot["assessments"][assessment_id]
        quality = assessment.get("quality")
        if quality in {"failed", "capability_limited", "qualified"}:
            return str(quality)
        inspection = self.shot["inspections"][assessment["inspection_id"]]
        return self._assessment_quality(inspection, assessment["status_by_check"])

    @staticmethod
    def _primary_use(candidate: Mapping[str, Any]) -> str:
        if candidate.get("target") == "video":
            return "final_video"
        return "condition" if candidate.get("purpose") == "first_frame" else "memory_slot"

    def _assessment_authorizes_primary_use(
        self, candidate_id: str, *, allow_failed_best_effort: bool = False
    ) -> bool:
        candidate = self.shot["candidates"][candidate_id]
        assessment = self.shot["assessments"].get(candidate.get("assessment_id"))
        if not assessment:
            return False
        primary = self._primary_use(candidate)
        for contract in assessment.get("use_contracts") or []:
            contract_id = contract.get("use_contract_id")
            status = assessment.get("use_status_by_contract", {}).get(contract_id)
            if (
                contract.get("use") == primary
                and contract.get("source_ids") == [candidate_id]
                and (
                    status in {"pass", "capability_blocked"}
                    or (allow_failed_best_effort and status in {"fail", "uncertain"})
                )
            ):
                return True
        return False

    def tool_submit_candidate_assessment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        inspection_id = str(arguments["inspection_id"])
        inspection = self.shot["inspections"].get(inspection_id)
        if not inspection:
            raise ToolRejected("unknown_inspection", f"unknown inspection: {inspection_id}")
        if arguments["inspection_revision_hash"] != inspection["checklist_sha256"]:
            raise ToolRejected("inspection_hash_mismatch", "inspection revision hash is stale")
        expected_assessment_contract = build_assessment_contract(
            inspection.get("checklist") or [],
            inspection.get("capability_mask") or {},
            inspection.get("use_contracts") or [],
        )
        if inspection.get("assessment_contract") != expected_assessment_contract:
            raise ToolRejected(
                "invalid_assessment_contract",
                "the stored assessment contract differs from the frozen inspection",
            )
        if (
            arguments.get("module_applicability")
            != expected_assessment_contract["module_applicability"]
        ):
            raise ToolRejected(
                "module_applicability_mismatch",
                "copy module_applicability exactly from the frozen assessment contract",
            )
        candidate = self.shot["candidates"][inspection["candidate_id"]]
        if candidate.get("assessment_id"):
            raise ToolRejected("candidate_already_assessed", "candidate assessment is immutable")
        check_results = list(arguments.get("check_results") or [])
        status_by_check = self._result_status_map(check_results)
        checklist_by_id = {item["check_id"]: item for item in inspection["checklist"]}
        expected_ids = set(checklist_by_id)
        if set(status_by_check) != expected_ids:
            raise ToolRejected(
                "incomplete_checklist",
                "assessment must cover every checklist item exactly once",
                details={
                    "missing": sorted(expected_ids - set(status_by_check)),
                    "extra": sorted(set(status_by_check) - expected_ids),
                },
            )
        check_result_by_id = {
            str(item.get("check_id") or ""): item for item in check_results
        }
        for forced_result in expected_assessment_contract["forced_check_results"]:
            if check_result_by_id.get(forced_result["check_id"]) != forced_result:
                raise ToolRejected(
                    "forced_check_result_mismatch",
                    "copy every forced check result exactly from the frozen assessment contract",
                    details={"check_id": forced_result["check_id"]},
                )
        evidence = inspection.get("evidence") or {}
        for result in check_results:
            check = checklist_by_id[result["check_id"]]
            result_value = result["result"]
            evidence_ids = list(result.get("evidence_ids") or [])
            allowed_ids = set(check.get("allowed_evidence_ids") or [])
            invalid_ids = set(evidence_ids) - allowed_ids
            if invalid_ids:
                raise ToolRejected(
                    "check_evidence_not_authorized",
                    f"check {result['check_id']} cites unauthorized evidence",
                    details={"evidence_ids": sorted(invalid_ids)},
                )
            if result_value in {"pass", "fail", "uncertain"} and not evidence_ids:
                raise ToolRejected(
                    "check_evidence_required",
                    f"check {result['check_id']} requires visible evidence",
                )
            if result_value == "not_applicable" and not check.get("not_applicable_allowed"):
                raise ToolRejected(
                    "not_applicable_forbidden",
                    f"check {result['check_id']} cannot be marked not_applicable",
                )
            if result_value == "capability_blocked":
                audio_blocked = (
                    check.get("module") == "speech_audio"
                    and not inspection.get("capability_mask", {}).get("audio", False)
                )
                if not audio_blocked:
                    raise ToolRejected(
                        "capability_blocked_not_authorized",
                        f"check {result['check_id']} is observable under the frozen capability mask",
                    )

        applicability_rows = list(arguments.get("module_applicability") or [])
        applicability: dict[str, str] = {}
        for row in applicability_rows:
            module = str(row.get("module") or "")
            if module in applicability:
                raise ToolRejected("duplicate_module_applicability", f"duplicate module: {module}")
            applicability[module] = str(row.get("applicability") or "")
        if set(applicability) != _SEMANTIC_MODULES:
            raise ToolRejected(
                "incomplete_module_applicability",
                "module_applicability must cover every semantic module exactly once",
                details={
                    "missing": sorted(_SEMANTIC_MODULES - set(applicability)),
                    "extra": sorted(set(applicability) - _SEMANTIC_MODULES),
                },
            )
        applicable_from_checks = {
            str(item["module"])
            for item in inspection["checklist"]
            if item.get("module") in _SEMANTIC_MODULES
        }
        mismatched_modules = [
            module
            for module in sorted(_SEMANTIC_MODULES)
            if (applicability[module] == "applicable") != (module in applicable_from_checks)
        ]
        if mismatched_modules:
            raise ToolRejected(
                "module_applicability_mismatch",
                "module applicability must agree with the frozen checklist",
                details={"modules": mismatched_modules},
            )

        use_contracts = list(inspection.get("use_contracts") or [])
        contracts_by_id = {item["use_contract_id"]: item for item in use_contracts}
        use_results = list(arguments.get("use_results") or [])
        forced_use_results = {
            str(item.get("use_contract_id") or ""): item
            for item in expected_assessment_contract.get("forced_use_results") or []
        }
        submitted_use_results = {
            str(item.get("use_contract_id") or ""): item for item in use_results
        }
        for contract_id, forced_result in forced_use_results.items():
            if submitted_use_results.get(contract_id) != forced_result:
                raise ToolRejected(
                    "forced_use_result_mismatch",
                    "copy every forced use result exactly from the frozen assessment contract",
                    details={"use_contract_id": contract_id},
                )
        use_status_by_contract: dict[str, str] = {}
        for result in use_results:
            contract_id = str(result.get("use_contract_id") or "")
            if contract_id in use_status_by_contract:
                raise ToolRejected("duplicate_use_result", f"duplicate use result: {contract_id}")
            contract = contracts_by_id.get(contract_id)
            if not contract:
                raise ToolRejected("unknown_use_contract", f"unknown use contract: {contract_id}")
            if result.get("use") != contract.get("use") or result.get("source_ids") != contract.get("source_ids"):
                raise ToolRejected("use_contract_mismatch", f"use result does not match {contract_id}")
            evidence_ids = list(result.get("evidence_ids") or [])
            invalid_ids = set(evidence_ids) - set(contract.get("allowed_evidence_ids") or [])
            if invalid_ids:
                raise ToolRejected(
                    "use_evidence_not_authorized",
                    f"use result {contract_id} cites unauthorized evidence",
                    details={"evidence_ids": sorted(invalid_ids)},
                )
            result_value = str(result.get("result") or "")
            if result_value in {"pass", "fail", "uncertain"} and not evidence_ids:
                raise ToolRejected("use_evidence_required", f"use result {contract_id} requires evidence")
            if result_value == "capability_blocked":
                if contract.get("modality") != "audio" or inspection.get("capability_mask", {}).get("audio", False):
                    raise ToolRejected(
                        "use_capability_blocked_not_authorized",
                        f"use contract {contract_id} is observable",
                    )
            use_status_by_contract[contract_id] = result_value
        if set(use_status_by_contract) != set(contracts_by_id):
            raise ToolRejected(
                "incomplete_use_results",
                "use_results must cover every use contract exactly once",
                details={
                    "missing": sorted(set(contracts_by_id) - set(use_status_by_contract)),
                    "extra": sorted(set(use_status_by_contract) - set(contracts_by_id)),
                },
            )
        if self.config.audio_perception != "verified":
            audio_checks = {
                item["check_id"]
                for item in inspection["checklist"]
                if item["module"] == "speech_audio"
            }
            invalid = [
                check_id
                for check_id in audio_checks
                if status_by_check[check_id] != "capability_blocked"
            ]
            if invalid:
                raise ToolRejected(
                    "audio_judgment_not_allowed",
                    "audio-disabled policy requires capability_blocked for every speech/audio check",
                )
        group = self.shot["generation_groups"][candidate["group_id"]]
        remaining = self._remaining_generation_attempts(candidate["target"])
        next_action = str(arguments["next_action"])
        retry_change = copy.deepcopy(arguments.get("retry_change"))
        selection = copy.deepcopy(arguments.get("selection"))
        current_quality = self._assessment_quality(inspection, status_by_check)
        primary_use = self._primary_use(candidate)
        primary_use_statuses = [
            use_status_by_contract.get(contract.get("use_contract_id"))
            for contract in use_contracts
            if contract.get("use") == primary_use
            and contract.get("source_ids") == [candidate["candidate_id"]]
        ]
        if any(status in {"fail", "uncertain"} for status in primary_use_statuses):
            current_quality = "failed"
        elif (
            current_quality == "qualified"
            and any(status == "capability_blocked" for status in primary_use_statuses)
        ):
            current_quality = "capability_limited"
        has_failure = current_quality == "failed"
        existing_qualified = [
            item for item in group["candidate_ids"] if item != candidate["candidate_id"] and self._candidate_quality(item) == "qualified"
        ]
        selected_candidate_id: str | None = None
        if next_action == "retry":
            if remaining <= 0:
                raise ToolRejected("retry_budget_exhausted", "budget is exhausted; select best assessed candidate")
            if not has_failure or existing_qualified:
                raise ToolRejected("unnecessary_retry", "retry requires an unresolved failure and no qualified candidate")
            if not isinstance(retry_change, dict) or selection is not None:
                raise ToolRejected("invalid_retry_decision", "retry requires retry_change and selection=null")
            axes = retry_change.get("axes") or []
            allowed = _IMAGE_AXES if candidate["target"] == "image" else _VIDEO_AXES
            if not axes or set(axes) - allowed or len(axes) != len(set(axes)):
                raise ToolRejected("invalid_retry_axes", "retry axes are empty, duplicate, or invalid")
            if (
                candidate["target"] == "video"
                and self.config.prompt_rewrite_mode == "passthrough_disabled"
                and "prompt" in axes
            ):
                raise ToolRejected(
                    "passthrough_retry_axes_forbidden",
                    "passthrough_disabled retries cannot change prompt text",
                )
            if (
                candidate["target"] == "image"
                and self.config.prompt_rewrite_mode == "passthrough_disabled"
                and "prompt" in axes
            ):
                raise ToolRejected(
                    "passthrough_retry_axes_forbidden",
                    "passthrough_disabled retries cannot change prompt text",
                )
            failed_ids = set(retry_change.get("failed_check_ids") or [])
            actual_failed_ids = {
                key
                for key, value in status_by_check.items()
                if value in {"fail", "uncertain"}
            }
            actual_failed_ids.update(
                key
                for key, value in use_status_by_contract.items()
                if value in {"fail", "uncertain"}
                and any(
                    contract.get("use_contract_id") == key
                    and contract.get("use") == primary_use
                    for contract in use_contracts
                )
            )
            if failed_ids != actual_failed_ids:
                raise ToolRejected("failed_check_ids_mismatch", "retry failed_check_ids must exactly cover fail/uncertain checks")
        elif next_action == "select":
            if retry_change is not None or not isinstance(selection, dict):
                raise ToolRejected("invalid_selection_decision", "select requires selection and retry_change=null")
            ranked = list(selection.get("ranked_candidate_ids") or [])
            assessed = [item for item in group["candidate_ids"] if self.shot["candidates"][item].get("assessment_id") or item == candidate["candidate_id"]]
            if set(ranked) != set(assessed) or len(ranked) != len(set(ranked)):
                raise ToolRejected("invalid_candidate_ranking", "ranking must cover all assessed candidates exactly once")
            selected_id = str(selection.get("selected_candidate_id") or "")
            if not ranked or ranked[0] != selected_id:
                raise ToolRejected("selected_candidate_not_first", "selected candidate must rank first")
            quality = current_quality if selected_id == candidate["candidate_id"] else self._candidate_quality(selected_id)
            quality_rank = {"failed": 0, "capability_limited": 1, "qualified": 2}
            better_available = any(
                quality_rank.get(
                    current_quality
                    if item == candidate["candidate_id"]
                    else self._candidate_quality(item),
                    -1,
                )
                > quality_rank.get(quality, -1)
                for item in assessed
                if item != selected_id
            )
            if better_available:
                raise ToolRejected(
                    "better_candidate_available",
                    "the selected candidate must have the highest available assessed quality",
                )
            mode = str(selection.get("mode") or "")
            if quality == "failed" and remaining > 0:
                raise ToolRejected("premature_best_effort", "failed output must retry while budget remains")
            expected_mode = "best_effort" if quality == "failed" else "capability_limited" if quality == "capability_limited" else "requirements_met"
            if mode != expected_mode:
                raise ToolRejected("selection_mode_mismatch", f"selection mode must be {expected_mode}")
            selected_candidate_id = selected_id
            allow_failed_primary = (
                quality == "failed" and remaining == 0 and mode == "best_effort"
            )
            if selected_id == candidate["candidate_id"]:
                primary_authorized = any(
                    contract.get("use") == self._primary_use(candidate)
                    and contract.get("source_ids") == [selected_id]
                    and use_status_by_contract.get(contract.get("use_contract_id"))
                    in (
                        {"pass", "capability_blocked", "fail", "uncertain"}
                        if allow_failed_primary
                        else {"pass", "capability_blocked"}
                    )
                    for contract in use_contracts
                )
            else:
                primary_authorized = self._assessment_authorizes_primary_use(
                    selected_id, allow_failed_best_effort=allow_failed_primary
                )
            if not primary_authorized:
                raise ToolRejected(
                    "selected_primary_use_not_authorized",
                    "selected candidate lacks a passing primary-use contract",
                )
        else:
            raise ToolRejected("invalid_next_action", "next_action must be retry or select")
        assessment_id = self.state.next_id("assess")
        assessment = {
            "assessment_id": assessment_id,
            "inspection_id": inspection_id,
            "candidate_id": candidate["candidate_id"],
            "module_applicability": copy.deepcopy(arguments.get("module_applicability") or []),
            "check_results": copy.deepcopy(arguments["check_results"]),
            "status_by_check": status_by_check,
            "quality": current_quality,
            "use_contracts": copy.deepcopy(use_contracts),
            "use_results": copy.deepcopy(use_results),
            "use_status_by_contract": use_status_by_contract,
            "next_action": next_action,
            "retry_change": retry_change,
            "selection": selection,
        }
        self.shot["assessments"][assessment_id] = assessment
        candidate["assessment_id"] = assessment_id
        self.shot["pending_candidate_id"] = None
        response: dict[str, Any] = {
            "ok": True,
            "assessment_id": assessment_id,
            "candidate_id": candidate["candidate_id"],
            "next_action": next_action,
            "remaining_attempts": remaining,
        }
        if next_action == "retry":
            artifact = self.shot["prompt_artifacts"][candidate["prompt_artifact_id"]]
            artifact["last_candidate_id"] = candidate["candidate_id"]
            group["retry_obligation"] = {
                "candidate_id": candidate["candidate_id"],
                "assessment_id": assessment_id,
                "axes": list(retry_change["axes"]),
            }
            response["required_retry_axes"] = retry_change["axes"]
            return response
        assert selected_candidate_id is not None
        selection_id = self.state.next_id("selection")
        selected = self.shot["candidates"][selected_candidate_id]
        selection_record = {
            "selection_id": selection_id,
            "target": candidate["target"],
            "purpose": group["purpose"],
            "group_id": candidate["group_id"],
            "selected_candidate_id": selected_candidate_id,
            "ranked_candidate_ids": list(selection["ranked_candidate_ids"]),
            "mode": selection["mode"],
            "reason": selection.get("reason") or "",
            "risk_manifest": [
                item for item in self.shot["assessments"][selected.get("assessment_id") or assessment_id]["check_results"]
                if item.get("result") in {"fail", "uncertain", "capability_blocked"}
            ],
        }
        self.shot["selections"][selection_id] = selection_record
        group["selection_id"] = selection_id
        group["retry_obligation"] = None
        response["selection_id"] = selection_id
        response[f"{candidate['target']}_selection_id"] = selection_id
        response["selected_candidate_id"] = selected_candidate_id
        response["mode"] = selection["mode"]
        if candidate["target"] == "video":
            self.shot["video_selection_id"] = selection_id
        return response

    def _writeback_source(
        self,
        item: Mapping[str, Any],
        *,
        selected_video_candidate_id: str,
    ) -> dict[str, Any]:
        """Resolve one writeback source through its exact passed use gate."""

        source_id = str(item["source_id"])
        assessment_id = str(item["assessment_id"])
        use_contract_id = str(item["use_contract_id"])
        observation_id = str(item["observation_id"])
        assessment = self.shot["assessments"].get(assessment_id)
        if not assessment:
            raise ToolRejected("unknown_writeback_assessment", f"unknown assessment: {assessment_id}")
        inspection = self.shot["inspections"].get(assessment.get("inspection_id"))
        if not inspection or inspection.get("observation_id") != observation_id:
            raise ToolRejected(
                "writeback_observation_mismatch",
                "observation_id must be the inspection observation bound to the assessment",
            )
        contract = next(
            (
                row
                for row in assessment.get("use_contracts") or []
                if row.get("use_contract_id") == use_contract_id
            ),
            None,
        )
        contract_status = assessment.get("use_status_by_contract", {}).get(
            use_contract_id
        )
        requested_transport_only_audio = bool(
            item.get("modality") == "audio"
            and item.get("verification_basis") == "trusted_user_declared"
            and self.config.audio_transport == "enabled"
            and self.config.audio_perception != "verified"
        )
        if (
            not contract
            or contract.get("use") != "memory_writeback"
            or contract.get("source_ids") != [source_id]
            or contract.get("modality") != item.get("modality")
            or contract_status
            != ("capability_blocked" if requested_transport_only_audio else "pass")
        ):
            raise ToolRejected(
                "writeback_use_not_authorized",
                "source requires its exact authorized memory_writeback use contract",
            )
        if requested_transport_only_audio and source_id not in self.state.data.get(
            "assets", {}
        ):
            raise ToolRejected(
                "audio_transport_not_authorized",
                "transport-only audio writeback is limited to trusted uploaded voice references",
            )

        candidate = self.shot["candidates"].get(source_id)
        if candidate and candidate.get("target") == "image":
            selected = any(
                row.get("target") == "image" and row.get("selected_candidate_id") == source_id
                for row in self.shot["selections"].values()
            )
            if not selected or inspection.get("candidate_id") != source_id:
                raise ToolRejected(
                    "generated_writeback_not_selected",
                    "generated image writeback requires its own selected, assessed candidate",
                )
            representation = candidate["representation"]
        else:
            representation = next(
                (
                    row
                    for row in inspection.get("derived_representations") or []
                    if source_id in {row.get("representation_id"), row.get("source_id")}
                ),
                None,
            )
            if representation is not None:
                if inspection.get("candidate_id") != selected_video_candidate_id:
                    raise ToolRejected(
                        "writeback_from_unselected_video",
                        "derived writeback media must come from the selected final video",
                    )
            else:
                asset = self.state.data["assets"].get(source_id)
                if not asset or source_id not in self.shot.get("upload_ids", []):
                    raise ToolRejected(
                        "invalid_writeback_source", f"writeback source is unknown: {source_id}"
                    )
                if inspection.get("candidate_id") != selected_video_candidate_id:
                    raise ToolRejected(
                        "upload_writeback_wrong_video",
                        "uploaded writeback authorization must come from the selected video inspection",
                    )
                evidence_rows = [
                    inspection.get("evidence", {}).get(evidence_id)
                    for evidence_id in contract.get("allowed_evidence_ids") or []
                ]
                representation_ids = {
                    str(row.get("representation_id") or "")
                    for row in evidence_rows
                    if isinstance(row, Mapping) and row.get("source_id") == source_id
                }
                if len(representation_ids) != 1:
                    raise ToolRejected(
                        "ambiguous_upload_writeback_representation",
                        "upload writeback must bind one exact inspected representation",
                    )
                representation_id = next(iter(representation_ids))
                representation = self._representation_for_asset(asset, representation_id)
                authorizations = self.shot.get("view_authorizations", {}).get(source_id) or []
                viewed_exact = any(
                    isinstance(row, Mapping)
                    and row.get("representation_id") == representation_id
                    for row in authorizations
                )
                if not viewed_exact:
                    raise ToolRejected(
                        "upload_writeback_not_viewed",
                        "upload writeback representation was not viewed in this shot",
                    )
                if requested_transport_only_audio:
                    subject_ids = [str(value) for value in item.get("subject_ids") or []]
                    if (
                        len(subject_ids) != 1
                        or not self._is_trusted_transport_audio(asset, subject_ids[0])
                    ):
                        raise ToolRejected(
                            "audio_transport_not_authorized",
                            "transport-only audio writeback requires the exact trusted uploaded speaker binding",
                        )

        media_type = str(representation.get("media_type") or "")
        actual_modality = media_type.split("/", 1)[0]
        if actual_modality != item.get("modality"):
            raise ToolRejected(
                "writeback_modality_mismatch",
                f"source is {actual_modality}, not {item.get('modality')}",
            )
        return copy.deepcopy(representation)

    def _condition_eligibility_for_writeback(
        self,
        item: Mapping[str, Any],
        *,
        representation: Mapping[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Derive future frame-zero eligibility from typed, passed evidence."""

        if item.get("modality") != "image":
            return False, {"policy": "storymem.condition_eligibility.v1", "basis": "not_image"}
        source_id = str(item.get("source_id") or "")
        assessment_id = str(item.get("assessment_id") or "")
        assessment = self.shot["assessments"].get(assessment_id) or {}

        def passed_contract(use: str) -> str | None:
            for contract in assessment.get("use_contracts") or []:
                contract_id = str(contract.get("use_contract_id") or "")
                if (
                    contract.get("use") == use
                    and contract.get("source_ids") == [source_id]
                    and assessment.get("use_status_by_contract", {}).get(contract_id)
                    == "pass"
                ):
                    return contract_id
            return None

        generated = self.shot["candidates"].get(source_id)
        if isinstance(generated, Mapping) and generated.get("target") == "image":
            condition_contract_id = passed_contract("condition")
            eligible = (
                generated.get("purpose") == "first_frame"
                and condition_contract_id is not None
            )
            if bool(item.get("condition_eligible", False)) and not eligible:
                raise ToolRejected(
                    "condition_eligibility_not_authorized",
                    "generated image is not a passed first_frame condition candidate",
                )
            return eligible, {
                "policy": "storymem.condition_eligibility.v1",
                "basis": (
                    "selected_generated_first_frame_passed_condition_use"
                    if eligible
                    else "generated_reference_asset"
                ),
                "assessment_id": assessment_id,
                "condition_use_contract_id": condition_contract_id,
            }

        inspection = self.shot["inspections"].get(assessment.get("inspection_id")) or {}
        derived = next(
            (
                row
                for row in inspection.get("derived_representations") or []
                if source_id in {row.get("source_id"), row.get("representation_id")}
            ),
            None,
        )
        if isinstance(derived, Mapping):
            frame_role = str(derived.get("frame_role") or "")
            condition_contract_id = passed_contract("condition")
            eligible = bool(item.get("condition_eligible", False)) and (
                frame_role in {"frame0", "mid", "tail"}
                and condition_contract_id is not None
            )
            if bool(item.get("condition_eligible", False)) and not eligible:
                raise ToolRejected(
                    "condition_eligibility_not_authorized",
                    "derived video frame requires its exact passed condition-use contract",
                )
            return eligible, {
                "policy": "storymem.condition_eligibility.v1",
                "basis": "selected_video_full_frame" if eligible else "derived_media_not_authorized",
                "frame_role": frame_role or None,
                "assessment_id": assessment_id,
                "condition_use_contract_id": condition_contract_id,
            }

        uploaded = self.state.data.get("assets", {}).get(source_id)
        if isinstance(uploaded, Mapping):
            eligible = bool(uploaded.get("condition_eligible", False))
            if bool(item.get("condition_eligible", False)) and not eligible:
                raise ToolRejected(
                    "condition_eligibility_not_authorized",
                    "uploaded source was not explicitly ingested as a complete frame-zero image",
                )
            return eligible, {
                "policy": "storymem.condition_eligibility.v1",
                "basis": "inherited_uploaded_declaration" if eligible else "uploaded_reference_only",
                "source_asset_id": source_id,
            }

        if bool(item.get("condition_eligible", False)):
            raise ToolRejected(
                "condition_eligibility_not_authorized",
                "writeback source cannot be authorized as a frame-zero condition",
            )
        return False, {
            "policy": "storymem.condition_eligibility.v1",
            "basis": "unsupported_source",
        }

    def tool_update_memory_bank(self, arguments: dict[str, Any]) -> dict[str, Any]:
        video_selection_id = str(arguments["video_selection_id"])
        if video_selection_id != self.shot.get("video_selection_id"):
            raise ToolRejected("wrong_video_selection", "memory update must bind the selected final video")
        if self.shot.get("staged_memory_update"):
            raise ToolRejected("memory_update_already_staged", "only one memory update transaction is allowed")
        video_selection = self.shot["selections"][video_selection_id]
        selected_video_candidate_id = str(video_selection["selected_candidate_id"])
        memories: list[dict[str, Any]] = []
        client_refs: set[str] = set()
        for item in arguments.get("add_memories") or []:
            client_ref = str(item["client_ref"])
            if client_ref in client_refs:
                raise ToolRejected("duplicate_client_ref", f"duplicate memory client_ref: {client_ref}")
            client_refs.add(client_ref)
            representation = self._writeback_source(
                item,
                selected_video_candidate_id=selected_video_candidate_id,
            )
            modality = str(item["modality"])
            roles = [str(value) for value in item.get("roles") or []]
            person_role = bool(set(roles) & _PERSON_MEMORY_ROLES)
            entity_role = bool(set(roles) & _ENTITY_MEMORY_ROLES)
            if person_role and entity_role:
                raise ToolRejected(
                    "mixed_person_entity_roles",
                    "one memory cannot mix human character and non-human entity roles",
                )
            subject_ids = self._validate_subject_ids(
                list(item.get("subject_ids") or []),
                label="update_memory_bank.add_memories.subject_ids",
                required=person_role,
                exactly_one=modality == "audio",
            )
            if entity_role and subject_ids:
                raise ToolRejected(
                    "entity_subject_ids_forbidden",
                    "entity_identity/entity_appearance memories require subject_ids=[]",
                )
            source_id = str(item["source_id"])
            source_subjects: list[str] | None = None
            generated = self.shot["candidates"].get(source_id)
            uploaded = self.state.data.get("assets", {}).get(source_id)
            if isinstance(generated, Mapping) and generated.get("target") == "image":
                source_subjects = [str(value) for value in generated.get("subject_ids") or []]
            elif isinstance(uploaded, Mapping) and str(
                representation.get("media_type") or ""
            ).split("/", 1)[0] in {"image", "audio"}:
                source_subjects = [str(value) for value in uploaded.get("subject_ids") or []]
            if source_subjects is not None and set(subject_ids) != set(source_subjects):
                raise ToolRejected(
                    "writeback_subject_mismatch",
                    "generated/uploaded writeback must preserve exact source subject_ids",
                )
            if modality == "audio" and "voice" not in roles:
                raise ToolRejected(
                    "audio_voice_role_required",
                    "audio memory writeback requires the voice role",
                )
            if modality == "audio" and self.config.audio_transport != "enabled":
                raise ToolRejected(
                    "audio_transport_disabled",
                    "audio memory writeback requires audio_transport=enabled",
                )
            verification_basis = str(item.get("verification_basis") or "")
            if modality == "audio" and self.config.audio_perception == "verified":
                if verification_basis != "agent_heard":
                    raise ToolRejected(
                        "audio_not_heard",
                        "audio perceived by the agent requires verification_basis=agent_heard",
                    )
            elif modality == "audio":
                if (
                    verification_basis != "trusted_user_declared"
                    or not isinstance(uploaded, Mapping)
                    or len(subject_ids) != 1
                    or not self._is_trusted_transport_audio(uploaded, subject_ids[0])
                ):
                    raise ToolRejected(
                        "audio_transport_not_authorized",
                        "transport-only voice memory requires its exact trusted uploaded speaker binding",
                    )
            if modality == "audio" and not self._audio_time_range_is_exact(
                representation
            ):
                raise ToolRejected(
                    "audio_time_range_required",
                    "audio writeback requires an exact nonzero time-range selector",
                )
            condition_eligible, condition_eligibility_provenance = (
                self._condition_eligibility_for_writeback(
                    item,
                    representation=representation,
                )
                if modality == "image"
                else (False, None)
            )
            memory_id = self.state.next_id("mem")
            memories.append(
                {
                    "memory_id": memory_id,
                    "client_ref": client_ref,
                    "modality": modality,
                    "representation": copy.deepcopy(representation),
                    "tag_refs": copy.deepcopy(item.get("tag_refs") or []),
                    "roles": roles,
                    "subject_ids": subject_ids,
                    "description": str(item["description"]),
                    **(
                        {
                            "condition_eligible": condition_eligible,
                            "condition_eligibility_provenance": condition_eligibility_provenance,
                        }
                        if modality == "image"
                        else {}
                    ),
                    "provenance": {
                        "source_id": item["source_id"],
                        "observation_id": item["observation_id"],
                        "assessment_id": item["assessment_id"],
                        "use_contract_id": item["use_contract_id"],
                    },
                    **(
                        {
                            "verification_basis": verification_basis,
                            "speaker_binding": copy.deepcopy(
                                uploaded.get("speaker_binding")
                                if isinstance(uploaded, Mapping)
                                else {
                                    "basis": verification_basis,
                                    "subject_id": subject_ids[0],
                                    "evidence_ids": [item["observation_id"]],
                                }
                            ),
                            "transport_eligible": True,
                            "transport_authorization_id": item["use_contract_id"],
                            "audio_quality_status": (
                                "agent_verified"
                                if verification_basis == "agent_heard"
                                else "unassessed"
                            ),
                            "quality_authorization_id": (
                                item["use_contract_id"]
                                if verification_basis == "agent_heard"
                                else None
                            ),
                            "time_range_ms": [
                                representation.get("selector", {}).get("start_ms", 0),
                                representation.get("selector", {}).get("end_ms", 0),
                            ],
                        }
                        if modality == "audio"
                        else {}
                    ),
                }
            )
        registry_updates = copy.deepcopy(arguments.get("registry_updates") or [])
        new_tag_refs: set[str] = set()
        new_tag_categories: dict[str, str] = {}
        for update in registry_updates:
            category = str(update.get("category") or "")
            if category not in {"person", "entity", "scene", "object", "style"}:
                raise ToolRejected(
                    "invalid_registry_category",
                    "registry category must be person, entity, scene, object, or style",
                )
            unknown = set(update.get("evidence_client_refs") or []) - client_refs
            if unknown:
                raise ToolRejected("unknown_evidence_client_ref", f"registry update references unknown memories: {sorted(unknown)}")
            tag_ref = update.get("tag_ref") or {}
            if update.get("operation") == "refine":
                tag = str(tag_ref.get("tag") or "")
                if tag_ref.get("kind") != "existing_tag" or tag not in self.state.data["registry"]["items"]:
                    raise ToolRejected(
                        "invalid_registry_refine",
                        "refine requires an existing registry tag",
                    )
                prior_category = str(
                    self.state.data["registry"]["items"][tag].get("category") or ""
                )
                if category != prior_category:
                    raise ToolRejected(
                        "registry_category_changed",
                        "refine must preserve the existing registry category",
                    )
            else:
                if category == "person":
                    raise ToolRejected(
                        "person_append_forbidden",
                        "people must be predeclared by story PE; append may not create person tags",
                    )
                client_tag_ref = str(tag_ref.get("client_ref") or "")
                if tag_ref.get("kind") != "new_tag" or not client_tag_ref:
                    raise ToolRejected(
                        "invalid_registry_append",
                        "append requires a non-empty transaction-local new_tag reference",
                    )
                if client_tag_ref in new_tag_refs:
                    raise ToolRejected("duplicate_new_tag_ref", f"duplicate new tag ref: {client_tag_ref}")
                new_tag_refs.add(client_tag_ref)
                new_tag_categories[client_tag_ref] = category
        memories_by_client = {str(memory["client_ref"]): memory for memory in memories}
        for memory in memories:
            person_tag_refs: set[str] = set()
            entity_tag_ref_count = 0
            for tag_ref in memory.get("tag_refs") or []:
                if tag_ref.get("kind") == "existing_tag":
                    tag = str(tag_ref.get("tag") or "")
                    registry_row = self.state.data["registry"]["items"].get(tag)
                    if not registry_row:
                        raise ToolRejected(
                            "unknown_registry_tag",
                            f"unknown registry tag: {tag}",
                        )
                    if registry_row.get("category") == "person":
                        person_tag_refs.add(tag)
                    elif registry_row.get("category") == "entity":
                        entity_tag_ref_count += 1
                else:
                    new_ref = str(tag_ref.get("client_ref") or "")
                    if new_ref not in new_tag_refs:
                        raise ToolRejected(
                            "unknown_new_tag_ref",
                            f"memory references an undeclared new tag: {new_ref}",
                        )
                    if new_tag_categories.get(new_ref) == "entity":
                        entity_tag_ref_count += 1
            if person_tag_refs != set(memory.get("subject_ids") or []):
                raise ToolRejected(
                    "memory_person_tag_subject_mismatch",
                    "a memory's person tag_refs must exactly equal its source-validated subject_ids",
                    details={
                        "person_tag_refs": sorted(person_tag_refs),
                        "subject_ids": sorted(memory.get("subject_ids") or []),
                    },
                )
            if set(memory.get("roles") or []) & _ENTITY_MEMORY_ROLES:
                if entity_tag_ref_count == 0:
                    raise ToolRejected(
                        "entity_tag_evidence_required",
                        "entity identity/appearance memory requires an ENTITY_N tag backlink",
                    )

        for update in registry_updates:
            target_ref = update.get("tag_ref") or {}
            category = str(update.get("category") or "")
            for client_ref in update.get("evidence_client_refs") or []:
                memory = memories_by_client[str(client_ref)]
                if (
                    memory.get("modality") == "audio"
                    and memory.get("verification_basis")
                    == "trusted_user_declared"
                ):
                    raise ToolRejected(
                        "transport_only_audio_registry_evidence_forbidden",
                        "transport-only audio cannot refine stable registry descriptions or aliases",
                    )
                if target_ref not in (memory.get("tag_refs") or []):
                    raise ToolRejected(
                        "registry_evidence_tag_mismatch",
                        "every registry update evidence memory must link back to the exact target tag_ref",
                        details={"evidence_client_ref": client_ref},
                    )
                if category == "person":
                    target_tag = str(target_ref.get("tag") or "")
                    if target_tag not in set(memory.get("subject_ids") or []):
                        raise ToolRejected(
                            "registry_person_evidence_subject_mismatch",
                            "person registry evidence must carry the same source-validated subject ID",
                            details={
                                "tag": target_tag,
                                "evidence_client_ref": client_ref,
                            },
                        )
        transaction_id = self.state.next_id("memupd")
        transaction = {
            "memory_update_id": transaction_id,
            "video_selection_id": video_selection_id,
            "memories": memories,
            "registry_updates": registry_updates,
            "created_at": utc_now(),
        }
        self.shot["staged_memory_update"] = transaction
        return {
            "ok": True,
            "memory_update_id": transaction_id,
            "staged_memory_ids": [item["memory_id"] for item in memories],
            "staged_registry_updates": len(registry_updates),
            "message": "memory transaction staged; submit_shot is still required",
        }

    def _commit_memory_update(self, transaction: dict[str, Any]) -> None:
        next_bank = copy.deepcopy(self.state.data["memory_bank"])
        registry = copy.deepcopy(self.state.data["registry"])
        by_client = {item["client_ref"]: item for item in transaction["memories"]}
        new_tags: dict[str, str] = {}
        category_counts: dict[str, int] = {}
        for update in transaction["registry_updates"]:
            tag_ref = update["tag_ref"]
            if tag_ref["kind"] == "new_tag":
                category = str(update["category"])
                base_count = category_counts.setdefault(
                    category,
                    sum(1 for row in registry["items"].values() if row.get("category") == category),
                )
                category_counts[category] = base_count + 1
                allocated = f"{category.upper()}_{base_count + 1}"
                while allocated in registry["items"] or allocated in new_tags.values():
                    category_counts[category] += 1
                    allocated = f"{category.upper()}_{category_counts[category]}"
                new_tags[str(tag_ref["client_ref"])] = allocated

        def resolve_tag(tag_ref: Mapping[str, Any]) -> str:
            if tag_ref.get("kind") == "existing_tag":
                tag = str(tag_ref.get("tag") or "")
                if tag not in registry["items"]:
                    raise ToolRejected("unknown_registry_tag", f"unknown registry tag: {tag}")
                return tag
            client_ref = str(tag_ref.get("client_ref") or "")
            tag = new_tags.get(client_ref)
            if not tag:
                raise ToolRejected("unknown_new_tag_ref", f"unknown new tag ref: {client_ref}")
            return tag

        for memory in transaction["memories"]:
            stored = copy.deepcopy(memory)
            stored.pop("client_ref", None)
            tag_refs = stored.pop("tag_refs", [])
            stored["tags"] = list(dict.fromkeys(resolve_tag(tag_ref) for tag_ref in tag_refs))
            next_bank[memory["memory_id"]] = stored
        for update in transaction["registry_updates"]:
            tag_ref = update.get("tag_ref") or {}
            if tag_ref.get("kind") == "existing_tag":
                tag = str(tag_ref.get("tag") or "")
                if tag not in registry["items"]:
                    raise ToolRejected("unknown_registry_tag", f"unknown registry tag: {tag}")
            else:
                tag = new_tags[str(tag_ref["client_ref"])]
            evidence_ids = [by_client[item]["memory_id"] for item in update.get("evidence_client_refs") or []]
            prior = registry["items"].get(tag) or {}
            registry["items"][tag] = {
                "tag": tag,
                "category": str(update.get("category") or prior.get("category") or "item"),
                "stable_description": str(update.get("stable_description") or prior.get("stable_description") or ""),
                "aliases": list(dict.fromkeys((prior.get("aliases") or []) + (update.get("aliases") or []))),
                "evidence_memory_ids": list(dict.fromkeys((prior.get("evidence_memory_ids") or []) + evidence_ids)),
                "version": int(prior.get("version", 0)) + 1,
            }
        if transaction["memories"] or transaction["registry_updates"]:
            registry["version"] = int(registry.get("version", 0)) + 1
        self.state.data["memory_bank"] = next_bank
        self.state.data["registry"] = registry

    def _unfinished_generation_obligations(self) -> dict[str, Any]:
        open_groups = [
            str(group_id)
            for group_id, group in self.shot.get("generation_groups", {}).items()
            if not group.get("selection_id")
        ]
        unconsumed_prompts = [
            str(artifact_id)
            for artifact_id, artifact in self.shot.get("prompt_artifacts", {}).items()
            if not artifact.get("consumed")
        ]
        unregistered_receipts = [
            str(receipt_id)
            for receipt_id, receipt in self.shot.get("pe_receipts", {}).items()
            if not receipt.get("registered_prompt_artifact_id")
        ]
        uninspected_candidates = [
            str(candidate_id)
            for candidate_id, candidate in self.shot.get("candidates", {}).items()
            if not candidate.get("inspection_id")
        ]
        unassessed_candidates = [
            str(candidate_id)
            for candidate_id, candidate in self.shot.get("candidates", {}).items()
            if candidate.get("inspection_id") and not candidate.get("assessment_id")
        ]
        return {
            "open_generation_group_ids": open_groups,
            "unregistered_pe_receipt_ids": unregistered_receipts,
            "unconsumed_prompt_artifact_ids": unconsumed_prompts,
            "uninspected_candidate_ids": uninspected_candidates,
            "unassessed_candidate_ids": unassessed_candidates,
            "retry_generation_group_ids": [
                group_id for group_id, _ in self._retry_obligations()
            ],
            "provider_call_inflight": self.shot.get("inflight") is not None,
        }

    def tool_submit_shot(self, arguments: dict[str, Any]) -> dict[str, Any]:
        video_selection_id = str(arguments["video_selection_id"])
        if video_selection_id != self.shot.get("video_selection_id"):
            raise ToolRejected("wrong_video_selection", "submit_shot requires the selected final video")
        selection = self.shot["selections"][video_selection_id]
        candidate = self.shot["candidates"][selection["selected_candidate_id"]]
        obligations = self._unfinished_generation_obligations()
        if any(
            bool(value)
            for value in obligations.values()
        ):
            raise ToolRejected(
                "unfinished_generation_obligations",
                "all started groups, prompts, candidates, retries, and provider calls must be closed before submit_shot",
                details=obligations,
            )
        memory_update = arguments["memory_update"]
        status = str(memory_update["status"])
        staged = self.shot.get("staged_memory_update")
        if status == "updated":
            if not staged or memory_update.get("memory_update_id") != staged["memory_update_id"]:
                raise ToolRejected("invalid_memory_transaction", "updated status requires the staged transaction ID")
        elif staged:
            raise ToolRejected("staged_memory_not_committed", "a staged memory transaction cannot be skipped")
        unconsumed = [
            selection_id
            for selection_id, row in self.shot["selections"].items()
            if row.get("target") == "image" and selection_id not in self.shot["consumed_image_selection_ids"]
        ]
        if unconsumed:
            raise ToolRejected("unconsumed_image_selections", "selected image groups remain unconsumed", details={"selection_ids": unconsumed})
        inspection = self.shot["inspections"].get(candidate.get("inspection_id")) or {}
        tail = next(
            (
                copy.deepcopy(item)
                for item in inspection.get("derived_representations") or []
                if item.get("frame_role") == "tail"
                and str(item.get("media_type") or "").startswith("image")
            ),
            None,
        )
        if tail:
            tail["source_id"] = str(tail.get("representation_id") or "")
        commit = {
            "video_selection_id": video_selection_id,
            "selected_candidate_id": candidate["candidate_id"],
            "video_representation": copy.deepcopy(candidate["representation"]),
            "tail_representation": tail,
            "tail_handle": (
                {
                    "kind": "previous_tail",
                    "id": tail["source_id"],
                    "representation_id": tail["representation_id"],
                }
                if tail
                else None
            ),
            "memory_update": copy.deepcopy(memory_update),
            "summary": str(arguments["summary"]),
            "committed_at": utc_now(),
        }
        return {
            "ok": True,
            "done": True,
            "shot_index": self.shot["shot_index"],
            "video_selection_id": video_selection_id,
            "video_representation": {key: value for key, value in candidate["representation"].items() if key != "path"},
            "_commit_payload": commit,
            "_memory_transaction": copy.deepcopy(staged) if status == "updated" else None,
        }

    def _committed_shot_path(self) -> Path:
        return (
            self.config.output_dir
            / "shots"
            / f"shot_{int(self.shot['shot_index']):03d}.json"
        )

    def _write_committed_shot(self) -> None:
        from .trajectory import (
            build_collector_semantic_lock,
            build_trajectory_shard,
            build_upload_private_lineage,
        )

        shot = self.shot
        run_manifest = self.state.data.get("run_manifest", {})
        expected_lock = run_manifest.get("collector_semantic_lock")
        if not isinstance(expected_lock, Mapping):
            raise RuntimeError(
                "state has no collector semantic lock; legacy checkpoints cannot emit canonical shards"
            )
        current_lock = build_collector_semantic_lock(
            runtime_version=RUNTIME_VERSION,
            pe_manifest_sha256=self.pe_resolver.manifest_sha256,
            tokenizer_profile_sha256=str(
                getattr(self.prompt_validator, "tokenizer_profile_sha256", "")
            ),
        )
        if dict(expected_lock) != current_lock:
            raise RuntimeError(
                "collector semantics changed before shot commit; refusing to relabel prior events"
            )
        source_assets_receipt = copy.deepcopy(
            run_manifest.get("source_assets_receipt")
        )
        if (
            not isinstance(source_assets_receipt, Mapping)
            or source_assets_receipt.get("receipt_sha256")
            != run_manifest.get("source_assets_receipt_sha256")
        ):
            raise RuntimeError(
                "run manifest source asset receipt is missing or has a broken hash reference"
            )
        output = build_trajectory_shard(
            runtime_version=RUNTIME_VERSION,
            story_id=self.state.data["story_id"],
            shot_index=int(shot["shot_index"]),
            shot_request_id=str(shot["shot_request_id"]),
            raw_user_request=str(shot["raw_user_request"]),
            source_sha256=str(run_manifest["source_sha256"]),
            source_assets_receipt=source_assets_receipt,
            upload_private_lineage=build_upload_private_lineage(
                assets=self.state.data.get("assets") or {},
                source_assets_receipt=source_assets_receipt,
            ),
            story_plan_sha256=str(
                run_manifest["story_plan_sha256"]
            ),
            story_pe_provenance=copy.deepcopy(
                run_manifest["story_pe_provenance"]
            ),
            pe_manifest_sha256=self.pe_resolver.manifest_sha256,
            collector_semantic_lock=current_lock,
            events=shot["events"],
            teacher_calls=shot["teacher_calls"],
            prompt_artifacts=shot["prompt_artifacts"],
            pe_receipts=shot["pe_receipts"],
            selections=shot["selections"],
            prompt_rewrite_mode=self.config.prompt_rewrite_mode,
        )
        path = self._committed_shot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, output)
