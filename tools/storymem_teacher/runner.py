from __future__ import annotations

import copy
import json
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .pe import PESkillResolver
from .contract import PROMPT_REWRITE_MODES
from .runtime import RUNTIME_VERSION, RuntimeConfig, StoryMemRuntime
from .source import (
    bind_assets_to_story_plan,
    build_source_asset_receipt,
    ingest_source_assets,
    load_story_source,
    persist_source_asset_receipt,
    prepare_story,
    validate_source_asset_receipt,
    validate_story_pe_provenance,
)
from .state import StateStore
from .trajectory import (
    TrajectoryValidationError,
    build_collector_semantic_lock,
    build_teacher_input_messages,
    parse_plain_json_tool_command,
)
from .util import atomic_write_json, canonical_json, safe_name, sha256_json, sha256_text, utc_now
from .validation import PromptValidator


class MalformedTeacherResponse(ValueError):
    """A paid teacher turn that did not yield one usable JSON tool command."""

    def __init__(self, message: str, *, provenance: Mapping[str, Any] | None = None):
        self.provenance = copy.deepcopy(dict(provenance or {}))
        super().__init__(message)


@dataclass(frozen=True)
class RunConfig:
    output_root: Path
    pe_manifest: Path
    tokenizer_path: Path
    story_pe_mode: str = "always"
    story_temperature: float = 0.7
    story_semantic_max_attempts: int = 3
    tool_temperature: float = 0.25
    max_turns_per_shot: int = 40
    max_hidden_turn_retries: int = 4
    image_attempts: int = 2
    video_attempts: int = 2
    width: int = 1280
    height: int = 736
    num_frames: int = 241
    max_memory_slots: int = 7
    audio_transport: str = "disabled"
    audio_perception: str = "disabled"
    prompt_rewrite_mode: str = "rewrite_enabled"
    teacher_media_limit: int = 64
    collector_semantic_lock: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        positive = {
            "max_turns_per_shot": self.max_turns_per_shot,
            "max_hidden_turn_retries": self.max_hidden_turn_retries,
            "story_semantic_max_attempts": self.story_semantic_max_attempts,
            "image_attempts": self.image_attempts,
            "video_attempts": self.video_attempts,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "max_memory_slots": self.max_memory_slots,
        }
        invalid = [name for name, value in positive.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"RunConfig values must be positive: {', '.join(invalid)}")
        if not 1 <= int(self.teacher_media_limit) <= 256:
            raise ValueError("teacher_media_limit must be between 1 and 256")
        if self.max_memory_slots > 7:
            raise ValueError("Echo 1.5 accepts at most seven memory slots")
        if self.story_pe_mode != "always":
            raise ValueError("canonical story_pe_mode must be always")
        if self.audio_perception not in {"disabled", "verified"}:
            raise ValueError("audio_perception must be disabled or verified")
        if self.audio_transport not in {"disabled", "enabled"}:
            raise ValueError("audio_transport must be disabled or enabled")
        if self.prompt_rewrite_mode not in PROMPT_REWRITE_MODES:
            raise ValueError(
                "prompt_rewrite_mode must be one of: "
                + ", ".join(PROMPT_REWRITE_MODES)
            )
        if not 0.0 <= float(self.story_temperature) <= 2.0:
            raise ValueError("story_temperature must be between 0 and 2")
        if not 0.0 <= float(self.tool_temperature) <= 2.0:
            raise ValueError("tool_temperature must be between 0 and 2")


def _plain_teacher_result(value: Any) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if isinstance(value, dict):
        call = value.get("tool_call") or value.get("call")
        reason = str(value.get("reason") or value.get("reasoning") or "")
        provenance = dict(value.get("provenance") or {})
    else:
        call = getattr(value, "tool_call", None) or getattr(value, "call", None)
        reason = str(getattr(value, "reason", "") or getattr(value, "reasoning", ""))
        provenance = dict(getattr(value, "provenance", {}) or {})
    if hasattr(call, "to_dict"):
        call = call.to_dict()
    elif hasattr(call, "__dict__"):
        call = vars(call)
    if not isinstance(call, dict):
        raise ValueError("teacher response contains no tool call")
    if "function" in call and isinstance(call["function"], dict):
        function = call["function"]
        raw_arguments = function.get("arguments") or {}
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        call = {"name": function.get("name"), "arguments": arguments}
    if "args" in call and "arguments" not in call:
        call["arguments"] = call.pop("args")
    if not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
        raise ValueError("teacher tool call must contain string name and object arguments")
    normalized_call = {"name": call["name"], "arguments": call["arguments"]}
    raw_text = provenance.get("raw_assistant_text")
    if not isinstance(raw_text, str):
        raise ValueError("teacher response provenance contains no raw assistant JSON text")
    try:
        parsed_call, parsed_reason = parse_plain_json_tool_command(raw_text)
    except TrajectoryValidationError as exc:
        raise ValueError(str(exc)) from exc
    if parsed_call != normalized_call:
        raise ValueError("raw assistant JSON differs from the parsed tool call")
    if reason and reason != parsed_reason:
        raise ValueError("teacher result reason differs from raw assistant JSON")
    return normalized_call, parsed_reason, provenance


def _phase(state: dict[str, Any]) -> str:
    shot = state["current_shot"]
    if shot["phase"] == "need_registry":
        return "discovery"
    if shot.get("pending_pe_receipt_id"):
        receipt = shot["pe_receipts"][shot["pending_pe_receipt_id"]]
        return "image_preparation" if receipt["target"] == "image" else "video_preparation"
    if shot.get("pending_prompt_artifact_id"):
        artifact = shot["prompt_artifacts"][shot["pending_prompt_artifact_id"]]
        return "image_preparation" if artifact["target"] == "image" else "video_preparation"
    candidate_id = shot.get("pending_candidate_id")
    if candidate_id:
        candidate = shot["candidates"][candidate_id]
        return "image_assessment" if candidate["target"] == "image" else "video_assessment"
    if shot.get("video_selection_id"):
        return "finalization"
    assessments = list(shot.get("assessments", {}).values())
    if assessments and assessments[-1].get("next_action") == "retry":
        candidate = shot["candidates"][assessments[-1]["candidate_id"]]
        return "image_assessment" if candidate["target"] == "image" else "video_assessment"
    return "video_preparation"


def _teacher_messages(
    runtime: StoryMemRuntime,
    media_limit: int,
) -> list[dict[str, Any]]:
    """Build the ordinary JSON dialogue for the next stateless Teacher turn.

    The closed endpoint never receives native tool calls or tool-result roles.
    It sees its exact prior JSON text followed by a USER-role observation, with
    all returned media kept adjacent to that observation.  Native tool syntax
    is introduced only by the deterministic SFT exporter.
    """

    committed_events = [
        event
        for event in runtime.shot.get("events") or []
        if event.get("committed_to_training")
    ]
    messages = build_teacher_input_messages(
        raw_user_request=str(runtime.shot["raw_user_request"]),
        events=committed_events,
    )

    media_count = sum(
        1
        for message in messages
        for block in (
            message.get("content") if isinstance(message.get("content"), list) else []
        )
        if isinstance(block, Mapping) and block.get("type") == "media"
    )
    if media_count > int(media_limit):
        raise RuntimeError(
            "closed teacher JSON dialogue requires "
            f"{media_count} media blocks but teacher_media_limit={media_limit}; "
            "increase --teacher-media-limit and resume (no observation was omitted)"
        )

    return messages


def _teacher_call(
    teacher_client: Any,
    *,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    from .providers import ProviderProtocolError

    try:
        result = teacher_client.call(
            system_prompt=system_prompt,
            user_content=messages,
            tools=tools,
            temperature=temperature,
        )
    except ProviderProtocolError as exc:
        raise MalformedTeacherResponse(
            str(exc), provenance=getattr(exc, "provenance", None)
        ) from exc
    try:
        return _plain_teacher_result(result)
    except (TypeError, ValueError) as exc:
        provenance = (
            result.get("provenance")
            if isinstance(result, Mapping)
            else getattr(result, "provenance", None)
        )
        raise MalformedTeacherResponse(str(exc), provenance=provenance) from exc


def _seed_registry(state: StateStore, story_plan: dict[str, Any]) -> None:
    registry = state.data["registry"]
    if registry["items"]:
        return
    for index, item in enumerate(story_plan.get("materials") or [], start=1):
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag") or f"ITEM_{index}")
        registry["items"][tag] = {
            "tag": tag,
            "category": str(item.get("category") or "item"),
            "stable_description": str(item.get("description") or ""),
            "aliases": list(item.get("aliases") or []),
            "evidence_memory_ids": [],
            "evidence_kind": "story_declared",
            "version": 1,
        }
    registry["version"] = 1 if registry["items"] else 0
    state.save()


def _create_or_resume_state(
    *,
    story_dir: Path,
    source: dict[str, Any],
    story_plan: dict[str, Any],
    story_pe_provenance: dict[str, Any],
    run_config: RunConfig,
    resolver: PESkillResolver,
    assets: dict[str, Any],
    source_assets_receipt: dict[str, Any],
    prompt_validator: PromptValidator,
) -> StateStore:
    story_resource_bundle = [
        {key: item[key] for key in ("resource_id", "revision", "sha256")}
        for item in resolver.story_preparation_bundle()
    ]
    story_plan_sha256 = sha256_json(story_plan)
    validated_story_pe_provenance = validate_story_pe_provenance(
        story_pe_provenance,
        expected_story_plan_sha256=story_plan_sha256,
        expected_pe_manifest_sha256=resolver.manifest_sha256,
        expected_resource_bundle=story_resource_bundle,
    )
    current_semantic_lock = build_collector_semantic_lock(
        runtime_version=RUNTIME_VERSION,
        pe_manifest_sha256=resolver.manifest_sha256,
        tokenizer_profile_sha256=prompt_validator.tokenizer_profile_sha256,
    )
    configured_semantic_lock = (
        copy.deepcopy(run_config.collector_semantic_lock)
        if run_config.collector_semantic_lock is not None
        else current_semantic_lock
    )
    if configured_semantic_lock != current_semantic_lock:
        raise RuntimeError(
            "collector semantics changed after launch; refusing to create or resume a story"
        )
    semantic_config = {
        # These values affect model-visible context, provider inputs, or the
        # number/type of candidates that may enter the trajectory.  They are
        # therefore immutable once state.json exists.  Scheduling-only knobs
        # such as turn/rejection limits intentionally remain resumable.
        "story_pe_mode": run_config.story_pe_mode,
        "story_temperature": float(run_config.story_temperature),
        "tool_temperature": float(run_config.tool_temperature),
        "image_attempts": int(run_config.image_attempts),
        "video_attempts": int(run_config.video_attempts),
        "width": int(run_config.width),
        "height": int(run_config.height),
        "num_frames": int(run_config.num_frames),
        "max_memory_slots": int(run_config.max_memory_slots),
        "audio_transport": run_config.audio_transport,
        "audio_perception": run_config.audio_perception,
        "prompt_rewrite_mode": run_config.prompt_rewrite_mode,
        "tokenizer_path": (
            str(prompt_validator.tokenizer_path)
            if prompt_validator.tokenizer_path is not None
            else None
        ),
        "tokenizer_profile_sha256": prompt_validator.tokenizer_profile_sha256,
        "collector_semantic_lock_sha256": current_semantic_lock[
            "semantic_lock_sha256"
        ],
    }
    state_path = story_dir / "state.json"
    source_sha256 = sha256_json(
        {key: value for key, value in source.items() if key != "_source_path"}
    )
    validated_source_assets_receipt = validate_source_asset_receipt(
        source_assets_receipt,
        expected_source_sha256=source_sha256,
    )
    run_manifest = {
        "runtime": RUNTIME_VERSION,
        "source_sha256": source_sha256,
        "story_plan_sha256": story_plan_sha256,
        "story_pe_provenance": validated_story_pe_provenance,
        "story_pe_provenance_sha256": sha256_json(
            validated_story_pe_provenance
        ),
        "story_pe_resource_bundle_sha256": validated_story_pe_provenance[
            "resource_bundle_sha256"
        ],
        "pe_manifest_sha256": resolver.manifest_sha256,
        "assets_sha256": sha256_json(assets),
        "source_assets_receipt": validated_source_assets_receipt,
        "source_assets_receipt_sha256": validated_source_assets_receipt[
            "receipt_sha256"
        ],
        "semantic_config": semantic_config,
        "semantic_config_sha256": sha256_json(semantic_config),
        "collector_semantic_lock": current_semantic_lock,
        "collector_semantic_lock_sha256": current_semantic_lock[
            "semantic_lock_sha256"
        ],
        "run_config": {**asdict(run_config), "output_root": str(run_config.output_root), "pe_manifest": str(run_config.pe_manifest), "tokenizer_path": str(run_config.tokenizer_path)},
    }
    if state_path.is_file():
        state = StateStore.load(state_path)
        current = state.data.get("run_manifest") or {}
        for key in (
            "source_sha256",
            "story_plan_sha256",
            "story_pe_provenance_sha256",
            "story_pe_resource_bundle_sha256",
            "pe_manifest_sha256",
            "assets_sha256",
            "source_assets_receipt_sha256",
            "semantic_config_sha256",
            "collector_semantic_lock_sha256",
        ):
            if current.get(key) != run_manifest.get(key):
                raise RuntimeError(f"cannot resume {story_plan['story_id']}: {key} changed")
        if current.get("collector_semantic_lock") != current_semantic_lock:
            raise RuntimeError(
                f"cannot resume {story_plan['story_id']}: collector_semantic_lock changed"
            )
        if current.get("source_assets_receipt") != validated_source_assets_receipt:
            raise RuntimeError(
                f"cannot resume {story_plan['story_id']}: source_assets_receipt changed"
            )
        return state
    state = StateStore.create(
        state_path,
        story_id=story_plan["story_id"],
        source={key: value for key, value in source.items() if key != "_source_path"},
        story_plan=story_plan,
        run_manifest=run_manifest,
        assets=assets,
    )
    _seed_registry(state, story_plan)
    return state


def run_story(
    *,
    source_path: str | Path,
    config: RunConfig,
    teacher_client: Any,
    image_provider: Any,
    video_provider: Any,
) -> dict[str, Any]:
    from .contract import get_tool_schemas
    from .prompts import build_teacher_system_prompt

    started = time.time()
    source = load_story_source(source_path)
    source_story_id = safe_name(source.get("story_id") or Path(source_path).stem, "story")
    story_dir = config.output_root / source_story_id
    story_dir.mkdir(parents=True, exist_ok=True)
    resolver = PESkillResolver(config.pe_manifest)
    prepared = prepare_story(
        source=source,
        resolver=resolver,
        teacher_client=teacher_client,
        cache_path=story_dir / "story_plan.json",
        mode=config.story_pe_mode,
        temperature=config.story_temperature,
        semantic_max_attempts=config.story_semantic_max_attempts,
    )
    if prepared.plan["story_id"] != source_story_id:
        raise RuntimeError("validated story PE plan has an unexpected story_id")
    source_assets_dir = story_dir / "source_assets"
    assets, uploads_by_shot = ingest_source_assets(
        source, cache_dir=source_assets_dir
    )
    assets = bind_assets_to_story_plan(
        source=source,
        story_plan=prepared.plan,
        assets=assets,
    )
    source_assets_receipt = build_source_asset_receipt(
        source=source,
        assets=assets,
        uploads_by_shot=uploads_by_shot,
        cache_dir=source_assets_dir,
    )
    persist_source_asset_receipt(
        source_assets_dir / "receipt.json", source_assets_receipt
    )
    validator = PromptValidator(config.tokenizer_path)
    state = _create_or_resume_state(
        story_dir=story_dir,
        source=source,
        story_plan=prepared.plan,
        story_pe_provenance=prepared.provenance,
        run_config=config,
        resolver=resolver,
        assets=assets,
        source_assets_receipt=source_assets_receipt,
        prompt_validator=validator,
    )
    runtime = StoryMemRuntime(
        state=state,
        config=RuntimeConfig(
            output_dir=story_dir,
            width=config.width,
            height=config.height,
            num_frames=config.num_frames,
            image_attempts=config.image_attempts,
            video_attempts=config.video_attempts,
            max_memory_slots=config.max_memory_slots,
            audio_transport=config.audio_transport,
            audio_perception=config.audio_perception,
            prompt_rewrite_mode=config.prompt_rewrite_mode,
        ),
        pe_resolver=resolver,
        prompt_validator=validator,
        image_provider=image_provider,
        video_provider=video_provider,
    )
    completed_indices = {int(item["shot_index"]) for item in state.data.get("committed_shots") or []}
    for shot_plan in prepared.plan["shots"]:
        shot_index = int(shot_plan["shot_index"])
        if shot_index in completed_indices:
            continue
        if state.data.get("current_shot") is None:
            state.start_shot(
                shot_index=shot_index,
                raw_user_request=shot_plan["raw_request"],
                uploads=uploads_by_shot.get(shot_index, []),
                budgets={
                    "image_attempts": config.image_attempts,
                    "video_attempts": config.video_attempts,
                },
            )
        elif int(state.shot["shot_index"]) != shot_index:
            raise RuntimeError("checkpoint active shot differs from the next uncommitted story shot")
        hidden_rejections: list[dict[str, Any]] = []
        successful_turns = len(state.shot.get("events") or [])
        while state.data.get("current_shot") is not None:
            if successful_turns >= config.max_turns_per_shot:
                raise RuntimeError(f"shot {shot_index} exceeded {config.max_turns_per_shot} committed turns")
            phase = _phase(state.data)
            # Keep operational budgets and hidden state out of the Teacher's
            # semantic decision.  Phase is advisory scaffolding; all shot facts
            # and generation budgets must arrive through real tool results.
            runtime_context: dict[str, Any] = {
                "phase": phase,
                "prompt_rewrite_mode": config.prompt_rewrite_mode,
            }
            if hidden_rejections:
                # Private repair scaffolding belongs to this one dynamic system
                # turn.  It must never create a second human USER request or
                # enter the committed open-model trajectory.
                runtime_context["private_retry_feedback"] = copy.deepcopy(
                    hidden_rejections
                )
            inflight = runtime.pending_provider_call()
            pending_teacher = state.pending_teacher_call()
            resumed_inflight = inflight is not None
            if inflight is not None:
                if (
                    pending_teacher is None
                    or pending_teacher.get("tool_call") != inflight.get("tool_call")
                    or pending_teacher.get("assistant_provenance")
                    != inflight.get("assistant_provenance")
                ):
                    raise RuntimeError(
                        "provider inflight record is not paired with the one pending "
                        "Teacher command"
                    )
                call = copy.deepcopy(inflight["tool_call"])
                provenance = copy.deepcopy(inflight["assistant_provenance"])
            else:
                if pending_teacher is not None:
                    call = copy.deepcopy(pending_teacher["tool_call"])
                    provenance = copy.deepcopy(
                        pending_teacher["assistant_provenance"]
                    )
                    currently_allowed = runtime.teacher_allowed_tool_names()
                    if call.get("name") not in currently_allowed:
                        raise RuntimeError(
                            "persisted Teacher command is no longer admitted by "
                            "the unchanged runtime checkpoint"
                        )
                else:
                    allowed_tool_names = runtime.teacher_allowed_tool_names()
                    if not allowed_tool_names:
                        raise RuntimeError(
                            f"shot {shot_index} has no Teacher action admitted by the runtime"
                        )
                    allowed_tool_schemas = get_tool_schemas(allowed_tool_names)
                    system_prompt = build_teacher_system_prompt(
                        phase,
                        allowed_tool_names=allowed_tool_names,
                        runtime_context=runtime_context,
                    )
                    teacher_messages = _teacher_messages(
                        runtime, config.teacher_media_limit
                    )
                    try:
                        call, reason, provenance = _teacher_call(
                            teacher_client,
                            system_prompt=system_prompt,
                            messages=teacher_messages,
                            tools=allowed_tool_schemas,
                            temperature=config.tool_temperature,
                        )
                    except MalformedTeacherResponse as exc:
                        malformed_provenance = copy.deepcopy(exc.provenance)
                        malformed_provenance.update(
                            {
                                "assistant_source": "remote_closed_model_api",
                                "dynamic_system_prompt_sha256": sha256_text(
                                    system_prompt
                                ),
                                "teacher_input_messages_sha256": sha256_json(
                                    teacher_messages
                                ),
                                "teacher_input_message_count": len(teacher_messages),
                                "teacher_history_event_count": len(
                                    state.shot.get("events") or []
                                ),
                                "private_retry_count": len(hidden_rejections),
                                "teacher_allowed_tools": list(allowed_tool_names),
                                "teacher_runtime_context": copy.deepcopy(
                                    runtime_context
                                ),
                                "phase": phase,
                                "teacher_reason": "",
                            }
                        )
                        state.record_teacher_call(
                            response_status="malformed_response",
                            provenance=malformed_provenance,
                            error={
                                "error_type": type(exc.__cause__ or exc).__name__,
                                "message": str(exc),
                            },
                        )
                        hidden_rejections.append(
                            {
                                "teacher_response_error": {
                                    "error_type": type(exc.__cause__ or exc).__name__,
                                    "message": str(exc),
                                    "provenance": malformed_provenance,
                                },
                                "instruction": (
                                    "Return exactly one strict JSON object shaped as "
                                    "{reason,tool_call:{name,arguments}}. "
                                    "This private retry is not inserted into the training trajectory."
                                ),
                            }
                        )
                        if len(hidden_rejections) >= config.max_hidden_turn_retries:
                            raise RuntimeError(
                                "teacher failed to return a usable JSON tool command after "
                                f"{len(hidden_rejections)} hidden retries: {exc}"
                            ) from exc
                        continue
                    provenance.update(
                        {
                            "assistant_source": "remote_closed_model_api",
                            "dynamic_system_prompt_sha256": sha256_text(system_prompt),
                            "teacher_input_messages_sha256": sha256_json(
                                teacher_messages
                            ),
                            "teacher_input_message_count": len(teacher_messages),
                            "teacher_history_event_count": len(
                                state.shot.get("events") or []
                            ),
                            "private_retry_count": len(hidden_rejections),
                            "teacher_allowed_tools": list(allowed_tool_names),
                            "teacher_runtime_context": copy.deepcopy(runtime_context),
                            "phase": phase,
                            "teacher_reason": reason,
                        }
                    )
                    teacher_call_audit_id = state.record_teacher_call(
                        response_status="plain_json_tool_command",
                        provenance=provenance,
                        tool_call=call,
                        teacher_reason=reason,
                    )
                    provenance["teacher_call_audit_id"] = teacher_call_audit_id
            observation = runtime.execute(
                call["name"], call["arguments"], assistant_provenance=provenance
            )
            if not observation.get("ok"):
                if resumed_inflight:
                    raise RuntimeError(
                        "persisted provider call could not be resumed: "
                        + canonical_json(observation)
                    )
                audit_id = provenance.get("teacher_call_audit_id")
                if isinstance(audit_id, str):
                    state.reject_teacher_call(audit_id, rejection=observation)
                hidden_rejections.append(
                    {
                        "rejected_tool_call": call,
                        "error": observation,
                        "instruction": "Correct the call yourself. The rejected call is audit-only and will not be inserted into the training trajectory.",
                    }
                )
                if len(hidden_rejections) >= config.max_hidden_turn_retries:
                    raise RuntimeError(
                        f"teacher failed to produce a valid tool call after {len(hidden_rejections)} hidden retries: {observation}"
                    )
                continue
            hidden_rejections.clear()
            successful_turns += 1
            if observation.get("done"):
                break
    story_result = {
        "story_id": source_story_id,
        "status": "complete",
        "num_shots": len(state.data.get("committed_shots") or []),
        "story_plan_sha256": state.data["run_manifest"]["story_plan_sha256"],
        "story_pe_provenance_sha256": state.data["run_manifest"][
            "story_pe_provenance_sha256"
        ],
        "story_pe_resource_bundle_sha256": state.data["run_manifest"][
            "story_pe_provenance"
        ]["resource_bundle_sha256"],
        "state_path": str(state.path),
        "story_plan_path": str(story_dir / "story_plan.json"),
        "shot_paths": [str(path) for path in sorted((story_dir / "shots").glob("shot_*.json"))],
        "elapsed_seconds": round(time.time() - started, 3),
        "completed_at": utc_now(),
    }
    atomic_write_json(story_dir / "complete.json", story_result)
    return story_result


def _validate_manifest_prompt_rewrite_mode(
    manifest: Any,
    records: list[Any],
    *,
    expected_prompt_rewrite_mode: str,
) -> None:
    """Bind a mode-bearing input manifest to the requested collector mode.

    Ordinary path manifests remain supported.  Once either the manifest or a
    record declares ``intended_prompt_rewrite_mode`` (as continuation cohort
    manifests do), both levels must declare the same exact canonical mode.
    This also rejects feeding the mixed, pre-partition continuation manifest
    directly to one collector invocation.
    """

    if expected_prompt_rewrite_mode not in PROMPT_REWRITE_MODES:
        raise ValueError(
            "expected_prompt_rewrite_mode must be one of: "
            + ", ".join(PROMPT_REWRITE_MODES)
        )
    top_level_declares_mode = isinstance(manifest, dict) and (
        "intended_prompt_rewrite_mode" in manifest
    )
    record_declares_mode = any(
        isinstance(item, dict) and "intended_prompt_rewrite_mode" in item
        for item in records
    )
    continuation_schema = (
        isinstance(manifest, dict)
        and manifest.get("schema_version")
        == "storymem.canonical.continuation-source.v1"
    )
    if not (top_level_declares_mode or record_declares_mode or continuation_schema):
        return

    if not isinstance(manifest, dict):
        raise ValueError("mode-bearing input manifest must be a JSON object")
    declared = manifest.get("intended_prompt_rewrite_mode")
    if declared != expected_prompt_rewrite_mode:
        raise ValueError(
            "input manifest intended_prompt_rewrite_mode does not match "
            f"--prompt-rewrite-mode: {declared!r} != "
            f"{expected_prompt_rewrite_mode!r}"
        )
    for index, item in enumerate(records):
        record_mode = (
            item.get("intended_prompt_rewrite_mode")
            if isinstance(item, dict)
            else None
        )
        if record_mode != expected_prompt_rewrite_mode:
            raise ValueError(
                f"input manifest record {index} intended_prompt_rewrite_mode "
                "does not match --prompt-rewrite-mode: "
                f"{record_mode!r} != {expected_prompt_rewrite_mode!r}"
            )


def load_manifest_sources(
    path: str | Path,
    *,
    expected_prompt_rewrite_mode: str | None = None,
) -> list[Path]:
    manifest_path = Path(path).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = value.get("records") if isinstance(value, dict) else value
    if not isinstance(records, list):
        raise ValueError("input manifest must be an array or an object with records[]")
    if expected_prompt_rewrite_mode is not None:
        _validate_manifest_prompt_rewrite_mode(
            value,
            records,
            expected_prompt_rewrite_mode=expected_prompt_rewrite_mode,
        )
    result: list[Path] = []
    for item in records:
        raw = item.get("path") if isinstance(item, dict) else item
        source = Path(str(raw or ""))
        if not source.is_absolute():
            workspace_candidate = (Path.cwd() / source).resolve()
            manifest_candidate = (manifest_path.parent / source).resolve()
            source = workspace_candidate if workspace_candidate.is_file() else manifest_candidate
        if not source.is_file():
            raise FileNotFoundError(f"story source is missing: {raw}")
        result.append(source)
    resolved = [str(item.resolve()) for item in result]
    if len(resolved) != len(set(resolved)):
        raise ValueError("input manifest contains duplicate source paths")
    return result


def validate_source_diversity(paths: Iterable[Path]) -> dict[str, Any]:
    fingerprints: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    lengths: list[int] = []
    for path in paths:
        source = load_story_source(path)
        source_copy = {key: value for key, value in source.items() if key != "_source_path"}
        fingerprints[sha256_json(source_copy)] += 1
        modes[str(source.get("mode") or source.get("metadata", {}).get("mode") or "unspecified")] += 1
        language = str(source.get("caption_language") or source.get("metadata", {}).get("caption_language") or "unspecified")
        languages[language] += 1
        shots = source.get("shots") or source.get("prompts") or []
        lengths.append(len(shots) if isinstance(shots, list) else 0)
    duplicates = sum(count - 1 for count in fingerprints.values() if count > 1)
    if duplicates:
        raise ValueError(f"source collection contains {duplicates} exact duplicate stories")
    return {
        "story_count": len(lengths),
        "shot_count": sum(lengths),
        "min_shots": min(lengths) if lengths else 0,
        "max_shots": max(lengths) if lengths else 0,
        "mode_counts": dict(sorted(modes.items())),
        "language_counts": dict(sorted(languages.items())),
        "exact_duplicate_count": duplicates,
    }


def audit_prepared_story_diversity(
    prepared: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Audit the story-PE outputs before any image or video generation starts.

    This is intentionally deterministic.  It rejects duplicate complete plans
    (which otherwise waste expensive media calls) and reports repeated shot
    requests without trying to replace semantic diversity with a heuristic
    score.
    """

    audit = _PreparedStoryDiversityAudit()
    for row in prepared:
        audit.add(row)
    return audit.snapshot(reject_duplicate_plans=True)


class _PreparedStoryDiversityAudit:
    """Incremental prepared-plan audit used by the long-running batch.

    Recomputing all prior plan and shot hashes after every story makes a 12k
    run quadratic and, because it happens under the preparation lock, blocks
    every worker.  This accumulator preserves the exact report fields while
    making each accepted plan O(number of shots in that plan).
    """

    def __init__(self) -> None:
        self.plan_fingerprints: Counter[str] = Counter()
        self.shot_fingerprints: Counter[str] = Counter()
        self.languages: Counter[str] = Counter()
        self.story_count = 0
        self.shot_count = 0
        self.min_shots: int | None = None
        self.max_shots = 0

    def add(self, row: Mapping[str, Any]) -> None:
        plan = row["plan"]
        requests = [str(item["raw_request"]) for item in plan["shots"]]
        self.plan_fingerprints[_prepared_plan_fingerprint(plan)] += 1
        for request in requests:
            self.shot_fingerprints[sha256_text(request.casefold())] += 1
        self.languages[str(plan["caption_language"])] += 1
        length = len(requests)
        self.story_count += 1
        self.shot_count += length
        self.min_shots = length if self.min_shots is None else min(self.min_shots, length)
        self.max_shots = max(self.max_shots, length)

    def snapshot(self, *, reject_duplicate_plans: bool) -> dict[str, Any]:
        plan_fingerprints = self.plan_fingerprints
        shot_fingerprints = self.shot_fingerprints
        languages = self.languages
        duplicate_plans = sum(
            count - 1 for count in plan_fingerprints.values() if count > 1
        )
        if reject_duplicate_plans and duplicate_plans:
            raise ValueError(
                f"story PE produced {duplicate_plans} duplicate complete story plan(s)"
            )
        repeated_shot_instances = sum(
            count - 1 for count in shot_fingerprints.values() if count > 1
        )
        return {
            "schema_version": "storymem.canonical.story_pe_audit.v1",
            "story_count": self.story_count,
            "shot_count": self.shot_count,
            "min_shots": self.min_shots or 0,
            "max_shots": self.max_shots,
            "caption_language_counts": dict(sorted(languages.items())),
            "duplicate_complete_plan_count": duplicate_plans,
            "repeated_exact_shot_instance_count": repeated_shot_instances,
            "unique_exact_shot_count": len(shot_fingerprints),
            "created_at": utc_now(),
        }


def _prepared_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    """Fingerprint semantic shot content for pre-media duplicate rejection."""

    requests = [str(item["raw_request"]) for item in plan.get("shots") or []]
    return sha256_json([item.casefold() for item in requests])


def _source_story_id(path: Path) -> str:
    source = load_story_source(path)
    return safe_name(source.get("story_id") or path.stem, "story")


def _prepare_batch_source(
    *,
    index: int,
    source_path: Path,
    config: RunConfig,
    teacher_client: Any,
) -> dict[str, Any]:
    source = load_story_source(source_path)
    story_id = safe_name(source.get("story_id") or source_path.stem, "story")
    story_dir = config.output_root / story_id
    story_dir.mkdir(parents=True, exist_ok=True)
    resolver = PESkillResolver(config.pe_manifest)
    prepared = prepare_story(
        source=source,
        resolver=resolver,
        teacher_client=teacher_client,
        cache_path=story_dir / "story_plan.json",
        mode=config.story_pe_mode,
        temperature=config.story_temperature,
        semantic_max_attempts=config.story_semantic_max_attempts,
    )
    plan = copy.deepcopy(prepared.plan)
    if plan["story_id"] != story_id:
        raise RuntimeError("validated story PE plan has an unexpected story_id")
    return {
        "index": index,
        "source": str(source_path),
        "story_id": story_id,
        "plan": plan,
        "provenance": copy.deepcopy(prepared.provenance),
    }


def run_batch(
    *,
    sources: list[Path],
    config: RunConfig,
    workers: int,
    component_factory: Callable[[int], tuple[Any, Any, Any]],
    max_in_flight: int | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if not sources:
        raise ValueError("sources must contain at least one story")
    config.output_root.mkdir(parents=True, exist_ok=True)
    diversity = validate_source_diversity(sources)
    story_ids = [_source_story_id(path) for path in sources]
    duplicate_story_ids = sorted(
        story_id for story_id, count in Counter(story_ids).items() if count > 1
    )
    if duplicate_story_ids:
        raise ValueError(
            "story sources resolve to duplicate output IDs: "
            + ", ".join(duplicate_story_ids[:20])
        )
    started = time.time()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    lock = threading.Lock()
    summary_path = config.output_root / "progress.jsonl"
    bound = max(1, int(max_in_flight or workers * 2))

    # PE and media are one bounded pipeline.  As soon as one story has a valid,
    # non-duplicate plan it may start its shot loop; a slow PE call elsewhere
    # never holds all Echo workers idle.  The coordinator claims complete-plan
    # fingerprints before media dispatch, so a duplicate discovered later is
    # rejected without invalidating media already produced by the first claim.
    prepared_audit = _PreparedStoryDiversityAudit()
    claimed_plan_fingerprints: dict[str, str] = {}
    preparation_lock = threading.Lock()

    def task(index: int, source: Path) -> dict[str, Any]:
        try:
            teacher, image, video = component_factory(index)
            prepared = _prepare_batch_source(
                index=index,
                source_path=source,
                config=config,
                teacher_client=teacher,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one story
            return {
                "story_id": story_ids[index],
                "status": "failed",
                "stage": "story_pe",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source": str(source),
            }

        fingerprint = _prepared_plan_fingerprint(prepared["plan"])
        with preparation_lock:
            first_story_id = claimed_plan_fingerprints.get(fingerprint)
            if first_story_id is not None:
                return {
                    "story_id": prepared["story_id"],
                    "status": "failed",
                    "stage": "story_pe_diversity",
                    "error_type": "DuplicatePreparedStoryPlan",
                    "error": (
                        "story PE produced a complete plan already claimed by "
                        f"{first_story_id}; media generation was not started"
                    ),
                    "duplicate_of_story_id": first_story_id,
                    "plan_fingerprint": fingerprint,
                    "source": str(source),
                }
            claimed_plan_fingerprints[fingerprint] = str(prepared["story_id"])
            prepared_audit.add(prepared)
            # This incremental report is useful while a long batch is still
            # running and remains valid even if the process is interrupted.
            atomic_write_json(
                config.output_root / "story_pe_audit.json",
                prepared_audit.snapshot(reject_duplicate_plans=False),
            )

        try:
            return run_story(
                source_path=source,
                config=config,
                teacher_client=teacher,
                image_provider=image,
                video_provider=video,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one story
            return {
                "story_id": prepared["story_id"],
                "status": "failed",
                "stage": "shot_runtime",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source": str(source),
            }

    pending: dict[Future[dict[str, Any]], tuple[int, Path]] = {}
    next_index = 0
    with summary_path.open("a", encoding="utf-8") as progress, ThreadPoolExecutor(max_workers=workers) as executor:
        for failure in failures:
            progress.write(json.dumps({**failure, "time": utc_now()}, ensure_ascii=False) + "\n")
        progress.flush()
        while next_index < len(sources) or pending:
            while next_index < len(sources) and len(pending) < bound:
                source_index = next_index
                source = sources[source_index]
                future = executor.submit(task, source_index, source)
                pending[future] = (source_index, source)
                next_index += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index, source = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # defensive: task normally isolates failures
                    result = {
                        "story_id": story_ids[index],
                        "status": "failed",
                        "stage": "pipeline_internal",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "source": str(source),
                    }
                if result.get("status") == "complete":
                    results.append(result)
                else:
                    failures.append(result)
                record = {"index": index, **result, "time": utc_now()}
                with lock:
                    progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress.flush()
    story_pe_audit = prepared_audit.snapshot(reject_duplicate_plans=True)
    atomic_write_json(config.output_root / "story_pe_audit.json", story_pe_audit)
    manifest = {
        "schema_version": "storymem.canonical.batch.v1",
        "status": "complete" if not failures else "partial",
        "requested_stories": len(sources),
        "complete_stories": len(results),
        "failed_stories": len(failures),
        "total_committed_shots": sum(int(item.get("num_shots", 0)) for item in results),
        "diversity": diversity,
        "story_pe_audit": story_pe_audit,
        "run_config": {**asdict(config), "output_root": str(config.output_root), "pe_manifest": str(config.pe_manifest), "tokenizer_path": str(config.tokenizer_path)},
        "records": sorted(results + failures, key=lambda item: str(item.get("story_id") or "")),
        "elapsed_seconds": round(time.time() - started, 3),
        "created_at": utc_now(),
    }
    atomic_write_json(config.output_root / "manifest.json", manifest)
    return manifest
