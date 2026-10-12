#!/usr/bin/env python3
"""Strict, artifact-backed acceptance checks for one StoryMem RL stage.

The verifier is intentionally independent from the training process.  A stage
passes only when its exact checkpoint exists, the trainer log proves that GRPO
produced non-degenerate advantages and a real update, and at least one rollout
from the current invocation completed every shot and received a positive
Gemini review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verl_recipe.storymem_memory_policy_rl.policy_profile import (
    TRAINING_SYSTEM_PROMPT,
    training_system_prompt_hash,
)
from tools.storymem_policy.contract import TOOL_PROTOCOL_VERSION
from .policy_profile import tool_schema_hash

from .contracts import REVIEW_SCHEMA_VERSION, TRAJECTORY_SCHEMA_VERSION


SNAPSHOT_SCHEMA_VERSION = "storymem.rl-stage-preexisting-rollouts.v1"
MERGE_RECEIPT_SCHEMA_VERSION = "storymem.rl-merged-hf-receipt.v1"
MERGE_RECEIPT_NAME = "storymem_merge_receipt.json"
STATE_COMPONENTS = ("registry_items", "memory_bank", "previous_evidence")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan)"


class StageVerificationError(RuntimeError):
    """The persisted artifacts do not prove a successful RL stage."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageVerificationError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageVerificationError(f"{label} must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise StageVerificationError(f"refusing to replace existing receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise StageVerificationError(f"refusing to replace existing receipt: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_rollouts(
    output_root: Path, destination: Path, *, stage_token: str
) -> dict[str, Any]:
    """Capture rollout paths that predate a stage invocation."""

    root = output_root.expanduser().resolve()
    if not stage_token:
        raise StageVerificationError("stage token must be non-empty")
    episode_root = root / "episodes"
    value = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "stage_token": stage_token,
        "output_root": str(root),
        "captured_at_unix_ns": __import__("time").time_ns(),
        "episode_paths": [
            str(path.resolve())
            for path in sorted(episode_root.iterdir() if episode_root.is_dir() else ())
            if path.is_dir()
        ],
        "trajectory_paths": [
            str(path.resolve())
            for path in sorted(episode_root.glob("*/trajectory.json"))
        ],
    }
    _atomic_json(destination.expanduser().resolve(), value)
    return value


def _load_snapshot(
    path: Path, output_root: Path, *, stage_token: str
) -> tuple[set[str], set[str], int]:
    value = _load_object(path, label="pre-stage rollout snapshot")
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise StageVerificationError(f"unsupported rollout snapshot schema: {path}")
    if value.get("output_root") != str(output_root.resolve()):
        raise StageVerificationError("rollout snapshot is bound to a different output root")
    if value.get("stage_token") != stage_token:
        raise StageVerificationError("rollout snapshot is bound to a different stage token")
    captured = value.get("captured_at_unix_ns")
    paths = value.get("trajectory_paths")
    episodes = value.get("episode_paths")
    if isinstance(captured, bool) or not isinstance(captured, int) or captured < 1:
        raise StageVerificationError("rollout snapshot has an invalid capture time")
    if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
        raise StageVerificationError("rollout snapshot has invalid trajectory paths")
    if not isinstance(episodes, list) or any(
        not isinstance(item, str) for item in episodes
    ):
        raise StageVerificationError("rollout snapshot has invalid episode paths")
    if len(paths) != len(set(paths)):
        raise StageVerificationError("rollout snapshot contains duplicate paths")
    if len(episodes) != len(set(episodes)):
        raise StageVerificationError("rollout snapshot contains duplicate episode paths")
    return set(paths), set(episodes), captured


def _require_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StageVerificationError(f"{label} is not a canonical SHA-256 digest")
    return value


def _require_state_hashes(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(STATE_COMPONENTS):
        raise StageVerificationError(
            f"{label} must contain exactly {', '.join(STATE_COMPONENTS)}"
        )
    return {
        name: _require_digest(value[name], label=f"{label}.{name}")
        for name in STATE_COMPONENTS
    }


def _expected_context_hash(requirement: str) -> str:
    messages = [
        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
        {"role": "user", "content": requirement},
    ]
    return _sha256_text(_canonical_json(messages))


def _validate_video(shot: Mapping[str, Any], *, label: str) -> Path:
    raw_path = shot.get("output_video_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise StageVerificationError(f"{label} has no output_video_path")
    path = Path(raw_path).expanduser()
    if not path.is_file() or path.stat().st_size <= 0:
        raise StageVerificationError(f"{label} video is missing or empty: {path}")
    declared = _require_digest(
        shot.get("output_video_sha256"), label=f"{label}.output_video_sha256"
    )
    actual = _sha256_file(path)
    if declared != actual:
        raise StageVerificationError(
            f"{label} video SHA-256 mismatch: declared={declared} actual={actual}"
        )
    return path.resolve()


def _validate_review(
    path: Path,
    *,
    trajectory_path: Path,
    mode: str,
    expected_shots: int,
    expected_model: str,
    video_paths: Sequence[Path],
    expected_reward_version: str = "v1",
) -> float:
    review = _load_object(path, label="reward review")
    if expected_reward_version == "reel_joint_v1" and mode == "multi":
        from .reel_joint_multishot import validate_online_review
        return validate_online_review(trajectory_path=trajectory_path, expected_model=expected_model)
    if expected_reward_version in {"single_v2", "single_memory_v3", "reel_joint_v1"}:
        if mode != "single" or expected_shots != 1:
            raise StageVerificationError("single reward verification requires exactly one shot")
        if expected_reward_version == "reel_joint_v1":
            from .reel_joint_reward import validate_online_review
        elif expected_reward_version == "single_memory_v3":
            from .memory_reward_v3 import validate_online_review
        else:
            from .online_reward_v2 import validate_online_review

        try:
            return validate_online_review(review, trajectory_path=trajectory_path, expected_model=expected_model)
        except Exception as exc:
            raise StageVerificationError(f"{expected_reward_version} receipt validation failed: {exc}") from exc
    if expected_reward_version != "v1":
        raise StageVerificationError("unsupported expected reward version")
    if review.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise StageVerificationError("reward review schema version mismatch")
    if review.get("mode") != mode:
        raise StageVerificationError("reward review mode differs from trajectory mode")
    if review.get("judge_called") is not True:
        raise StageVerificationError("reward review did not call Gemini")
    if review.get("policy_failure") is not False:
        raise StageVerificationError("reward review marks the trajectory as a policy failure")
    reward = review.get("reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or float(reward) <= 0.0
        or float(reward) > 1.0
    ):
        raise StageVerificationError("reward review has no finite positive reward")
    local = review.get("local_shot_rewards")
    if (
        not isinstance(local, list)
        or len(local) != expected_shots
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
            for item in local
        )
    ):
        raise StageVerificationError("reward review has invalid per-shot rewards")
    raw_trajectory = review.get("trajectory_path")
    if not isinstance(raw_trajectory, str) or Path(raw_trajectory).resolve() != trajectory_path:
        raise StageVerificationError("reward review is bound to a different trajectory")
    raw_videos = review.get("video_paths")
    if not isinstance(raw_videos, list) or [
        Path(item).expanduser().resolve() if isinstance(item, str) else None
        for item in raw_videos
    ] != list(video_paths):
        raise StageVerificationError("reward review video list differs from trajectory videos")

    judge = review.get("judge")
    if not isinstance(judge, dict) or judge.get("model") != expected_model:
        raise StageVerificationError(
            f"reward review did not use expected Gemini model {expected_model!r}"
        )
    if judge.get("api_style") != "gemini":
        raise StageVerificationError("reward review did not use the Gemini API style")
    repeats = judge.get("review_repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise StageVerificationError("reward review has an invalid review_repeats value")
    groups = judge.get("per_shot")
    if not isinstance(groups, list) or len(groups) != expected_shots:
        raise StageVerificationError(
            f"reward review must contain exactly {expected_shots} Gemini receipt groups"
        )
    for position, group in enumerate(groups):
        if not isinstance(group, dict) or group.get("episode_position") != position:
            raise StageVerificationError("Gemini receipt groups are not position-aligned")
        receipts = group.get("receipts")
        if (
            not isinstance(receipts, list)
            or len(receipts) != repeats
            or group.get("review_repeats") != repeats
        ):
            raise StageVerificationError(
                f"Gemini receipt group {position} does not contain exactly {repeats} requests"
            )
        for repeat, receipt in enumerate(receipts, start=1):
            if not isinstance(receipt, dict) or receipt.get("episode_position") != position:
                raise StageVerificationError("Gemini receipt is not position-aligned")
            if receipt.get("repeat") != repeat:
                raise StageVerificationError("Gemini receipt repeats are not contiguous")
            attempts = receipt.get("attempts")
            if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
                raise StageVerificationError("Gemini receipt has no positive attempt count")
            provenance = receipt.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("model") != expected_model:
                raise StageVerificationError(
                    f"Gemini receipt does not prove model {expected_model!r}"
                )
            if (
                provenance.get("assistant_source") != "remote_closed_model_api"
                or provenance.get("api_style") != "gemini"
            ):
                raise StageVerificationError("Gemini receipt has invalid provider provenance")
            for field in (
                "endpoint_sha256",
                "request_payload_sha256",
                "request_sha256",
                "response_sha256",
                "assistant_text_sha256",
            ):
                _require_digest(provenance.get(field), label=f"Gemini receipt {field}")
            if provenance["request_payload_sha256"] != provenance["request_sha256"]:
                raise StageVerificationError("Gemini request hashes disagree")
    return float(reward)


def _validate_trajectory(
    trajectory_path: Path,
    *,
    mode: str,
    expected_shots: int,
    expected_model: str,
    expected_max_format_rejections: int,
    expected_infra_max_attempts: int,
    expected_reward_version: str = "v1",
) -> float:
    trajectory = _load_object(trajectory_path, label="trajectory")
    if trajectory.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise StageVerificationError("trajectory schema version mismatch")
    if trajectory.get("mode") != mode:
        raise StageVerificationError("trajectory mode mismatch")
    if trajectory.get("expected_shots") != expected_shots:
        raise StageVerificationError("trajectory expected_shots mismatch")
    if trajectory.get("completed_shots") != expected_shots:
        raise StageVerificationError("trajectory is incomplete")
    if trajectory.get("fresh_conversation_per_shot") is not True:
        raise StageVerificationError("trajectory does not require fresh per-shot context")
    if trajectory.get("system_prompt_sha256") != training_system_prompt_hash():
        raise StageVerificationError("trajectory system prompt hash mismatch")
    if trajectory.get("tool_protocol_version") != TOOL_PROTOCOL_VERSION:
        raise StageVerificationError("trajectory tool protocol version mismatch")
    if trajectory.get("tool_schema_sha256") != tool_schema_hash():
        raise StageVerificationError("trajectory tool schema hash mismatch")
    attempt = trajectory.get("infrastructure_attempt")
    max_attempts = trajectory.get("max_infrastructure_attempts")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 1 <= attempt <= expected_infra_max_attempts
        or max_attempts != expected_infra_max_attempts
    ):
        raise StageVerificationError("trajectory infrastructure retry receipt mismatch")
    failures = trajectory.get("infrastructure_failures")
    if not isinstance(failures, list) or failures:
        raise StageVerificationError("trajectory contains infrastructure failures")
    shots = trajectory.get("shots")
    if not isinstance(shots, list) or len(shots) != expected_shots:
        raise StageVerificationError("trajectory does not contain exactly K shots")

    videos: list[Path] = []
    previous_final: dict[str, str] | None = None
    for position, shot in enumerate(shots):
        label = f"shots[{position}]"
        if not isinstance(shot, dict) or shot.get("episode_position") != position:
            raise StageVerificationError("trajectory shots are not position-aligned")
        if shot.get("completed") is not True or shot.get("phase") != "complete":
            raise StageVerificationError(f"{label} is not complete")
        if shot.get("fresh_conversation") is not True:
            raise StageVerificationError(f"{label} did not start a fresh conversation")
        if shot.get("protocol_error") is not None:
            raise StageVerificationError(f"{label} contains a protocol error")
        shot_failures = shot.get("infrastructure_failures")
        if not isinstance(shot_failures, list) or shot_failures:
            raise StageVerificationError(f"{label} contains infrastructure failures")
        format_rejections = shot.get("format_rejections")
        if (
            not isinstance(format_rejections, list)
            or len(format_rejections) > expected_max_format_rejections
        ):
            raise StageVerificationError(f"{label} format-rejection receipt is invalid")
        requirement = shot.get("shot_requirement")
        if not isinstance(requirement, str) or not requirement:
            raise StageVerificationError(f"{label} has no shot requirement")
        if shot.get("shot_requirement_sha256") != _sha256_text(requirement):
            raise StageVerificationError(f"{label} shot requirement hash mismatch")
        if shot.get("initial_context_sha256") != _expected_context_hash(requirement):
            raise StageVerificationError(
                f"{label} does not prove a fresh canonical system+user context"
            )
        initial = _require_state_hashes(
            shot.get("initial_state_sha256"), label=f"{label}.initial_state_sha256"
        )
        final = _require_state_hashes(
            shot.get("final_state_sha256"), label=f"{label}.final_state_sha256"
        )
        if previous_final is not None and initial != previous_final:
            raise StageVerificationError(
                f"{label} did not inherit the previous shot's complete online state"
            )
        previous_final = final
        videos.append(_validate_video(shot, label=label))

    declared_videos = trajectory.get("shot_video_paths")
    if not isinstance(declared_videos, list) or [
        Path(item).expanduser().resolve() if isinstance(item, str) else None
        for item in declared_videos
    ] != videos:
        raise StageVerificationError("trajectory shot_video_paths is not position-aligned")
    return _validate_review(
        trajectory_path.parent / "reward_review.json",
        trajectory_path=trajectory_path.resolve(),
        mode=mode,
        expected_shots=expected_shots,
        expected_model=expected_model,
        video_paths=videos,
        expected_reward_version=expected_reward_version,
    )


def _metric_value(line: str, key: str) -> float | None:
    match = re.search(rf"(?<![A-Za-z0-9_./-]){re.escape(key)}:({_NUMBER})(?![A-Za-z0-9_.])", line)
    return None if match is None else float(match.group(1))


def verify_training_log(log_path: Path, *, expected_step: int) -> dict[str, bool]:
    try:
        text = _ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        raise StageVerificationError(f"cannot read training log {log_path}: {exc}") from exc
    flags = {
        "expected_step": False,
        "score_spread": False,
        "signed_advantages": False,
        "positive_grad_norm": False,
    }
    observed_steps: list[int] = []
    for line in text.splitlines():
        raw_step = _metric_value(line, "training/global_step")
        if raw_step is not None and math.isfinite(raw_step) and raw_step.is_integer():
            step = int(raw_step)
            observed_steps.append(step)
            flags["expected_step"] |= step == expected_step
        score_max = _metric_value(line, "critic/score/max")
        score_min = _metric_value(line, "critic/score/min")
        if score_max is not None and score_min is not None:
            flags["score_spread"] |= (
                math.isfinite(score_max)
                and math.isfinite(score_min)
                and score_max > score_min
            )
        advantage_max = _metric_value(line, "critic/advantages/max")
        advantage_min = _metric_value(line, "critic/advantages/min")
        if advantage_max is None:
            advantage_max = _metric_value(line, "advantages/max")
        if advantage_min is None:
            advantage_min = _metric_value(line, "advantages/min")
        if advantage_max is not None and advantage_min is not None:
            flags["signed_advantages"] |= (
                math.isfinite(advantage_max)
                and math.isfinite(advantage_min)
                and advantage_max > 0.0
                and advantage_min < 0.0
            )
        grad_norm = _metric_value(line, "actor/grad_norm")
        if grad_norm is not None:
            flags["positive_grad_norm"] |= math.isfinite(grad_norm) and grad_norm > 0.0
    if observed_steps and max(observed_steps) > expected_step:
        raise StageVerificationError(
            f"training log advanced past expected step {expected_step}: max={max(observed_steps)}"
        )
    missing = [name for name, present in flags.items() if not present]
    if missing:
        raise StageVerificationError(
            "training log lacks acceptance evidence: " + ", ".join(missing)
        )
    return flags


def verify_checkpoint(checkpoint_dir: Path, *, expected_step: int) -> Path:
    root = checkpoint_dir.expanduser().resolve()
    marker = root / "latest_checkpointed_iteration.txt"
    try:
        raw_step = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise StageVerificationError(f"cannot read checkpoint marker {marker}: {exc}") from exc
    if raw_step != str(expected_step):
        raise StageVerificationError(
            f"checkpoint marker is step {raw_step!r}, expected {expected_step}"
        )
    actor = root / f"global_step_{expected_step}" / "actor"
    if not actor.is_dir() or not (actor / "huggingface" / "config.json").is_file():
        raise StageVerificationError(f"exact actor checkpoint is incomplete: {actor}")
    later = []
    for path in root.glob("global_step_*"):
        try:
            step = int(path.name.removeprefix("global_step_"))
        except ValueError:
            continue
        if step > expected_step:
            later.append(step)
    if later:
        raise StageVerificationError(
            f"checkpoint directory contains steps beyond {expected_step}: {sorted(later)}"
        )
    return actor


def verify_stage(
    *,
    label: str,
    mode: str,
    output_root: Path,
    checkpoint_dir: Path,
    log_path: Path,
    expected_step: int,
    expected_shots: int,
    expected_model: str,
    minimum_judged: int,
    preexisting_snapshot: Path,
    stage_token: str,
    expected_max_format_rejections: int,
    expected_infra_max_attempts: int,
    expected_reward_version: str = "v1",
) -> dict[str, Any]:
    if mode not in {"single", "multi"}:
        raise StageVerificationError("mode must be single or multi")
    if expected_step < 1 or expected_shots < 1 or minimum_judged < 1:
        raise StageVerificationError("step, shot, and completion counts must be positive")
    if expected_max_format_rejections < 0 or expected_infra_max_attempts < 1:
        raise StageVerificationError("retry limits are invalid")
    if (mode == "single") != (expected_shots == 1):
        raise StageVerificationError("single mode requires K=1 and multi mode requires K>=2")
    root = output_root.expanduser().resolve()
    preexisting, preexisting_episodes, _captured_at_ns = _load_snapshot(
        preexisting_snapshot, root, stage_token=stage_token
    )
    actor = verify_checkpoint(checkpoint_dir, expected_step=expected_step)
    log = log_path.expanduser().resolve()
    if not log.is_file():
        raise StageVerificationError(f"training log does not exist: {log}")
    log_flags = verify_training_log(log, expected_step=expected_step)

    fresh_paths = [
        path.resolve()
        for path in sorted((root / "episodes").glob("*/trajectory.json"))
        if str(path.resolve()) not in preexisting
        and str(path.parent.resolve()) not in preexisting_episodes
    ]
    accepted: list[dict[str, Any]] = []
    rejected: list[str] = []
    for path in fresh_paths:
        try:
            reward = _validate_trajectory(
                path,
                mode=mode,
                expected_shots=expected_shots,
                expected_model=expected_model,
                expected_max_format_rejections=expected_max_format_rejections,
                expected_infra_max_attempts=expected_infra_max_attempts,
                expected_reward_version=expected_reward_version,
            )
        except StageVerificationError as exc:
            rejected.append(f"{path}: {exc}")
            continue
        accepted.append({"trajectory_path": str(path), "reward": reward})
    if len(accepted) < minimum_judged:
        detail = " | ".join(rejected[-3:]) if rejected else "no fresh trajectories"
        raise StageVerificationError(
            f"{label} has {len(accepted)} current complete positive Gemini-reviewed "
            f"trajectories; required {minimum_judged}. Last rejections: {detail}"
        )
    return {
        "label": label,
        "mode": mode,
        "reward_version": expected_reward_version,
        "expected_step": expected_step,
        "expected_shots": expected_shots,
        "checkpoint_actor": str(actor),
        "fresh_trajectory_count": len(fresh_paths),
        "accepted_trajectory_count": len(accepted),
        "accepted": accepted,
        "training_log": log_flags,
    }


def _merged_artifacts(merged_dir: Path) -> list[dict[str, Any]]:
    config = merged_dir / "config.json"
    if not config.is_file() or config.stat().st_size <= 0:
        raise StageVerificationError(f"merged checkpoint has no config.json: {merged_dir}")
    index = merged_dir / "model.safetensors.index.json"
    single = merged_dir / "model.safetensors"
    if index.is_file():
        value = _load_object(index, label="safetensors index")
        weight_map = value.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise StageVerificationError("safetensors index has no weight_map")
        weight_names = sorted(set(weight_map.values()))
        if any(not isinstance(name, str) or Path(name).name != name for name in weight_names):
            raise StageVerificationError("safetensors index contains unsafe shard names")
        expected_weights = {str((merged_dir / name).resolve()) for name in weight_names}
        actual_weights = {
            str(path.resolve()) for path in merged_dir.glob("*.safetensors")
        }
        if actual_weights != expected_weights:
            raise StageVerificationError(
                "merged checkpoint safetensors files differ from its weight index"
            )
    elif single.is_file():
        if set(merged_dir.glob("*.safetensors")) != {single}:
            raise StageVerificationError("single-file merged checkpoint has extra weight shards")
    else:
        raise StageVerificationError("merged checkpoint has no safetensors weights")
    paths = sorted(
        path
        for path in merged_dir.rglob("*")
        if path.is_file() and path.name != MERGE_RECEIPT_NAME
    )
    artifacts = []
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise StageVerificationError(f"merged checkpoint artifact is missing or empty: {path}")
        artifacts.append(
            {
                "relative_path": str(path.relative_to(merged_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return artifacts


def _expected_merge_receipt(
    merged_dir: Path, source_actor: Path, source_step: int
) -> dict[str, Any]:
    merged = merged_dir.expanduser().resolve()
    actor = source_actor.expanduser().resolve()
    expected_suffix = (f"global_step_{source_step}", "actor")
    if tuple(actor.parts[-2:]) != expected_suffix:
        raise StageVerificationError(
            f"source actor path is not global_step_{source_step}/actor: {actor}"
        )
    if not (actor / "huggingface" / "config.json").is_file():
        raise StageVerificationError(f"source actor checkpoint is incomplete: {actor}")
    return {
        "schema_version": MERGE_RECEIPT_SCHEMA_VERSION,
        "source_actor_path": str(actor),
        "source_global_step": source_step,
        "merged_hf_path": str(merged),
        "artifacts": _merged_artifacts(merged),
    }


def write_merge_receipt(
    merged_dir: Path, *, source_actor: Path, source_step: int
) -> Path:
    value = _expected_merge_receipt(merged_dir, source_actor, source_step)
    path = merged_dir.expanduser().resolve() / MERGE_RECEIPT_NAME
    _atomic_json(path, value, exclusive=True)
    return path


def verify_merge_receipt(
    merged_dir: Path, *, source_actor: Path, source_step: int
) -> Path:
    merged = merged_dir.expanduser().resolve()
    path = merged / MERGE_RECEIPT_NAME
    actual = _load_object(path, label="merged checkpoint receipt")
    expected = _expected_merge_receipt(merged, source_actor, source_step)
    if actual != expected:
        raise StageVerificationError(
            "merged checkpoint receipt does not match its exact source actor, step, or files"
        )
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot-rollouts")
    snapshot.add_argument("--output-root", type=Path, required=True)
    snapshot.add_argument("--destination", type=Path, required=True)
    snapshot.add_argument("--stage-token", required=True)

    stage = subparsers.add_parser("stage")
    stage.add_argument("--label", required=True)
    stage.add_argument("--mode", choices=("single", "multi"), required=True)
    stage.add_argument("--output-root", type=Path, required=True)
    stage.add_argument("--checkpoint-dir", type=Path, required=True)
    stage.add_argument("--log-path", type=Path, required=True)
    stage.add_argument("--expected-step", type=int, required=True)
    stage.add_argument("--expected-shots", type=int, required=True)
    stage.add_argument("--expected-gemini-model", required=True)
    stage.add_argument("--expected-reward-version", choices=("v1", "single_v2", "single_memory_v3", "reel_joint_v1"), default="v1")
    stage.add_argument("--expected-max-format-rejections", type=int, required=True)
    stage.add_argument("--expected-infra-max-attempts", type=int, required=True)
    stage.add_argument("--minimum-judged", type=int, default=1)
    stage.add_argument("--preexisting-snapshot", type=Path, required=True)
    stage.add_argument("--stage-token", required=True)

    for name in ("write-merge-receipt", "verify-merge-receipt"):
        merge = subparsers.add_parser(name)
        merge.add_argument("--merged-dir", type=Path, required=True)
        merge.add_argument("--source-actor", type=Path, required=True)
        merge.add_argument("--source-step", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "snapshot-rollouts":
            result: Any = snapshot_rollouts(
                args.output_root, args.destination, stage_token=args.stage_token
            )
        elif args.command == "stage":
            result = verify_stage(
                label=args.label,
                mode=args.mode,
                output_root=args.output_root,
                checkpoint_dir=args.checkpoint_dir,
                log_path=args.log_path,
                expected_step=args.expected_step,
                expected_shots=args.expected_shots,
                expected_model=args.expected_gemini_model,
                minimum_judged=args.minimum_judged,
                preexisting_snapshot=args.preexisting_snapshot,
                stage_token=args.stage_token,
                expected_max_format_rejections=args.expected_max_format_rejections,
                expected_infra_max_attempts=args.expected_infra_max_attempts,
                expected_reward_version=args.expected_reward_version,
            )
        elif args.command == "write-merge-receipt":
            result = {
                "receipt": str(
                    write_merge_receipt(
                        args.merged_dir,
                        source_actor=args.source_actor,
                        source_step=args.source_step,
                    )
                )
            }
        else:
            result = {
                "receipt": str(
                    verify_merge_receipt(
                        args.merged_dir,
                        source_actor=args.source_actor,
                        source_step=args.source_step,
                    )
                )
            }
    except StageVerificationError as exc:
        raise SystemExit(f"[storymem-stage-verifier][fatal] {exc}") from None
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
