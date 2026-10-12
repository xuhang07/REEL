#!/usr/bin/env python3
"""Freeze completed Memory Policy trajectories for Qwen3.5 SFT.

The production shards contain native OpenAI messages.  This exporter validates
their receipts, restores private media paths, and writes LLaMA-Factory v1
native-message JSONL.  It deliberately does not project the data to legacy
alternating ShareGPT: production trajectories may contain a tool result followed
by a trusted runtime user observation, which is valid for Qwen3.5's own chat
template but cannot be represented losslessly by the legacy schema.

To get a useful first checkpoint quickly, each shot becomes a small number of
turn-level samples.  Every sample is a prefix ending at exactly one supervised
assistant turn; earlier assistant turns are retained as context with zero loss.
The evidence-plan and finish turns are always selected, plus deterministic
coverage turns when ``--targets-per-shot`` is greater than two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


PROGRESS_SCHEMA = "storymem.memory-policy.progress.v1"
SFT_SCHEMA = "storymem.memory-policy.sft.sample.v3"
TRAJECTORY_SCHEMA = "storymem.memory-policy.trajectory.v2"
SNAPSHOT_SCHEMA = "storymem.memory-policy.qwen35-sft-snapshot.v2"
MEDIA_BLOCK_TYPES = {"image": "image_url", "video": "video_url"}
REQUIRED_TARGETS = ("submit_evidence_plan", "finish_memory_session")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    else:
        text = canonical_json(value)
    return (text + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_stable_bytes(path: Path) -> bytes:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed while reading: {path}")
    return payload


def read_stable_json(path: Path) -> tuple[dict[str, Any], bytes, str]:
    payload = read_stable_bytes(path)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    # Production receipts hash canonical JSON, not the pretty-printed bytes.
    return value, payload, sha256_bytes(canonical_json(value).encode("utf-8"))


def require_digest(actual: str, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or expected != actual:
        raise ValueError(f"{label} digest mismatch: expected={expected!r} actual={actual!r}")


def ranked_story_ids(story_ids: list[str], seed: str, purpose: str) -> list[str]:
    return sorted(
        story_ids,
        key=lambda story_id: (
            hashlib.sha256(f"{seed}:{purpose}:{story_id}".encode()).hexdigest(),
            story_id,
        ),
    )


def _normalize_tool_calls(message: dict[str, Any], *, message_index: int) -> tuple[list[dict[str, str]], list[str]]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError(f"assistant message {message_index} has no tool calls")
    blocks: list[dict[str, str]] = []
    names: list[str] = []
    for call_index, call in enumerate(tool_calls):
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError(f"assistant message {message_index} tool call {call_index} is invalid")
        function = call.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"assistant message {message_index} tool call {call_index} has no function")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError(f"assistant message {message_index} tool call {call_index} has invalid name")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"assistant message {message_index} tool call {call_index} has invalid arguments JSON"
                ) from exc
        if not isinstance(arguments, dict):
            raise ValueError(
                f"assistant message {message_index} tool call {call_index} arguments must decode to an object"
            )
        blocks.append({"type": "tool_call", "value": canonical_json({"name": name, "arguments": arguments})})
        names.append(name)
    return blocks, names


def _normalize_content(
    content: Any,
    *,
    message_index: int,
    public_media: dict[tuple[int, int], dict[str, Any]],
    private_media: dict[str, dict[str, Any]],
    check_media: bool,
    verify_media_sha256: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    if content is None:
        return [], []
    if isinstance(content, str):
        return [{"type": "text", "value": content}], []
    if not isinstance(content, list):
        raise ValueError(f"message {message_index} has unsupported content type {type(content).__name__}")

    blocks: list[dict[str, str]] = []
    used_media_ids: list[str] = []
    for content_index, block in enumerate(content):
        if not isinstance(block, dict):
            raise ValueError(f"message {message_index} content {content_index} is not an object")
        block_type = block.get("type")
        if block_type == "text":
            value = block.get("text")
            if not isinstance(value, str):
                raise ValueError(f"message {message_index} content {content_index} has invalid text")
            blocks.append({"type": "text", "value": value})
            continue
        if block_type != "media":
            raise ValueError(
                f"message {message_index} content {content_index} has unsupported block type {block_type!r}"
            )

        public = public_media.get((message_index, content_index))
        if public is None:
            raise ValueError(f"missing public media entry for message {message_index} content {content_index}")
        media_id = block.get("media_id")
        if not isinstance(media_id, str) or public.get("media_id") != media_id:
            raise ValueError(f"media id mismatch at message {message_index} content {content_index}")
        private = private_media.get(media_id)
        if private is None:
            raise ValueError(f"missing private media path for {media_id}")

        media_type = block.get("media_type")
        if media_type not in MEDIA_BLOCK_TYPES:
            raise ValueError(f"Qwen3.5 snapshot does not support media type {media_type!r} ({media_id})")
        for key in ("media_type", "mime_type", "sha256", "size_bytes", "representation_id"):
            if block.get(key) != public.get(key) or public.get(key) != private.get(key):
                raise ValueError(f"public/private media field {key!r} mismatch for {media_id}")

        path_value = private.get("path")
        if not isinstance(path_value, str) or not os.path.isabs(path_value):
            raise ValueError(f"private media path must be absolute for {media_id}: {path_value!r}")
        media_path = Path(path_value)
        if check_media:
            if not media_path.is_file():
                raise ValueError(f"missing media file for {media_id}: {media_path}")
            expected_size = private.get("size_bytes")
            if not isinstance(expected_size, int) or media_path.stat().st_size != expected_size:
                raise ValueError(f"media size mismatch for {media_id}: {media_path}")
            if verify_media_sha256:
                require_digest(sha256_file(media_path), private.get("sha256"), f"media {media_id}")

        blocks.append({"type": MEDIA_BLOCK_TYPES[media_type], "value": path_value})
        used_media_ids.append(media_id)
    return blocks, used_media_ids


def _select_target_indices(
    messages: list[dict[str, Any]], *, story_id: str, shot_index: int, seed: str, targets_per_shot: int
) -> list[int]:
    assistant_calls: dict[int, list[str]] = {}
    for message_index, message in enumerate(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            _, names = _normalize_tool_calls(message, message_index=message_index)
            assistant_calls[message_index] = names

    finish = [index for index, names in assistant_calls.items() if names == ["finish_memory_session"]]
    if len(finish) != 1:
        raise ValueError(f"expected exactly one finish_memory_session turn, found {len(finish)}")
    evidence = [index for index, names in assistant_calls.items() if "submit_evidence_plan" in names]
    if not evidence:
        raise ValueError("missing submit_evidence_plan turn")

    required = [evidence[-1], finish[0]]
    candidates = [
        index
        for index, names in assistant_calls.items()
        if index not in required and not any(name in REQUIRED_TARGETS for name in names)
    ]
    # Prefer distinct tool names before repeated calls, while keeping selection deterministic.
    ordered = sorted(
        candidates,
        key=lambda index: hashlib.sha256(
            f"{seed}:target:{story_id}:{shot_index}:{','.join(assistant_calls[index])}:{index}".encode()
        ).hexdigest(),
    )
    extras: list[int] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for index in ordered:
        signature = tuple(assistant_calls[index])
        if signature not in seen_signatures:
            extras.append(index)
            seen_signatures.add(signature)
            if len(extras) == targets_per_shot - len(required):
                break
    if len(extras) < targets_per_shot - len(required):
        for index in ordered:
            if index not in extras:
                extras.append(index)
                if len(extras) == targets_per_shot - len(required):
                    break
    return sorted(set(required + extras))


def project_samples(
    sft: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    story_id: str,
    shot_index: int,
    sft_path: Path,
    trajectory_path: Path,
    sft_sha256: str,
    trajectory_sha256: str,
    check_media: bool,
    verify_media_sha256: bool,
    split_seed: str,
    targets_per_shot: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sft.get("schema_version") != SFT_SCHEMA:
        raise ValueError(f"unsupported SFT schema in {sft_path}: {sft.get('schema_version')!r}")
    if trajectory.get("schema_version") != TRAJECTORY_SCHEMA:
        raise ValueError(
            f"unsupported trajectory schema in {trajectory_path}: {trajectory.get('schema_version')!r}"
        )
    require_digest(trajectory_sha256, sft.get("source_trajectory_sha256"), str(sft_path))
    for key in ("request_id", "session_id", "raw_request_sha256", "messages_sha256", "media_manifest_sha256"):
        if sft.get(key) != trajectory.get(key):
            raise ValueError(f"SFT/trajectory field {key!r} mismatch for {sft_path}")

    messages = sft.get("messages")
    tools = sft.get("tools")
    public_manifest = sft.get("media_manifest")
    private_manifest = trajectory.get("private_media_manifest")
    if not isinstance(messages, list) or len(messages) < 4:
        raise ValueError(f"missing or short messages in {sft_path}")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"missing tools in {sft_path}")
    if not isinstance(public_manifest, list) or not isinstance(private_manifest, list):
        raise ValueError(f"missing media manifests for {sft_path}")
    require_digest(sha256_bytes(canonical_json(messages).encode()), sft.get("messages_sha256"), f"{sft_path} messages")
    require_digest(
        sha256_bytes(canonical_json(public_manifest).encode()),
        sft.get("media_manifest_sha256"),
        f"{sft_path} media manifest",
    )

    public_by_position: dict[tuple[int, int], dict[str, Any]] = {}
    public_ids: set[str] = set()
    for item in public_manifest:
        if not isinstance(item, dict):
            raise ValueError(f"invalid public media manifest item in {sft_path}")
        position = (item.get("message_index"), item.get("content_index"))
        media_id = item.get("media_id")
        if not all(isinstance(value, int) for value in position) or not isinstance(media_id, str):
            raise ValueError(f"invalid public media coordinates in {sft_path}")
        if position in public_by_position or media_id in public_ids:
            raise ValueError(f"duplicate public media binding in {sft_path}: {media_id}")
        public_by_position[position] = item
        public_ids.add(media_id)

    private_by_id: dict[str, dict[str, Any]] = {}
    for item in private_manifest:
        if not isinstance(item, dict) or not isinstance(item.get("media_id"), str):
            raise ValueError(f"invalid private media manifest item in {trajectory_path}")
        media_id = item["media_id"]
        if media_id in private_by_id:
            raise ValueError(f"duplicate private media id in {trajectory_path}: {media_id}")
        private_by_id[media_id] = item
    if public_ids != set(private_by_id):
        raise ValueError(f"public/private media id set mismatch for {sft_path}")

    normalized: list[dict[str, Any]] = []
    used_media_ids: list[str] = []
    assistant_names: dict[int, list[str]] = {}
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {message_index} is not an object in {sft_path}")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported role {role!r} at {sft_path}:{message_index}")
        if role == "assistant" and message.get("tool_calls"):
            if message.get("content") not in (None, "", []):
                raise ValueError(f"assistant tool call has unsupported mixed content at {sft_path}:{message_index}")
            content, names = _normalize_tool_calls(message, message_index=message_index)
            assistant_names[message_index] = names
        else:
            content, media_ids = _normalize_content(
                message.get("content"),
                message_index=message_index,
                public_media=public_by_position,
                private_media=private_by_id,
                check_media=check_media,
                verify_media_sha256=verify_media_sha256,
            )
            used_media_ids.extend(media_ids)
            if role == "assistant":
                assistant_names[message_index] = ["<assistant_text>"]
        normalized.append({"role": role, "content": content, "loss_weight": 0.0})

    if set(used_media_ids) != public_ids or len(used_media_ids) != len(public_ids):
        raise ValueError(f"not every media manifest entry is used exactly once in {sft_path}")
    if normalized[0]["role"] != "system":
        raise ValueError(f"first message is not system in {sft_path}")
    if normalized[-1]["role"] != "tool" or normalized[-2]["role"] != "assistant":
        raise ValueError(f"trajectory does not end in assistant/tool in {sft_path}")
    if assistant_names.get(len(normalized) - 2) != ["finish_memory_session"]:
        raise ValueError(f"final assistant turn is not finish_memory_session in {sft_path}")
    terminal_call_id = messages[-1].get("tool_call_id")
    final_calls = messages[-2].get("tool_calls") or []
    if terminal_call_id is not None and (len(final_calls) != 1 or final_calls[0].get("id") != terminal_call_id):
        raise ValueError(f"terminal tool acknowledgement id mismatch in {sft_path}")

    normalized = normalized[:-1]  # terminal acknowledgement is never a model target/context.
    source_messages = messages[:-1]
    target_indices = _select_target_indices(
        source_messages,
        story_id=story_id,
        shot_index=shot_index,
        seed=split_seed,
        targets_per_shot=targets_per_shot,
    )
    rows: list[dict[str, Any]] = []
    target_names: Counter[str] = Counter()
    media_refs = 0
    for target_index in target_indices:
        prefix = [
            {"role": item["role"], "content": item["content"], "loss_weight": 0.0}
            for item in normalized[: target_index + 1]
        ]
        if prefix[-1]["role"] != "assistant":
            raise AssertionError("selected target is not assistant")
        prefix[-1]["loss_weight"] = 1.0
        names = assistant_names[target_index]
        target_names.update(names)
        media_refs += sum(
            block["type"] in {"image_url", "video_url"}
            for message in prefix
            for block in message["content"]
        )
        sample_id = f"{story_id}:shot:{shot_index:04d}:turn:{target_index:04d}"
        extra_info = {
            "id": sample_id,
            "story_id": story_id,
            "shot_index": shot_index,
            "target_message_index": target_index,
            "target_tool_names": names,
            "source_sft_path": str(sft_path),
            "source_sft_sha256": sft_sha256,
            "source_trajectory_path": str(trajectory_path),
            "source_trajectory_sha256": trajectory_sha256,
        }
        rows.append({"messages": prefix, "tools": canonical_json(tools), "extra_info": canonical_json(extra_info)})

    return rows, {
        "source_messages": len(normalized),
        "source_images": sum(item.get("media_type") == "image" for item in public_manifest),
        "source_videos": sum(item.get("media_type") == "video" for item in public_manifest),
        "sample_media_references": media_refs,
        "target_tool_names": target_names,
    }


class HashedWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: BinaryIO = path.open("wb")
        self.digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0

    def write(self, value: dict[str, Any]) -> None:
        payload = json_bytes(value)
        self.handle.write(payload)
        self.digest.update(payload)
        self.rows += 1
        self.bytes += len(payload)

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()

    def summary(self) -> dict[str, Any]:
        return {
            "file": self.path.name,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": "sha256:" + self.digest.hexdigest(),
        }


def _write_dataset_yaml(path: Path, *, dataset_name: str, jsonl_path: Path) -> None:
    # JSON strings are valid YAML scalars and avoid path quoting surprises.
    value = (
        f"{dataset_name}:\n"
        f"  path: {json.dumps(str(jsonl_path), ensure_ascii=False)}\n"
        "  source: local\n"
    )
    path.write_text(value, encoding="utf-8")


def export_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    progress_path = data_root / "progress.json"
    progress, progress_payload, progress_sha256 = read_stable_json(progress_path)
    if progress.get("schema_version") != PROGRESS_SCHEMA:
        raise ValueError(f"unsupported progress schema: {progress.get('schema_version')!r}")
    story_map = progress.get("stories")
    if not isinstance(story_map, dict):
        raise ValueError("progress.json has no stories object")

    completed: dict[str, dict[str, Any]] = {
        story_id: record
        for story_id, record in story_map.items()
        if isinstance(record, dict) and record.get("status") == "complete"
    }
    if len(completed) < 2:
        raise ValueError("at least two complete stories are required")
    selected_ids = ranked_story_ids(list(completed), args.split_seed, "snapshot")
    if args.max_stories:
        selected_ids = selected_ids[: args.max_stories]
    if len(selected_ids) < 2:
        raise ValueError("selection must contain at least two stories")

    val_count = args.val_stories
    if val_count is None:
        val_count = max(1, round(len(selected_ids) * args.val_fraction))
    if not 1 <= val_count < len(selected_ids):
        raise ValueError("validation story count must be between 1 and selected_stories - 1")
    val_ids = set(ranked_story_ids(selected_ids, args.split_seed, "validation")[:val_count])

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite snapshot: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)

    train_name = f"{args.dataset_name}_train"
    val_name = f"{args.dataset_name}_val"
    train_writer = HashedWriter(staging / "train.jsonl")
    val_writer = HashedWriter(staging / "val.jsonl")
    source_writer = HashedWriter(staging / "source_index.jsonl")
    totals: Counter[str] = Counter()
    target_tool_counts: Counter[str] = Counter()
    story_summaries: list[dict[str, Any]] = []

    try:
        for story_number, story_id in enumerate(sorted(selected_ids), start=1):
            record = completed[story_id]
            if record.get("story_id") != story_id:
                raise ValueError(f"progress story key/id mismatch for {story_id}")
            shots = record.get("shots")
            if not isinstance(shots, dict):
                raise ValueError(f"complete story has no shots object: {story_id}")
            try:
                num_shots = int(record["num_shots"])
                shot_indices = sorted(int(index) for index in shots)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid shot metadata for {story_id}") from exc
            if shot_indices != list(range(num_shots)):
                raise ValueError(f"incomplete/non-contiguous shots for {story_id}: {shot_indices}")

            story_rows = 0
            split = "validation" if story_id in val_ids else "train"
            writer = val_writer if story_id in val_ids else train_writer
            for shot_index in shot_indices:
                shot = shots[str(shot_index)]
                if not isinstance(shot, dict) or shot.get("status") != "complete":
                    raise ValueError(f"story {story_id} shot {shot_index} is not complete")
                sft_path = Path(str(shot.get("sft_path") or ""))
                trajectory_path = Path(str(shot.get("trajectory_path") or ""))
                sft, _, sft_sha256 = read_stable_json(sft_path)
                trajectory, _, trajectory_sha256 = read_stable_json(trajectory_path)
                require_digest(sft_sha256, shot.get("sft_sha256"), str(sft_path))
                require_digest(trajectory_sha256, shot.get("trajectory_sha256"), str(trajectory_path))
                rows, stats = project_samples(
                    sft,
                    trajectory,
                    story_id=story_id,
                    shot_index=shot_index,
                    sft_path=sft_path,
                    trajectory_path=trajectory_path,
                    sft_sha256=sft_sha256,
                    trajectory_sha256=trajectory_sha256,
                    check_media=not args.skip_media_check,
                    verify_media_sha256=args.verify_media_sha256,
                    split_seed=args.split_seed,
                    targets_per_shot=args.targets_per_shot,
                )
                for row in rows:
                    writer.write(row)
                    source_writer.write(json.loads(row["extra_info"]))
                story_rows += len(rows)
                totals.update(
                    {
                        "source_shots": 1,
                        "samples": len(rows),
                        "source_messages": stats["source_messages"],
                        "source_images": stats["source_images"],
                        "source_videos": stats["source_videos"],
                        "sample_media_references": stats["sample_media_references"],
                    }
                )
                target_tool_counts.update(stats["target_tool_names"])

            story_summaries.append({"story_id": story_id, "split": split, "samples": story_rows})
            if story_number % 25 == 0 or story_number == len(selected_ids):
                print(
                    canonical_json(
                        {
                            "event": "validated",
                            "stories": story_number,
                            "selected_stories": len(selected_ids),
                            "samples": totals["samples"],
                        }
                    ),
                    flush=True,
                )

        train_writer.close()
        val_writer.close()
        source_writer.close()
        if train_writer.rows == 0 or val_writer.rows == 0:
            raise RuntimeError("empty train or validation split")

        final_train_jsonl = output_dir / "train.jsonl"
        final_val_jsonl = output_dir / "val.jsonl"
        _write_dataset_yaml(staging / "train_dataset.yaml", dataset_name=train_name, jsonl_path=final_train_jsonl)
        _write_dataset_yaml(staging / "val_dataset.yaml", dataset_name=val_name, jsonl_path=final_val_jsonl)
        (staging / "train_story_ids.txt").write_text(
            "\n".join(sorted(set(selected_ids) - val_ids)) + "\n", encoding="utf-8"
        )
        (staging / "val_story_ids.txt").write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")
        with (staging / "story_index.jsonl").open("wb") as handle:
            for item in story_summaries:
                handle.write(json_bytes(item))

        manifest = {
            "schema_version": SNAPSHOT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_data_root": str(data_root),
            "source_progress_path": str(progress_path),
            "source_progress_sha256": progress_sha256,
            "source_progress_file_sha256": sha256_bytes(progress_payload),
            "source_progress_bytes": len(progress_payload),
            "source_complete_stories": len(completed),
            "selection": {
                "seed": args.split_seed,
                "max_stories": args.max_stories,
                "targets_per_shot": args.targets_per_shot,
                "selected_stories": len(selected_ids),
                "train_stories": len(selected_ids) - len(val_ids),
                "val_stories": len(val_ids),
                "story_overlap": 0,
            },
            "datasets": {
                train_name: train_writer.summary(),
                val_name: val_writer.summary(),
            },
            "source_index": source_writer.summary(),
            "totals": dict(totals),
            "target_tool_counts": dict(sorted(target_tool_counts.items())),
            "media_policy": {
                "paths": "absolute_private_manifest_paths",
                "copied": False,
                "existence_and_size_checked": not args.skip_media_check,
                "sha256_checked": args.verify_media_sha256,
                "supported_types": sorted(MEDIA_BLOCK_TYPES),
            },
            "projection": {
                "format": "llamafactory_v1_native_messages_turn_snapshot",
                "terminal_tool_acknowledgement": "dropped",
                "historical_assistant_loss_weight": 0.0,
                "target_assistant_loss_weight": 1.0,
                "assistant_tool_arguments": "decoded_json_objects_in_tool_call_blocks",
                "chat_template": "model_native_qwen3_5_no_think",
            },
        }
        (staging / "snapshot_manifest.json").write_bytes(json_bytes(manifest, pretty=True))
        os.replace(staging, output_dir)
    except BaseException:
        for writer in (train_writer, val_writer, source_writer):
            if not writer.handle.closed:
                writer.handle.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="storymem_memory_policy_qwen35_v1")
    parser.add_argument("--max-stories", type=int, default=0, help="0 captures every complete story")
    parser.add_argument("--val-stories", type=int)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--targets-per-shot", type=int, default=3)
    parser.add_argument("--split-seed", default="storymem-memory-policy-qwen35-v1")
    parser.add_argument("--skip-media-check", action="store_true")
    parser.add_argument("--verify-media-sha256", action="store_true")
    args = parser.parse_args(argv)
    if args.max_stories < 0:
        parser.error("--max-stories must be non-negative")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be in (0, 1)")
    if args.val_stories is not None and args.val_stories < 1:
        parser.error("--val-stories must be positive")
    if args.targets_per_shot < len(REQUIRED_TARGETS):
        parser.error(f"--targets-per-shot must be at least {len(REQUIRED_TARGETS)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_snapshot(args)
    print(canonical_json({"event": "complete", "output_dir": str(args.output_dir.resolve()), **manifest}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(1)
