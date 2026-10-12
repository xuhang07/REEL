#!/usr/bin/env python3
"""Run Qwen3.5 base or SFT as the native nine-tool policy on ST-Bench.

The model-facing contract is loaded from the exact clean SFT snapshot.  Every
shot starts a fresh conversation with that same system prompt; memory persists
only through the policy runtime's explicit registry and memory-bank state.

This runner deliberately uses greedy decoding and the same global-shot seed as
``run_storymem_stbench_baselines.py``.  It records every model turn, rejected
runtime call, durable Echo reservation, and completed video so an interrupted
run can resume without silently issuing another generation job.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import fcntl
import hashlib
import json
import queue
import re
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

import run_storymem_stbench_baselines as stbench
from evaluate_storymem_qwen35_sft_smoke import (
    model_inputs_to_device,
    parse_qwen35_tool_call,
)
from storymem_adapter.echo15 import (
    Echo15BackendAdapter,
    Echo15Capabilities,
)
from storymem_policy.contract import CANONICAL_TOOL_NAMES, get_tool_schemas
from storymem_policy.runtime import MemoryPolicyRuntime
from storymem_policy.trajectory import (
    _merged_events,
    _observation_content,
    _prepare_media,
)
from storymem_policy_collector import (
    CollectorConfig,
    CollectorError,
    Echo15WireTransport,
    EchoTerminalFailure,
    StoryCollector,
    _atomic_json,
    _json_copy,
    _safe_id,
    prepare_story_source,
)
from storymem_teacher.util import (
    canonical_json,
    safe_name,
    sha256_file,
    sha256_json,
    sha256_text,
)


RUNNER_VERSION = "storymem.stbench.qwen35-native-policy.v4"
TRACE_VERSION = "storymem.stbench.qwen35-native-policy-trace.v1"
RUN_CONFIG_VERSION = "storymem.stbench.qwen35-native-policy-run.v3"
PROGRESS_VERSION = "storymem.stbench.qwen35-native-policy-progress.v1"
SHOT_HOST_VERSION = "storymem.stbench.qwen35-native-policy-host.v1"

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/external/normalized/storymem_stbench/manifest.json"
)
DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "sft_snapshot"
DEFAULT_BASE_MODEL = Path(__file__).resolve().parents[1] / "models" / "Qwen3.5-9B"
DEFAULT_SFT_MODEL = Path(__file__).resolve().parents[1] / "models" / "REEL"
DEFAULT_ENDPOINTS = tuple(f"http://127.0.0.1:{port}" for port in range(8421, 8425))
IMAGE_CREATION_ATTEMPT_LIMIT = 0
SUPPORTED_MEDIA = {"image": "image_url", "video": "video_url"}
TRAINING_RENDERER_SOURCE_COMMIT = "97b32d3133b501432141a82949d5c7bc4d94f23a"
RENDERER_PARITY_FIXTURE = {
    "schema_version": "storymem.qwen35-renderer-parity-fixture.v1",
    "reference_implementation": "llamafactory.v1.core.rendering.Renderer",
    "reference_commit": TRAINING_RENDERER_SOURCE_COMMIT,
    "snapshot_schema_version": "continuity-evidence.qwen35-sft-snapshot.v3",
    "validation_jsonl_sha256": (
        "sha256:7027376e74e9ec78112221f4e8bdb879eda6b98e96e9113f5ae8efd7ca132394"
    ),
    "row_index": 0,
    "message_prefix_length": 2,
    "message_prefix_sha256": (
        "sha256:28d9cf6cbd57d481f1873d6aeb7ab8793a102bf3fc2a20996c8dd01b19744578"
    ),
    "tools_text_sha256": (
        "sha256:c72d3729b9406be129983ff82c5cd880c198fc7a799dddfee79075b3cb689278"
    ),
    "render_kwargs": {"is_generate": True, "enable_thinking": False},
    "input_token_count": 8208,
    "input_ids_sha256": (
        "sha256:4e502ebee2af562e10bb75554179575448a8939cb09f76dd8fef05eb20b8b2ac"
    ),
    "input_ids_prefix": [248045, 8678, 198, 2, 13455, 271, 2523, 599],
    "input_ids_suffix": [198, 248045, 74455, 198, 248068, 271, 248069, 271],
}
_MULTIMODAL_PASSTHROUGH_KEYS = frozenset(
    {
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
        "video_second_per_grid",
        "input_features",
        "feature_attention_mask",
    }
)


class NativePolicyError(RuntimeError):
    """Base class carrying a stable error category for experiment accounting."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class NativeModelOutputError(NativePolicyError):
    def __init__(
        self,
        category: str,
        message: str,
        *,
        raw_output: str,
        diagnostics: Mapping[str, Any],
    ) -> None:
        super().__init__(category, message)
        self.raw_output = raw_output
        self.diagnostics = _json_copy(dict(diagnostics))


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotContract:
    root: Path
    system_prompt: str
    tools: tuple[dict[str, Any], ...]
    tools_text: str
    system_prompt_sha256: str
    tool_schema_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class ModelTurn:
    raw_output: str
    tool_call: dict[str, Any]
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    ended_on_im_end: bool
    input_messages_sha256: str
    input_message_count: int


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def load_snapshot_contract(path: str | Path) -> SnapshotContract:
    root = Path(path).expanduser().resolve()
    system_path = root / "system_prompt.txt"
    tools_path = root / "tools.json"
    manifest_path = root / "snapshot_manifest.json"
    for required in (system_path, tools_path, manifest_path):
        if not required.is_file():
            raise NativePolicyError("snapshot_missing", f"missing snapshot artifact: {required}")

    manifest = _strict_json(manifest_path)
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(files, Mapping):
        raise NativePolicyError("snapshot_invalid", "snapshot manifest has no files map")
    for name, artifact in (("system_prompt.txt", system_path), ("tools.json", tools_path)):
        row = files.get(name)
        if not isinstance(row, Mapping) or row.get("sha256") != _file_sha256(artifact):
            raise NativePolicyError(
                "snapshot_hash_mismatch", f"snapshot manifest does not seal {name}"
            )

    system_prompt = system_path.read_text(encoding="utf-8")
    tools = _strict_json(tools_path)
    if not isinstance(tools, list):
        raise NativePolicyError("tool_contract_invalid", "tools.json must contain an array")
    names = [
        item.get("function", {}).get("name")
        for item in tools
        if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)
    ]
    if tuple(names) != CANONICAL_TOOL_NAMES or len(tools) != 9:
        raise NativePolicyError(
            "tool_contract_invalid",
            f"snapshot must expose exactly the canonical nine tools, found {names}",
        )
    if tools != get_tool_schemas():
        raise NativePolicyError(
            "tool_contract_drift",
            "snapshot tools differ from the active MemoryPolicyRuntime contract",
        )
    tools_text = canonical_json(tools)
    return SnapshotContract(
        root=root,
        system_prompt=system_prompt,
        tools=tuple(copy.deepcopy(tools)),
        tools_text=tools_text,
        system_prompt_sha256=sha256_text(system_prompt),
        tool_schema_sha256=sha256_json(tools),
    )


def model_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NativePolicyError("model_missing", f"model directory does not exist: {root}")
    files: dict[str, Any] = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "training_report.json",
    ):
        candidate = root / name
        if candidate.is_file():
            files[name] = {
                "size_bytes": candidate.stat().st_size,
                "sha256": _file_sha256(candidate),
            }
    shards = sorted(
        {
            *root.glob("model-*-of-*.safetensors"),
            *root.glob("model.safetensors-*-of-*.safetensors"),
        }
    )
    if not shards:
        single = root / "model.safetensors"
        if single.is_file():
            shards = [single]
    if not shards:
        raise NativePolicyError("model_incomplete", f"model has no weight shards: {root}")
    files["weight_shards"] = [
        {"name": item.name, "size_bytes": item.stat().st_size} for item in shards
    ]
    identity = {"path": str(root), "files": files}
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def _convert_observation_blocks(
    blocks: Sequence[Mapping[str, Any]], private_by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            result.append({"type": "text", "value": str(block.get("text") or "")})
            continue
        if block_type != "media":
            raise NativePolicyError("dialogue_invalid", f"unsupported observation block: {block_type}")
        media_id = str(block.get("media_id") or "")
        private = private_by_id.get(media_id)
        if private is None:
            raise NativePolicyError("dialogue_invalid", f"media {media_id!r} has no local bytes")
        media_type = str(block.get("media_type") or "")
        output_type = SUPPORTED_MEDIA.get(media_type)
        if output_type is None:
            raise NativePolicyError(
                "unsupported_model_media", f"Qwen3.5 cannot receive {media_type!r} media"
            )
        media_path = Path(str(private.get("path") or "")).expanduser().resolve()
        if not media_path.is_file():
            raise NativePolicyError("media_missing", f"model media is missing: {media_path}")
        if private.get("sha256") != sha256_file(media_path):
            raise NativePolicyError("media_hash_mismatch", f"model media changed: {media_path}")
        result.append({"type": output_type, "value": str(media_path)})
    return result


def build_native_messages(
    *,
    runtime: MemoryPolicyRuntime,
    contract: SnapshotContract,
    repair_context: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Reconstruct the exact LLaMA-Factory v1 dialogue for the current shot."""

    events = runtime.state.data.get("events") or []
    host_events = runtime.state.data.get("host_events") or []
    media_by_event, private_manifest = _prepare_media(events)
    private_by_id = {
        str(item["media_id"]): item for item in private_manifest if isinstance(item, Mapping)
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "value": contract.system_prompt}],
            "loss_weight": 0.0,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "value": str(runtime.state.data["shot_requirement"])}
            ],
            "loss_weight": 0.0,
        },
    ]
    for kind, event in _merged_events(events, host_events, require_complete=False):
        event_id = str(event["event_id"])
        if kind == "tool":
            call = event["tool_call"]
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "value": canonical_json(
                                {"name": call["name"], "arguments": call["arguments"]}
                            ),
                        }
                    ],
                    "loss_weight": 0.0,
                }
            )
            blocks = _observation_content(
                "trusted_tool_observation",
                str(call["name"]),
                event["observation"],
                media_by_event.get(event_id, ()),
            )
            messages.append(
                {
                    "role": "tool",
                    "content": _convert_observation_blocks(blocks, private_by_id),
                    "loss_weight": 0.0,
                }
            )
        else:
            blocks = _observation_content(
                "trusted_runtime_observation",
                str(event["name"]),
                event["observation"],
            )
            messages.append(
                {
                    "role": "user",
                    "content": _convert_observation_blocks(blocks, private_by_id),
                    "loss_weight": 0.0,
                }
            )

    # Rejected calls and malformed native outputs are appended as local repair
    # turns; the frozen system prompt and tool contract remain unchanged.
    for rejected in repair_context:
        result = rejected.get("result")
        if rejected.get("kind") == "model_output_rejected":
            raw_output = rejected.get("raw_output")
            if not isinstance(raw_output, str) or not isinstance(result, Mapping):
                raise NativePolicyError(
                    "checkpoint_invalid", "model-output repair context is malformed"
                )
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "value": raw_output}],
                        "loss_weight": 0.0,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "value": canonical_json(
                                    {
                                        "trusted_runtime_observation": {
                                            "name": "native_tool_call_validation",
                                            "result": result,
                                        }
                                    }
                                ),
                            }
                        ],
                        "loss_weight": 0.0,
                    },
                ]
            )
            continue

        call = rejected.get("tool_call")
        if not isinstance(call, Mapping) or not isinstance(result, Mapping):
            raise NativePolicyError("checkpoint_invalid", "repair context is malformed")
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "value": canonical_json(
                                {"name": call["name"], "arguments": call["arguments"]}
                            ),
                        }
                    ],
                    "loss_weight": 0.0,
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "text",
                            "value": canonical_json(
                                {
                                    "trusted_tool_observation": {
                                        "name": call["name"],
                                        "result": result,
                                    }
                                }
                            ),
                        }
                    ],
                    "loss_weight": 0.0,
                },
            ]
        )
    return messages


def _training_renderer_specials(tokenizer: Any) -> list[str]:
    return sorted(
        (
            str(token)
            for token in tokenizer.added_tokens_decoder.values()
            if getattr(token, "special", False) and len(str(token)) >= 2
        ),
        key=len,
        reverse=True,
    )


def _training_renderer_escape(
    text: str, specials: Sequence[str], special_ids: set[int], tokenizer: Any
) -> str:
    if not any(token in text for token in specials):
        return text
    escaped = text
    for token in specials:
        if token in escaped:
            escaped = escaped.replace(token, token[0] + "\u200b" + token[1:])
    if special_ids.intersection(
        tokenizer(escaped, add_special_tokens=False)["input_ids"]
    ):
        raise ValueError("special-token escape failed for model-controlled rendering")
    return escaped


def _training_renderer_escape_messages(
    messages: Sequence[Mapping[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    specials = _training_renderer_specials(tokenizer)
    special_ids = {
        int(token_id)
        for token_id, token in tokenizer.added_tokens_decoder.items()
        if getattr(token, "special", False)
    }
    if not specials:
        return _json_copy(list(messages))
    escaped: list[dict[str, Any]] = []
    for message in messages:
        content_rows: list[dict[str, Any]] = []
        for content in message["content"]:
            row = _json_copy(dict(content))
            if row["type"] in {"text", "reasoning"}:
                row["value"] = _training_renderer_escape(
                    str(row["value"]), specials, special_ids, tokenizer
                )
            elif row["type"] == "tool_call":
                try:
                    call = json.loads(row["value"])
                except (json.JSONDecodeError, TypeError):
                    pass
                else:
                    if isinstance(call, dict) and isinstance(call.get("arguments"), dict):
                        call["arguments"] = {
                            key: (
                                _training_renderer_escape(
                                    value, specials, special_ids, tokenizer
                                )
                                if isinstance(value, str)
                                else value
                            )
                            for key, value in call["arguments"].items()
                        }
                        row["value"] = json.dumps(call)
            content_rows.append(row)
        escaped.append({**_json_copy(dict(message)), "content": content_rows})
    return escaped


def _training_renderer_hf_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    hf_messages: list[dict[str, Any]] = []
    for message in messages:
        hf_content: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_content = ""
        for content in message["content"]:
            content_type = content["type"]
            if content_type == "text":
                hf_content.append({"type": "text", "text": content["value"]})
            elif content_type == "reasoning":
                reasoning_content += str(content["value"])
            elif content_type == "tool_call":
                call = json.loads(content["value"])
                if (
                    not isinstance(call, dict)
                    or "name" not in call
                    or "arguments" not in call
                ):
                    raise ValueError("tool_call must contain name and arguments")
                tool_calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                )
            elif content_type == "image_url":
                hf_content.append({"type": "image", "image": content["value"]})
            elif content_type == "video_url":
                hf_content.append({"type": "video", "video": content["value"]})
            elif content_type == "audio_url":
                hf_content.append({"type": "audio", "audio": content["value"]})
        hf_message: dict[str, Any] = {
            "role": message["role"],
            "content": hf_content,
        }
        if tool_calls:
            hf_message["tool_calls"] = tool_calls
        if reasoning_content:
            hf_message["reasoning_content"] = reasoning_content
        hf_messages.append(hf_message)
    return hf_messages


def _training_renderer_media(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], list[Any], list[Any]]:
    images: list[Any] = []
    videos: list[Any] = []
    audios: list[Any] = []
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image_url":
                images.append(content["value"])
            elif content["type"] == "video_url":
                videos.append(content["value"])
            elif content["type"] == "audio_url":
                audios.append(content["value"])
    return images, videos, audios


def _check_training_renderer_placeholders(
    processor: Any,
    tokenizer: Any,
    rendered_text: str,
    *,
    image_count: int,
    video_count: int,
    audio_count: int,
) -> None:
    for attribute, count, kind in (
        ("image_token_id", image_count, "image"),
        ("video_token_id", video_count, "video"),
        ("audio_token_id", audio_count, "audio"),
    ):
        token_id = getattr(processor, attribute, None)
        if token_id is None:
            token_id = getattr(tokenizer, attribute, None)
        if token_id is None:
            continue
        placeholder = tokenizer.convert_ids_to_tokens(token_id)
        if rendered_text.count(placeholder) != count:
            raise ValueError(
                f"{kind} placeholder count does not match the number of {kind} blocks"
            )


class CheckpointChatTemplateRenderer:
    """Render the training wire using only Transformers and checkpoint assets.

    The implementation is intentionally local: benchmark execution never imports
    the mutable LLaMA-Factory source checkout.  ``validate_renderer_parity`` pins
    its output to an oracle produced by the training renderer at ``source_commit``.
    """

    source_commit = TRAINING_RENDERER_SOURCE_COMMIT

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    @staticmethod
    def _load_audios(values: Sequence[Any], sampling_rate: int) -> list[Any]:
        import numpy as np
        import torchaudio

        results: list[Any] = []
        for value in values:
            if isinstance(value, np.ndarray):
                results.append(value)
                continue
            waveform, source_rate = torchaudio.load(value)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if source_rate != sampling_rate:
                waveform = torchaudio.functional.resample(
                    waveform, source_rate, sampling_rate
                )
            results.append(waveform.squeeze(0).numpy())
        return results

    def render_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools_text: str | None = None,
        *,
        is_generate: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tokenizer = self.processor.tokenizer
        escaped = _training_renderer_escape_messages(messages, tokenizer)
        hf_messages = _training_renderer_hf_messages(escaped)
        tools = None
        if tools_text:
            specials = _training_renderer_specials(tokenizer)
            special_ids = {
                int(token_id)
                for token_id, token in tokenizer.added_tokens_decoder.items()
                if getattr(token, "special", False)
            }
            tools = json.loads(
                _training_renderer_escape(
                    tools_text, specials, special_ids, tokenizer
                )
            )
            if not isinstance(tools, list):
                tools = [tools]
        rendered_text = self.processor.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=is_generate,
            tools=tools,
            **kwargs,
        )
        images, videos, audios = _training_renderer_media(escaped)
        outputs: Mapping[str, Any] | None = None
        if images or videos or audios:
            _check_training_renderer_placeholders(
                self.processor,
                tokenizer,
                rendered_text,
                image_count=len(images),
                video_count=len(videos),
                audio_count=len(audios),
            )
            processor_kwargs: dict[str, Any] = {
                "text": rendered_text,
                "return_tensors": "pt",
            }
            if images:
                processor_kwargs["images"] = images
            if videos:
                processor_kwargs["videos"] = videos
            if audios:
                processor_kwargs["audio"] = self._load_audios(
                    audios, self.processor.feature_extractor.sampling_rate
                )
            outputs = self.processor(**processor_kwargs)
            input_ids = outputs["input_ids"][0].tolist()
        else:
            input_ids = tokenizer(
                rendered_text, add_special_tokens=False
            )["input_ids"]
        result: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(input_ids),
            "loss_weights": [0.0] * len(input_ids),
        }
        if outputs is not None:
            for key in _MULTIMODAL_PASSTHROUGH_KEYS:
                if key in outputs:
                    result[key] = outputs[key]
            mm_type_ids = (
                outputs["mm_token_type_ids"][0].tolist()
                if "mm_token_type_ids" in outputs
                else None
            )
            for attribute, marker in (
                ("image_token_id", 1),
                ("video_token_id", 2),
                ("audio_token_id", 3),
            ):
                token_id = getattr(self.processor, attribute, None)
                if token_id is None:
                    token_id = getattr(tokenizer, attribute, None)
                if token_id is None or token_id not in input_ids:
                    continue
                if mm_type_ids is not None and marker in mm_type_ids:
                    continue
                if mm_type_ids is None:
                    mm_type_ids = [0] * len(input_ids)
                mm_type_ids = [
                    marker if item == token_id else token_type
                    for item, token_type in zip(input_ids, mm_type_ids)
                ]
            if mm_type_ids is not None:
                result["mm_token_type_ids"] = mm_type_ids
        return result


# Compatibility for any already-imported local callers; new evidence identifies
# the implementation by the checkpoint/Transformers name above.
FrozenTrainingRenderer = CheckpointChatTemplateRenderer


def _snapshot_validation_row(contract: SnapshotContract, row_index: int) -> dict[str, Any]:
    validation_path = contract.root / "validation.jsonl"
    if not validation_path.is_file():
        raise NativePolicyError(
            "renderer_fixture_missing", f"missing renderer fixture source: {validation_path}"
        )
    expected_source_hash = str(RENDERER_PARITY_FIXTURE["validation_jsonl_sha256"])
    if sha256_file(validation_path) != expected_source_hash:
        raise NativePolicyError(
            "renderer_fixture_mismatch",
            "validation.jsonl does not match the sealed clean-snapshot fixture",
        )
    with validation_path.open("r", encoding="utf-8") as handle:
        for current_index, line in enumerate(handle):
            if current_index == row_index:
                value = json.loads(line)
                if not isinstance(value, dict):
                    break
                return value
    raise NativePolicyError(
        "renderer_fixture_missing", f"validation row {row_index} is unavailable"
    )


def validate_renderer_parity(
    *, processor: Any, contract: SnapshotContract, model_path: str | Path
) -> dict[str, Any]:
    """Verify checkpoint chat-template tokenization against the frozen SFT oracle."""

    model_root = Path(model_path).expanduser().resolve()
    chat_template_path = model_root / "chat_template.jinja"
    if not chat_template_path.is_file():
        raise NativePolicyError(
            "checkpoint_chat_template_missing",
            f"checkpoint has no chat_template.jinja: {model_root}",
        )
    fixture = RENDERER_PARITY_FIXTURE
    manifest = _strict_json(contract.root / "snapshot_manifest.json")
    if manifest.get("schema_version") != fixture["snapshot_schema_version"]:
        raise NativePolicyError(
            "renderer_fixture_mismatch", "snapshot schema does not match renderer fixture"
        )
    row = _snapshot_validation_row(contract, int(fixture["row_index"]))
    prefix = row.get("messages")
    tools_text = row.get("tools")
    prefix_length = int(fixture["message_prefix_length"])
    if not isinstance(prefix, list) or not isinstance(tools_text, str):
        raise NativePolicyError("renderer_fixture_mismatch", "renderer fixture row is malformed")
    prefix = prefix[:prefix_length]
    if (
        sha256_json(prefix) != fixture["message_prefix_sha256"]
        or sha256_text(tools_text) != fixture["tools_text_sha256"]
        or tools_text != contract.tools_text
    ):
        raise NativePolicyError(
            "renderer_fixture_mismatch", "renderer fixture inputs differ from the clean snapshot"
        )
    rendered = CheckpointChatTemplateRenderer(processor).render_messages(
        prefix, tools_text, **fixture["render_kwargs"]
    )
    input_ids = [int(value) for value in rendered["input_ids"]]
    if (
        len(input_ids) != fixture["input_token_count"]
        or sha256_json(input_ids) != fixture["input_ids_sha256"]
        or input_ids[:8] != fixture["input_ids_prefix"]
        or input_ids[-8:] != fixture["input_ids_suffix"]
    ):
        raise NativePolicyError(
            "renderer_parity_failure",
            "checkpoint chat-template rendering differs from the frozen training oracle",
        )
    return {
        "status": "passed",
        "fixture_schema_version": fixture["schema_version"],
        "reference_implementation": fixture["reference_implementation"],
        "reference_commit": fixture["reference_commit"],
        "validation_jsonl_sha256": fixture["validation_jsonl_sha256"],
        "row_index": fixture["row_index"],
        "message_prefix_length": fixture["message_prefix_length"],
        "input_token_count": len(input_ids),
        "input_ids_sha256": sha256_json(input_ids),
        "checkpoint_chat_template_sha256": _file_sha256(chat_template_path),
    }


def configure_media_processor(
    processor: Any,
    *,
    image_max_pixels: int,
    video_max_pixels_per_frame: int,
    video_max_frames: int,
) -> None:
    """Apply the frozen image/video budget to the checkpoint processor."""

    processor.image_processor.size["longest_edge"] = int(image_max_pixels)
    processor.video_processor.max_frames = int(video_max_frames)
    processor.video_processor.size["longest_edge"] = (
        int(video_max_pixels_per_frame) * int(video_max_frames)
    )
    # Transformers < 5.22 defaults this compatibility switch to False.  Without
    # setting it explicitly, ``longest_edge`` is treated as a whole-video budget
    # during some code paths and as a per-frame budget during others.
    processor.video_processor.cap_pixels_per_frame = True


class LocalQwen35NativePolicy:
    """One local Qwen3.5 model using the frozen training renderer and tool wire."""

    def __init__(
        self,
        model_path: str | Path,
        contract: SnapshotContract,
        *,
        device: str,
        dtype: str,
        max_new_tokens: int,
        image_max_pixels: int,
        video_max_pixels_per_frame: int,
        video_max_frames: int,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.contract = contract
        self.identity = model_identity(model_path)
        self.device_name = device
        self.max_new_tokens = int(max_new_tokens)
        self._torch = torch
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        configure_media_processor(
            processor,
            image_max_pixels=image_max_pixels,
            video_max_pixels_per_frame=video_max_pixels_per_frame,
            video_max_frames=video_max_frames,
        )
        torch_dtype: Any = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch_dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map={"": device},
        ).eval()
        self.processor = processor
        self.renderer = CheckpointChatTemplateRenderer(processor)
        self.device = next(self.model.parameters()).device
        self.im_end_id = int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.eos_ids = sorted(
            {self.im_end_id, int(processor.tokenizer.eos_token_id)}
        )
        self.pad_token_id = int(processor.tokenizer.pad_token_id)
        self.schema_by_name = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in contract.tools
        }
        self._lock = threading.Lock()

    def generate(self, messages: Sequence[Mapping[str, Any]]) -> ModelTurn:
        with self._lock:
            rendered = self.renderer.render_messages(
                list(messages),
                self.contract.tools_text,
                is_generate=True,
                enable_thinking=False,
            )
            model_inputs = model_inputs_to_device(rendered, self.device)
            prompt_tokens = int(model_inputs["input_ids"].shape[-1])
            started = time.monotonic()
            with self._torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.eos_ids,
                    pad_token_id=self.pad_token_id,
                )
            elapsed = time.monotonic() - started
            generated = output_ids[0, prompt_tokens:].detach().cpu()
            raw_output = self.processor.tokenizer.decode(
                generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            parsed, format_errors = parse_qwen35_tool_call(raw_output)
            diagnostics = {
                "format_errors": format_errors,
                "schema_errors": [],
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(generated.numel()),
                "elapsed_seconds": round(elapsed, 3),
                "ended_on_im_end": bool(
                    generated.numel() and int(generated[-1]) == self.im_end_id
                ),
                "input_messages_sha256": sha256_json(list(messages)),
                "input_message_count": len(messages),
            }
            if parsed is None or format_errors:
                raise NativeModelOutputError(
                    "model_wire_format",
                    "model did not emit exactly one native Qwen3.5 tool call",
                    raw_output=raw_output,
                    diagnostics=diagnostics,
                )
            schema = self.schema_by_name.get(str(parsed.get("name") or ""))
            if schema is None:
                diagnostics["schema_errors"] = ["tool is not in snapshot tools.json"]
                raise NativeModelOutputError(
                    "model_unknown_tool",
                    f"model called unknown tool {parsed.get('name')!r}",
                    raw_output=raw_output,
                    diagnostics=diagnostics,
                )
            schema_errors = sorted(
                Draft202012Validator(schema).iter_errors(parsed["arguments"]),
                key=lambda error: list(error.path),
            )
            if schema_errors:
                diagnostics["schema_errors"] = [error.message for error in schema_errors]
                raise NativeModelOutputError(
                    "model_schema_invalid",
                    "model tool arguments fail the snapshot JSON Schema",
                    raw_output=raw_output,
                    diagnostics=diagnostics,
                )
            return ModelTurn(
                raw_output=raw_output,
                tool_call=_json_copy(parsed),
                prompt_tokens=prompt_tokens,
                generated_tokens=int(generated.numel()),
                elapsed_seconds=round(elapsed, 3),
                ended_on_im_end=diagnostics["ended_on_im_end"],
                input_messages_sha256=diagnostics["input_messages_sha256"],
                input_message_count=len(messages),
            )


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyReplica:
    """One independently loaded policy model and its scheduling identity."""

    replica_id: int
    device: str
    model: Any


class PolicyReplicaPool:
    """Dynamically schedule model turns across independent policy replicas.

    A replica is removed from the bounded queue for the complete ``generate``
    call and is returned in ``finally``.  Consequently one replica can never
    execute two turns concurrently, while unrelated stories can infer on
    different GPUs at the same time.  Scheduling individual turns rather than
    pinning an entire story also lets a replica serve another story while the
    first one is waiting for Echo.
    """

    def __init__(self, replicas: Sequence[PolicyReplica]) -> None:
        self._replicas = tuple(replicas)
        if not self._replicas:
            raise ValueError("at least one policy replica is required")
        expected_ids = tuple(range(len(self._replicas)))
        actual_ids = tuple(replica.replica_id for replica in self._replicas)
        if actual_ids != expected_ids:
            raise ValueError("policy replica IDs must be contiguous from zero")
        self._available: queue.Queue[PolicyReplica] = queue.Queue(
            maxsize=len(self._replicas)
        )
        for replica in self._replicas:
            self._available.put_nowait(replica)

    @property
    def replica_count(self) -> int:
        return len(self._replicas)

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(replica.device for replica in self._replicas)

    def generate(self, messages: Sequence[Mapping[str, Any]]) -> ModelTurn:
        replica = self._available.get()
        try:
            return replica.model.generate(messages)
        finally:
            self._available.put_nowait(replica)


@dataclasses.dataclass(frozen=True, slots=True)
class _SeededCompilation:
    delegate: Any
    seed: int

    @property
    def document(self) -> dict[str, Any]:
        return self.delegate.document

    @property
    def reservation_input(self) -> dict[str, Any]:
        return self.delegate.reservation_input

    @property
    def echo_request_template(self) -> dict[str, Any]:
        value = self.delegate.echo_request_template
        value["seed"] = self.seed
        return value

    def reserve(self, runtime: Any) -> dict[str, Any]:
        return self.delegate.reserve(runtime)

    def build_invocation_request(self, reservation: Mapping[str, Any]) -> dict[str, Any]:
        value = self.delegate.build_invocation_request(reservation)
        value["seed"] = self.seed
        return value


class SeededBenchmarkAdapter:
    def __init__(
        self,
        delegate: Echo15BackendAdapter,
        *,
        base_seed: int,
        story_offsets: Mapping[str, int],
    ) -> None:
        self.delegate = delegate
        self.base_seed = int(base_seed)
        self.story_offsets = {str(key): int(value) for key, value in story_offsets.items()}

    def compile(self, *, state: Any, request: Any, **kwargs: Any) -> _SeededCompilation:
        data = getattr(state, "data", state)
        story_id = str(data.get("source_story_id") or "")
        if story_id not in self.story_offsets or request.sequence_index is None:
            raise NativePolicyError(
                "seed_contract_invalid", "cannot resolve canonical global-shot seed"
            )
        seed = self.base_seed + self.story_offsets[story_id] + int(request.sequence_index)
        if seed < 0 or seed > 2**31 - 1:
            raise NativePolicyError("seed_contract_invalid", f"seed is out of range: {seed}")
        return _SeededCompilation(
            self.delegate.compile(state=state, request=request, **kwargs), seed
        )


class SeededEcho15WireTransport(Echo15WireTransport):
    """Benchmark-only transport accepting a seed already frozen by the adapter."""

    def _wire_payload(
        self,
        request: Mapping[str, Any],
        *,
        reserved_canonical_conditions: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        value = _json_copy(dict(request))
        seed = value.pop("seed", None)
        if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
            raise CollectorError("seeded benchmark Echo request has an invalid seed")
        payload, media = super()._wire_payload(
            value, reserved_canonical_conditions=reserved_canonical_conditions
        )
        payload["seed"] = seed
        return payload, media

    def invoke(self, request: Mapping[str, Any], **kwargs: Any) -> Any:
        result = super().invoke(request, **kwargs)
        seed = int(request["seed"])
        receipt = {**result.receipt, "seed_present": True, "seed": seed}
        output_dir = Path(kwargs["output_dir"]).expanduser().resolve()
        # The parent transport writes the canonical receipt with ``safe_name``.
        # Reuse exactly that filename so adding the frozen seed updates the same
        # durable sidecar even when generation IDs contain ':' characters.
        receipt_path = output_dir / f"{safe_name(result.generation_id)}.echo_receipt.json"
        _atomic_json(receipt_path, receipt)
        return dataclasses.replace(result, receipt=receipt)


class _PassthroughOnlyTextClient:
    def generate_text(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("passthrough benchmark must never call prompt rewriting")


class _ImageCreationDisabled:
    """Defense-in-depth guard for the benchmark's zero image-creation budget."""

    def create(self, request: Any) -> Any:
        raise NativePolicyError(
            "image_creation_protocol_violation",
            f"generate_image reached its provider despite a zero attempt budget: {request.request_id}",
        )


class NativePolicyDriver:
    def __init__(
        self,
        model: Any,
        contract: SnapshotContract,
        *,
        max_rejected_calls: int,
    ) -> None:
        self.model = model
        self.contract = contract
        self.max_rejected_calls = int(max_rejected_calls)

    @staticmethod
    def _pending_committed(runtime: MemoryPolicyRuntime, pending: Mapping[str, Any]) -> bool:
        events = runtime.state.data.get("events") or []
        event_count_before = pending.get("runtime_event_count_before")
        if type(event_count_before) is not int or event_count_before < 0:
            raise NativePolicyError(
                "checkpoint_invalid", "pending policy call has no runtime event boundary"
            )
        if len(events) == event_count_before:
            return False
        if len(events) != event_count_before + 1:
            raise NativePolicyError(
                "checkpoint_invalid", "runtime advanced by an unexpected number of events"
            )
        if events[event_count_before].get("tool_call") != pending.get("tool_call"):
            raise NativePolicyError(
                "checkpoint_invalid", "runtime event does not match the pending policy call"
            )
        return True

    @staticmethod
    def _turn(host: dict[str, Any], turn_id: str) -> dict[str, Any]:
        matches = [row for row in host.get("model_turns") or [] if row.get("turn_id") == turn_id]
        if len(matches) != 1:
            raise NativePolicyError("checkpoint_invalid", f"missing model turn {turn_id}")
        return matches[0]

    def _raise_if_repair_exhausted(
        self, repair_context: Sequence[Mapping[str, Any]]
    ) -> None:
        if len(repair_context) < self.max_rejected_calls:
            return
        model_output_failure = (
            repair_context
            and repair_context[-1].get("kind") == "model_output_rejected"
        )
        raise NativePolicyError(
            (
                "model_output_repair_exhausted"
                if model_output_failure
                else "runtime_rejection_exhausted"
            ),
            (
                f"model produced {len(repair_context)} consecutive invalid outputs"
                if model_output_failure
                else f"model produced {len(repair_context)} consecutive rejected tool calls"
            ),
        )

    def step(
        self,
        *,
        runtime: MemoryPolicyRuntime,
        host: dict[str, Any],
        save_host: Callable[[], None],
    ) -> None:
        pending = host.get("pending_policy")
        if isinstance(pending, Mapping) and self._pending_committed(runtime, pending):
            row = self._turn(host, str(pending["turn_id"]))
            row["status"] = "committed_recovered"
            row["runtime_event_id"] = runtime.state.data["events"][-1]["event_id"]
            host["pending_policy"] = None
            host["repair_context"] = []
            save_host()
            return

        if not isinstance(pending, Mapping):
            repairs = host.get("repair_context") or []
            if not isinstance(repairs, list) or not all(
                isinstance(item, Mapping) for item in repairs
            ):
                raise NativePolicyError(
                    "checkpoint_invalid", "repair context must be an array of objects"
                )
            self._raise_if_repair_exhausted(repairs)
            messages = build_native_messages(
                runtime=runtime,
                contract=self.contract,
                repair_context=repairs,
            )
            try:
                generated = self.model.generate(messages)
            except NativeModelOutputError as exc:
                ordinal = len(host.setdefault("model_turns", [])) + 1
                host["model_turns"].append(
                    {
                        "turn_id": f"model_turn_{ordinal:06d}",
                        "phase": runtime.phase,
                        "status": "invalid_model_output",
                        "error_category": exc.category,
                        "error_message": str(exc),
                        "raw_output": exc.raw_output,
                        **_json_copy(exc.diagnostics),
                    }
                )
                host.setdefault("error_counts", {})[exc.category] = (
                    int(host["error_counts"].get(exc.category, 0)) + 1
                )
                repairs = list(repairs)
                repairs.append(
                    {
                        "kind": "model_output_rejected",
                        "raw_output": exc.raw_output,
                        "result": {
                            "ok": False,
                            "error": exc.category,
                            "message": str(exc),
                            "format_errors": _json_copy(
                                exc.diagnostics.get("format_errors") or []
                            ),
                            "schema_errors": _json_copy(
                                exc.diagnostics.get("schema_errors") or []
                            ),
                            "instruction": (
                                "Return exactly one native Qwen3.5 tool call using a "
                                "supplied tool and arguments that satisfy its JSON Schema."
                            ),
                        },
                    }
                )
                host["repair_context"] = repairs
                save_host()
                self._raise_if_repair_exhausted(repairs)
                return
            ordinal = len(host.setdefault("model_turns", [])) + 1
            turn_id = f"model_turn_{ordinal:06d}"
            runtime_event_count_before = len(runtime.state.data.get("events") or [])
            row = {
                "turn_id": turn_id,
                "phase": runtime.phase,
                "status": "pending_runtime",
                "runtime_event_count_before": runtime_event_count_before,
                "raw_output": generated.raw_output,
                "tool_call": _json_copy(generated.tool_call),
                "prompt_tokens": generated.prompt_tokens,
                "generated_tokens": generated.generated_tokens,
                "elapsed_seconds": generated.elapsed_seconds,
                "ended_on_im_end": generated.ended_on_im_end,
                "input_messages_sha256": generated.input_messages_sha256,
                "input_message_count": generated.input_message_count,
            }
            host["model_turns"].append(row)
            pending = {
                "turn_id": turn_id,
                "tool_call": _json_copy(generated.tool_call),
                "runtime_event_count_before": runtime_event_count_before,
            }
            host["pending_policy"] = pending
            save_host()

        call = pending["tool_call"]
        result = runtime.execute(str(call["name"]), call["arguments"])
        row = self._turn(host, str(pending["turn_id"]))
        if result.get("ok") is True:
            row["status"] = "committed"
            row["runtime_event_id"] = runtime.state.data["events"][-1]["event_id"]
            row["runtime_result_sha256"] = sha256_json(
                {key: value for key, value in result.items() if key != "_media_attachments"}
            )
            host["pending_policy"] = None
            host["repair_context"] = []
            save_host()
            return

        public_result = {
            key: _json_copy(value)
            for key, value in result.items()
            if key != "_media_attachments"
        }
        row["status"] = "runtime_rejected"
        row["runtime_error"] = public_result
        host["pending_policy"] = None
        repairs = list(host.get("repair_context") or [])
        repairs.append({"tool_call": _json_copy(call), "result": public_result})
        host["repair_context"] = repairs
        category = "runtime_rejected." + str(public_result.get("error") or "unknown")
        host.setdefault("error_counts", {})[category] = (
            int(host["error_counts"].get(category, 0)) + 1
        )
        save_host()
        self._raise_if_repair_exhausted(repairs)


class QwenBenchmarkStoryCollector(StoryCollector):
    def __init__(
        self,
        *,
        policy_driver: NativePolicyDriver,
        model_contract: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(teacher_client=None, **kwargs)
        self.policy_driver = policy_driver
        self.model_contract = _json_copy(dict(model_contract))
        self.model_contract_sha256 = sha256_json(self.model_contract)

    def _create_or_load_host(
        self, *, shot_dir: Path, raw_prompt: str, pe_mode: str
    ) -> dict[str, Any]:
        existed = self._shot_host_path(shot_dir).is_file()
        host = super()._create_or_load_host(
            shot_dir=shot_dir, raw_prompt=raw_prompt, pe_mode=pe_mode
        )
        if existed:
            if (
                host.get("native_policy_host_version") != SHOT_HOST_VERSION
                or host.get("model_contract_sha256") != self.model_contract_sha256
            ):
                raise NativePolicyError(
                    "checkpoint_conflict", "shot checkpoint belongs to another policy model"
                )
        else:
            host.update(
                {
                    "native_policy_host_version": SHOT_HOST_VERSION,
                    "model_contract_sha256": self.model_contract_sha256,
                    "pending_policy": None,
                    "repair_context": [],
                    "model_turns": [],
                    "error_counts": {},
                }
            )
            self._save_host(shot_dir, host)
        return host

    def _teacher_turn(
        self, *, runtime: MemoryPolicyRuntime, host: dict[str, Any], shot_dir: Path
    ) -> None:
        self.policy_driver.step(
            runtime=runtime,
            host=host,
            save_host=lambda: self._save_host(shot_dir, host),
        )

    def _export_policy_trace(
        self, *, runtime: MemoryPolicyRuntime, host: Mapping[str, Any], shot_dir: Path
    ) -> dict[str, Any]:
        messages = build_native_messages(runtime=runtime, contract=self.policy_driver.contract)
        trace = {
            "schema_version": TRACE_VERSION,
            "runner_version": RUNNER_VERSION,
            "dialogue_scope": "fresh_conversation_per_shot",
            "decoding": {"strategy": "greedy", "do_sample": False},
            "system_prompt_sha256": self.policy_driver.contract.system_prompt_sha256,
            "tool_schema_sha256": self.policy_driver.contract.tool_schema_sha256,
            "tool_names": list(CANONICAL_TOOL_NAMES),
            "model_contract": _json_copy(self.model_contract),
            "session_id": runtime.state.data["session_id"],
            "request_id": runtime.state.data["request_id"],
            "raw_user_request": runtime.state.data["shot_requirement"],
            "raw_request_sha256": runtime.state.data["shot_requirement_sha256"],
            "final_phase": runtime.phase,
            "model_turns": _json_copy(host.get("model_turns") or []),
            "error_counts": _json_copy(host.get("error_counts") or {}),
            "native_messages": messages,
            "runtime_events": _json_copy(runtime.state.data.get("events") or []),
            "host_events": _json_copy(runtime.state.data.get("host_events") or []),
        }
        trace["trace_sha256"] = sha256_json(trace)
        _atomic_json(shot_dir / "policy_trajectory.json", trace)
        return trace

    def run_story(self, source_path: str | Path) -> dict[str, Any]:
        source_value = json.loads(Path(source_path).read_text(encoding="utf-8"))
        story_id = _safe_id(source_value.get("story_id") or Path(source_path).stem, "story")
        story_dir = self.config.output_root / "stories" / story_id
        story_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_story_source(source_path, story_dir=story_dir)
        checkpoint = self._create_or_load_story(
            prepared, story_dir=story_dir, pe_mode="passthrough"
        )
        existing_contract = checkpoint.get("model_contract_sha256")
        if existing_contract not in {None, self.model_contract_sha256}:
            raise NativePolicyError(
                "checkpoint_conflict", "story checkpoint belongs to another policy model"
            )
        checkpoint["model_contract_sha256"] = self.model_contract_sha256
        self._save_story(story_dir, checkpoint)
        if checkpoint.get("status") == "complete":
            complete_path = story_dir / "complete.json"
            if complete_path.is_file():
                return json.loads(complete_path.read_text(encoding="utf-8"))

        started = time.time()
        for shot_index, raw_prompt in enumerate(prepared["prompts"]):
            if shot_index < int(checkpoint["next_shot_index"]):
                continue
            shot_dir = story_dir / "shots" / f"shot_{shot_index:04d}"
            shot_dir.mkdir(parents=True, exist_ok=True)
            host = self._create_or_load_host(
                shot_dir=shot_dir, raw_prompt=raw_prompt, pe_mode="passthrough"
            )
            runtime = self._runtime(
                story_id=story_id,
                shot_index=shot_index,
                shot_dir=shot_dir,
                raw_prompt=raw_prompt,
                registry_items=checkpoint["registry_items"],
                memory_bank=checkpoint["memory_bank"],
                uploaded_evidence=prepared["uploads_by_shot"].get(shot_index, {}),
                previous_evidence=checkpoint["previous_evidence"],
            )
            while runtime.phase != "complete":
                if runtime.phase == "awaiting_generation":
                    self._generation_step(
                        runtime=runtime,
                        host=host,
                        shot_dir=shot_dir,
                        story_id=story_id,
                        shot_index=shot_index,
                        sequence_index=shot_index,
                        raw_prompt=raw_prompt,
                        pe_mode="passthrough",
                    )
                elif runtime.phase == "awaiting_final_evidence":
                    self._final_evidence_step(runtime=runtime, host=host, shot_dir=shot_dir)
                else:
                    if len(host.get("model_turns") or []) >= self.config.max_turns_per_shot:
                        raise NativePolicyError(
                            "turn_limit_exhausted",
                            f"shot {shot_index} exceeded {self.config.max_turns_per_shot} model turns",
                        )
                    self._teacher_turn(runtime=runtime, host=host, shot_dir=shot_dir)

            trace = self._export_policy_trace(runtime=runtime, host=host, shot_dir=shot_dir)
            final_bundle = host.get("final_bundle") or {}
            tail_rows = [
                row
                for row in final_bundle.get("derived_evidence") or []
                if isinstance(row, Mapping)
                and isinstance(row.get("selector"), Mapping)
                and row["selector"].get("mode") == "tail_frame"
            ]
            previous_evidence: dict[str, Any] = {}
            if len(tail_rows) == 1:
                row = _json_copy(tail_rows[0])
                evidence_id = str(row["evidence_id"])
                representation = row["representation"]
                previous_evidence[evidence_id] = {
                    "evidence_id": evidence_id,
                    "source_kind": "sequence_context",
                    "modality": "image",
                    "description": row["description"],
                    "semantic_roles": row.get("semantic_roles") or ["continuity"],
                    "subject_ids": row.get("subject_ids") or [],
                    "default_representation_id": representation["representation_id"],
                    "representations": {
                        representation["representation_id"]: representation
                    },
                }
            selected_id = str(
                (runtime.state.data.get("selected_generation") or {}).get("generation_id")
                or ""
            )
            selected_host = (host.get("generations") or {}).get(selected_id) or {}
            model_turns = host.get("model_turns") or []
            checkpoint["registry_items"] = _json_copy(runtime.state.data["registry"]["items"])
            checkpoint["memory_bank"] = _json_copy(runtime.state.data["memory_bank"])
            checkpoint["previous_evidence"] = previous_evidence
            checkpoint["shots"][str(shot_index)] = {
                "status": "complete",
                "request_id": runtime.state.data["request_id"],
                "policy_trajectory_path": str(
                    (shot_dir / "policy_trajectory.json").resolve()
                ),
                "policy_trajectory_sha256": trace["trace_sha256"],
                "selected_generation_id": selected_id,
                "seed": int(selected_host.get("compilation", {}).get("echo_request_template", {}).get("seed", -1))
                if selected_host.get("compilation")
                else None,
                "output_path": selected_host.get("output_path"),
                "output_sha256": selected_host.get("output_sha256"),
                "model_turns": len(model_turns),
                "policy_inference_seconds": round(
                    sum(float(row.get("elapsed_seconds") or 0.0) for row in model_turns),
                    3,
                ),
                "policy_prompt_tokens": sum(
                    int(row.get("prompt_tokens") or 0) for row in model_turns
                ),
                "policy_generated_tokens": sum(
                    int(row.get("generated_tokens") or 0) for row in model_turns
                ),
                "runtime_rejections": sum(
                    int(value)
                    for key, value in (host.get("error_counts") or {}).items()
                    if str(key).startswith("runtime_rejected.")
                ),
            }
            # The seed is present in the exact invocation request hash and Echo
            # receipt; expose the canonical formula explicitly in the result.
            checkpoint["shots"][str(shot_index)]["seed"] = (
                self.backend_adapter.base_seed
                + self.backend_adapter.story_offsets[story_id]
                + shot_index
            )
            checkpoint["next_shot_index"] = shot_index + 1
            self._save_story(story_dir, checkpoint)

        checkpoint["status"] = "complete"
        checkpoint["elapsed_seconds"] = round(time.time() - started, 3)
        checkpoint["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._save_story(story_dir, checkpoint)
        result = {
            "schema_version": "storymem.stbench.qwen35-story-result.v1",
            "story_id": story_id,
            "status": "complete",
            "source_path": prepared["source_path"],
            "source_sha256": prepared["source_sha256"],
            "num_shots": len(prepared["prompts"]),
            "shots": _json_copy(checkpoint["shots"]),
            "policy_inference_seconds": round(
                sum(
                    float(row.get("policy_inference_seconds") or 0.0)
                    for row in checkpoint["shots"].values()
                ),
                3,
            ),
            "policy_model_turns": sum(
                int(row.get("model_turns") or 0)
                for row in checkpoint["shots"].values()
            ),
            "elapsed_seconds": checkpoint["elapsed_seconds"],
            "completed_at": checkpoint["completed_at"],
        }
        _atomic_json(story_dir / "complete.json", result)
        return result


def _selected_stories(
    stories: Sequence[stbench.StorySpec],
    *,
    story_ids: Sequence[str],
    max_stories: int | None,
) -> list[stbench.StorySpec]:
    selected = list(stories)
    if story_ids:
        wanted = set(story_ids)
        selected = [story for story in selected if story.story_id in wanted]
        missing = wanted - {story.story_id for story in selected}
        if missing:
            raise NativePolicyError("story_selection_invalid", f"unknown story IDs: {sorted(missing)}")
    if max_stories is not None:
        selected = selected[:max_stories]
    if not selected:
        raise NativePolicyError("story_selection_invalid", "no ST-Bench stories selected")
    return selected


def _materialize_source(
    story: stbench.StorySpec, *, output_root: Path, max_shots: int | None
) -> Path:
    if max_shots is None or max_shots >= len(story.prompts):
        return story.prompt_path
    if max_shots < 1:
        raise ValueError("max_shots_per_story must be positive")
    source = json.loads(story.prompt_path.read_text(encoding="utf-8"))
    source["prompts"] = list(story.prompts[:max_shots])
    source["benchmark_subset"] = {
        "source_path": str(story.prompt_path),
        "source_sha256": story.prompt_file_sha256,
        "selected_shots": max_shots,
    }
    path = output_root / "benchmark_inputs" / f"{story.story_id}.json"
    _atomic_json(path, source)
    return path


def _classify_error(exc: BaseException) -> str:
    if isinstance(exc, NativePolicyError):
        return exc.category
    if isinstance(exc, EchoTerminalFailure):
        return "echo_terminal_failure"
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    if "cuda" in message or "out of memory" in message:
        return "model_cuda_failure"
    if "image" in name or "image" in message:
        return "image_provider_failure"
    if "echo" in name or "echo" in message:
        return "echo_transport_failure"
    if "adapter" in name or "compil" in message:
        return "adapter_failure"
    return "unclassified_failure"


def _resolve_policy_devices(
    *, legacy_device: str, policy_devices: Sequence[str]
) -> tuple[str, ...]:
    """Resolve the legacy singleton option and the repeatable replica option."""

    requested = list(policy_devices) if policy_devices else [legacy_device]
    devices = tuple(str(device).strip() for device in requested)
    if any(not device for device in devices):
        raise ValueError("policy devices must be non-empty")
    if len(set(devices)) != len(devices):
        raise ValueError("policy devices must be unique; use one replica per device")
    return devices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--story-id", action="append", default=[])
    parser.add_argument("--max-stories", type=int)
    parser.add_argument("--max-shots-per-story", type=int)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument(
        "--story-workers",
        type=int,
        default=0,
        help="Concurrent stories; 0 uses one worker per configured Echo endpoint.",
    )
    parser.add_argument(
        "--expected-endpoints",
        type=int,
        default=4,
        help="Require this many unique local Echo endpoints; 0 disables the check.",
    )
    parser.add_argument(
        "--device",
        default="cuda:4",
        help=(
            "Legacy single policy device (default: cuda:4). Ignored when one or "
            "more --policy-device options are supplied."
        ),
    )
    parser.add_argument(
        "--policy-device",
        action="append",
        default=[],
        metavar="DEVICE",
        help=(
            "Load one independent policy replica on DEVICE. Repeat for multiple "
            "GPUs; model turns borrow an idle replica from a thread-safe queue."
        ),
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "float32"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-turns-per-shot", type=int, default=48)
    parser.add_argument("--max-rejected-calls", type=int, default=4)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--video-max-pixels-per-frame", type=int, default=65536)
    parser.add_argument("--video-max-frames", type=int, default=12)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.base_seed < 0 or args.base_seed > 2**31 - 1:
        raise ValueError("base_seed must be in [0, 2**31-1]")
    if args.max_stories is not None and args.max_stories < 1:
        raise ValueError("max_stories must be positive")
    if args.max_shots_per_story is not None and args.max_shots_per_story < 1:
        raise ValueError("max_shots_per_story must be positive")
    if args.story_workers < 0:
        raise ValueError("story_workers must be non-negative")
    if args.expected_endpoints < 0:
        raise ValueError("expected_endpoints must be non-negative")
    for name in (
        "max_new_tokens",
        "max_turns_per_shot",
        "max_rejected_calls",
        "image_max_pixels",
        "video_max_pixels_per_frame",
        "video_max_frames",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    policy_devices = _resolve_policy_devices(
        legacy_device=args.device, policy_devices=args.policy_device
    )

    contract = load_snapshot_contract(args.snapshot)
    identity = model_identity(args.model)
    from transformers import AutoProcessor

    parity_processor = AutoProcessor.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    renderer_parity = validate_renderer_parity(
        processor=parity_processor, contract=contract, model_path=args.model
    )
    del parity_processor
    stories = stbench.load_stbench(args.manifest)
    selected = _selected_stories(
        stories, story_ids=args.story_id, max_stories=args.max_stories
    )
    endpoints = stbench.normalize_endpoints(args.endpoint or DEFAULT_ENDPOINTS)
    if args.expected_endpoints and len(endpoints) != args.expected_endpoints:
        raise NativePolicyError(
            "endpoint_contract_invalid",
            f"expected {args.expected_endpoints} unique Echo endpoints, found {len(endpoints)}",
        )
    story_workers = min(args.story_workers or len(endpoints), len(selected))
    if args.fail_fast:
        story_workers = 1
    output_root = args.output_root.expanduser().resolve()
    selected_shots = {
        story.story_id: min(
            len(story.prompts), args.max_shots_per_story or len(story.prompts)
        )
        for story in selected
    }
    run_config = {
        "schema_version": RUN_CONFIG_VERSION,
        "runner_version": RUNNER_VERSION,
        "method_name": args.method_name,
        "model": identity,
        "snapshot": {
            "path": str(contract.root),
            "system_prompt_sha256": contract.system_prompt_sha256,
            "tool_schema_sha256": contract.tool_schema_sha256,
            "tool_names": list(CANONICAL_TOOL_NAMES),
            "tool_count": len(CANONICAL_TOOL_NAMES),
            "tool_availability": {
                name: "registered_in_runtime" for name in CANONICAL_TOOL_NAMES
            },
        },
        "protocol": {
            "dialogue_scope": "fresh_conversation_per_shot",
            "system_prompt_reused_across_shots": True,
            "native_tool_execution": True,
            "decoding": "greedy",
            "prompt_engineering": "passthrough",
            "width": stbench.CANONICAL_WIDTH,
            "height": stbench.CANONICAL_HEIGHT,
            "num_frames": stbench.CANONICAL_FRAMES,
            "fps": stbench.CANONICAL_FPS,
            "base_seed": args.base_seed,
            "seed_formula": "base_seed + canonical_global_shot_index",
            "max_echo_generations_per_shot": 1,
            "condition_revision_budget": 0,
            "image_creation_attempt_limit": IMAGE_CREATION_ATTEMPT_LIMIT,
            "image_creation_provider": "disabled_by_benchmark_protocol",
            "generate_image_schema_retained": "generate_image" in CANONICAL_TOOL_NAMES,
            "external_policy_calls": 0,
        },
        "policy_inference": {
            "implementation": "local_huggingface_transformers",
            "renderer": "checkpoint_chat_template_transformers",
            "renderer_parity": renderer_parity,
            "device": policy_devices[0] if len(policy_devices) == 1 else None,
            "devices": list(policy_devices),
            "dtype": args.dtype,
            "model_instances": len(policy_devices),
            "replica_count": len(policy_devices),
            "replicas": [
                {"replica_id": index, "device": device}
                for index, device in enumerate(policy_devices)
            ],
            "batch_size": 1,
            "batching": "none",
            "max_new_tokens": args.max_new_tokens,
            "max_turns_per_shot": args.max_turns_per_shot,
            "max_rejected_calls": args.max_rejected_calls,
            "image_max_pixels": args.image_max_pixels,
            "video_max_pixels_per_frame": args.video_max_pixels_per_frame,
            "video_max_frames": args.video_max_frames,
            "video_cap_pixels_per_frame": True,
            "scheduling": "dynamic_model_turn_replica_queue",
            "synchronization": "thread_safe_bounded_queue_and_per_replica_lock",
            "max_parallel_generate_calls": min(
                len(policy_devices), story_workers
            ),
            "timing_source": "sum(model_turns[*].elapsed_seconds) in policy trajectories",
        },
        "echo_scheduling": {
            "story_workers": story_workers,
            "endpoint_count": len(endpoints),
            "global_concurrency": min(story_workers, len(endpoints)),
            "per_endpoint_concurrency": 1,
            "new_request_routing": "idle_round_robin_v1",
            "resume_routing": "persisted_endpoint_binding",
        },
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "selected_stories": [story.story_id for story in selected],
        "selected_shots": selected_shots,
        "endpoints": list(endpoints),
    }
    run_config["run_config_sha256"] = sha256_json(run_config)
    if args.validate_only:
        print(json.dumps({"ok": True, "run_config": run_config}, ensure_ascii=False))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".runner.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise NativePolicyError("run_locked", f"another process owns {output_root}") from exc
        config_path = output_root / "run_config.json"
        if config_path.is_file():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != run_config:
                raise NativePolicyError("run_config_conflict", "output root has another run config")
        else:
            _atomic_json(config_path, run_config)

        echo = SeededEcho15WireTransport(
            endpoints, concurrency=story_workers, per_endpoint_concurrency=1
        )
        echo.preflight(require_audio_memory=True)
        visual_creator = _ImageCreationDisabled()
        replicas: list[PolicyReplica] = []
        for replica_id, device in enumerate(policy_devices):
            print(
                json.dumps(
                    {
                        "event": "policy_replica_load_start",
                        "replica_id": replica_id,
                        "device": device,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            replica_model = LocalQwen35NativePolicy(
                args.model,
                contract,
                device=device,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                image_max_pixels=args.image_max_pixels,
                video_max_pixels_per_frame=args.video_max_pixels_per_frame,
                video_max_frames=args.video_max_frames,
            )
            replicas.append(
                PolicyReplica(
                    replica_id=replica_id,
                    device=device,
                    model=replica_model,
                )
            )
            print(
                json.dumps(
                    {
                        "event": "policy_replica_load_complete",
                        "replica_id": replica_id,
                        "device": device,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        policy_pool = PolicyReplicaPool(replicas)
        offsets = {story.story_id: story.global_shot_start for story in stories}
        adapter = SeededBenchmarkAdapter(
            Echo15BackendAdapter.from_official_v7(
                artifact_root=output_root / "adapter",
                gemini=_PassthroughOnlyTextClient(),
                capabilities=Echo15Capabilities(max_memory_slots=7, audio_memory=True),
            ),
            base_seed=args.base_seed,
            story_offsets=offsets,
        )
        namespace_digest = hashlib.sha256(
            canonical_json(
                {
                    "runner": RUNNER_VERSION,
                    "run_config_sha256": run_config["run_config_sha256"],
                    "output_root": str(output_root),
                }
            ).encode("utf-8")
        ).hexdigest()[:24]
        collector = QwenBenchmarkStoryCollector(
            config=CollectorConfig(
                output_root=output_root,
                width=stbench.CANONICAL_WIDTH,
                height=stbench.CANONICAL_HEIGHT,
                num_frames=stbench.CANONICAL_FRAMES,
                story_workers=story_workers,
                echo_concurrency=story_workers,
                image_concurrency=1,
                teacher_qps=1.0,
                max_turns_per_shot=args.max_turns_per_shot,
                visual_creation_budget=IMAGE_CREATION_ATTEMPT_LIMIT,
                max_condition_revisions=0,
                rewrite_fraction=0.0,
            ),
            policy_driver=NativePolicyDriver(
                policy_pool, contract, max_rejected_calls=args.max_rejected_calls
            ),
            model_contract={
                "model": identity,
                "method_name": args.method_name,
                "system_prompt_sha256": contract.system_prompt_sha256,
                "tool_schema_sha256": contract.tool_schema_sha256,
                "decoding": "greedy",
            },
            visual_evidence_creator=visual_creator,
            backend_adapter=adapter,
            echo=echo,
            run_namespace=f"storymem_run_{namespace_digest}",
        )

        progress_path = output_root / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("run_config_sha256") != run_config["run_config_sha256"]:
                raise NativePolicyError("run_config_conflict", "progress belongs to another run")
        else:
            progress = {
                "schema_version": PROGRESS_VERSION,
                "run_config_sha256": run_config["run_config_sha256"],
                "stories": {},
            }

        sources = {
            story.story_id: _materialize_source(
                story, output_root=output_root, max_shots=args.max_shots_per_story
            )
            for story in selected
        }

        def run_one_story(story: stbench.StorySpec) -> tuple[str, dict[str, Any]]:
            print(
                json.dumps(
                    {"event": "story_start", "story_id": story.story_id},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            started = time.time()
            try:
                result = collector.run_story(sources[story.story_id])
            except Exception as exc:
                category = _classify_error(exc)
                failure = {
                    "status": "failed",
                    "category": category,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
                return story.story_id, failure
            return story.story_id, result

        with ThreadPoolExecutor(
            max_workers=story_workers, thread_name_prefix="stbench-story"
        ) as executor:
            future_to_story = {
                executor.submit(run_one_story, story): story for story in selected
            }
            for future in as_completed(future_to_story):
                story = future_to_story[future]
                story_id, row = future.result()
                progress["stories"][story_id] = row
                _atomic_json(progress_path, progress)
                if row.get("status") == "complete":
                    print(
                        json.dumps(
                            {
                                "event": "story_complete",
                                "story_id": story_id,
                                "shots": row["num_shots"],
                                "elapsed_seconds": row["elapsed_seconds"],
                                "policy_inference_seconds": row.get(
                                    "policy_inference_seconds"
                                ),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                else:
                    print(
                        json.dumps(
                            {
                                "event": "story_failed",
                                "story_id": story_id,
                                "category": row.get("category"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if args.fail_fast:
                        for pending in future_to_story:
                            if pending is not future:
                                pending.cancel()
                        raise NativePolicyError(
                            str(row.get("category") or "unclassified_failure"),
                            f"story {story.story_id} failed: {row.get('message')}",
                        )

        counts = Counter(
            row.get("category")
            for row in progress["stories"].values()
            if row.get("status") == "failed"
        )
        complete = sum(row.get("status") == "complete" for row in progress["stories"].values())
        summary = {
            "schema_version": "storymem.stbench.qwen35-native-policy-summary.v1",
            "run_config_sha256": run_config["run_config_sha256"],
            "stories_requested": len(selected),
            "stories_complete": complete,
            "stories_failed": sum(counts.values()),
            "shots_requested": sum(selected_shots.values()),
            "shots_complete": sum(
                int(row.get("num_shots", 0))
                for row in progress["stories"].values()
                if row.get("status") == "complete"
            ),
            "policy_inference_seconds": round(
                sum(
                    float(row.get("policy_inference_seconds") or 0.0)
                    for row in progress["stories"].values()
                    if row.get("status") == "complete"
                ),
                3,
            ),
            "policy_model_turns": sum(
                int(row.get("policy_model_turns") or 0)
                for row in progress["stories"].values()
                if row.get("status") == "complete"
            ),
            "error_counts": dict(sorted(counts.items())),
            "complete": complete == len(selected),
        }
        _atomic_json(output_root / "summary.json", summary)
        print(json.dumps({"event": "complete", "summary": summary}, ensure_ascii=False), flush=True)
        return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
