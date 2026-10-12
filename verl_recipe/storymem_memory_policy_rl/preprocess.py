#!/usr/bin/env python3
"""Build leak-free single- and multi-shot VERL parquet datasets.

The source run stores the *final* state of every completed shot.  That final
state cannot be used wholesale as the input to the same shot: it may contain
memory written by the reference trajectory.  This exporter reconstructs the
input seed as follows:

* shot 0 gets the registry from ``prepared_source.json`` and an empty memory
  bank;
* shot i gets its registry and memory bank from shot i-1's final state;
* shot-local uploads come from ``prepared_source.json`` (the final state can
  contain session-local inspection/authorization fields), while prior-tail
  evidence, budgets, and policy limits are read from shot i's state.

For multi-shot rows, every shot remains a fresh system+user conversation.  The
first shot is initialized from its source seed.  Subsequent shot specs are
marked so the online loop replaces source registry/memory/previous-evidence
with the preceding *rollout* shot's final values.  Source seeds remain in the
row as reproducible fixtures and evaluator provenance, never as permission to
reset an in-progress multi-shot rollout to the reference branch.

Complex state is encoded as canonical JSON inside a fixed Arrow schema.  This
is intentional: storing memory dictionaries directly would make every opaque
memory ID an Arrow struct field and render a full-data parquet impractical.

Every shot is also rendered with the selected Qwen3.5 chat template as
``system + shot user + canonical nine tools``.  The v2 dataset records that
exact token count and excludes any episode containing an overlong shot.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ECHO_LONGVIDEO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_DIR = Path(__file__).resolve().parent
TOOLS_ROOT = ECHO_LONGVIDEO_ROOT / "tools"
# Direct execution puts this recipe directory (which also contains tools.py)
# at sys.path[0].  Always move the project root to the front so ``tools.*``
# resolves to echo_longvideo/tools rather than this recipe-local module.
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve()
    not in {ECHO_LONGVIDEO_ROOT, RECIPE_DIR, TOOLS_ROOT}
]
sys.path.insert(0, str(ECHO_LONGVIDEO_ROOT))
sys.path.insert(1, str(TOOLS_ROOT))

from verl_recipe.storymem_memory_policy_rl.policy_profile import (  # noqa: E402
    PROMPT_AUTHORING,
    TRAINING_SYSTEM_PROMPT,
    TRAINING_SYSTEM_PROMPT_VERSION,
    training_system_prompt_hash,
    get_tool_schemas,
    tool_schema_hash,
)
from tools.storymem_policy.contract import (  # noqa: E402
    CANONICAL_TOOL_NAMES,
    TOOL_PROTOCOL_VERSION,
)
from tools.storymem_policy.runtime import RUNTIME_VERSION  # noqa: E402
from tools.storymem_policy.state import (  # noqa: E402
    STATE_SCHEMA_VERSION as SOURCE_STATE_SCHEMA_VERSION,
)
from tools.storymem_policy.trajectory import (  # noqa: E402
    TRAJECTORY_SCHEMA_VERSION as SOURCE_TRAJECTORY_SCHEMA_VERSION,
)
from verl_recipe.storymem_memory_policy_rl.contracts import (  # noqa: E402
    AGENT_NAME,
    DATASET_SCHEMA_VERSION,
    DATA_SOURCE_BY_MODE,
    MANIFEST_SCHEMA_VERSION,
    PROMPT_VALIDATION_SCHEMA_VERSION,
    SHOT_SPEC_SCHEMA_VERSION,
    STATE_SEED_ENCODING,
    STATE_SEED_SCHEMA_VERSION,
)


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "sft_snapshot"
# This is the current schema emitted by the source collector, not an RL legacy
# format.  RL artifacts themselves accept only the v2 contracts above.
SOURCE_PROGRESS_SCHEMA_VERSION = "storymem.memory-policy.progress.v1"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MEMORY_ID_RE = re.compile(r"memory_(\d+)\Z")
_VALID_MODALITIES = {"image", "video", "audio", "text"}


class PreprocessError(ValueError):
    """The requested dataset cannot be built without guessing or leakage."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PreprocessError(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def read_stable_json(path: Path, *, label: str) -> tuple[Any, bytes]:
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise PreprocessError(f"cannot read {label}: {path}: {exc}") from exc
    if not payload:
        raise PreprocessError(f"{label} is empty: {path}")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PreprocessError(f"{label} changed while it was read: {path}")
    return strict_json_bytes(payload, label=label), payload


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreprocessError(f"{label} must be an object")
    return value


def require_nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PreprocessError(f"{label} must be a non-negative integer")
    return value


def require_positive_int(value: Any, *, label: str) -> int:
    result = require_nonnegative_int(value, label=label)
    if result == 0:
        raise PreprocessError(f"{label} must be positive")
    return result


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PreprocessError(f"{label} is not a canonical SHA-256 digest")
    return value


def require_file(path_value: Any, *, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise PreprocessError(f"{label} path is missing")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise PreprocessError(f"{label} path is not absolute: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PreprocessError(f"{label} is unavailable: {path}: {exc}") from exc
    if not resolved.is_file():
        raise PreprocessError(f"{label} is not a regular file: {resolved}")
    return resolved


def ranked_story_ids(
    story_ids: Iterable[str], *, seed: str, purpose: str
) -> list[str]:
    return sorted(
        story_ids,
        key=lambda story_id: (
            hashlib.sha256(
                f"{seed}\0{purpose}\0{story_id}".encode("utf-8")
            ).hexdigest(),
            story_id,
        ),
    )


def _memory_counter_floor(memory_bank: Mapping[str, Any]) -> int:
    ordinals = [
        int(match.group(1))
        for key in memory_bank
        for match in [_MEMORY_ID_RE.fullmatch(str(key))]
        if match is not None
    ]
    return max(ordinals, default=0)


def _verify_declared_hash(path: Path, expected: Any, *, label: str) -> None:
    declared = require_sha256(expected, label=f"{label} declared hash")
    actual = sha256_file(path)
    if actual != declared:
        raise PreprocessError(
            f"{label} hash mismatch: declared={declared} actual={actual}"
        )


def _verify_declared_json_hash(value: Any, expected: Any, *, label: str) -> None:
    declared = require_sha256(expected, label=f"{label} declared hash")
    actual = sha256_json(value)
    if actual != declared:
        raise PreprocessError(
            f"{label} canonical hash mismatch: declared={declared} actual={actual}"
        )


@dataclass(frozen=True, slots=True)
class SourceOptions:
    data_root: Path
    verify_declared_hashes: bool


class PromptLengthValidator:
    """Tokenize the exact fresh-shot prompt used by the online agent loop."""

    def __init__(
        self,
        *,
        model_path: Path,
        tool_config_path: Path,
        max_prompt_length: int,
    ) -> None:
        try:
            import tokenizers
            import transformers
            import yaml
            from transformers import AutoProcessor
        except ImportError as exc:
            raise PreprocessError(
                "prompt validation requires transformers and PyYAML"
            ) from exc

        self.model_path = model_path.expanduser().resolve()
        self.tool_config_path = tool_config_path.expanduser().resolve()
        self.max_prompt_length = max_prompt_length
        if not self.model_path.is_dir():
            raise PreprocessError(f"model directory does not exist: {self.model_path}")
        if not self.tool_config_path.is_file():
            raise PreprocessError(
                f"tool configuration does not exist: {self.tool_config_path}"
            )

        try:
            tool_config = yaml.safe_load(
                self.tool_config_path.read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError) as exc:
            raise PreprocessError(
                f"cannot load tool configuration {self.tool_config_path}: {exc}"
            ) from exc
        rows = tool_config.get("tools") if isinstance(tool_config, Mapping) else None
        if not isinstance(rows, list):
            raise PreprocessError("tool configuration must contain a tools array")
        configured_names = [
            str((row.get("config") or {}).get("tool_name") or "")
            if isinstance(row, Mapping)
            else ""
            for row in rows
        ]
        if configured_names != list(CANONICAL_TOOL_NAMES):
            raise PreprocessError(
                "tool configuration is not the ordered canonical nine-tool contract: "
                f"{configured_names!r}"
            )

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
                use_fast=True,
            )
        except Exception as exc:
            raise PreprocessError(
                f"cannot load multimodal processor from {self.model_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise PreprocessError("selected Qwen3.5 processor has no tokenizer")
        if not callable(getattr(self.processor, "apply_chat_template", None)):
            raise PreprocessError("selected processor has no apply_chat_template method")
        if not callable(self.processor):
            raise PreprocessError("selected processor cannot encode rendered prompts")
        self.transformers_version = str(transformers.__version__)
        self.tokenizers_version = str(tokenizers.__version__)
        self.processor_class = type(self.processor).__name__

        # SFT stores the tool catalog as canonical JSON and the dataset loader
        # parses that string before rendering it.  Preserve the same recursive
        # key order here so the declared prompt-token proof is for the exact
        # prompt used by the online runtime, not just a schema-equivalent one.
        self.tools = json.loads(canonical_json(get_tool_schemas()))
        template = getattr(self.tokenizer, "chat_template", None)
        if isinstance(template, Mapping):
            template_payload = canonical_json(dict(template))
        elif isinstance(template, str) and template:
            template_payload = template
        else:
            raise PreprocessError("selected tokenizer has no non-empty chat template")
        self.chat_template_sha256 = sha256_bytes(template_payload.encode("utf-8"))
        self.tool_config_sha256 = sha256_file(self.tool_config_path)
        artifact_hashes: dict[str, str] = {}
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "processor_config.json",
        ):
            path = self.model_path / name
            if path.is_file():
                artifact_hashes[name] = sha256_file(path)
        if "tokenizer.json" not in artifact_hashes:
            raise PreprocessError(f"tokenizer.json is missing under {self.model_path}")
        self.tokenizer_artifact_hashes = artifact_hashes
        self.tokenizer_contract_sha256 = sha256_bytes(
            canonical_json(
                {
                    "chat_template_sha256": self.chat_template_sha256,
                    "processor_class": self.processor_class,
                    "tokenizer_class": type(self.tokenizer).__name__,
                    "transformers_version": self.transformers_version,
                    "tokenizers_version": self.tokenizers_version,
                    "artifacts": artifact_hashes,
                }
            ).encode("utf-8")
        )

        # A tokenizer can accept ``tools=`` while silently ignoring it.  Reject
        # that case before exporting a dataset whose length proof is invalid.
        rendered = self._apply("prompt validation probe", tokenize=False)
        if not isinstance(rendered, str) or not all(
            name in rendered for name in CANONICAL_TOOL_NAMES
        ):
            raise PreprocessError(
                "Qwen3.5 chat template did not serialize all nine tool schemas"
            )

    def _apply(self, requirement: str, *, tokenize: bool) -> Any:
        messages = [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": requirement},
        ]
        try:
            return self.processor.apply_chat_template(
                messages,
                tools=self.tools,
                add_generation_prompt=True,
                tokenize=tokenize,
                enable_thinking=False,
            )
        except Exception as exc:
            raise PreprocessError(
                f"Qwen3.5 chat-template rendering failed: {type(exc).__name__}: {exc}"
            ) from exc

    def count(self, requirement: str) -> int:
        rendered = self._apply(requirement, tokenize=False)
        try:
            value = self.processor(
                text=[rendered],
                images=None,
                videos=None,
                return_tensors="pt",
            )["input_ids"]
        except Exception as exc:
            raise PreprocessError(
                f"Qwen3.5 processor failed to tokenize the rendered prompt: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, Mapping):
            value = value.get("input_ids")
            if hasattr(value, "tolist"):
                value = value.tolist()
        if (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], list)
        ):
            value = value[0]
        if not isinstance(value, list) or not all(type(token) is int for token in value):
            raise PreprocessError(
                "Qwen3.5 chat template returned an unsupported token-id structure"
            )
        if not value:
            raise PreprocessError("Qwen3.5 chat template returned an empty prompt")
        return len(value)

    def manifest_record(self) -> dict[str, Any]:
        return {
            "schema_version": PROMPT_VALIDATION_SCHEMA_VERSION,
            "model_path": str(self.model_path),
            "processor_class": self.processor_class,
            "tokenizer_class": type(self.tokenizer).__name__,
            "transformers_version": self.transformers_version,
            "tokenizers_version": self.tokenizers_version,
            "tokenizer_contract_sha256": self.tokenizer_contract_sha256,
            "tokenizer_artifact_hashes": dict(self.tokenizer_artifact_hashes),
            "chat_template_sha256": self.chat_template_sha256,
            "tool_config_path": str(self.tool_config_path),
            "tool_config_sha256": self.tool_config_sha256,
            "tool_schema_sha256": tool_schema_hash(),
            "tool_count": len(self.tools),
            "messages": "system + current shot user",
            "add_generation_prompt": True,
            "enable_thinking": False,
            "max_prompt_length": self.max_prompt_length,
        }


def _state_input_seed(
    *,
    current_state: Mapping[str, Any],
    registry_items: Mapping[str, Any],
    memory_bank: Mapping[str, Any],
    uploaded_evidence: Mapping[str, Any],
    previous_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    condition_budget = require_mapping(
        current_state.get("condition_budget"), label="condition_budget"
    )
    policy_limits = require_mapping(
        current_state.get("policy_limits"), label="policy_limits"
    )
    if (policy_limits.get("prompt_authoring") is True) != PROMPT_AUTHORING:
        raise PreprocessError("source policy mode differs from the selected RL profile")
    visual_creation_budget = require_nonnegative_int(
        current_state.get("visual_creation_budget"),
        label="visual_creation_budget",
    )
    max_condition_revisions = require_nonnegative_int(
        current_state.get("max_condition_revisions"),
        label="max_condition_revisions",
    )
    policy_revision_limit = require_nonnegative_int(
        policy_limits.get("max_condition_revisions"),
        label="policy_limits.max_condition_revisions",
    )
    if policy_revision_limit != max_condition_revisions:
        raise PreprocessError(
            "state max_condition_revisions disagrees with policy_limits"
        )
    max_image_reference_inputs = require_nonnegative_int(
        policy_limits.get("max_image_reference_inputs"),
        label="policy_limits.max_image_reference_inputs",
    )
    raw_modalities = policy_limits.get("perceivable_modalities")
    if not isinstance(raw_modalities, list) or not all(
        isinstance(value, str) for value in raw_modalities
    ):
        raise PreprocessError("policy_limits.perceivable_modalities must be a string array")
    if len(set(raw_modalities)) != len(raw_modalities) or set(raw_modalities) - _VALID_MODALITIES:
        raise PreprocessError("policy_limits.perceivable_modalities is invalid")

    # These keys map one-to-one to MemoryPolicyState.create.  The online loop
    # supplies new session/request IDs, a new state path, and shot_requirement.
    return {
        "schema_version": STATE_SEED_SCHEMA_VERSION,
        **({"prompt_authoring": True} if PROMPT_AUTHORING else {}),
        "condition_budget": copy.deepcopy(dict(condition_budget)),
        "registry_items": copy.deepcopy(dict(registry_items)),
        "memory_bank": copy.deepcopy(dict(memory_bank)),
        "uploaded_evidence": copy.deepcopy(dict(uploaded_evidence)),
        "previous_evidence": copy.deepcopy(dict(previous_evidence)),
        "visual_creation_budget": visual_creation_budget,
        "max_condition_revisions": max_condition_revisions,
        "perceivable_modalities": list(raw_modalities),
        "max_image_reference_inputs": max_image_reference_inputs,
        "memory_counter_floor": _memory_counter_floor(memory_bank),
    }


def _validate_final_state(
    state: Any,
    *,
    story_id: str,
    shot_index: int,
    expected_prompt: str,
    expected_request_id: Any,
    label: str,
) -> Mapping[str, Any]:
    row = require_mapping(state, label=label)
    if row.get("schema_version") != SOURCE_STATE_SCHEMA_VERSION:
        raise PreprocessError(
            f"{label} has unsupported schema {row.get('schema_version')!r}"
        )
    if row.get("phase") != "complete":
        raise PreprocessError(f"{label} is not a complete final state")
    if row.get("source_story_id") != story_id:
        raise PreprocessError(f"{label} belongs to another story")
    if row.get("shot_requirement") != expected_prompt:
        raise PreprocessError(f"{label} shot requirement differs from prepared source")
    expected_prompt_hash = sha256_bytes(expected_prompt.encode("utf-8"))
    if row.get("shot_requirement_sha256") != expected_prompt_hash:
        raise PreprocessError(f"{label} shot requirement hash is invalid")
    if expected_request_id is not None and row.get("request_id") != expected_request_id:
        raise PreprocessError(f"{label} request_id differs from progress.json")
    request_id = row.get("request_id")
    if not isinstance(request_id, str) or not request_id.endswith(
        f":shot:{shot_index:04d}"
    ):
        raise PreprocessError(f"{label} request_id has the wrong shot suffix")
    registry = require_mapping(row.get("registry"), label=f"{label}.registry")
    require_mapping(registry.get("items"), label=f"{label}.registry.items")
    require_mapping(row.get("memory_bank"), label=f"{label}.memory_bank")
    return row


def load_story_shot_specs(
    story_id: str,
    record: Mapping[str, Any],
    *,
    options: SourceOptions,
) -> list[dict[str, Any]]:
    """Load one complete source story and reconstruct every pre-shot seed."""

    if record.get("story_id") != story_id or record.get("status") != "complete":
        raise PreprocessError(f"progress record is not a complete story: {story_id}")
    num_shots = require_positive_int(
        record.get("num_shots"), label=f"{story_id}.num_shots"
    )
    shot_records = require_mapping(record.get("shots"), label=f"{story_id}.shots")
    expected_keys = {str(index) for index in range(num_shots)}
    if set(shot_records) != expected_keys:
        raise PreprocessError(f"{story_id} does not have contiguous complete shots")

    story_dir = (options.data_root / "stories" / story_id).resolve()
    prepared_path = story_dir / "source_assets" / "prepared_source.json"
    prepared, _ = read_stable_json(
        prepared_path, label=f"{story_id} prepared source"
    )
    prepared = require_mapping(prepared, label=f"{story_id} prepared source")
    if prepared.get("story_id") != story_id:
        raise PreprocessError(f"prepared source belongs to another story: {story_id}")
    prompts = prepared.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != num_shots or not all(
        isinstance(prompt, str) and prompt for prompt in prompts
    ):
        raise PreprocessError(f"prepared source has invalid prompts: {story_id}")
    prepared_registry = require_mapping(
        prepared.get("registry_items"), label=f"{story_id}.registry_items"
    )
    uploads_by_shot = require_mapping(
        prepared.get("uploads_by_shot"), label=f"{story_id}.uploads_by_shot"
    )

    result: list[dict[str, Any]] = []
    previous_state: Mapping[str, Any] | None = None
    previous_state_path = ""
    for shot_index in range(num_shots):
        shot_label = f"{story_id}/shot_{shot_index:04d}"
        shot_record = require_mapping(
            shot_records[str(shot_index)], label=f"{shot_label} progress record"
        )
        if shot_record.get("status") != "complete":
            raise PreprocessError(f"{shot_label} is not complete")
        shot_dir = story_dir / "shots" / f"shot_{shot_index:04d}"
        state_path = require_file(
            str(shot_dir / "policy_state.json"), label=f"{shot_label} policy state"
        )
        state_value, _ = read_stable_json(state_path, label=f"{shot_label} policy state")
        current_state = _validate_final_state(
            state_value,
            story_id=story_id,
            shot_index=shot_index,
            expected_prompt=prompts[shot_index],
            expected_request_id=shot_record.get("request_id"),
            label=f"{shot_label} policy state",
        )

        if shot_index == 0:
            # Explicitly exclude shot 0's final memory and final registry.
            registry_items: Mapping[str, Any] = prepared_registry
            memory_bank: Mapping[str, Any] = {}
            registry_memory_source_shot_index = -1
            if current_state.get("previous_evidence") != {}:
                raise PreprocessError(f"{shot_label} unexpectedly has previous evidence")
        else:
            if previous_state is None:
                raise AssertionError("previous state was not retained")
            previous_registry = require_mapping(
                previous_state.get("registry"),
                label=f"{story_id}/shot_{shot_index - 1:04d}.registry",
            )
            registry_items = require_mapping(
                previous_registry.get("items"),
                label=f"{story_id}/shot_{shot_index - 1:04d}.registry.items",
            )
            memory_bank = require_mapping(
                previous_state.get("memory_bank"),
                label=f"{story_id}/shot_{shot_index - 1:04d}.memory_bank",
            )
            registry_memory_source_shot_index = shot_index - 1

        source_uploads = uploads_by_shot.get(str(shot_index))
        if source_uploads is None:
            source_uploads = {}
        source_uploads = require_mapping(
            source_uploads, label=f"{shot_label} prepared uploads"
        )
        # Runtime viewing/assessment can add session-local authorization fields
        # to final-state upload records.  The immutable prepared copy is the
        # actual pre-shot input and therefore the only leak-free seed.
        previous_evidence = require_mapping(
            current_state.get("previous_evidence"),
            label=f"{shot_label}.previous_evidence",
        )

        state_seed = _state_input_seed(
            current_state=current_state,
            registry_items=registry_items,
            memory_bank=memory_bank,
            uploaded_evidence=source_uploads,
            previous_evidence=previous_evidence,
        )
        encoded_seed = canonical_json(state_seed)

        trajectory_path = require_file(
            shot_record.get("trajectory_path"), label=f"{shot_label} trajectory"
        )
        trajectory_value, _ = read_stable_json(
            trajectory_path, label=f"{shot_label} trajectory"
        )
        trajectory = require_mapping(
            trajectory_value, label=f"{shot_label} trajectory"
        )
        if trajectory.get("schema_version") != SOURCE_TRAJECTORY_SCHEMA_VERSION:
            raise PreprocessError(
                f"{shot_label} trajectory is not the current source schema"
            )
        if trajectory.get("runtime_version") != RUNTIME_VERSION:
            raise PreprocessError(
                f"{shot_label} trajectory is not from the current runtime"
            )
        if trajectory.get("tool_protocol_version") != TOOL_PROTOCOL_VERSION:
            raise PreprocessError(
                f"{shot_label} trajectory is not from the current tool protocol"
            )
        if trajectory.get("tool_schema_sha256") != tool_schema_hash():
            raise PreprocessError(
                f"{shot_label} trajectory tool schema hash is not current"
            )
        if trajectory.get("final_phase") != "complete":
            raise PreprocessError(f"{shot_label} trajectory is not complete")
        if trajectory.get("raw_user_request") != prompts[shot_index]:
            raise PreprocessError(
                f"{shot_label} trajectory request differs from prepared source"
            )
        if trajectory.get("raw_request_sha256") != sha256_bytes(
            prompts[shot_index].encode("utf-8")
        ):
            raise PreprocessError(
                f"{shot_label} trajectory request hash is invalid"
            )
        if trajectory.get("request_id") != current_state.get("request_id"):
            raise PreprocessError(
                f"{shot_label} trajectory request_id differs from policy state"
            )
        sft_path = require_file(
            shot_record.get("sft_path"), label=f"{shot_label} SFT source"
        )
        output_path = require_file(
            shot_record.get("output_path"), label=f"{shot_label} reference video"
        )
        trajectory_sha256 = require_sha256(
            shot_record.get("trajectory_sha256"),
            label=f"{shot_label} trajectory_sha256",
        )
        sft_sha256 = require_sha256(
            shot_record.get("sft_sha256"), label=f"{shot_label} sft_sha256"
        )
        output_sha256 = require_sha256(
            shot_record.get("output_sha256"), label=f"{shot_label} output_sha256"
        )
        if options.verify_declared_hashes:
            _verify_declared_json_hash(
                trajectory, trajectory_sha256, label=f"{shot_label} trajectory"
            )
            sft_value, _ = read_stable_json(
                sft_path, label=f"{shot_label} SFT source"
            )
            _verify_declared_json_hash(
                sft_value, sft_sha256, label=f"{shot_label} SFT source"
            )
            _verify_declared_hash(
                output_path, output_sha256, label=f"{shot_label} reference video"
            )

        result.append(
            {
                "schema_version": SHOT_SPEC_SCHEMA_VERSION,
                "episode_position": 0,  # Set when a single/window row is built.
                "shot_index": shot_index,
                "shot_requirement": prompts[shot_index],
                "shot_requirement_sha256": str(
                    current_state["shot_requirement_sha256"]
                ),
                "source_story_dir": str(story_dir),
                "source_shot_dir": str(shot_dir.resolve()),
                "source_policy_state_path": str(state_path),
                "source_previous_policy_state_path": previous_state_path,
                "source_trajectory_path": str(trajectory_path),
                "source_trajectory_sha256": trajectory_sha256,
                "source_sft_path": str(sft_path),
                "source_sft_sha256": sft_sha256,
                "reference_output_path": str(output_path),
                "reference_output_sha256": output_sha256,
                "source_request_id": str(current_state["request_id"]),
                "registry_memory_source_shot_index": registry_memory_source_shot_index,
                "state_seed": encoded_seed,
                "state_seed_sha256": sha256_bytes(encoded_seed.encode("utf-8")),
                "inherit_registry_from_previous_rollout_shot": False,
                "inherit_memory_bank_from_previous_rollout_shot": False,
                "inherit_previous_evidence_from_previous_rollout_shot": False,
                "fresh_conversation": True,
            }
        )
        previous_state = current_state
        previous_state_path = str(state_path)
    return result


def episode_windows(
    shot_specs: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    window_size: int,
    window_stride: int,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    if mode == "single":
        starts = range(len(shot_specs))
        size = 1
    elif mode == "multi":
        if len(shot_specs) < window_size:
            return
        starts = range(0, len(shot_specs) - window_size + 1, window_stride)
        size = window_size
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    for start in starts:
        window: list[dict[str, Any]] = []
        for episode_position, source_spec in enumerate(
            shot_specs[start : start + size]
        ):
            spec = dict(source_spec)
            spec["episode_position"] = episode_position
            inherit = mode == "multi" and episode_position > 0
            spec["inherit_registry_from_previous_rollout_shot"] = inherit
            spec["inherit_memory_bank_from_previous_rollout_shot"] = inherit
            spec["inherit_previous_evidence_from_previous_rollout_shot"] = inherit
            window.append(spec)
        yield start, window


def _episode_id(story_id: str, *, mode: str, start: int, size: int) -> str:
    if mode == "single":
        return f"{story_id}:single:shot:{start:04d}"
    return f"{story_id}:multi:start:{start:04d}:k:{size:02d}"


def build_row(
    *,
    story_id: str,
    split: str,
    index: int,
    mode: str,
    start: int,
    shot_specs: list[dict[str, Any]],
    prompt_contract: Mapping[str, Any],
) -> dict[str, Any]:
    episode_id = _episode_id(
        story_id, mode=mode, start=start, size=len(shot_specs)
    )
    shot_indices = [int(spec["shot_index"]) for spec in shot_specs]
    source_story_dir = str(shot_specs[0]["source_story_dir"])
    ground_truth = {
        "episode_id": episode_id,
        "story_id": story_id,
        "shot_indices": shot_indices,
        "source_story_dir": source_story_dir,
        "source_shot_dirs": [str(spec["source_shot_dir"]) for spec in shot_specs],
        "source_trajectory_paths": [
            str(spec["source_trajectory_path"]) for spec in shot_specs
        ],
        "reference_output_paths": [
            str(spec["reference_output_path"]) for spec in shot_specs
        ],
    }
    return {
        "data_source": DATA_SOURCE_BY_MODE[mode],
        # The custom loop creates one fresh user conversation per shot.  There
        # must never be a synthetic episode-level user message here.
        "prompt": [{"role": "system", "content": TRAINING_SYSTEM_PROMPT}],
        "ability": "reel_joint_prompt_memory" if PROMPT_AUTHORING else "storymem_continuity_evidence_policy",
        "reward_model": {"style": "custom", "ground_truth": ground_truth},
        "extra_info": {
            "schema_version": DATASET_SCHEMA_VERSION,
            "split": split,
            "index": index,
            "mode": mode,
            "episode_id": episode_id,
            "story_id": story_id,
            "window_start": start,
            "window_size": len(shot_specs),
            "fresh_conversation_per_shot": True,
            "system_prompt_version": TRAINING_SYSTEM_PROMPT_VERSION,
            "system_prompt_sha256": training_system_prompt_hash(),
            "tool_protocol_version": TOOL_PROTOCOL_VERSION,
            "tool_schema_sha256": tool_schema_hash(),
            "prompt_validation_schema_version": PROMPT_VALIDATION_SCHEMA_VERSION,
            "max_prompt_length": int(prompt_contract["max_prompt_length"]),
            "initial_prompt_token_counts": [
                int(spec["initial_prompt_token_count"]) for spec in shot_specs
            ],
            "tokenizer_contract_sha256": str(
                prompt_contract["tokenizer_contract_sha256"]
            ),
            "tool_config_sha256": str(prompt_contract["tool_config_sha256"]),
            "state_seed_encoding": STATE_SEED_ENCODING,
            "source_story_dir": source_story_dir,
            "shot_indices": shot_indices,
            "shot_specs": shot_specs,
            "agent_name": AGENT_NAME,
        },
    }


SHOT_SPEC_ARROW_TYPE = pa.struct(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("episode_position", pa.int32(), nullable=False),
        pa.field("shot_index", pa.int32(), nullable=False),
        pa.field("shot_requirement", pa.string(), nullable=False),
        pa.field("shot_requirement_sha256", pa.string(), nullable=False),
        pa.field("initial_prompt_token_count", pa.int32(), nullable=False),
        pa.field("source_story_dir", pa.string(), nullable=False),
        pa.field("source_shot_dir", pa.string(), nullable=False),
        pa.field("source_policy_state_path", pa.string(), nullable=False),
        pa.field("source_previous_policy_state_path", pa.string(), nullable=False),
        pa.field("source_trajectory_path", pa.string(), nullable=False),
        pa.field("source_trajectory_sha256", pa.string(), nullable=False),
        pa.field("source_sft_path", pa.string(), nullable=False),
        pa.field("source_sft_sha256", pa.string(), nullable=False),
        pa.field("reference_output_path", pa.string(), nullable=False),
        pa.field("reference_output_sha256", pa.string(), nullable=False),
        pa.field("source_request_id", pa.string(), nullable=False),
        pa.field("registry_memory_source_shot_index", pa.int32(), nullable=False),
        pa.field("state_seed", pa.string(), nullable=False),
        pa.field("state_seed_sha256", pa.string(), nullable=False),
        pa.field(
            "inherit_registry_from_previous_rollout_shot",
            pa.bool_(),
            nullable=False,
        ),
        pa.field(
            "inherit_memory_bank_from_previous_rollout_shot",
            pa.bool_(),
            nullable=False,
        ),
        pa.field(
            "inherit_previous_evidence_from_previous_rollout_shot",
            pa.bool_(),
            nullable=False,
        ),
        pa.field("fresh_conversation", pa.bool_(), nullable=False),
    ]
)


PARQUET_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string(), nullable=False),
        pa.field(
            "prompt",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string(), nullable=False),
                        pa.field("content", pa.string(), nullable=False),
                    ]
                )
            ),
            nullable=False,
        ),
        pa.field("ability", pa.string(), nullable=False),
        pa.field(
            "reward_model",
            pa.struct(
                [
                    pa.field("style", pa.string(), nullable=False),
                    pa.field(
                        "ground_truth",
                        pa.struct(
                            [
                                pa.field("episode_id", pa.string(), nullable=False),
                                pa.field("story_id", pa.string(), nullable=False),
                                pa.field(
                                    "shot_indices",
                                    pa.list_(pa.int32()),
                                    nullable=False,
                                ),
                                pa.field(
                                    "source_story_dir", pa.string(), nullable=False
                                ),
                                pa.field(
                                    "source_shot_dirs",
                                    pa.list_(pa.string()),
                                    nullable=False,
                                ),
                                pa.field(
                                    "source_trajectory_paths",
                                    pa.list_(pa.string()),
                                    nullable=False,
                                ),
                                pa.field(
                                    "reference_output_paths",
                                    pa.list_(pa.string()),
                                    nullable=False,
                                ),
                            ]
                        ),
                        nullable=False,
                    ),
                ]
            ),
            nullable=False,
        ),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    pa.field("schema_version", pa.string(), nullable=False),
                    pa.field("split", pa.string(), nullable=False),
                    pa.field("index", pa.int64(), nullable=False),
                    pa.field("mode", pa.string(), nullable=False),
                    pa.field("episode_id", pa.string(), nullable=False),
                    pa.field("story_id", pa.string(), nullable=False),
                    pa.field("window_start", pa.int32(), nullable=False),
                    pa.field("window_size", pa.int32(), nullable=False),
                    pa.field(
                        "fresh_conversation_per_shot", pa.bool_(), nullable=False
                    ),
                    pa.field("system_prompt_version", pa.string(), nullable=False),
                    pa.field("system_prompt_sha256", pa.string(), nullable=False),
                    pa.field("tool_protocol_version", pa.string(), nullable=False),
                    pa.field("tool_schema_sha256", pa.string(), nullable=False),
                    pa.field(
                        "prompt_validation_schema_version",
                        pa.string(),
                        nullable=False,
                    ),
                    pa.field("max_prompt_length", pa.int32(), nullable=False),
                    pa.field(
                        "initial_prompt_token_counts",
                        pa.list_(pa.int32()),
                        nullable=False,
                    ),
                    pa.field(
                        "tokenizer_contract_sha256", pa.string(), nullable=False
                    ),
                    pa.field("tool_config_sha256", pa.string(), nullable=False),
                    pa.field("state_seed_encoding", pa.string(), nullable=False),
                    pa.field("source_story_dir", pa.string(), nullable=False),
                    pa.field(
                        "shot_indices", pa.list_(pa.int32()), nullable=False
                    ),
                    pa.field(
                        "shot_specs",
                        pa.list_(SHOT_SPEC_ARROW_TYPE),
                        nullable=False,
                    ),
                    pa.field("agent_name", pa.string(), nullable=False),
                ]
            ),
            nullable=False,
        ),
    ],
    metadata={
        b"schema_version": DATASET_SCHEMA_VERSION.encode("utf-8"),
        b"state_seed_encoding": STATE_SEED_ENCODING.encode("utf-8"),
    },
)


class BufferedParquetWriter:
    def __init__(
        self,
        path: Path,
        *,
        batch_size: int,
        compression: str,
    ) -> None:
        self.path = path
        self.batch_size = batch_size
        self.rows = 0
        self._buffer: list[dict[str, Any]] = []
        self._closed = False
        self._writer = pq.ParquetWriter(
            str(path),
            PARQUET_SCHEMA,
            compression=compression,
            use_dictionary=True,
        )

    def write(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        self.rows += 1
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=PARQUET_SCHEMA)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._writer.close()
        self._closed = True


def _bounded_story_loads(
    story_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    *,
    options: SourceOptions,
    workers: int,
) -> Iterator[tuple[str, list[dict[str, Any]] | BaseException]]:
    if workers == 1:
        for story_id in story_ids:
            try:
                yield story_id, load_story_shot_specs(
                    story_id, records[story_id], options=options
                )
            except BaseException as exc:  # Report with the owning story ID.
                yield story_id, exc
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[tuple[str, Future[list[dict[str, Any]]]]] = deque()
        source = iter(story_ids)

        def submit_one() -> bool:
            try:
                story_id = next(source)
            except StopIteration:
                return False
            pending.append(
                (
                    story_id,
                    executor.submit(
                        load_story_shot_specs,
                        story_id,
                        records[story_id],
                        options=options,
                    ),
                )
            )
            return True

        for _ in range(min(len(story_ids), workers * 2)):
            submit_one()
        while pending:
            story_id, future = pending.popleft()
            try:
                value: list[dict[str, Any]] | BaseException = future.result()
            except BaseException as exc:
                value = exc
            yield story_id, value
            submit_one()


def _file_summary(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise PreprocessError(f"data root does not exist: {data_root}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    prompt_validator = PromptLengthValidator(
        model_path=args.model_path,
        tool_config_path=args.tool_config_path,
        max_prompt_length=args.max_prompt_length,
    )
    prompt_contract = prompt_validator.manifest_record()

    progress_path = data_root / "progress.json"
    progress_value, progress_payload = read_stable_json(
        progress_path, label="source progress"
    )
    progress = require_mapping(progress_value, label="source progress")
    if progress.get("schema_version") != SOURCE_PROGRESS_SCHEMA_VERSION:
        raise PreprocessError(
            f"unsupported progress schema {progress.get('schema_version')!r}"
        )
    stories = require_mapping(progress.get("stories"), label="progress.stories")
    complete: dict[str, Mapping[str, Any]] = {
        str(story_id): require_mapping(record, label=f"story {story_id}")
        for story_id, record in stories.items()
        if isinstance(story_id, str)
        and isinstance(record, Mapping)
        and record.get("status") == "complete"
    }
    selected_ids = ranked_story_ids(
        complete, seed=args.split_seed, purpose="selection"
    )
    if args.max_stories:
        selected_ids = selected_ids[: args.max_stories]
    if len(selected_ids) < 2:
        raise PreprocessError("at least two complete selected stories are required")

    if args.val_stories is None:
        val_count = max(1, round(len(selected_ids) * args.val_fraction))
    else:
        val_count = args.val_stories
    if not 1 <= val_count < len(selected_ids):
        raise PreprocessError(
            "validation story count must be in [1, selected_stories - 1]"
        )
    val_ids = set(
        ranked_story_ids(
            selected_ids, seed=args.split_seed, purpose="validation"
        )[:val_count]
    )
    train_ids = set(selected_ids) - val_ids
    if train_ids & val_ids:
        raise AssertionError("story-level split overlap")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    train_staging = staging / "train.parquet"
    val_staging = staging / "val.parquet"
    train_writer = BufferedParquetWriter(
        train_staging,
        batch_size=args.write_batch_size,
        compression=args.compression,
    )
    val_writer = BufferedParquetWriter(
        val_staging,
        batch_size=args.write_batch_size,
        compression=args.compression,
    )
    writers = {"train": train_writer, "val": val_writer}
    story_episode_counts: Counter[str] = Counter()
    written_story_ids: dict[str, set[str]] = {"train": set(), "val": set()}
    skipped: list[dict[str, str]] = []
    overlong_shots: list[dict[str, Any]] = []
    skipped_overlong_episodes = 0
    checked_prompt_count = 0
    min_prompt_tokens: int | None = None
    max_prompt_tokens = 0
    global_index = 0
    processed_stories = 0
    options = SourceOptions(
        data_root=data_root,
        verify_declared_hashes=args.verify_declared_hashes,
    )

    try:
        for story_id, loaded in _bounded_story_loads(
            selected_ids,
            complete,
            options=options,
            workers=args.workers,
        ):
            if isinstance(loaded, BaseException):
                if not args.skip_invalid_stories:
                    raise PreprocessError(
                        f"failed to build story {story_id}: {loaded}"
                    ) from loaded
                skipped.append(
                    {"story_id": story_id, "error": f"{type(loaded).__name__}: {loaded}"}
                )
                continue
            for spec in loaded:
                token_count = prompt_validator.count(str(spec["shot_requirement"]))
                spec["initial_prompt_token_count"] = token_count
                checked_prompt_count += 1
                min_prompt_tokens = (
                    token_count
                    if min_prompt_tokens is None
                    else min(min_prompt_tokens, token_count)
                )
                max_prompt_tokens = max(max_prompt_tokens, token_count)
                if token_count > args.max_prompt_length:
                    overlong_shots.append(
                        {
                            "story_id": story_id,
                            "shot_index": int(spec["shot_index"]),
                            "prompt_tokens": token_count,
                        }
                    )
            split = "val" if story_id in val_ids else "train"
            episodes_for_story = 0
            for start, window in episode_windows(
                loaded,
                mode=args.mode,
                window_size=args.window_size,
                window_stride=args.window_stride,
            ):
                if any(
                    int(spec["initial_prompt_token_count"])
                    > args.max_prompt_length
                    for spec in window
                ):
                    skipped_overlong_episodes += 1
                    continue
                row = build_row(
                    story_id=story_id,
                    split=split,
                    index=global_index,
                    mode=args.mode,
                    start=start,
                    shot_specs=window,
                    prompt_contract=prompt_contract,
                )
                writers[split].write(row)
                global_index += 1
                episodes_for_story += 1
            story_episode_counts[split] += episodes_for_story
            if episodes_for_story:
                written_story_ids[split].add(story_id)
            processed_stories += 1
            if processed_stories == 1 or processed_stories % 100 == 0:
                print(
                    canonical_json(
                        {
                            "event": "progress",
                            "processed_stories": processed_stories,
                            "selected_stories": len(selected_ids),
                            "episodes": global_index,
                            "story_id": story_id,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )

        train_writer.close()
        val_writer.close()
        if train_writer.rows == 0 or val_writer.rows == 0:
            raise PreprocessError(
                "both train and validation must contain at least one episode; "
                "reduce --window-size or select more stories"
            )

        valid_train_ids = sorted(written_story_ids["train"])
        valid_val_ids = sorted(written_story_ids["val"])
        (staging / "train_story_ids.txt").write_text(
            "\n".join(valid_train_ids) + "\n", encoding="utf-8"
        )
        (staging / "val_story_ids.txt").write_text(
            "\n".join(valid_val_ids) + "\n", encoding="utf-8"
        )

        final_train_path = output_dir / "train.parquet"
        final_val_path = output_dir / "val.parquet"
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "data_root": str(data_root),
                "progress_path": str(progress_path),
                "progress_bytes": len(progress_payload),
                "progress_sha256": sha256_bytes(progress_payload),
                "progress_schema_version": SOURCE_PROGRESS_SCHEMA_VERSION,
                "complete_stories_available": len(complete),
                "declared_artifact_hashes_verified": args.verify_declared_hashes,
            },
            "selection": {
                "split_seed": args.split_seed,
                "algorithm": "sha256(seed + NUL + purpose + NUL + story_id)",
                "max_stories": args.max_stories,
                "selected_stories": len(selected_ids),
                "processed_stories": processed_stories,
                "train_stories": len(valid_train_ids),
                "val_stories": len(valid_val_ids),
                "story_overlap": len(set(valid_train_ids) & set(valid_val_ids)),
                "skipped_invalid_stories": skipped,
            },
            "construction": {
                "mode": args.mode,
                "window_size": 1 if args.mode == "single" else args.window_size,
                "window_stride": 1 if args.mode == "single" else args.window_stride,
                "fresh_conversation_per_shot": True,
                "top_level_prompt": "system_only; loop appends one user request per shot",
                "system_prompt_version": TRAINING_SYSTEM_PROMPT_VERSION,
                "system_prompt_sha256": training_system_prompt_hash(),
                "tool_protocol_version": TOOL_PROTOCOL_VERSION,
                "tool_schema_sha256": tool_schema_hash(),
                "state_seed_schema_version": STATE_SEED_SCHEMA_VERSION,
                "state_seed_encoding": STATE_SEED_ENCODING,
                "prompt_validation": {
                    **prompt_contract,
                    "checked_shots": checked_prompt_count,
                    "minimum_prompt_tokens": min_prompt_tokens,
                    "maximum_prompt_tokens": max_prompt_tokens,
                    "overlong_shots": overlong_shots,
                    "skipped_overlong_episodes": skipped_overlong_episodes,
                },
                "future_leakage_policy": {
                    "shot_0_registry": "prepared_source.registry_items",
                    "shot_0_memory_bank": "empty",
                    "shot_i_registry_and_memory": "shot_i_minus_1.final_policy_state",
                    "shot_i_uploads": "prepared_source.uploads_by_shot",
                    "shot_i_other_local_inputs": "shot_i.policy_state immutable inputs",
                    "current_shot_final_registry_and_memory": "excluded",
                    "multi_later_shots": "must inherit preceding rollout final state",
                },
            },
            "datasets": {
                "train": _file_summary(train_staging, train_writer.rows),
                "val": _file_summary(val_staging, val_writer.rows),
            },
            "episodes": {
                "total": global_index,
                "train": story_episode_counts["train"],
                "val": story_episode_counts["val"],
            },
            "parquet": {
                "compression": args.compression,
                "write_batch_size": args.write_batch_size,
                "schema_version": DATASET_SCHEMA_VERSION,
            },
        }
        manifest["datasets"]["train"]["path"] = str(final_train_path)
        manifest["datasets"]["val"]["path"] = str(final_val_path)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except BaseException:
        # ParquetWriter.close is safe here even if conversion raised midway.
        for writer in (train_writer, val_writer):
            try:
                writer.close()
            except BaseException:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "multi"), required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Qwen3.5 checkpoint whose tokenizer/chat template will be used in training.",
    )
    parser.add_argument(
        "--tool-config-path",
        type=Path,
        required=True,
        help="Canonical nine-tool VERL configuration used by the rollout.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        required=True,
        help="Reject episodes whose real fresh-shot prompt exceeds this token count.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Number of consecutive shots per multi-shot episode.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Distance between consecutive multi-shot window starts.",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=0,
        help="Deterministically select at most this many complete stories; 0 means all.",
    )
    parser.add_argument("--val-stories", type=int)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--split-seed", default="storymem-memory-policy-qwen35-rl-v2"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--write-batch-size", type=int, default=128)
    parser.add_argument(
        "--compression", choices=("zstd", "snappy", "gzip", "none"), default="zstd"
    )
    parser.add_argument(
        "--verify-declared-hashes",
        action="store_true",
        help="Also hash every trajectory, SFT file, and reference video (slow).",
    )
    parser.add_argument(
        "--skip-invalid-stories",
        action="store_true",
        help="Skip a corrupt complete story instead of failing closed.",
    )
    args = parser.parse_args(argv)
    if args.mode == "multi" and args.window_size < 2:
        parser.error("--window-size must be at least 2 for --mode multi")
    if args.window_size < 1:
        parser.error("--window-size must be positive")
    if args.window_stride < 1:
        parser.error("--window-stride must be positive")
    if args.max_prompt_length < 1:
        parser.error("--max-prompt-length must be positive")
    if args.max_stories < 0:
        parser.error("--max-stories must be non-negative")
    if args.val_stories is not None and args.val_stories < 1:
        parser.error("--val-stories must be positive")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be in (0, 1)")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.write_batch_size < 1:
        parser.error("--write-batch-size must be positive")
    if args.compression == "none":
        args.compression = "NONE"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_dataset(args)
    print(
        canonical_json(
            {
                "event": "complete",
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "episodes": manifest["episodes"],
                "selected_stories": manifest["selection"]["selected_stories"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
