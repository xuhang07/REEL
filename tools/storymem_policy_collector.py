#!/usr/bin/env python3
"""Production host for model-independent StoryMem Memory Policy trajectories.

This is intentionally a new collector, not a compatibility wrapper around the
legacy ``storymem_teacher`` state machine.  The only reused code is transport
and deterministic media I/O:

* the policy agent is a closed Gemini endpoint speaking ordinary JSON text;
* :class:`MemoryPolicyRuntime` is the sole owner of the nine policy tools;
* PE is private to one ``Echo15BackendAdapter.compile`` call;
* one durable generation reservation authorizes one real Echo job ID;
* Echo retries replay that same ID and never create a new sampling attempt;
* completed runtime state is sealed directly as a trajectory and native-tool
  SFT sample.

The module is importable for offline tests.  Merely importing it, or invoking
``--help``, never creates a dataset root or calls a provider.
"""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import json
import mimetypes
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urljoin, urlparse

import requests

from storymem_adapter.echo15 import (
    ECHO15_ADAPTER_VERSION,
    ECHO15_COMPILATION_RECEIPT_SCHEMA_VERSION,
    ECHO15_CONDITION_PROJECTION_SCHEMA_VERSION,
    Echo15AdapterError,
    Echo15BackendAdapter,
    Echo15Capabilities,
    Echo15RequestSpec,
    official_v7_profile_manifest,
)
from storymem_adapter.gemini import IsolatedGeminiTextClient
from storymem_policy.contract import (
    TOOL_PROTOCOL_VERSION,
    get_tool_schemas,
    tool_schema_hash,
)
from storymem_policy.ports import (
    VisualEvidenceCreationRequest,
    VisualEvidenceCreationResult,
)
from storymem_policy.prompts import OPEN_MODEL_SYSTEM_PROMPT, system_prompt_for_mode
from storymem_policy.runtime import MemoryPolicyRuntime, PolicyRejected, RUNTIME_VERSION
from storymem_policy.sft_export import (
    SFT_PROJECTION_VERSION,
    SFT_SAMPLE_SCHEMA_VERSION,
    build_sft_sample,
)
from storymem_policy.state import MemoryPolicyState, STATE_SCHEMA_VERSION
from storymem_policy.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    build_teacher_input_messages,
    build_trajectory_shard,
)
from storymem_teacher.media import (
    atomic_write_bytes,
    detect_mime,
    download_url,
    extract_video_frames,
    extract_video_tail_frame,
    require_file,
    selector_catalog,
)
from storymem_teacher.providers import (
    EchoVideoProvider,
    GenericImageProvider,
    ImageGenerationRequest,
    ProviderProtocolError,
    RemoteTeacherClient,
    SharedRequestRateLimiter,
    ToolCall,
)
from storymem_teacher.util import canonical_json, safe_name, sha256_file, sha256_json, sha256_text


COLLECTOR_VERSION = "storymem.memory-policy.production-collector.v6"
RUN_NAMESPACE_VERSION = "storymem.memory-policy.provider-run-namespace.v3"
RUN_INCARNATION_VERSION = "storymem.memory-policy.run-incarnation.v1"
ADAPTER_EPOCH_MARKER_VERSION = "storymem.memory-policy.adapter-epoch.v2"
STORY_CHECKPOINT_VERSION = "storymem.memory-policy.story-checkpoint.v1"
SHOT_HOST_CHECKPOINT_VERSION = "storymem.memory-policy.shot-host-checkpoint.v2"
RUN_MANIFEST_VERSION = "storymem.memory-policy.production-run.v3"
PREFLIGHT_HISTORY_VERSION = "storymem.memory-policy.preflight-history.v1"
RUN_SEMANTIC_HISTORY_VERSION = "storymem.memory-policy.semantic-history.v1"
OPERATIONAL_PATCH_VERSION = "storymem.memory-policy.operational-patch.v1"
ECHO_CAPABILITY_PROFILE_VERSION = "storymem.echo-capability-profile.v1"
ECHO_WIRE_RECEIPT_VERSION = "storymem.echo15-wire-invocation.v1"
ECHO_ENDPOINT_BINDING_VERSION = "storymem.echo15-endpoint-binding.v1"
ECHO_WIRE_ROUTE_VERSION = "storymem.echo15-wire-route.v2"
PE_FAILURE_AUDIT_VERSION = "storymem.adapter.pe-failure-audit.v2"
PROGRESS_SCHEMA_VERSION = "storymem.memory-policy.progress.v1"
DATASET_MANIFEST_SCHEMA_VERSION = "storymem.memory-policy.dataset-manifest.v2"
FAILURE_FUSE_SCHEMA_VERSION = "storymem.memory-policy.failure-fuse.v1"
DEFAULT_FAILURE_FUSE_MIN_OUTCOMES = 16
DEFAULT_FAILURE_FUSE_RATE = 0.5
DEFAULT_FAILURE_FUSE_MAX_FAILURES = 24
TEACHER_REQUEST_TIMEOUT_SECONDS = 120.0
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_CONFIG = REPO_ROOT / "Director_Agent" / ".config.local.json"

# Only validator-owned codes may cross from an exception ledger into durable
# batch progress.  Unknown extension codes are reduced to one generic marker
# so an untrusted provider/custom port cannot smuggle prompt or credential text
# through a syntactically plausible dotted string.
_PE_AUDIT_VIOLATION_CODES = frozenset(
    {
        "validation.rejected",
        "validation.unclassified",
        "output.empty",
        "output.not_text",
        "format.outer_whitespace",
        "format.single_paragraph",
        "format.markdown_wrapper",
        "format.ellipsis",
        "format.forbidden_prefix",
        "format.required_prefix_missing",
        "length.below_minimum",
        "length.above_maximum",
        "length.out_of_range",
        "language.invalid_lock",
        "temporal.preamble_count",
        "temporal.preamble_language",
        "temporal.incomplete_prompt",
        "temporal.unexpected_preamble",
        "shots.no_cut_syntax",
        "shots.prefix_language",
        "shots.labels_sequence",
        "shots.prefix_adjacency",
        "shots.labels_malformed",
        "shots.required_prefix_missing",
        "declarations.mixed_languages",
        "declarations.no_ocr_count",
        "declarations.no_subtitle_count",
        "declarations.closing_adjacency",
        "dialogue.silence_declarations",
        "dialogue.vocal_declaration",
        "dialogue.canonical_quoted_line",
        "dialogue.unexpected",
        "dialogue.requested_speech_missing",
        "dialogue.requested_speech_count",
        "dialogue.exact_lines_mismatch",
        "subject.too_many_ids",
        "subject.identity_anchor_count",
        "subject.clothing_anchor_count",
        "subject.canonical_ids_mismatch",
        "subject.person_count_mismatch",
        "subject.nonhuman_identity",
        "semantic.missing_concepts",
        "semantic.added_concepts",
        "semantic.negated_concepts",
        "state_changes.ordered_transition_missing",
        "privacy.private_material_tag",
        "privacy.numbered_reference_image",
        "privacy.path_url_or_evidence_handle",
    }
)

# Cleartext diagnostic concepts are restricted to the validator's public,
# finite taxonomy.  Open-vocabulary subject/object attributes are represented
# only by bound hashes below; this keeps progress useful without persisting a
# private raw prompt token verbatim.
_PE_AUDIT_CLEARTEXT_CONCEPTS = frozenset(
    """
    alien animal bear bird cat creature dog dragon horse monster robot soldier
    vampire wolf wizard appearance build face
    arrive attack break burn carry catch close cross destroy discover drink drop
    eat enter exit explode fight follow give hand hold hug inspect kill kidnap
    leave lift look meet move murder open pick place point present read repair
    rescue return run shift shoot sit speak stand steal take touch turn walk
    watch write
    alley archive attic beach bridge cafe castle city corridor courtyard desert
    dock forest garden greenhouse harbor hospital house kitchen laboratory market
    mountain observatory ocean office palace pier platform restaurant river room
    rooftop shop shore station street studio subway temple train village warehouse
    workshop
    anime cartoon claymation cyberpunk documentary dreamlike dreamy fantasy film
    historical ink wash monochrome naturalism noir painterly photorealistic realism
    realistic retro storybook surreal watercolor
    """.split()
)

_SUCCESS_STATES = frozenset({"completed", "done", "success", "succeeded"})
_FAILURE_STATES = frozenset({"cancelled", "canceled", "error", "failed"})
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUN_NAMESPACE_RE = re.compile(r"^storymem_run_[0-9a-f]{24}$")
_RUN_INCARNATION_RE = re.compile(r"^storymem_incarnation_[0-9a-f]{32}$")


class CollectorError(RuntimeError):
    """A fail-closed collector contract violation."""


class EchoTerminalFailure(CollectorError):
    """The one authorized Echo job reached a terminal failure state."""


class EchoInvocationPort(Protocol):
    def invoke(
        self,
        request: Mapping[str, Any],
        *,
        output_dir: str | Path,
        resume: bool,
        reserved_canonical_conditions: Sequence[Mapping[str, Any]] | None = None,
        endpoint_binding: Mapping[str, Any] | None = None,
        persist_endpoint_binding: Callable[[Mapping[str, Any]], None] | None = None,
        legacy_route_fallback: bool = False,
    ) -> "EchoInvocationResult": ...


class FinalEvidencePort(Protocol):
    def derive(
        self,
        *,
        video_path: str | Path,
        output_dir: str | Path,
        generation_id: str,
        subject_ids: Sequence[str],
        semantic_roles: Sequence[str],
    ) -> list[dict[str, Any]]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def new_run_incarnation_id() -> str:
    """Return a high-entropy identity that is persisted before generation."""

    return "storymem_incarnation_" + secrets.token_hex(16)


def validate_run_incarnation_id(value: Any) -> str:
    incarnation = str(value or "")
    if not _RUN_INCARNATION_RE.fullmatch(incarnation):
        raise CollectorError("run incarnation ID is malformed")
    return incarnation


def run_namespace_for_semantic_epoch(
    output_root: str | Path,
    *,
    semantic_epoch: int,
    semantic_lock_sha256: str,
    run_incarnation_id: str,
) -> str:
    """Return the provider namespace for one sealed semantic epoch.

    Provider idempotency journals outlive local dataset checkpoints.  The
    persisted incarnation prevents a deleted/recreated root at the same path
    from reusing provider job IDs, while the epoch and semantic lock isolate
    migrations.  All inputs stay stable for ordinary crash recovery.
    """

    if type(semantic_epoch) is not int or semantic_epoch < 1:
        raise ValueError("semantic_epoch must be a positive integer")
    if not _SHA256_RE.fullmatch(str(semantic_lock_sha256)):
        raise ValueError("semantic_lock_sha256 must be a canonical SHA-256")
    incarnation = validate_run_incarnation_id(run_incarnation_id)
    identity = canonical_json(
        {
            "schema_version": RUN_NAMESPACE_VERSION,
            "output_root": str(Path(output_root).expanduser().resolve()),
            "collector_version": COLLECTOR_VERSION,
            "semantic_epoch": semantic_epoch,
            "semantic_lock_sha256": semantic_lock_sha256,
            "run_incarnation_id": incarnation,
        }
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"storymem_run_{digest}"


def semantic_epoch_for_run_manifest(run: Mapping[str, Any]) -> int:
    """Return the active epoch while rejecting ambiguous history."""

    history = run.get("semantic_history")
    if history is None:
        return 1
    if not isinstance(history, list) or not history:
        raise CollectorError("run semantic history is malformed")
    active = history[-1]
    epoch = active.get("epoch") if isinstance(active, Mapping) else None
    if type(epoch) is not int or epoch < 1 or epoch != len(history):
        raise CollectorError("run semantic history has no valid active epoch")
    return epoch


def adapter_artifact_root(output_root: str | Path, run_namespace: str) -> Path:
    """Return the adapter root isolated to a single provider namespace."""

    if not _RUN_NAMESPACE_RE.fullmatch(str(run_namespace)):
        raise CollectorError("run namespace is malformed")
    return (
        Path(output_root).expanduser().resolve()
        / "adapter_epochs"
        / str(run_namespace)
    )


def build_production_semantic_lock() -> dict[str, Any]:
    """Seal every code/prompt contract whose change can alter training data."""

    source_paths = [
        Path(__file__).resolve(),
        *sorted((REPO_ROOT / "tools" / "storymem_policy").glob("*.py")),
        *sorted((REPO_ROOT / "tools" / "storymem_adapter").glob("*.py")),
        REPO_ROOT / "tools" / "storymem_teacher" / "media.py",
        REPO_ROOT / "tools" / "storymem_teacher" / "providers.py",
        REPO_ROOT / "tools" / "storymem_teacher" / "util.py",
        REPO_ROOT / "tools" / "storymem_shot_assets.py",
        REPO_ROOT / "tools" / "storymem_provider_pool.py",
        REPO_ROOT / "tools" / "storymem_data_pipeline.py",
        REPO_ROOT / "ltx-core" / "src" / "ltx_core" / "model" / "transformer" / "attention.py",
    ]
    unique_paths = sorted(
        {path.resolve() for path in source_paths if path.is_file()},
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )
    source_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in unique_paths
    }
    body = {
        "schema_version": "storymem.memory-policy.semantic-lock.v1",
        "collector_version": COLLECTOR_VERSION,
        "run_manifest_version": RUN_MANIFEST_VERSION,
        "run_namespace_version": RUN_NAMESPACE_VERSION,
        "run_incarnation_version": RUN_INCARNATION_VERSION,
        "adapter_epoch_marker_version": ADAPTER_EPOCH_MARKER_VERSION,
        "story_checkpoint_version": STORY_CHECKPOINT_VERSION,
        "shot_host_checkpoint_version": SHOT_HOST_CHECKPOINT_VERSION,
        "echo_endpoint_binding_version": ECHO_ENDPOINT_BINDING_VERSION,
        "echo_wire_route_version": ECHO_WIRE_ROUTE_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "tool_protocol_version": TOOL_PROTOCOL_VERSION,
        "tool_schema_sha256": tool_schema_hash(),
        "open_model_system_prompt_sha256": sha256_text(OPEN_MODEL_SYSTEM_PROMPT),
        "prompt_authoring_tool_schema_sha256": tool_schema_hash(prompt_authoring=True),
        "prompt_authoring_system_prompt_sha256": sha256_text(system_prompt_for_mode(True)),
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "sft_sample_schema_version": SFT_SAMPLE_SCHEMA_VERSION,
        "sft_projection_version": SFT_PROJECTION_VERSION,
        "echo15_adapter_version": ECHO15_ADAPTER_VERSION,
        "echo15_condition_projection_schema_version": (
            ECHO15_CONDITION_PROJECTION_SCHEMA_VERSION
        ),
        "echo15_compilation_receipt_schema_version": (
            ECHO15_COMPILATION_RECEIPT_SCHEMA_VERSION
        ),
        "official_v7_pe": official_v7_profile_manifest(REPO_ROOT),
        "collector_source_files_sha256": source_hashes,
        "collector_source_bundle_sha256": sha256_json(source_hashes),
    }
    body["semantic_lock_sha256"] = sha256_json(body)
    return body


def validate_compatible_run_manifest(
    actual: Mapping[str, Any], expected_body: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate either an original run manifest or an audited semantic migration.

    ``expected_body`` is the current canonical run body without ``run_sha256``.
    Operational migrations may add history, but may not change source, policy
    configuration, provider identity, or any other canonical field.
    """

    value = _json_copy(dict(actual))
    run_sha256 = value.pop("run_sha256", None)
    if not isinstance(run_sha256, str) or run_sha256 != sha256_json(value):
        raise CollectorError("existing run manifest seal is invalid")

    semantic_history = value.pop("semantic_history", None)
    patch_history = value.pop("operational_patch_history", None)
    if value != _json_copy(dict(expected_body)):
        raise CollectorError("output root belongs to a different run")
    expected_incarnation = validate_run_incarnation_id(
        expected_body.get("run_incarnation_id")
    )
    if semantic_history is None and patch_history is None:
        return _json_copy(dict(actual))
    if not isinstance(semantic_history, list) or len(semantic_history) < 2:
        raise CollectorError("migrated run has no valid semantic history")
    if not isinstance(patch_history, list) or not patch_history:
        raise CollectorError("migrated run has no operational patch history")
    if len(semantic_history) != len(patch_history) + 1:
        raise CollectorError("semantic and operational histories have different lengths")

    for index, epoch in enumerate(semantic_history, start=1):
        if (
            not isinstance(epoch, Mapping)
            or epoch.get("schema_version") != RUN_SEMANTIC_HISTORY_VERSION
            or epoch.get("epoch") != index
            or not _SHA256_RE.fullmatch(str(epoch.get("semantic_lock_sha256") or ""))
            or not isinstance(epoch.get("activated_at"), str)
            or not epoch["activated_at"]
        ):
            raise CollectorError("migrated run semantic history is malformed")
        lock = epoch.get("collector_semantic_lock")
        if (
            not isinstance(lock, Mapping)
            or lock.get("semantic_lock_sha256") != epoch["semantic_lock_sha256"]
        ):
            raise CollectorError("migrated run semantic epoch lock is malformed")
        if index < len(semantic_history):
            if (
                not isinstance(epoch.get("retired_at"), str)
                or not epoch["retired_at"]
                or not _SHA256_RE.fullmatch(str(epoch.get("run_sha256") or ""))
            ):
                raise CollectorError("retired semantic epoch is missing its audit boundary")
        elif "retired_at" in epoch or "run_sha256" in epoch:
            raise CollectorError("active semantic epoch is marked retired")
        if index == len(semantic_history):
            if (
                epoch.get("run_namespace") != expected_body.get("run_namespace")
                or epoch.get("run_incarnation_id") != expected_incarnation
            ):
                raise CollectorError(
                    "active semantic history identity differs from the run"
                )

    active_lock = expected_body.get("collector_semantic_lock")
    if (
        not isinstance(active_lock, Mapping)
        or semantic_history[-1].get("collector_semantic_lock") != active_lock
    ):
        raise CollectorError("migrated run active semantic epoch is not current")
    for index, patch in enumerate(patch_history, start=1):
        if (
            not isinstance(patch, Mapping)
            or patch.get("schema_version") != OPERATIONAL_PATCH_VERSION
            or patch.get("sequence") != index
            or not _SHA256_RE.fullmatch(str(patch.get("migration_id") or ""))
            or not _SHA256_RE.fullmatch(str(patch.get("previous_run_sha256") or ""))
            or not _SHA256_RE.fullmatch(
                str(patch.get("previous_semantic_lock_sha256") or "")
            )
            or not _SHA256_RE.fullmatch(
                str(patch.get("active_semantic_lock_sha256") or "")
            )
            or type(patch.get("executor_workers")) is not int
            or patch["executor_workers"] < 1
            or not isinstance(patch.get("applied_at"), str)
            or not patch["applied_at"]
            or not isinstance(patch.get("changes"), list)
            or not patch["changes"]
        ):
            raise CollectorError("migrated run operational patch history is malformed")
        if patch["previous_semantic_lock_sha256"] != semantic_history[index - 1][
            "semantic_lock_sha256"
        ] or patch["active_semantic_lock_sha256"] != semantic_history[index][
            "semantic_lock_sha256"
        ]:
            raise CollectorError("operational patch does not join adjacent semantic epochs")
        if patch["previous_run_sha256"] != semantic_history[index - 1]["run_sha256"]:
            raise CollectorError("operational patch previous run boundary is inconsistent")
        if index == len(patch_history):
            previous_namespace = patch.get("previous_run_namespace")
            active_namespace = patch.get("active_run_namespace")
            previous_incarnation = patch.get("previous_run_incarnation_id")
            active_incarnation = patch.get("active_run_incarnation_id")
            if (
                not isinstance(previous_namespace, str)
                or not previous_namespace
                or active_namespace != expected_body.get("run_namespace")
                or previous_namespace == active_namespace
                or "previous_run_incarnation_id" not in patch
                or (
                    previous_incarnation is not None
                    and validate_run_incarnation_id(previous_incarnation)
                    == expected_incarnation
                )
                or active_incarnation != expected_incarnation
            ):
                raise CollectorError(
                    "operational patch run identity transition is malformed"
                )
    if patch_history[-1]["active_semantic_lock_sha256"] != active_lock.get(
        "semantic_lock_sha256"
    ):
        raise CollectorError("operational patch history does not end at the current lock")
    return _json_copy(dict(actual))


def _output_lock_path(output_root: str | Path) -> Path:
    destination = Path(output_root).expanduser().resolve()
    digest = hashlib.sha256(str(destination).encode("utf-8")).hexdigest()[:20]
    return destination.parent / f".{_safe_id(destination.name, 'storymem')}.{digest}.lock"


@contextmanager
def _exclusive_output_root_lock(output_root: str | Path):
    """Claim a dataset root before any adapter artifact is created."""

    destination = Path(output_root).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _output_lock_path(destination)
    handle = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CollectorError(
                f"another collector already owns output root {destination}"
            ) from exc
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _atomic_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    # Do not resolve the leaf: resolving it would follow an attacker- or
    # race-created symlink and replace the symlink target.  The parent paths in
    # production are already rooted under the canonical CollectorConfig root.
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise CollectorError(f"refusing to replace symlink: {destination}")
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            # The directory entry must be durable before any provider POST can
            # rely on the checkpoint or endpoint binding it contains.
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _build_echo_endpoint_binding(
    *,
    generation_id: str,
    request_payload_sha256: str,
    routing: str,
    endpoint_sha256: str,
    endpoint_count_at_assignment: int,
    per_endpoint_concurrency_at_assignment: int,
    assigned_at: str | None = None,
) -> dict[str, Any]:
    body = {
        "schema_version": ECHO_ENDPOINT_BINDING_VERSION,
        "generation_id": generation_id,
        "request_payload_sha256": request_payload_sha256,
        "routing": routing,
        "endpoint_sha256": endpoint_sha256,
        "endpoint_count_at_assignment": endpoint_count_at_assignment,
        "per_endpoint_concurrency_at_assignment": (
            per_endpoint_concurrency_at_assignment
        ),
        "assigned_at": assigned_at or _utc_now(),
    }
    return {**body, "binding_sha256": sha256_json(body)}


def _validated_echo_endpoint_binding(
    value: Mapping[str, Any],
    *,
    generation_id: str,
    request_payload_sha256: str,
) -> dict[str, Any]:
    binding = _json_copy(dict(value))
    expected_keys = {
        "schema_version",
        "generation_id",
        "request_payload_sha256",
        "routing",
        "endpoint_sha256",
        "endpoint_count_at_assignment",
        "per_endpoint_concurrency_at_assignment",
        "assigned_at",
        "binding_sha256",
    }
    body = {key: item for key, item in binding.items() if key != "binding_sha256"}
    if (
        set(binding) != expected_keys
        or binding.get("schema_version") != ECHO_ENDPOINT_BINDING_VERSION
        or binding.get("generation_id") != generation_id
        or binding.get("request_payload_sha256") != request_payload_sha256
        or binding.get("routing")
        not in {"idle_round_robin_v1", "legacy_generation_hash_recovery_v1"}
        or not _SHA256_RE.fullmatch(str(binding.get("endpoint_sha256") or ""))
        or type(binding.get("endpoint_count_at_assignment")) is not int
        or binding["endpoint_count_at_assignment"] < 1
        or type(binding.get("per_endpoint_concurrency_at_assignment")) is not int
        or binding["per_endpoint_concurrency_at_assignment"] < 1
        or not isinstance(binding.get("assigned_at"), str)
        or not binding["assigned_at"]
        or binding.get("binding_sha256") != sha256_json(body)
    ):
        raise CollectorError("durable Echo endpoint binding is malformed or unsealed")
    return binding


def _read_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectorError(f"{label} is unreadable: {source}") from exc
    if not isinstance(value, dict):
        raise CollectorError(f"{label} must be a JSON object: {source}")
    return value


def _prevalidate_expected_run_incarnation(
    output_root: str | Path,
    expected_run_incarnation_id: str | None,
) -> str | None:
    """Reject a bad launch identity before constructing or calling providers."""

    if expected_run_incarnation_id is None:
        return None
    expected = validate_run_incarnation_id(expected_run_incarnation_id)
    run_path = Path(output_root).expanduser().resolve() / "run.json"
    if not (run_path.exists() or run_path.is_symlink()):
        return expected
    if not run_path.is_file() or run_path.is_symlink():
        raise CollectorError("existing run manifest path is unsafe")
    run = _read_object(run_path, label="run manifest")
    run_body = {key: item for key, item in run.items() if key != "run_sha256"}
    if run.get("run_sha256") != sha256_json(run_body):
        raise CollectorError("existing run manifest seal is invalid")
    if validate_run_incarnation_id(run.get("run_incarnation_id")) != expected:
        raise CollectorError(
            "existing run incarnation differs from the expected launch identity"
        )
    return expected


def _validated_run_namespace(
    run: Mapping[str, Any],
    *,
    output_root: str | Path,
    active_semantic_lock: Mapping[str, Any],
) -> str:
    """Validate the persisted namespace against the active sealed epoch."""

    active_lock_sha256 = str(active_semantic_lock.get("semantic_lock_sha256") or "")
    if not _SHA256_RE.fullmatch(active_lock_sha256):
        raise CollectorError("active semantic lock has no valid SHA-256")
    if run.get("collector_semantic_lock") != dict(active_semantic_lock):
        raise CollectorError("run semantic lock does not match the active code")
    collector_version = str(run.get("collector_version") or "")
    if (
        not collector_version
        or collector_version != active_semantic_lock.get("collector_version")
        or collector_version != COLLECTOR_VERSION
    ):
        raise CollectorError("run collector version does not match the active code")
    namespace = str(run.get("run_namespace") or "")
    incarnation = validate_run_incarnation_id(run.get("run_incarnation_id"))
    expected = run_namespace_for_semantic_epoch(
        output_root,
        semantic_epoch=semantic_epoch_for_run_manifest(run),
        semantic_lock_sha256=active_lock_sha256,
        run_incarnation_id=incarnation,
    )
    if namespace != expected:
        raise CollectorError("run namespace does not match its semantic epoch")
    return namespace


def prepare_adapter_artifact_root(
    output_root: str | Path,
    run: Mapping[str, Any],
    *,
    active_semantic_lock: Mapping[str, Any] | None = None,
) -> Path:
    """Create or validate the adapter directory owned by ``run``.

    The sealed marker makes a directory resumable while preventing a foreign
    or pre-epoch cache from being accepted merely because its path matches.
    """

    root = Path(output_root).expanduser().resolve()
    lock = (
        build_production_semantic_lock()
        if active_semantic_lock is None
        else _json_copy(dict(active_semantic_lock))
    )
    namespace = _validated_run_namespace(
        run, output_root=root, active_semantic_lock=lock
    )
    run_body = {key: item for key, item in run.items() if key != "run_sha256"}
    if run.get("run_sha256") != sha256_json(run_body):
        raise CollectorError("run manifest seal is invalid before adapter setup")

    parent = root / "adapter_epochs"
    if parent.exists() or parent.is_symlink():
        if not parent.is_dir() or parent.is_symlink():
            raise CollectorError("adapter epoch parent is unsafe")
    else:
        parent.mkdir(mode=0o750)
    artifact_root = adapter_artifact_root(root, namespace)
    if artifact_root.exists() or artifact_root.is_symlink():
        if not artifact_root.is_dir() or artifact_root.is_symlink():
            raise CollectorError("adapter epoch root is unsafe")
    else:
        artifact_root.mkdir(mode=0o750)

    marker_body = {
        "schema_version": ADAPTER_EPOCH_MARKER_VERSION,
        "output_root": str(root),
        "collector_version": str(run["collector_version"]),
        "run_namespace_version": RUN_NAMESPACE_VERSION,
        "run_namespace": namespace,
        "run_incarnation_id": validate_run_incarnation_id(
            run.get("run_incarnation_id")
        ),
        "semantic_epoch": semantic_epoch_for_run_manifest(run),
        "semantic_lock_sha256": lock["semantic_lock_sha256"],
        "run_sha256": run["run_sha256"],
    }
    expected_marker = {
        **marker_body,
        "marker_sha256": sha256_json(marker_body),
    }
    marker_path = artifact_root / "adapter_epoch.json"
    if marker_path.exists() or marker_path.is_symlink():
        if not marker_path.is_file() or marker_path.is_symlink():
            raise CollectorError("adapter epoch marker is unsafe")
        if _read_object(marker_path, label="adapter epoch marker") != expected_marker:
            raise CollectorError("adapter epoch marker conflicts with this run")
    else:
        if any(artifact_root.iterdir()):
            raise CollectorError("unmarked adapter epoch root is not empty")
        _atomic_json(marker_path, expected_marker)
    return artifact_root


def _safe_id(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "")).strip("_.:-")
    return text[:120] or fallback


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _digest_or_none(value: Any) -> str | None:
    text = str(value or "")
    return text if _SHA256_RE.fullmatch(text) is not None else None


def build_echo_capability_profile(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project live Echo receipts to a stable, run-sealed capability identity.

    Worker counts, response bytes, retry counts, timestamps, PIDs, and readiness
    snapshots deliberately stay in ``preflight_history.json``.  Only endpoint
    identity and the schema that proved the conditioning contract enter the run
    identity.  The current Echo health contract does not independently expose a
    model-weight or server-build identifier, so this limitation is recorded
    explicitly rather than inferred from a URL or a response hash.
    """

    if not receipts:
        raise CollectorError("Echo capability profile requires at least one receipt")
    endpoints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(receipts):
        if not isinstance(raw, Mapping):
            raise CollectorError(f"Echo receipt {index} must be an object")
        endpoint_sha256 = _digest_or_none(raw.get("endpoint_sha256"))
        capability_sha256 = _digest_or_none(raw.get("capability_schema_sha256"))
        capability_source = str(raw.get("capability_source") or "")
        if endpoint_sha256 is None or endpoint_sha256 in seen:
            raise CollectorError("Echo capability receipts have invalid/duplicate endpoints")
        if capability_sha256 is None or capability_source not in {
            "health.r2v_contract",
            "openapi.r2v_memory_slot_audio_url",
        }:
            raise CollectorError("Echo capability receipt lacks a stable schema proof")
        if raw.get("audio_memory") is not True:
            raise CollectorError("Echo capability receipt does not prove audio memory")
        max_slots = raw.get("max_memory_slots")
        if max_slots is not None and (
            type(max_slots) is not int or max_slots < 7
        ):
            raise CollectorError(
                "Echo capability receipt advertises fewer than seven memory slots"
            )
        seen.add(endpoint_sha256)
        endpoints.append(
            {
                "endpoint_sha256": endpoint_sha256,
                "capability_source": capability_source,
                "capability_schema_sha256": capability_sha256,
                "audio_memory": True,
                "max_memory_slots": max_slots,
            }
        )
    body: dict[str, Any] = {
        "schema_version": ECHO_CAPABILITY_PROFILE_VERSION,
        "required_audio_memory": True,
        "required_memory_slots": 7,
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "all_capacity_limits_proven": all(
            type(row["max_memory_slots"]) is int for row in endpoints
        ),
        "backend_build_identity": "not_independently_advertised_or_verified",
        "backend_model_identity": "not_independently_advertised_or_verified",
    }
    body["profile_sha256"] = sha256_json(body)
    return body


def bind_echo_capability_profile(
    provider_profile: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a provider profile bound to the stable Echo capability proof."""

    profile = _json_copy(dict(provider_profile))
    echo = profile.get("echo")
    if not isinstance(echo, dict):
        raise CollectorError("provider profile has no Echo object")
    capability_profile = build_echo_capability_profile(receipts)
    if echo.get("endpoint_count") != capability_profile["endpoint_count"]:
        raise CollectorError("Echo provider/receipt endpoint counts differ")
    advertised = echo.get("endpoint_sha256")
    receipt_endpoints = [
        row["endpoint_sha256"] for row in capability_profile["endpoints"]
    ]
    if advertised != receipt_endpoints:
        raise CollectorError("Echo provider/receipt endpoint identities differ")
    echo["capability_profile"] = capability_profile
    return profile


def _project_pe_request_identity(value: Any) -> dict[str, Any]:
    """Project a PE request to hashes/counts; never retain prompt text or headers."""

    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("request_payload_sha256", "system_instruction_sha256"):
        digest = _digest_or_none(value.get(key))
        if digest is not None:
            result[key] = digest
    for key in ("user_turn_count",):
        child = value.get(key)
        if type(child) is int and child >= 0:
            result[key] = child
    for key in ("api_style", "transport", "response_mime_type"):
        child = value.get(key)
        if isinstance(child, str):
            result[key] = child
    parts = []
    for raw in value.get("parts") or []:
        if not isinstance(raw, Mapping):
            continue
        part: dict[str, Any] = {}
        for key in ("kind", "mime_type"):
            child = raw.get(key)
            if isinstance(child, str):
                part[key] = child
        digest = _digest_or_none(raw.get("sha256"))
        if digest is not None:
            part["sha256"] = digest
        size = raw.get("size_bytes")
        if type(size) is int and size >= 0:
            part["size_bytes"] = size
        if part:
            parts.append(part)
    if parts:
        result["parts"] = parts
    return result


def _project_pe_http_attempts(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return result
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        row: dict[str, Any] = {}
        for key in (
            "attempt",
            "http_status",
            "latency_ms",
            "response_size_bytes",
            "redirect_count",
            "rate_limit_wait_seconds",
            "retry_after_seconds",
            "retry_delay_seconds",
            "shared_cooldown_seconds",
        ):
            child = raw.get(key)
            if isinstance(child, (int, float)) and not isinstance(child, bool):
                row[key] = child
        for key in ("trace_id", "response_request_id", "error_type"):
            child = raw.get(key)
            if isinstance(child, str):
                row[key] = child
        for key in ("response_sha256", "final_url_sha256"):
            digest = _digest_or_none(raw.get(key))
            if digest is not None:
                row[key] = digest
        if type(raw.get("retryable")) is bool:
            row["retryable"] = raw["retryable"]
        if row:
            result.append(row)
    return result


def _project_pe_response_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "finish_reason",
        "candidate_count",
        "part_count",
        "final_text_part_count",
        "thought_part_count",
        "non_text_part_count",
        "discarded_final_text_character_count",
    ):
        child = value.get(key)
        if isinstance(child, (str, int)) and not isinstance(child, bool):
            result[key] = child
    for key in (
        "raw_final_text_sha256",
        "discarded_final_text_sha256",
    ):
        digest = _digest_or_none(value.get(key))
        if digest is not None:
            result[key] = digest
    for key in ("final_text_part_sha256", "thought_part_sha256"):
        digests = [
            digest
            for child in (value.get(key) or [])
            if (digest := _digest_or_none(child)) is not None
        ]
        if digests:
            result[key] = digests
    return result


def _project_pe_provider_provenance(value: Any) -> dict[str, Any]:
    """Allowlist provider metadata; omit text, payloads, URLs, and credentials."""

    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "assistant_source",
        "api_style",
        "model",
        "response_id",
        "response_request_id",
        "returned_model",
        "transport",
        "response_mime_type",
        "failure_stage",
        "failure_kind",
    ):
        child = value.get(key)
        if isinstance(child, str):
            result[key] = child
    for key in (
        "endpoint_sha256",
        "request_payload_sha256",
        "request_sha256",
        "request_manifest_sha256",
        "response_sha256",
        "assistant_text_sha256",
    ):
        digest = _digest_or_none(value.get(key))
        if digest is not None:
            result[key] = digest
    for key in ("user_turn_count",):
        child = value.get(key)
        if type(child) is int and child >= 0:
            result[key] = child
    for key in ("conversation_reused", "retryable_for_fresh_pe_attempt"):
        child = value.get(key)
        if type(child) is bool:
            result[key] = child
    response_manifest = _project_pe_response_manifest(value.get("response_manifest"))
    if response_manifest:
        result["response_manifest"] = response_manifest
    attempts = _project_pe_http_attempts(value.get("attempts"))
    if attempts:
        result["transport_attempts"] = attempts
    return result


def _project_pe_generation_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "strategy",
        "retryable_provider",
        "provider_error_fallback",
    ):
        child = value.get(key)
        if isinstance(child, str):
            result[key] = child
    attempts = value.get("max_generation_attempts")
    if type(attempts) is int and attempts >= 1:
        result["max_generation_attempts"] = attempts
    for key in ("retryable_finish_reasons", "retryable_failure_kinds"):
        reasons = value.get(key)
        if isinstance(reasons, list) and all(
            isinstance(item, str) for item in reasons
        ):
            result[key] = list(reasons)
    for key in (
        "semantic_validation_uses_fresh_turn",
        "passthrough_fallback",
    ):
        child = value.get(key)
        if type(child) is bool:
            result[key] = child
    return result


def _project_pe_validation(value: Any) -> dict[str, Any]:
    """Project deterministic rejection reasons without provider or prompt text."""

    if not isinstance(value, Mapping) or type(value.get("accepted")) is not bool:
        return {}
    result: dict[str, Any] = {"accepted": value["accepted"]}
    error = value.get("error")
    if isinstance(error, str):
        result["error_sha256"] = sha256_text(error)
    schema_version = value.get("violation_schema_version")
    if schema_version == "storymem.pe-validation-violations.v1":
        result["violation_schema_version"] = schema_version

    codes: list[str] = []

    def add_code(raw: Any) -> None:
        if not isinstance(raw, str) or not raw:
            return
        codes.append(
            raw
            if raw in _PE_AUDIT_VIOLATION_CODES
            else "validation.unclassified"
        )

    raw_codes = value.get("violation_codes")
    if isinstance(raw_codes, list):
        for code in raw_codes[:128]:
            add_code(code)

    concept_failures: dict[str, dict[str, list[str]]] = {}
    concept_failure_hashes: dict[str, dict[str, list[str]]] = {}
    concept_failure_counts: dict[str, dict[str, int]] = {}
    violations = value.get("violations")
    if isinstance(violations, list):
        for violation in violations[:128]:
            if not isinstance(violation, Mapping):
                continue
            add_code(violation.get("code"))
            category = violation.get("category")
            if not isinstance(category, str) or category not in {
                "subject",
                "event",
                "scene",
                "style",
                "state_changes",
            }:
                continue
            for source_key, result_key in (
                ("missing_concepts", "missing"),
                ("added_concepts", "added"),
                ("negated_concepts", "negated"),
            ):
                raw_concepts = violation.get(source_key)
                if not isinstance(raw_concepts, list):
                    continue
                concepts = sorted(
                    {
                        concept
                        for concept in raw_concepts[:128]
                        if isinstance(concept, str)
                        and 1 <= len(concept) <= 512
                    }
                )
                if not concepts:
                    continue
                cleartext = [
                    concept
                    for concept in concepts
                    if concept in _PE_AUDIT_CLEARTEXT_CONCEPTS
                ]
                if cleartext:
                    concept_failures.setdefault(result_key, {})[
                        category
                    ] = cleartext
                concept_failure_hashes.setdefault(result_key, {})[category] = [
                    sha256_text(f"{result_key}:{category}:{concept}")
                    for concept in concepts
                ]
                concept_failure_counts.setdefault(result_key, {})[
                    category
                ] = len(concepts)
    if codes:
        result["violation_codes"] = list(dict.fromkeys(codes))
    if concept_failures:
        result["concept_failures"] = concept_failures
    if concept_failure_hashes:
        result["concept_failure_hashes"] = concept_failure_hashes
        result["concept_failure_counts"] = concept_failure_counts
    return result


def _project_pe_attempt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    attempt = value.get("attempt")
    if type(attempt) is int and attempt >= 1:
        result["attempt"] = attempt
    outcome = value.get("outcome")
    if isinstance(outcome, str):
        result["outcome"] = outcome
    request = _project_pe_request_identity(value.get("request_manifest"))
    if request:
        result["request_identity"] = request
    provenance = _project_pe_provider_provenance(value.get("provider_provenance"))
    if provenance:
        result["provider_provenance"] = provenance
    failure = value.get("failure")
    if isinstance(failure, Mapping):
        projected_failure: dict[str, Any] = {}
        for key in ("error_type", "finish_reason"):
            child = failure.get(key)
            if isinstance(child, str):
                projected_failure[key] = child
        retryable = failure.get("retryable_as_fresh_pe_attempt")
        if type(retryable) is bool:
            projected_failure["retryable_as_fresh_pe_attempt"] = retryable
        if projected_failure:
            result["failure"] = projected_failure
    normalization = value.get("output_normalization")
    if isinstance(normalization, Mapping):
        projected_normalization: dict[str, Any] = {}
        method = normalization.get("method")
        if isinstance(method, str):
            projected_normalization["method"] = method
        if type(normalization.get("changed")) is bool:
            projected_normalization["changed"] = normalization["changed"]
        for key in ("raw_text_sha256", "normalized_text_sha256"):
            digest = _digest_or_none(normalization.get(key))
            if digest is not None:
                projected_normalization[key] = digest
        if projected_normalization:
            result["output_normalization"] = projected_normalization
    output_character_count = value.get("output_character_count")
    if type(output_character_count) is int and output_character_count >= 0:
        result["output_character_count"] = output_character_count
    validation = value.get("validation")
    projected_validation = _project_pe_validation(validation)
    if projected_validation:
        result["validation"] = projected_validation
    return result


def _pe_failure_audit(exc: BaseException) -> dict[str, Any] | None:
    """Build a durable, JSON-safe, secret-free audit of terminal PE attempts."""

    attempts = getattr(exc, "attempts", None)
    policy = getattr(exc, "pe_generation_policy", None)
    if not isinstance(attempts, list) or not attempts or not isinstance(policy, Mapping):
        return None
    try:
        source_attempts = _json_copy(attempts)
        projected_attempts = [
            projected
            for item in source_attempts
            if (projected := _project_pe_attempt(item))
        ]
        projected_policy = _project_pe_generation_policy(policy)
        if not projected_attempts or not projected_policy:
            return None
        body: dict[str, Any] = {
            "schema_version": PE_FAILURE_AUDIT_VERSION,
            "error_type": type(exc).__name__,
            "pe_generation_policy": projected_policy,
            "attempt_count": len(projected_attempts),
            "attempts": projected_attempts,
            "source_attempt_ledger_sha256": sha256_json(source_attempts),
            "projected_attempt_ledger_sha256": sha256_json(projected_attempts),
        }
        terminal = _project_pe_provider_provenance(
            getattr(exc, "provenance", None)
        )
        if terminal:
            body["terminal_provider_provenance"] = terminal
        body["audit_sha256"] = sha256_json(body)
        return body
    except (TypeError, ValueError):
        # Never mask the original provider failure because an untrusted custom
        # port attached malformed audit metadata.
        return None


def assign_pe_mode(story_id: str, *, rewrite_fraction: float = 0.5) -> str:
    """Assign a story, not an individual turn, to one stable PE mode."""

    if not 0.0 <= float(rewrite_fraction) <= 1.0:
        raise ValueError("rewrite_fraction must be in [0, 1]")
    bucket = int(hashlib.sha256(story_id.encode("utf-8")).hexdigest()[:16], 16)
    threshold = int(float(rewrite_fraction) * (1 << 64))
    return "gemini_rewrite" if bucket < threshold else "passthrough"


def build_closed_teacher_system_prompt(
    *,
    allowed_tool_names: Sequence[str],
    phase: str,
    current_action_contract: Mapping[str, Any] | None = None,
    private_retry_feedback: Sequence[Mapping[str, Any]] = (),
    prompt_authoring: bool = False,
) -> str:
    """Build the exact ordinary-JSON protocol prompt for one closed-model turn.

    Tool schemas are text in this system message.  They are deliberately not
    sent as provider-native function definitions.  The open-model SFT export
    continues to use ``OPEN_MODEL_SYSTEM_PROMPT`` and native calls.
    """

    schemas = get_tool_schemas(allowed_tool_names, prompt_authoring=prompt_authoring)
    base = system_prompt_for_mode(prompt_authoring)
    native_suffix = (
        "Use native structured tool calls only. Do not emit XML calls, narrate an "
        "intended call instead of making it, reveal hidden reasoning, or end with a "
        "natural-language answer. Continue until finish_memory_session returns "
        "done=true or the runtime reports a recoverable pending checkpoint."
    )
    if native_suffix not in base:
        raise CollectorError("open-model prompt terminal transport clause changed")
    policy = base.replace(native_suffix, "").rstrip()
    retry = ""
    if private_retry_feedback:
        retry = (
            "\n\nPRIVATE REPAIR FEEDBACK FOR THIS TURN ONLY\n"
            + canonical_json(list(private_retry_feedback))
        )
    action_contract = ""
    if current_action_contract:
        action_contract = (
            "\n\nCURRENT RUNTIME ACTION CONTRACT\n"
            "This object contains deterministic FSM prerequisites, not a semantic "
            "policy decision. Obey every exact field. When exact_arguments is "
            "present, return that tool and those arguments verbatim.\n"
            + canonical_json(dict(current_action_contract))
        )
    return (
        policy
        + "\n\nCLOSED-TEACHER TRANSPORT (not part of the learned open-model prompt)\n"
        + "You are in phase "
        + phase
        + ". Only the tools listed below are legal now. Return exactly one strict "
        + "JSON object and no markdown or surrounding text, with exactly this shape: "
        + '{"reason":"concise action justification","tool_call":{"name":"one '
        + 'allowed name","arguments":{}}}. The arguments object must satisfy the '
        + "selected tool schema exactly. Do not emit provider-native tool calls.\n"
        + "AVAILABLE TOOL SCHEMAS:\n"
        + canonical_json(schemas)
        + action_contract
        + retry
    )


def _closed_teacher_action_contract(
    runtime: MemoryPolicyRuntime,
    allowed_tool_names: Sequence[str],
) -> dict[str, Any]:
    """Expose only deterministic next-step facts already present in history.

    The contract removes no valid policy choice.  It makes causal FSM
    prerequisites explicit so the plain-JSON Teacher does not pay for an
    action which the runtime can prove will be rejected before execution.
    """

    allowed = tuple(allowed_tool_names)
    state = runtime.state.data
    contract: dict[str, Any] = {
        "phase": runtime.phase,
        "allowed_tool_names": list(allowed),
    }
    if runtime.phase == "need_registry":
        contract["exact_arguments"] = {
            "tool_name": "view_material_registry",
            "arguments": {"snapshot_id": None, "cursor": None, "limit": 200},
        }
        return contract

    snapshot_id = state.get("registry_snapshot_id")
    if runtime.phase == "generation_review":
        candidate_id = state.get("condition_review_target_id")
        bundle = state.get("generation_bundle_private") or {}
        if allowed == ("view_memories",):
            contract["exact_arguments"] = {
                "tool_name": "view_memories",
                "arguments": {
                    "snapshot_id": snapshot_id,
                    "items": [
                        {
                            "source_kind": "generation_derived",
                            "source_id": candidate_id,
                            "discovery_id": bundle.get("bundle_id"),
                            "representation_id": None,
                            "time_range": None,
                        }
                    ],
                },
            }
            return contract
        if allowed == ("inspect_candidate",):
            plan = state.get("accepted_evidence_plan") or {}
            comparison_ids = [
                row["evidence_id"]
                for row in plan.get("ranked_bindings") or []
                if row.get("modality") in runtime.perceivable_modalities
            ]
            contract["exact_arguments"] = {
                "tool_name": "inspect_candidate",
                "arguments": {
                    "candidate_id": candidate_id,
                    "requested_uses": ["condition_adherence"],
                    "comparison_evidence_ids": comparison_ids,
                },
            }
            return contract

    if runtime.phase == "writeback":
        # An adopted reusable uploaded voice is a mandatory durable-memory
        # obligation.  Surface its deterministic causal prefix so the closed
        # Teacher cannot accidentally jump from condition review straight to
        # update_memory_bank: the exact audio range must first be viewed and
        # independently inspected for metadata-level reuse suitability.
        selected_plan = runtime._selected_generation_plan()  # noqa: SLF001
        missing_voice_records = runtime._missing_required_uploaded_voice_memories(  # noqa: SLF001
            selected_plan
        )
        if missing_voice_records:
            contract["required_uploaded_voice_writebacks"] = _json_copy(
                missing_voice_records
            )
            for required in missing_voice_records:
                evidence_id = str(required["evidence_id"])
                representation_id = str(required["representation_id"])
                source_selector = dict(required["source_selector"])
                matching_views = [
                    row
                    for row in state.get("views", {}).values()
                    if isinstance(row, Mapping)
                    and row.get("source_kind")
                    in {"uploaded_material", "uploaded_asset"}
                    and row.get("evidence_id") == evidence_id
                    and row.get("representation_id") == representation_id
                    and row.get("selector") == source_selector
                ]
                upload = state.get("uploaded_evidence", {}).get(evidence_id) or {}
                if not matching_views and "view_memories" in allowed:
                    contract["exact_arguments"] = {
                        "tool_name": "view_memories",
                        "arguments": {
                            "snapshot_id": snapshot_id,
                            "items": [
                                {
                                    "source_kind": "uploaded_material",
                                    "source_id": evidence_id,
                                    "discovery_id": snapshot_id,
                                    "representation_id": representation_id,
                                    "time_range": {
                                        "start_ms": source_selector["start_ms"],
                                        "end_ms": source_selector["end_ms"],
                                    },
                                }
                            ],
                        },
                    }
                    return contract
                if not upload.get("inspection_id") and "inspect_candidate" in allowed:
                    contract["exact_arguments"] = {
                        "tool_name": "inspect_candidate",
                        "arguments": {
                            "candidate_id": evidence_id,
                            "requested_uses": ["memory_writeback"],
                            "comparison_evidence_ids": [],
                        },
                    }
                    return contract
            contract["writeback_gate"] = (
                "Before finish_memory_session, independently pass the pending "
                "metadata-level audio assessment and include every required exact-"
                "range voice record in update_memory_bank. Copy the viewed "
                "writeback_record_template, including source_selector, exactly."
            )

    pending_candidate_id = state.get("pending_candidate_id")
    if allowed == ("inspect_candidate",) and pending_candidate_id:
        candidate = state.get("created_evidence", {}).get(pending_candidate_id) or {}
        purpose = str(candidate.get("purpose") or "reference_evidence")
        contract["required_arguments"] = {
            "candidate_id": pending_candidate_id,
            "requested_uses_must_include": purpose,
        }
    pending_assessment_id = state.get("pending_assessment_id")
    if allowed == ("submit_candidate_assessment",) and pending_assessment_id:
        inspection = state.get("inspections", {}).get(pending_assessment_id) or {}
        contract["required_arguments"] = {
            "inspection_id": pending_assessment_id,
            "candidate_id": inspection.get("candidate_id"),
            "check_ids_exactly": [
                row.get("check_id") for row in inspection.get("checklist") or []
            ],
            "use_ids_exactly": [
                row.get("use_id") for row in inspection.get("requested_uses") or []
            ],
            "copy_deterministic_check_results_exactly": {
                row["check_id"]: row["deterministic_result"]
                for row in inspection.get("checklist") or []
                if row.get("deterministic_result") is not None
            },
        }

    if runtime.phase == "planning":
        plan = state.get("accepted_evidence_plan")
        contract["evidence_plan_immutable_fields"] = {
            "request_id": state.get("request_id"),
            "raw_request_sha256": state.get("shot_requirement_sha256"),
            "condition_budget_id": (state.get("condition_budget") or {}).get(
                "budget_id"
            ),
            "next_parent_plan_id": (
                plan.get("plan_id") if isinstance(plan, Mapping) else None
            ),
            "next_version": (
                int(plan.get("version", 0)) + 1
                if isinstance(plan, Mapping)
                else 1
            ),
            "revision_requirements_rule": (
                "copy_parent_requirements_exactly"
                if isinstance(plan, Mapping)
                else "author_from_raw_request_and_registry"
            ),
        }
    return contract


def _hydrate_teacher_media(
    messages: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    *,
    perceivable_modalities: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Privately attach verified local bytes to pathless public history blocks."""

    allowed = set(perceivable_modalities)
    private_by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = str(event.get("event_id") or "")
        for index, attachment in enumerate(event.get("media_attachments_private") or []):
            if not isinstance(attachment, Mapping):
                raise CollectorError("runtime media attachment is malformed")
            path = require_file(
                str(attachment.get("path") or ""), label="Teacher media attachment"
            )
            digest = sha256_file(path)
            representation_id = str(attachment.get("representation_id") or "")
            media_id = "media_" + hashlib.sha256(
                f"{event_id}\0{index}\0{representation_id}\0{digest}".encode("utf-8")
            ).hexdigest()[:24]
            if media_id in private_by_id:
                raise CollectorError("Teacher media attachment ID collision")
            private_by_id[media_id] = {
                "path": str(path.resolve()),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "representation_id": representation_id,
                "media_type": str(attachment.get("type") or ""),
            }
            for field in ("parent_sha256", "derivation"):
                if attachment.get(field) is not None:
                    private_by_id[media_id][field] = _json_copy(
                        attachment[field]
                    )

    hydrated = _json_copy(list(messages))
    manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    for message_index, message in enumerate(hydrated):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "media":
                continue
            media_id = str(block.get("media_id") or "")
            private = private_by_id.get(media_id)
            if private is None:
                raise CollectorError(f"Teacher media {media_id!r} has no private bytes")
            media_type = str(block.get("media_type") or "")
            if media_type not in allowed:
                raise CollectorError(
                    f"non-perceivable {media_type!r} media reached the Teacher payload"
                )
            for field in ("sha256", "size_bytes", "representation_id", "media_type"):
                if block.get(field) != private[field]:
                    raise CollectorError(
                        f"Teacher media {media_id!r} differs from its private manifest"
                    )
            for field in ("parent_sha256", "derivation"):
                if block.get(field) != private.get(field):
                    raise CollectorError(
                        f"Teacher media {media_id!r} differs in {field} derivation"
                    )
            block["path"] = private["path"]
            seen.add(media_id)
            manifest.append(
                {
                    "message_index": message_index,
                    "content_index": content_index,
                    "media_id": media_id,
                    "media_type": media_type,
                    "mime_type": block.get("mime_type"),
                    "size_bytes": private["size_bytes"],
                    "sha256": private["sha256"],
                    "representation_id": private["representation_id"],
                    **(
                        {"parent_sha256": private["parent_sha256"]}
                        if private.get("parent_sha256") is not None
                        else {}
                    ),
                    **(
                        {"derivation": _json_copy(private["derivation"])}
                        if private.get("derivation") is not None
                        else {}
                    ),
                }
            )
    if seen != set(private_by_id):
        raise CollectorError("private Teacher media is not represented in public history")
    return hydrated, manifest


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    output_root: Path
    width: int = 1280
    height: int = 736
    num_frames: int = 241
    duration_sec: float | None = None
    story_workers: int = 64
    echo_concurrency: int = 32
    image_concurrency: int = 32
    teacher_qps: float = 2.8
    tool_temperature: float = 0.25
    max_turns_per_shot: int = 48
    max_hidden_turn_retries: int = 4
    teacher_media_limit: int = 64
    visual_creation_budget: int = 2
    max_condition_revisions: int = 1
    max_image_reference_inputs: int = 4
    rewrite_fraction: float = 0.5
    max_total_conditions: int = 15
    max_image_conditions: int = 8
    max_audio_conditions: int = 7
    perceivable_modalities: tuple[str, ...] = ("image", "video", "text")
    prompt_authoring: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root).expanduser().resolve())
        object.__setattr__(self, "perceivable_modalities", tuple(self.perceivable_modalities))
        for name in (
            "width",
            "height",
            "num_frames",
            "story_workers",
            "echo_concurrency",
            "image_concurrency",
            "max_turns_per_shot",
            "max_hidden_turn_retries",
            "teacher_media_limit",
            "max_image_reference_inputs",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.visual_creation_budget) is not int or self.visual_creation_budget < 0:
            raise ValueError("visual_creation_budget must be a non-negative integer")
        if isinstance(self.teacher_qps, bool) or not isinstance(
            self.teacher_qps, (int, float)
        ) or self.teacher_qps <= 0:
            raise ValueError("teacher_qps must be a positive finite number")
        if not float("-inf") < float(self.teacher_qps) < float("inf"):
            raise ValueError("teacher_qps must be a positive finite number")
        if type(self.max_condition_revisions) is not int or self.max_condition_revisions < 0:
            raise ValueError("max_condition_revisions must be non-negative")
        if not 0 <= self.rewrite_fraction <= 1:
            raise ValueError("rewrite_fraction must be in [0, 1]")
        if self.max_total_conditions < self.max_image_conditions + self.max_audio_conditions:
            raise ValueError(
                "max_total_conditions must permit the advertised image/audio binding budget"
            )

    def semantic_manifest(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_root"] = str(self.output_root)
        value["perceivable_modalities"] = list(self.perceivable_modalities)
        return value


@dataclass(frozen=True, slots=True)
class EchoInvocationResult:
    output_path: str
    generation_id: str
    request_payload_sha256: str
    output_sha256: str
    receipt: dict[str, Any]


class ProviderVisualEvidenceCreator:
    """Adapt the real image transport to Memory Policy's idempotent port."""

    def __init__(
        self,
        provider: Any,
        *,
        width: int,
        height: int,
        semaphore: threading.Semaphore | None = None,
        request_namespace: str | None = None,
    ) -> None:
        self.provider = provider
        self.width = int(width)
        self.height = int(height)
        self.semaphore = semaphore
        self.request_namespace = request_namespace

    def create(
        self, request: VisualEvidenceCreationRequest
    ) -> VisualEvidenceCreationResult:
        provider_request_id = request.request_id
        if self.request_namespace is not None:
            identity = f'{self.request_namespace}:{Path(request.output_dir).resolve()}:{request.request_id}'
            provider_request_id = 'storymem_image_' + hashlib.sha256(identity.encode()).hexdigest()[:40]
        provider_request = ImageGenerationRequest(
            request_id=provider_request_id,
            prompt=str(request.creation_spec["visual_brief"]),
            reference_paths=[str(path) for path in request.input_paths],
            seed=None,
            width=self.width,
            height=self.height,
            output_dir=request.output_dir,
        )
        if self.semaphore is None:
            result = self.provider.generate(provider_request)
        else:
            with self.semaphore:
                result = self.provider.generate(provider_request)
        if str(result.request_id) != provider_request_id:
            raise CollectorError("image provider changed the idempotency key")
        return VisualEvidenceCreationResult(
            request_id=request.request_id,
            output_path=Path(result.output_path).resolve(),
            private_receipt=result.to_dict(),
        )


class Echo15WireTransport:
    """Submit adapter-compiled requests without adding a seed or retry axis."""

    _ALLOWED_REQUEST_KEYS = frozenset(
        {
            "work_id",
            "shot_id",
            "job_id",
            "prompt",
            "memory_slots",
            "condition_img",
            "num_frames",
            "duration_sec",
            "width",
            "height",
            "seed",
        }
    )

    def __init__(
        self,
        base_urls: Sequence[str],
        *,
        concurrency: int = 32,
        per_endpoint_concurrency: int = 1,
        generation_timeout: float = 7200.0,
        request_timeout: float = 120.0,
        poll_seconds: float = 5.0,
        http_attempts: int = 3,
        allow_seed: bool = False,
    ) -> None:
        urls = tuple(dict.fromkeys(str(item).rstrip("/") for item in base_urls if str(item).strip()))
        if not urls:
            raise ValueError("at least one Echo URL is required")
        if concurrency < 1 or per_endpoint_concurrency < 1:
            raise ValueError("Echo concurrency limits must be positive")
        self.base_urls = urls
        self.allow_seed = allow_seed
        # EchoVideoProvider is reused only for its HTTP retry/poll/download
        # transport.  Its generate() method is never called because that
        # legacy method injects a seed and terminal -infraN jobs.
        self._clients = tuple(
            EchoVideoProvider(
                url,
                timeout=generation_timeout,
                request_timeout=request_timeout,
                poll_seconds=poll_seconds,
                max_attempts=http_attempts,
                max_terminal_attempts=1,
                media_transport="data_uri",
            )
            for url in urls
        )
        # Route assignment and capacity acquisition are one operation.  A
        # fixed hash selected before semaphore acquisition can queue several
        # jobs on one node while peer nodes are idle.  These counters let a
        # new generation claim an actually available endpoint, while a
        # replay can wait for its already-persisted endpoint without drifting.
        self._global_capacity = int(concurrency)
        self._per_endpoint_capacity = int(per_endpoint_concurrency)
        self._global_active = 0
        self._endpoint_active = [0 for _ in urls]
        self._tie_cursor = 0
        self._capacity_changed = threading.Condition()

    @staticmethod
    def _response_sha256(response: Any, body: Mapping[str, Any]) -> str:
        payload = getattr(response, "content", None)
        if not isinstance(payload, bytes):
            payload = canonical_json(body).encode("utf-8")
        return _sha_bytes(payload)

    @staticmethod
    def _resolve_openapi_schema(
        document: Mapping[str, Any], node: Any
    ) -> Mapping[str, Any] | None:
        current = node
        seen: set[str] = set()
        while isinstance(current, Mapping) and isinstance(current.get("$ref"), str):
            reference = str(current["$ref"])
            if not reference.startswith("#/") or reference in seen:
                return None
            seen.add(reference)
            resolved: Any = document
            for raw_part in reference[2:].split("/"):
                part = raw_part.replace("~1", "/").replace("~0", "~")
                if not isinstance(resolved, Mapping) or part not in resolved:
                    return None
                resolved = resolved[part]
            current = resolved
        return current if isinstance(current, Mapping) else None

    @classmethod
    def _openapi_audio_memory_proof(
        cls, document: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        paths = document.get("paths")
        r2v = paths.get("/r2v") if isinstance(paths, Mapping) else None
        post = r2v.get("post") if isinstance(r2v, Mapping) else None
        request_body = post.get("requestBody") if isinstance(post, Mapping) else None
        content = (
            request_body.get("content")
            if isinstance(request_body, Mapping)
            else None
        )
        media = content.get("application/json") if isinstance(content, Mapping) else None
        request_schema = cls._resolve_openapi_schema(
            document, media.get("schema") if isinstance(media, Mapping) else None
        )
        properties = (
            request_schema.get("properties")
            if isinstance(request_schema, Mapping)
            else None
        )
        memory_slots = (
            properties.get("memory_slots") if isinstance(properties, Mapping) else None
        )
        slot_schema = cls._resolve_openapi_schema(
            document,
            memory_slots.get("items") if isinstance(memory_slots, Mapping) else None,
        )
        slot_properties = (
            slot_schema.get("properties")
            if isinstance(slot_schema, Mapping)
            else None
        )
        if not isinstance(slot_properties, Mapping) or "audio_url" not in slot_properties:
            return None
        max_slots = memory_slots.get("maxItems") if isinstance(memory_slots, Mapping) else None
        return {
            "request_schema": copy.deepcopy(dict(request_schema)),
            "memory_slot_schema": copy.deepcopy(dict(slot_schema)),
            "max_memory_slots": max_slots,
        }

    def preflight(self, *, require_audio_memory: bool = True) -> list[dict[str, Any]]:
        """Verify every configured endpoint without enforcing prompt policy server-side."""

        receipts: list[dict[str, Any]] = []
        for client in self._clients:
            response, attempts = client._request(  # noqa: SLF001 - transport reuse
                "GET", f"{client.base_url}/health", headers=client.extra_headers
            )
            body = self._response_json(response)
            inference = body.get("inference")
            workers = inference.get("workers") if isinstance(inference, Mapping) else None
            scheduler = body.get("scheduler")
            scheduler_workers = (
                scheduler.get("gpu_workers")
                if isinstance(scheduler, Mapping)
                else None
            )
            unready_states = {
                "starting",
                "resource_error",
                "waiting_resources",
                "failed",
            }

            def worker_is_ready(worker: Any) -> bool:
                if not isinstance(worker, Mapping):
                    return False
                state = str(worker.get("state") or "").strip().lower()
                if not state or state in unready_states or worker.get("error"):
                    return False
                if "ready" in worker:
                    # Do not coerce strings, integers, or null into readiness.
                    return worker.get("ready") is True
                # Two deployed legacy nodes predate the boolean mirror.  Their
                # authoritative lifecycle state is exactly ``ready``.
                return state == "ready"

            ready_workers = [
                worker
                for worker in workers or []
                if worker_is_ready(worker)
            ]
            if (
                body.get("status") != "ok"
                or not isinstance(inference, Mapping)
                or inference.get("enabled") is not True
                or not isinstance(workers, list)
                or not ready_workers
                or type(scheduler_workers) is not int
                or scheduler_workers < 1
                or scheduler_workers != len(workers)
            ):
                raise CollectorError(
                    "Echo endpoint is not inference-ready: "
                    + sha256_text(client.base_url)
                )
            contract = body.get("r2v_contract")
            capability_source = None
            capability_schema_sha256 = None
            openapi_response_sha256 = None
            capability_attempt_count = 0
            openapi_slots = None
            if isinstance(contract, Mapping) and contract.get(
                "enable_audio_memory"
            ) is True:
                capability_source = "health.r2v_contract"
                capability_schema_sha256 = sha256_json(contract)
            elif require_audio_memory:
                if isinstance(contract, Mapping) and contract.get(
                    "enable_audio_memory"
                ) is False:
                    raise CollectorError(
                        "Echo endpoint explicitly disables audio-memory support: "
                        + sha256_text(client.base_url)
                    )
                openapi_response, openapi_attempts = client._request(  # noqa: SLF001
                    "GET",
                    f"{client.base_url}/openapi.json",
                    headers=client.extra_headers,
                )
                openapi = self._response_json(openapi_response)
                proof = self._openapi_audio_memory_proof(openapi)
                if proof is None:
                    raise CollectorError(
                        "Echo endpoint cannot prove audio-memory support: "
                        + sha256_text(client.base_url)
                    )
                capability_source = "openapi.r2v_memory_slot_audio_url"
                capability_schema_sha256 = sha256_json(proof)
                openapi_response_sha256 = self._response_sha256(
                    openapi_response, openapi
                )
                capability_attempt_count = len(openapi_attempts)
                openapi_slots = proof.get("max_memory_slots")
            advertised_slots = (
                contract.get("max_memory_slots")
                if isinstance(contract, Mapping)
                else None
            )
            if advertised_slots is None:
                advertised_slots = openapi_slots
            if advertised_slots is not None and (
                type(advertised_slots) is not int or advertised_slots < 7
            ):
                raise CollectorError(
                    "Echo endpoint advertises fewer than seven memory slots: "
                    + sha256_text(client.base_url)
                )
            receipts.append(
                {
                    "endpoint_sha256": sha256_text(client.base_url),
                    "status": "ok",
                    "worker_count": len(workers),
                    "ready_worker_count": len(ready_workers),
                    "scheduler_worker_count": scheduler_workers,
                    "audio_memory": (
                        True
                        if capability_source is not None
                        else (
                            contract.get("enable_audio_memory")
                            if isinstance(contract, Mapping)
                            else None
                        )
                    ),
                    "max_memory_slots": advertised_slots,
                    "capability_source": capability_source,
                    "capability_schema_sha256": capability_schema_sha256,
                    "health_response_sha256": self._response_sha256(response, body),
                    "openapi_response_sha256": openapi_response_sha256,
                    "attempt_count": len(attempts),
                    "capability_attempt_count": capability_attempt_count,
                }
            )
        return receipts

    @staticmethod
    def _response_json(response: Any) -> dict[str, Any]:
        try:
            value = response.json()
        except Exception as exc:  # noqa: BLE001 - provider boundary
            raise ProviderProtocolError("Echo returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ProviderProtocolError("Echo response must be a JSON object")
        return value

    @staticmethod
    def _media_to_wire(path_value: Any, *, expected: str, label: str) -> tuple[str, dict[str, Any]]:
        path = require_file(str(path_value), label=label)
        mime_type = detect_mime(path)
        if not mime_type.startswith(expected + "/"):
            raise CollectorError(f"{label} must be {expected}, got {mime_type}")
        # Read once: the digest below describes the exact bytes embedded in the
        # request, rather than a second read which could race with replacement.
        payload = path.read_bytes()
        if not payload:
            raise CollectorError(f"{label} must not be empty")
        encoded = "data:" + mime_type + ";base64," + base64.b64encode(payload).decode("ascii")
        return encoded, {
            "label": label,
            "mime_type": mime_type,
            "size_bytes": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        }

    @staticmethod
    def _verify_reserved_media(
        media: Mapping[str, Any],
        reserved_canonical_conditions: Sequence[Mapping[str, Any]] | None,
    ) -> None:
        wire_rows: list[tuple[str, str, str]] = []
        condition = media.get("condition")
        if isinstance(condition, Mapping):
            wire_rows.append(
                ("image", str(condition.get("sha256") or ""), "condition_img")
            )
        for raw_slot in media.get("memory_slots") or []:
            if not isinstance(raw_slot, Mapping):
                raise CollectorError("wire media manifest contains an invalid slot")
            ordinal = raw_slot.get("ordinal")
            image = raw_slot.get("image")
            audio = raw_slot.get("audio")
            reuses_opening = raw_slot.get("image_reuses_opening_condition")
            if not isinstance(reuses_opening, bool):
                raise CollectorError(
                    "wire media manifest has no explicit opening-image reuse flag"
                )
            condition_digest = (
                str(condition.get("sha256") or "")
                if isinstance(condition, Mapping)
                else ""
            )
            image_digest = (
                str(image.get("sha256") or "") if isinstance(image, Mapping) else ""
            )
            actually_reuses_opening = bool(
                condition_digest and image_digest == condition_digest
            )
            if reuses_opening != actually_reuses_opening:
                raise CollectorError(
                    "opening-image reuse flag differs from the physical wire media"
                )
            if reuses_opening and not isinstance(audio, Mapping):
                raise CollectorError(
                    "opening-image reuse is legal only for a slot carrying audio"
                )
            if isinstance(image, Mapping):
                wire_rows.append(
                    (
                        "image",
                        str(image.get("sha256") or ""),
                        f"memory_slots[{ordinal}].image_url",
                    )
                )
            if isinstance(audio, Mapping):
                wire_rows.append(
                    (
                        "audio",
                        str(audio.get("sha256") or ""),
                        f"memory_slots[{ordinal}].audio_url",
                    )
                )

        if reserved_canonical_conditions is None:
            if wire_rows:
                raise CollectorError(
                    "conditioned Echo request has no reserved canonical media manifest"
                )
            return

        expected_rows: list[tuple[str, str]] = []
        for ordinal, raw in enumerate(reserved_canonical_conditions, start=1):
            if (
                not isinstance(raw, Mapping)
                or set(raw) != {
                    "ordinal",
                    "modality",
                    "content_sha256",
                    "applied_effect",
                }
                or raw.get("ordinal") != ordinal
                or raw.get("modality") not in {"image", "audio"}
                or _SHA256_RE.fullmatch(str(raw.get("content_sha256") or "")) is None
                or not isinstance(raw.get("applied_effect"), str)
                or not raw["applied_effect"]
            ):
                raise CollectorError("reserved canonical condition manifest is malformed")
            expected_rows.append(
                (str(raw["modality"]), str(raw["content_sha256"]))
            )

        # The Echo adapter may repeat the opening image once as the visual half
        # of its paired voice slot.  That is the only legal duplicate physical
        # input; canonical conditions intentionally count those bytes once.
        counts: dict[tuple[str, str], list[str]] = {}
        normalized_wire: list[tuple[str, str]] = []
        for modality, digest, location in wire_rows:
            key = (modality, digest)
            locations = counts.setdefault(key, [])
            locations.append(location)
            if len(locations) == 1:
                normalized_wire.append(key)
        for key, locations in counts.items():
            if len(locations) == 1:
                continue
            duplicate_is_paired_opening = (
                len(locations) == 2
                and key[0] == "image"
                and locations[0] == "condition_img"
                and locations[1].endswith(".image_url")
                and any(
                    slot.get("image_reuses_opening_condition") is True
                    and str((slot.get("image") or {}).get("sha256") or "") == key[1]
                    and isinstance(slot.get("audio"), Mapping)
                    for slot in media.get("memory_slots") or []
                    if isinstance(slot, Mapping)
                )
            )
            if not duplicate_is_paired_opening:
                raise CollectorError(
                    "Echo wire request duplicates media outside the reserved opening/voice pairing"
                )
        if normalized_wire != expected_rows:
            raise CollectorError(
                "Echo wire media bytes differ from the durable generation reservation"
            )

    def _wire_payload(
        self,
        request: Mapping[str, Any],
        *,
        reserved_canonical_conditions: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        value = _json_copy(dict(request))
        extra = set(value) - self._ALLOWED_REQUEST_KEYS
        if extra:
            raise CollectorError(f"adapter Echo request has unknown keys: {sorted(extra)}")
        if "seed" in value and not self.allow_seed:
            raise CollectorError("Memory Policy Echo invocation must not contain a seed")
        if "seed" in value and (type(value["seed"]) is not int or not 0 <= value["seed"] < 2 ** 32):
            raise CollectorError("Echo seed must be a host-owned uint32")
        for key in ("work_id", "shot_id", "job_id", "prompt"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise CollectorError(f"adapter Echo request has invalid {key}")
        if value["prompt"] != value["prompt"].strip():
            raise CollectorError("Echo prompt must already be trimmed by the adapter")
        slots = value.get("memory_slots")
        if not isinstance(slots, list) or len(slots) > 7:
            raise CollectorError("Echo memory_slots must be an array of at most seven slots")
        media = {"condition": None, "memory_slots": []}
        if value.get("condition_img") is not None:
            value["condition_img"], media["condition"] = self._media_to_wire(
                value["condition_img"], expected="image", label="condition_img"
            )
        encoded_slots = []
        for index, raw_slot in enumerate(slots):
            if not isinstance(raw_slot, Mapping):
                raise CollectorError(f"memory_slots[{index}] must be an object")
            slot = _json_copy(dict(raw_slot))
            if set(slot) - {
                "image_url",
                "audio_url",
                "audio_mode",
                "image_reuses_opening_condition",
            }:
                raise CollectorError(f"memory_slots[{index}] has unknown fields")
            reuses_opening = slot.pop("image_reuses_opening_condition", None)
            if not isinstance(reuses_opening, bool):
                raise CollectorError(
                    f"memory_slots[{index}] must declare image_reuses_opening_condition"
                )
            image_value, image_manifest = self._media_to_wire(
                slot.get("image_url"),
                expected="image",
                label=f"memory_slots[{index}].image_url",
            )
            encoded = {"image_url": image_value}
            audio_manifest = None
            if slot.get("audio_url") is not None:
                encoded["audio_url"], audio_manifest = self._media_to_wire(
                    slot["audio_url"],
                    expected="audio",
                    label=f"memory_slots[{index}].audio_url",
                )
            else:
                if slot.get("audio_mode") not in {None, "empty"}:
                    raise CollectorError("unsupported Echo audio_mode")
                encoded["audio_mode"] = "empty"
            encoded_slots.append(encoded)
            media["memory_slots"].append(
                {
                    "ordinal": index + 1,
                    "image": image_manifest,
                    "audio": audio_manifest,
                    "image_reuses_opening_condition": reuses_opening,
                }
            )
        value["memory_slots"] = encoded_slots
        self._verify_reserved_media(media, reserved_canonical_conditions)
        return value, media

    def _endpoint_index(self, generation_id: str) -> int:
        """Return the legacy route used before durable idle-node assignment."""

        digest = hashlib.sha256(generation_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % len(self._clients)

    @staticmethod
    def _route_path(destination: Path, generation_id: str) -> Path:
        return destination / f"{safe_name(generation_id)}.echo_route.json"

    def _read_route(
        self,
        route_path: Path,
        *,
        generation_id: str,
        request_payload_sha256: str,
    ) -> dict[str, Any] | None:
        if route_path.is_symlink():
            raise CollectorError("persisted Echo endpoint route is unsafe")
        if not route_path.exists():
            return None
        if not route_path.is_file():
            raise CollectorError("persisted Echo endpoint route is unsafe")
        route = _read_object(route_path, label="Echo endpoint route")
        route_body = {
            key: item for key, item in route.items() if key != "route_sha256"
        }
        if (
            route.get("schema_version") != ECHO_WIRE_ROUTE_VERSION
            or set(route) != {"schema_version", "endpoint_binding", "route_sha256"}
            or route.get("route_sha256") != sha256_json(route_body)
            or not isinstance(route.get("endpoint_binding"), Mapping)
        ):
            raise CollectorError("persisted Echo endpoint route is malformed or unsealed")
        return _validated_echo_endpoint_binding(
            route["endpoint_binding"],
            generation_id=generation_id,
            request_payload_sha256=request_payload_sha256,
        )

    @staticmethod
    def _sealed_route(endpoint_binding: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "schema_version": ECHO_WIRE_ROUTE_VERSION,
            "endpoint_binding": _json_copy(dict(endpoint_binding)),
        }
        return {**body, "route_sha256": sha256_json(body)}

    def _route_endpoint_index(self, route: Mapping[str, Any]) -> int:
        endpoint_sha256 = route.get("endpoint_sha256")
        matches = [
            index
            for index, client in enumerate(self._clients)
            if sha256_text(client.base_url) == endpoint_sha256
        ]
        if len(matches) != 1:
            raise CollectorError(
                "persisted Echo endpoint route is absent or ambiguous; refusing to drift"
            )
        return matches[0]

    def _acquire_endpoint(self, preferred_index: int | None = None) -> int:
        """Atomically claim either one free endpoint or one specific replay route."""

        if preferred_index is not None and not 0 <= preferred_index < len(self._clients):
            raise ValueError("preferred Echo endpoint index is out of range")
        with self._capacity_changed:
            while True:
                if self._global_active < self._global_capacity:
                    if preferred_index is None:
                        count = len(self._clients)
                        candidates = [
                            (self._tie_cursor + offset) % count
                            for offset in range(count)
                        ]
                        available = [
                            index
                            for index in candidates
                            if self._endpoint_active[index]
                            < self._per_endpoint_capacity
                        ]
                        endpoint_index = min(
                            available,
                            key=lambda index: (
                                self._endpoint_active[index],
                                candidates.index(index),
                            ),
                            default=None,
                        )
                    elif (
                        self._endpoint_active[preferred_index]
                        < self._per_endpoint_capacity
                    ):
                        endpoint_index = preferred_index
                    else:
                        endpoint_index = None
                    if endpoint_index is not None:
                        self._global_active += 1
                        self._endpoint_active[endpoint_index] += 1
                        if preferred_index is None:
                            self._tie_cursor = (endpoint_index + 1) % len(self._clients)
                        return endpoint_index
                self._capacity_changed.wait()

    def _release_endpoint(self, endpoint_index: int) -> None:
        with self._capacity_changed:
            if (
                self._global_active < 1
                or self._endpoint_active[endpoint_index] < 1
            ):
                raise RuntimeError("Echo endpoint capacity counter underflow")
            self._global_active -= 1
            self._endpoint_active[endpoint_index] -= 1
            self._capacity_changed.notify_all()

    def _completed_result(
        self,
        *,
        receipt_path: Path,
        output_path: Path,
        generation_id: str,
        request_payload_sha256: str,
    ) -> EchoInvocationResult | None:
        if not (receipt_path.is_file() and output_path.is_file()):
            return None
        receipt = _read_object(receipt_path, label="Echo invocation receipt")
        if (
            receipt.get("generation_id") != generation_id
            or receipt.get("request_payload_sha256") != request_payload_sha256
            or receipt.get("output_sha256") != sha256_file(output_path)
        ):
            raise CollectorError("persisted Echo invocation receipt conflicts with bytes")
        return EchoInvocationResult(
            output_path=str(output_path),
            generation_id=generation_id,
            request_payload_sha256=request_payload_sha256,
            output_sha256=receipt["output_sha256"],
            receipt=receipt,
        )

    def invoke(
        self,
        request: Mapping[str, Any],
        *,
        output_dir: str | Path,
        resume: bool,
        reserved_canonical_conditions: Sequence[Mapping[str, Any]] | None = None,
        endpoint_binding: Mapping[str, Any] | None = None,
        persist_endpoint_binding: Callable[[Mapping[str, Any]], None] | None = None,
        legacy_route_fallback: bool = False,
    ) -> EchoInvocationResult:
        payload, media_manifest = self._wire_payload(
            request,
            reserved_canonical_conditions=reserved_canonical_conditions,
        )
        generation_id = str(payload["job_id"])
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        receipt_path = destination / f"{safe_name(generation_id)}.echo_receipt.json"
        output_path = destination / f"{safe_name(generation_id)}.mp4"
        payload_hash = sha256_json(payload)
        validated_binding = (
            None
            if endpoint_binding is None
            else _validated_echo_endpoint_binding(
                endpoint_binding,
                generation_id=generation_id,
                request_payload_sha256=payload_hash,
            )
        )
        if validated_binding is not None and not resume:
            raise CollectorError(
                "a durable Echo endpoint binding may only be used for a replay"
            )
        if validated_binding is None and resume and not legacy_route_fallback:
            raise CollectorError(
                "v4 Echo replay has no sealed endpoint binding; refusing to guess a route"
            )
        if validated_binding is None and persist_endpoint_binding is None:
            raise CollectorError(
                "Echo invocation cannot POST before its endpoint binding is persisted"
            )
        completed = self._completed_result(
            receipt_path=receipt_path,
            output_path=output_path,
            generation_id=generation_id,
            request_payload_sha256=payload_hash,
        )
        if completed is not None:
            if not resume:
                raise CollectorError(
                    "fresh Echo invocation unexpectedly found a completed generation"
                )
            return completed
        route_path = self._route_path(destination, generation_id)
        route_lock_path = route_path.with_suffix(route_path.suffix + ".lock")
        # A dataset root has one owning collector process, and this file lock
        # additionally serializes accidental duplicate invocations of the same
        # durable generation ID.  The second caller observes the first receipt
        # instead of submitting the idempotency key twice in parallel.
        with route_lock_path.open("a+b") as route_lock:
            fcntl.flock(route_lock.fileno(), fcntl.LOCK_EX)
            try:
                return self._invoke_locked(
                    payload=payload,
                    media_manifest=media_manifest,
                    generation_id=generation_id,
                    destination=destination,
                    receipt_path=receipt_path,
                    output_path=output_path,
                    route_path=route_path,
                    payload_hash=payload_hash,
                    resume=resume,
                    endpoint_binding=validated_binding,
                    persist_endpoint_binding=persist_endpoint_binding,
                    legacy_route_fallback=legacy_route_fallback,
                )
            finally:
                fcntl.flock(route_lock.fileno(), fcntl.LOCK_UN)

    def _invoke_locked(
        self,
        *,
        payload: Mapping[str, Any],
        media_manifest: Mapping[str, Any],
        generation_id: str,
        destination: Path,
        receipt_path: Path,
        output_path: Path,
        route_path: Path,
        payload_hash: str,
        resume: bool,
        endpoint_binding: Mapping[str, Any] | None,
        persist_endpoint_binding: Callable[[Mapping[str, Any]], None] | None,
        legacy_route_fallback: bool,
    ) -> EchoInvocationResult:
        completed = self._completed_result(
            receipt_path=receipt_path,
            output_path=output_path,
            generation_id=generation_id,
            request_payload_sha256=payload_hash,
        )
        if completed is not None:
            # Old completed runs have no route sidecar.  Their sealed receipt
            # remains sufficient and is deliberately not rewritten.
            return completed

        durable_binding = (
            None
            if endpoint_binding is None
            else _validated_echo_endpoint_binding(
                endpoint_binding,
                generation_id=generation_id,
                request_payload_sha256=payload_hash,
            )
        )
        if durable_binding is not None:
            preferred_index = self._route_endpoint_index(durable_binding)
            routing = str(durable_binding["routing"])
        elif resume and legacy_route_fallback:
            # This fallback is legal only when the caller has positively
            # identified a pre-binding checkpoint.  Absence of a route file is
            # never, by itself, evidence that a v4 reservation is legacy.
            preferred_index = self._endpoint_index(generation_id)
            routing = "legacy_generation_hash_recovery_v1"
        elif not resume:
            preferred_index = None
            routing = "idle_round_robin_v1"
        else:  # guarded in invoke(), retained as a fail-closed inner boundary
            raise CollectorError("v4 Echo replay has no sealed endpoint binding")

        endpoint_index = self._acquire_endpoint(preferred_index)
        client = self._clients[endpoint_index]
        try:
            if durable_binding is None:
                durable_binding = _build_echo_endpoint_binding(
                    generation_id=generation_id,
                    request_payload_sha256=payload_hash,
                    routing=routing,
                    endpoint_sha256=sha256_text(client.base_url),
                    endpoint_count_at_assignment=len(self._clients),
                    per_endpoint_concurrency_at_assignment=(
                        self._per_endpoint_capacity
                    ),
                )
                assert persist_endpoint_binding is not None
                # This callback atomically updates and fsyncs the generation's
                # host checkpoint.  No provider request is legal before it
                # returns successfully.
                persist_endpoint_binding(_json_copy(durable_binding))
            elif durable_binding["endpoint_sha256"] != sha256_text(client.base_url):
                raise CollectorError("Echo endpoint binding resolved to a different endpoint")

            route_matches = False
            if route_path.is_symlink():
                raise CollectorError("persisted Echo endpoint route is unsafe")
            if route_path.exists():
                try:
                    route_matches = (
                        self._read_route(
                            route_path,
                            generation_id=generation_id,
                            request_payload_sha256=payload_hash,
                        )
                        == durable_binding
                    )
                except CollectorError:
                    # The host checkpoint is the sealed source of truth.  A
                    # regular derivative sidecar can be rebuilt without route
                    # drift; an unsafe symlink was rejected above.
                    route_matches = False
            if not route_matches:
                _atomic_json(route_path, self._sealed_route(durable_binding))
            endpoint = f"{client.base_url}/r2v"
            response, submit_attempts = client._request(  # noqa: SLF001 - transport reuse
                "POST",
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Idempotency-Key": generation_id,
                    **client.extra_headers,
                },
                json_payload=payload,
            )
            accepted = self._response_json(response)
            final, poll_attempts = client._poll(accepted)  # noqa: SLF001
        finally:
            self._release_endpoint(endpoint_index)
        status = str(final.get("status") or "").lower()
        if status in _FAILURE_STATES:
            failure = {
                "schema_version": ECHO_WIRE_RECEIPT_VERSION,
                "generation_id": generation_id,
                "request_payload_sha256": payload_hash,
                "status": status,
                "resume": bool(resume),
                "endpoint_sha256": sha256_text(client.base_url),
                "media_inputs": media_manifest,
                "accepted": accepted,
                "final": final,
                "submit_attempts": submit_attempts,
                "poll_attempts": poll_attempts,
                "failed_at": _utc_now(),
            }
            _atomic_json(destination / f"{safe_name(generation_id)}.failed.json", failure)
            raise EchoTerminalFailure(
                "the one authorized Echo job failed; no alternate job ID was created"
            )
        if status not in _SUCCESS_STATES:
            raise ProviderProtocolError(f"Echo returned unknown terminal status {status!r}")
        result = final.get("result") if isinstance(final.get("result"), Mapping) else {}
        local_source = result.get("output_path") or final.get("output_path")
        video_url = (
            result.get("video_url")
            or final.get("video_url")
            or result.get("artifact_url")
            or final.get("artifact_url")
        )
        if local_source and Path(str(local_source)).expanduser().is_file():
            source = Path(str(local_source)).expanduser().resolve()
            temporary = output_path.with_name(f".{output_path.name}.tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, output_path)
            output_transport = "shared_local_path"
            download_attempts: list[dict[str, Any]] = []
        elif isinstance(video_url, str) and video_url:
            raw, download_attempts, _ = client._download_video(video_url)  # noqa: SLF001
            atomic_write_bytes(output_path, raw)
            output_transport = "download_url"
        else:
            raise ProviderProtocolError("successful Echo result has no accessible video")
        if not detect_mime(output_path).startswith("video/"):
            raise ProviderProtocolError("Echo output is not recognized as video")
        receipt = {
            "schema_version": ECHO_WIRE_RECEIPT_VERSION,
            "generation_id": generation_id,
            "request_payload_sha256": payload_hash,
            "prompt_sha256": sha256_text(str(payload["prompt"])),
            "seed_present": False,
            "status": status,
            "resume": bool(resume),
            "endpoint_sha256": sha256_text(client.base_url),
            "media_inputs": media_manifest,
            "accepted_response_sha256": sha256_json(accepted),
            "final_response_sha256": sha256_json(final),
            "submit_attempts": submit_attempts,
            "poll_attempts": poll_attempts,
            "download_attempts": download_attempts,
            "output_transport": output_transport,
            "output_sha256": sha256_file(output_path),
            "completed_at": _utc_now(),
        }
        _atomic_json(receipt_path, receipt)
        return EchoInvocationResult(
            output_path=str(output_path.resolve()),
            generation_id=generation_id,
            request_payload_sha256=payload_hash,
            output_sha256=receipt["output_sha256"],
            receipt=receipt,
        )


class FFmpegFinalEvidence:
    """Derive immutable first/tail frame evidence without a model call."""

    def derive(
        self,
        *,
        video_path: str | Path,
        output_dir: str | Path,
        generation_id: str,
        subject_ids: Sequence[str],
        semantic_roles: Sequence[str],
    ) -> list[dict[str, Any]]:
        source = require_file(video_path, label="accepted Echo video")
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        opening = destination / "opening.png"
        tail = destination / "tail.png"
        if not opening.is_file():
            extracted = extract_video_frames(source, destination / ".opening", timestamps=[0.0])
            opening.parent.mkdir(parents=True, exist_ok=True)
            os.replace(extracted[0], opening)
        if not tail.is_file():
            extract_video_tail_frame(source, tail)
        parent_hash = sha256_file(source)
        rows = []
        for role, path, time_selector in (
            ("opening", opening, {"mode": "frame", "time_ms": 0}),
            ("tail", tail, {"mode": "tail_frame"}),
        ):
            evidence_id = f"{role}_{_safe_id(generation_id, 'generation')}"
            representation_id = f"repr_{evidence_id}"
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "modality": "image",
                    "description": (
                        "Accepted opening frame of this shot."
                        if role == "opening"
                        else "Final frame of the immediately preceding accepted shot."
                    ),
                    "semantic_roles": sorted(set(semantic_roles) | {"continuity"}),
                    "subject_ids": list(dict.fromkeys(subject_ids)),
                    "representation": {
                        "representation_id": representation_id,
                        "modality": "image",
                        "mime_type": detect_mime(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "path": str(path.resolve()),
                    },
                    "parent_output_sha256": parent_hash,
                    "selector": time_selector,
                    "derivation": "deterministic_ffmpeg_frame_extract.v1",
                    # The collector deterministically carries the tail into the
                    # next shot as sequence_context.  Neither automatic frame
                    # should be duplicated in durable memory merely for
                    # adjacent-shot continuity.
                    "writeback_candidate": False,
                }
            )
        return rows


def _semantic_roles(asset_type: str) -> list[str]:
    value = str(asset_type or "").lower()
    if "voice" in value or "audio" in value:
        return ["voice"]
    if "character" in value or "person" in value:
        return ["identity", "appearance", "wardrobe"]
    if "scene" in value or "environment" in value:
        return ["scene", "environment"]
    if "object" in value or "prop" in value:
        return ["object_state"]
    if "style" in value:
        return ["style"]
    if "frame" in value or "image" in value:
        return ["visual_reference"]
    return ["generic_reference"]


def _normalized_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def _source_prompts(source: Mapping[str, Any]) -> list[str]:
    raw = source.get("prompts")
    if not isinstance(raw, list) or not raw:
        raw = source.get("shots")
    prompts = []
    for row in raw if isinstance(raw, list) else []:
        value = (
            row
            if isinstance(row, str)
            else row.get("raw_request") or row.get("prompt")
            if isinstance(row, Mapping)
            else ""
        )
        if isinstance(value, str) and value:
            prompts.append(value)
    if not prompts:
        raise CollectorError("story source contains no non-empty shot prompts")
    return prompts


def _build_registry(source: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    library = source.get("material_library") or {}
    if not isinstance(library, Mapping):
        raise CollectorError("material_library must be an object")
    registry: dict[str, Any] = {}
    tag_to_subject: dict[str, str] = {}
    for ordinal, (material_id, raw) in enumerate(sorted(library.items()), start=1):
        if not isinstance(raw, Mapping):
            raise CollectorError(f"material_library.{material_id} must be an object")
        source_tag = str(raw.get("tag") or material_id)
        subject_id = "subject_" + _safe_id(source_tag, f"material_{ordinal}").lower()
        aliases = [
            str(value)
            for value in [raw.get("name"), *(raw.get("aliases") or [])]
            if isinstance(value, str) and value.strip()
        ]
        item = {
            "subject_id": subject_id,
            "tag": source_tag,
            "category": str(raw.get("category") or raw.get("asset_type") or "material"),
            "description": str(raw.get("description") or raw.get("name") or source_tag),
            "stable_description": str(
                raw.get("description") or raw.get("name") or source_tag
            ),
            "aliases": list(dict.fromkeys(aliases)),
            "semantic_roles": _semantic_roles(str(raw.get("asset_type") or "")),
            **({"first_shot": int(raw["first_shot"])} if "first_shot" in raw else {}),
        }
        registry[subject_id] = item
        for tag in [source_tag, *(raw.get("tags") or [])]:
            if isinstance(tag, str) and tag:
                tag_to_subject[tag] = subject_id

    # Canonical sources normally contain material_library.  These structured
    # fallbacks preserve useful text-only registry entries for arbitrary user
    # story specs without generating or inspecting media.
    metadata = source.get("metadata") or {}
    candidates: list[tuple[str, Mapping[str, Any], str]] = []
    if isinstance(metadata, Mapping):
        for key in ("primary_character", "secondary_character", "prop", "style"):
            row = metadata.get(key)
            if isinstance(row, Mapping):
                candidates.append((key, row, key))
        for index, row in enumerate(metadata.get("scenes") or []):
            if isinstance(row, Mapping):
                candidates.append((f"scene_{index + 1}", row, "scene"))
    for ordinal, (key, raw, category) in enumerate(candidates, start=len(registry) + 1):
        source_tag = str(raw.get("tag") or key)
        if source_tag in tag_to_subject:
            continue
        subject_id = "subject_" + _safe_id(source_tag, f"material_{ordinal}").lower()
        name = str(raw.get("role") or raw.get("name") or source_tag)
        description = str(raw.get("description") or name)
        registry[subject_id] = {
            "subject_id": subject_id,
            "tag": source_tag,
            "category": category,
            "description": f"{name}: {description}" if description != name else name,
            "stable_description": description,
            "aliases": [name],
            "semantic_roles": _semantic_roles(category),
        }
        tag_to_subject[source_tag] = subject_id
    candidates_by_label: dict[str, set[str]] = {}
    for subject_id, item in registry.items():
        for label in [item.get("tag"), *(item.get("aliases") or [])]:
            normalized = _normalized_name(label)
            if normalized:
                candidates_by_label.setdefault(normalized, set()).add(subject_id)
    # Only unique normalized labels are safe automatic bindings. Ambiguous
    # human names remain visible in the registry and require explicit policy
    # disambiguation rather than silently choosing one subject.
    tag_to_subject = {
        label: next(iter(subject_ids))
        for label, subject_ids in candidates_by_label.items()
        if len(subject_ids) == 1
    }
    return registry, tag_to_subject


def _copy_source_asset(
    source_path: Path,
    row: Mapping[str, Any],
    *,
    cache_root: Path,
) -> Path:
    locator = next(
        (
            row.get(key)
            for key in ("path", "image_path", "video_path", "audio_path", "document_path")
            if row.get(key)
        ),
        None,
    )
    if locator is None and row.get("url"):
        downloaded = download_url(str(row["url"]), cache_root / "downloads")
        candidate = Path(downloaded["path"])
    elif locator is not None:
        candidate = Path(str(locator)).expanduser()
        if not candidate.is_absolute():
            candidate = source_path.parent / candidate
    else:
        raise CollectorError("uploaded material has no path or URL")
    candidate = require_file(candidate, label="uploaded source material")
    digest = sha256_file(candidate)
    declared = row.get("sha256") or row.get("content_sha256")
    if declared is not None:
        normalized = str(declared)
        normalized = normalized if normalized.startswith("sha256:") else "sha256:" + normalized
        if normalized != digest:
            raise CollectorError("uploaded source material differs from declared sha256")
    suffix = candidate.suffix.lower()
    if not suffix or len(suffix) > 12:
        suffix = mimetypes.guess_extension(detect_mime(candidate)) or ".bin"
    destination = cache_root / "objects" / f"{digest.removeprefix('sha256:')}{suffix}"
    if destination.exists():
        if sha256_file(destination) != digest:
            raise CollectorError(f"content-addressed source asset is corrupt: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copyfile(candidate, temporary)
        os.replace(temporary, destination)
    return destination.resolve()


def prepare_story_source(source_path: str | Path, *, story_dir: str | Path,
                         streaming_shot_index: int | None = None) -> dict[str, Any]:
    path = Path(source_path).expanduser().resolve()
    source = _read_object(path, label="story source")
    story_id = _safe_id(source.get("story_id") or path.stem, "story")
    prompts = _source_prompts(source)
    registry, tag_to_subject = _build_registry(source)
    schedule = source.get("uploaded_assets_by_shot") or {}
    streaming = source.get("streaming_asset_schema") is not None
    pack_hash = None
    if streaming:
        if source["streaming_asset_schema"] != "storymem.shot-driven-assets.v1":
            raise CollectorError("unsupported streaming asset schema")
        if schedule or Path(source.get("streaming_asset_root") or "").resolve() != path.parent:
            raise CollectorError("streaming source must use only its own immutable shot packs")
        registry = {identifier: item for identifier, item in registry.items()
                    if int(item.get("first_shot", 0)) <= (streaming_shot_index or 0)}
        if streaming_shot_index is not None:
            if not 0 <= streaming_shot_index < len(prompts):
                raise CollectorError("streaming shot index is outside the story")
            pack_path = path.parent / "shots" / f"{streaming_shot_index:04d}" / "upload_pack.json"
            pack = _read_object(pack_path, label="published shot upload pack")
            pack_hash = pack.get("pack_sha256")
            if (pack_hash != sha256_json({key: value for key, value in pack.items() if key != "pack_sha256"})
                    or pack.get("story_id") != story_id
                    or pack.get("schema_version") != source["streaming_asset_schema"]
                    or pack.get("shot_index") != streaming_shot_index
                    or pack.get("raw_prompt_sha256") != sha256_json(prompts[streaming_shot_index])):
                raise CollectorError("published upload pack lineage/hash mismatch")
            rows = []
            purposes = {"identity": "character_reference", "scene": "scene_plate", "object": "object_closeup", "style": "style_reference"}
            for asset in pack["assets"]:
                image = Path(asset["image_path"]).resolve()
                if not image.is_relative_to(path.parent) or asset.get("source_kind") != "new_seedream_simulated_upload":
                    raise CollectorError("streaming uploads must be newly generated within this story")
                if any(asset.get("assessment", {}).get(key) is not True for key in ("pass", "identity_preserved", "requirement_preserved", "usable_for_purpose")):
                    raise CollectorError("streaming upload did not pass independent image review")
                if any("subject_" + identifier not in registry for identifier in asset["entity_ids"]):
                    raise CollectorError("streaming upload refers to a future/unknown entity")
                rows.append({"asset_id": asset["asset_id"], "image_path": str(image), "sha256": asset["sha256"],
                             "tags": asset["entity_ids"], "user_description": asset["user_description"],
                             "asset_type": purposes.get(asset["purpose"], "image_reference")})
            schedule = {str(streaming_shot_index): rows}
    if not isinstance(schedule, Mapping):
        raise CollectorError("uploaded_assets_by_shot must be an object")
    cache_root = Path(story_dir).expanduser().resolve() / "source_assets"
    uploads: dict[int, dict[str, Any]] = {}
    for raw_index, rows in schedule.items():
        try:
            shot_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise CollectorError("uploaded asset schedule has a non-integer shot index") from exc
        if not 0 <= shot_index < len(prompts) or not isinstance(rows, list):
            raise CollectorError("uploaded asset schedule is outside the story")
        shot_uploads: dict[str, Any] = {}
        for ordinal, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise CollectorError("uploaded material entry must be an object")
            if "one_shot" in row and type(row.get("one_shot")) is not bool:
                raise CollectorError(
                    "uploaded material one_shot must be a boolean when present"
                )
            evidence_id = _safe_id(
                row.get("upload_id") or row.get("asset_id"),
                f"upload_s{shot_index + 1:03d}_{ordinal:03d}",
            )
            local = _copy_source_asset(path, row, cache_root=cache_root)
            mime_type = detect_mime(local)
            modality = mime_type.split("/", 1)[0]
            if modality not in {"image", "video", "audio", "text"}:
                if mime_type in {"application/json", "application/pdf", "application/xml"}:
                    modality = "text"
                else:
                    raise CollectorError(f"unsupported uploaded media type: {mime_type}")
            tags = [str(value) for value in row.get("tags") or [] if str(value)]
            subject_ids = list(
                dict.fromkeys(
                    tag_to_subject[normalized]
                    for tag in tags
                    for normalized in [_normalized_name(tag)]
                    if normalized in tag_to_subject
                )
            )
            asset_type = str(row.get("declared_asset_type") or row.get("asset_type") or modality)
            roles = _semantic_roles(asset_type)
            description = str(
                row.get("user_description") or row.get("description") or asset_type
            )
            speaker_binding = None
            speaker_name = None
            if modality == "audio":
                raw_binding = row.get("speaker_binding")
                speaker_name = str(
                    row.get("speaker_name")
                    or row.get("role_name")
                    or (raw_binding.get("declared_name") if isinstance(raw_binding, Mapping) else "")
                    or ""
                ).strip()
                if not speaker_name:
                    raise CollectorError(
                        f"audio upload {evidence_id} must carry an explicit character name"
                    )
                matched_subject = tag_to_subject.get(_normalized_name(speaker_name))
                matches = [matched_subject] if matched_subject is not None else []
                if len(matches) != 1:
                    raise CollectorError(
                        f"audio upload {evidence_id} character name must exactly match one registry entry"
                    )
                subject_ids = matches
                roles = ["voice"]
                speaker_binding = {
                    "basis": "trusted_user_declared",
                    "declared_name": speaker_name,
                    "subject_id": matches[0],
                    "matching_method": "exact_normalized_name",
                }
            representation_id = "repr_" + evidence_id
            representation = {
                "representation_id": representation_id,
                "modality": modality,
                "mime_type": mime_type,
                "size_bytes": local.stat().st_size,
                "sha256": sha256_file(local),
                "path": str(local),
            }
            selector_info = selector_catalog(local)
            if modality == "audio":
                full_ranges = [
                    selector
                    for selector in selector_info["selectors"]
                    if selector.get("mode") == "time_range"
                ]
                if len(full_ranges) != 1:
                    raise CollectorError(
                        f"audio upload {evidence_id} has no unique full-range selector"
                    )
                representation["selector"] = _json_copy(full_ranges[0])
            shot_uploads[evidence_id] = {
                "evidence_id": evidence_id,
                "source_kind": "uploaded_material",
                "modality": modality,
                "description": description,
                "declared_tags": tags,
                "semantic_roles": roles,
                "subject_ids": subject_ids,
                "speaker_name": speaker_name,
                "speaker_binding": speaker_binding,
                "trusted_stable_facts": [] if streaming else [description],
                # Selector ranges are derived from the immutable copied bytes,
                # never trusted from user metadata.  Audio writeback must name
                # one exact range from this catalog.
                "selectors": _json_copy(selector_info["selectors"]),
                "selector_limits": _json_copy(selector_info["limits"]),
                "default_representation_id": representation_id,
                "representations": {representation_id: representation},
            }
            if row.get("one_shot") is True:
                shot_uploads[evidence_id]["one_shot"] = True
        uploads[shot_index] = shot_uploads
    prepared = {
        "story_id": story_id,
        "title": str(source.get("title") or story_id),
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "prompts": prompts,
        "registry_items": registry,
        "uploads_by_shot": uploads,
        **({"streaming": True, "upload_pack_sha256": pack_hash} if streaming else {}),
    }
    prepared_name = f"prepared_shot_{streaming_shot_index:04d}.json" if streaming_shot_index is not None else "prepared_source.json"
    _atomic_json(cache_root / prepared_name, prepared)
    return prepared


def load_source_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = value.get("records") if isinstance(value, Mapping) else value
    if not isinstance(records, list) or not records:
        raise CollectorError("input manifest must contain a non-empty records array")
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(records):
        if isinstance(raw, str):
            row: Mapping[str, Any] = {"path": raw}
        elif isinstance(raw, Mapping):
            row = raw
        else:
            raise CollectorError(f"manifest record {index} is invalid")
        source = Path(str(row.get("path") or "")).expanduser()
        if not source.is_absolute():
            source = manifest_path.parent / source
        source = source.resolve()
        if not source.is_file():
            raise CollectorError(f"manifest source is missing: {source}")
        actual_hash = sha256_file(source)
        declared_hash = row.get("sha256")
        if declared_hash is not None and declared_hash != actual_hash:
            raise CollectorError(f"manifest source hash mismatch: {source}")
        source_value = _read_object(source, label="story source")
        story_id = _safe_id(row.get("story_id") or source_value.get("story_id") or source.stem, "story")
        if story_id in seen:
            raise CollectorError(f"duplicate story_id in input manifest: {story_id}")
        seen.add(story_id)
        resolved.append(
            {
                "index": index,
                "story_id": story_id,
                "path": str(source),
                "sha256": actual_hash,
                "expected_shots": len(_source_prompts(source_value)),
            }
        )
    return resolved


class StoryCollector:
    """Run one story sequentially while allowing independent stories in parallel."""

    def __init__(
        self,
        *,
        config: CollectorConfig,
        teacher_client: Any,
        visual_evidence_creator: Any,
        backend_adapter: Any,
        echo: EchoInvocationPort,
        finalizer: FinalEvidencePort | None = None,
        run_namespace: str | None = None,
    ) -> None:
        self.config = config
        self.teacher_client = teacher_client
        self.visual_evidence_creator = visual_evidence_creator
        self.backend_adapter = backend_adapter
        self.echo = echo
        self.finalizer = finalizer or FFmpegFinalEvidence()
        if run_namespace is None:
            active_lock = build_production_semantic_lock()
            run_path = config.output_root / "run.json"
            if run_path.is_file() and not run_path.is_symlink():
                persisted_run = _read_object(run_path, label="run manifest")
                run_namespace = _validated_run_namespace(
                    persisted_run,
                    output_root=config.output_root,
                    active_semantic_lock=active_lock,
                )
            else:
                raise CollectorError(
                    "StoryCollector requires an initialized run manifest or an "
                    "explicit test-only run namespace"
                )
        if not _RUN_NAMESPACE_RE.fullmatch(str(run_namespace)):
            raise CollectorError("run namespace is malformed")
        self.run_namespace = str(run_namespace)

    @staticmethod
    def _story_checkpoint_path(story_dir: Path) -> Path:
        return story_dir / "story_checkpoint.json"

    @staticmethod
    def _shot_host_path(shot_dir: Path) -> Path:
        return shot_dir / "host_checkpoint.json"

    def _create_or_load_story(
        self, prepared: Mapping[str, Any], *, story_dir: Path, pe_mode: str
    ) -> dict[str, Any]:
        path = self._story_checkpoint_path(story_dir)
        fingerprint = sha256_json(
            {
                "collector_version": COLLECTOR_VERSION,
                "config": self.config.semantic_manifest(),
                "source_sha256": prepared["source_sha256"],
                "pe_mode": pe_mode,
                "run_namespace": self.run_namespace,
            }
        )
        if path.is_file():
            value = _read_object(path, label="story checkpoint")
            if (
                value.get("schema_version") != STORY_CHECKPOINT_VERSION
                or value.get("story_id") != prepared["story_id"]
                or value.get("run_fingerprint") != fingerprint
                or value.get("run_namespace") != self.run_namespace
            ):
                raise CollectorError("story checkpoint conflicts with this run")
            return value
        value = {
            "schema_version": STORY_CHECKPOINT_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "story_id": prepared["story_id"],
            "run_namespace": self.run_namespace,
            "source_path": prepared["source_path"],
            "source_sha256": prepared["source_sha256"],
            "run_fingerprint": fingerprint,
            "pe_mode": pe_mode,
            "next_shot_index": 0,
            "registry_items": _json_copy(prepared["registry_items"]),
            "memory_bank": {},
            "previous_evidence": {},
            "shots": {},
            "status": "running",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        _atomic_json(path, value)
        return value

    def _save_story(self, story_dir: Path, checkpoint: dict[str, Any]) -> None:
        checkpoint["updated_at"] = _utc_now()
        _atomic_json(self._story_checkpoint_path(story_dir), checkpoint)

    def _create_or_load_host(
        self,
        *,
        shot_dir: Path,
        raw_prompt: str,
        pe_mode: str,
    ) -> dict[str, Any]:
        path = self._shot_host_path(shot_dir)
        raw_hash = sha256_text(raw_prompt)
        if path.is_file():
            value = _read_object(path, label="shot host checkpoint")
            if (
                value.get("schema_version") != SHOT_HOST_CHECKPOINT_VERSION
                or value.get("raw_request_sha256") != raw_hash
                or value.get("pe_mode") != pe_mode
            ):
                raise CollectorError("shot host checkpoint conflicts with this shot")
            return value
        value = {
            "schema_version": SHOT_HOST_CHECKPOINT_VERSION,
            "raw_request_sha256": raw_hash,
            "pe_mode": pe_mode,
            "pending_teacher": None,
            "hidden_rejections": [],
            "generations": {},
            "final_bundle": None,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        _atomic_json(path, value)
        return value

    def _save_host(self, shot_dir: Path, host: dict[str, Any]) -> None:
        host["updated_at"] = _utc_now()
        _atomic_json(self._shot_host_path(shot_dir), host)

    def _runtime(
        self,
        *,
        story_id: str,
        shot_index: int,
        shot_dir: Path,
        raw_prompt: str,
        registry_items: Mapping[str, Any],
        memory_bank: Mapping[str, Any],
        uploaded_evidence: Mapping[str, Any],
        previous_evidence: Mapping[str, Any],
    ) -> MemoryPolicyRuntime:
        state_path = shot_dir / "policy_state.json"
        session_id = f"{self.run_namespace}:{story_id}"
        request_id = f"{session_id}:shot:{shot_index:04d}"
        if state_path.is_file():
            state = MemoryPolicyState.load(state_path)
            if (
                state.data["shot_requirement"] != raw_prompt
                or state.data["session_id"] != session_id
                or state.data["request_id"] != request_id
                or state.data.get("source_story_id") != story_id
                or state.data.get("run_namespace") != self.run_namespace
            ):
                raise CollectorError("persisted policy state conflicts with story input")
        else:
            state = MemoryPolicyState.create(
                state_path,
                session_id=session_id,
                request_id=request_id,
                shot_requirement=raw_prompt,
                condition_budget={
                    # This identifier is model-visible through the registry and
                    # EvidencePlan, so it must describe only the portable policy
                    # contract rather than the currently selected backend.
                    "budget_id": "condition-budget-v1",
                    "max_total_conditions": self.config.max_total_conditions,
                    "max_image_conditions": self.config.max_image_conditions,
                    "max_video_conditions": 0,
                    "max_audio_conditions": self.config.max_audio_conditions,
                    "max_text_conditions": 0,
                },
                registry_items=registry_items,
                memory_bank=memory_bank,
                uploaded_evidence=uploaded_evidence,
                previous_evidence=previous_evidence,
                visual_creation_budget=self.config.visual_creation_budget,
                max_condition_revisions=self.config.max_condition_revisions,
                perceivable_modalities=self.config.perceivable_modalities,
                max_image_reference_inputs=self.config.max_image_reference_inputs,
                prompt_authoring=self.config.prompt_authoring,
            )
            state.data["source_story_id"] = story_id
            state.data["run_namespace"] = self.run_namespace
            # Inherited memory IDs must never be overwritten by a shot-local
            # counter restart.  The runtime still owns allocation thereafter.
            existing_ordinals = [
                int(match.group(1))
                for key in memory_bank
                for match in [re.fullmatch(r"memory_(\d+)", str(key))]
                if match
            ]
            if existing_ordinals:
                state.data["counters"]["memory"] = max(existing_ordinals)
            state.save()
        return MemoryPolicyRuntime(
            state=state,
            visual_evidence_creator=self.visual_evidence_creator,
            output_dir=shot_dir / "policy_runtime",
        )

    @staticmethod
    def _pending_is_committed(runtime: MemoryPolicyRuntime, pending: Mapping[str, Any]) -> bool:
        events = runtime.state.data.get("events") or []
        if not events:
            return False
        last = events[-1]
        return (
            last.get("tool_call") == pending.get("tool_call")
            and last.get("assistant_provenance") == pending.get("provenance")
        )

    def _teacher_turn(
        self,
        *,
        runtime: MemoryPolicyRuntime,
        host: dict[str, Any],
        shot_dir: Path,
    ) -> None:
        pending = host.get("pending_teacher")
        if isinstance(pending, Mapping) and self._pending_is_committed(runtime, pending):
            host["pending_teacher"] = None
            host["hidden_rejections"] = []
            self._save_host(shot_dir, host)
            return
        if not isinstance(pending, Mapping):
            allowed = runtime.allowed_tool_names()
            if not allowed:
                raise CollectorError(f"policy phase {runtime.phase!r} admits no teacher action")
            messages = build_teacher_input_messages(
                raw_user_request=runtime.state.data["shot_requirement"],
                events=runtime.state.data.get("events") or [],
                host_events=runtime.state.data.get("host_events") or [],
            )
            provider_messages, provider_media_manifest = _hydrate_teacher_media(
                messages,
                runtime.state.data.get("events") or [],
                perceivable_modalities=self.config.perceivable_modalities,
            )
            media_count = sum(
                1
                for message in messages
                for block in (
                    message.get("content")
                    if isinstance(message.get("content"), list)
                    else []
                )
                if isinstance(block, Mapping) and block.get("type") == "media"
            )
            if media_count > self.config.teacher_media_limit:
                raise CollectorError(
                    f"teacher history needs {media_count} media blocks, exceeding "
                    f"teacher_media_limit={self.config.teacher_media_limit}"
                )
            system_prompt = build_closed_teacher_system_prompt(
                allowed_tool_names=allowed,
                phase=runtime.phase,
                current_action_contract=_closed_teacher_action_contract(
                    runtime, allowed
                ),
                private_retry_feedback=host.get("hidden_rejections") or [],
                prompt_authoring=self.config.prompt_authoring,
            )
            try:
                result = self.teacher_client.call(
                    system_prompt=system_prompt,
                    user_content=provider_messages,
                    tools=get_tool_schemas(allowed, prompt_authoring=self.config.prompt_authoring),
                    temperature=self.config.tool_temperature,
                )
            except Exception as exc:
                feedback = list(host.get("hidden_rejections") or [])
                feedback.append(
                    {
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "instruction": (
                            "Return exactly one strict JSON {reason,tool_call} object "
                            "using an allowed schema."
                        ),
                    }
                )
                host["hidden_rejections"] = feedback[-self.config.max_hidden_turn_retries :]
                self._save_host(shot_dir, host)
                if len(feedback) >= self.config.max_hidden_turn_retries:
                    raise CollectorError("closed Teacher exhausted malformed-turn retries") from exc
                return
            if not isinstance(result, ToolCall) and not all(
                hasattr(result, name) for name in ("name", "arguments", "provenance")
            ):
                raise CollectorError("teacher client did not return a ToolCall")
            provenance = copy.deepcopy(dict(result.provenance))
            if provenance.get("assistant_source") not in {None, "remote_closed_model_api"}:
                raise CollectorError("teacher provenance is not a closed-model API response")
            provenance.update(
                {
                    "assistant_source": "remote_closed_model_api",
                    "dynamic_system_prompt_sha256": sha256_text(system_prompt),
                    "teacher_input_messages_sha256": sha256_json(messages),
                    "teacher_input_message_count": len(messages),
                    "teacher_provider_input_manifest_sha256": sha256_json(
                        {
                            "public_messages_sha256": sha256_json(messages),
                            "media": provider_media_manifest,
                        }
                    ),
                    "teacher_provider_media_manifest": provider_media_manifest,
                    "teacher_history_event_count": len(runtime.state.data.get("events") or []),
                    "private_retry_count": len(host.get("hidden_rejections") or []),
                    "teacher_allowed_tools": list(allowed),
                    "teacher_tool_schema_sha256": tool_schema_hash(allowed, prompt_authoring=self.config.prompt_authoring),
                    "phase": runtime.phase,
                    "teacher_reason": str(getattr(result, "reason", "")),
                }
            )
            pending = {
                "tool_call": {
                    "name": str(result.name),
                    "arguments": _json_copy(result.arguments),
                },
                "provenance": provenance,
            }
            host["pending_teacher"] = pending
            self._save_host(shot_dir, host)
        call = pending["tool_call"]
        observation = runtime.execute(
            str(call["name"]),
            call["arguments"],
            assistant_provenance=pending["provenance"],
        )
        if observation.get("ok") is not True:
            feedback = list(host.get("hidden_rejections") or [])
            feedback.append(
                {
                    "rejected_tool_call": call,
                    "error": {k: v for k, v in observation.items() if k != "_media_attachments"},
                    "instruction": "Correct the call; this rejected action is not training data.",
                }
            )
            host["pending_teacher"] = None
            host["hidden_rejections"] = feedback[-self.config.max_hidden_turn_retries :]
            self._save_host(shot_dir, host)
            if len(feedback) >= self.config.max_hidden_turn_retries:
                raise CollectorError("closed Teacher exhausted rejected-tool retries")
            return
        host["pending_teacher"] = None
        host["hidden_rejections"] = []
        self._save_host(shot_dir, host)

    @staticmethod
    def _compilation_request(
        compilation: Any, reservation: Mapping[str, Any]
    ) -> dict[str, Any]:
        if reservation.get("invoke_allowed") is True:
            return compilation.build_invocation_request(reservation)
        if reservation.get("replayed") is not True:
            raise CollectorError("generation reservation has invalid replay flags")
        # This is recovery of the already-authorized idempotent request, not a
        # second generation.  Its exact body is reconstructed from the sealed
        # adapter compilation and the runtime-returned global job ID.
        request = compilation.echo_request_template
        request["job_id"] = str(reservation["generation_id"])
        return request

    def _generation_step(
        self,
        *,
        runtime: MemoryPolicyRuntime,
        host: dict[str, Any],
        shot_dir: Path,
        story_id: str,
        shot_index: int,
        sequence_index: int,
        raw_prompt: str,
        pe_mode: str,
    ) -> None:
        request_spec = Echo15RequestSpec(
            work_id=str(runtime.state.data["session_id"]),
            shot_id=f"shot_{shot_index:04d}",
            sequence_index=sequence_index,
            num_frames=self.config.num_frames,
            width=self.config.width,
            height=self.config.height,
            duration_sec=self.config.duration_sec,
            seed=self._generation_seed(runtime) if getattr(self.config, 'prompt_authoring', False) else None,
        )
        try:
            compilation = self.backend_adapter.compile(
                state=runtime.state,
                raw_prompt=raw_prompt,
                request=request_spec,
                pe_mode=pe_mode,
            )
        except Echo15AdapterError as exc:
            reason_code = None
            if exc.code == "condition_not_compilable":
                candidate = exc.details.get("reason_code")
                if candidate in {
                    "condition.conflict",
                    "condition.unsupported",
                    "condition.capacity_limited",
                    "condition.invalid_evidence",
                    "condition.unavailable",
                }:
                    reason_code = str(candidate)
            elif exc.code == "evidence_unavailable":
                reason_code = "condition.unavailable"
            if reason_code is None:
                # Prompt/PE lineage, evidence integrity, stale state, and host
                # protocol defects are not policy choices and must stay fatal.
                raise
            plan = runtime.state.data.get("accepted_evidence_plan") or {}
            try:
                runtime.attach_compilation_feedback(
                    {
                        "request_id": runtime.state.data["request_id"],
                        "raw_request_sha256": runtime.state.data[
                            "shot_requirement_sha256"
                        ],
                        "plan_id": plan.get("plan_id"),
                        "plan_sha256": plan.get("plan_sha256"),
                        "binding_id": exc.binding_id,
                        "reason_code": reason_code,
                    }
                )
            except PolicyRejected as feedback_error:
                raise CollectorError(
                    "adapter could not compile the accepted condition plan and "
                    f"policy revision failed: {feedback_error.code}"
                ) from feedback_error
            return
        reservation = compilation.reserve(runtime)
        request = self._compilation_request(compilation, reservation)
        generation_id = str(reservation["generation_id"])
        reservation_private = runtime.state.data["generation_reservations"][
            reservation["generation_reservation_id"]
        ]
        generations = host.setdefault("generations", {})
        row = generations.get(generation_id)
        if row is None:
            row = {
                "generation_id": generation_id,
                "status": "reserved",
                "reservation": _json_copy(reservation),
                "invocation_request_sha256": sha256_json(request),
                "compilation": compilation.document,
                **({"base_prompt_artifact": compilation.base_prompt_artifact.to_dict()} if self.config.prompt_authoring else {}),
                "routing_contract": ECHO_ENDPOINT_BINDING_VERSION,
                "dispatch_state": "pre_endpoint_binding",
                "endpoint_binding": None,
                "created_at": _utc_now(),
            }
            generations[generation_id] = row
            self._save_host(shot_dir, host)
        elif row.get("invocation_request_sha256") != sha256_json(request):
            raise CollectorError("resumed Echo request differs from its host checkpoint")
        routing_contract = row.get("routing_contract")
        if routing_contract not in {
            ECHO_ENDPOINT_BINDING_VERSION,
            "legacy_generation_hash_recovery_v1",
        }:
            raise CollectorError("generation checkpoint has no explicit routing contract")
        dispatch_state = row.get("dispatch_state")
        if routing_contract == ECHO_ENDPOINT_BINDING_VERSION:
            if row.get("endpoint_binding") is None:
                if dispatch_state != "pre_endpoint_binding":
                    raise CollectorError(
                        "v4 generation checkpoint lost its durable endpoint binding"
                    )
            elif dispatch_state != "endpoint_binding_persisted":
                raise CollectorError(
                    "v4 generation checkpoint endpoint binding state is inconsistent"
                )

        def persist_endpoint_binding(value: Mapping[str, Any]) -> None:
            payload_hash = str(value.get("request_payload_sha256") or "")
            binding = _validated_echo_endpoint_binding(
                value,
                generation_id=generation_id,
                request_payload_sha256=payload_hash,
            )
            existing = row.get("endpoint_binding")
            if existing is not None and existing != binding:
                raise CollectorError(
                    "generation checkpoint conflicts with the selected Echo endpoint"
                )
            if existing is None:
                row["endpoint_binding"] = binding
                row["dispatch_state"] = "endpoint_binding_persisted"
                self._save_host(shot_dir, host)

        if row.get("status") != "completed":
            reservation_replay = reservation.get("invoke_allowed") is not True
            pre_binding_recovery = (
                routing_contract == ECHO_ENDPOINT_BINDING_VERSION
                and row.get("endpoint_binding") is None
                and dispatch_state == "pre_endpoint_binding"
            )
            try:
                result = self.echo.invoke(
                    request,
                    output_dir=shot_dir / "echo",
                    resume=reservation_replay and not pre_binding_recovery,
                    reserved_canonical_conditions=reservation_private[
                        "canonical_conditions"
                    ],
                    endpoint_binding=row.get("endpoint_binding"),
                    persist_endpoint_binding=persist_endpoint_binding,
                    legacy_route_fallback=(
                        routing_contract == "legacy_generation_hash_recovery_v1"
                    ),
                )
            except Exception as exc:
                row["status"] = "provider_pending_or_failed"
                row["last_error"] = {"type": type(exc).__name__, "message": str(exc)}
                self._save_host(shot_dir, host)
                raise
            if result.generation_id != generation_id:
                raise CollectorError("Echo transport changed the reserved generation ID")
            if routing_contract == ECHO_ENDPOINT_BINDING_VERSION:
                persisted_binding = row.get("endpoint_binding")
                if (
                    not isinstance(persisted_binding, Mapping)
                    or row.get("dispatch_state") != "endpoint_binding_persisted"
                ):
                    raise CollectorError(
                        "Echo transport returned before its endpoint binding "
                        "was durably persisted"
                    )
                _validated_echo_endpoint_binding(
                    persisted_binding,
                    generation_id=generation_id,
                    request_payload_sha256=result.request_payload_sha256,
                )
            row.update(
                {
                    "status": "completed",
                    "output_path": str(Path(result.output_path).resolve()),
                    "output_sha256": result.output_sha256,
                    "provider_receipt": _json_copy(result.receipt),
                    "completed_at": _utc_now(),
                }
            )
            self._save_host(shot_dir, host)
        video_path = require_file(row["output_path"], label="Echo candidate")
        if sha256_file(video_path) != row["output_sha256"]:
            raise CollectorError("Echo candidate bytes changed after completion")
        plan = runtime.state.data["accepted_evidence_plan"]
        bundle_id = "candidate_bundle_" + _safe_id(generation_id, "generation")
        candidate_id = "candidate_" + _safe_id(generation_id, "generation")
        bundle = {
            "bundle_kind": "candidate_feedback",
            "bundle_id": bundle_id,
            "request_id": runtime.state.data["request_id"],
            "raw_request_sha256": runtime.state.data["shot_requirement_sha256"],
            "plan_id": plan["plan_id"],
            "plan_sha256": plan["plan_sha256"],
            "generation_reservation_id": reservation["generation_reservation_id"],
            "base_prompt_artifact_id": reservation_private["base_prompt_artifact_id"],
            "base_prompt_artifact_sha256": reservation_private[
                "base_prompt_artifact_sha256"
            ],
            "wire_prompt_artifact_id": reservation_private["wire_prompt_artifact_id"],
            "wire_prompt_artifact_sha256": reservation_private[
                "wire_prompt_artifact_sha256"
            ],
            "compilation_receipt_id": reservation_private["compilation_receipt_id"],
            "compilation_receipt_sha256": reservation_private[
                "compilation_receipt_sha256"
            ],
            "condition_signature_sha256": reservation[
                "condition_signature_sha256"
            ],
            "generation_id": generation_id,
            "generation_output_sha256": row["output_sha256"],
            "review_target_evidence_id": candidate_id,
            "binding_outcomes": _json_copy(compilation.reservation_input["binding_outcomes"]),
            "derived_evidence": [
                {
                    "evidence_id": candidate_id,
                    "modality": "video",
                    "description": (
                        "Real generated video candidate for memory-condition "
                        "adherence review."
                    ),
                    "semantic_roles": sorted(
                        {
                            role
                            for binding in plan.get("ranked_bindings") or []
                            for role in binding.get("semantic_roles") or []
                        }
                    ),
                    "subject_ids": list(
                        dict.fromkeys(
                            subject
                            for binding in plan.get("ranked_bindings") or []
                            for subject in binding.get("subject_ids") or []
                        )
                    ),
                    "representation": {
                        "representation_id": "repr_" + candidate_id,
                        "modality": "video",
                        "mime_type": detect_mime(video_path),
                        "size_bytes": video_path.stat().st_size,
                        "sha256": row["output_sha256"],
                        "path": str(video_path.resolve()),
                    },
                    "writeback_candidate": False,
                }
            ],
        }
        row["candidate_bundle_id"] = bundle_id
        row["candidate_id"] = candidate_id
        row["plan_id"] = plan["plan_id"]
        row["plan_sha256"] = plan["plan_sha256"]
        row["lineage"] = {
            key: bundle[key]
            for key in (
                "generation_reservation_id",
                "base_prompt_artifact_id",
                "base_prompt_artifact_sha256",
                "wire_prompt_artifact_id",
                "wire_prompt_artifact_sha256",
                "compilation_receipt_id",
                "compilation_receipt_sha256",
                "condition_signature_sha256",
                "generation_id",
                "generation_output_sha256",
            )
        }
        self._save_host(shot_dir, host)
        runtime.attach_generation_evidence(bundle)

    def _final_evidence_step(
        self,
        *,
        runtime: MemoryPolicyRuntime,
        host: dict[str, Any],
        shot_dir: Path,
    ) -> None:
        selected = runtime.state.data.get("selected_generation") or {}
        generation_id = str(selected.get("generation_id") or "")
        candidate = (host.get("generations") or {}).get(generation_id)
        if not isinstance(candidate, Mapping) or candidate.get("status") != "completed":
            raise CollectorError("selected generation is absent from the host checkpoint")
        plan_id = str(selected["plan_id"])
        plans = [
            *(runtime.state.data.get("plan_history") or []),
            runtime.state.data.get("accepted_evidence_plan"),
        ]
        matches = [row for row in plans if isinstance(row, Mapping) and row.get("plan_id") == plan_id]
        if len(matches) != 1:
            raise CollectorError("selected generation plan lineage is ambiguous")
        plan = matches[0]
        subject_ids = list(
            dict.fromkeys(
                subject
                for binding in plan.get("ranked_bindings") or []
                for subject in binding.get("subject_ids") or []
            )
        )
        semantic_roles = sorted(
            {
                role
                for binding in plan.get("ranked_bindings") or []
                for role in binding.get("semantic_roles") or []
            }
        )
        if host.get("final_bundle") is None:
            derived = self.finalizer.derive(
                video_path=candidate["output_path"],
                output_dir=shot_dir / "final_evidence" / _safe_id(generation_id, "generation"),
                generation_id=generation_id,
                subject_ids=subject_ids,
                semantic_roles=semantic_roles,
            )
            lineage = candidate["lineage"]
            bundle = {
                "bundle_kind": "final_accepted",
                "bundle_id": "final_bundle_" + _safe_id(generation_id, "generation"),
                "request_id": runtime.state.data["request_id"],
                "raw_request_sha256": runtime.state.data["shot_requirement_sha256"],
                "plan_id": plan["plan_id"],
                "plan_sha256": plan["plan_sha256"],
                **_json_copy(lineage),
                "accepted_candidate_bundle_id": candidate["candidate_bundle_id"],
                "selection_disposition": (
                    "condition_pass"
                    if selected.get("decision") == "accept_conditions"
                    else "budget_best_effort"
                ),
                "finalization_receipt_id": "finalize_" + _safe_id(generation_id, "generation"),
                "derived_evidence": derived,
            }
            host["final_bundle"] = bundle
            self._save_host(shot_dir, host)
        runtime.attach_generation_evidence(host["final_bundle"])

    def _export_shot(
        self, *, runtime: MemoryPolicyRuntime, shot_dir: Path
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        trajectory = build_trajectory_shard(runtime_state=runtime.state)
        sft = build_sft_sample(trajectory, source_name=COLLECTOR_VERSION)
        _atomic_json(shot_dir / "trajectory.json", trajectory)
        _atomic_json(shot_dir / "sft.json", sft)
        return trajectory, sft

    @staticmethod
    def _story_result(
        prepared: Mapping[str, Any], checkpoint: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema_version": "storymem.memory-policy.story-result.v1",
            "story_id": prepared["story_id"],
            "run_namespace": checkpoint["run_namespace"],
            "status": "complete",
            "source_path": prepared["source_path"],
            "source_sha256": prepared["source_sha256"],
            "pe_mode": checkpoint["pe_mode"],
            "num_shots": len(prepared["prompts"]),
            "shots": _json_copy(checkpoint["shots"]),
            "elapsed_seconds": checkpoint.get("elapsed_seconds"),
            "completed_at": checkpoint["completed_at"],
        }

    @staticmethod
    def _generation_seed(runtime: MemoryPolicyRuntime) -> int:
        plans = [*(runtime.state.data.get("plan_history") or []), runtime.state.data.get("accepted_evidence_plan") or {}]
        resamples = sum(plan.get("attempt_kind") == "resample" for plan in plans)
        identity = f'{runtime.state.data["request_id"]}:resample:{resamples}'
        return int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16)

    def run_story(self, source_path: str | Path, *, max_shots: int | None = None) -> dict[str, Any]:
        source_value = _read_object(source_path, label="story source")
        story_id = _safe_id(source_value.get("story_id") or Path(source_path).stem, "story")
        story_dir = self.config.output_root / "stories" / story_id
        story_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_story_source(source_path, story_dir=story_dir)
        pe_mode = assign_pe_mode(story_id, rewrite_fraction=self.config.rewrite_fraction)
        checkpoint = self._create_or_load_story(prepared, story_dir=story_dir, pe_mode=pe_mode)
        if checkpoint.get("status") == "complete":
            complete_path = story_dir / "complete.json"
            if complete_path.is_file():
                return _read_object(complete_path, label="story result")
            result = self._story_result(prepared, checkpoint)
            _atomic_json(complete_path, result)
            return result

        started = time.time()
        completed_in_call = 0
        for shot_index, raw_prompt in enumerate(prepared["prompts"]):
            if shot_index < int(checkpoint["next_shot_index"]):
                continue
            if max_shots is not None and completed_in_call >= max_shots:
                return {"status": "partial", "story_id": story_id, "next_shot_index": shot_index}
            current_prepared = prepared
            if prepared.get("streaming"):
                pack_path = Path(source_path).resolve().parent / "shots" / f"{shot_index:04d}" / "upload_pack.json"
                if not pack_path.is_file():
                    return {"status": "waiting_assets", "story_id": story_id, "next_shot_index": shot_index}
                current_prepared = prepare_story_source(source_path, story_dir=story_dir, streaming_shot_index=shot_index)
                for identifier, item in current_prepared["registry_items"].items():
                    checkpoint["registry_items"].setdefault(identifier, _json_copy(item))
            shot_dir = story_dir / "shots" / f"shot_{shot_index:04d}"
            shot_dir.mkdir(parents=True, exist_ok=True)
            host = self._create_or_load_host(
                shot_dir=shot_dir,
                raw_prompt=raw_prompt,
                pe_mode=pe_mode,
            )
            if prepared.get("streaming"):
                if host.get("source_upload_pack_sha256") not in {None, current_prepared["upload_pack_sha256"]}:
                    raise CollectorError("shot upload pack changed after trajectory started")
                host["source_upload_pack_sha256"] = current_prepared["upload_pack_sha256"]
                self._save_host(shot_dir, host)
            runtime = self._runtime(
                story_id=story_id,
                shot_index=shot_index,
                shot_dir=shot_dir,
                raw_prompt=raw_prompt,
                registry_items=checkpoint["registry_items"],
                memory_bank=checkpoint["memory_bank"],
                uploaded_evidence=current_prepared["uploads_by_shot"].get(shot_index, {}),
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
                        pe_mode=pe_mode,
                    )
                elif runtime.phase == "awaiting_final_evidence":
                    self._final_evidence_step(runtime=runtime, host=host, shot_dir=shot_dir)
                else:
                    if (
                        len(runtime.state.data.get("events") or [])
                        >= self.config.max_turns_per_shot
                    ):
                        raise CollectorError(
                            f"shot {shot_index} exceeded "
                            f"{self.config.max_turns_per_shot} committed Teacher turns"
                        )
                    self._teacher_turn(runtime=runtime, host=host, shot_dir=shot_dir)

            if self.config.prompt_authoring:
                selected_id = str((runtime.state.data.get("selected_generation") or {}).get("generation_id") or "")
                selected_host = (host.get("generations") or {}).get(selected_id) or {}
                self.backend_adapter.commit_selected_prompt(
                    state=runtime.state, raw_prompt=raw_prompt,
                    request=Echo15RequestSpec(work_id=str(runtime.state.data["session_id"]),
                                              shot_id=f"shot_{shot_index:04d}", sequence_index=shot_index),
                    base_prompt_artifact=selected_host["base_prompt_artifact"],
                )
            trajectory, sft = self._export_shot(runtime=runtime, shot_dir=shot_dir)
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
            selected_id = str((runtime.state.data.get("selected_generation") or {}).get("generation_id") or "")
            selected_host = (host.get("generations") or {}).get(selected_id) or {}
            checkpoint["registry_items"] = _json_copy(runtime.state.data["registry"]["items"])
            checkpoint["memory_bank"] = _json_copy(runtime.state.data["memory_bank"])
            checkpoint["previous_evidence"] = previous_evidence
            checkpoint["shots"][str(shot_index)] = {
                "status": "complete",
                "request_id": runtime.state.data["request_id"],
                "trajectory_path": str((shot_dir / "trajectory.json").resolve()),
                "trajectory_sha256": sha256_json(trajectory),
                "sft_path": str((shot_dir / "sft.json").resolve()),
                "sft_sha256": sha256_json(sft),
                "selected_generation_id": selected_id,
                "output_path": selected_host.get("output_path"),
                "output_sha256": selected_host.get("output_sha256"),
            }
            checkpoint["next_shot_index"] = shot_index + 1
            self._save_story(story_dir, checkpoint)
            completed_in_call += 1

        if checkpoint.get("status") not in {"finalizing", "complete"}:
            checkpoint["status"] = "finalizing"
            checkpoint["elapsed_seconds"] = round(time.time() - started, 3)
            checkpoint["completed_at"] = _utc_now()
            self._save_story(story_dir, checkpoint)
        result = self._story_result(prepared, checkpoint)
        _atomic_json(story_dir / "complete.json", result)
        checkpoint["status"] = "complete"
        self._save_story(story_dir, checkpoint)
        return result


class BatchCollector:
    def __init__(
        self,
        *,
        story_collector: StoryCollector | None,
        config: CollectorConfig,
        executor_workers: int | None = None,
        max_in_flight: int | None = None,
        failure_fuse_min_outcomes: int = DEFAULT_FAILURE_FUSE_MIN_OUTCOMES,
        failure_fuse_rate: float = DEFAULT_FAILURE_FUSE_RATE,
        failure_fuse_max_failures: int = DEFAULT_FAILURE_FUSE_MAX_FAILURES,
        provider_profile: Mapping[str, Any] | None = None,
        echo_health_receipts: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.story_collector = story_collector
        self.config = config
        self.executor_workers = (
            config.story_workers if executor_workers is None else executor_workers
        )
        if type(self.executor_workers) is not int or self.executor_workers < 1:
            raise ValueError("executor_workers must be a positive integer")
        self.max_in_flight = (
            self.executor_workers if max_in_flight is None else max_in_flight
        )
        if type(self.max_in_flight) is not int or self.max_in_flight < 1:
            raise ValueError("max_in_flight must be a positive integer")
        if self.max_in_flight > self.executor_workers:
            raise ValueError("max_in_flight must not exceed executor_workers")
        if (
            type(failure_fuse_min_outcomes) is not int
            or failure_fuse_min_outcomes < 1
        ):
            raise ValueError("failure_fuse_min_outcomes must be a positive integer")
        if (
            isinstance(failure_fuse_rate, bool)
            or not isinstance(failure_fuse_rate, (int, float))
            or not 0 < float(failure_fuse_rate) <= 1
        ):
            raise ValueError("failure_fuse_rate must be in (0, 1]")
        if (
            type(failure_fuse_max_failures) is not int
            or failure_fuse_max_failures < 1
        ):
            raise ValueError("failure_fuse_max_failures must be a positive integer")
        self.failure_fuse_min_outcomes = failure_fuse_min_outcomes
        self.failure_fuse_rate = float(failure_fuse_rate)
        self.failure_fuse_max_failures = failure_fuse_max_failures
        self.echo_health_receipts = _json_copy(list(echo_health_receipts))
        raw_provider_profile = _json_copy(dict(provider_profile or {}))
        self.provider_profile = (
            bind_echo_capability_profile(
                raw_provider_profile, self.echo_health_receipts
            )
            if self.echo_health_receipts
            else raw_provider_profile
        )
        self._progress_lock = threading.Lock()

    def _initialize(
        self,
        source_manifest: Path,
        records: Sequence[Mapping[str, Any]],
        *,
        expected_run_incarnation_id: str | None = None,
    ) -> dict[str, Any]:
        root = self.config.output_root
        expected_incarnation = _prevalidate_expected_run_incarnation(
            root, expected_run_incarnation_id
        )
        root.mkdir(parents=True, exist_ok=True)
        launch_path = root / "run.json"
        source_rows = [_json_copy(dict(row)) for row in records]
        assignment = {
            row["story_id"]: assign_pe_mode(
                str(row["story_id"]), rewrite_fraction=self.config.rewrite_fraction
            )
            for row in source_rows
        }
        existing_run: dict[str, Any] | None = None
        if launch_path.exists() or launch_path.is_symlink():
            if not launch_path.is_file() or launch_path.is_symlink():
                raise CollectorError("existing run manifest path is unsafe")
            existing_run = _read_object(launch_path, label="run manifest")
        active_lock = build_production_semantic_lock()
        semantic_epoch = (
            semantic_epoch_for_run_manifest(existing_run)
            if existing_run is not None
            else 1
        )
        if existing_run is not None:
            run_incarnation_id = validate_run_incarnation_id(
                existing_run.get("run_incarnation_id")
            )
            if (
                expected_incarnation is not None
                and run_incarnation_id != expected_incarnation
            ):
                raise CollectorError(
                    "existing run incarnation differs from the expected launch identity"
                )
        else:
            run_incarnation_id = expected_incarnation or new_run_incarnation_id()
        run_namespace = run_namespace_for_semantic_epoch(
            root,
            semantic_epoch=semantic_epoch,
            semantic_lock_sha256=active_lock["semantic_lock_sha256"],
            run_incarnation_id=run_incarnation_id,
        )
        body = {
            "schema_version": RUN_MANIFEST_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "run_incarnation_id": run_incarnation_id,
            "run_namespace": run_namespace,
            "source_manifest_path": str(source_manifest.resolve()),
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_records_sha256": sha256_json(source_rows),
            "story_count": len(source_rows),
            "expected_shots": sum(int(row["expected_shots"]) for row in source_rows),
            "pe_assignment": assignment,
            "pe_assignment_sha256": sha256_json(assignment),
            "config": self.config.semantic_manifest(),
            "provider_profile": self.provider_profile,
            "collector_semantic_lock": active_lock,
        }
        body["run_sha256"] = sha256_json(body)
        if existing_run is not None:
            expected_body = {key: value for key, value in body.items() if key != "run_sha256"}
            body = validate_compatible_run_manifest(
                existing_run, expected_body
            )
        else:
            unexpected = [item for item in root.iterdir() if item.name != "run.json"]
            if unexpected:
                raise CollectorError("new output root is non-empty and has no matching run manifest")
            _atomic_json(launch_path, body)
        if self.echo_health_receipts:
            health_path = root / "preflight_history.json"
            if health_path.is_file():
                health_history = _read_object(
                    health_path, label="Echo preflight history"
                )
                if (
                    health_history.get("schema_version")
                    != PREFLIGHT_HISTORY_VERSION
                    or health_history.get("run_sha256") != body["run_sha256"]
                    or not isinstance(health_history.get("checks"), list)
                ):
                    raise CollectorError(
                        "Echo preflight history conflicts with this run"
                    )
            else:
                health_history = {
                    "schema_version": PREFLIGHT_HISTORY_VERSION,
                    "run_sha256": body["run_sha256"],
                    "checks": [],
                }
            receipts_sha256 = sha256_json(self.echo_health_receipts)
            checks = health_history["checks"]
            if not checks or checks[-1].get("receipts_sha256") != receipts_sha256:
                checks.append(
                    {
                        "sequence": len(checks) + 1,
                        "checked_at": _utc_now(),
                        "receipts": self.echo_health_receipts,
                        "receipts_sha256": receipts_sha256,
                    }
                )
                _atomic_json(health_path, health_history)
        progress_path = root / "progress.json"
        if not progress_path.exists():
            _atomic_json(
                progress_path,
                {
                    "schema_version": PROGRESS_SCHEMA_VERSION,
                    "run_sha256": body["run_sha256"],
                    "stories": {},
                    "updated_at": _utc_now(),
                },
            )
        return body

    def _validated_progress(
        self,
        run: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Load progress only after binding every terminal row to durable evidence."""

        path = self.config.output_root / "progress.json"
        value = _read_object(path, label="progress checkpoint")
        if value.get("schema_version") != PROGRESS_SCHEMA_VERSION:
            raise CollectorError("progress checkpoint schema is incompatible")
        if value.get("run_sha256") != run.get("run_sha256"):
            raise CollectorError("progress checkpoint belongs to another run")
        stories = value.get("stories")
        if not isinstance(stories, Mapping):
            raise CollectorError("progress checkpoint stories must be an object")
        records_by_id = {str(row["story_id"]): row for row in records}
        if len(records_by_id) != len(records):
            raise CollectorError("source records contain duplicate story IDs")
        unknown = sorted(set(stories) - set(records_by_id))
        if unknown:
            raise CollectorError(
                "progress checkpoint contains stories outside the active source: "
                + ", ".join(unknown[:5])
            )
        for story_id, raw in stories.items():
            if not isinstance(story_id, str) or not isinstance(raw, Mapping):
                raise CollectorError("progress checkpoint contains a malformed story row")
            if raw.get("story_id") != story_id:
                raise CollectorError(
                    f"progress story identity is inconsistent for {story_id}"
                )
            status = raw.get("status")
            if status not in {"complete", "failed"}:
                raise CollectorError(
                    f"progress story {story_id} has an unsupported status"
                )
            history = raw.get("failure_history")
            if history is not None and not isinstance(history, list):
                raise CollectorError(
                    f"progress story {story_id} has malformed failure history"
                )
            if status == "failed":
                if not all(
                    isinstance(raw.get(key), str) and bool(raw.get(key))
                    for key in ("error_type", "message", "failed_at")
                ):
                    raise CollectorError(
                        f"progress story {story_id} has malformed failure evidence"
                    )
                continue

            record = records_by_id[story_id]
            complete_path = (
                self.config.output_root / "stories" / story_id / "complete.json"
            )
            if not complete_path.is_file() or complete_path.is_symlink():
                raise CollectorError(
                    f"progress marks {story_id} complete without a safe complete artifact"
                )
            complete = _read_object(complete_path, label=f"story {story_id} result")
            progress_result = {
                key: item
                for key, item in raw.items()
                if key not in {"batch_attempt_count", "failure_history"}
            }
            if complete != progress_result:
                raise CollectorError(
                    f"progress and complete artifact disagree for {story_id}"
                )
            if (
                complete.get("schema_version")
                != "storymem.memory-policy.story-result.v1"
                or complete.get("story_id") != story_id
                or complete.get("run_namespace") != run.get("run_namespace")
                or complete.get("source_path") != str(Path(record["path"]).resolve())
                or complete.get("source_sha256") != record.get("sha256")
                or complete.get("num_shots") != record.get("expected_shots")
            ):
                raise CollectorError(
                    f"complete artifact lineage is invalid for {story_id}"
                )
        return value

    @staticmethod
    def _failure_history_entry(result: Mapping[str, Any]) -> dict[str, Any]:
        entry = {
            "error_type": str(result.get("error_type") or "UnknownError"),
            "message": str(result.get("message") or "story attempt failed"),
            "failed_at": str(result.get("failed_at") or _utc_now()),
        }
        audit = result.get("pe_failure_audit")
        if isinstance(audit, Mapping):
            # ``_pe_failure_audit`` is already a secret-free projection.  Keep
            # the complete prior ledger as well as its digest: a digest alone
            # cannot explain or reproduce an earlier failed attempt after a
            # later retry replaces the top-level failure row.
            projected = _json_copy(dict(audit))
            entry["pe_failure_audit"] = projected
            entry["pe_failure_audit_sha256"] = sha256_json(projected)
        return entry

    def _progress(self, run_sha256: str, story_id: str, result: Mapping[str, Any]) -> None:
        path = self.config.output_root / "progress.json"
        with self._progress_lock:
            value = _read_object(path, label="progress checkpoint")
            if (
                value.get("schema_version") != PROGRESS_SCHEMA_VERSION
                or value.get("run_sha256") != run_sha256
                or not isinstance(value.get("stories"), dict)
            ):
                raise CollectorError("progress checkpoint belongs to another run")
            existing = value["stories"].get(story_id)
            merged = _json_copy(dict(result))
            history: list[dict[str, Any]] = []
            if isinstance(existing, Mapping):
                previous_history = existing.get("failure_history")
                if isinstance(previous_history, list):
                    history = _json_copy(previous_history)
                elif existing.get("status") == "failed":
                    history = [self._failure_history_entry(existing)]
            if merged.get("status") == "failed":
                history.append(self._failure_history_entry(merged))
            if history:
                merged["failure_history"] = history
                merged["batch_attempt_count"] = len(history) + (
                    1 if merged.get("status") == "complete" else 0
                )
            value["stories"][story_id] = merged
            value["updated_at"] = _utc_now()
            _atomic_json(path, value)

    def _failure_fuse_reason(
        self, *, outcomes: int, failures: int
    ) -> str | None:
        if failures >= self.failure_fuse_max_failures:
            return "absolute_failure_limit"
        if (
            outcomes >= self.failure_fuse_min_outcomes
            and failures / outcomes >= self.failure_fuse_rate
        ):
            return "failure_rate_limit"
        return None

    def run(
        self,
        source_manifest: str | Path,
        *,
        max_stories: int | None = None,
        retry_failed: bool = False,
    ) -> dict[str, Any]:
        if self.story_collector is None:
            raise CollectorError("BatchCollector.run requires a StoryCollector")
        manifest_path = Path(source_manifest).expanduser().resolve()
        records = load_source_manifest(manifest_path)
        if max_stories is not None:
            if max_stories < 1:
                raise ValueError("max_stories must be positive")
            records = records[:max_stories]
        run = self._initialize(manifest_path, records)
        progress = self._validated_progress(run, records)
        completed = {
            story_id
            for story_id, row in (progress.get("stories") or {}).items()
            if isinstance(row, Mapping) and row.get("status") == "complete"
        }
        recorded_failed = {
            story_id
            for story_id, row in (progress.get("stories") or {}).items()
            if isinstance(row, Mapping) and row.get("status") == "failed"
        }
        never_recorded = [
            row
            for row in records
            if row["story_id"] not in completed
            and row["story_id"] not in recorded_failed
        ]
        failed_retries = [
            row
            for row in records
            if retry_failed and row["story_id"] in recorded_failed
        ]
        # Resume unfinished work before retrying terminal rows.  Otherwise a
        # persistent early failure cohort can repeatedly trip the fuse and
        # starve every never-attempted story that follows it in source order.
        pending = [*never_recorded, *failed_retries]
        futures: dict[Future[Any], Mapping[str, Any]] = {}
        next_pending = 0
        submitted = 0
        session_outcomes = 0
        session_failures = 0
        cancelled_before_start = 0
        fuse_trigger: dict[str, Any] | None = None
        active_limit = min(
            self.max_in_flight,
            self.failure_fuse_min_outcomes,
        )
        executor = ThreadPoolExecutor(max_workers=self.executor_workers)
        try:
            while next_pending < len(pending) and len(futures) < active_limit:
                row = pending[next_pending]
                next_pending += 1
                future = executor.submit(self.story_collector.run_story, row["path"])
                futures[future] = row
                submitted += 1

            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in sorted(
                    done,
                    key=lambda item: int(futures[item].get("index", 0)),
                ):
                    row = futures.pop(future)
                    if future.cancelled():
                        cancelled_before_start += 1
                        continue
                    try:
                        result = future.result()
                    except Exception as exc:  # independent stories stay isolated
                        pe_failure_audit = _pe_failure_audit(exc)
                        result = {
                            "story_id": row["story_id"],
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "message": (
                                "terminal backend PE failure; see pe_failure_audit"
                                if pe_failure_audit is not None
                                else str(exc)
                            ),
                            "failed_at": _utc_now(),
                        }
                        if pe_failure_audit is not None:
                            result["pe_failure_audit"] = pe_failure_audit
                    self._progress(run["run_sha256"], str(row["story_id"]), result)
                    session_outcomes += 1
                    if result.get("status") != "complete":
                        session_failures += 1
                reason = self._failure_fuse_reason(
                    outcomes=session_outcomes,
                    failures=session_failures,
                )
                if reason is not None and fuse_trigger is None:
                    fuse_trigger = {
                        "reason": reason,
                        "triggered_at": _utc_now(),
                        "outcomes": session_outcomes,
                        "failures": session_failures,
                        "failure_rate": session_failures / session_outcomes,
                    }
                    for queued in futures:
                        queued.cancel()

                if fuse_trigger is not None:
                    continue
                # Treat the first ``failure_fuse_min_outcomes`` submissions as
                # a closed startup cohort.  Do not replenish that cohort while
                # it is still running; otherwise fast systemic failures could
                # enqueue nearly twice the intended diagnostic sample before
                # the rate fuse has enough evidence to trip.  Configurations
                # whose maximum window is smaller advance in complete waves.
                if (
                    session_outcomes < self.failure_fuse_min_outcomes
                    and futures
                ):
                    continue
                if session_outcomes >= self.failure_fuse_min_outcomes:
                    active_limit = self.max_in_flight
                while next_pending < len(pending) and len(futures) < active_limit:
                    row = pending[next_pending]
                    next_pending += 1
                    future = executor.submit(
                        self.story_collector.run_story, row["path"]
                    )
                    futures[future] = row
                    submitted += 1
        finally:
            executor.shutdown(wait=True, cancel_futures=fuse_trigger is not None)

        progress = self._validated_progress(run, records)
        results = progress.get("stories") or {}
        complete = sum(
            1 for row in results.values() if isinstance(row, Mapping) and row.get("status") == "complete"
        )
        failed = sum(
            1 for row in results.values() if isinstance(row, Mapping) and row.get("status") == "failed"
        )
        manifest = {
            "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
            "run_sha256": run["run_sha256"],
            "source_story_count": len(records),
            "complete_stories": complete,
            "failed_stories": failed,
            "all_complete": complete == len(records) and failed == 0,
            "run_session": {
                "retry_failed": retry_failed,
                "eligible_stories": len(pending),
                "submitted_stories": submitted,
                "finished_outcomes": session_outcomes,
                "failed_outcomes": session_failures,
                "cancelled_before_start": cancelled_before_start,
                "deferred_recorded_failures": (
                    0 if retry_failed else len(recorded_failed)
                ),
            },
            "failure_fuse": {
                "schema_version": FAILURE_FUSE_SCHEMA_VERSION,
                "tripped": fuse_trigger is not None,
                "minimum_outcomes": self.failure_fuse_min_outcomes,
                "failure_rate_limit": self.failure_fuse_rate,
                "absolute_failure_limit": self.failure_fuse_max_failures,
                "startup_in_flight": min(
                    self.max_in_flight, self.failure_fuse_min_outcomes
                ),
                "maximum_in_flight": self.max_in_flight,
                "trigger": fuse_trigger,
            },
            "story_results": _json_copy(results),
            "updated_at": _utc_now(),
        }
        manifest["manifest_sha256"] = sha256_json(manifest)
        _atomic_json(self.config.output_root / "manifest.json", manifest)
        return manifest


def _production_output_root(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    allowed = Path(os.environ.get("REEL_OUTPUT_ROOT", str(Path(__file__).resolve().parents[1] / "output"))).expanduser().resolve()
    try:
        path.relative_to(allowed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"production output must be a new directory under {allowed}"
        ) from exc
    if path == allowed:
        raise argparse.ArgumentTypeError("output must not be the configured output root itself")
    return path


def _default_echo_urls() -> list[str]:
    return [f"http://127.0.0.1:{port}" for port in range(8321, 8329)]


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _load_api_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        if candidate == DEFAULT_API_CONFIG.resolve():
            return {}
        raise FileNotFoundError(f"API config does not exist: {candidate}")
    value = _read_object(candidate, label="API config")
    providers = value.get("providers")
    custom = providers.get("custom") if isinstance(providers, Mapping) else None
    if custom is not None and not isinstance(custom, Mapping):
        raise CollectorError("API config providers.custom must be an object")
    storymem = value.get("storymem")
    if storymem is not None and not isinstance(storymem, Mapping):
        raise CollectorError("API config storymem must be an object")
    return value


def _split_echo_urls(values: Sequence[str] | None) -> list[str]:
    raw = values or []
    urls = [
        item.rstrip("/")
        for value in raw
        for item in re.split(r"[,\s]+", str(value).strip())
        if item
    ]
    if not urls:
        return _default_echo_urls()
    if len(urls) != len(set(urls)):
        raise CollectorError("Echo endpoint list contains duplicates")
    return urls


def _resolve_provider_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve credentials in memory; callers must persist only ``public``."""

    config = _load_api_config(args.api_config)
    providers = config.get("providers")
    custom = providers.get("custom") if isinstance(providers, Mapping) else {}
    custom = custom if isinstance(custom, Mapping) else {}
    storymem = config.get("storymem")
    storymem = storymem if isinstance(storymem, Mapping) else {}
    base = str(custom.get("apiBase") or "").strip().rstrip("/")
    config_key = str(custom.get("apiKey") or "").strip() or None

    teacher_style = str(
        args.teacher_api_style
        or _first_env("RECAPTION_API_STYLE")
        or storymem.get("teacherApiStyle")
        or "gemini"
    ).strip()
    teacher_url = str(
        args.teacher_url
        or _first_env("RECAPTION_URL")
        or storymem.get("teacherUrl")
        or (f"{base}/responses" if base else "")
    ).strip()
    teacher_model = str(
        args.teacher_model
        or _first_env("RECAPTION_MODEL", "STORYMEM_TEACHER_MODEL")
        or storymem.get("teacherModel")
        or "Gemini-3.5-Flash-Third"
    ).strip()
    teacher_key = (
        _first_env(args.teacher_api_key_env, "API_KEY") or config_key
    )

    pe_url = str(
        args.pe_url
        or _first_env("STORYMEM_PE_URL")
        or storymem.get("peUrl")
        or (f"{base}/responses" if base else teacher_url)
    ).strip()
    pe_model = str(
        args.pe_model
        or _first_env("STORYMEM_PE_MODEL")
        or storymem.get("peModel")
        or teacher_model
    ).strip()
    pe_key = _first_env(args.pe_api_key_env, args.teacher_api_key_env, "API_KEY") or config_key

    image_adapter = str(
        args.image_adapter
        or _first_env("GEN_API_STYLE")
        or storymem.get("imageAdapter")
        or "seedream"
    ).strip()
    if image_adapter not in {"seedream", "openai_image", "gemini"}:
        raise CollectorError("image adapter must be seedream, openai_image, or gemini")
    image_route = {
        "seedream": "imageEdit/generations",
        "openai_image": "images/generations",
        "gemini": "images/gemini_flash/generations",
    }[image_adapter]
    image_url = str(
        args.image_url
        or _first_env("GEN_URL")
        or storymem.get("imageUrl")
        or (f"{base}/{image_route}" if base else "")
    ).strip()
    image_model = str(
        args.image_model
        or _first_env("GEN_MODEL", "IMAGE_GEN_MODEL", "SEEDREAM_MODEL")
        or storymem.get("imageModel")
        or "doubao-seedream-4-5-251128"
    ).strip()
    image_key = _first_env(args.image_api_key_env, "API_KEY") or config_key
    for label, value in (
        ("teacher URL", teacher_url),
        ("teacher model", teacher_model),
        ("PE URL", pe_url),
        ("PE model", pe_model),
        ("image URL", image_url),
        ("image model", image_model),
    ):
        if not value:
            raise CollectorError(f"{label} is required")
    echo_values = args.echo_urls
    if not echo_values:
        configured = _first_env(
            "ECHO15_R2V_SERVER_URLS",
            "ECHO15_R2V_SERVER_URL",
            "ECHO_R2V_SERVER_URL",
        )
        if configured:
            echo_values = [configured]
    echo_urls = _split_echo_urls(echo_values)
    public = {
        "teacher": {
            "endpoint_sha256": sha256_text(teacher_url.rstrip("/")),
            "model": teacher_model,
            "api_style": teacher_style,
            "transport": "plain_json_tool_dialogue",
        },
        "pe": {
            "endpoint_sha256": sha256_text(pe_url.rstrip("/")),
            "model": pe_model,
            "transport": "isolated_gemini_single_user_turn",
            "owner": "backend_adapter",
        },
        "image": {
            "endpoint_sha256": sha256_text(image_url.rstrip("/")),
            "model": image_model,
            "adapter": image_adapter,
        },
        "echo": {
            "endpoint_sha256": [sha256_text(url) for url in echo_urls],
            "endpoint_count": len(echo_urls),
            "audio_memory": True,
        },
    }
    return {
        "teacher_url": teacher_url,
        "teacher_model": teacher_model,
        "teacher_style": teacher_style,
        "teacher_key": teacher_key,
        "pe_url": pe_url,
        "pe_model": pe_model,
        "pe_key": pe_key,
        "image_url": image_url,
        "image_model": image_model,
        "image_adapter": image_adapter,
        "image_key": image_key,
        "echo_urls": echo_urls,
        "public": public,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect real Memory Policy trajectories through an Echo 1.5 adapter."
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=_production_output_root, required=True)
    parser.add_argument("--max-stories", type=int)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Explicitly retry stories already recorded as failed. Without this "
            "flag, resume processes only never-recorded or interrupted stories."
        ),
    )
    parser.add_argument("--expected-story-count", type=int)
    parser.add_argument("--expected-shot-count", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate provider configuration, source inventory, and every Echo "
            "endpoint without creating the output root or calling generation providers."
        ),
    )
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--story-workers", type=int, default=64)
    parser.add_argument(
        "--executor-workers",
        type=int,
        help=(
            "Operational thread-pool size. Defaults to --story-workers and is "
            "intentionally excluded from semantic run/checkpoint fingerprints."
        ),
    )
    parser.add_argument("--echo-concurrency", type=int, default=32)
    parser.add_argument("--image-concurrency", type=int, default=32)
    parser.add_argument("--teacher-qps", type=float, default=2.8)
    parser.add_argument("--rewrite-fraction", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=736)
    parser.add_argument("--num-frames", type=int, default=241)
    parser.add_argument("--teacher-url")
    parser.add_argument("--teacher-model")
    parser.add_argument(
        "--teacher-api-style", choices=("openai_chat", "gemini")
    )
    parser.add_argument("--teacher-api-key-env", default="RECAPTION_API_KEY")
    parser.add_argument("--pe-url")
    parser.add_argument("--pe-model")
    parser.add_argument("--pe-api-key-env", default="STORYMEM_PE_API_KEY")
    parser.add_argument("--image-url")
    parser.add_argument("--image-model")
    parser.add_argument(
        "--image-adapter", choices=("seedream", "openai_image", "gemini")
    )
    parser.add_argument("--image-api-key-env", default="GEN_API_KEY")
    parser.add_argument("--echo-url", action="append", dest="echo_urls")
    parser.add_argument(
        "--expected-run-incarnation-id",
        help=(
            "Optional launch-attempt identity. A new run adopts it exactly; an "
            "existing run must already contain the same sealed identity."
        ),
    )
    parser.add_argument(
        "--expected-provider-profile-sha256",
        help=(
            "Optional launch-attempt lock. Abort before provider construction or "
            "generation when the fresh Echo preflight yields another public profile."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_run_incarnation_id = _prevalidate_expected_run_incarnation(
        args.output_root, args.expected_run_incarnation_id
    )
    providers = _resolve_provider_configuration(args)
    config = CollectorConfig(
        output_root=args.output_root,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        story_workers=args.story_workers,
        echo_concurrency=args.echo_concurrency,
        image_concurrency=args.image_concurrency,
        teacher_qps=args.teacher_qps,
        rewrite_fraction=args.rewrite_fraction,
    )
    executor_workers = (
        config.story_workers if args.executor_workers is None else args.executor_workers
    )
    if type(executor_workers) is not int or executor_workers < 1:
        raise ValueError("executor_workers must be a positive integer")
    if args.max_stories is not None and args.max_stories < 1:
        raise ValueError("max_stories must be positive")
    if args.expected_story_count is not None and args.expected_story_count < 1:
        raise ValueError("expected_story_count must be positive")
    if args.expected_shot_count is not None and args.expected_shot_count < 1:
        raise ValueError("expected_shot_count must be positive")
    all_records = load_source_manifest(args.input_manifest)
    story_count = len(all_records)
    shot_count = sum(int(row["expected_shots"]) for row in all_records)
    if (
        args.expected_story_count is not None
        and story_count != args.expected_story_count
    ):
        raise CollectorError(
            f"source story count {story_count} != expected {args.expected_story_count}"
        )
    if args.expected_shot_count is not None and shot_count != args.expected_shot_count:
        raise CollectorError(
            f"source shot count {shot_count} != expected {args.expected_shot_count}"
        )
    selected_records = (
        all_records[: args.max_stories]
        if args.max_stories is not None
        else all_records
    )
    echo = Echo15WireTransport(
        providers["echo_urls"], concurrency=config.echo_concurrency
    )
    echo_health = echo.preflight(require_audio_memory=True)
    provider_profile = bind_echo_capability_profile(
        providers["public"], echo_health
    )
    provider_profile_sha256 = sha256_json(provider_profile)
    if args.expected_provider_profile_sha256 is not None:
        if not _SHA256_RE.fullmatch(args.expected_provider_profile_sha256):
            raise CollectorError("expected provider-profile SHA-256 is malformed")
        if args.expected_provider_profile_sha256 != provider_profile_sha256:
            raise CollectorError(
                "fresh provider capability profile differs from the launch attempt"
            )
    if args.validate_only:
        report = {
            "ok": True,
            "validation_mode": "read_only_no_output_root_no_generation",
            "source_manifest_sha256": sha256_file(args.input_manifest),
            "story_count": story_count,
            "shot_count": shot_count,
            "selected_story_count": len(selected_records),
            "selected_shot_count": sum(
                int(row["expected_shots"]) for row in selected_records
            ),
            "executor_workers": executor_workers,
            "provider_profile": provider_profile,
            "provider_profile_sha256": provider_profile_sha256,
            "collector_semantic_lock": build_production_semantic_lock(),
            "echo_health_receipts": echo_health,
            "output_root_created": args.output_root.exists(),
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    limiter = SharedRequestRateLimiter(config.teacher_qps)
    # These are separate clients and separate HTTP sessions.  Sharing only the
    # QPS limiter does not share dialogue history or request payload state.
    teacher = RemoteTeacherClient(
        providers["teacher_url"],
        api_key=providers["teacher_key"],
        model=providers["teacher_model"],
        api_style=providers["teacher_style"],
        rate_limiter=limiter,
        timeout=TEACHER_REQUEST_TIMEOUT_SECONDS,
        max_output_tokens=32768,
    )
    pe_client = IsolatedGeminiTextClient(
        providers["pe_url"],
        api_key=providers["pe_key"],
        model=providers["pe_model"],
        rate_limiter=limiter,
    )
    image_provider = GenericImageProvider(
        api_url=providers["image_url"],
        api_key=providers["image_key"],
        model=providers["image_model"],
        adapter=providers["image_adapter"],
    )
    visual_creator = ProviderVisualEvidenceCreator(
        image_provider,
        width=config.width,
        height=config.height,
        semaphore=threading.BoundedSemaphore(config.image_concurrency),
    )
    with _exclusive_output_root_lock(config.output_root):
        bootstrap = BatchCollector(
            story_collector=None,
            config=config,
            executor_workers=executor_workers,
            provider_profile=provider_profile,
            echo_health_receipts=echo_health,
        )
        run = bootstrap._initialize(
            args.input_manifest.resolve(),
            selected_records,
            expected_run_incarnation_id=expected_run_incarnation_id,
        )
        adapter_root = prepare_adapter_artifact_root(
            config.output_root,
            run,
            active_semantic_lock=run["collector_semantic_lock"],
        )
        adapter = Echo15BackendAdapter.from_official_v7(
            artifact_root=adapter_root,
            gemini=pe_client,
            capabilities=Echo15Capabilities(max_memory_slots=7, audio_memory=True),
        )
        story_collector = StoryCollector(
            config=config,
            teacher_client=teacher,
            visual_evidence_creator=visual_creator,
            backend_adapter=adapter,
            echo=echo,
            run_namespace=run["run_namespace"],
        )
        manifest = BatchCollector(
            story_collector=story_collector,
            config=config,
            executor_workers=executor_workers,
            provider_profile=provider_profile,
            echo_health_receipts=echo_health,
        ).run(
            args.input_manifest,
            max_stories=args.max_stories,
            retry_failed=args.retry_failed,
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["all_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BatchCollector",
    "COLLECTOR_VERSION",
    "RUN_NAMESPACE_VERSION",
    "RUN_INCARNATION_VERSION",
    "ADAPTER_EPOCH_MARKER_VERSION",
    "STORY_CHECKPOINT_VERSION",
    "SHOT_HOST_CHECKPOINT_VERSION",
    "ECHO_ENDPOINT_BINDING_VERSION",
    "ECHO_WIRE_ROUTE_VERSION",
    "CollectorConfig",
    "CollectorError",
    "Echo15WireTransport",
    "EchoInvocationResult",
    "FFmpegFinalEvidence",
    "ProviderVisualEvidenceCreator",
    "StoryCollector",
    "adapter_artifact_root",
    "assign_pe_mode",
    "bind_echo_capability_profile",
    "build_echo_capability_profile",
    "build_closed_teacher_system_prompt",
    "load_source_manifest",
    "prepare_story_source",
    "prepare_adapter_artifact_root",
    "new_run_incarnation_id",
    "run_namespace_for_semantic_epoch",
    "semantic_epoch_for_run_manifest",
    "validate_run_incarnation_id",
    "validate_compatible_run_manifest",
]
