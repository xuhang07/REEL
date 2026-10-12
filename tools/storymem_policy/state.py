"""Durable state for the two-stage provider-neutral Memory Policy FSM."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


STATE_SCHEMA_VERSION = "storymem.memory-policy.state.v2"
POLICY_PHASES = frozenset(
    {
        "need_registry",
        "planning",
        "awaiting_generation",
        "generation_review",
        "awaiting_final_evidence",
        "writeback",
        "complete",
    }
)


class StateConflictError(RuntimeError):
    """Raised when a stale in-memory state attempts to replace newer bytes."""


def _serialized_state(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _payload_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Persist the directory entry as well as the file contents.  The state
        # is the authorization ledger for external generation calls.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class MemoryPolicyState:
    """Small JSON-backed store with explicit phase transitions and counters."""

    def __init__(
        self,
        path: str | Path,
        data: Mapping[str, Any],
        *,
        persisted_sha256: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.data = copy.deepcopy(dict(data))
        self._persisted_sha256 = persisted_sha256
        self._validate()

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        session_id: str,
        request_id: str,
        shot_requirement: str,
        condition_budget: Mapping[str, Any],
        registry_items: Mapping[str, Any] | None = None,
        memory_bank: Mapping[str, Any] | None = None,
        uploaded_evidence: Mapping[str, Any] | None = None,
        previous_evidence: Mapping[str, Any] | None = None,
        visual_creation_budget: int = 2,
        max_condition_revisions: int = 1,
        perceivable_modalities: tuple[str, ...] = ("image", "video", "text"),
        max_image_reference_inputs: int = 4,
        prompt_authoring: bool = False,
    ) -> "MemoryPolicyState":
        if type(visual_creation_budget) is not int or visual_creation_budget < 0:
            raise ValueError("visual_creation_budget must be a non-negative integer")
        if type(max_condition_revisions) is not int or max_condition_revisions < 0:
            raise ValueError("max_condition_revisions must be a non-negative integer")
        if type(max_image_reference_inputs) is not int or max_image_reference_inputs < 0:
            raise ValueError("max_image_reference_inputs must be a non-negative integer")
        perception = tuple(perceivable_modalities)
        if len(set(perception)) != len(perception) or set(perception) - {
            "image",
            "video",
            "audio",
            "text",
        }:
            raise ValueError("perceivable_modalities contains an invalid modality")
        data = {
            "schema_version": STATE_SCHEMA_VERSION,
            "session_id": str(session_id),
            "request_id": str(request_id),
            "shot_requirement": str(shot_requirement),
            "shot_requirement_sha256": "sha256:"
            + hashlib.sha256(str(shot_requirement).encode("utf-8")).hexdigest(),
            "condition_budget": copy.deepcopy(dict(condition_budget)),
            "policy_limits": {
                "max_condition_revisions": max_condition_revisions,
                "perceivable_modalities": list(perception),
                "max_image_reference_inputs": max_image_reference_inputs,
                **({"prompt_authoring": True} if prompt_authoring else {}),
            },
            "phase": "need_registry",
            "registry": {
                "version": 1,
                "items": copy.deepcopy(dict(registry_items or {})),
            },
            "memory_bank": copy.deepcopy(dict(memory_bank or {})),
            "uploaded_evidence": copy.deepcopy(dict(uploaded_evidence or {})),
            "previous_evidence": copy.deepcopy(dict(previous_evidence or {})),
            "created_evidence": {},
            "generation_evidence": {},
            "searches": {},
            "views": {},
            "creation_groups": {},
            "inspections": {},
            "assessments": {},
            "accepted_evidence_plan": None,
            "plan_history": [],
            "condition_review": None,
            "condition_review_target_id": None,
            "max_condition_revisions": max_condition_revisions,
            "revision_count": 0,
            "generation_bundle_private": None,
            "generation_bundle_history_private": [],
            "base_prompt_artifact_lock": None,
            "generation_reservations": {},
            "active_generation_reservation_id": None,
            "used_condition_signatures": [],
            "reviewed_generations": [],
            "selected_generation": None,
            "staged_transaction": None,
            "pending_candidate_id": None,
            "pending_assessment_id": None,
            "pending_retry_group_id": None,
            "inflight": None,
            "events": [],
            "host_events": [],
            "compilation_feedback_history_private": [],
            "visual_creation_budget": int(visual_creation_budget),
            "visual_creation_count": 0,
            "counters": {},
        }
        state = cls(path, data, persisted_sha256=None)
        state.save()
        return state

    @classmethod
    def load(cls, path: str | Path) -> "MemoryPolicyState":
        resolved = Path(path).expanduser().resolve()
        payload = resolved.read_bytes()
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("memory policy state is not valid UTF-8 JSON") from exc
        return cls(resolved, value, persisted_sha256=_payload_sha256(payload))

    @property
    def persisted_sha256(self) -> str | None:
        """Return the digest used by the compare-and-swap persistence guard."""

        return self._persisted_sha256

    def reload(self) -> None:
        """Replace both data and CAS metadata with the durable state."""

        refreshed = type(self).load(self.path)
        self.data = refreshed.data
        self._persisted_sha256 = refreshed._persisted_sha256

    def _validate(self) -> None:
        if self.data.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("memory policy state schema is incompatible")
        if self.data.get("phase") not in POLICY_PHASES:
            raise ValueError("memory policy state has an invalid phase")
        for key in ("session_id", "request_id", "shot_requirement"):
            if not isinstance(self.data.get(key), str) or not self.data[key]:
                raise ValueError(f"memory policy state has invalid {key}")
        expected_requirement_sha256 = "sha256:" + hashlib.sha256(
            self.data["shot_requirement"].encode("utf-8")
        ).hexdigest()
        if self.data.get("shot_requirement_sha256") != expected_requirement_sha256:
            raise ValueError("memory policy state raw request hash mismatch")
        if not isinstance(self.data.get("condition_budget"), dict):
            raise ValueError("memory policy state has no condition budget")
        if not isinstance(self.data.get("policy_limits"), dict):
            raise ValueError("memory policy state has no policy limits")
        if not isinstance(self.data.get("used_condition_signatures"), list):
            raise ValueError("memory policy state condition signatures must be an array")
        if not isinstance(self.data.get("generation_reservations"), dict):
            raise ValueError("memory policy state generation reservations must be an object")
        active_reservation = self.data.get("active_generation_reservation_id")
        if active_reservation is not None and active_reservation not in self.data[
            "generation_reservations"
        ]:
            raise ValueError("memory policy state active generation reservation is unknown")
        if not isinstance(self.data.get("reviewed_generations"), list):
            raise ValueError("memory policy state reviewed generations must be an array")
        if not isinstance(
            self.data.get("compilation_feedback_history_private"), list
        ):
            raise ValueError(
                "memory policy state compilation feedback history must be an array"
            )
        if not isinstance(self.data.get("counters"), dict):
            raise ValueError("memory policy state counters must be an object")

    @property
    def phase(self) -> str:
        return str(self.data["phase"])

    def transition(self, expected: str, target: str) -> None:
        if self.phase != expected:
            raise RuntimeError(
                f"invalid memory policy transition {self.phase!r}; expected {expected!r}"
            )
        if target not in POLICY_PHASES:
            raise ValueError(f"unknown memory policy phase: {target}")
        self.data["phase"] = target

    def next_id(self, prefix: str) -> str:
        counters = self.data["counters"]
        counters[prefix] = int(counters.get(prefix, 0)) + 1
        return f"{prefix}_{counters[prefix]:06d}"

    def save(self) -> None:
        self._validate()
        payload = _serialized_state(self.data)
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                current_sha256 = (
                    _payload_sha256(self.path.read_bytes())
                    if self.path.exists()
                    else None
                )
                if current_sha256 != self._persisted_sha256:
                    raise StateConflictError(
                        "memory policy state changed since it was loaded; refusing stale write"
                    )
                _atomic_write_json(self.path, payload)
                self._persisted_sha256 = _payload_sha256(payload)
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "MemoryPolicyState",
    "POLICY_PHASES",
    "STATE_SCHEMA_VERSION",
    "StateConflictError",
]
