from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .util import atomic_write_json, read_json, sha256_json, sha256_text, utc_now


STATE_SCHEMA_VERSION = "storymem.canonical.state.v3"
STATE_INTEGRITY_SCHEMA_VERSION = "storymem.canonical.state-integrity.v1"
TEACHER_CALL_LEDGER_SCHEMA_VERSION = "storymem.canonical.teacher-call.v3"

_LOCATOR_KEYS = frozenset(
    {
        "url",
        "path",
        "image_path",
        "video_path",
        "audio_path",
        "document_path",
    }
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _is_locator_key(key: str) -> bool:
    normalized = key.casefold()
    return (
        normalized in _LOCATOR_KEYS
        or normalized.endswith("_url")
        or normalized.endswith("_path")
    )


def audit_safe_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the source metadata that may safely be persisted in state.json.

    Runtime asset ingestion has already copied the referenced bytes into a
    content-addressed story-local store.  The original locator is therefore
    neither needed for replay nor safe to retain: URLs may carry expiring
    signatures and local paths may disclose user filesystem layout.  Hashing
    every locator still lets an auditor distinguish and compare inputs without
    retaining the secret-bearing string.
    """

    def project(value: Any, *, key: str | None = None) -> Any:
        if key is not None and _is_locator_key(key) and value not in (None, ""):
            serialized = str(value)
            locator_kind = (
                "http_url"
                if serialized.casefold().startswith(("http://", "https://"))
                else "local_path"
            )
            return {
                "redacted": True,
                "locator_kind": locator_kind,
                "locator_sha256": sha256_text(serialized),
            }
        if isinstance(value, Mapping):
            return {
                str(child_key): project(child, key=str(child_key))
                for child_key, child in value.items()
                if str(child_key) != "_source_path"
            }
        if isinstance(value, list):
            return [project(item) for item in value]
        if isinstance(value, tuple):
            return [project(item) for item in value]
        return copy.deepcopy(value)

    return project(dict(source))


def _state_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if key != "integrity"
    }


def _seal_state(value: dict[str, Any]) -> None:
    # Uploaded assets gain deterministic derived representations after a view
    # selector is materialized, so this named digest tracks the current asset
    # table rather than only its launch-time contents.
    value["assets_sha256"] = sha256_json(value.get("assets") or {})
    payload = _state_payload(value)
    value["integrity"] = {
        "schema_version": STATE_INTEGRITY_SCHEMA_VERSION,
        "payload_sha256": sha256_json(payload),
    }


def _validate_state(value: Mapping[str, Any], *, source: Path) -> None:
    if value.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(f"incompatible StoryMem state: {source}")
    integrity = value.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("schema_version") != STATE_INTEGRITY_SCHEMA_VERSION
        or not isinstance(integrity.get("payload_sha256"), str)
        or not _DIGEST_RE.fullmatch(str(integrity.get("payload_sha256")))
        or integrity.get("payload_sha256") != sha256_json(_state_payload(value))
    ):
        raise ValueError(f"StoryMem state integrity check failed: {source}")

    required_objects = (
        "source",
        "story_plan",
        "run_manifest",
        "assets",
        "registry",
        "memory_bank",
        "id_counters",
    )
    if any(not isinstance(value.get(key), Mapping) for key in required_objects):
        raise ValueError(f"StoryMem state has an invalid object field: {source}")
    if not isinstance(value.get("committed_shots"), list):
        raise ValueError(f"StoryMem state committed_shots is invalid: {source}")
    if value.get("current_shot") is not None and not isinstance(
        value.get("current_shot"), Mapping
    ):
        raise ValueError(f"StoryMem state current_shot is invalid: {source}")
    if type(value.get("event_sequence")) is not int or int(
        value.get("event_sequence", -1)
    ) < 0:
        raise ValueError(f"StoryMem state event_sequence is invalid: {source}")
    if not isinstance(value.get("story_id"), str) or not str(
        value.get("story_id")
    ).strip():
        raise ValueError(f"StoryMem state story_id is invalid: {source}")

    source_audit = value.get("source")
    if value.get("source_audit_sha256") != sha256_json(source_audit):
        raise ValueError(f"StoryMem state source audit hash mismatch: {source}")

    def validate_safe_locators(item: Any, *, key: str | None = None) -> None:
        if key == "_source_path":
            raise ValueError(f"StoryMem state retained the source path: {source}")
        if key is not None and _is_locator_key(key):
            if item in (None, ""):
                return
            if (
                not isinstance(item, Mapping)
                or item.get("redacted") is not True
                or item.get("locator_kind") not in {"http_url", "local_path"}
                or not isinstance(item.get("locator_sha256"), str)
                or not _DIGEST_RE.fullmatch(str(item.get("locator_sha256")))
            ):
                raise ValueError(
                    f"StoryMem state retained an unsafe source locator: {source}"
                )
            return
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                validate_safe_locators(child, key=str(child_key))
        elif isinstance(item, list):
            for child in item:
                validate_safe_locators(child)

    validate_safe_locators(source_audit)
    if value.get("story_plan_sha256") != sha256_json(value.get("story_plan")):
        raise ValueError(f"StoryMem state story plan hash mismatch: {source}")
    if value.get("assets_sha256") != sha256_json(value.get("assets")):
        raise ValueError(f"StoryMem state assets hash mismatch: {source}")

    # The outer state seal catches every byte.  These named closures make the
    # reason for a damaged semantic lock/config explicit rather than silently
    # accepting a self-consistent but internally contradictory checkpoint.
    run_manifest = value.get("run_manifest") or {}
    semantic_config = run_manifest.get("semantic_config")
    if isinstance(semantic_config, Mapping) and run_manifest.get(
        "semantic_config_sha256"
    ) != sha256_json(semantic_config):
        raise ValueError(f"StoryMem state semantic config hash mismatch: {source}")
    semantic_lock = run_manifest.get("collector_semantic_lock")
    if isinstance(semantic_lock, Mapping):
        lock_body = {
            key: item
            for key, item in semantic_lock.items()
            if key != "semantic_lock_sha256"
        }
        lock_hash = semantic_lock.get("semantic_lock_sha256")
        if (
            lock_hash != sha256_json(lock_body)
            or run_manifest.get("collector_semantic_lock_sha256") != lock_hash
        ):
            raise RuntimeError(
                f"collector_semantic_lock changed or failed its hash closure: {source}"
            )

    seen_teacher_calls: set[str] = set()
    seen_event_ids: set[str] = set()
    shots = [
        *(
            row
            for row in value.get("committed_shots") or []
            if isinstance(row, Mapping)
        ),
        *(
            [value["current_shot"]]
            if isinstance(value.get("current_shot"), Mapping)
            else []
        ),
    ]
    for shot in shots:
        events = shot.get("events") or []
        teacher_calls = shot.get("teacher_calls") or []
        if not isinstance(events, list) or not isinstance(teacher_calls, list):
            raise ValueError(f"StoryMem state shot ledger is invalid: {source}")
        local_events: dict[str, Mapping[str, Any]] = {}
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError(f"StoryMem state event is invalid: {source}")
            event_id = str(event.get("event_id") or "")
            if not re.fullmatch(r"event_[0-9]{8}", event_id):
                raise ValueError(f"StoryMem state event ID is invalid: {source}")
            if event_id in seen_event_ids:
                raise ValueError(f"StoryMem state has duplicate event IDs: {source}")
            seen_event_ids.add(event_id)
            local_events[event_id] = event
        for record in teacher_calls:
            if not isinstance(record, Mapping):
                raise ValueError(f"StoryMem state teacher call is invalid: {source}")
            _validate_teacher_call_record(record)
            audit_id = str(record["teacher_call_audit_id"])
            if audit_id in seen_teacher_calls:
                raise ValueError(
                    f"StoryMem state has duplicate teacher call audit IDs: {source}"
                )
            seen_teacher_calls.add(audit_id)
            event_id = record.get("event_id")
            if record.get("execution_status") == "committed":
                event = local_events.get(str(event_id))
                if (
                    event is None
                    or event.get("tool_call") != record.get("tool_call")
                    or (event.get("assistant_provenance") or {}).get(
                        "teacher_call_audit_id"
                    )
                    != audit_id
                ):
                    raise ValueError(
                        f"StoryMem state teacher/event linkage is invalid: {source}"
                    )
        received = [
            record
            for record in teacher_calls
            if record.get("execution_status") == "received"
        ]
        if len(received) > 1:
            raise ValueError(
                f"StoryMem state has multiple pending Teacher commands: {source}"
            )
        if received:
            if shot is not value.get("current_shot") or teacher_calls[-1] is not received[0]:
                raise ValueError(
                    f"StoryMem state has a non-tail pending Teacher command: {source}"
                )


def _validate_teacher_call_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != TEACHER_CALL_LEDGER_SCHEMA_VERSION:
        raise ValueError("teacher call ledger has an incompatible schema")
    audit_id = str(record.get("teacher_call_audit_id") or "")
    if not re.fullmatch(r"teachercall_[0-9]{6}", audit_id):
        raise ValueError("teacher call ledger has an invalid audit ID")
    response_status = record.get("response_status")
    execution_status = record.get("execution_status")
    if response_status not in {"plain_json_tool_command", "malformed_response"}:
        raise ValueError("teacher call ledger has an invalid response status")
    if execution_status not in {"not_applicable", "received", "committed", "rejected"}:
        raise ValueError("teacher call ledger has an invalid execution status")
    tool_call = record.get("tool_call")
    if response_status == "plain_json_tool_command":
        if (
            not isinstance(tool_call, Mapping)
            or not isinstance(tool_call.get("name"), str)
            or not isinstance(tool_call.get("arguments"), Mapping)
        ):
            raise ValueError("plain-JSON teacher command has no exact tool call")
        if execution_status == "not_applicable":
            raise ValueError("plain-JSON teacher command cannot be marked not_applicable")
    elif tool_call is not None or execution_status != "not_applicable":
        raise ValueError("malformed teacher response must not contain a parsed tool call")
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("teacher call ledger entry has no API provenance")
    if execution_status == "committed" and not isinstance(record.get("event_id"), str):
        raise ValueError("committed teacher call ledger entry has no event ID")
    if execution_status == "rejected" and not isinstance(
        record.get("runtime_rejection"), Mapping
    ):
        raise ValueError("rejected teacher call ledger entry has no runtime rejection")


@dataclass
class StateStore:
    path: Path
    data: dict[str, Any]

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        story_id: str,
        source: dict[str, Any],
        story_plan: dict[str, Any],
        run_manifest: dict[str, Any],
        assets: dict[str, Any] | None = None,
    ) -> "StateStore":
        destination = Path(path)
        # Provider idempotency keys must be globally unique, not merely unique
        # inside one story state. Persist the random run component before any
        # provider request is opened so crash-resume reconstructs the same key.
        # The story digest keeps logs attributable without putting an arbitrary
        # (and potentially very long) user story ID on the wire.
        run_instance_id = uuid.uuid4().hex
        story_digest = sha256_text(story_id).removeprefix("sha256:")[:12]
        source_audit = audit_safe_source(source)
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "story_id": story_id,
            "run_instance_id": run_instance_id,
            "provider_request_namespace": f"storymem-{run_instance_id}-{story_digest}",
            "source": source_audit,
            "source_audit_sha256": sha256_json(source_audit),
            "story_plan": copy.deepcopy(story_plan),
            "story_plan_sha256": sha256_json(story_plan),
            "run_manifest": copy.deepcopy(run_manifest),
            "assets": copy.deepcopy(assets or {}),
            "assets_sha256": sha256_json(assets or {}),
            "registry": {"version": 0, "items": {}},
            "memory_bank": {},
            "committed_shots": [],
            "current_shot": None,
            "id_counters": {},
            "event_sequence": 0,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        store = cls(destination, state)
        store.save()
        return store

    @classmethod
    def load(cls, path: str | Path) -> "StateStore":
        source = Path(path)
        value = read_json(source)
        if not isinstance(value, dict):
            raise ValueError(f"incompatible StoryMem state: {source}")
        _validate_state(value, source=source)
        return cls(source, value)

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        _seal_state(self.data)
        atomic_write_json(self.path, self.data)

    def next_id(self, prefix: str) -> str:
        counters = self.data.setdefault("id_counters", {})
        value = int(counters.get(prefix, 0)) + 1
        counters[prefix] = value
        return f"{prefix}_{value:06d}"

    def next_event_id(self) -> str:
        self.data["event_sequence"] = int(self.data.get("event_sequence", 0)) + 1
        return f"event_{self.data['event_sequence']:08d}"

    @property
    def shot(self) -> dict[str, Any]:
        value = self.data.get("current_shot")
        if not isinstance(value, dict):
            raise RuntimeError("no active shot")
        return value

    def start_shot(
        self,
        *,
        shot_index: int,
        raw_user_request: str,
        uploads: list[str],
        budgets: dict[str, int],
    ) -> dict[str, Any]:
        if self.data.get("current_shot") is not None:
            raise RuntimeError("cannot start a new shot before committing the active shot")
        request_id = self.next_id("shotreq")
        shot = {
            "shot_index": int(shot_index),
            "shot_request_id": request_id,
            "raw_user_request": raw_user_request,
            "raw_user_prompt_sha256": sha256_text(raw_user_request),
            "upload_ids": list(uploads),
            "budgets": {key: int(value) for key, value in budgets.items()},
            "phase": "need_registry",
            "observations": {},
            "observation_order": [],
            "latest_search_id": None,
            "searches": {},
            "view_authorizations": {},
            "pe_receipts": {},
            "pending_pe_receipt_id": None,
            "prompt_artifacts": {},
            "pending_prompt_artifact_id": None,
            "generation_groups": {},
            "candidates": {},
            "inspections": {},
            "assessments": {},
            "selections": {},
            "consumed_image_selection_ids": [],
            "staged_memory_update": None,
            "events": [],
            "teacher_calls": [],
            "started_at": utc_now(),
        }
        self.data["current_shot"] = shot
        self.save()
        return shot

    def record_teacher_call(
        self,
        *,
        response_status: str,
        provenance: Mapping[str, Any],
        tool_call: Mapping[str, Any] | None = None,
        teacher_reason: str = "",
        error: Mapping[str, Any] | None = None,
    ) -> str:
        """Persist one paid closed-model HTTP turn outside training messages."""

        audit_id = self.next_id("teachercall")
        record = {
            "schema_version": TEACHER_CALL_LEDGER_SCHEMA_VERSION,
            "teacher_call_audit_id": audit_id,
            "sequence": len(self.shot.setdefault("teacher_calls", [])) + 1,
            "response_status": str(response_status),
            "execution_status": (
                "received"
                if response_status == "plain_json_tool_command"
                else "not_applicable"
            ),
            "tool_call": copy.deepcopy(dict(tool_call)) if tool_call is not None else None,
            "teacher_reason": str(teacher_reason),
            "provenance": copy.deepcopy(dict(provenance)),
            "error": copy.deepcopy(dict(error)) if error is not None else None,
            "event_id": None,
            "runtime_rejection": None,
            "created_at": utc_now(),
            "finalized_at": (
                utc_now() if response_status == "malformed_response" else None
            ),
        }
        _validate_teacher_call_record(record)
        self.shot["teacher_calls"].append(record)
        self.save()
        return audit_id

    def _teacher_call_record(self, audit_id: str) -> dict[str, Any]:
        matches = [
            row
            for row in self.shot.get("teacher_calls") or []
            if isinstance(row, dict)
            and row.get("teacher_call_audit_id") == audit_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"teacher call audit ID must resolve exactly once: {audit_id}"
            )
        return matches[0]

    def pending_teacher_call(self) -> dict[str, Any] | None:
        """Return the one durable received command that still needs execution.

        A paid Teacher response is saved before its deterministic tool runs.
        If the process dies in that narrow window, replaying this exact command
        avoids both an extra API call and an orphaned ``received`` audit row.
        """

        if not isinstance(self.data.get("current_shot"), Mapping):
            return None
        records = self.shot.get("teacher_calls") or []
        pending = [
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("execution_status") == "received"
        ]
        if not pending:
            return None
        if len(pending) != 1 or records[-1] is not pending[0]:
            raise RuntimeError("Teacher command ledger has an ambiguous pending command")
        record = pending[0]
        _validate_teacher_call_record(record)
        call = record.get("tool_call")
        provenance = record.get("provenance")
        if not isinstance(call, Mapping) or not isinstance(provenance, Mapping):
            raise RuntimeError("pending Teacher command is incomplete")
        replay_provenance = copy.deepcopy(dict(provenance))
        replay_provenance["teacher_call_audit_id"] = str(
            record["teacher_call_audit_id"]
        )
        return {
            "tool_call": copy.deepcopy(dict(call)),
            "assistant_provenance": replay_provenance,
        }

    def reject_teacher_call(
        self, audit_id: str, *, rejection: Mapping[str, Any]
    ) -> None:
        record = self._teacher_call_record(audit_id)
        if record.get("response_status") != "plain_json_tool_command":
            raise RuntimeError("only a parsed plain-JSON tool command can be runtime-rejected")
        if record.get("execution_status") not in {"received", "rejected"}:
            raise RuntimeError("teacher call was already committed")
        normalized = copy.deepcopy(dict(rejection))
        if record.get("execution_status") == "rejected":
            if record.get("runtime_rejection") != normalized:
                raise RuntimeError("teacher call rejection changed after it was recorded")
            return
        record["execution_status"] = "rejected"
        record["runtime_rejection"] = normalized
        record["finalized_at"] = utc_now()
        _validate_teacher_call_record(record)
        self.save()

    def _link_teacher_call_to_event(
        self,
        *,
        tool_call: Mapping[str, Any],
        event_id: str,
        assistant_provenance: Mapping[str, Any],
        committed: bool,
        observation: Mapping[str, Any],
    ) -> None:
        audit_id = assistant_provenance.get("teacher_call_audit_id")
        if audit_id is None:
            # Unit tests and explicitly non-collector uses may execute the
            # deterministic runtime directly. Canonical shard validation still
            # requires real remote provenance and a linked teacher ledger.
            return
        if not isinstance(audit_id, str):
            raise RuntimeError("teacher_call_audit_id must be a string")
        record = self._teacher_call_record(audit_id)
        if record.get("response_status") != "plain_json_tool_command":
            raise RuntimeError("event points to a malformed teacher response")
        if record.get("tool_call") != dict(tool_call):
            raise RuntimeError("event tool call differs from its teacher audit record")
        target_status = "committed" if committed else "rejected"
        current_status = record.get("execution_status")
        if current_status not in {"received", target_status}:
            raise RuntimeError(
                f"teacher call {audit_id} already finalized as {current_status}"
            )
        record["execution_status"] = target_status
        record["event_id"] = event_id if committed else None
        record["runtime_rejection"] = (
            None if committed else copy.deepcopy(dict(observation))
        )
        record["finalized_at"] = utc_now()
        _validate_teacher_call_record(record)

    def add_observation(self, kind: str, payload: dict[str, Any]) -> str:
        observation_id = self.next_id("obs")
        record = {
            "observation_id": observation_id,
            "kind": kind,
            "sequence": len(self.shot["observation_order"]) + 1,
            "payload": copy.deepcopy(payload),
        }
        self.shot["observations"][observation_id] = record
        self.shot["observation_order"].append(observation_id)
        return observation_id

    def record_event(
        self,
        *,
        tool_call: dict[str, Any],
        observation: dict[str, Any],
        assistant_provenance: dict[str, Any] | None = None,
        committed: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        event = {
            "event_id": self.next_event_id(),
            "tool_call": copy.deepcopy(tool_call),
            "observation": copy.deepcopy(observation),
            "assistant_provenance": copy.deepcopy(assistant_provenance or {}),
            "committed_to_training": bool(committed),
            "created_at": utc_now(),
        }
        self.shot["events"].append(event)
        self._link_teacher_call_to_event(
            tool_call=tool_call,
            event_id=event["event_id"],
            assistant_provenance=assistant_provenance or {},
            committed=bool(committed),
            observation=observation,
        )
        if save:
            self.save()
        return event

    def commit_shot(self, committed: dict[str, Any]) -> None:
        # Build and persist the complete next state in one atomic file replace.
        # Keeping ``self.data`` unchanged until that succeeds also makes a
        # caller-level retry safe when serialization or storage fails.
        next_data = copy.deepcopy(self.data)
        shot = next_data.get("current_shot")
        if not isinstance(shot, dict):
            raise RuntimeError("no active shot")
        shot["phase"] = "committed"
        shot["committed_at"] = utc_now()
        shot["commit"] = copy.deepcopy(committed)
        next_data["committed_shots"].append(copy.deepcopy(shot))
        next_data["current_shot"] = None
        next_data["updated_at"] = utc_now()
        _seal_state(next_data)
        atomic_write_json(self.path, next_data)
        self.data = next_data


__all__ = [
    "STATE_INTEGRITY_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "TEACHER_CALL_LEDGER_SCHEMA_VERSION",
    "StateStore",
    "audit_safe_source",
]
