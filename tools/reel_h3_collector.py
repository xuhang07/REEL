"""H3 host adapter using the unchanged REEL planner and evidence runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from reel_h3_adapter import H3PEError, H3R2AVPromptAdapter
from reel_h3_transport import H3RenderSettings, H3VideoClient, atomic_json, durable_lock, file_sha256, stage_request
from storymem_adapter.models import canonical_json, sha256_json, sha256_text
from storymem_policy_collector import CollectorError, StoryCollector, _safe_id, _utc_now


VERSION = "reel.h3.collector-adapter.v1"


@dataclass(frozen=True)
class H3CompiledGeneration:
    document_json: str

    @property
    def document(self) -> dict:
        return json.loads(self.document_json)

    @property
    def request(self) -> dict:
        return self.document["h3_request"]

    @property
    def reservation_input(self) -> dict:
        return self.document["reservation_input"]


class H3BackendAdapter:
    def __init__(
        self, *, artifact_root: Path, spool_root: Path, gemini_client: Any,
        settings: H3RenderSettings | None = None, max_references: int = 12,
    ) -> None:
        self.artifact_root = artifact_root.resolve()
        self.spool_root = spool_root.resolve()
        self.settings = settings or H3RenderSettings()
        self.settings.validate()
        self.pe = H3R2AVPromptAdapter(artifact_root=self.artifact_root, gemini_client=gemini_client,
                                    max_references=max_references)

    def compile(self, *, state: Any, raw_prompt: str, seed: int) -> H3CompiledGeneration:
        data = state if isinstance(state, Mapping) else state.data
        plan = data["accepted_evidence_plan"]
        pe = self.pe.compile(state=state, raw_prompt=raw_prompt)
        request, lineage = stage_request(pe, spool_root=self.spool_root, settings=replace(self.settings, seed=seed))
        base = {
            "schema_version": "reel.h3.planner-prompt.v1", "request_id": data["request_id"],
            "raw_prompt_sha256": data["shot_requirement_sha256"], "plan_sha256": plan["plan_sha256"],
            "prompt": plan["generation_prompt"], "prompt_sha256": sha256_text(plan["generation_prompt"]),
        }
        base_hash = sha256_json(base)
        wire = {"schema_version": "reel.h3.wire-prompt.v1", "prompt": request["prompt"],
                "pe_artifact_sha256": pe.document["artifact_sha256"], "references": lineage["references"]}
        wire_hash = sha256_json(wire)
        outcomes = [{"binding_id": reference["binding"]["binding_id"], "outcome": "applied_as_requested",
                     "applied_effect": reference["binding"]["intended_effect"], "reason_code": "condition.applied"}
                    for reference in lineage["references"]]
        conditions = [{"ordinal": index, "binding_ids": [reference["binding"]["binding_id"]],
                       "modality": "image", "content_path": reference["shared_path"],
                       "applied_effect": reference["binding"]["intended_effect"]}
                      for index, reference in enumerate(lineage["references"], 1)]
        receipt = {"schema_version": VERSION, "backend": "h3", "request_sha256": sha256_json(request),
                   "pe_artifact_sha256": pe.document["artifact_sha256"], "base_prompt_sha256": base_hash,
                   "wire_prompt_sha256": wire_hash, "binding_outcomes": outcomes, "compiled_conditions": conditions,
                   "temporal_roles_enforcement": lineage["temporal_roles_enforcement"]}
        receipt_hash = sha256_json(receipt)
        reservation = {
            "request_id": data["request_id"], "raw_request_sha256": data["shot_requirement_sha256"],
            "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"],
            "base_prompt_artifact_id": "h3_base_" + base_hash.removeprefix("sha256:"),
            "base_prompt_artifact_sha256": base_hash,
            "wire_prompt_artifact_id": "h3_wire_" + wire_hash.removeprefix("sha256:"),
            "wire_prompt_artifact_sha256": wire_hash,
            "compilation_receipt_id": "h3_compilation_" + receipt_hash.removeprefix("sha256:"),
            "compilation_receipt_sha256": receipt_hash,
            "binding_outcomes": outcomes, "compiled_conditions": conditions,
            "generator_prompt_sha256": sha256_text(request["prompt"]), "generator_seed": seed,
        }
        document = {"schema_version": VERSION, "backend": "h3", "h3_request": request, "lineage": lineage,
                    "base_prompt_artifact": base, "wire_prompt_artifact": wire, "compilation_receipt": receipt,
                    "reservation_input": reservation}
        path = self.artifact_root / "compilations" / (receipt_hash.removeprefix("sha256:") + ".json")
        with durable_lock(path.with_suffix(".lock")):
            if path.exists():
                if json.loads(path.read_text()) != document:
                    raise CollectorError("H3 compilation audit changed after it was sealed")
            else:
                atomic_json(path, document)
        return H3CompiledGeneration(canonical_json(document))

    def commit_selected_prompt(self, *, state: Any, request: Any, raw_prompt: str, base_prompt_artifact: Mapping[str, Any]) -> None:
        data = state if isinstance(state, Mapping) else state.data
        selected = data.get("selected_generation") or {}
        if data.get("phase") != "complete" or not selected.get("generation_id"):
            raise CollectorError("Only a finalized H3 candidate can be selected")
        base = dict(base_prompt_artifact)
        if base.get("raw_prompt_sha256") != sha256_text(raw_prompt) or base.get("request_id") != data["request_id"]:
            raise CollectorError("Selected H3 planner prompt belongs to a different request")
        matches = [row for row in data["generation_reservations"].values() if row["generation_id"] == selected["generation_id"]]
        if len(matches) != 1 or matches[0]["base_prompt_artifact_sha256"] != sha256_json(base):
            raise CollectorError("Selected H3 planner prompt does not match its actual generation reservation")
        audit = {"backend": "h3", "request_id": data["request_id"], "generation_id": selected["generation_id"],
                 "base_prompt_artifact": base, "work_id": request.work_id, "shot_id": request.shot_id}
        path = self.artifact_root / "selected" / (sha256_text(data["request_id"]).removeprefix("sha256:") + ".json")
        with durable_lock(path.with_suffix(".lock")):
            if path.exists() and json.loads(path.read_text()) != audit:
                raise CollectorError("Finalized H3 selection changed during resume")
            atomic_json(path, audit)


class H3StoryCollector(StoryCollector):
    def __init__(self, *, h3: H3VideoClient, **kwargs) -> None:
        super().__init__(echo=h3, **kwargs)
        self.h3 = h3

    def _generation_step(
        self, *, runtime, host: dict, shot_dir: Path, story_id: str, shot_index: int,
        sequence_index: int, raw_prompt: str, pe_mode: str,
    ) -> None:
        try:
            compilation = self.backend_adapter.compile(state=runtime.state, raw_prompt=raw_prompt,
                                                       seed=self._generation_seed(runtime))
        except H3PEError as error:
            reasons = {
                "reference.capacity": "condition.unsupported",
                "reference.modality": "condition.unsupported",
                "reference.effect": "condition.unsupported",
                "reference.opening_conflict": "condition.conflict",
                "reference.ending_conflict": "condition.conflict",
            }
            if error.code not in reasons:
                raise
            plan = runtime.state.data["accepted_evidence_plan"]
            runtime.attach_compilation_feedback({
                "request_id": runtime.state.data["request_id"],
                "raw_request_sha256": runtime.state.data["shot_requirement_sha256"],
                "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"],
                "binding_id": None, "reason_code": reasons[error.code],
            })
            return
        reservation = runtime.reserve_generation(compilation.reservation_input)
        generation_id = str(reservation["generation_id"])
        reserved = runtime.state.data["generation_reservations"][reservation["generation_reservation_id"]]
        request = compilation.request
        document = compilation.document
        generations = host.setdefault("generations", {})
        row = generations.get(generation_id)
        if row is None:
            row = {"generation_id": generation_id, "backend": "h3", "status": "reserved",
                   "reservation": reservation, "invocation_request_sha256": sha256_json(request),
                   "compilation": document, "base_prompt_artifact": document["base_prompt_artifact"],
                   "created_at": _utc_now()}
            generations[generation_id] = row
            self._save_host(shot_dir, host)
        elif row.get("backend") != "h3" or row.get("invocation_request_sha256") != sha256_json(request):
            raise CollectorError("Resumed H3 generation differs from its durable host checkpoint")
        if row["status"] != "completed":
            try:
                receipt = self.h3.generate(generation_key=generation_id, request=request, lineage=document["lineage"])
            except Exception as error:
                row.update(status="provider_pending_or_failed", last_error={"type": type(error).__name__,
                           "code": getattr(error, "code", None), "message": str(error)})
                self._save_host(shot_dir, host)
                raise
            if receipt["generation_key"] != generation_id or receipt["request_sha256"] != sha256_json(request):
                raise CollectorError("H3 result differs from the reserved generation input")
            row.update(status="completed", output_path=receipt["output_path"], output_sha256=receipt["output_sha256"],
                       h3_job_id=receipt["h3_job_id"], provider_receipt=receipt, completed_at=_utc_now())
            self._save_host(shot_dir, host)
        video_path = Path(row["output_path"])
        if not video_path.is_file() or file_sha256(video_path) != row["output_sha256"]:
            raise CollectorError("H3 candidate bytes changed after completion")
        plan = runtime.state.data["accepted_evidence_plan"]
        bundle_id = "candidate_bundle_" + _safe_id(generation_id, "generation")
        candidate_id = "candidate_" + _safe_id(generation_id, "generation")
        lineage_keys = (
            "base_prompt_artifact_id", "base_prompt_artifact_sha256", "wire_prompt_artifact_id",
            "wire_prompt_artifact_sha256", "compilation_receipt_id", "compilation_receipt_sha256",
        )
        bundle = {
            "bundle_kind": "candidate_feedback", "bundle_id": bundle_id,
            "request_id": runtime.state.data["request_id"], "raw_request_sha256": runtime.state.data["shot_requirement_sha256"],
            "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"],
            "generation_reservation_id": reservation["generation_reservation_id"],
            **{key: reserved[key] for key in lineage_keys},
            "condition_signature_sha256": reservation["condition_signature_sha256"], "generation_id": generation_id,
            "generation_output_sha256": row["output_sha256"], "review_target_evidence_id": candidate_id,
            "binding_outcomes": compilation.reservation_input["binding_outcomes"],
            "derived_evidence": [{
                "evidence_id": candidate_id, "modality": "video",
                "description": "Real generated video candidate for memory-condition adherence review.",
                "semantic_roles": sorted({role for binding in plan["ranked_bindings"] for role in binding["semantic_roles"]}),
                "subject_ids": list(dict.fromkeys(subject for binding in plan["ranked_bindings"] for subject in binding["subject_ids"])),
                "representation": {"representation_id": "repr_" + candidate_id, "modality": "video",
                                   "mime_type": "video/mp4", "size_bytes": video_path.stat().st_size,
                                   "sha256": row["output_sha256"], "path": str(video_path.resolve())},
                "writeback_candidate": False,
            }],
        }
        row.update(candidate_bundle_id=bundle_id, candidate_id=candidate_id, plan_id=plan["plan_id"], plan_sha256=plan["plan_sha256"])
        row["lineage"] = {key: bundle[key] for key in (
            "generation_reservation_id", *lineage_keys, "condition_signature_sha256", "generation_id", "generation_output_sha256",
        )}
        self._save_host(shot_dir, host)
        runtime.attach_generation_evidence(bundle)
