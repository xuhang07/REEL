"""Independent, image-grounded H3 PE; no H3 service transport is assumed."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from storymem_adapter.echo15 import (
    Echo15BackendAdapter,
    Echo15Capabilities,
    _Echo15ConditionCompiler,
)
from storymem_adapter.gemini import GeminiInlineImage, GeminiTextResult
from storymem_adapter.models import (
    PromptEngineeringProfile,
    PromptOutputContract,
    canonical_json,
    sha256_json,
    sha256_text,
)
from storymem_adapter.prompt_engineering import (
    GeminiTextPort,
    _atomic_write_json,
    _exclusive_lock,
)
from storymem_policy.models import EvidenceBinding


VERSION = "reel.h3.r2av-pe.v1"
MAX_REFERENCES = 12
SKILL_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts/h3-r2av-multiref-pe/SKILL.md"
)
HEADINGS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
RETENTION_MODES = frozenset(
    {"fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"}
)
GROUNDING_BASES = frozenset(
    {"visible_reference", "request_only", "requested_change", "uncertain"}
)
LABEL_RE = re.compile(r"<(Image|Subject) (\d+)>")
EDITORIAL_RE = re.compile(
    r"\bshot\s*\d+\b|\[\s*shot\b[^\]]*\]|^\s*shot\s*[:#]"
    r"|\bF\d{2,}\b|\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b"
    r"|\b(?![12]\d{2}0s\b)\d+(?:\.\d+)?\s*(?:ms|s|secs?|seconds?|mins?|minutes?)\b",
    re.IGNORECASE | re.MULTILINE,
)
LITERAL_RE = re.compile(r"<d>.*?</d>|\"[^\"\n]*\"|“[^”\n]*”", re.DOTALL)
PRIVATE_RE = re.compile(
    r"(?:/pfs2?/|/Users/|/root/|file://|https?://|sha256:|storymem_)"
    r"|\b(?:ID_[A-Z][A-Z0-9_]*|ENTITY_\d+|SCENE_\d+|OBJECT_\d+)\b"
)


class H3PEError(ValueError):
    def __init__(self, code: str, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = json.loads(canonical_json(details or {}))


def load_h3_profile(path: Path = SKILL_PATH) -> PromptEngineeringProfile:
    return PromptEngineeringProfile.from_sources(
        profile_id="reel.h3.r2av-multiref.v1",
        sources=[("h3-r2av-multiref-pe/SKILL.md", path)],
        output_contract=PromptOutputContract(
            single_paragraph=False,
            max_characters=24000,
            required_any_prefixes=("subject_definitions:",),
        ),
    )


def _strings(value: Any, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (not nonempty or bool(value))
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _role(binding: Mapping[str, Any]) -> tuple[str, str]:
    effect = binding["intended_effect"]
    timeline = binding["timeline_role"]
    if effect in {"exact_opening_state", "temporal_continuation"}:
        if timeline != "opening":
            raise H3PEError("reference.timeline", "Opening evidence needs opening scope")
        return "first_frame", "the first frame of the target video"
    if effect == "exact_ending_state":
        if timeline != "ending":
            raise H3PEError("reference.timeline", "Ending evidence needs ending scope")
        return "keyframe", "a keyframe of the target video at the end"
    if effect == "composition_anchor" and timeline in {"middle", "ending"}:
        phase = "during the middle of the requested action" if timeline == "middle" else "at the end"
        return "keyframe", f"a keyframe of the target video {phase}"
    return "reference", "a reference image"


def _project_text(text: str, subjects: Mapping[str, str]) -> str:
    projected = text
    for identifier, label in sorted(subjects.items(), key=lambda item: -len(item[0])):
        projected = re.sub(
            rf"(?<!\w){re.escape(identifier)}(?!\w)",
            lambda match: label,
            projected,
        )
    if PRIVATE_RE.search(projected):
        raise H3PEError("input.private_metadata", "Unresolved private metadata in PE text")
    return projected


def _summary_prefix(references: list[dict]) -> str:
    prefix = "[reference generation]"
    if any(reference["role"] != "reference" for reference in references):
        prefix += " + [keyframe completion]"
    return prefix


@dataclass(frozen=True, slots=True)
class H3PreparedPrompt:
    document_json: str
    images: tuple[GeminiInlineImage, ...]

    @property
    def document(self) -> dict[str, Any]:
        return json.loads(self.document_json)

    @property
    def prepared_sha256(self) -> str:
        return sha256_text(self.document_json)


@dataclass(frozen=True, slots=True)
class H3PromptCompilation:
    document_json: str
    artifact_path: Path

    @property
    def document(self) -> dict[str, Any]:
        return json.loads(self.document_json)

    @property
    def prompt(self) -> str:
        return self.document["prompt"]


def validate_observation(text: str, prepared: H3PreparedPrompt) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as error:
        raise H3PEError("grounding.json", "Observation must be strict JSON") from error
    if not isinstance(value, dict) or set(value) != {
        "status", "references", "request_mapping", "conflicts", "blocked_reasons"
    }:
        raise H3PEError("grounding.schema", "Observation fields differ from the contract")
    if not isinstance(value["status"], str) or value["status"] not in {"ready", "blocked"}:
        raise H3PEError("grounding.status", "Unknown grounding status")
    expected = prepared.document["input"]["references"]
    references = value["references"]
    if not isinstance(references, list) or len(references) != len(expected):
        raise H3PEError("grounding.coverage", "Every supplied image must be observed")
    for actual, reference in zip(references, expected):
        if not isinstance(actual, dict) or set(actual) != {
            "label", "role", "visible_facts", "uncertainties"
        }:
            raise H3PEError("grounding.schema", "Malformed reference observation")
        if actual["label"] != reference["label"] or actual["role"] != reference["role"]:
            raise H3PEError("grounding.binding", "Observation changed image order or role")
        if not _strings(actual["visible_facts"], nonempty=True) or not _strings(actual["uncertainties"]):
            raise H3PEError("grounding.facts", "Reference facts and uncertainty must be explicit")
    mappings = value["request_mapping"]
    if not isinstance(mappings, list) or not mappings:
        raise H3PEError("grounding.request", "The request needs an evidence mapping")
    labels = {reference["label"] for reference in expected}
    for mapping in mappings:
        if not isinstance(mapping, dict) or set(mapping) != {"constraint", "reference_labels", "basis"}:
            raise H3PEError("grounding.request", "Malformed request mapping")
        if not isinstance(mapping["constraint"], str) or not mapping["constraint"].strip():
            raise H3PEError("grounding.request", "Request constraint must be nonempty")
        if not _strings(mapping["reference_labels"]) or not set(mapping["reference_labels"]).issubset(labels):
            raise H3PEError("grounding.request", "Request mapping uses an unavailable image")
        if not isinstance(mapping["basis"], str) or mapping["basis"] not in GROUNDING_BASES:
            raise H3PEError("grounding.request", "Unknown grounding basis")
        if mapping["basis"] == "visible_reference" and not mapping["reference_labels"]:
            raise H3PEError("grounding.request", "A visible claim needs image evidence")
    if not _strings(value["conflicts"]) or not _strings(value["blocked_reasons"]):
        raise H3PEError("grounding.schema", "Grounding failures must be string arrays")
    if value["status"] == "blocked" or value["blocked_reasons"]:
        raise H3PEError("grounding.blocked", "Reference grounding reported an unresolved problem")
    if value["conflicts"]:
        raise H3PEError(
            "grounding.inconsistent_status", "Ready status contradicts unresolved conflicts",
            details={"rule": "Reassess the evidence: unresolved contradictions require blocked; resolved differences belong in observations or uncertainty, not conflicts. Never clear a genuine unresolved conflict to pass."},
        )
    return value


def validate_h3_prompt(text: str, prepared: H3PreparedPrompt) -> dict[str, str]:
    if not isinstance(text, str) or not 1 <= len(text) <= 24000 or text != text.strip():
        raise H3PEError("output.length", "Invalid prompt length or outer whitespace")
    if "```" in text or PRIVATE_RE.search(text):
        raise H3PEError("output.private_or_wrapped", "Prompt contains private metadata or a code fence")
    matches = list(re.finditer(r"(?m)^([a-z_]+):[ \t]*$", text))
    if tuple(match.group(1) for match in matches) != HEADINGS or matches[0].start() != 0:
        raise H3PEError("output.sections", "Expected exactly the six ordered H3 sections")
    sections = {
        match.group(1): text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)].strip()
        for index, match in enumerate(matches)
    }
    if not all(sections.values()):
        raise H3PEError("output.sections", "Empty H3 section")
    pack = prepared.document["input"]
    references = pack["references"]
    subject_labels = [subject["label"] for subject in pack["subjects"]]
    image_labels = [reference["label"] for reference in references]
    available = set(subject_labels + image_labels)
    if any(match.group(0) not in available for match in LABEL_RE.finditer(text)):
        raise H3PEError("output.labels", "Prompt mentions an undefined image or subject")
    for token in re.findall(r"<[^<>\n]*>", text):
        if token not in available and token not in {"<d>", "</d>"}:
            raise H3PEError("output.labels", "Unsupported or malformed reference label")
    definitions = [line.strip() for line in sections["subject_definitions"].splitlines() if line.strip()]
    expected_definitions = subject_labels + image_labels
    declared = [match.group(0) for line in definitions if (match := LABEL_RE.match(line))]
    if declared != expected_definitions:
        raise H3PEError(
            "output.definitions", "Define each subject and image exactly once in order",
            details={"expected_line_prefixes": expected_definitions, "actual_line_prefixes": declared},
        )
    for reference in references:
        line = next(line for line in definitions if line.startswith(reference["label"]))
        stem = reference["definition"].removesuffix(".")
        if not line.startswith(stem) or re.match(r"[.,;:](?:\s|$)", line[len(stem):]) is None:
            raise H3PEError(
                "output.reference_role", "The required image-use definition changed",
                details={"image_definitions": [row["definition"] for row in references],
                         "rule": "Each definition already begins with its label; do not prepend another label."},
            )
    retention_lines = [line.strip() for line in sections["retention_analysis"].splitlines() if line.strip()]
    if len(retention_lines) != len(references):
        raise H3PEError("output.retention", "Each reference needs its own retention line")
    for reference, line in zip(references, retention_lines):
        match = re.match(r"^(<Image \d+>)\s*(?::|—|-)?\s*([a-z_]+)\b", line)
        if not match or match.group(1) != reference["label"] or match.group(2) not in RETENTION_MODES:
            raise H3PEError(
                "output.retention", "Invalid retention label, order, or mode",
                details={"expected_labels": image_labels, "allowed_modes": sorted(RETENTION_MODES)},
            )
        if reference["role"] != "reference" and match.group(2) in {"weak_reference", "attribute_transfer"}:
            raise H3PEError("output.reference_role", "A hard frame anchor cannot become a weak reference")
    expected_summary = _summary_prefix(references)
    summary_match = re.match(r"^\[reference generation\](?: \+ \[keyframe completion\])?", sections["summary"])
    if summary_match is None or summary_match.group(0) != expected_summary:
        raise H3PEError("output.task_type", "Incorrect task-type prefix",
                        details={"expected_summary_prefix": expected_summary})
    if "[video editing]" in text or "[video continuation]" in text:
        raise H3PEError("output.task_type", "Image references do not establish a source video")
    if EDITORIAL_RE.search(LITERAL_RE.sub("", text)):
        raise H3PEError("output.timing", "Use semantic phases, not shot labels or numeric timing")
    return sections


def normalize_h3_prompt(text: str) -> str:
    def normalize_segment(segment: str) -> str:
        segment = re.sub(r"`(<(?:Image|Subject) \d+>)`", r"\1", segment)
        segment = re.sub(
            r"(?<![<\w])(Image|Subject)\s+(\d+)(?![\w>])",
            lambda match: f"<{match.group(1).capitalize()} {match.group(2)}>",
            segment,
            flags=re.IGNORECASE,
        )
        return re.sub(
            r"(?m)^([ \t]*)(<(?:Image|Subject) \d+>)[ \t]*:?[ \t]*\2[ \t]+",
            r"\1\2 ",
            segment,
        )

    parts = []
    cursor = 0
    for literal in LITERAL_RE.finditer(text):
        parts.append(normalize_segment(text[cursor:literal.start()]))
        parts.append(literal.group(0))
        cursor = literal.end()
    parts.append(normalize_segment(text[cursor:]))
    return "".join(parts)


class H3R2AVPromptAdapter:
    """Prepare authorized references and PE artifacts without mutating REEL state.

    Only the existing evidence-authentication helpers are shared with Echo.
    Its seven-slot allocation, V7 PE, reservation, and HTTP payload are not used.
    Service submission remains unavailable until the deployment contract is supplied.
    """

    def __init__(
        self,
        *,
        artifact_root: Path,
        gemini_client: GeminiTextPort,
        max_references: int = MAX_REFERENCES,
        max_pe_attempts: int = 2,
        profile: PromptEngineeringProfile | None = None,
    ) -> None:
        if type(max_references) is not int or not 1 <= max_references <= MAX_REFERENCES:
            raise ValueError("max_references must be between one and twelve")
        if type(max_pe_attempts) is not int or not 1 <= max_pe_attempts <= 3:
            raise ValueError("max_pe_attempts must be between one and three")
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.gemini_client = gemini_client
        self.max_references = max_references
        self.max_pe_attempts = max_pe_attempts
        self.profile = profile or load_h3_profile()
        self._evidence = _Echo15ConditionCompiler(
            capabilities=Echo15Capabilities(),
            asset_root=self.artifact_root / "assets",
        )

    def prepare(
        self,
        *,
        state: Any,
        raw_prompt: str,
        evidence_plan: Mapping[str, Any] | None = None,
    ) -> H3PreparedPrompt:
        data = state if isinstance(state, Mapping) else state.data
        if not isinstance(data, Mapping):
            raise TypeError("state must expose a mapping")
        if not isinstance(raw_prompt, str) or sha256_text(raw_prompt) != data.get("shot_requirement_sha256"):
            raise H3PEError("input.raw_prompt", "Request differs from the immutable REEL request")
        plan = self._evidence._accepted_plan(data, evidence_plan)
        generation_prompt = plan.get("generation_prompt")
        if not isinstance(generation_prompt, str) or not 10 <= len(generation_prompt.strip()) <= 24000:
            raise H3PEError("input.generation_prompt", "An accepted REEL-authored generation prompt is required")
        bindings = plan["ranked_bindings"]
        if not 1 <= len(bindings) <= self.max_references:
            raise H3PEError("reference.capacity", f"R2AV requires one to twelve selected references (configured maximum {self.max_references}, received {len(bindings)}); none may be silently dropped")
        if len({binding["binding_id"] for binding in bindings}) != len(bindings):
            raise H3PEError("reference.duplicate_binding", "Binding identifiers must be unique")
        for binding in bindings:
            EvidenceBinding(**binding).validate()
            if binding["modality"] != "image":
                raise H3PEError("reference.modality", "This PE profile cannot inspect audio or video inputs; do not drop them")
            if binding["intended_effect"] in {"voice_reference", "audio_reference", "soft_audio_reference"}:
                raise H3PEError("reference.effect", "Images cannot provide audio or voice evidence")
        roles = [_role(binding) for binding in bindings]
        if sum(role == "first_frame" for role, _ in roles) > 1:
            raise H3PEError("reference.opening_conflict", "More than one first-frame constraint is unresolved")
        if sum(role == "keyframe" and binding["timeline_role"] == "ending" for binding, (role, _) in zip(bindings, roles)) > 1:
            raise H3PEError("reference.ending_conflict", "More than one final keyframe is unresolved")
        registry = (data.get("registry") or {}).get("items") or {}
        identifiers = list(dict.fromkeys(
            identifier
            for row in [*plan.get("requirements", []), *bindings]
            for identifier in row.get("subject_ids", [])
        ))
        for identifier in registry:
            if identifier not in identifiers and re.search(rf"(?<!\w){re.escape(identifier)}(?!\w)", raw_prompt + "\n" + generation_prompt):
                identifiers.append(identifier)
        subject_map = {identifier: f"<Subject {index}>" for index, identifier in enumerate(identifiers, 1)}
        subjects = [
            {
                "label": subject_map[identifier],
                "description": _project_text(
                    Echo15BackendAdapter._natural_subject_label(
                        registry.get(identifier) or {}, fallback="the selected visual subject"
                    ),
                    subject_map,
                ),
            }
            for identifier in identifiers
        ]
        requirements = {row["requirement_id"]: row for row in plan.get("requirements", [])}
        references = []
        lineage = []
        images = []
        for index, (binding, (role, usage)) in enumerate(zip(bindings, roles), 1):
            resolved = self._evidence._resolve_binding(data, binding)
            payload = Path(resolved.content_path).read_bytes()
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    image_format = image.format
                    width, height = image.size
                    image.verify()
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                raise H3PEError("reference.decode", "An authorized reference is not a decodable image") from error
            actual_mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(image_format)
            if actual_mime != resolved.mime_type:
                raise H3PEError("reference.mime", "Image encoding and authorized MIME type differ or are unsupported")
            label = f"<Image {index}>"
            targets = [subject_map[identifier] for identifier in binding["subject_ids"]]
            if role == "reference":
                target_text = ", ".join(targets) or "the selected " + ", ".join(binding["semantic_roles"])
                usage += f" for {target_text}, used for {binding['intended_effect'].replace('_', ' ')}"
            reference = {
                "label": label,
                "role": role,
                "definition": f"{label} is {usage}.",
                "subject_labels": targets,
                "semantic_roles": binding["semantic_roles"],
                "intended_effect": binding["intended_effect"],
                "timeline_role": binding["timeline_role"],
                "requirements": [
                    _project_text(str(requirements[identifier]["description"]), subject_map)
                    for identifier in binding["requirement_ids"]
                ],
            }
            references.append(reference)
            lineage.append({
                "label": label,
                "binding": binding,
                "content_path": resolved.content_path,
                "content_sha256": resolved.content_sha256,
                "size_bytes": resolved.size_bytes,
                "mime_type": resolved.mime_type,
                "width": width,
                "height": height,
                "creation_purpose": resolved.purpose,
                "role": role,
            })
            images.append(GeminiInlineImage.from_bytes(mime_type=actual_mime, data=payload))
        document = {
            "schema_version": VERSION,
            "request_id": data["request_id"],
            "plan_sha256": plan["plan_sha256"],
            "raw_prompt_sha256": sha256_text(raw_prompt),
            "generation_prompt_sha256": sha256_text(generation_prompt),
            "input": {
                "original_request": _project_text(raw_prompt, subject_map),
                "generation_prompt": _project_text(generation_prompt, subject_map),
                "summary_prefix": _summary_prefix(references),
                "subjects": subjects,
                "references": references,
            },
            "reference_lineage": lineage,
        }
        return H3PreparedPrompt(canonical_json(document), tuple(images))

    def _stage(self, stage: str, prepared: H3PreparedPrompt, observation: dict | None = None) -> dict:
        correction = None
        for attempt in range(1, self.max_pe_attempts + 1):
            request = {"stage": stage, **prepared.document["input"]}
            if observation is not None:
                request["observation"] = observation
            if correction is not None:
                request["correction"] = correction
            result = self.gemini_client.generate_text(
                system_instruction=self.profile.system_instruction,
                user_text=canonical_json(request),
                images=prepared.images,
            )
            if not isinstance(result, GeminiTextResult):
                raise TypeError("PE transport must return GeminiTextResult")
            raw_text = result.text.strip()
            text = normalize_h3_prompt(raw_text) if stage == "rewrite" else raw_text
            try:
                if stage == "observe":
                    output = validate_observation(text, prepared)
                else:
                    validate_h3_prompt(text, prepared)
                    output = text
            except H3PEError as error:
                rejection = {
                    "schema_version": VERSION,
                    "status": "rejected_private_pe_output",
                    "h3_submitted": False,
                    "prepared_sha256": prepared.prepared_sha256,
                    "profile_sha256": self.profile.profile_sha256,
                    "stage": stage,
                    "attempt": attempt,
                    "code": error.code,
                    "details": error.details,
                    "request_manifest": result.request_manifest,
                    "response_sha256": sha256_text(result.text),
                    "response_excerpt": raw_text[:24000],
                }
                _atomic_write_json(
                    self.artifact_root / "rejected_pe" / (sha256_json(rejection).removeprefix("sha256:") + ".json"),
                    rejection,
                )
                if error.code == "grounding.blocked" or attempt == self.max_pe_attempts:
                    raise
                correction = {
                    "code": error.code,
                    "details": error.details,
                    "instruction": "Return a fresh response satisfying the system contract and this diagnostic. Keep genuinely unresolved grounding problems blocked.",
                }
                continue
            return {
                "output": output,
                "request_manifest": result.request_manifest,
                "attempt": attempt,
                "response_sha256": sha256_text(result.text),
                "output_text_sha256": sha256_text(text),
                "label_markup_normalized": text != raw_text,
                "model": result.request_manifest.get("model"),
            }
        raise RuntimeError("PE attempts exhausted")

    def compile(
        self,
        *,
        state: Any,
        raw_prompt: str,
        evidence_plan: Mapping[str, Any] | None = None,
    ) -> H3PromptCompilation:
        prepared = self.prepare(state=state, raw_prompt=raw_prompt, evidence_plan=evidence_plan)
        cache_key = sha256_json({
            "prepared": prepared.prepared_sha256,
            "profile": self.profile.profile_sha256,
            "version": VERSION,
        }).removeprefix("sha256:")
        directory = self.artifact_root / "pe" / cache_key
        artifact_path = directory / "compilation.json"
        with _exclusive_lock(directory / ".lock"):
            if artifact_path.exists():
                artifact = self._read_cache(artifact_path, cache_key)
                if artifact["prepared"] != prepared.document or artifact["profile_sha256"] != self.profile.profile_sha256:
                    raise H3PEError("cache.corrupt", "Cached PE input differs from this request")
                validate_observation(canonical_json(artifact["observation"]), prepared)
                validate_h3_prompt(artifact["prompt"], prepared)
                return H3PromptCompilation(canonical_json(artifact), artifact_path)
            observation_path = directory / "observation.json"
            if observation_path.exists():
                observed = self._read_cache(observation_path, cache_key)
                validate_observation(canonical_json(observed["result"]["output"]), prepared)
            else:
                observed = {"cache_key": cache_key, "result": self._stage("observe", prepared)}
                self._write_cache(observation_path, observed)
            observation = observed["result"]["output"]
            rewritten = self._stage("rewrite", prepared, observation)
            artifact = {
                "schema_version": VERSION,
                "status": "pe_only",
                "h3_submitted": False,
                "cache_key": cache_key,
                "profile_sha256": self.profile.profile_sha256,
                "profile_sources": json.loads(self.profile.source_manifest_json),
                "prepared": prepared.document,
                "observation": observation,
                "prompt": rewritten["output"],
                "prompt_sha256": sha256_text(rewritten["output"]),
                "pe_requests": [
                    {key: value for key, value in result.items() if key != "output"}
                    for result in (observed["result"], rewritten)
                ],
            }
            self._write_cache(artifact_path, artifact)
            return H3PromptCompilation(canonical_json(artifact), artifact_path)

    @staticmethod
    def _write_cache(path: Path, value: dict) -> None:
        value["artifact_sha256"] = sha256_json(value)
        _atomic_write_json(path, value)

    @staticmethod
    def _read_cache(path: Path, cache_key: str) -> dict:
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("cache is not an object")
            digest = value.pop("artifact_sha256")
            if digest != sha256_json(value) or value["cache_key"] != cache_key:
                raise ValueError("cache identity differs")
            value["artifact_sha256"] = digest
            return value
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise H3PEError("cache.corrupt", "PE cache is corrupt and will not be overwritten") from error
