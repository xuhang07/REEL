"""Deterministic Echo 1.5 backend adapter for REEL Memory Policy.

The public adapter boundary accepts a raw shot prompt plus an accepted
provider-neutral EvidencePlan.  Prompt engineering is an implementation
detail owned by this adapter: callers never need to orchestrate prompt
artifacts or expose PE as a Memory Policy tool.

No network call to Echo is made here.  The returned reservation payload must
first be committed through ``MemoryPolicyRuntime.reserve_generation``.  Only
then may :meth:`Echo15Compilation.build_invocation_request` construct the one
request authorized by that reservation.
"""

from __future__ import annotations

import copy
import fcntl
import json
import mimetypes
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from storymem_teacher.source import _declared_character_category

from .models import (
    BasePromptArtifact,
    BasePromptRequest,
    OpeningCondition,
    PromptEngineeringProfile,
    PromptOutputContract,
    WirePromptArtifact,
    canonical_json,
    sha256_bytes,
    sha256_json,
    sha256_text,
    strict_json_copy,
)
from .prompt_engineering import (
    DEFAULT_MAX_PE_GENERATION_ATTEMPTS,
    AdapterPromptEngineer,
    PromptArtifactConflict,
    pe_generation_policy_manifest,
)
from .gemini import IsolatedGeminiTextClient
from .echo_prompt_budget import get_echo_prompt_budget
from .v7_validation import (
    DirectorV7PromptValidator,
    V7PromptValidationError,
    build_semantic_preservation_contract,
    infer_caption_language,
)


ECHO15_ADAPTER_VERSION = "storymem.backend.echo15-adapter.v7"
ECHO15_CONDITION_PROJECTION_SCHEMA_VERSION = (
    "storymem.backend.echo15-condition-projection.v1"
)
ECHO15_COMPILATION_RECEIPT_SCHEMA_VERSION = (
    "storymem.backend.echo15-compilation-receipt.v3"
)
ECHO15_STORY_PE_CONTEXT_SCHEMA_VERSION = "storymem.backend.echo15-story-pe-context.v1"
ECHO15_SERVER_MAX_MEMORY_SLOTS = 7
ECHO15_MAX_PE_GENERATION_ATTEMPTS = DEFAULT_MAX_PE_GENERATION_ATTEMPTS

_OPENING_EFFECTS = frozenset({"exact_opening_state", "temporal_continuation"})
_SOURCE_TABLES = {
    "retrieved_memory": "memory_bank",
    "uploaded_material": "uploaded_evidence",
    "sequence_context": "previous_evidence",
    "generated_evidence": "created_evidence",
    "generation_derived": "generation_evidence",
}
_FALLBACKS = frozenset({"forbidden", "allow_as_reference", "allow_drop"})
_CAPTION_LANGUAGES = frozenset({"English", "Simplified Chinese"})
_TRUSTED_AUDIO_BINDING_BASIS = "trusted_user_declared"
_TRUSTED_AUDIO_MATCHING_METHOD = "exact_normalized_name"
_LEGACY_MATERIAL_TAG_PATTERN = (
    r"<\s*(?:id|entity|scene|obj|object|prop|style)_[1-9][0-9]*\s*>"
)
_CANONICAL_NONPERSON_MATERIAL_TAG_PATTERN = (
    r"(?<![A-Za-z0-9_])(?:ENTITY|SCENE|OBJECT|OBJ|PROP|STYLE)_"
    r"[A-Z0-9][A-Z0-9_]*(?![A-Za-z0-9_])"
)
_NON_NATURAL_SUBJECT_TAG_RE = re.compile(
    r"(?ix)(?:"
    + r"(?<![A-Za-z0-9_])ID_[A-Z][A-Z0-9_]*(?![A-Za-z0-9_])|"
    + _CANONICAL_NONPERSON_MATERIAL_TAG_PATTERN
    + r"|"
    + _LEGACY_MATERIAL_TAG_PATTERN
    + r")"
)
_PRIVATE_MATERIAL_TAG_RE = re.compile(
    r"(?ix)(?:"
    + _CANONICAL_NONPERSON_MATERIAL_TAG_PATTERN
    + r"|"
    + _LEGACY_MATERIAL_TAG_PATTERN
    + r")"
)
_SUBJECT_SEMANTIC_ROLES = frozenset(
    {
        "identity",
        "appearance",
        "wardrobe",
        "character",
        "person",
        "voice",
        "character_identity",
        "character_appearance",
        "entity_identity",
        "entity_appearance",
    }
)


class Echo15AdapterError(RuntimeError):
    """A condition plan cannot be compiled without changing its semantics."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        binding_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.binding_id = binding_id
        self.details = copy.deepcopy(dict(details or {}))


class _ReservationRuntime(Protocol):
    def reserve_generation(self, compilation: Mapping[str, Any]) -> dict[str, Any]: ...


def _as_state_data(state: Any) -> Mapping[str, Any]:
    value = state if isinstance(state, Mapping) else getattr(state, "data", None)
    if not isinstance(value, Mapping):
        raise TypeError("state must be a mapping or expose a mapping .data")
    return value


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Echo15AdapterError("invalid_input", f"{field_name} must be non-empty")
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
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
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _safe_suffix(path: Path, mime_type: str) -> str:
    suffix = path.suffix.lower()
    if suffix and len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    guessed = mimetypes.guess_extension(mime_type) or ".bin"
    return ".jpg" if guessed == ".jpe" else guessed


@dataclass(frozen=True, slots=True)
class _OfficialV7PromptValidator:
    caption_language: str
    semantic_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.caption_language not in _CAPTION_LANGUAGES:
            raise ValueError("official v7 validator has an invalid caption language")
        if not isinstance(self.semantic_contract, Mapping):
            raise TypeError("official v7 validator requires a semantic contract")

    @property
    def validation_id(self) -> str:
        language = self.caption_language.lower().replace(" ", "-")
        tokenizer = get_echo_prompt_budget().tokenizer_sha256
        return f"storymem.echo15.official-v7.{language}.semantic-preservation.v12.source-cast-gemma1024.{tokenizer}"

    @property
    def output_budget_contract(self) -> Mapping[str, Any]:
        return get_echo_prompt_budget().contract

    def validate_budget(self, text: str) -> Mapping[str, Any]:
        return get_echo_prompt_budget().validate(text)

    def _semantic_validator(self) -> DirectorV7PromptValidator:
        # The profile's prose-length target cannot override the encoder's
        # physical capacity. Neither language should pad a short request just
        # to reach the legacy 1500-character floor.
        return DirectorV7PromptValidator(
            semantic_contract=self.semantic_contract,
            min_characters=1,
            max_characters=None,
        )

    def validate(
        self,
        text: str,
        *,
        raw_prompt: str,
        condition_effect: str | None,
        expected_caption_language: str | None,
    ) -> Mapping[str, Any]:
        if expected_caption_language != self.caption_language:
            raise ValueError("active validator differs from the locked caption language")
        if _PRIVATE_MATERIAL_TAG_RE.search(text):
            raise V7PromptValidationError(
                "wire prompt must describe non-human entities naturally and must not "
                "contain registry or legacy material tags",
                violations=[{"code": "privacy.private_material_tag"}],
            )
        budget = get_echo_prompt_budget().validate(text)
        result = self._semantic_validator().validate(
            text,
            raw_prompt=raw_prompt,
            condition_effect=condition_effect,
            expected_caption_language=self.caption_language,
        )
        if result.get("caption_language") != self.caption_language:
            raise ValueError("wire prompt differs from the locked caption language")
        if re.search(
            r"(?i)(?:"
            r"(?:reference\s+)?(?:image|picture|photo)\s*"
            r"(?:no\.?\s*|number\s*|[#]\s*)?[1-9][0-9]*|"
            r"第\s*[一二三四五六七八九十百千0-9]+\s*张?\s*"
            r"(?:参考)?(?:图像|图片|图)|"
            r"参考(?:图像|图片|图)\s*"
            r"[一二三四五六七八九十百千0-9]+"
            r")",
            text,
        ):
            raise V7PromptValidationError(
                "wire prompt must not name numbered reference images",
                violations=[{"code": "privacy.numbered_reference_image"}],
            )
        if re.search(
            r"(?ix)(?:https?://|file://|data:image/|(?:^|\s)/(?:pfs|tmp|home|root)/|"
            r"\b(?:binding|evidence|representation|viewauth|upload|mem)_[a-z0-9_.:-]+\b)",
            text,
        ):
            raise V7PromptValidationError(
                "wire prompt contains a private path, URL, or evidence handle",
                violations=[{"code": "privacy.path_url_or_evidence_handle"}],
            )
        return {**result, "encoder_budget": budget, "validation_id": self.validation_id}

    def correction_context(
        self,
        *,
        raw_prompt: str,
        condition_effect: str | None,
        expected_caption_language: str | None,
    ) -> Mapping[str, Any]:
        if expected_caption_language != self.caption_language:
            raise ValueError("active validator differs from the locked caption language")
        context = self._semantic_validator().correction_context(
            raw_prompt=raw_prompt,
            condition_effect=condition_effect,
            expected_caption_language=self.caption_language,
        )
        return {**context, "encoder_budget": get_echo_prompt_budget().contract}


@dataclass(frozen=True, slots=True)
class Echo15Capabilities:
    """Private limits used by the Echo compiler, never by Memory Policy."""

    max_memory_slots: int = ECHO15_SERVER_MAX_MEMORY_SLOTS
    audio_memory: bool = True

    def __post_init__(self) -> None:
        if type(self.max_memory_slots) is not int or not (
            0 <= self.max_memory_slots <= ECHO15_SERVER_MAX_MEMORY_SLOTS
        ):
            raise ValueError("max_memory_slots must be in [0, 7]")
        if type(self.audio_memory) is not bool:
            raise ValueError("audio_memory must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_memory_slots": self.max_memory_slots,
            "audio_memory": self.audio_memory,
        }


@dataclass(frozen=True, slots=True)
class Echo15RequestSpec:
    """Host-owned non-semantic fields for one Echo request."""

    work_id: str
    shot_id: str
    sequence_index: int | None = None
    num_frames: int | None = None
    width: int | None = None
    height: int | None = None
    duration_sec: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("work_id", "shot_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.sequence_index is not None and (
            type(self.sequence_index) is not int or self.sequence_index < 0
        ):
            raise ValueError("sequence_index must be null or a non-negative integer")
        for name in ("num_frames", "width", "height"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be null or a positive integer")
        if self.duration_sec is not None and (
            isinstance(self.duration_sec, bool)
            or not isinstance(self.duration_sec, (int, float))
            or self.duration_sec <= 0
        ):
            raise ValueError("duration_sec must be null or positive")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2 ** 32):
            raise ValueError("seed must be null or a uint32")

    def apply(self, request: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(request)
        value["work_id"] = self.work_id.strip()
        value["shot_id"] = self.shot_id.strip()
        for name in ("num_frames", "width", "height", "duration_sec", "seed"):
            child = getattr(self, name)
            if child is not None:
                value[name] = child
        return value


@dataclass(frozen=True, slots=True)
class _ResolvedBinding:
    binding: dict[str, Any]
    source_path: str
    content_path: str
    content_sha256: str
    size_bytes: int
    mime_type: str
    purpose: str | None

    @property
    def binding_id(self) -> str:
        return str(self.binding["binding_id"])

    @property
    def modality(self) -> str:
        return str(self.binding["modality"])

    @property
    def intended_effect(self) -> str:
        return str(self.binding["intended_effect"])

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.binding.get("subject_ids") or [])


@dataclass(frozen=True, slots=True)
class _PreparedEcho15Conditions:
    """Sealed internal result shared by PE projection and final compilation."""

    _document_json: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "_PreparedEcho15Conditions":
        document = strict_json_copy(value, "prepared Echo conditions")
        if document.get("schema_version") != ECHO15_CONDITION_PROJECTION_SCHEMA_VERSION:
            raise ValueError("prepared Echo condition schema is invalid")
        stored = document.get("prepared_sha256")
        unhashed = copy.deepcopy(document)
        unhashed.pop("prepared_sha256", None)
        if stored != sha256_json(unhashed):
            raise ValueError("prepared Echo condition hash is invalid")
        return cls(canonical_json(document))

    @property
    def document(self) -> dict[str, Any]:
        return copy.deepcopy(json.loads(self._document_json))

    @property
    def prepared_sha256(self) -> str:
        return str(self.document["prepared_sha256"])

    @property
    def wire_projection(self) -> dict[str, Any]:
        return self.document["wire_projection"]

    @property
    def opening_condition(self) -> OpeningCondition | None:
        opening = self.document.get("opening_condition")
        if not isinstance(opening, Mapping):
            return None
        return OpeningCondition(
            condition_id=str(opening["binding_id"]),
            image_path=str(opening["content_path"]),
            semantic_effect=str(opening["semantic_effect"]),
            mime_type=str(opening["mime_type"]),
        )


@dataclass(frozen=True, slots=True)
class Echo15Compilation:
    """Immutable compiler output bound to exact prompt and condition bytes."""

    _document_json: str
    _base_artifact_json: str
    _wire_artifact_json: str

    @classmethod
    def create(
        cls,
        *,
        document: Mapping[str, Any],
        base: BasePromptArtifact,
        wire: WirePromptArtifact,
    ) -> "Echo15Compilation":
        value = strict_json_copy(document, "Echo compilation")
        return cls(
            canonical_json(value),
            canonical_json(base.to_dict()),
            canonical_json(wire.to_dict()),
        )

    @property
    def document(self) -> dict[str, Any]:
        return copy.deepcopy(json.loads(self._document_json))

    @property
    def reservation_input(self) -> dict[str, Any]:
        return self.document["reservation_input"]

    @property
    def echo_request_template(self) -> dict[str, Any]:
        return self.document["echo_request_template"]

    @property
    def receipt(self) -> dict[str, Any]:
        return self.document["compilation_receipt"]

    @property
    def provenance(self) -> dict[str, Any]:
        return self.document["provenance"]

    @property
    def base_prompt_artifact(self) -> BasePromptArtifact:
        return BasePromptArtifact.from_dict(json.loads(self._base_artifact_json))

    @property
    def wire_prompt_artifact(self) -> WirePromptArtifact:
        return WirePromptArtifact.from_dict(json.loads(self._wire_artifact_json))

    def reserve(self, runtime: _ReservationRuntime) -> dict[str, Any]:
        return runtime.reserve_generation(self.reservation_input)

    def build_invocation_request(
        self, reservation: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Bind the private request to one freshly committed reservation."""

        if reservation.get("invoke_allowed") is not True or reservation.get(
            "replayed"
        ) is not False:
            raise Echo15AdapterError(
                "invocation_not_authorized",
                "only a fresh durable reservation may authorize an Echo call",
            )
        generation_id = _require_text(reservation.get("generation_id"), "generation_id")
        request = self.echo_request_template
        request["job_id"] = generation_id
        return request


class _Echo15ConditionCompiler:
    def __init__(
        self,
        *,
        capabilities: Echo15Capabilities,
        asset_root: Path,
    ) -> None:
        self.capabilities = capabilities
        self.asset_root = asset_root
        self.asset_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _accepted_plan(
        data: Mapping[str, Any], evidence_plan: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        accepted = data.get("accepted_evidence_plan")
        if not isinstance(accepted, Mapping):
            raise Echo15AdapterError("plan_missing", "state has no accepted EvidencePlan")
        plan = accepted if evidence_plan is None else evidence_plan
        if not isinstance(plan, Mapping):
            raise TypeError("evidence_plan must be a mapping")
        value = strict_json_copy(plan, "EvidencePlan")
        if (
            value.get("plan_id") != accepted.get("plan_id")
            or value.get("plan_sha256") != accepted.get("plan_sha256")
        ):
            raise Echo15AdapterError(
                "stale_plan", "only the state's currently accepted EvidencePlan may compile"
            )
        stored_hash = value.get("plan_sha256")
        unhashed = copy.deepcopy(value)
        unhashed.pop("plan_sha256", None)
        if stored_hash != sha256_json(unhashed):
            raise Echo15AdapterError("plan_corrupt", "EvidencePlan hash is invalid")
        if (
            value.get("request_id") != data.get("request_id")
            or value.get("raw_request_sha256") != data.get("shot_requirement_sha256")
        ):
            raise Echo15AdapterError(
                "plan_lineage_mismatch", "EvidencePlan does not belong to this request"
            )
        rows = value.get("ranked_bindings")
        if not isinstance(rows, list):
            raise Echo15AdapterError("plan_corrupt", "ranked_bindings must be an array")
        if [row.get("rank") for row in rows if isinstance(row, Mapping)] != list(
            range(1, len(rows) + 1)
        ):
            raise Echo15AdapterError(
                "plan_corrupt", "binding ranks must be list-ordered and contiguous"
            )
        representation_keys: dict[tuple[str, str, str], str] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise Echo15AdapterError("plan_corrupt", "binding is not an object")
            binding_id = str(row.get("binding_id") or "")
            key = (
                str(row.get("source_kind") or ""),
                str(row.get("evidence_id") or ""),
                str(row.get("representation_id") or ""),
            )
            prior_binding_id = representation_keys.get(key)
            if prior_binding_id is not None:
                raise Echo15AdapterError(
                    "plan_corrupt",
                    "one evidence representation is repeated across bindings",
                    binding_id=binding_id or None,
                    details={
                        "first_binding_id": prior_binding_id,
                        "duplicate_binding_id": binding_id,
                    },
                )
            representation_keys[key] = binding_id
        return value

    def _materialize(self, path: Path, payload: bytes, mime_type: str) -> Path:
        digest = sha256_bytes(payload)
        target = self.asset_root / (
            digest.removeprefix("sha256:") + _safe_suffix(path, mime_type)
        )
        with _exclusive_lock(target.with_suffix(target.suffix + ".lock")):
            if target.exists():
                if sha256_bytes(target.read_bytes()) != digest:
                    raise Echo15AdapterError(
                        "adapter_asset_corrupt",
                        f"content-addressed adapter asset is corrupt: {target}",
                    )
            else:
                _atomic_write_bytes(target, payload)
        return target.resolve()

    def _resolve_binding(
        self, data: Mapping[str, Any], binding: Mapping[str, Any]
    ) -> _ResolvedBinding:
        binding_id = _require_text(binding.get("binding_id"), "binding_id")
        source_kind = str(binding.get("source_kind") or "")
        table_name = _SOURCE_TABLES.get(source_kind)
        if table_name is None:
            raise Echo15AdapterError(
                "invalid_binding", "unsupported source kind", binding_id=binding_id
            )
        evidence_id = _require_text(binding.get("evidence_id"), "evidence_id")
        representation_id = _require_text(
            binding.get("representation_id"), "representation_id"
        )
        authorization_id = _require_text(
            binding.get("authorization_id"), "authorization_id"
        )
        records = data.get(table_name)
        record = records.get(evidence_id) if isinstance(records, Mapping) else None
        if not isinstance(record, Mapping):
            raise Echo15AdapterError(
                "evidence_unavailable", "evidence record is missing", binding_id=binding_id
            )
        view_key = "|".join(
            [source_kind, evidence_id, representation_id, authorization_id]
        )
        views = data.get("views")
        if not isinstance(views, Mapping) or view_key not in views:
            raise Echo15AdapterError(
                "evidence_not_authorized",
                "binding lacks exact current-session view authorization",
                binding_id=binding_id,
            )
        representations = record.get("representations")
        representation = (
            representations.get(representation_id)
            if isinstance(representations, Mapping)
            else None
        )
        if not isinstance(representation, Mapping):
            direct = record.get("representation")
            representation = direct if isinstance(direct, Mapping) else None
        if not isinstance(representation, Mapping):
            raise Echo15AdapterError(
                "evidence_unavailable",
                "binding representation is missing",
                binding_id=binding_id,
            )
        modality = str(binding.get("modality") or "")
        if modality not in {"image", "video", "audio", "text"}:
            raise Echo15AdapterError(
                "invalid_binding", "binding modality is invalid", binding_id=binding_id
            )
        if representation.get("modality") != modality or (
            record.get("modality") is not None and record.get("modality") != modality
        ):
            raise Echo15AdapterError(
                "evidence_modality_mismatch",
                "binding and representation modalities differ",
                binding_id=binding_id,
            )
        subjects = binding.get("subject_ids") or []
        if (
            not isinstance(subjects, list)
            or any(not isinstance(item, str) or not item for item in subjects)
            or len(set(subjects)) != len(subjects)
            or list(record.get("subject_ids") or []) != subjects
        ):
            raise Echo15AdapterError(
                "subject_binding_mismatch",
                "binding subjects differ from the authorized evidence",
                binding_id=binding_id,
            )
        if modality == "audio":
            speaker_binding = record.get("speaker_binding")
            if not isinstance(speaker_binding, Mapping):
                raise Echo15AdapterError(
                    "untrusted_audio_speaker_binding",
                    "audio evidence has no trusted named-speaker binding",
                    binding_id=binding_id,
                )
            declared_name = str(speaker_binding.get("declared_name") or "").strip()
            speaker_subject_id = str(speaker_binding.get("subject_id") or "")
            speaker_name = record.get("speaker_name")
            if (
                speaker_binding.get("basis") != _TRUSTED_AUDIO_BINDING_BASIS
                or speaker_binding.get("matching_method")
                != _TRUSTED_AUDIO_MATCHING_METHOD
                or not declared_name
                or len(subjects) != 1
                or speaker_subject_id != subjects[0]
                or (
                    speaker_name is not None
                    and str(speaker_name).strip() != declared_name
                )
            ):
                raise Echo15AdapterError(
                    "untrusted_audio_speaker_binding",
                    "audio evidence must retain one exact trusted named-speaker "
                    "binding matching the EvidencePlan subject",
                    binding_id=binding_id,
                )
        path = Path(str(representation.get("path") or "")).expanduser().resolve()
        if not path.is_file():
            raise Echo15AdapterError(
                "evidence_unavailable", "evidence bytes are missing", binding_id=binding_id
            )
        payload = path.read_bytes()
        if not payload:
            raise Echo15AdapterError(
                "evidence_unavailable", "evidence bytes are empty", binding_id=binding_id
            )
        content_hash = sha256_bytes(payload)
        if representation.get("sha256") != content_hash or representation.get(
            "size_bytes"
        ) != len(payload):
            raise Echo15AdapterError(
                "evidence_integrity_mismatch",
                "evidence bytes differ from the authorized representation",
                binding_id=binding_id,
            )
        mime_type = str(
            representation.get("mime_type")
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        if modality in {"image", "video", "audio"} and not mime_type.startswith(
            modality + "/"
        ):
            raise Echo15AdapterError(
                "evidence_modality_mismatch",
                "representation MIME type differs from its modality",
                binding_id=binding_id,
            )
        materialized = self._materialize(path, payload, mime_type)
        return _ResolvedBinding(
            binding=strict_json_copy(binding, "binding"),
            source_path=str(path),
            content_path=str(materialized),
            content_sha256=content_hash,
            size_bytes=len(payload),
            mime_type=mime_type,
            purpose=(str(record.get("purpose")) if record.get("purpose") else None),
        )

    @staticmethod
    def _outcome(
        item: _ResolvedBinding,
        *,
        outcome: str,
        applied_effect: str | None,
        reason_code: str,
    ) -> dict[str, Any]:
        return {
            "binding_id": item.binding_id,
            "outcome": outcome,
            "applied_effect": applied_effect,
            "reason_code": reason_code,
        }

    @staticmethod
    def _cannot_apply(
        item: _ResolvedBinding, *, reason_code: str, message: str
    ) -> dict[str, Any]:
        fallback = str(item.binding.get("fallback") or "")
        if fallback not in _FALLBACKS:
            raise Echo15AdapterError(
                "invalid_binding", "binding fallback is invalid", binding_id=item.binding_id
            )
        if fallback == "allow_drop":
            return _Echo15ConditionCompiler._outcome(
                item,
                outcome="not_applied",
                applied_effect=None,
                reason_code=reason_code,
            )
        raise Echo15AdapterError(
            "condition_not_compilable",
            message,
            binding_id=item.binding_id,
            details={"reason_code": reason_code},
        )

    def prepare(
        self,
        *,
        state: Any,
        evidence_plan: Mapping[str, Any] | None = None,
    ) -> _PreparedEcho15Conditions:
        data = _as_state_data(state)
        if data.get("phase") != "awaiting_generation":
            raise Echo15AdapterError(
                "wrong_phase", "Echo conditions require an accepted plan awaiting generation"
            )
        plan = self._accepted_plan(data, evidence_plan)
        resolved: list[_ResolvedBinding] = []
        precomputed_outcomes: dict[str, dict[str, Any]] = {}
        diagnostics: list[dict[str, Any]] = []
        for raw in plan["ranked_bindings"]:
            if not isinstance(raw, Mapping):
                raise Echo15AdapterError("plan_corrupt", "binding is not an object")
            binding_id = str(raw.get("binding_id") or "")
            try:
                item = self._resolve_binding(data, raw)
            except Echo15AdapterError as exc:
                fallback = str(raw.get("fallback") or "")
                if fallback == "allow_drop":
                    precomputed_outcomes[binding_id] = {
                        "binding_id": binding_id,
                        "outcome": "not_applied",
                        "applied_effect": None,
                        "reason_code": (
                            "condition.unavailable"
                            if exc.code == "evidence_unavailable"
                            else "condition.invalid_evidence"
                        ),
                    }
                    diagnostics.append(
                        {
                            "binding_id": binding_id,
                            "code": exc.code,
                            "message": str(exc),
                        }
                    )
                    continue
                raise
            resolved.append(item)

        opening: _ResolvedBinding | None = None
        slot_candidates: list[tuple[_ResolvedBinding, str, str]] = []
        audio_candidates: list[_ResolvedBinding] = []

        for item in resolved:
            effect = item.intended_effect
            if item.modality == "image":
                generated_opening = item.purpose == "opening_frame_evidence"
                generated_reference = item.purpose == "reference_evidence"
                opening_intent = effect in _OPENING_EFFECTS
                if generated_opening and not opening_intent:
                    precomputed_outcomes[item.binding_id] = self._cannot_apply(
                        item,
                        reason_code="condition.invalid_evidence",
                        message="opening-frame evidence requires an opening semantic effect",
                    )
                    continue
                if opening_intent and generated_reference:
                    if item.binding.get("fallback") == "allow_as_reference":
                        slot_candidates.append((item, "allowed_fallback", "visual_reference"))
                    else:
                        precomputed_outcomes[item.binding_id] = self._cannot_apply(
                            item,
                            reason_code="condition.invalid_evidence",
                            message="reference-only generated evidence cannot become frame zero",
                        )
                    continue
                if opening_intent or generated_opening:
                    if opening is None:
                        opening = item
                    elif item.binding.get("fallback") == "allow_as_reference":
                        slot_candidates.append((item, "allowed_fallback", "visual_reference"))
                    else:
                        precomputed_outcomes[item.binding_id] = self._cannot_apply(
                            item,
                            reason_code="condition.conflict",
                            message="Echo accepts only one frame-zero condition",
                        )
                    continue
                if effect == "exact_ending_state":
                    if item.binding.get("fallback") == "allow_as_reference":
                        slot_candidates.append((item, "allowed_fallback", "visual_reference"))
                    else:
                        precomputed_outcomes[item.binding_id] = self._cannot_apply(
                            item,
                            reason_code="condition.unsupported",
                            message="Echo cannot express an exact ending state with this image",
                        )
                    continue
                slot_candidates.append((item, "applied_as_requested", effect))
            elif item.modality == "audio":
                audio_candidates.append(item)
            else:
                precomputed_outcomes[item.binding_id] = self._cannot_apply(
                    item,
                    reason_code="condition.unsupported",
                    message=f"Echo cannot directly consume {item.modality} evidence",
                )

        slots: list[dict[str, Any]] = []
        applied: dict[str, dict[str, Any]] = {}
        for item, status, applied_effect in sorted(
            slot_candidates, key=lambda row: int(row[0].binding["rank"])
        ):
            if len(slots) >= self.capabilities.max_memory_slots:
                precomputed_outcomes[item.binding_id] = self._cannot_apply(
                    item,
                    reason_code="condition.capacity_limited",
                    message="Echo memory-slot capacity cannot preserve this binding",
                )
                continue
            slots.append(
                {
                    "ordinal": len(slots) + 1,
                    "image": item,
                    "image_effect": applied_effect,
                    "audio": None,
                    "audio_effect": None,
                    "reuses_opening_image": False,
                }
            )
            applied[item.binding_id] = self._outcome(
                item,
                outcome=status,
                applied_effect=applied_effect,
                reason_code=(
                    "condition.applied"
                    if status == "applied_as_requested"
                    else "condition.fallback"
                ),
            )

        if opening is not None:
            applied[opening.binding_id] = self._outcome(
                opening,
                outcome="applied_as_requested",
                applied_effect=opening.intended_effect,
                reason_code="condition.applied",
            )

        for audio in sorted(audio_candidates, key=lambda item: int(item.binding["rank"])):
            if not self.capabilities.audio_memory:
                precomputed_outcomes[audio.binding_id] = self._cannot_apply(
                    audio,
                    reason_code="condition.unsupported",
                    message="Echo audio-memory transport is disabled",
                )
                continue
            subjects = audio.subjects
            matching = [
                slot
                for slot in slots
                if len(subjects) == 1
                and slot["image"].subjects == subjects
                and slot["audio"] is None
            ]
            if (
                not matching
                and opening is not None
                and len(subjects) == 1
                and opening.subjects == subjects
            ):
                if len(slots) >= self.capabilities.max_memory_slots:
                    precomputed_outcomes[audio.binding_id] = self._cannot_apply(
                        audio,
                        reason_code="condition.capacity_limited",
                        message=(
                            "Echo has no free slot for the opening-image copy required "
                            "to transport this audio binding"
                        ),
                    )
                    continue
                target = {
                    "ordinal": 0,
                    "image": opening,
                    "image_effect": opening.intended_effect,
                    "audio": None,
                    "audio_effect": None,
                    "reuses_opening_image": True,
                }
                slots.append(target)
                slots.sort(key=lambda slot: int(slot["image"].binding["rank"]))
                for index, slot in enumerate(slots, start=1):
                    slot["ordinal"] = index
                matching = [target]
            if not matching:
                precomputed_outcomes[audio.binding_id] = self._cannot_apply(
                    audio,
                    reason_code="condition.conflict",
                    message=(
                        "audio requires one unused image slot with the exact same "
                        "single subject binding"
                    ),
                )
                continue
            target = matching[0]
            target["audio"] = audio
            target["audio_effect"] = audio.intended_effect
            applied[audio.binding_id] = self._outcome(
                audio,
                outcome="applied_as_requested",
                applied_effect=audio.intended_effect,
                reason_code="condition.applied",
            )

        outcomes: list[dict[str, Any]] = []
        for binding in plan["ranked_bindings"]:
            binding_id = str(binding["binding_id"])
            outcome = applied.get(binding_id) or precomputed_outcomes.get(binding_id)
            if outcome is None:
                raise RuntimeError(f"compiler lost binding outcome: {binding_id}")
            outcomes.append(outcome)

        compiled_conditions: list[dict[str, Any]] = []
        ordinal = 0
        if opening is not None:
            ordinal += 1
            compiled_conditions.append(
                {
                    "ordinal": ordinal,
                    "binding_ids": [opening.binding_id],
                    "modality": "image",
                    "content_path": opening.content_path,
                    "applied_effect": opening.intended_effect,
                }
            )
        for slot in slots:
            image: _ResolvedBinding = slot["image"]
            if not slot["reuses_opening_image"]:
                ordinal += 1
                compiled_conditions.append(
                    {
                        "ordinal": ordinal,
                        "binding_ids": [image.binding_id],
                        "modality": "image",
                        "content_path": image.content_path,
                        "applied_effect": slot["image_effect"],
                    }
                )
            audio = slot["audio"]
            if isinstance(audio, _ResolvedBinding):
                ordinal += 1
                compiled_conditions.append(
                    {
                        "ordinal": ordinal,
                        "binding_ids": [audio.binding_id],
                        "modality": "audio",
                        "content_path": audio.content_path,
                        "applied_effect": slot["audio_effect"],
                    }
                )

        resolved_by_binding_id = {item.binding_id: item for item in resolved}
        wire_media_keys: dict[tuple[str, str], str] = {}
        for condition in compiled_conditions:
            for binding_id in condition["binding_ids"]:
                item = resolved_by_binding_id[binding_id]
                key = (item.modality, item.content_sha256)
                prior_binding_id = wire_media_keys.get(key)
                if prior_binding_id is not None:
                    raise Echo15AdapterError(
                        "duplicate_wire_media",
                        "distinct bindings compile to duplicate physical wire media",
                        binding_id=item.binding_id,
                        details={
                            "first_binding_id": prior_binding_id,
                            "duplicate_binding_id": item.binding_id,
                            "modality": item.modality,
                            "content_sha256": item.content_sha256,
                        },
                    )
                wire_media_keys[key] = item.binding_id

        def projection_media(item: _ResolvedBinding, effect: str) -> dict[str, Any]:
            return {
                "binding_id": item.binding_id,
                "semantic_effect": effect,
                "content_sha256": item.content_sha256,
                "subject_ids": list(item.subjects),
            }

        wire_projection = {
            "schema_version": ECHO15_CONDITION_PROJECTION_SCHEMA_VERSION,
            "plan_id": plan["plan_id"],
            "plan_sha256": plan["plan_sha256"],
            "opening": (
                projection_media(opening, opening.intended_effect)
                if opening is not None
                else None
            ),
            "memory": [
                {
                    "ordinal": slot["ordinal"],
                    "image_reuses_opening_condition": slot[
                        "reuses_opening_image"
                    ],
                    "image": projection_media(slot["image"], slot["image_effect"]),
                    "audio": (
                        projection_media(slot["audio"], slot["audio_effect"])
                        if isinstance(slot["audio"], _ResolvedBinding)
                        else None
                    ),
                }
                for slot in slots
            ],
            "binding_outcomes": copy.deepcopy(outcomes),
        }

        all_resolved = resolved_by_binding_id
        media_provenance = [
            {
                "binding_id": binding["binding_id"],
                "source_kind": binding["source_kind"],
                "evidence_id": binding["evidence_id"],
                "representation_id": binding["representation_id"],
                "semantic_effect": binding["intended_effect"],
                "source_path": all_resolved[binding["binding_id"]].source_path,
                "content_path": all_resolved[binding["binding_id"]].content_path,
                "content_sha256": all_resolved[binding["binding_id"]].content_sha256,
                "size_bytes": all_resolved[binding["binding_id"]].size_bytes,
                "mime_type": all_resolved[binding["binding_id"]].mime_type,
            }
            for binding in plan["ranked_bindings"]
            if binding["binding_id"] in all_resolved
        ]

        document: dict[str, Any] = {
            "schema_version": ECHO15_CONDITION_PROJECTION_SCHEMA_VERSION,
            "adapter_version": ECHO15_ADAPTER_VERSION,
            "request_id": data["request_id"],
            "raw_request_sha256": data["shot_requirement_sha256"],
            "plan_id": plan["plan_id"],
            "plan_sha256": plan["plan_sha256"],
            "capabilities": self.capabilities.to_dict(),
            "wire_projection": wire_projection,
            "opening_condition": (
                {
                    "binding_id": opening.binding_id,
                    "semantic_effect": opening.intended_effect,
                    "content_path": opening.content_path,
                    "content_sha256": opening.content_sha256,
                    "mime_type": opening.mime_type,
                }
                if opening is not None
                else None
            ),
            "slots": [
                {
                    "ordinal": slot["ordinal"],
                    "image_binding_id": slot["image"].binding_id,
                    "image_path": slot["image"].content_path,
                    "image_effect": slot["image_effect"],
                    "audio_binding_id": (
                        slot["audio"].binding_id
                        if isinstance(slot["audio"], _ResolvedBinding)
                        else None
                    ),
                    "audio_path": (
                        slot["audio"].content_path
                        if isinstance(slot["audio"], _ResolvedBinding)
                        else None
                    ),
                    "audio_effect": slot["audio_effect"],
                    "image_reuses_opening_condition": slot[
                        "reuses_opening_image"
                    ],
                }
                for slot in slots
            ],
            "binding_outcomes": outcomes,
            "requested_effects": {
                binding["binding_id"]: binding["intended_effect"]
                for binding in plan["ranked_bindings"]
            },
            "compiled_conditions": compiled_conditions,
            "media_provenance": media_provenance,
            "diagnostics": diagnostics,
        }
        document["prepared_sha256"] = sha256_json(document)
        return _PreparedEcho15Conditions.from_dict(document)


def _official_v7_profiles(
    source_root: str | Path | None = None,
) -> tuple[PromptEngineeringProfile, dict[str, PromptEngineeringProfile]]:
    """Load the byte-exact official profiles without creating adapter state."""

    root = (
        Path(source_root).expanduser().resolve()
        if source_root is not None
        else Path(__file__).resolve().parents[2]
    )
    ordinary = (
        root
        / "Director_Agent"
        / "pe"
        / "v7_cinematic_full"
        / "references"
        / "shot-prompt-writer.md"
    )
    tail = (
        root
        / "Director_Agent"
        / "pe"
        / "v7_cinematic_full"
        / "skills"
        / "i2v-tail-frame-prompt-rewriter"
        / "SKILL.md"
    )
    budget = ("echo15_encoder_budget.md", root / "prompts" / "echo15_encoder_budget.md")
    output_contract = PromptOutputContract(max_characters=None)
    base_profile = PromptEngineeringProfile.from_sources(
        profile_id="director.v7_cinematic_full.base",
        sources=(("shot-prompt-writer.md", ordinary), budget),
        output_contract=output_contract,
    )
    temporal_profile = PromptEngineeringProfile.from_sources(
        profile_id="director.v7_cinematic_full.temporal-continuation",
        sources=(
            ("shot-prompt-writer.md", ordinary),
            ("i2v-tail-frame-prompt-rewriter/SKILL.md", tail),
            budget,
        ),
        output_contract=PromptOutputContract(
            max_characters=None,
            required_any_prefixes=(
                "以当前图片作为视频首帧，并基于首帧中已有的人物、物体、环境、构图、机位、光线和动作状态自然延续。",
                "Use the current image as the first frame of the video, and continue naturally from the characters, objects, environment, composition, camera position, lighting, and action state already shown in it.",
            ),
        ),
    )
    exact_profile = PromptEngineeringProfile.from_sources(
        profile_id="director.v7_cinematic_full.exact-opening-state",
        sources=(("shot-prompt-writer.md", ordinary), budget),
        output_contract=output_contract,
    )
    return base_profile, {
        "exact_opening_state": exact_profile,
        "temporal_continuation": temporal_profile,
    }


def official_v7_profile_manifest(
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return pathless resource and profile hashes for run semantic locking."""

    base, opening = _official_v7_profiles(source_root)
    profiles = [base, *(opening[key] for key in sorted(opening))]
    return {
        "pe_generation_policy": pe_generation_policy_manifest(
            ECHO15_MAX_PE_GENERATION_ATTEMPTS
        ),
        "profiles": {
            profile.profile_id: {
                "profile_sha256": profile.profile_sha256,
                "source_manifest": json.loads(profile.source_manifest_json),
                "output_contract": profile.output_contract.to_dict(),
            }
            for profile in profiles
        }
    }


class _DisabledPromptEngineer:
    def generate_text(self, **kwargs: Any) -> Any:
        raise RuntimeError("PE is disabled for policy_direct; no provider may rewrite the prompt")


class Echo15BackendAdapter:
    """Unified Echo adapter: condition compilation and isolated PE in one call."""

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        gemini: Any,
        base_profile: PromptEngineeringProfile,
        opening_profiles: Mapping[str, PromptEngineeringProfile] | None,
        capabilities: Echo15Capabilities | None = None,
        receipt_root: str | Path | None = None,
        strict_v7: bool = False,
        caption_language_lock: str | None = None,
        max_pe_generation_attempts: int = ECHO15_MAX_PE_GENERATION_ATTEMPTS,
    ) -> None:
        if not callable(getattr(gemini, "generate_text", None)):
            raise TypeError("gemini must implement generate_text")
        if not isinstance(base_profile, PromptEngineeringProfile):
            raise TypeError("base_profile must be PromptEngineeringProfile")
        profiles = dict(opening_profiles or {})
        if set(profiles) - _OPENING_EFFECTS or any(
            not isinstance(value, PromptEngineeringProfile)
            for value in profiles.values()
        ):
            raise TypeError(
                "opening_profiles may contain only exact_opening_state and "
                "temporal_continuation PromptEngineeringProfiles"
            )
        root = Path(artifact_root).expanduser().resolve()
        self._base_profile = base_profile
        self._opening_profiles = profiles
        if type(strict_v7) is not bool:
            raise TypeError("strict_v7 must be boolean")
        if caption_language_lock not in {None, *_CAPTION_LANGUAGES}:
            raise ValueError(
                "caption_language_lock must be English, Simplified Chinese, or null"
            )
        if (
            type(max_pe_generation_attempts) is not int
            or max_pe_generation_attempts < 1
        ):
            raise ValueError("max_pe_generation_attempts must be positive")
        self._strict_v7 = strict_v7
        self._caption_language_lock = caption_language_lock
        self._max_pe_generation_attempts = max_pe_generation_attempts
        self._pe_generation_policy = pe_generation_policy_manifest(
            max_pe_generation_attempts
        )
        self._gemini = gemini
        self._prompt_artifact_root = root / "prompt_artifacts"
        self.capabilities = capabilities or Echo15Capabilities()
        self.receipt_root = Path(
            receipt_root or (root / "echo15_compilations")
        ).expanduser().resolve()
        self.receipt_root.mkdir(parents=True, exist_ok=True)
        self._conditions = _Echo15ConditionCompiler(
            capabilities=self.capabilities,
            asset_root=self.receipt_root / "assets",
        )

    @classmethod
    def from_official_v7(
        cls,
        *,
        artifact_root: str | Path,
        gemini: Any | None = None,
        capabilities: Echo15Capabilities | None = None,
        receipt_root: str | Path | None = None,
        source_root: str | Path | None = None,
        caption_language_lock: str | None = None,
        max_pe_generation_attempts: int = ECHO15_MAX_PE_GENERATION_ATTEMPTS,
    ) -> "Echo15BackendAdapter":
        """Build the production adapter from the repository-owned v7 PE files."""

        base_profile, opening_profiles = _official_v7_profiles(source_root)
        return cls(
            artifact_root=artifact_root,
            gemini=gemini or IsolatedGeminiTextClient.from_env(),
            base_profile=base_profile,
            opening_profiles=opening_profiles,
            capabilities=capabilities,
            receipt_root=receipt_root,
            strict_v7=True,
            caption_language_lock=caption_language_lock,
            max_pe_generation_attempts=max_pe_generation_attempts,
        )

    @staticmethod
    def _scope_id(data: Mapping[str, Any]) -> str:
        return f"{data['session_id']}:{data['request_id']}"

    @classmethod
    def from_policy_direct(cls, *, artifact_root: str | Path,
                           capabilities: Echo15Capabilities | None = None) -> "Echo15BackendAdapter":
        """Compile policy text and conditions without loading a PE template/client."""
        return cls(artifact_root=artifact_root, gemini=_DisabledPromptEngineer(),
                   base_profile=PromptEngineeringProfile.from_text(
                       profile_id="echo15.policy-direct.v1", system_instruction="No prompt rewriting."),
                   opening_profiles={}, capabilities=capabilities)

    def _compile_policy_direct(self, *, data: Mapping[str, Any], plan: Mapping[str, Any],
                               raw_prompt: str, request: Echo15RequestSpec) -> Echo15Compilation:
        draft = plan.get("generation_prompt")
        if not isinstance(draft, str) or not draft.strip() or draft != draft.strip():
            raise Echo15AdapterError("policy_prompt_invalid", "generation_prompt must be nonempty and trimmed")
        budget = get_echo_prompt_budget().validate(draft)
        prepared = self._conditions.prepare(state=data, evidence_plan=plan)
        engineer = AdapterPromptEngineer(root=self._prompt_artifact_root, gemini=_DisabledPromptEngineer())
        base = self._ensure_base_prompt(
            data=data, prompt_engineer=engineer, plan=plan, raw_prompt=raw_prompt,
            pe_mode="policy_direct", pe_input_text=draft, caption_language=None,
            origin_plan_projection=self._default_origin_projection(plan),
            initial_condition_projection=prepared.wire_projection)
        wire = engineer.compile_wire(base=base, condition_projection=prepared.wire_projection)
        validation = self._validate_wire(base=base, wire=wire, prepared=prepared,
            pe_mode="policy_direct", raw_prompt=raw_prompt, strict_v7=False,
            caption_language=None, semantic_contract=None)
        if wire.prompt_text != draft:
            raise Echo15AdapterError("policy_prompt_changed", "direct wire prompt must equal the accepted policy draft")
        validation.update(profile="policy_direct", validation_id="echo15.policy-direct.gemma1024.v1",
                          encoder_budget=budget, provider_call_count=0, byte_exact_policy_prompt=True)
        return self._finalize(prepared=prepared, base=base, wire=wire, request=request,
                              prompt_validation=validation, prompt_authoring=True)

    @staticmethod
    def _default_origin_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "storymem.adapter.prompt-origin.v1",
            "request_id": plan["request_id"],
            "raw_request_sha256": plan["raw_request_sha256"],
            # Requirements are immutable across condition revisions.
            "requirements": copy.deepcopy(plan.get("requirements") or []),
        }

    def _story_context_path(self, work_id: str) -> Path:
        component = sha256_text(work_id).removeprefix("sha256:")
        return self._prompt_artifact_root / "story_contexts" / f"{component}.json"

    @staticmethod
    def _seal_story_context(value: Mapping[str, Any]) -> dict[str, Any]:
        document = strict_json_copy(value, "story PE context")
        document.pop("context_sha256", None)
        document["context_sha256"] = sha256_json(document)
        return document

    @staticmethod
    def _validate_story_context(
        value: Any,
        *,
        request: Echo15RequestSpec,
        pe_mode: str,
        profile_sha256: str,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise PromptArtifactConflict("story PE context is not an object")
        document = strict_json_copy(value, "story PE context")
        stored_hash = document.pop("context_sha256", None)
        if stored_hash != sha256_json(document):
            raise PromptArtifactConflict("story PE context integrity check failed")
        document["context_sha256"] = stored_hash
        if (
            document.get("schema_version") != ECHO15_STORY_PE_CONTEXT_SCHEMA_VERSION
            or document.get("adapter_version") != ECHO15_ADAPTER_VERSION
            or document.get("work_id") != request.work_id.strip()
            or document.get("pe_mode") != pe_mode
            or document.get("base_profile_sha256") != profile_sha256
            or not isinstance(document.get("shots"), dict)
        ):
            raise PromptArtifactConflict(
                "story PE context is locked to different adapter input"
            )
        language = document.get("caption_language")
        if language not in {None, *_CAPTION_LANGUAGES}:
            raise PromptArtifactConflict("story PE context has an invalid language lock")
        return document

    def _story_pe_inputs(
        self,
        *,
        data: Mapping[str, Any],
        request: Echo15RequestSpec,
        raw_prompt: str,
        pe_mode: str,
    ) -> tuple[str | None, str | None]:
        """Return the adapter-owned language lock and prior finished prompt.

        ``sequence_index`` is host ordering metadata, not a PE artifact.  When
        present it gives the adapter enough information to persist and resume
        story-level language/cast continuity without making the collector
        construct a Gemini turn.
        """

        if pe_mode != "gemini_rewrite":
            return None, None
        if request.sequence_index is not None and request.work_id.strip() != str(
            data.get("session_id") or ""
        ):
            raise PromptArtifactConflict(
                "ordered story PE requires request.work_id to match state.session_id"
            )
        inferred = (
            self._caption_language_lock
            or (infer_caption_language(raw_prompt) if self._strict_v7 else None)
        )
        if request.sequence_index is None:
            return inferred, None

        path = self._story_context_path(request.work_id)
        with _exclusive_lock(path.with_suffix(".lock")):
            if path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise PromptArtifactConflict(
                        f"cannot read story PE context: {path}"
                    ) from exc
                context = self._validate_story_context(
                    loaded,
                    request=request,
                    pe_mode=pe_mode,
                    profile_sha256=self._base_profile.profile_sha256,
                )
                language = context.get("caption_language")
                if self._caption_language_lock is not None and (
                    language != self._caption_language_lock
                ):
                    raise PromptArtifactConflict(
                        "story PE context conflicts with the adapter language lock"
                    )
            else:
                if request.sequence_index != 0:
                    raise PromptArtifactConflict(
                        "story PE context must begin at sequence_index=0"
                    )
                context = self._seal_story_context(
                    {
                        "schema_version": ECHO15_STORY_PE_CONTEXT_SCHEMA_VERSION,
                        "adapter_version": ECHO15_ADAPTER_VERSION,
                        "work_id": request.work_id.strip(),
                        "pe_mode": pe_mode,
                        "base_profile_sha256": self._base_profile.profile_sha256,
                        "caption_language": inferred,
                        "shots": {},
                    }
                )
                _atomic_write_json(path, context)
                language = inferred

            shots = context["shots"]
            index = request.sequence_index
            current = shots.get(str(index))
            if current is not None and (
                not isinstance(current, Mapping)
                or current.get("shot_id") != request.shot_id.strip()
                or current.get("scope_id") != self._scope_id(data)
                or current.get("raw_prompt_sha256") != sha256_text(raw_prompt)
            ):
                raise PromptArtifactConflict(
                    "sequence position is already locked to another shot"
                )
            previous_prompt = None
            if index > 0:
                previous = shots.get(str(index - 1))
                if not isinstance(previous, Mapping) or not isinstance(
                    previous.get("base_prompt_text"), str
                ):
                    raise PromptArtifactConflict(
                        "previous accepted shot PE context is unavailable"
                    )
                previous_prompt = str(previous["base_prompt_text"])
            return (str(language) if language is not None else None), previous_prompt

    def _commit_story_pe_context(
        self,
        *,
        data: Mapping[str, Any],
        request: Echo15RequestSpec,
        raw_prompt: str,
        pe_mode: str,
        base: BasePromptArtifact,
    ) -> None:
        if pe_mode != "gemini_rewrite" or request.sequence_index is None:
            return
        path = self._story_context_path(request.work_id)
        with _exclusive_lock(path.with_suffix(".lock")):
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PromptArtifactConflict(
                    f"cannot read story PE context: {path}"
                ) from exc
            context = self._validate_story_context(
                loaded,
                request=request,
                pe_mode=pe_mode,
                profile_sha256=self._base_profile.profile_sha256,
            )
            index = request.sequence_index
            shots = context["shots"]
            existing = shots.get(str(index))
            row = {
                "sequence_index": index,
                "shot_id": request.shot_id.strip(),
                "scope_id": self._scope_id(data),
                "raw_prompt_sha256": sha256_text(raw_prompt),
                "base_prompt_artifact_id": base.artifact_id,
                "base_prompt_artifact_sha256": base.artifact_sha256,
                "base_prompt_sha256": base.prompt_sha256,
                "base_prompt_text": base.prompt_text,
            }
            if existing is not None:
                if existing != row:
                    raise PromptArtifactConflict(
                        "story sequence position has conflicting PE lineage"
                    )
                return
            indices = sorted(int(key) for key in shots)
            if indices != list(range(index)):
                raise PromptArtifactConflict(
                    "story PE context must be committed in contiguous sequence order"
                )
            shots[str(index)] = row
            _atomic_write_json(path, self._seal_story_context(context))

    @staticmethod
    def _registry_subject_kind(
        subject_id: str, raw: Mapping[str, Any]
    ) -> str | None:
        """Return a registry-backed person/entity kind, never an asset kind.

        EvidencePlan ``subject_ids`` may point at any registry item, including a
        scene or style plate.  Only a taxonomy-checked person or non-human
        entity is safe to project into the PE subject channel.
        """

        if not isinstance(raw, Mapping):
            return None
        roles = {
            str(value).strip().casefold().replace(" ", "_")
            for value in raw.get("semantic_roles") or []
            if isinstance(value, str) and value.strip()
        }
        category_text = re.sub(
            r"[\s-]+",
            "_",
            str(raw.get("category") or "").strip().casefold(),
        )
        explicit_asset = any(
            category_text == token or category_text.startswith(token + "_")
            for token in (
                "scene",
                "environment",
                "location",
                "object",
                "prop",
                "style",
                "voice",
                "audio",
            )
        )
        if explicit_asset:
            return None
        explicit_entity = any(
            category_text == token or category_text.startswith(token + "_")
            for token in (
                "entity",
                "nonhuman",
                "non_human",
                "animal",
                "robot",
                "android",
                "creature",
                "mascot",
            )
        )
        if explicit_entity:
            return "entity"
        character_like = bool(
            roles.intersection(_SUBJECT_SEMANTIC_ROLES)
            or any(
                token in category_text
                for token in ("character", "person", "actor", "human")
            )
        )
        if not character_like:
            return None
        aliases = [
            str(value).strip()
            for value in raw.get("aliases") or []
            if isinstance(value, str) and value.strip()
        ]
        descriptor = {
            key: copy.deepcopy(raw[key])
            for key in (
                "category",
                "entity_type",
                "subject_type",
                "species",
                "is_human",
                "name",
                "role",
                "type",
                "description",
            )
            if key in raw
        }
        if not str(descriptor.get("name") or "").strip() and aliases:
            descriptor["name"] = aliases[0]
        if not str(descriptor.get("role") or "").strip() and aliases:
            descriptor["role"] = " ".join(aliases)
        descriptor["description"] = str(
            raw.get("stable_description")
            or raw.get("description")
            or descriptor.get("description")
            or ""
        )
        category = _declared_character_category(
            descriptor,
            label=f"registry.items[{subject_id!r}]",
            fail_if_ambiguous=False,
        )
        registry_tag = str(raw.get("tag") or "")
        if category is None and (
            re.fullmatch(r"ID_[A-Z][A-Z0-9_]*", str(subject_id))
            or re.fullmatch(r"ID_[A-Z][A-Z0-9_]*", registry_tag)
        ):
            return "person"
        return category

    @staticmethod
    def _canonical_cast_map(data: Mapping[str, Any]) -> dict[str, str]:
        registry = data.get("registry")
        items = registry.get("items") if isinstance(registry, Mapping) else None
        if not isinstance(items, Mapping):
            return {}
        person_subjects = []
        for subject_id, raw in sorted(items.items(), key=lambda item: str(item[0])):
            if not isinstance(raw, Mapping):
                continue
            if Echo15BackendAdapter._registry_subject_kind(
                str(subject_id), raw
            ) == "person":
                person_subjects.append(str(subject_id))

        def letters(position: int) -> str:
            value = position + 1
            output = ""
            while value:
                value, remainder = divmod(value - 1, 26)
                output = chr(ord("A") + remainder) + output
            return output

        mapping: dict[str, str] = {}
        used = {
            subject_id
            for subject_id in person_subjects
            if re.fullmatch(r"ID_[A-Z][A-Z0-9_]*", subject_id)
        }
        next_index = 0
        for subject_id in person_subjects:
            if subject_id in used:
                mapping[subject_id] = subject_id
                continue
            while f"ID_{letters(next_index)}" in used:
                next_index += 1
            canonical_id = f"ID_{letters(next_index)}"
            next_index += 1
            used.add(canonical_id)
            mapping[subject_id] = canonical_id
        return mapping

    @staticmethod
    def _natural_subject_text(value: Any) -> str:
        """Return PE-safe natural prose without public or legacy subject tags."""

        if not isinstance(value, str) or not value.strip():
            return ""
        text = _NON_NATURAL_SUBJECT_TAG_RE.sub(" ", value)
        text = " ".join(text.split()).strip(" \t\r\n:：,，;；-–—")
        return text

    @staticmethod
    def _natural_subject_label(
        row: Mapping[str, Any], *, fallback: str = "selected subject"
    ) -> str:
        aliases = row.get("aliases")
        candidates = [
            *(aliases if isinstance(aliases, list) else []),
            row.get("name"),
            row.get("role"),
            row.get("stable_description"),
            row.get("description"),
            row.get("tag"),
        ]
        return next(
            (
                text
                for text in (
                    Echo15BackendAdapter._natural_subject_text(value)
                    for value in candidates
                )
                if text
            ),
            fallback,
        )

    @staticmethod
    def _natural_requirement_description(
        value: Any, *, subject_replacements: Mapping[str, str]
    ) -> str:
        """Replace registry-only subject tags before exposing a requirement to PE."""

        text = str(value or "")
        for private_label, natural_label in sorted(
            subject_replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if private_label and natural_label and private_label != natural_label:
                text = re.sub(
                    re.escape(private_label),
                    lambda _: natural_label,
                    text,
                    flags=re.IGNORECASE,
                )
        # A malformed requirement may mention an unbound entity/legacy tag.
        # Keep that metadata out of the provider context even when no natural
        # registry description is available for a lossless replacement.
        text = _PRIVATE_MATERIAL_TAG_RE.sub("selected subject", text)
        return " ".join(text.split())

    @staticmethod
    def _stable_pe_context(
        *,
        data: Mapping[str, Any],
        plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Project immutable semantic requirements into PE-safe continuity facts.

        Exact selected evidence is deliberately absent.  It may change during
        a condition-only revision while the shot prompt is locked.  The active
        bytes are compiled through the native condition channels, and an
        opening image gets its own image-aware wire rewrite.
        """

        cast_map = Echo15BackendAdapter._canonical_cast_map(data)
        registry = data.get("registry")
        registry_items = (
            registry.get("items") if isinstance(registry, Mapping) else {}
        )
        registry_items = registry_items if isinstance(registry_items, Mapping) else {}
        cast: dict[str, dict[str, Any]] = {}
        requirements: list[dict[str, Any]] = []
        for requirement in plan.get("requirements") or []:
            if not isinstance(requirement, Mapping):
                continue
            subject_labels: list[str] = []
            subject_replacements: dict[str, str] = {}
            for subject in requirement.get("subject_ids") or []:
                subject_id = str(subject)
                raw_registry_row = registry_items.get(subject_id)
                has_registry_row = isinstance(raw_registry_row, Mapping)
                registry_row = raw_registry_row if has_registry_row else {}
                canonical_id = cast_map.get(subject_id)
                if (
                    canonical_id is None
                    and not has_registry_row
                    and re.fullmatch(r"ID_[A-Z][A-Z0-9_]*", subject_id)
                ):
                    # Protocol-level IDs with no registry row predate the richer
                    # registry taxonomy.  They are human by definition.  An
                    # explicit registry entity row always wins and is excluded.
                    canonical_id = subject_id
                aliases = registry_row.get("aliases")
                aliases = aliases if isinstance(aliases, list) else []
                display_name = Echo15BackendAdapter._natural_subject_label(
                    registry_row,
                    fallback=canonical_id or "selected subject",
                )
                subject_label = canonical_id or display_name
                subject_labels.append(subject_label)
                subject_replacements[subject_id] = subject_label
                registry_tag = registry_row.get("tag")
                if isinstance(registry_tag, str) and registry_tag.strip():
                    subject_replacements[registry_tag.strip()] = subject_label
                if canonical_id is not None:
                    natural_aliases = list(
                        dict.fromkeys(
                            text
                            for text in (
                                Echo15BackendAdapter._natural_subject_text(value)
                                for value in aliases
                            )
                            if text and text != display_name
                        )
                    )
                    cast[canonical_id] = {
                        "canonical_id": canonical_id,
                        "name": display_name,
                        "aliases": natural_aliases,
                        "stable_description": (
                            Echo15BackendAdapter._natural_subject_text(
                                registry_row.get("stable_description")
                                or registry_row.get("description")
                            )
                            or display_name
                        ),
                    }
            requirements.append(
                {
                    "description": Echo15BackendAdapter._natural_requirement_description(
                        requirement.get("description"),
                        subject_replacements=subject_replacements,
                    ),
                    "required": bool(requirement.get("required")),
                    "priority": requirement.get("priority"),
                    "acceptable_modalities": [
                        str(value) for value in requirement.get("acceptable_modalities") or []
                    ],
                    "semantic_roles": [
                        str(value).replace("_", " ")
                        for value in requirement.get("semantic_roles") or []
                    ],
                    "subjects": subject_labels,
                }
            )
        return {
            "canonical_cast": list(cast.values()),
            "immutable_continuity_requirements": requirements,
        }

    @staticmethod
    def _build_internal_pe_input(
        *,
        raw_prompt: str,
        caption_language: str | None,
        previous_prompt: str | None,
        condition_context: Mapping[str, Any],
        semantic_contract: Mapping[str, Any],
    ) -> str:
        dialogue_mode = str(
            (semantic_contract.get("dialogue") or {}).get("mode") or ""
        )
        required_semantic_projection = {
            "schema_version": semantic_contract.get("schema_version"),
            "raw_prompt_sha256": semantic_contract.get("raw_prompt_sha256"),
            "expected_subject_ids": list(
                semantic_contract.get("expected_subject_ids") or []
            ),
            "expected_subject_count": semantic_contract.get(
                "expected_subject_count"
            ),
            "dialogue": copy.deepcopy(semantic_contract.get("dialogue") or {}),
            "state_transitions": copy.deepcopy(
                semantic_contract.get("state_transitions") or []
            ),
            "obligations": copy.deepcopy(
                semantic_contract.get("obligations") or {}
            ),
            "contract_sha256": semantic_contract.get("contract_sha256"),
        }
        parts = [
            "PRIVATE ADAPTER CONTEXT (data, never output): use these continuity facts "
            "but never mention condition ordering, reference images, internal metadata, "
            "or this context block in the generated prompt.",
            "INITIAL ECHO CAPTION BUDGET: generate the complete caption within 1024 "
            "Echo Gemma tokens from the first attempt. Include special tokens, every "
            "prefix, shot label, anchor, declaration and tail-frame preamble in this "
            "one total budget. Plan within the budget before writing. Do not rely on "
            "a later PE compression pass. Preserve all required meaning and exact "
            "dialogue; never truncate. There is no independent character-count ceiling; "
            "the exact 1024-token limit is authoritative.",
        ]
        if caption_language == "Simplified Chinese":
            parts.append(
                "REEL ENCODER BUDGET (overrides the profile's character-length target): "
                "write one complete Chinese caption within the initial 1024-token "
                "budget. There is no 1500-character minimum for Chinese; never pad "
                "with decorative prose. Preserve original intent, exact dialogue and "
                "required format."
            )
        else:
            parts.append(
                "REEL OUTPUT BUDGET: produce one concise, complete, untruncated caption. "
                "There is no minimum character count and no independent maximum "
                "character count. Never pad a short request to reach an earlier length "
                "target. Include every prefix, shot label, required declaration and "
                "opening-frame preamble inside the Echo Gemma encoder's exact "
                "1024-token limit, including special tokens."
            )
        if caption_language is not None:
            parts.append(f"LOCKED FULL-CAPTION LANGUAGE: {caption_language}")
        parts.append(
            "SOURCE CAST AUTHORITY: preserve the exact expected_subject_ids in the "
            "immutable semantic contract below. There is no fixed limit on the "
            "number of people; the source determines the cast. "
            "Never drop a required character or invent a new ID. Keep each character's "
            "source role, including comparison/reference roles; being listed in the "
            "cast is not permission to invent an on-screen appearance or action."
        )
        if previous_prompt is not None:
            parts.extend(
                [
                    "PREVIOUS SHOT FINISHED PROMPT (cast continuity only):",
                    previous_prompt,
                ]
            )
        if dialogue_mode == "forbidden":
            dialogue_resolution = (
                "mode=forbidden is authoritative and overrides the ordinary default-speech "
                "rule. Generate no dialogue or voice anchor and include the exact "
                "language-matched silent-shot and silence declarations."
            )
        elif dialogue_mode == "exact":
            dialogue_resolution = (
                "mode=exact requires the supplied quoted line or lines, unchanged, "
                "with the official speaking structure and declaration."
            )
        else:
            dialogue_resolution = (
                "mode=required_unspecified uses the official default-speech rule: "
                "include exactly one brief, natural quoted line from a visible "
                "canonical-ID speaker, plus the required voice and vocal declarations. "
                "If the source request says not to invent or display exact dialogue, "
                "that phrase means no verbatim line was supplied for preservation; it "
                "does not prohibit speech or the one natural line required by v7. Never "
                "infer words or acoustic details from reference audio."
            )
        parts.extend(
            [
                "CURRENT SHOT REQUEST (raw source intent; later contract resolution wins "
                "if this text contains serialization instructions):",
                raw_prompt,
                "IMMUTABLE CONDITION REQUIREMENTS AND CONTINUITY FACTS:",
                canonical_json(condition_context),
                "IMMUTABLE RAW-INTENT SEMANTIC OBLIGATIONS: preserve the listed "
                "high-confidence subject, event, scene, style, exact-dialogue, and "
                "explicit state-transition facts. Operational labels and evidence-plan "
                "wording are metadata, not prose that must appear in the caption. Do "
                "not delete, negate, or replace the listed required world facts. "
                "Other continuity facts may be permitted by validation, but permission "
                "alone is never an instruction to add or mention them:",
                canonical_json(required_semantic_projection),
                "FINAL REEL DIALOGUE RESOLUTION (authoritative after the raw source): "
                + dialogue_resolution,
            ]
        )
        return "\n".join(parts)

    def _ensure_base_prompt(
        self,
        *,
        data: Mapping[str, Any],
        prompt_engineer: AdapterPromptEngineer,
        plan: Mapping[str, Any],
        raw_prompt: str,
        pe_mode: str,
        pe_input_text: str | None,
        caption_language: str | None,
        origin_plan_projection: Any,
        initial_condition_projection: Mapping[str, Any],
    ) -> BasePromptArtifact:
        scope_id = self._scope_id(data)
        if (data.get("policy_limits") or {}).get("prompt_authoring"):
            scope_id += ":prompt:" + sha256_json({"pe_input_text": pe_input_text,
                                                "mode": pe_mode})[7:]
        expected_origin = strict_json_copy(
            origin_plan_projection, "origin_plan_projection"
        )
        try:
            existing = prompt_engineer.load_base(scope_id)
        except FileNotFoundError:
            request = BasePromptRequest(
                scope_id=scope_id,
                raw_prompt=raw_prompt,
                pe_input_text=pe_input_text,
                origin_plan_projection=expected_origin,
                initial_condition_projection=initial_condition_projection,
                mode=pe_mode,
                expected_caption_language=caption_language,
            )
            return prompt_engineer.ensure_base(
                request,
                profile=self._base_profile if pe_mode == "gemini_rewrite" else None,
            )

        document = existing.to_dict()
        lineage = document.get("lineage") or {}
        if (
            existing.raw_prompt_sha256 != sha256_text(raw_prompt)
            or lineage.get("raw_prompt") != raw_prompt
            or document.get("mode") != pe_mode
            or lineage.get("origin_plan_projection") != expected_origin
            or lineage.get("expected_caption_language") != caption_language
        ):
            raise PromptArtifactConflict(
                "the shot-level base prompt is already locked with different input"
            )
        # The first accepted condition plan contributes private PE context to
        # the immutable base prompt.  Later condition-only revisions must reuse
        # that base even when their selected evidence descriptions differ.
        profile = document.get("profile")
        if pe_mode == "gemini_rewrite":
            if not isinstance(profile, Mapping) or profile.get(
                "profile_sha256"
            ) != self._base_profile.profile_sha256:
                raise PromptArtifactConflict(
                    "the shot-level base prompt is locked to another PE profile"
                )
            request_manifest = document.get("request_manifest")
            if not isinstance(request_manifest, Mapping) or request_manifest.get(
                "generation_policy"
            ) != prompt_engineer.generation_policy:
                raise PromptArtifactConflict(
                    "the shot-level base prompt is locked to another PE generation policy"
                )
        elif profile is not None:
            raise PromptArtifactConflict("passthrough artifact unexpectedly claims a PE profile")
        return existing

    @staticmethod
    def _validate_wire(
        *,
        base: BasePromptArtifact,
        wire: WirePromptArtifact,
        prepared: _PreparedEcho15Conditions,
        pe_mode: str,
        raw_prompt: str,
        strict_v7: bool,
        caption_language: str | None,
        semantic_contract: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        document = wire.to_dict()
        if (
            wire.base_artifact_id != base.artifact_id
            or document.get("base_artifact_sha256") != base.artifact_sha256
            or wire.scope_id != base.scope_id
        ):
            raise Echo15AdapterError(
                "prompt_lineage_mismatch", "wire prompt is not derived from the locked base"
            )
        lineage = document.get("lineage") or {}
        if lineage.get("active_condition_projection") != prepared.wire_projection:
            raise Echo15AdapterError(
                "prompt_condition_mismatch",
                "wire prompt was compiled for different condition bytes",
            )
        opening = prepared.document.get("opening_condition")
        opening_manifest = lineage.get("opening_condition")
        if pe_mode == "gemini_rewrite" and opening is not None:
            if wire.derivation != "opening_image_rewrite" or not isinstance(
                opening_manifest, Mapping
            ):
                raise Echo15AdapterError(
                    "opening_prompt_missing",
                    "rewritten mode requires an opening-aware wire prompt",
                )
            if (
                opening_manifest.get("condition_id") != opening.get("binding_id")
                or opening_manifest.get("semantic_effect")
                != opening.get("semantic_effect")
                or opening_manifest.get("sha256") != opening.get("content_sha256")
            ):
                raise Echo15AdapterError(
                    "opening_prompt_mismatch",
                    "wire prompt used a different frame-zero image",
                )
        else:
            if wire.derivation != "identity" or wire.prompt_text != base.prompt_text:
                raise Echo15AdapterError(
                    "unexpected_prompt_rewrite",
                    "this compilation requires an identity wire prompt",
                )
        if (
            not wire.prompt_text.strip()
            or wire.prompt_text != wire.prompt_text.strip()
        ):
            raise Echo15AdapterError(
                "wire_prompt_invalid",
                "Echo wire prompt must be nonempty and trimmed",
            )
        if strict_v7:
            if caption_language is None:
                raise Echo15AdapterError(
                    "caption_language_required",
                    "official v7 compilation requires the locked caption language",
                )
            return dict(
                _OfficialV7PromptValidator(
                    caption_language,
                    semantic_contract
                    or build_semantic_preservation_contract(raw_prompt),
                ).validate(
                    wire.prompt_text,
                    raw_prompt=raw_prompt,
                    condition_effect=(
                        str(opening["semantic_effect"])
                        if isinstance(opening, Mapping)
                        else None
                    ),
                    expected_caption_language=caption_language,
                )
            )
        return {
            "profile": "custom",
            "character_count": len(wire.prompt_text),
            "encoder_budget": get_echo_prompt_budget().validate(wire.prompt_text),
        }

    def _finalize(
        self,
        *,
        prepared: _PreparedEcho15Conditions,
        base: BasePromptArtifact,
        wire: WirePromptArtifact,
        request: Echo15RequestSpec,
        prompt_validation: Mapping[str, Any],
        prompt_authoring: bool = False,
    ) -> Echo15Compilation:
        conditions = prepared.document
        slots = []
        for slot in conditions["slots"]:
            row: dict[str, Any] = {
                "image_url": slot["image_path"],
                "image_reuses_opening_condition": slot[
                    "image_reuses_opening_condition"
                ],
            }
            if slot["audio_path"] is not None:
                row["audio_url"] = slot["audio_path"]
            else:
                row["audio_mode"] = "empty"
            slots.append(row)
        request_template: dict[str, Any] = {
            "prompt": wire.prompt_text,
            "memory_slots": slots,
        }
        opening = conditions.get("opening_condition")
        if isinstance(opening, Mapping):
            request_template["condition_img"] = opening["content_path"]
        request_template = request.apply(request_template)

        # Request payloads deliberately contain no policy/evidence identifiers.
        if any(
            forbidden in request_template
            for forbidden in ("binding_id", "evidence_id", "representation_id")
        ):
            raise RuntimeError("private identifiers leaked into the Echo request")

        public_outcomes = copy.deepcopy(conditions["binding_outcomes"])
        full_outcomes = []
        requested_by_id = conditions["requested_effects"]
        for row in public_outcomes:
            full_outcomes.append(
                {**row, "requested_effect": requested_by_id.get(row["binding_id"])}
            )

        provenance = {
            "adapter_version": ECHO15_ADAPTER_VERSION,
            "capabilities": self.capabilities.to_dict(),
            "pe_generation_policy": copy.deepcopy(self._pe_generation_policy),
            "base_prompt": {
                "artifact_id": base.artifact_id,
                "artifact_sha256": base.artifact_sha256,
                "prompt_sha256": base.prompt_sha256,
                "mode": base.to_dict()["mode"],
            },
            "wire_prompt": {
                "artifact_id": wire.artifact_id,
                "artifact_sha256": wire.artifact_sha256,
                "prompt_sha256": wire.prompt_sha256,
                "derivation": wire.derivation,
            },
            "prepared_conditions_sha256": prepared.prepared_sha256,
            "media": copy.deepcopy(conditions["media_provenance"]),
            "diagnostics": copy.deepcopy(conditions["diagnostics"]),
            "echo_request_template_sha256": sha256_json(request_template),
            "prompt_validation": strict_json_copy(
                prompt_validation, "prompt_validation"
            ),
            "private_packing": {
                "condition_img": copy.deepcopy(conditions.get("opening_condition")),
                "memory_slots": copy.deepcopy(conditions["slots"]),
            },
        }
        receipt_body = {
            "schema_version": ECHO15_COMPILATION_RECEIPT_SCHEMA_VERSION,
            "adapter_version": ECHO15_ADAPTER_VERSION,
            "pe_generation_policy": copy.deepcopy(self._pe_generation_policy),
            "request_id": conditions["request_id"],
            "raw_request_sha256": conditions["raw_request_sha256"],
            "plan_id": conditions["plan_id"],
            "plan_sha256": conditions["plan_sha256"],
            "base_prompt_artifact_id": base.artifact_id,
            "base_prompt_artifact_sha256": base.artifact_sha256,
            "base_prompt_sha256": base.prompt_sha256,
            "wire_prompt_artifact_id": wire.artifact_id,
            "wire_prompt_artifact_sha256": wire.artifact_sha256,
            "wire_prompt_sha256": wire.prompt_sha256,
            "binding_outcomes": full_outcomes,
            "compiled_conditions": copy.deepcopy(conditions["compiled_conditions"]),
            "private_backend_projection": {
                "condition_img": copy.deepcopy(conditions.get("opening_condition")),
                "memory_slots": copy.deepcopy(conditions["slots"]),
            },
            "provenance": provenance,
        }
        receipt_id = "compile-echo15-" + sha256_json(receipt_body).removeprefix(
            "sha256:"
        )[:32]
        receipt = {**receipt_body, "compilation_receipt_id": receipt_id}
        receipt["receipt_sha256"] = sha256_json(receipt)

        receipt_path = self.receipt_root / "receipts" / f"{receipt_id}.json"
        with _exclusive_lock(receipt_path.with_suffix(".lock")):
            if receipt_path.exists():
                try:
                    existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise Echo15AdapterError(
                        "receipt_corrupt", f"cannot read compilation receipt: {receipt_path}"
                    ) from exc
                if existing != receipt:
                    raise Echo15AdapterError(
                        "receipt_conflict", "compilation receipt ID has conflicting content"
                    )
            else:
                _atomic_write_json(receipt_path, receipt)

        reservation_input = {
            "request_id": conditions["request_id"],
            "raw_request_sha256": conditions["raw_request_sha256"],
            "plan_id": conditions["plan_id"],
            "plan_sha256": conditions["plan_sha256"],
            "base_prompt_artifact_id": base.artifact_id,
            "base_prompt_artifact_sha256": base.artifact_sha256,
            "wire_prompt_artifact_id": wire.artifact_id,
            "wire_prompt_artifact_sha256": wire.artifact_sha256,
            "compilation_receipt_id": receipt_id,
            "compilation_receipt_sha256": receipt["receipt_sha256"],
            "binding_outcomes": public_outcomes,
            "compiled_conditions": copy.deepcopy(conditions["compiled_conditions"]),
            **({"generator_prompt_sha256": wire.prompt_sha256, "generator_seed": request.seed}
               if prompt_authoring else {}),
        }
        document = {
            "adapter_version": ECHO15_ADAPTER_VERSION,
            "reservation_input": reservation_input,
            "echo_request_template": request_template,
            "compilation_receipt": receipt,
            "compilation_receipt_path": str(receipt_path.resolve()),
            "provenance": provenance,
        }
        return Echo15Compilation.create(document=document, base=base, wire=wire)

    def compile(
        self,
        *,
        state: Any,
        raw_prompt: str,
        request: Echo15RequestSpec,
        evidence_plan: Mapping[str, Any] | None = None,
        pe_mode: str = "gemini_rewrite",
    ) -> Echo15Compilation:
        """Compile raw intent plus portable evidence into one native Echo request.

        PE is an internal adapter stage, not a Memory Policy tool or a
        caller-orchestrated pre-step.  Its Gemini client uses an isolated fresh
        turn for every attempt.  The base semantic artifact is locked once per
        request; condition revisions reuse it while the adapter recompiles the
        active condition layout.
        """

        if not isinstance(request, Echo15RequestSpec):
            raise TypeError("request must be Echo15RequestSpec")
        if pe_mode not in {"gemini_rewrite", "passthrough", "policy_direct"}:
            raise ValueError("unsupported prompt mode")
        raw_prompt = _require_text(raw_prompt, "raw_prompt")
        data = _as_state_data(state)
        prompt_authoring = (data.get("policy_limits") or {}).get("prompt_authoring") is True
        if prompt_authoring and (pe_mode not in {"gemini_rewrite", "policy_direct"} or request.seed is None):
            raise Echo15AdapterError("prompt_authoring_contract", "prompt authoring requires an authored-prompt mode and an explicit host-owned seed")
        if pe_mode == "policy_direct" and not prompt_authoring:
            raise Echo15AdapterError("prompt_authoring_contract", "policy_direct requires prompt authoring")
        if sha256_text(raw_prompt) != data.get("shot_requirement_sha256"):
            raise Echo15AdapterError(
                "raw_prompt_mismatch", "raw_prompt differs from the immutable request"
        )
        plan = self._conditions._accepted_plan(data, evidence_plan)
        if pe_mode == "policy_direct":
            return self._compile_policy_direct(data=data, plan=plan, raw_prompt=raw_prompt, request=request)
        prepared = self._conditions.prepare(state=data, evidence_plan=plan)
        condition_context = self._stable_pe_context(data=data, plan=plan)
        expected_subject_ids = {
            str(row.get("canonical_id"))
            for row in condition_context.get("canonical_cast") or []
            if isinstance(row, Mapping)
            and re.fullmatch(r"ID_[A-Z][A-Z0-9_]*", str(row.get("canonical_id") or ""))
        }
        registry = data.get("registry")
        registry_items = (
            registry.get("items") if isinstance(registry, Mapping) else {}
        )
        registry_items = registry_items if isinstance(registry_items, Mapping) else {}
        # An ID_X supplied without a registry row is already a canonical
        # protocol-level human identifier.  Once a registry row exists, its
        # source-derived taxonomy is authoritative; this prevents a legacy
        # robot stored under an ID-shaped key from increasing the person count.
        expected_subject_ids.update(
            str(subject_id)
            for requirement in plan.get("requirements") or []
            if isinstance(requirement, Mapping)
            for subject_id in requirement.get("subject_ids") or []
            if isinstance(subject_id, str)
            and re.fullmatch(r"ID_[A-Z][A-Z0-9_]*", subject_id)
            and subject_id not in registry_items
        )
        semantic_contract = build_semantic_preservation_contract(
            raw_prompt,
            expected_subject_ids=sorted(expected_subject_ids),
            semantic_requirements=[
                *(condition_context.get("immutable_continuity_requirements") or []),
                # PE sees these source-derived cast facts. Permit them without
                # turning every descriptor into a mandatory new caption fact.
                *({"description": row["stable_description"], "required": False,
                   "semantic_roles": ["identity", "appearance", "wardrobe"]}
                  for row in condition_context.get("canonical_cast") or []
                  if isinstance(row, Mapping) and row.get("stable_description")),
            ],
        )
        caption_language, previous_prompt = self._story_pe_inputs(
            data=data,
            request=request,
            raw_prompt=raw_prompt,
            pe_mode=pe_mode,
        )
        if self._strict_v7 and pe_mode == "gemini_rewrite" and (
            caption_language not in _CAPTION_LANGUAGES
        ):
            raise Echo15AdapterError(
                "caption_language_required",
                "official v7 adapter could not establish a caption-language lock",
            )
        pe_input_text = (
            self._build_internal_pe_input(
                raw_prompt=raw_prompt,
                caption_language=caption_language,
                previous_prompt=previous_prompt,
                condition_context=condition_context,
                semantic_contract=semantic_contract,
            )
            if pe_mode == "gemini_rewrite"
            else None
        )
        if prompt_authoring:
            draft = _require_text(plan.get("generation_prompt"), "generation_prompt")
            pe_input_text += "\nPOLICY-AUTHORED FULL VIDEO DRAFT (subordinate to the immutable raw user intent and semantic contract; enhance this draft, never emit tool IDs):\n" + draft
        output_validator = (
            _OfficialV7PromptValidator(
                str(caption_language), semantic_contract
            )
            if self._strict_v7 and pe_mode == "gemini_rewrite"
            else None
        )
        prompt_engineer = AdapterPromptEngineer(
            root=self._prompt_artifact_root,
            gemini=self._gemini,
            output_validator=output_validator,
            max_generation_attempts=self._max_pe_generation_attempts,
        )
        origin = self._default_origin_projection(plan)
        base = self._ensure_base_prompt(
            data=data,
            prompt_engineer=prompt_engineer,
            plan=plan,
            raw_prompt=raw_prompt,
            pe_mode=pe_mode,
            pe_input_text=pe_input_text,
            caption_language=caption_language,
            origin_plan_projection=origin,
            initial_condition_projection=prepared.wire_projection,
        )
        if not prompt_authoring:
            self._commit_story_pe_context(
                data=data, request=request, raw_prompt=raw_prompt, pe_mode=pe_mode, base=base,
            )
        if pe_mode == "gemini_rewrite" and prepared.opening_condition is not None:
            opening = prepared.opening_condition
            opening_profile = self._opening_profiles.get(opening.semantic_effect)
            if opening_profile is None:
                raise Echo15AdapterError(
                    "opening_profile_missing",
                    "a semantic-effect-specific opening PE profile is required",
                )
        else:
            # The passthrough cohort never rewrites user text, including when
            # an opening image is present.
            opening = None
            opening_profile = None
        wire = prompt_engineer.compile_wire(
            base=base,
            condition_projection=prepared.wire_projection,
            opening=opening,
            opening_profile=opening_profile,
        )
        prompt_validation = self._validate_wire(
            base=base,
            wire=wire,
            prepared=prepared,
            pe_mode=pe_mode,
            raw_prompt=raw_prompt,
            strict_v7=self._strict_v7 and pe_mode == "gemini_rewrite",
            caption_language=caption_language,
            semantic_contract=(
                semantic_contract
                if self._strict_v7 and pe_mode == "gemini_rewrite"
                else None
            ),
        )
        return self._finalize(
            prepared=prepared,
            base=base,
            wire=wire,
            request=request,
            prompt_validation=prompt_validation,
            prompt_authoring=prompt_authoring,
        )

    def commit_selected_prompt(self, *, state: Any, request: Echo15RequestSpec,
                               raw_prompt: str, base_prompt_artifact: Mapping[str, Any]) -> None:
        if base_prompt_artifact.get("mode") == "policy_direct":
            return  # Direct prompts need no cross-shot PE context.
        data = _as_state_data(state)
        if data.get("phase") != "complete" or not data.get("selected_generation"):
            raise PromptArtifactConflict("only a finalized selected generation can advance story PE context")
        base = BasePromptArtifact.from_dict(base_prompt_artifact)
        if base.raw_prompt_sha256 != sha256_text(raw_prompt):
            raise PromptArtifactConflict("selected prompt does not belong to this raw request")
        self._commit_story_pe_context(data=data, request=request, raw_prompt=raw_prompt,
                                      pe_mode="gemini_rewrite", base=base)


__all__ = [
    "ECHO15_ADAPTER_VERSION",
    "ECHO15_COMPILATION_RECEIPT_SCHEMA_VERSION",
    "ECHO15_CONDITION_PROJECTION_SCHEMA_VERSION",
    "ECHO15_MAX_PE_GENERATION_ATTEMPTS",
    "Echo15AdapterError",
    "Echo15BackendAdapter",
    "Echo15Capabilities",
    "Echo15Compilation",
    "Echo15RequestSpec",
    "official_v7_profile_manifest",
]
