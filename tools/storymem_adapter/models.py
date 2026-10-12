"""Immutable, provider-private prompt artifacts used by backend adapters."""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_PROMPT_ARTIFACT_SCHEMA_VERSION = "storymem.adapter.prompt.base.v1"
WIRE_PROMPT_ARTIFACT_SCHEMA_VERSION = "storymem.adapter.prompt.wire.v1"
OPENING_REWRITE_CACHE_SCHEMA_VERSION = "storymem.adapter.prompt.opening-cache.v1"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MODES = frozenset({"gemini_rewrite", "passthrough", "policy_direct"})
_OPENING_EFFECTS = frozenset({"exact_opening_state", "temporal_continuation"})


def canonical_json(value: Any) -> str:
    """Serialize strict JSON deterministically, rejecting NaN and infinities."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def require_sha256(value: Any, field_name: str) -> str:
    text = str(value or "")
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def strict_json_copy(value: Any, field_name: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain finite JSON data") from exc


@dataclass(frozen=True, slots=True)
class PromptOutputContract:
    """Mechanical checks applied to a profile's text-only response.

    Backend-specific semantic rules remain in the profile itself.  These
    checks only prevent common transport/format failures from being committed
    as immutable artifacts.
    """

    single_paragraph: bool = True
    min_characters: int = 1
    max_characters: int | None = None
    forbidden_prefixes: tuple[str, ...] = ("```", "# ")
    required_any_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "forbidden_prefixes", tuple(self.forbidden_prefixes))
        object.__setattr__(
            self, "required_any_prefixes", tuple(self.required_any_prefixes)
        )
        if type(self.single_paragraph) is not bool:
            raise ValueError("single_paragraph must be boolean")
        if type(self.min_characters) is not int or self.min_characters < 1:
            raise ValueError("min_characters must be a positive integer")
        if self.max_characters is not None and (
            type(self.max_characters) is not int
            or self.max_characters < self.min_characters
        ):
            raise ValueError("max_characters must be null or at least min_characters")
        for name, values in (
            ("forbidden_prefixes", self.forbidden_prefixes),
            ("required_any_prefixes", self.required_any_prefixes),
        ):
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{name} must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "single_paragraph": self.single_paragraph,
            "min_characters": self.min_characters,
            "max_characters": self.max_characters,
            "forbidden_prefixes": list(self.forbidden_prefixes),
            "required_any_prefixes": list(self.required_any_prefixes),
        }


@dataclass(frozen=True, slots=True)
class PromptEngineeringProfile:
    """One exact backend-owned PE system instruction and its audit manifest."""

    profile_id: str
    system_instruction: str
    source_manifest_json: str
    output_contract: PromptOutputContract = field(default_factory=PromptOutputContract)
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be non-empty")
        if not isinstance(self.system_instruction, str) or not self.system_instruction:
            raise ValueError("system_instruction must be non-empty")
        manifest = strict_json_copy(
            json.loads(self.source_manifest_json), "source_manifest"
        )
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("source_manifest must be a non-empty array")
        for index, item in enumerate(manifest):
            if not isinstance(item, dict) or set(item) != {
                "source_id",
                "sha256",
                "size_bytes",
            }:
                raise ValueError(f"source_manifest[{index}] is malformed")
            if not isinstance(item["source_id"], str) or not item["source_id"]:
                raise ValueError(f"source_manifest[{index}].source_id is invalid")
            require_sha256(item["sha256"], f"source_manifest[{index}].sha256")
            if type(item["size_bytes"]) is not int or item["size_bytes"] < 1:
                raise ValueError(f"source_manifest[{index}].size_bytes is invalid")
        canonical_manifest = canonical_json(manifest)
        object.__setattr__(self, "source_manifest_json", canonical_manifest)
        expected = sha256_json(
            {
                "profile_id": self.profile_id,
                "system_instruction": self.system_instruction,
                "source_manifest": manifest,
                "output_contract": self.output_contract.to_dict(),
            }
        )
        if self.profile_sha256 and self.profile_sha256 != expected:
            raise ValueError("profile_sha256 does not match the exact PE profile")
        object.__setattr__(self, "profile_sha256", expected)

    @classmethod
    def from_sources(
        cls,
        *,
        profile_id: str,
        sources: Sequence[tuple[str, str | Path]],
        output_contract: PromptOutputContract | None = None,
        separator: str = "\n\n",
    ) -> "PromptEngineeringProfile":
        """Load and hash complete profile files without backend assumptions.

        For the official opening-frame rewrite, callers pass the ordinary shot
        PE reference first and the tail-frame skill second.  Their byte-exact
        contents then form one isolated Gemini system instruction.
        """

        if not sources:
            raise ValueError("sources must not be empty")
        texts: list[str] = []
        manifest: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source_id, raw_path in sources:
            if not isinstance(source_id, str) or not source_id or source_id in seen:
                raise ValueError("source IDs must be non-empty and unique")
            path = Path(raw_path).expanduser().resolve()
            payload = path.read_bytes()
            if not payload:
                raise ValueError(f"PE source is empty: {path}")
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"PE source is not UTF-8: {path}") from exc
            seen.add(source_id)
            texts.append(text)
            manifest.append(
                {
                    "source_id": source_id,
                    "sha256": sha256_bytes(payload),
                    "size_bytes": len(payload),
                }
            )
        return cls(
            profile_id=profile_id,
            system_instruction=separator.join(texts),
            source_manifest_json=canonical_json(manifest),
            output_contract=output_contract or PromptOutputContract(),
        )

    @classmethod
    def from_text(
        cls,
        *,
        profile_id: str,
        system_instruction: str,
        output_contract: PromptOutputContract | None = None,
        source_id: str = "inline",
    ) -> "PromptEngineeringProfile":
        payload = system_instruction.encode("utf-8")
        return cls(
            profile_id=profile_id,
            system_instruction=system_instruction,
            source_manifest_json=canonical_json(
                [
                    {
                        "source_id": source_id,
                        "sha256": sha256_bytes(payload),
                        "size_bytes": len(payload),
                    }
                ]
            ),
            output_contract=output_contract or PromptOutputContract(),
        )

    @property
    def source_manifest(self) -> list[dict[str, Any]]:
        return copy.deepcopy(json.loads(self.source_manifest_json))

    def manifest(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "system_instruction_sha256": sha256_text(self.system_instruction),
            "sources": self.source_manifest,
            "output_contract": self.output_contract.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BasePromptRequest:
    """The frozen semantic input used for exactly one shot-level base PE call."""

    scope_id: str
    raw_prompt: str
    origin_plan_projection: Any
    initial_condition_projection: Any
    pe_input_text: str | None = None
    mode: str = "gemini_rewrite"
    expected_caption_language: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope_id, str) or not self.scope_id.strip():
            raise ValueError("scope_id must be non-empty")
        if not isinstance(self.raw_prompt, str) or not self.raw_prompt:
            raise ValueError("raw_prompt must be non-empty")
        if self.mode not in _MODES:
            raise ValueError(f"unsupported PE mode: {self.mode!r}")
        if self.expected_caption_language not in {
            None,
            "English",
            "Simplified Chinese",
        }:
            raise ValueError(
                "expected_caption_language must be English, Simplified Chinese, or null"
            )
        if self.pe_input_text is not None:
            if not isinstance(self.pe_input_text, str) or not self.pe_input_text:
                raise ValueError("pe_input_text must be null or non-empty")
            if self.mode != "policy_direct" and self.raw_prompt not in self.pe_input_text:
                raise ValueError("pe_input_text must contain raw_prompt byte-for-byte")
        if self.mode == "policy_direct" and self.pe_input_text is None:
            raise ValueError("policy_direct requires the policy-authored prompt")
        strict_json_copy(self.origin_plan_projection, "origin_plan_projection")
        strict_json_copy(
            self.initial_condition_projection, "initial_condition_projection"
        )

    @property
    def exact_pe_input_text(self) -> str:
        return self.pe_input_text if self.pe_input_text is not None else self.raw_prompt

    def fingerprint(
        self,
        profile: PromptEngineeringProfile | None,
        *,
        output_validator_id: str = "mechanical-only.v1",
    ) -> str:
        if not isinstance(output_validator_id, str) or not output_validator_id:
            raise ValueError("output_validator_id must be non-empty")
        profile_hash = profile.profile_sha256 if profile is not None else None
        return sha256_json(
            {
                "scope_id": self.scope_id,
                "raw_prompt": self.raw_prompt,
                "pe_input_text": self.exact_pe_input_text,
                "origin_plan_projection": strict_json_copy(
                    self.origin_plan_projection, "origin_plan_projection"
                ),
                "initial_condition_projection": strict_json_copy(
                    self.initial_condition_projection,
                    "initial_condition_projection",
                ),
                "mode": self.mode,
                "expected_caption_language": self.expected_caption_language,
                "profile_sha256": profile_hash,
                "output_validator_id": output_validator_id,
            }
        )


@dataclass(frozen=True, slots=True)
class OpeningCondition:
    """An actual image whose bytes must be treated as generator frame zero."""

    condition_id: str
    image_path: str | Path
    semantic_effect: str
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, str) or not self.condition_id:
            raise ValueError("condition_id must be non-empty")
        if self.semantic_effect not in _OPENING_EFFECTS:
            raise ValueError(
                "opening semantic_effect must be exact_opening_state or "
                "temporal_continuation"
            )

    def read(self) -> tuple[Path, str, bytes, str]:
        path = Path(self.image_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"opening condition image is missing: {path}")
        payload = path.read_bytes()
        if not payload:
            raise ValueError(f"opening condition image is empty: {path}")
        mime_type = self.mime_type or mimetypes.guess_type(path.name)[0] or ""
        if not mime_type.startswith("image/"):
            raise ValueError("opening condition must be an image")
        return path, mime_type, payload, sha256_bytes(payload)


class _ImmutableArtifact:
    """A validated canonical JSON document with no mutable nested references."""

    _document_json: str

    @property
    def document(self) -> dict[str, Any]:
        return copy.deepcopy(json.loads(self._document_json))

    def to_dict(self) -> dict[str, Any]:
        return self.document

    def _get(self, *keys: str) -> Any:
        value: Any = json.loads(self._document_json)
        for key in keys:
            value = value[key]
        return value

    @property
    def artifact_id(self) -> str:
        return str(self._get("artifact_id"))

    @property
    def artifact_sha256(self) -> str:
        return str(self._get("artifact_sha256"))

    @property
    def scope_id(self) -> str:
        return str(self._get("scope_id"))

    @property
    def prompt_text(self) -> str:
        return str(self._get("prompt", "text"))

    @property
    def prompt_sha256(self) -> str:
        return str(self._get("prompt", "sha256"))

    @property
    def request_fingerprint(self) -> str:
        return str(self._get("request_fingerprint"))


@dataclass(frozen=True, slots=True)
class BasePromptArtifact(_ImmutableArtifact):
    _document_json: str

    def __post_init__(self) -> None:
        document = json.loads(self._document_json)
        _validate_artifact_document(
            document,
            expected_schema=BASE_PROMPT_ARTIFACT_SCHEMA_VERSION,
            expected_layer="base",
        )
        object.__setattr__(self, "_document_json", canonical_json(document))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BasePromptArtifact":
        return cls(canonical_json(strict_json_copy(value, "base prompt artifact")))

    @property
    def raw_prompt_sha256(self) -> str:
        return str(self._get("lineage", "raw_prompt_sha256"))

    @property
    def origin_plan_projection_sha256(self) -> str:
        return str(self._get("lineage", "origin_plan_projection_sha256"))


@dataclass(frozen=True, slots=True)
class WirePromptArtifact(_ImmutableArtifact):
    _document_json: str

    def __post_init__(self) -> None:
        document = json.loads(self._document_json)
        _validate_artifact_document(
            document,
            expected_schema=WIRE_PROMPT_ARTIFACT_SCHEMA_VERSION,
            expected_layer="wire",
        )
        if not isinstance(document.get("base_artifact_id"), str) or not document[
            "base_artifact_id"
        ]:
            raise ValueError("wire base_artifact_id is invalid")
        if document.get("derivation") not in {"identity", "opening_image_rewrite"}:
            raise ValueError("wire derivation is invalid")
        object.__setattr__(self, "_document_json", canonical_json(document))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WirePromptArtifact":
        return cls(canonical_json(strict_json_copy(value, "wire prompt artifact")))

    @property
    def base_artifact_id(self) -> str:
        return str(self._get("base_artifact_id"))

    @property
    def derivation(self) -> str:
        return str(self._get("derivation"))


def seal_artifact(document: Mapping[str, Any]) -> dict[str, Any]:
    value = strict_json_copy(document, "artifact")
    value.pop("artifact_sha256", None)
    value["artifact_sha256"] = sha256_json(value)
    return value


def _validate_artifact_document(
    document: Any, *, expected_schema: str, expected_layer: str
) -> None:
    if not isinstance(document, dict):
        raise ValueError("prompt artifact must be an object")
    if document.get("schema_version") != expected_schema:
        raise ValueError("prompt artifact schema version is invalid")
    if document.get("layer") != expected_layer:
        raise ValueError("prompt artifact layer is invalid")
    for field_name in ("artifact_id", "scope_id", "request_fingerprint", "created_at"):
        if not isinstance(document.get(field_name), str) or not document[field_name]:
            raise ValueError(f"prompt artifact {field_name} is invalid")
    require_sha256(document["request_fingerprint"], "request_fingerprint")
    prompt = document.get("prompt")
    if not isinstance(prompt, dict) or set(prompt) != {"text", "sha256"}:
        raise ValueError("prompt artifact prompt is malformed")
    if not isinstance(prompt["text"], str) or not prompt["text"]:
        raise ValueError("prompt artifact text is empty")
    if prompt["sha256"] != sha256_text(prompt["text"]):
        raise ValueError("prompt artifact text hash is invalid")
    stored_hash = require_sha256(document.get("artifact_sha256"), "artifact_sha256")
    unhashed = copy.deepcopy(document)
    unhashed.pop("artifact_sha256", None)
    if stored_hash != sha256_json(unhashed):
        raise ValueError("prompt artifact integrity hash is invalid")


__all__ = [
    "BASE_PROMPT_ARTIFACT_SCHEMA_VERSION",
    "OPENING_REWRITE_CACHE_SCHEMA_VERSION",
    "WIRE_PROMPT_ARTIFACT_SCHEMA_VERSION",
    "BasePromptArtifact",
    "BasePromptRequest",
    "OpeningCondition",
    "PromptEngineeringProfile",
    "PromptOutputContract",
    "WirePromptArtifact",
    "canonical_json",
    "require_sha256",
    "seal_artifact",
    "sha256_bytes",
    "sha256_json",
    "sha256_text",
    "strict_json_copy",
]
