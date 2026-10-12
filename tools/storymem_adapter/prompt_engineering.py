"""Crash-safe two-layer prompt engineering owned by a backend adapter."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

from storymem_teacher.providers import ProviderError

from .gemini import (
    GeminiEmptyOutput,
    GeminiInlineImage,
    GeminiOutputTruncated,
    GeminiTextResult,
)
from .models import (
    BASE_PROMPT_ARTIFACT_SCHEMA_VERSION,
    OPENING_REWRITE_CACHE_SCHEMA_VERSION,
    WIRE_PROMPT_ARTIFACT_SCHEMA_VERSION,
    BasePromptArtifact,
    BasePromptRequest,
    OpeningCondition,
    PromptEngineeringProfile,
    PromptOutputContract,
    WirePromptArtifact,
    canonical_json,
    seal_artifact,
    require_sha256,
    sha256_json,
    sha256_text,
    strict_json_copy,
)
from .v7_validation import V7PromptValidationError


PE_GENERATION_STRATEGY = "fresh_isolated_turn_bounded.v5"
VALIDATION_VIOLATION_SCHEMA_VERSION = "storymem.pe-validation-violations.v1"
PE_RETRYABLE_FINISH_REASONS = ("MAX_TOKENS",)
PE_RETRYABLE_FAILURE_KINDS = (
    "gemini_output_truncated",
    "gemini_empty_output",
)
DEFAULT_MAX_PE_GENERATION_ATTEMPTS = 5


def pe_generation_policy_manifest(max_generation_attempts: int) -> dict[str, Any]:
    """Return the semantic retry policy sealed into artifacts and receipts."""

    if type(max_generation_attempts) is not int or max_generation_attempts < 1:
        raise ValueError("max_generation_attempts must be a positive integer")
    return {
        "strategy": PE_GENERATION_STRATEGY,
        "max_generation_attempts": max_generation_attempts,
        "retryable_provider": "gemini",
        "retryable_finish_reasons": list(PE_RETRYABLE_FINISH_REASONS),
        "retryable_failure_kinds": list(PE_RETRYABLE_FAILURE_KINDS),
        "semantic_validation_uses_fresh_turn": True,
        "encoder_length_failure": "reject_and_regenerate_original_input_without_draft",
        "encoder_length_compression": False,
        "provider_error_fallback": "fail_closed",
        "passthrough_fallback": False,
    }


class PromptEngineeringError(RuntimeError):
    pass


class PromptArtifactConflict(PromptEngineeringError):
    """A caller tried to mutate an already locked shot-level artifact."""


class PromptArtifactCorrupt(PromptEngineeringError):
    """A durable artifact cannot be trusted and will not be overwritten."""


class PromptOutputRejected(PromptEngineeringError):
    """Gemini returned text that violates the selected profile's wire contract.

    ``violations`` is deliberately machine-readable.  The provider still sees
    the human-readable exception text during a bounded correction, while the
    collector can persist only the allowlisted codes and concept names needed
    to diagnose a failed run without retaining provider output.
    """

    def __init__(
        self,
        message: str,
        *,
        violations: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        normalized: list[dict[str, Any]] = []
        for value in violations:
            if not isinstance(value, Mapping):
                continue
            code = value.get("code")
            if not isinstance(code, str) or not code.strip():
                continue
            try:
                row = strict_json_copy(dict(value), "validation violation")
            except (TypeError, ValueError):
                continue
            row["code"] = code.strip()
            normalized.append(row)
        if not normalized:
            normalized = [{"code": "validation.rejected"}]
        self.violation_schema_version = VALIDATION_VIOLATION_SCHEMA_VERSION
        self.violations = normalized
        self.violation_codes = list(
            dict.fromkeys(str(row["code"]) for row in normalized)
        )


class GeminiTextPort(Protocol):
    def generate_text(
        self,
        *,
        system_instruction: str,
        user_text: str,
        images: Sequence[GeminiInlineImage] = (),
    ) -> GeminiTextResult: ...


class PromptOutputValidator(Protocol):
    validation_id: str

    def validate(
        self,
        text: str,
        *,
        raw_prompt: str,
        condition_effect: str | None,
        expected_caption_language: str | None,
    ) -> Mapping[str, Any]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_component(value: str) -> str:
    readable = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    ).strip("_")[:48]
    suffix = sha256_text(value).removeprefix("sha256:")[:20]
    return f"{readable or 'shot'}-{suffix}"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
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


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromptArtifactCorrupt(f"cannot read durable artifact {path}") from exc
    if not isinstance(value, dict):
        raise PromptArtifactCorrupt(f"durable artifact is not an object: {path}")
    return value


def _validate_output(text: str, contract: PromptOutputContract) -> None:
    if not isinstance(text, str) or not text:
        raise PromptOutputRejected(
            "PE output is empty",
            violations=[{"code": "output.empty"}],
        )
    if text != text.strip():
        raise PromptOutputRejected(
            "PE output has leading or trailing whitespace",
            violations=[{"code": "format.outer_whitespace"}],
        )
    if contract.single_paragraph and ("\n" in text or "\r" in text):
        raise PromptOutputRejected(
            "PE output is not one paragraph",
            violations=[{"code": "format.single_paragraph"}],
        )
    if len(text) < contract.min_characters:
        raise PromptOutputRejected(
            "PE output has "
            f"{len(text)} characters, below the profile minimum of "
            f"{contract.min_characters}",
            violations=[
                {
                    "code": "length.below_minimum",
                    "actual_count": len(text),
                    "expected_minimum": contract.min_characters,
                }
            ],
        )
    if contract.max_characters is not None and len(text) > contract.max_characters:
        raise PromptOutputRejected(
            "PE output has "
            f"{len(text)} characters, exceeding the profile hard limit of "
            f"{contract.max_characters}",
            violations=[
                {
                    "code": "length.above_maximum",
                    "actual_count": len(text),
                    "expected_maximum": contract.max_characters,
                }
            ],
        )
    if any(text.startswith(prefix) for prefix in contract.forbidden_prefixes):
        raise PromptOutputRejected(
            "PE output starts with forbidden wrapper text",
            violations=[{"code": "format.forbidden_prefix"}],
        )
    if contract.required_any_prefixes and not any(
        text.startswith(prefix) for prefix in contract.required_any_prefixes
    ):
        raise PromptOutputRejected(
            "PE output lacks a required profile prefix",
            violations=[{"code": "format.required_prefix_missing"}],
        )


def _canonicalize_provider_output(text: Any) -> tuple[str, dict[str, Any]]:
    """Apply the official deterministic whitespace-only PE normalization.

    The unmodified model text remains in provider provenance.  This operation
    neither invents tokens nor asks the model for a correction; it only maps
    runs of Unicode whitespace to one ASCII space so harmless response framing
    cannot poison a large batch.
    """

    if not isinstance(text, str):
        raise PromptOutputRejected(
            "PE output is not text",
            violations=[{"code": "output.not_text"}],
        )
    normalized = " ".join(text.split()).strip()
    return normalized, {
        "method": "unicode_whitespace_collapse.v1",
        "changed": normalized != text,
        "raw_text_sha256": sha256_text(text),
        "normalized_text_sha256": sha256_text(normalized),
    }


def _correction_user_turn(
    *,
    original_user_turn: str,
    rejected_text: str,
    validation_errors: Sequence[str],
    validation_violations: Sequence[Mapping[str, Any]],
    validation_context: Mapping[str, Any] | None,
    output_contract: PromptOutputContract,
) -> str:
    context = (
        "\nLOCKED VALIDATION CONTEXT:\n"
        + canonical_json(validation_context)
        if validation_context
        else ""
    )
    actual_characters = len(rejected_text)
    effective_minimum = output_contract.min_characters
    effective_maximum = output_contract.max_characters
    required_range = (
        validation_context.get("required_character_range")
        if validation_context
        else None
    )
    if (
        isinstance(required_range, Sequence)
        and not isinstance(required_range, (str, bytes, bytearray))
        and len(required_range) == 2
        and all(type(value) is int for value in required_range)
        and required_range[0] >= 1
        and required_range[1] >= required_range[0]
    ):
        effective_minimum = max(effective_minimum, required_range[0])
        effective_maximum = (
            required_range[1]
            if effective_maximum is None
            else min(effective_maximum, required_range[1])
        )
    if effective_maximum is not None and effective_maximum < effective_minimum:
        raise PromptEngineeringError(
            "output contract and validator character ranges do not intersect"
        )
    # Prefer the conservative production window used by official v7, but
    # intersect it with the active profile instead of making a backend-generic
    # prompt engineer impossible to retry for smaller or larger legal ranges.
    target_minimum = max(effective_minimum, 1550)
    target_maximum = (
        1750
        if effective_maximum is None
        else min(effective_maximum, 1750)
    )
    if target_minimum > target_maximum:
        target_minimum = effective_minimum
        target_maximum = (
            effective_maximum
            if effective_maximum is not None
            else max(effective_minimum + 200, 1750)
        )
    hard_maximum = (
        str(effective_maximum) if effective_maximum is not None else "unbounded"
    )
    unique_errors = list(
        dict.fromkeys(str(item) for item in validation_errors if str(item))
    )
    failures = "\n".join(
        f"{index}. {message}" for index, message in enumerate(unique_errors, start=1)
    )
    structured_violations = [
        strict_json_copy(dict(item), "validation violation")
        for item in validation_violations
        if isinstance(item, Mapping)
    ]
    violation_block = canonical_json(
        {
            "schema_version": VALIDATION_VIOLATION_SCHEMA_VERSION,
            "current_violations": structured_violations,
        }
    )
    dialogue_mode = (
        str(validation_context.get("dialogue_mode") or "")
        if validation_context
        else ""
    )
    if dialogue_mode == "required_unspecified":
        dialogue_resolution = (
            "Dialogue mode is required_unspecified. A source phrase saying that no "
            "exact dialogue was supplied means there is no verbatim line to preserve; "
            "it does not forbid speech. Compose exactly one brief, natural quoted line "
            "for a visible canonical-ID speaker and include the required vocal "
            "declarations. Do not infer words from reference audio."
        )
    elif dialogue_mode == "exact":
        dialogue_resolution = (
            "Dialogue mode is exact. Preserve only the locked exact line or lines and "
            "the required vocal declarations."
        )
    elif dialogue_mode == "forbidden":
        dialogue_resolution = (
            "Dialogue mode is forbidden. Include no quoted speech or voice anchor and "
            "use the required silence declarations."
        )
    else:
        dialogue_resolution = (
            "Follow the dialogue mode in the locked validation context exactly."
        )
    length_instruction = (
        "\nLENGTH TARGET (mandatory on every correction):\n"
        f"- The rejected draft is {actual_characters} Unicode characters.\n"
        f"- Rewrite the complete prompt to {target_minimum}-{target_maximum} "
        "characters inclusive.\n"
        f"- The absolute permitted range is {effective_minimum}-{hard_maximum} "
        "characters.\n"
        "- Reach the target by compositionally rewriting secondary description "
        "while preserving the shot beat, required identity and clothing anchors, "
        "dialogue, condition relationship, and fixed safety declarations. Never "
        "slice, mechanically truncate, or leave an incomplete sentence or segment.\n"
    )
    encoder_budget = (validation_context or {}).get("encoder_budget")
    if isinstance(encoder_budget, Mapping):
        # Semantic repairs retain the original encoder budget. In particular,
        # never expand a Chinese caption to the legacy 1550-character target.
        length_instruction = (
            "\nINITIAL ENCODER BUDGET (also applies to this semantic repair):\n"
            f"- The COMPLETE caption must fit {encoder_budget['max_prompt_tokens']} "
            "Echo Gemma tokens including special tokens, labels, anchors, dialogue, "
            "required declarations and any opening-frame preamble.\n"
            f"- The permitted character range is {effective_minimum}-{hard_maximum}. "
            "Do not add prose just to reach a larger character target.\n"
            "- Plan within both bounds before writing; preserve every required "
            "meaning and return one complete caption without truncation.\n"
        )
    return (
        "This is a fresh bounded correction request. Follow the same system "
        "contract and return only the complete corrected prompt. Preserve the "
        "original shot intent. Fix every deterministic validation failure below; "
        "do not discuss the correction. This instruction wrapper is not evidence "
        "for choosing the output language."
        f"{context}\n\n"
        f"CUMULATIVE VALIDATION FAILURES (fix all of them):\n{failures}\n\n"
        f"CURRENT STRUCTURED VIOLATIONS:\n{violation_block}\n\n"
        f"{length_instruction}"
        f"ORIGINAL INPUT:\n{original_user_turn}\n\n"
        f"REJECTED DRAFT:\n{rejected_text}\n\n"
        "FINAL CORRECTION AUTHORITY (applies after all source and draft text):\n"
        f"{dialogue_resolution} Use the locked validation checklist above, fix every "
        "current structured violation, and return only one complete corrected prompt."
    )


def _validate_gemini_result(
    result: GeminiTextResult,
    *,
    profile: PromptEngineeringProfile,
    user_text: str,
    images: Sequence[GeminiInlineImage],
) -> None:
    """Bind a provider result to the exact isolated request we authorized."""

    manifest = result.request_manifest
    provenance = result.provenance
    try:
        require_sha256(
            manifest.get("request_payload_sha256"), "request_payload_sha256"
        )
        if manifest.get("system_instruction_sha256") != sha256_text(
            profile.system_instruction
        ):
            raise ValueError("system instruction hash")
        if manifest.get("user_turn_count") != 1:
            raise ValueError("user turn count")
        expected_parts = [
            {
                "kind": "text",
                "sha256": sha256_text(user_text),
                "size_bytes": len(user_text.encode("utf-8")),
            },
            *[
                {
                    "kind": "inline_image",
                    "mime_type": image.mime_type,
                    "sha256": image.content_sha256,
                    "size_bytes": len(image.data),
                }
                for image in images
            ],
        ]
        if manifest.get("parts") != expected_parts:
            raise ValueError("request parts")
        if provenance.get("request_payload_sha256") != manifest.get(
            "request_payload_sha256"
        ):
            raise ValueError("request payload hash")
        require_sha256(provenance.get("response_sha256"), "response_sha256")
        if provenance.get("assistant_text_sha256") != sha256_text(result.text):
            raise ValueError("assistant text hash")
        if provenance.get("raw_assistant_text") != result.text:
            raise ValueError("raw assistant text")
        if provenance.get("conversation_reused") is not False:
            raise ValueError("conversation isolation")
        if provenance.get("user_turn_count") != 1:
            raise ValueError("provenance user turn count")
    except (AttributeError, TypeError, ValueError) as exc:
        raise PromptEngineeringError(
            "Gemini result is not bound to the authorized isolated PE request"
        ) from exc


def _failure_attempt_record(
    exc: ProviderError,
    *,
    attempt_number: int,
    profile: PromptEngineeringProfile,
    user_text: str,
    images: Sequence[GeminiInlineImage],
    retryable_kind: str | None,
) -> dict[str, Any]:
    """Validate and retain the request identity of one failed provider call."""

    if retryable_kind not in {None, "truncation", "empty_output"}:
        raise ValueError(f"unsupported retryable Gemini failure: {retryable_kind!r}")

    provenance = strict_json_copy(exc.provenance, "Gemini failure provenance")
    manifest = provenance.get("request_manifest")
    try:
        if not isinstance(manifest, Mapping):
            raise ValueError("request manifest")
        request_manifest = strict_json_copy(manifest, "failed Gemini request manifest")
        require_sha256(
            request_manifest.get("request_payload_sha256"),
            "request_payload_sha256",
        )
        if request_manifest.get("system_instruction_sha256") != sha256_text(
            profile.system_instruction
        ):
            raise ValueError("system instruction hash")
        expected_parts = [
            {
                "kind": "text",
                "sha256": sha256_text(user_text),
                "size_bytes": len(user_text.encode("utf-8")),
            },
            *[
                {
                    "kind": "inline_image",
                    "mime_type": image.mime_type,
                    "sha256": image.content_sha256,
                    "size_bytes": len(image.data),
                }
                for image in images
            ],
        ]
        if request_manifest.get("user_turn_count") != 1:
            raise ValueError("user turn count")
        if request_manifest.get("parts") != expected_parts:
            raise ValueError("request parts")
        if provenance.get("request_payload_sha256") != request_manifest.get(
            "request_payload_sha256"
        ):
            raise ValueError("request payload hash")
        if provenance.get("request_manifest_sha256") != sha256_json(
            request_manifest
        ):
            raise ValueError("request manifest hash")
        if provenance.get("conversation_reused") is not False:
            raise ValueError("conversation isolation")
        if provenance.get("user_turn_count") != 1:
            raise ValueError("provenance user turn count")
        if retryable_kind is not None:
            response_manifest = provenance.get("response_manifest")
            if not isinstance(response_manifest, Mapping):
                raise ValueError("response manifest")
            if provenance.get("retryable_for_fresh_pe_attempt") is not True:
                raise ValueError("retryable marker")
            require_sha256(provenance.get("response_sha256"), "response_sha256")
            if retryable_kind == "truncation":
                if not isinstance(exc, GeminiOutputTruncated):
                    raise ValueError("truncation exception type")
                if response_manifest.get("finish_reason") not in (
                    PE_RETRYABLE_FINISH_REASONS
                ):
                    raise ValueError("retryable finish reason")
                if provenance.get("failure_kind") != "gemini_output_truncated":
                    raise ValueError("truncation failure kind")
            else:
                if not isinstance(exc, GeminiEmptyOutput):
                    raise ValueError("empty-output exception type")
                if response_manifest.get("finish_reason") != "STOP":
                    raise ValueError("empty-output finish reason")
                if provenance.get("failure_kind") != "gemini_empty_output":
                    raise ValueError("empty-output failure kind")
                raw_text = provenance.get("raw_assistant_text")
                if not isinstance(raw_text, str) or raw_text.strip():
                    raise ValueError("empty-output raw assistant text")
                if provenance.get("assistant_text_sha256") != sha256_text(raw_text):
                    raise ValueError("empty-output assistant text hash")
    except (AttributeError, TypeError, ValueError) as binding_error:
        error = PromptEngineeringError(
            "Gemini failure is not bound to the authorized isolated PE request"
        )
        error.provider_error = exc  # type: ignore[attr-defined]
        raise error from binding_error

    response_manifest = provenance.get("response_manifest")
    finish_reason = (
        str(response_manifest.get("finish_reason") or "")
        if isinstance(response_manifest, Mapping)
        else ""
    )
    return {
        "attempt": attempt_number,
        "outcome": {
            "truncation": "retryable_provider_truncation",
            "empty_output": "retryable_provider_empty_output",
            None: "terminal_provider_failure",
        }[retryable_kind],
        "request_manifest": request_manifest,
        "provider_provenance": provenance,
        "failure": {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "finish_reason": finish_reason or None,
            "failure_kind": provenance.get("failure_kind"),
            "retryable_as_fresh_pe_attempt": retryable_kind is not None,
        },
    }


def _attach_attempt_ledger(
    exc: BaseException,
    *,
    attempt_records: Sequence[Mapping[str, Any]],
    generation_policy: Mapping[str, Any],
) -> None:
    """Attach a strict audit ledger to a terminal exception without masking it."""

    ledger = strict_json_copy(list(attempt_records), "PE attempt ledger")
    policy = strict_json_copy(generation_policy, "PE generation policy")
    exc.attempts = copy.deepcopy(ledger)  # type: ignore[attr-defined]
    exc.pe_generation_policy = copy.deepcopy(policy)  # type: ignore[attr-defined]
    if isinstance(exc, ProviderError):
        provenance = copy.deepcopy(exc.provenance)
        provenance.update(
            {
                "pe_generation_policy": policy,
                "pe_attempt_count": len(ledger),
                "pe_attempt_ledger": ledger,
                "pe_attempt_ledger_sha256": sha256_json(ledger),
            }
        )
        exc.provenance = provenance


def _opening_user_turn(base_prompt: str, opening: OpeningCondition) -> str:
    # The image itself is a separate part in the same fresh USER message.  No
    # Memory Policy dialogue, tool history, or main-agent reasoning is sent.
    return (
        "Rewrite the complete generation prompt below under the system "
        "instruction. The attached image is the authoritative frame-zero "
        f"condition ({opening.semantic_effect}). Preserve unrelated content.\n\n"
        "COMPLETE BASE PROMPT:\n"
        f"{base_prompt}"
    )


class AdapterPromptEngineer:
    """Create immutable base prompts and condition-derived wire prompts.

    A base prompt is locked by ``scope_id`` after its first successful write.
    Ordinary condition revisions only produce new identity wire wrappers and
    never call Gemini.  An exact opening image may derive a rewritten wire
    prompt; that private call is cached by base prompt bytes, opening-image
    bytes, and the exact rewrite profile.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        gemini: GeminiTextPort,
        output_validator: PromptOutputValidator | None = None,
        max_generation_attempts: int = DEFAULT_MAX_PE_GENERATION_ATTEMPTS,
    ) -> None:
        if type(max_generation_attempts) is not int or max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be a positive integer")
        if output_validator is not None:
            validation_id = getattr(output_validator, "validation_id", None)
            if not isinstance(validation_id, str) or not validation_id:
                raise TypeError("output_validator must expose a non-empty validation_id")
            if not callable(getattr(output_validator, "validate", None)):
                raise TypeError("output_validator must implement validate")
        self.root = Path(root).expanduser().resolve()
        self.gemini = gemini
        self.output_validator = output_validator
        self.max_generation_attempts = max_generation_attempts
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def generation_policy(self) -> dict[str, Any]:
        return pe_generation_policy_manifest(self.max_generation_attempts)

    @property
    def output_validator_id(self) -> str:
        return (
            self.output_validator.validation_id
            if self.output_validator is not None
            else "mechanical-only.v1"
        )

    def _opening_validation_identity(
        self,
        *,
        base: BasePromptArtifact,
        opening: OpeningCondition,
    ) -> dict[str, Any]:
        """Seal every input that can change opening-rewrite acceptance."""

        lineage = base.to_dict().get("lineage")
        if not isinstance(lineage, Mapping) or not isinstance(
            lineage.get("raw_prompt"), str
        ):
            raise PromptArtifactCorrupt("base artifact has invalid raw-prompt lineage")
        semantic_contract = (
            getattr(self.output_validator, "semantic_contract", None)
            if self.output_validator is not None
            else None
        )
        semantic_contract_sha256: str | None = None
        if semantic_contract is not None:
            if not isinstance(semantic_contract, Mapping):
                raise TypeError("output_validator.semantic_contract must be a mapping")
            semantic_contract_sha256 = sha256_json(
                strict_json_copy(
                    semantic_contract,
                    "output validator semantic contract",
                )
            )
        return {
            "schema_version": "opening_validation_identity.v1",
            "output_validator_id": self.output_validator_id,
            "raw_prompt_sha256": sha256_text(lineage["raw_prompt"]),
            "expected_caption_language": lineage.get("expected_caption_language"),
            "condition_effect": opening.semantic_effect,
            "semantic_contract_sha256": semantic_contract_sha256,
        }

    def _validate_compiled_output(
        self,
        text: str,
        *,
        profile: PromptEngineeringProfile,
        raw_prompt: str,
        condition_effect: str | None,
        expected_caption_language: str | None,
    ) -> dict[str, Any]:
        validate_budget = getattr(self.output_validator, "validate_budget", None)
        if callable(validate_budget):
            # Check tokens before character/format checks so an overlong answer
            # cannot accidentally enter the semantic-correction/compression path.
            try:
                validate_budget(text)
            except V7PromptValidationError as exc:
                raise PromptOutputRejected(str(exc), violations=exc.violations) from exc
        _validate_output(text, profile.output_contract)
        if self.output_validator is None:
            return {"validation_id": self.output_validator_id}
        try:
            result = self.output_validator.validate(
                text,
                raw_prompt=raw_prompt,
                condition_effect=condition_effect,
                expected_caption_language=expected_caption_language,
            )
        except PromptOutputRejected:
            raise
        except V7PromptValidationError as exc:
            raise PromptOutputRejected(
                str(exc),
                violations=getattr(exc, "violations", ()),
            ) from exc
        return strict_json_copy(result, "output validation result")

    def _generate_with_bounded_correction(
        self,
        *,
        profile: PromptEngineeringProfile,
        original_user_turn: str,
        images: Sequence[GeminiInlineImage],
        raw_prompt: str,
        condition_effect: str | None,
        expected_caption_language: str | None,
    ) -> tuple[
        str,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        attempt_records: list[dict[str, Any]] = []
        validation_errors: list[str] = []
        user_turn = original_user_turn
        generation_policy = self.generation_policy
        for attempt_number in range(1, self.max_generation_attempts + 1):
            try:
                result = self.gemini.generate_text(
                    system_instruction=profile.system_instruction,
                    user_text=user_turn,
                    images=images,
                )
            except (GeminiOutputTruncated, GeminiEmptyOutput) as exc:
                retryable_kind = (
                    "truncation"
                    if isinstance(exc, GeminiOutputTruncated)
                    else "empty_output"
                )
                try:
                    record = _failure_attempt_record(
                        exc,
                        attempt_number=attempt_number,
                        profile=profile,
                        user_text=user_turn,
                        images=images,
                        retryable_kind=retryable_kind,
                    )
                except PromptEngineeringError as binding_error:
                    _attach_attempt_ledger(
                        binding_error,
                        attempt_records=attempt_records,
                        generation_policy=generation_policy,
                    )
                    raise
                attempt_records.append(record)
                if attempt_number < self.max_generation_attempts:
                    # An unusable provider response is never fed back to Gemini.
                    # Retry the same authorized USER turn as a wholly fresh PE
                    # request within the shared bounded attempt budget.
                    continue
                _attach_attempt_ledger(
                    exc,
                    attempt_records=attempt_records,
                    generation_policy=generation_policy,
                )
                raise
            except ProviderError as exc:
                # HTTP failures and non-truncation protocol failures are not PE
                # correction signals and must never consume another fresh turn.
                try:
                    record = _failure_attempt_record(
                        exc,
                        attempt_number=attempt_number,
                        profile=profile,
                        user_text=user_turn,
                        images=images,
                        retryable_kind=None,
                    )
                except PromptEngineeringError as binding_error:
                    _attach_attempt_ledger(
                        binding_error,
                        attempt_records=attempt_records,
                        generation_policy=generation_policy,
                    )
                    raise
                attempt_records.append(record)
                _attach_attempt_ledger(
                    exc,
                    attempt_records=attempt_records,
                    generation_policy=generation_policy,
                )
                raise

            record = {
                "attempt": attempt_number,
                "outcome": "provider_response_received",
                "request_manifest": strict_json_copy(
                    result.request_manifest, "Gemini request manifest"
                ),
                "provider_provenance": strict_json_copy(
                    result.provenance, "Gemini provider provenance"
                ),
            }
            try:
                _validate_gemini_result(
                    result,
                    profile=profile,
                    user_text=user_turn,
                    images=images,
                )
            except PromptEngineeringError as exc:
                record["outcome"] = "terminal_result_binding_failure"
                record["failure"] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "retryable_as_fresh_pe_attempt": False,
                }
                attempt_records.append(record)
                _attach_attempt_ledger(
                    exc,
                    attempt_records=attempt_records,
                    generation_policy=generation_policy,
                )
                raise
            normalized, normalization = _canonicalize_provider_output(result.text)
            record["output_normalization"] = normalization
            record["output_character_count"] = len(normalized)
            try:
                validation = self._validate_compiled_output(
                    normalized,
                    profile=profile,
                    raw_prompt=raw_prompt,
                    condition_effect=condition_effect,
                    expected_caption_language=expected_caption_language,
                )
            except PromptOutputRejected as exc:
                violations = strict_json_copy(
                    getattr(exc, "violations", [{"code": "validation.rejected"}]),
                    "validation violations",
                )
                record["outcome"] = "deterministic_validation_rejected"
                record["validation"] = {
                    "accepted": False,
                    "error": str(exc),
                    "violation_schema_version": getattr(
                        exc,
                        "violation_schema_version",
                        VALIDATION_VIOLATION_SCHEMA_VERSION,
                    ),
                    "violation_codes": list(
                        dict.fromkeys(
                            str(item.get("code"))
                            for item in violations
                            if isinstance(item, Mapping) and item.get("code")
                        )
                    ),
                    "violations": violations,
                }
                attempt_records.append(record)
                validation_errors.append(str(exc))
                budget_rejection = (
                    getattr(self.output_validator, "output_budget_contract", None) is not None
                    and bool(set(record["validation"]["violation_codes"]) & {
                        "length.encoder_token_limit", "length.above_maximum",
                    })
                )
                if budget_rejection:
                    record["validation"]["recovery"] = "fresh_original_input_no_compression"
                if attempt_number >= self.max_generation_attempts:
                    error = PromptOutputRejected(
                        "PE output failed deterministic validation after "
                        f"{self.max_generation_attempts} fresh attempt(s): {exc}",
                        violations=violations,
                    )
                    _attach_attempt_ledger(
                        error,
                        attempt_records=attempt_records,
                        generation_policy=generation_policy,
                    )
                    raise error from exc
                if budget_rejection:
                    # Discard the invalid output. Retry the exact original input,
                    # never give the oversized text to PE for a compression pass.
                    user_turn = original_user_turn
                    validation_errors.clear()
                    continue
                try:
                    validation_context = (
                        strict_json_copy(
                            self.output_validator.correction_context(
                                raw_prompt=raw_prompt,
                                condition_effect=condition_effect,
                                expected_caption_language=expected_caption_language,
                            ),
                            "validator correction context",
                        )
                        if self.output_validator is not None
                        and callable(
                            getattr(self.output_validator, "correction_context", None)
                        )
                        else None
                    )
                    user_turn = _correction_user_turn(
                        original_user_turn=original_user_turn,
                        rejected_text=normalized or result.text,
                        validation_errors=validation_errors,
                        validation_violations=violations,
                        validation_context=validation_context,
                        output_contract=profile.output_contract,
                    )
                except Exception as correction_error:
                    _attach_attempt_ledger(
                        correction_error,
                        attempt_records=attempt_records,
                        generation_policy=generation_policy,
                    )
                    raise
                continue

            record["outcome"] = "accepted"
            record["validation"] = {
                "accepted": True,
                "result": validation,
            }
            attempt_records.append(record)
            final_request = copy.deepcopy(result.request_manifest)
            final_request.update(
                {
                    "correction_strategy": PE_GENERATION_STRATEGY,
                    "generation_policy": copy.deepcopy(generation_policy),
                    "max_generation_attempts": self.max_generation_attempts,
                    "attempt_count": attempt_number,
                    "attempts": [
                        copy.deepcopy(item["request_manifest"])
                        for item in attempt_records
                    ],
                    "attempt_ledger": [
                        {
                            "attempt": item["attempt"],
                            "outcome": item["outcome"],
                            "request_manifest": copy.deepcopy(
                                item["request_manifest"]
                            ),
                        }
                        for item in attempt_records
                    ],
                }
            )
            final_provenance = copy.deepcopy(result.provenance)
            final_provenance.update(
                {
                    "correction_strategy": PE_GENERATION_STRATEGY,
                    "pe_generation_policy": copy.deepcopy(generation_policy),
                    "max_generation_attempts": self.max_generation_attempts,
                    "attempt_count": attempt_number,
                    "attempts": attempt_records,
                    "pe_attempt_ledger": copy.deepcopy(attempt_records),
                    "pe_attempt_ledger_sha256": sha256_json(attempt_records),
                }
            )
            return (
                normalized,
                final_request,
                final_provenance,
                normalization,
                validation,
            )
        raise AssertionError("bounded PE loop terminated without a result")

    def _scope_dir(self, scope_id: str) -> Path:
        return self.root / "scopes" / _artifact_component(scope_id)

    def _base_path(self, scope_id: str) -> Path:
        return self._scope_dir(scope_id) / "base_prompt.json"

    def ensure_base(
        self,
        request: BasePromptRequest,
        *,
        profile: PromptEngineeringProfile | None,
    ) -> BasePromptArtifact:
        if request.mode == "gemini_rewrite" and profile is None:
            raise ValueError("gemini_rewrite requires a PE profile")
        if request.mode in {"passthrough", "policy_direct"} and profile is not None:
            raise ValueError("direct prompt modes must not claim a PE profile")
        request_fingerprint = request.fingerprint(
            profile, output_validator_id=self.output_validator_id
        )
        fingerprint = (
            sha256_json(
                {
                    "request_fingerprint": request_fingerprint,
                    "pe_generation_policy": self.generation_policy,
                }
            )
            if request.mode == "gemini_rewrite"
            else request_fingerprint
        )
        path = self._base_path(request.scope_id)
        with _exclusive_lock(path.with_suffix(".lock")):
            if path.exists():
                artifact = self._load_base(path)
                if artifact.request_fingerprint != fingerprint:
                    raise PromptArtifactConflict(
                        "base prompt is already locked for this scope with different input"
                    )
                self._validate_stored_base(artifact)
                return artifact

            if request.mode in {"passthrough", "policy_direct"}:
                output = request.exact_pe_input_text if request.mode == "policy_direct" else request.raw_prompt
                request_manifest: dict[str, Any] = {
                    "transport": "local_" + request.mode,
                    "provider_call_count": 0,
                    "user_turn_count": 0,
                    "raw_prompt_sha256": sha256_text(request.raw_prompt),
                }
                provenance: dict[str, Any] = {
                    "transport": "local_" + request.mode,
                    "provider_call_count": 0,
                }
                profile_manifest = None
            else:
                assert profile is not None
                (
                    output,
                    request_manifest,
                    provenance,
                    output_normalization,
                    output_validation,
                ) = self._generate_with_bounded_correction(
                    profile=profile,
                    original_user_turn=request.exact_pe_input_text,
                    images=(),
                    raw_prompt=request.raw_prompt,
                    condition_effect=None,
                    expected_caption_language=request.expected_caption_language,
                )
                profile_manifest = profile.manifest()
            if request.mode in {"passthrough", "policy_direct"}:
                output_normalization = {
                    "method": "none_" + request.mode,
                    "changed": False,
                    "raw_text_sha256": sha256_text(output),
                    "normalized_text_sha256": sha256_text(output),
                }
                output_validation = {
                    "validation_id": request.mode + ".byte_exact.v1"
                }

            origin_plan = strict_json_copy(
                request.origin_plan_projection, "origin_plan_projection"
            )
            initial_conditions = strict_json_copy(
                request.initial_condition_projection,
                "initial_condition_projection",
            )
            prompt_hash = sha256_text(output)
            artifact_id = "prompt-base-" + sha256_json(
                {
                    "request_fingerprint": fingerprint,
                    "prompt_sha256": prompt_hash,
                }
            ).removeprefix("sha256:")
            document = seal_artifact(
                {
                    "schema_version": BASE_PROMPT_ARTIFACT_SCHEMA_VERSION,
                    "layer": "base",
                    "artifact_id": artifact_id,
                    "scope_id": request.scope_id,
                    "request_fingerprint": fingerprint,
                    "mode": request.mode,
                    "lineage": {
                        "raw_prompt": request.raw_prompt,
                        "raw_prompt_sha256": sha256_text(request.raw_prompt),
                        "pe_input_text": request.exact_pe_input_text,
                        "pe_input_text_sha256": sha256_text(
                            request.exact_pe_input_text
                        ),
                        "origin_plan_projection": origin_plan,
                        "origin_plan_projection_sha256": sha256_json(origin_plan),
                        "initial_condition_projection": initial_conditions,
                        "initial_condition_projection_sha256": sha256_json(
                            initial_conditions
                        ),
                        "expected_caption_language": request.expected_caption_language,
                    },
                    "profile": profile_manifest,
                    "request_manifest": request_manifest,
                    "provider_provenance": provenance,
                    "output_normalization": output_normalization,
                    "output_validation": output_validation,
                    "prompt": {"text": output, "sha256": prompt_hash},
                    "created_at": _utc_now(),
                }
            )
            artifact = BasePromptArtifact.from_dict(document)
            _atomic_write_json(path, artifact.to_dict())
            return artifact

    def load_base(self, scope_id: str) -> BasePromptArtifact:
        path = self._base_path(scope_id)
        if not path.exists():
            raise FileNotFoundError(path)
        artifact = self._load_base(path)
        self._validate_stored_base(artifact)
        return artifact

    def _validate_stored_base(self, artifact: BasePromptArtifact) -> None:
        document = artifact.to_dict()
        if document.get("mode") in {"passthrough", "policy_direct"}:
            if (document.get("output_validation") or {}).get(
                "validation_id"
            ) != document["mode"] + ".byte_exact.v1":
                raise PromptArtifactCorrupt(
                    "direct base artifact has invalid validation lineage"
                )
            source_key = "pe_input_text" if document["mode"] == "policy_direct" else "raw_prompt"
            if artifact.prompt_text != document["lineage"][source_key]:
                raise PromptArtifactCorrupt("direct artifact differs from its source text")
            return
        profile = document.get("profile")
        contract = profile.get("output_contract") if isinstance(profile, Mapping) else None
        if not isinstance(contract, Mapping):
            raise PromptArtifactCorrupt("base artifact has no output contract")
        try:
            _validate_output(
                artifact.prompt_text,
                PromptOutputContract(
                    single_paragraph=contract.get("single_paragraph", True),
                    min_characters=contract.get("min_characters", 1),
                    max_characters=contract.get("max_characters"),
                    forbidden_prefixes=tuple(contract.get("forbidden_prefixes") or ()),
                    required_any_prefixes=tuple(
                        contract.get("required_any_prefixes") or ()
                    ),
                ),
            )
            validation = (
                {"validation_id": self.output_validator_id}
                if self.output_validator is None
                else strict_json_copy(
                    self.output_validator.validate(
                        artifact.prompt_text,
                        raw_prompt=str(document["lineage"]["raw_prompt"]),
                        condition_effect=None,
                        expected_caption_language=document["lineage"].get(
                            "expected_caption_language"
                        ),
                    ),
                    "stored output validation",
                )
            )
        except Exception as exc:
            raise PromptArtifactCorrupt(
                "base artifact fails the active output validator"
            ) from exc
        if document.get("output_validation") != validation:
            raise PromptArtifactConflict(
                "base artifact was locked under a different output validator"
            )

    def compile_wire(
        self,
        *,
        base: BasePromptArtifact,
        condition_projection: Any,
        opening: OpeningCondition | None = None,
        opening_profile: PromptEngineeringProfile | None = None,
    ) -> WirePromptArtifact:
        # Re-validate a detached artifact before trusting it as prompt lineage.
        base = BasePromptArtifact.from_dict(base.to_dict())
        projection = strict_json_copy(condition_projection, "condition_projection")
        projection_hash = sha256_json(projection)

        if opening is None:
            if opening_profile is not None:
                raise ValueError("opening_profile is legal only with an opening image")
            prompt_text = base.prompt_text
            derivation = "identity"
            opening_manifest = None
            rewrite_profile_manifest = None
            rewrite_cache_key = None
            rewrite_request_manifest = {
                "transport": "identity",
                "provider_call_count": 0,
            }
            rewrite_provenance = {
                "transport": "identity",
                "provider_call_count": 0,
            }
            output_validation = copy.deepcopy(
                base.to_dict().get("output_validation") or {}
            )
        else:
            if opening_profile is None:
                raise ValueError("an opening image requires an opening rewrite profile")
            path, mime_type, image_bytes, image_hash = opening.read()
            validation_identity = self._opening_validation_identity(
                base=base,
                opening=opening,
            )
            cache_key = sha256_json(
                {
                    "base_prompt_sha256": base.prompt_sha256,
                    "opening_image_sha256": image_hash,
                    "profile_sha256": opening_profile.profile_sha256,
                    "pe_generation_policy": self.generation_policy,
                    "validation_identity": validation_identity,
                }
            )
            receipt = self._ensure_opening_rewrite(
                base=base,
                opening=opening,
                image_path=path,
                mime_type=mime_type,
                image_bytes=image_bytes,
                image_sha256=image_hash,
                profile=opening_profile,
                cache_key=cache_key,
                validation_identity=validation_identity,
            )
            prompt_text = str(receipt["prompt"]["text"])
            derivation = "opening_image_rewrite"
            opening_manifest = {
                "condition_id": opening.condition_id,
                "semantic_effect": opening.semantic_effect,
                "source_path": str(path),
                "mime_type": mime_type,
                "size_bytes": len(image_bytes),
                "sha256": image_hash,
            }
            rewrite_profile_manifest = opening_profile.manifest()
            rewrite_cache_key = cache_key
            rewrite_request_manifest = copy.deepcopy(receipt["request_manifest"])
            rewrite_provenance = copy.deepcopy(receipt["provider_provenance"])
            output_validation = copy.deepcopy(receipt["output_validation"])

        wire_fingerprint = sha256_json(
            {
                "base_artifact_id": base.artifact_id,
                "base_artifact_sha256": base.artifact_sha256,
                "condition_projection_sha256": projection_hash,
                "derivation": derivation,
                "opening": opening_manifest,
                "rewrite_profile_sha256": (
                    opening_profile.profile_sha256
                    if opening_profile is not None
                    else None
                ),
                "rewrite_cache_key": rewrite_cache_key,
                "prompt_sha256": sha256_text(prompt_text),
                "output_validator_id": self.output_validator_id,
            }
        )
        path = (
            self._scope_dir(base.scope_id)
            / "wire_prompts"
            / f"{wire_fingerprint.removeprefix('sha256:')}.json"
        )
        with _exclusive_lock(path.with_suffix(".lock")):
            if path.exists():
                artifact = self._load_wire(path)
                if artifact.request_fingerprint != wire_fingerprint:
                    raise PromptArtifactCorrupt("wire artifact filename/content mismatch")
                return artifact
            artifact_id = "prompt-wire-" + sha256_json(
                {
                    "request_fingerprint": wire_fingerprint,
                    "prompt_sha256": sha256_text(prompt_text),
                }
            ).removeprefix("sha256:")
            base_document = base.to_dict()
            document = seal_artifact(
                {
                    "schema_version": WIRE_PROMPT_ARTIFACT_SCHEMA_VERSION,
                    "layer": "wire",
                    "artifact_id": artifact_id,
                    "scope_id": base.scope_id,
                    "request_fingerprint": wire_fingerprint,
                    "base_artifact_id": base.artifact_id,
                    "base_artifact_sha256": base.artifact_sha256,
                    "derivation": derivation,
                    "lineage": {
                        "raw_prompt_sha256": base_document["lineage"][
                            "raw_prompt_sha256"
                        ],
                        "origin_plan_projection_sha256": base_document["lineage"][
                            "origin_plan_projection_sha256"
                        ],
                        "active_condition_projection": projection,
                        "active_condition_projection_sha256": projection_hash,
                        "opening_condition": opening_manifest,
                    },
                    "profile": rewrite_profile_manifest,
                    "rewrite_cache_key": rewrite_cache_key,
                    "request_manifest": rewrite_request_manifest,
                    "provider_provenance": rewrite_provenance,
                    "output_validation": output_validation,
                    "prompt": {
                        "text": prompt_text,
                        "sha256": sha256_text(prompt_text),
                    },
                    "created_at": _utc_now(),
                }
            )
            artifact = WirePromptArtifact.from_dict(document)
            _atomic_write_json(path, artifact.to_dict())
            return artifact

    def _ensure_opening_rewrite(
        self,
        *,
        base: BasePromptArtifact,
        opening: OpeningCondition,
        image_path: Path,
        mime_type: str,
        image_bytes: bytes,
        image_sha256: str,
        profile: PromptEngineeringProfile,
        cache_key: str,
        validation_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        cache_path = (
            self.root
            / "opening_rewrite_cache"
            / f"{cache_key.removeprefix('sha256:')}.json"
        )
        with _exclusive_lock(cache_path.with_suffix(".lock")):
            if cache_path.exists():
                receipt = _read_json(cache_path)
                self._validate_opening_receipt(
                    receipt,
                    cache_key=cache_key,
                    base_prompt_sha256=base.prompt_sha256,
                    image_sha256=image_sha256,
                    profile_sha256=profile.profile_sha256,
                    validation_identity=validation_identity,
                )
                validation = self._validate_compiled_output(
                    str(receipt["prompt"]["text"]),
                    profile=profile,
                    raw_prompt=str(base.to_dict()["lineage"]["raw_prompt"]),
                    condition_effect=opening.semantic_effect,
                    expected_caption_language=base.to_dict()["lineage"].get(
                        "expected_caption_language"
                    ),
                )
                if receipt.get("output_validation") != validation:
                    raise PromptArtifactConflict(
                        "opening rewrite cache was created under a different validator"
                    )
                return receipt

            user_turn = _opening_user_turn(base.prompt_text, opening)
            inline_image = GeminiInlineImage.from_bytes(
                mime_type=mime_type,
                data=image_bytes,
            )
            (
                prompt_text,
                request_manifest,
                provider_provenance,
                output_normalization,
                output_validation,
            ) = self._generate_with_bounded_correction(
                profile=profile,
                original_user_turn=user_turn,
                images=(inline_image,),
                raw_prompt=str(base.to_dict()["lineage"]["raw_prompt"]),
                condition_effect=opening.semantic_effect,
                expected_caption_language=base.to_dict()["lineage"].get(
                    "expected_caption_language"
                ),
            )
            receipt: dict[str, Any] = {
                "schema_version": OPENING_REWRITE_CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "base_prompt_sha256": base.prompt_sha256,
                "opening_image": {
                    "source_path": str(image_path),
                    "mime_type": mime_type,
                    "size_bytes": len(image_bytes),
                    "sha256": image_sha256,
                },
                "profile": profile.manifest(),
                "validation_identity": strict_json_copy(
                    validation_identity,
                    "opening validation identity",
                ),
                "validation_identity_sha256": sha256_json(validation_identity),
                "user_turn": user_turn,
                "user_turn_sha256": sha256_text(user_turn),
                "request_manifest": request_manifest,
                "provider_provenance": provider_provenance,
                "output_normalization": output_normalization,
                "output_validation": output_validation,
                "prompt": {
                    "text": prompt_text,
                    "sha256": sha256_text(prompt_text),
                },
                "created_at": _utc_now(),
            }
            receipt["receipt_sha256"] = sha256_json(receipt)
            self._validate_opening_receipt(
                receipt,
                cache_key=cache_key,
                base_prompt_sha256=base.prompt_sha256,
                image_sha256=image_sha256,
                profile_sha256=profile.profile_sha256,
                validation_identity=validation_identity,
            )
            _atomic_write_json(cache_path, receipt)
            return receipt

    @staticmethod
    def _validate_opening_receipt(
        receipt: Mapping[str, Any],
        *,
        cache_key: str,
        base_prompt_sha256: str,
        image_sha256: str,
        profile_sha256: str,
        validation_identity: Mapping[str, Any],
    ) -> None:
        try:
            expected_validation_identity = strict_json_copy(
                validation_identity,
                "opening validation identity",
            )
            if receipt.get("schema_version") != OPENING_REWRITE_CACHE_SCHEMA_VERSION:
                raise ValueError("schema")
            if receipt.get("cache_key") != cache_key:
                raise ValueError("cache key")
            if receipt.get("base_prompt_sha256") != base_prompt_sha256:
                raise ValueError("base prompt")
            opening_image = receipt.get("opening_image")
            if not isinstance(opening_image, Mapping) or opening_image.get(
                "sha256"
            ) != image_sha256:
                raise ValueError("opening image")
            profile = receipt.get("profile")
            if not isinstance(profile, Mapping) or profile.get(
                "profile_sha256"
            ) != profile_sha256:
                raise ValueError("profile")
            if receipt.get("validation_identity") != expected_validation_identity:
                raise ValueError("validation identity")
            if receipt.get("validation_identity_sha256") != sha256_json(
                expected_validation_identity
            ):
                raise ValueError("validation identity hash")
            prompt = receipt.get("prompt")
            if (
                not isinstance(prompt, Mapping)
                or not isinstance(prompt.get("text"), str)
                or prompt.get("sha256") != sha256_text(prompt["text"])
            ):
                raise ValueError("prompt")
            stored = receipt.get("receipt_sha256")
            unhashed = copy.deepcopy(dict(receipt))
            unhashed.pop("receipt_sha256", None)
            if stored != sha256_json(unhashed):
                raise ValueError("integrity hash")
        except (KeyError, TypeError, ValueError) as exc:
            raise PromptArtifactCorrupt("opening rewrite cache is corrupt") from exc

    @staticmethod
    def _load_base(path: Path) -> BasePromptArtifact:
        try:
            return BasePromptArtifact.from_dict(_read_json(path))
        except PromptArtifactCorrupt:
            raise
        except (TypeError, ValueError) as exc:
            raise PromptArtifactCorrupt(f"base prompt artifact is corrupt: {path}") from exc

    @staticmethod
    def _load_wire(path: Path) -> WirePromptArtifact:
        try:
            return WirePromptArtifact.from_dict(_read_json(path))
        except PromptArtifactCorrupt:
            raise
        except (TypeError, ValueError) as exc:
            raise PromptArtifactCorrupt(f"wire prompt artifact is corrupt: {path}") from exc


__all__ = [
    "DEFAULT_MAX_PE_GENERATION_ATTEMPTS",
    "PE_GENERATION_STRATEGY",
    "PE_RETRYABLE_FAILURE_KINDS",
    "PE_RETRYABLE_FINISH_REASONS",
    "AdapterPromptEngineer",
    "GeminiTextPort",
    "PromptArtifactConflict",
    "PromptArtifactCorrupt",
    "PromptEngineeringError",
    "PromptOutputRejected",
    "pe_generation_policy_manifest",
]
