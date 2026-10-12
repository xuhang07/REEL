"""A stateless Gemini text-output transport private to the backend adapter."""

from __future__ import annotations

import base64
import copy
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from storymem_teacher.http_session import create_provider_session

from storymem_teacher.providers import (
    ProviderError,
    ProviderProtocolError,
    RemoteTeacherClient,
    SharedRequestRateLimiter,
)

from .models import sha256_bytes, sha256_json, sha256_text


_SUCCESS_FINISH_REASONS = frozenset({"STOP"})
_RETRYABLE_TRUNCATION_FINISH_REASONS = frozenset({"MAX_TOKENS"})


class GeminiOutputTruncated(ProviderProtocolError):
    """Gemini explicitly exhausted its output budget before completing PE.

    This dedicated type is one of the explicitly retryable generation
    outcomes.  All HTTP failures and every other protocol/finish reason remain
    terminal.
    """


class GeminiEmptyOutput(ProviderProtocolError):
    """Gemini stopped cleanly but returned no usable non-thought text.

    A whitespace-only final part is a transient generation outcome rather than
    a malformed provider envelope.  The prompt engineer may therefore spend
    another slot from its existing bounded fresh-attempt budget on it.
    """


@dataclass(frozen=True, slots=True)
class GeminiInlineImage:
    mime_type: str
    data: bytes
    content_sha256: str

    @classmethod
    def from_bytes(cls, *, mime_type: str, data: bytes) -> "GeminiInlineImage":
        payload = bytes(data)
        if not mime_type.startswith("image/"):
            raise ValueError("Gemini PE inline condition must use an image MIME type")
        if not payload:
            raise ValueError("Gemini PE inline condition must not be empty")
        return cls(
            mime_type=mime_type,
            data=payload,
            content_sha256=sha256_bytes(payload),
        )


@dataclass(frozen=True, slots=True)
class GeminiTextResult:
    text: str
    request_manifest: dict[str, Any]
    provenance: dict[str, Any]


class IsolatedGeminiTextClient:
    """Issue one fresh, ordinary Gemini USER turn per PE operation.

    The HTTP session belongs solely to this client.  Reusing a connection does
    not reuse a model conversation: every payload contains exactly one USER
    content and no assistant/tool history.  The response is text, never a
    provider-native tool call or JSON command.
    """

    def __init__(
        self,
        api_url: str,
        *,
        api_key: str | None,
        model: str,
        timeout: float = 180.0,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        max_output_tokens: int = 32768,
        temperature: float = 0.0,
        extra_headers: Mapping[str, str] | None = None,
        rate_limiter: SharedRequestRateLimiter | None = None,
        session_factory: Callable[[], Any] = create_provider_session,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self.temperature = float(temperature)
        self.max_output_tokens = max_output_tokens
        # Constructing the session here, instead of accepting the main
        # agent's client/session, is the isolation boundary.
        private_session = session_factory()
        self._transport = RemoteTeacherClient(
            api_url,
            api_key=api_key,
            model=model,
            api_style="gemini",
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            max_output_tokens=max_output_tokens,
            session=private_session,
            extra_headers=extra_headers,
            rate_limiter=rate_limiter,
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> "IsolatedGeminiTextClient":
        api_base = os.getenv("RECAPTION_OPENAI_BASE_URL")
        api_url = (
            os.getenv("STORYMEM_PE_URL")
            or os.getenv("RECAPTION_URL")
            or (f"{api_base.rstrip('/')}/responses" if api_base else None)
            or "https://modelservice.jdcloud.com/v1/responses"
        )
        values: dict[str, Any] = {
            "api_url": api_url,
            "api_key": os.getenv("STORYMEM_PE_API_KEY")
            or os.getenv("RECAPTION_API_KEY")
            or os.getenv("API_KEY"),
            "model": os.getenv("STORYMEM_PE_MODEL")
            or os.getenv("RECAPTION_MODEL")
            or "Gemini-3.5-Flash-Third",
        }
        values.update(overrides)
        return cls(**values)

    def generate_text(
        self,
        *,
        system_instruction: str,
        user_text: str,
        images: Sequence[GeminiInlineImage] = (),
    ) -> GeminiTextResult:
        if not isinstance(system_instruction, str) or not system_instruction:
            raise ValueError("system_instruction must be non-empty")
        if not isinstance(user_text, str) or not user_text:
            raise ValueError("user_text must be non-empty")
        part_manifest: list[dict[str, Any]] = [
            {
                "kind": "text",
                "sha256": sha256_text(user_text),
                "size_bytes": len(user_text.encode("utf-8")),
            }
        ]
        for image in images:
            if not isinstance(image, GeminiInlineImage):
                raise TypeError("images must contain GeminiInlineImage values")
            if image.content_sha256 != sha256_bytes(image.data):
                raise ValueError("inline image hash no longer matches its bytes")
            part_manifest.append(
                {
                    "kind": "inline_image",
                    "mime_type": image.mime_type,
                    "sha256": image.content_sha256,
                    "size_bytes": len(image.data),
                }
            )
        parts: list[dict[str, Any]] = [{"text": user_text}]
        for image in images:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": image.mime_type,
                        "data": base64.b64encode(image.data).decode("ascii"),
                    }
                }
            )
        payload: dict[str, Any] = {
            "model": self._transport.model,
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            # This is deliberately always one fresh USER turn.
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "text/plain",
            },
        }
        request_manifest = {
            "request_payload_sha256": sha256_json(payload),
            "system_instruction_sha256": sha256_text(system_instruction),
            "user_turn_count": 1,
            "parts": part_manifest,
            "model": self._transport.model,
            "api_style": "gemini",
            "transport": "gemini_native_generate_content",
            "response_mime_type": "text/plain",
        }
        try:
            body, provenance = self._transport._invoke(payload)
        except ProviderError as exc:
            failed = self._failure_provenance(
                exc.provenance,
                request_manifest=request_manifest,
                failure_stage="transport_invoke",
            )
            raise type(exc)(str(exc), provenance=failed) from exc
        try:
            text, response_manifest = self._final_text(body, provenance)
        except ProviderProtocolError as exc:
            failed = self._failure_provenance(
                exc.provenance,
                request_manifest=request_manifest,
                failure_stage="response_protocol",
            )
            raise type(exc)(str(exc), provenance=failed) from exc
        provenance = copy.deepcopy(provenance)
        provenance.update(
            {
                "raw_assistant_text": text,
                "assistant_text_sha256": sha256_text(text),
                "transport": "isolated_gemini_single_user_turn",
                "conversation_reused": False,
                "user_turn_count": 1,
                "response_mime_type": "text/plain",
                "response_manifest": response_manifest,
            }
        )
        if provenance.get("request_payload_sha256") != request_manifest[
            "request_payload_sha256"
        ]:
            raise RuntimeError("Gemini transport changed the sealed PE request")
        return GeminiTextResult(
            text=text,
            request_manifest=request_manifest,
            provenance=provenance,
        )

    @staticmethod
    def _failure_provenance(
        provenance: Mapping[str, Any],
        *,
        request_manifest: Mapping[str, Any],
        failure_stage: str,
    ) -> dict[str, Any]:
        """Attach the sealed, secret-free request identity to a failed call."""

        failed = copy.deepcopy(dict(provenance))
        request = copy.deepcopy(dict(request_manifest))
        expected_hash = request["request_payload_sha256"]
        observed_hash = failed.get("request_payload_sha256")
        if observed_hash not in {None, expected_hash}:
            raise RuntimeError("Gemini failure provenance changed the sealed PE request")
        failed.update(
            {
                "request_payload_sha256": expected_hash,
                "request_manifest": request,
                "request_manifest_sha256": sha256_json(request),
                "transport": "isolated_gemini_single_user_turn",
                "conversation_reused": False,
                "user_turn_count": 1,
                "response_mime_type": "text/plain",
                "failure_stage": failure_stage,
            }
        )
        return failed

    @staticmethod
    def _final_text(
        body: Mapping[str, Any], provenance: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Extract only non-thought Gemini parts and reject truncation.

        Some Gemini endpoints return internal reasoning as text parts marked
        ``thought=true`` before the actual answer.  Concatenating every text
        part leaks that reasoning into the generator prompt and can make a
        truncated response look valid, so the PE transport has a stricter
        extractor than the legacy Teacher client.
        """

        def fail(
            message: str,
            *,
            manifest: Mapping[str, Any] | None = None,
            truncated: bool = False,
            empty_output: bool = False,
            raw_assistant_text: str = "",
        ) -> None:
            if truncated and empty_output:
                raise AssertionError(
                    "Gemini failure cannot be both truncated and empty"
                )
            failed = copy.deepcopy(dict(provenance))
            failed["response_manifest"] = copy.deepcopy(dict(manifest or {}))
            failed["raw_assistant_text"] = raw_assistant_text
            failed["assistant_text_sha256"] = sha256_text(raw_assistant_text)
            if truncated:
                failure_kind = "gemini_output_truncated"
                error_type = GeminiOutputTruncated
            elif empty_output:
                failure_kind = "gemini_empty_output"
                error_type = GeminiEmptyOutput
            else:
                failure_kind = "gemini_protocol_error"
                error_type = ProviderProtocolError
            failed["failure_kind"] = failure_kind
            failed["retryable_for_fresh_pe_attempt"] = truncated or empty_output
            raise error_type(message, provenance=failed)

        candidates = body.get("candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 1
            or not isinstance(candidates[0], Mapping)
        ):
            fail("Gemini PE response must contain exactly one candidate")
        candidate = candidates[0]
        finish_reason = str(
            candidate.get("finishReason") or candidate.get("finish_reason") or ""
        ).upper()
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else None
        if not isinstance(parts, list):
            fail("Gemini PE candidate has no content parts")

        final_text_parts: list[str] = []
        final_part_hashes: list[str] = []
        thought_part_hashes: list[str] = []
        non_text_part_count = 0
        for part in parts:
            if not isinstance(part, Mapping):
                non_text_part_count += 1
                continue
            part_text = part.get("text")
            if not isinstance(part_text, str):
                non_text_part_count += 1
                continue
            if part.get("thought") is True:
                thought_part_hashes.append(sha256_text(part_text))
                continue
            final_text_parts.append(part_text)
            final_part_hashes.append(sha256_text(part_text))
        manifest = {
            "finish_reason": finish_reason,
            "candidate_count": 1,
            "part_count": len(parts),
            "final_text_part_count": len(final_text_parts),
            "final_text_part_sha256": final_part_hashes,
            "thought_part_count": len(thought_part_hashes),
            "thought_part_sha256": thought_part_hashes,
            "non_text_part_count": non_text_part_count,
        }
        if finish_reason not in _SUCCESS_FINISH_REASONS:
            if final_text_parts:
                discarded_text = "".join(final_text_parts)
                manifest.update(
                    {
                        "discarded_final_text_sha256": sha256_text(discarded_text),
                        "discarded_final_text_character_count": len(discarded_text),
                    }
                )
            fail(
                f"Gemini PE response did not finish cleanly: {finish_reason or 'unknown'}",
                manifest=manifest,
                truncated=finish_reason in _RETRYABLE_TRUNCATION_FINISH_REASONS,
            )
        text = "".join(final_text_parts)
        manifest["raw_final_text_sha256"] = sha256_text(text)
        if not text.strip():
            fail(
                "Gemini PE response has no non-thought assistant text",
                manifest=manifest,
                empty_output=True,
                raw_assistant_text=text,
            )
        return text, manifest


__all__ = [
    "GeminiEmptyOutput",
    "GeminiInlineImage",
    "GeminiOutputTruncated",
    "GeminiTextResult",
    "IsolatedGeminiTextClient",
]
