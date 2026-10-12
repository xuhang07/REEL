"""Production API transports for the canonical StoryMem teacher.

The adapters in this module are deliberately policy-free: they do not choose
tools, rewrite prompts, evaluate generations, manufacture placeholder media, or
fall back to legacy endpoints.  They transport an already-authorized request
and retain enough exact request/response provenance for later auditing.
"""

from __future__ import annotations

import base64
import copy
import email.utils
import fcntl
import io
import json
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote, urljoin, urlparse

import requests
from PIL import Image, ImageOps

from .http_session import create_provider_session

from .media import (
    MediaInput,
    atomic_write_bytes,
    content_block,
    data_uri,
    describe_media,
    detect_mime,
    require_file,
)
from .util import (
    atomic_write_json,
    canonical_json,
    safe_name,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
)


_OPENAI_STYLES = frozenset({"openai", "openai_chat", "chat", "chat_completions"})
_GEMINI_STYLES = frozenset({"gemini", "gemini_style", "generate_content"})
_PLAIN_JSON_TOOL_TRANSPORT = "plain_json_tool_dialogue"
_RETRYABLE_STATUS = frozenset({408, 425, 429})
_SUCCESS_STATES = frozenset({"completed", "done", "success", "succeeded"})
_FAILURE_STATES = frozenset({"cancelled", "canceled", "error", "failed"})


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, provenance: Mapping[str, Any] | None = None):
        self.provenance = copy.deepcopy(dict(provenance or {}))
        super().__init__(message)


class ProviderHTTPError(ProviderError):
    pass


class ProviderProtocolError(ProviderError):
    pass


@dataclass(frozen=True)
class ToolCall:
    """One tool action parsed from the closed Teacher's JSON text reply.

    Closed Teacher APIs are deliberately used as ordinary chat/JSON APIs.  A
    ``ToolCall`` is therefore a local interpretation of assistant text, not a
    provider-native function call.  ``call_id`` remains as a compatibility
    field for callers that consume this value generically; plain-JSON commands
    do not populate it.
    """

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict, compare=False)
    reason: str = ""

    @property
    def tool_call(self) -> dict[str, Any]:
        """Compatibility view consumed by the package runner."""

        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        value = {"name": self.name, "arguments": copy.deepcopy(self.arguments)}
        if self.call_id:
            value["call_id"] = self.call_id
        return value


@dataclass(frozen=True)
class JsonCallResult:
    value: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"value": copy.deepcopy(self.value), "provenance": copy.deepcopy(self.provenance)}


@dataclass(frozen=True)
class ImageGenerationRequest:
    request_id: str
    prompt: str
    reference_paths: Sequence[str | Path]
    seed: int | None
    width: int
    height: int
    output_dir: str | Path


@dataclass(frozen=True)
class VideoGenerationRequest:
    request_id: str
    work_id: str
    shot_id: str
    prompt: str
    condition_path: str | Path | None
    memory_slots: Sequence[Mapping[str, Any]]
    seed: int
    width: int
    height: int
    num_frames: int
    output_dir: str | Path


@dataclass(frozen=True)
class ProviderResult:
    output_path: str
    request_payload: dict[str, Any]
    request_sha256: str
    request_id: str
    response_id: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


IMAGE_GENERATION_REQUEST_SCHEMA: dict[str, Any] = {
    "$id": "storymem.provider.image.request.v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "request_id",
        "prompt",
        "reference_paths",
        "seed",
        "width",
        "height",
        "output_dir",
    ],
    "properties": {
        "request_id": {"type": "string", "minLength": 1},
        "prompt": {"type": "string", "minLength": 1},
        "reference_paths": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "seed": {"type": ["integer", "null"]},
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "output_dir": {"type": "string", "minLength": 1},
    },
}


VIDEO_GENERATION_REQUEST_SCHEMA: dict[str, Any] = {
    "$id": "storymem.provider.echo15.r2v.request.v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "request_id",
        "work_id",
        "shot_id",
        "prompt",
        "condition_path",
        "memory_slots",
        "seed",
        "width",
        "height",
        "num_frames",
        "output_dir",
    ],
    "properties": {
        "request_id": {"type": "string", "minLength": 1},
        "work_id": {"type": "string", "minLength": 1},
        "shot_id": {"type": "string", "minLength": 1},
        "prompt": {"type": "string", "minLength": 1},
        "condition_path": {"type": ["string", "null"]},
        "memory_slots": {
            "type": "array",
            "maxItems": 7,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["image_path", "audio_path", "metadata"],
                "properties": {
                    "image_path": {"type": "string", "minLength": 1},
                    "audio_path": {"type": ["string", "null"]},
                    "metadata": {"type": "object"},
                },
            },
        },
        "seed": {"type": "integer"},
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "num_frames": {"type": "integer", "minimum": 1},
        "output_dir": {"type": "string", "minLength": 1},
    },
}


def normalize_endpoint_url(url: str) -> str:
    """Return the single endpoint identity used by launch and wire receipts.

    Query strings and fragments are intentionally excluded: deployments often
    put short-lived credentials there, and neither is part of the stable
    provider identity.  A root slash, host/scheme case, and default ports are
    normalized so equivalent CLI spellings cannot create different resume
    fingerprints or ambiguous Echo routes.
    """

    value = str(url).strip()
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"endpoint must be an absolute URL: {url!r}")
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError(f"endpoint has no hostname: {url!r}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"endpoint has an invalid port: {url!r}") from exc
    host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    # Userinfo is not a stable endpoint property and may contain a secret.
    path = parsed.path.rstrip("/")
    return parsed._replace(
        scheme=scheme,
        netloc=host,
        path=path,
        params="",
        query="",
        fragment="",
    ).geturl()


def endpoint_sha256(url: str) -> str:
    return sha256_text(normalize_endpoint_url(url))


# Internal compatibility name retained for the provider implementations below.
_endpoint_sha256 = endpoint_sha256


def _response_bytes(response: Any) -> bytes:
    content = getattr(response, "content", b"")
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    return bytes(content or b"")


def _response_json(response: Any) -> dict[str, Any]:
    try:
        value = response.json()
    except Exception as exc:  # noqa: BLE001 - provider JSON errors need context
        raise ProviderProtocolError("provider returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ProviderProtocolError("provider response must be a JSON object")
    return value


def _request_header(response: Any, *names: str) -> str:
    headers = getattr(response, "headers", {}) or {}
    for name in names:
        value = headers.get(name) or headers.get(name.lower()) or headers.get(name.title())
        if value:
            return str(value)
    return ""


def _retry_after_seconds(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        seconds = parsed.timestamp() - time.time()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _is_retryable(status: int) -> bool:
    return status in _RETRYABLE_STATUS or 500 <= status <= 599


def _is_jdcloud_upstream_failure(status: int, url: str, response: Any) -> bool:
    """Recognize the observed upstream-failure envelope, not arbitrary HTTP 400."""
    if status != 400 or urlparse(url).hostname != "modelservice.jdcloud.com":
        return False
    try:
        body = response.json()
    except (TypeError, ValueError):
        return False
    error = body.get("error") if isinstance(body, Mapping) else None
    return (
        isinstance(error, Mapping)
        and error.get("code") == 400
        and error.get("status") == "FAILED_RESPONSE"
        and error.get("message") == "模型服务返回异常信息，无法处理"
        and not error.get("cause")
        and body.get("result") is None
    )


class SharedRequestRateLimiter:
    """Smooth one provider request stream and share 429 cooldowns across threads."""

    def __init__(
        self,
        max_qps: float,
        *,
        cooldown_seconds: float = 75.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(float(max_qps)) or float(max_qps) <= 0:
            raise ValueError("max_qps must be a finite positive number")
        if (
            not math.isfinite(float(cooldown_seconds))
            or float(cooldown_seconds) < 0
        ):
            raise ValueError("cooldown_seconds must be finite and nonnegative")
        self.max_qps = float(max_qps)
        self.cooldown_seconds = float(cooldown_seconds)
        self._interval = 1.0 / self.max_qps
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0
        self._cooldown_until = 0.0

    def acquire(self) -> float:
        """Reserve one evenly spaced request slot and return seconds waited."""

        started = self._clock()
        while True:
            with self._lock:
                now = self._clock()
                allowed_at = max(self._next_allowed_at, self._cooldown_until)
                delay = allowed_at - now
                if delay <= 0:
                    self._next_allowed_at = now + self._interval
                    return max(0.0, self._clock() - started)
            self._sleep(delay)

    def extend_cooldown(self, seconds: float | None = None) -> float:
        """Atomically extend the cooldown seen by every sharing client."""

        delay = self.cooldown_seconds if seconds is None else float(seconds)
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("cooldown delay must be finite and nonnegative")
        with self._lock:
            now = self._clock()
            self._cooldown_until = max(self._cooldown_until, now + delay)
            self._next_allowed_at = max(
                self._next_allowed_at, self._cooldown_until
            )
            return max(0.0, self._cooldown_until - now)


class _HTTPClient:
    def __init__(
        self,
        *,
        timeout: float,
        max_attempts: int,
        backoff_seconds: float,
        session: Any | None,
        sleep: Callable[[float], None],
        rate_limiter: SharedRequestRateLimiter | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.timeout = float(timeout)
        self.max_attempts = int(max_attempts)
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        # Supplied sessions retain their caller-owned adapters and settings.
        self.session = session if session is not None else create_provider_session()
        self.sleep = sleep
        self.rate_limiter = rate_limiter

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_payload: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        retryable_statuses: frozenset[int] = frozenset(),
        allow_redirects: bool = False,
    ) -> tuple[Any, list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            rate_limit_wait = (
                self.rate_limiter.acquire() if self.rate_limiter is not None else 0.0
            )
            started = time.monotonic()
            trace_id = str(uuid.uuid4())
            request_headers = dict(headers or {})
            request_headers.setdefault("Trace-Id", trace_id)
            try:
                response = self.session.request(
                    method,
                    url,
                    headers=request_headers,
                    json=dict(json_payload) if json_payload is not None else None,
                    timeout=float(timeout or self.timeout),
                    allow_redirects=allow_redirects,
                )
            except (requests.RequestException, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                record = {
                    "attempt": attempt,
                    "trace_id": trace_id,
                    "http_status": 0,
                    "timeout_seconds": float(timeout or self.timeout),
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "retryable": True,
                }
                if self.rate_limiter is not None:
                    record["rate_limit_wait_seconds"] = round(
                        rate_limit_wait, 6
                    )
                attempts.append(record)
                if attempt >= self.max_attempts:
                    break
                delay = self.backoff_seconds * (2 ** (attempt - 1))
                record["retry_delay_seconds"] = delay
                self.sleep(delay)
                continue

            status = int(getattr(response, "status_code", 0))
            response_content = _response_bytes(response)
            record = {
                "attempt": attempt,
                "trace_id": trace_id,
                "http_status": status,
                "timeout_seconds": float(timeout or self.timeout),
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "response_sha256": sha256_bytes(response_content),
                "response_size_bytes": len(response_content),
                "response_request_id": _request_header(
                    response, "x-request-id", "request-id", "x-jdcloud-request-id"
                ),
                "redirect_count": len(getattr(response, "history", []) or []),
                "final_url_sha256": _endpoint_sha256(str(getattr(response, "url", url) or url)),
                "retryable": False,
            }
            if self.rate_limiter is not None:
                record["rate_limit_wait_seconds"] = round(rate_limit_wait, 6)
            attempts.append(record)
            if 200 <= status < 300:
                return response, attempts
            upstream_failure = _is_jdcloud_upstream_failure(status, url, response)
            retryable = _is_retryable(status) or status in retryable_statuses or upstream_failure
            record["retryable"] = retryable
            if upstream_failure:
                record["retry_reason"] = "jdcloud_upstream_failed_response"
            detail = getattr(response, "text", "") or ""
            # Some gateways echo the rejected credential in their JSON error
            # body.  Never propagate that value into exceptions, Ray logs, or
            # persisted rollout receipts.
            if self.api_key:
                detail = str(detail).replace(self.api_key, "[REDACTED_API_KEY]")
            retry_after = _retry_after_seconds(_request_header(response, "Retry-After"))
            delay = max(self.backoff_seconds * (2 ** (attempt - 1)), retry_after or 0.0)
            if upstream_failure:
                delay = max(delay, 10.0 * (2 ** (attempt - 1)))
            record["retry_after_seconds"] = retry_after
            record["retry_delay_seconds"] = delay
            if status == 429 and self.rate_limiter is not None:
                shared_delay = self.rate_limiter.extend_cooldown(
                    max(self.rate_limiter.cooldown_seconds, delay)
                )
                record["shared_cooldown_seconds"] = round(shared_delay, 6)
            # ProviderError snapshots provenance on construction.  Build it
            # only after the retry/cooldown receipt is complete so a terminal
            # 429 remains auditable even when max_attempts == 1.
            last_error = ProviderHTTPError(
                f"HTTP {status} from provider: {str(detail)[:1000]}",
                provenance={"attempts": attempts},
            )
            if not retryable or attempt >= self.max_attempts:
                break
            if status == 429 and self.rate_limiter is not None:
                # The next acquire observes the shared deadline. Sleeping here
                # as well would double-apply the same cooldown.
                continue
            self.sleep(delay)
        if isinstance(last_error, ProviderHTTPError):
            raise last_error
        raise ProviderHTTPError(
            f"provider request failed after {len(attempts)} attempt(s): {last_error}",
            provenance={"attempts": attempts},
        ) from last_error


def _function_definitions(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(tools):
        if not isinstance(raw, Mapping):
            raise TypeError(f"tools[{index}] must be an object")
        function = raw.get("function") if raw.get("type") == "function" else raw
        if not isinstance(function, Mapping):
            raise ValueError(f"tools[{index}] has no function definition")
        name = str(function.get("name") or "")
        if not name or name in names:
            raise ValueError(f"tools[{index}] has an empty or duplicate name: {name!r}")
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"tools[{index}].parameters must be an object")
        definition = {
            "name": name,
            "description": str(function.get("description") or ""),
            "parameters": copy.deepcopy(dict(parameters)),
        }
        if "strict" in function:
            definition["strict"] = bool(function["strict"])
        definitions.append(definition)
        names.add(name)
    if not definitions:
        raise ValueError("tools must contain at least one native function definition")
    return definitions


def _generic_user_items(content: Any) -> list[Any]:
    if isinstance(content, (str, Path, MediaInput)):
        return [content]
    if isinstance(content, Mapping):
        if "type" in content or "text" in content or "path" in content or "inlineData" in content:
            return [content]
        return [canonical_json(content)]
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        return list(content)
    raise TypeError("user_content must be text, a JSON object, media, or a sequence of those")


def _openai_user_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for item in _generic_user_items(content):
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, (Path, MediaInput)):
            blocks.append(content_block(item, api_style="openai_chat"))
        elif isinstance(item, Mapping):
            item_type = str(item.get("type") or "")
            if item_type == "text" or ("text" in item and not item_type):
                blocks.append({"type": "text", "text": str(item.get("text") or "")})
            elif item_type == "media" or "path" in item:
                blocks.append(
                    content_block(
                        MediaInput(
                            path=str(item.get("path") or ""),
                            mime_type=str(item.get("mime_type") or "") or None,
                            detail=str(item.get("detail") or "") or None,
                        ),
                        api_style="openai_chat",
                    )
                )
            elif item_type in {"image_url", "video_url", "input_audio", "file_url"}:
                blocks.append(copy.deepcopy(dict(item)))
            else:
                blocks.append({"type": "text", "text": canonical_json(item)})
        else:
            raise TypeError(f"unsupported user content item: {type(item).__name__}")
    if not blocks:
        raise ValueError("user_content must not be empty")
    return blocks


def _gemini_user_parts(content: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for item in _generic_user_items(content):
        if isinstance(item, str):
            parts.append({"text": item})
        elif isinstance(item, (Path, MediaInput)):
            parts.append(content_block(item, api_style="gemini"))
        elif isinstance(item, Mapping):
            item_type = str(item.get("type") or "")
            if item_type == "text" or ("text" in item and not item_type):
                parts.append({"text": str(item.get("text") or "")})
            elif item_type == "media" or "path" in item:
                parts.append(
                    content_block(
                        MediaInput(
                            path=str(item.get("path") or ""),
                            mime_type=str(item.get("mime_type") or "") or None,
                        ),
                        api_style="gemini",
                    )
                )
            elif "inlineData" in item or "inline_data" in item:
                inline = item.get("inlineData") or item.get("inline_data")
                if not isinstance(inline, Mapping):
                    raise ValueError("inline media part must be an object")
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": str(inline.get("mimeType") or inline.get("mime_type") or ""),
                            "data": str(inline.get("data") or ""),
                        }
                    }
                )
            else:
                parts.append({"text": canonical_json(item)})
        else:
            raise TypeError(f"unsupported user content item: {type(item).__name__}")
    if not parts:
        raise ValueError("user_content must not be empty")
    return parts


def _is_teacher_dialogue(content: Any) -> bool:
    """Return whether ``content`` is a provider-neutral ordinary dialogue.

    A list is also the public representation for one multimodal USER message,
    so it counts as dialogue only when every item explicitly carries a chat
    role.  There is intentionally no ``tool`` role in this transport.
    """

    return (
        isinstance(content, Sequence)
        and not isinstance(content, (str, bytes, bytearray))
        and bool(content)
        and all(
            isinstance(item, Mapping)
            and item.get("role") in {"user", "assistant"}
            for item in content
        )
    )


def _teacher_dialogue(content: Any) -> list[dict[str, Any]]:
    """Validate or wrap a closed-Teacher JSON-text conversation.

    Tool actions are assistant JSON strings and tool observations are ordinary
    USER messages.  Requiring strict alternation makes it impossible for a
    stale native ``tool_calls``/``tool`` transcript to leak onto the wire.
    """

    looks_like_chat = (
        isinstance(content, Sequence)
        and not isinstance(content, (str, bytes, bytearray))
        and any(isinstance(item, Mapping) and "role" in item for item in content)
    )
    if looks_like_chat and not _is_teacher_dialogue(content):
        raise ValueError(
            "Teacher history must use only ordinary user/assistant dialogue roles"
        )
    if not _is_teacher_dialogue(content):
        return [{"role": "user", "content": copy.deepcopy(content)}]

    messages = [copy.deepcopy(dict(item)) for item in content]
    if messages[0].get("role") != "user":
        raise ValueError("Teacher dialogue must begin with the raw USER request")
    if messages[-1].get("role") != "user":
        raise ValueError("Teacher dialogue must end with a USER tool observation")
    for index, message in enumerate(messages):
        extra = set(message) - {"role", "content"}
        if extra:
            raise ValueError(f"messages[{index}] has unknown fields: {sorted(extra)}")
        expected_role = "user" if index % 2 == 0 else "assistant"
        role = message.get("role")
        if role != expected_role:
            raise ValueError(
                f"messages[{index}] must have role {expected_role!r}, got {role!r}"
            )
        if "content" not in message:
            raise ValueError(f"messages[{index}] has no content")
        if role == "assistant":
            raw_text = message["content"]
            if not isinstance(raw_text, str):
                raise ValueError(
                    f"messages[{index}] assistant content must be raw JSON text"
                )
            # Historical commands were admitted by the runtime state of their
            # own turn.  The action set passed to ``call`` belongs only to the
            # new response and must not be retroactively applied here.
            _parse_plain_json_tool_command_text(raw_text, allowed_names=None)
    return messages


def _openai_conversation(content: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in _teacher_dialogue(content):
        role = str(message["role"])
        if role == "assistant":
            result.append({"role": "assistant", "content": message["content"]})
        else:
            result.append(
                {"role": "user", "content": _openai_user_content(message["content"])}
            )
    return result


def _gemini_conversation(content: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in _teacher_dialogue(content):
        role = str(message["role"])
        if role == "assistant":
            parts = [{"text": message["content"]}]
            wire_role = "model"
        else:
            parts = _gemini_user_parts(message["content"])
            wire_role = "user"
        result.append({"role": wire_role, "parts": parts})
    return result


def _strict_json_object(text: str) -> dict[str, Any]:
    if not text.strip():
        raise ProviderProtocolError("teacher JSON response is empty")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text.strip(),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProviderProtocolError("teacher response is not one strict JSON value") from exc
    if not isinstance(value, dict):
        raise ProviderProtocolError("teacher JSON response must be an object")
    return value


def _parse_plain_json_tool_command_text(
    text: str, *, allowed_names: set[str] | None
) -> tuple[str, str, dict[str, Any]]:
    """Parse the sole accepted closed-Teacher response envelope."""

    value = _strict_json_object(text)
    if set(value) != {"reason", "tool_call"}:
        raise ProviderProtocolError(
            "teacher JSON response must contain exactly reason and tool_call"
        )
    reason = value.get("reason")
    raw_call = value.get("tool_call")
    if not isinstance(reason, str):
        raise ProviderProtocolError("teacher response reason must be a string")
    if not isinstance(raw_call, dict) or set(raw_call) != {"name", "arguments"}:
        raise ProviderProtocolError(
            "teacher tool_call must contain exactly name and arguments"
        )
    name = raw_call.get("name")
    arguments = raw_call.get("arguments")
    if not isinstance(name, str) or not name:
        raise ProviderProtocolError("teacher tool_call name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise ProviderProtocolError("teacher tool_call arguments must be a JSON object")
    if allowed_names is not None and name not in allowed_names:
        raise ProviderProtocolError(f"teacher returned unknown tool {name!r}")
    try:
        json.dumps(arguments, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProviderProtocolError(
            "teacher tool_call arguments are not strict JSON"
        ) from exc
    return reason, name, copy.deepcopy(arguments)


class RemoteTeacherClient(_HTTPClient):
    """Stateless ordinary-dialogue client for a closed multimodal Teacher.

    The remote API is never asked to perform provider-native function calls.
    It returns one strict JSON object in text; canonical tools are supplied to
    :meth:`call` solely for local name validation and an audit hash.
    """

    def __init__(
        self,
        api_url: str,
        *,
        api_key: str | None,
        model: str,
        api_style: str = "openai_chat",
        timeout: float = 180.0,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        max_output_tokens: int = 8192,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        extra_headers: Mapping[str, str] | None = None,
        rate_limiter: SharedRequestRateLimiter | None = None,
    ) -> None:
        super().__init__(
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            session=session,
            sleep=sleep,
            rate_limiter=rate_limiter,
        )
        style = str(api_style).strip().lower()
        if style not in _OPENAI_STYLES | _GEMINI_STYLES:
            raise ValueError(f"unsupported teacher API style: {api_style!r}")
        if not str(api_url).strip() or not str(model).strip():
            raise ValueError("api_url and model are required")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self.api_url = str(api_url).rstrip("/")
        self.api_key = str(api_key or "")
        self.model = str(model)
        self.api_style = "openai_chat" if style in _OPENAI_STYLES else "gemini"
        self.max_output_tokens = int(max_output_tokens)
        self.extra_headers = dict(extra_headers or {})

    @classmethod
    def from_env(cls, **overrides: Any) -> "RemoteTeacherClient":
        base = os.getenv("RECAPTION_OPENAI_BASE_URL", "https://modelservice.jdcloud.com/v1")
        values = {
            "api_url": os.getenv("RECAPTION_URL") or f"{base.rstrip('/')}/chat/completions",
            "api_key": os.getenv("RECAPTION_API_KEY") or os.getenv("API_KEY"),
            "model": os.getenv("RECAPTION_MODEL", "Gemini-3.5-Flash-Third"),
            "api_style": os.getenv("RECAPTION_API_STYLE", "openai_chat"),
        }
        values.update(overrides)
        return cls(**values)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(
        self,
        system_prompt: str,
        user_content: Any,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValueError("system_prompt must be a non-empty string")
        if self.api_style in _OPENAI_STYLES:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *_openai_conversation(user_content),
                ],
                "temperature": float(temperature),
                "max_tokens": int(max_output_tokens),
                "stream": False,
                "response_format": {"type": "json_object"},
            }
            return payload

        payload = {
            "model": self.model,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": _gemini_conversation(user_content),
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_output_tokens),
                "responseMimeType": "application/json",
            },
        }
        return payload

    def _invoke(self, payload: Mapping[str, Any], *, timeout: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        response, attempts = self._request(
            "POST",
            self.api_url,
            headers=self._headers(),
            json_payload=payload,
            timeout=timeout,
        )
        response_content = _response_bytes(response)
        provenance = {
            "assistant_source": "remote_closed_model_api",
            "api_style": self.api_style,
            "model": self.model,
            "endpoint_sha256": _endpoint_sha256(self.api_url),
            "request_payload_sha256": sha256_json(payload),
            "request_sha256": sha256_json(payload),
            "response_sha256": sha256_bytes(response_content),
            "response_id": "",
            "response_request_id": _request_header(
                response, "x-request-id", "request-id", "x-jdcloud-request-id"
            ),
            "returned_model": "",
            "usage": {},
            "attempts": attempts,
        }
        try:
            body = _response_json(response)
        except ProviderProtocolError as exc:
            raise ProviderProtocolError(str(exc), provenance=provenance) from exc
        provenance.update(
            {
                "response_id": str(
                    body.get("id")
                    or body.get("responseId")
                    or body.get("response_id")
                    or ""
                ),
                "returned_model": str(
                    body.get("model") or body.get("modelVersion") or ""
                ),
                "usage": copy.deepcopy(
                    body.get("usage") or body.get("usageMetadata") or {}
                ),
            }
        )
        return body, provenance

    def _assistant_text(
        self, body: Mapping[str, Any], provenance: Mapping[str, Any]
    ) -> str:
        """Extract the exact logical assistant text from one JSON response."""

        def no_text_error(message: str) -> ProviderProtocolError:
            failed_provenance = copy.deepcopy(dict(provenance))
            failed_provenance["raw_assistant_text"] = ""
            failed_provenance["assistant_text_sha256"] = sha256_text("")
            return ProviderProtocolError(message, provenance=failed_provenance)

        if self.api_style in _OPENAI_STYLES:
            choices = body.get("choices")
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], Mapping)
            ):
                raise no_text_error(
                    "OpenAI JSON response must contain exactly one choice"
                )
            message = choices[0].get("message")
            raw_content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(raw_content, list):
                text = "".join(
                    str(item.get("text") or "")
                    for item in raw_content
                    if isinstance(item, Mapping)
                    and item.get("type") in {None, "text", "output_text"}
                )
            elif isinstance(raw_content, str):
                text = raw_content
            else:
                text = ""
        else:
            candidates = body.get("candidates")
            if (
                not isinstance(candidates, list)
                or len(candidates) != 1
                or not isinstance(candidates[0], Mapping)
            ):
                raise no_text_error(
                    "Gemini JSON response must contain exactly one candidate"
                )
            candidate_content = candidates[0].get("content")
            parts = (
                candidate_content.get("parts")
                if isinstance(candidate_content, Mapping)
                else None
            )
            text = "".join(
                str(item.get("text"))
                for item in (parts if isinstance(parts, list) else [])
                if isinstance(item, Mapping) and item.get("text") is not None
            )
        if not text.strip():
            raise no_text_error("teacher JSON response has no assistant text")
        return text

    def call(
        self,
        system_prompt: str,
        user_content: Any,
        tools: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> ToolCall:
        definitions = _function_definitions(tools)
        allowed_names = {item["name"] for item in definitions}
        payload = self._payload(
            system_prompt,
            user_content,
            temperature=temperature,
            max_output_tokens=max_output_tokens or self.max_output_tokens,
        )
        # Seal the exact canonical definitions used for local validation.  The
        # schemas are intentionally absent from the provider request itself.
        schema_provenance = {
            "tool_schema_sha256": sha256_json(
                [{"type": "function", "function": item} for item in definitions]
            ),
            "teacher_transport": _PLAIN_JSON_TOOL_TRANSPORT,
            "plain_json_tool_command": False,
        }
        try:
            body, provenance = self._invoke(payload)
        except ProviderProtocolError as exc:
            failed_provenance = copy.deepcopy(exc.provenance)
            failed_provenance.update(schema_provenance)
            failed_provenance.setdefault("raw_assistant_text", "")
            failed_provenance.setdefault(
                "assistant_text_sha256", sha256_text("")
            )
            raise ProviderProtocolError(
                str(exc), provenance=failed_provenance
            ) from exc
        provenance.update(schema_provenance)
        try:
            text = self._assistant_text(body, provenance)
        except ProviderProtocolError as exc:
            failed_provenance = copy.deepcopy(exc.provenance)
            failed_provenance.update(schema_provenance)
            raise ProviderProtocolError(str(exc), provenance=failed_provenance) from exc
        provenance["raw_assistant_text"] = text
        provenance["assistant_text_sha256"] = sha256_text(text)
        try:
            reason, name, arguments = _parse_plain_json_tool_command_text(
                text, allowed_names=allowed_names
            )
        except ProviderProtocolError as exc:
            raise ProviderProtocolError(str(exc), provenance=provenance) from exc
        provenance["plain_json_tool_command"] = True
        return ToolCall(
            name=name,
            arguments=arguments,
            reason=reason,
            provenance=provenance,
        )

    def call_json(
        self,
        system_prompt: str,
        user_content: Any,
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        timeout: float | None = None,
    ) -> JsonCallResult:
        payload = self._payload(
            system_prompt,
            user_content,
            temperature=temperature,
            max_output_tokens=max_output_tokens or self.max_output_tokens,
        )
        body, provenance = self._invoke(payload, timeout=timeout) if timeout is not None else self._invoke(payload)
        text = self._assistant_text(body, provenance)
        provenance["raw_assistant_text"] = text
        provenance["assistant_text_sha256"] = sha256_text(text)
        try:
            value = _strict_json_object(text)
        except ProviderProtocolError as exc:
            raise ProviderProtocolError(str(exc), provenance=provenance) from exc
        return JsonCallResult(value=value, provenance=provenance)


def _choose_ratio(width: int, height: int, choices: Sequence[tuple[str, float]]) -> str:
    ratio = float(width) / float(height)
    return min(choices, key=lambda item: abs(item[1] - ratio))[0]


def _seedream_size(width: int, height: int, *, model: str = "") -> str:
    # Seedream 4.5 and 5.0 reject images below 3,686,400 pixels (the 2560x1440
    # baseline).  Preserve the caller's aspect ratio with the same explicit
    # width×height contract used by the proven production adapter; a generic
    # ``2K`` preset may silently choose a different aspect ratio from a
    # reference image.  Earlier Seedream generations retain their 1K floor.
    normalized_model = str(model).casefold()
    minimum_pixels = (
        3_686_400
        if any(version in normalized_model for version in ("4-5", "4.5", "5-0", "5.0"))
        else 1024 * 1024
    )
    scale = max(1.0, math.sqrt(minimum_pixels / float(width * height)))
    scaled_width = int(math.ceil(width * scale / 2.0) * 2)
    scaled_height = int(math.ceil(height * scale / 2.0) * 2)
    return f"{scaled_width}x{scaled_height}"


def _decode_data_url(value: str) -> bytes:
    header, separator, encoded = value.partition(",")
    if not separator or ";base64" not in header.lower():
        raise ProviderProtocolError("provider returned an invalid data URL")
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ProviderProtocolError("provider returned invalid base64 media") from exc


def _audit_safe_payload(value: Any, *, key: str | None = None) -> Any:
    """Replace inline media bytes with hashes before persisting a request."""

    if isinstance(value, str) and value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        if separator and ";base64" in header.lower():
            raw = base64.b64decode(encoded)
            return {
                "$inline_media": {
                    "mime_type": header[5:].split(";", 1)[0],
                    "sha256": sha256_bytes(raw),
                    "size_bytes": len(raw),
                }
            }
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        mime = str(value.get("mimeType") or value.get("mime_type") or "")
        for child_key, child in value.items():
            if child_key == "data" and mime and isinstance(child, str):
                try:
                    raw = base64.b64decode(child, validate=True)
                except Exception:  # noqa: BLE001
                    output[child_key] = "<invalid-inline-base64>"
                else:
                    output[child_key] = {
                        "$inline_media": {
                            "mime_type": mime,
                            "sha256": sha256_bytes(raw),
                            "size_bytes": len(raw),
                        }
                    }
            else:
                output[str(child_key)] = _audit_safe_payload(child, key=str(child_key))
        return output
    if isinstance(value, list):
        return [_audit_safe_payload(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [_audit_safe_payload(item, key=key) for item in value]
    return copy.deepcopy(value)


def _image_response_source(body: Mapping[str, Any]) -> tuple[bytes | None, str | None]:
    data = body.get("data")
    if isinstance(data, list) and data and isinstance(data[0], Mapping):
        item = data[0]
        encoded = item.get("b64_json") or item.get("b64") or item.get("base64")
        if isinstance(encoded, str) and encoded:
            return (_decode_data_url(encoded) if encoded.startswith("data:") else base64.b64decode(encoded), None)
        if isinstance(item.get("url"), str) and item.get("url"):
            return None, str(item["url"])
    candidates = body.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            content = candidate.get("content") if isinstance(candidate, Mapping) else None
            parts = content.get("parts") if isinstance(content, Mapping) else None
            for part in parts if isinstance(parts, list) else []:
                inline = part.get("inlineData") or part.get("inline_data") if isinstance(part, Mapping) else None
                if isinstance(inline, Mapping) and inline.get("data"):
                    return base64.b64decode(str(inline["data"])), None
    for key in ("image_url", "output_url", "url"):
        if isinstance(body.get(key), str) and body.get(key):
            return None, str(body[key])
    raise ProviderProtocolError("image provider response contains no image bytes or URL")


def _response_id(body: Mapping[str, Any]) -> str:
    return str(
        body.get("id")
        or body.get("response_id")
        or body.get("responseId")
        or body.get("task_id")
        or body.get("version_id")
        or ""
    )


def _assert_no_reported_prompt_rewrite(body: Mapping[str, Any], prompt: str) -> None:
    candidates: list[Mapping[str, Any]] = [body]
    data = body.get("data")
    if isinstance(data, list):
        candidates.extend(item for item in data if isinstance(item, Mapping))
    for item in candidates:
        for key in ("revised_prompt", "optimized_prompt", "rewritten_prompt"):
            value = item.get(key)
            if isinstance(value, str) and value != prompt:
                raise ProviderProtocolError(
                    f"provider reported a hidden prompt rewrite in {key}; canonical generation aborted"
                )


def _fit_image(raw: bytes, *, width: int, height: int) -> tuple[bytes, dict[str, Any]]:
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            oriented = ImageOps.exif_transpose(opened)
            source_size = oriented.size
            if oriented.mode not in {"RGB", "RGBA"}:
                oriented = oriented.convert("RGB")
            scale = max(width / oriented.width, height / oriented.height)
            resized_size = (
                max(width, int(math.ceil(oriented.width * scale))),
                max(height, int(math.ceil(oriented.height * scale))),
            )
            resized = oriented.resize(resized_size, Image.Resampling.LANCZOS)
            left = (resized.width - width) // 2
            top = (resized.height - height) // 2
            fitted = resized.crop((left, top, left + width, top + height))
            output = io.BytesIO()
            fitted.save(output, format="PNG", optimize=False)
    except Exception as exc:  # noqa: BLE001
        raise ProviderProtocolError("image provider returned undecodable image bytes") from exc
    transform = {
        "method": "aspect_preserving_scale_then_center_crop",
        "source_size": list(source_size),
        "resized_size": list(resized_size),
        "crop_box": [left, top, left + width, top + height],
        "output_size": [width, height],
        "resampler": "PIL.Image.Resampling.LANCZOS",
    }
    return output.getvalue(), transform


class GenericImageProvider(_HTTPClient):
    """One provider-neutral image interface with explicit concrete adapters."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str | None,
        model: str,
        adapter: str,
        timeout: float = 360.0,
        max_attempts: int = 3,
        backoff_seconds: float = 1.5,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            session=session,
            sleep=sleep,
        )
        normalized = str(adapter).strip().lower()
        aliases = {
            "seedream_image_edit": "seedream",
            "openai": "openai_image",
            "openai_image_json": "openai_image",
            "gemini_flash": "gemini",
            "gemini_flash_image": "gemini",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"seedream", "openai_image", "gemini"}:
            raise ValueError(f"unsupported image adapter: {adapter!r}")
        if not str(api_url).strip() or not str(model).strip():
            raise ValueError("api_url and model are required")
        self.api_url = str(api_url)
        self.api_key = str(api_key or "")
        self.model = str(model)
        self.adapter = normalized
        self.extra_headers = dict(extra_headers or {})

    @classmethod
    def from_env(cls, **overrides: Any) -> "GenericImageProvider":
        adapter = os.getenv("GEN_API_STYLE", "gemini")
        base_url = os.getenv("BASE_URL", "https://modelservice.jdcloud.com/v1").rstrip("/")
        normalized = {
            "seedream_image_edit": "seedream",
            "openai": "openai_image",
            "openai_image_json": "openai_image",
            "gemini_flash": "gemini",
            "gemini_flash_image": "gemini",
        }.get(adapter.strip().lower(), adapter.strip().lower())
        if normalized == "seedream":
            api_url = os.getenv("SEEDREAM_IMAGE_URL", f"{base_url}/imageEdit/generations")
        elif normalized == "openai_image":
            api_url = os.getenv("OPENAI_IMAGE_URL", f"{base_url}/images/generations")
        else:
            api_url = os.getenv("GEN_URL", f"{base_url}/images/gemini_flash/generations")
        values = {
            "api_url": api_url,
            "api_key": os.getenv("GEN_API_KEY") or os.getenv("API_KEY"),
            "model": os.getenv("GEN_MODEL", ""),
            "adapter": adapter,
        }
        values.update(overrides)
        return cls(**values)

    def _headers(self, request_id: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": request_id,
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _validate(request: ImageGenerationRequest) -> list[Path]:
        if not str(request.request_id).strip():
            raise ValueError("request_id is required")
        if not isinstance(request.prompt, str) or not request.prompt:
            raise ValueError("prompt must be a non-empty string")
        if request.prompt != request.prompt.strip():
            raise ValueError("prompt has leading/trailing whitespace; provider will not normalize it")
        if int(request.width) <= 0 or int(request.height) <= 0:
            raise ValueError("width and height must be positive")
        refs = [require_file(path, label=f"reference_paths[{index}]") for index, path in enumerate(request.reference_paths)]
        for path in refs:
            if not detect_mime(path).startswith("image/"):
                raise ValueError(f"reference is not an image: {path}")
        return refs

    def build_payload(self, request: ImageGenerationRequest) -> dict[str, Any]:
        refs = self._validate(request)
        if self.adapter == "seedream":
            payload: dict[str, Any] = {
                "model": self.model,
                "prompt": request.prompt,
                "size": _seedream_size(
                    int(request.width), int(request.height), model=self.model
                ),
                "response_format": "url",
                "watermark": False,
            }
            if refs:
                payload["image"] = [data_uri(path) for path in refs]
            if request.seed is not None:
                payload["seed"] = int(request.seed)
            # Deliberately no optimize_prompt_options: the legacy path enabled
            # it and silently changed the registered prompt.
            return payload
        if self.adapter == "openai_image":
            size = _choose_ratio(
                int(request.width),
                int(request.height),
                (("1024x1024", 1.0), ("1536x1024", 1.5), ("1024x1536", 2 / 3)),
            )
            payload = {
                "model": self.model,
                "prompt": request.prompt,
                "size": size,
                "n": 1,
                "response_format": "b64_json",
            }
            if refs:
                encoded = [data_uri(path) for path in refs]
                payload["image"] = encoded if len(encoded) > 1 else encoded[0]
                payload["input_fidelity"] = "high"
            if request.seed is not None:
                payload["seed"] = int(request.seed)
            return payload
        ratio = _choose_ratio(
            int(request.width),
            int(request.height),
            (
                ("1:1", 1.0),
                ("16:9", 16 / 9),
                ("9:16", 9 / 16),
                ("4:3", 4 / 3),
                ("3:4", 3 / 4),
                ("3:2", 3 / 2),
                ("2:3", 2 / 3),
                ("21:9", 21 / 9),
            ),
        )
        parts = []
        for path in refs:
            media = describe_media(path)
            parts.append(
                {
                    "inline_data": {
                        "mimeType": media.mime_type,
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    }
                }
            )
        parts.append({"text": request.prompt})
        generation_config: dict[str, Any] = {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": ratio},
        }
        if request.seed is not None:
            generation_config["seed"] = int(request.seed)
        return {
            "model": self.model,
            "contents": [{"role": "USER", "parts": parts}],
            "generation_config": {
                "response_modalities": generation_config["responseModalities"],
                "image_config": {
                    "aspect_ratio": generation_config["imageConfig"]["aspectRatio"]
                },
                **({"seed": generation_config["seed"]} if "seed" in generation_config else {}),
            },
        }

    def _download(self, url: str) -> tuple[bytes, list[dict[str, Any]], str]:
        if url.startswith("data:"):
            return _decode_data_url(url), [], "data_url"
        response, attempts = self._request(
            "GET", url, headers=self.extra_headers, allow_redirects=True
        )
        payload = _response_bytes(response)
        if not payload:
            raise ProviderProtocolError("image download returned empty bytes")
        return payload, attempts, _request_header(response, "content-type")

    def generate(self, request: ImageGenerationRequest) -> ProviderResult:
        refs = self._validate(request)
        payload = self.build_payload(request)
        response, attempts = self._request(
            "POST",
            self.api_url,
            headers=self._headers(str(request.request_id)),
            json_payload=payload,
        )
        body = _response_json(response)
        _assert_no_reported_prompt_rewrite(body, request.prompt)
        raw, url = _image_response_source(body)
        download_attempts: list[dict[str, Any]] = []
        transport = "inline_base64"
        if raw is None:
            assert url is not None
            raw, download_attempts, _ = self._download(url)
            transport = "download_url"
        if not raw:
            raise ProviderProtocolError("image provider returned empty image bytes")
        fitted, transform = _fit_image(raw, width=int(request.width), height=int(request.height))
        output_dir = Path(request.output_dir).expanduser().resolve()
        output_path = atomic_write_bytes(output_dir / f"{safe_name(request.request_id)}.png", fitted)
        provenance = {
            "provider_kind": "image",
            "adapter": self.adapter,
            "model": self.model,
            "endpoint_sha256": _endpoint_sha256(self.api_url),
            "request_id": str(request.request_id),
            "request_payload_sha256": sha256_json(payload),
            "prompt_sha256": sha256_text(request.prompt),
            "prompt_transport": "verbatim",
            "hidden_prompt_rewrite_enabled": False,
            "reference_inputs": [describe_media(path).to_dict() for path in refs],
            "response_sha256": sha256_bytes(_response_bytes(response)),
            "provider_image_sha256": sha256_bytes(raw),
            "output_sha256": sha256_file(output_path),
            "output_mime_type": detect_mime(output_path),
            "output_transform": transform,
            "output_transport": transport,
            "response_request_id": _request_header(
                response, "x-request-id", "request-id", "x-jdcloud-request-id"
            ),
            "http_attempts": attempts,
            "download_attempts": download_attempts,
        }
        return ProviderResult(
            output_path=str(output_path),
            request_payload=_audit_safe_payload(payload),
            request_sha256=sha256_json(payload),
            request_id=str(request.request_id),
            response_id=_response_id(body),
            provenance=provenance,
        )


class EchoVideoProvider(_HTTPClient):
    """Direct, fail-closed Echo 1.5 ``POST /r2v`` adapter."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 7200.0,
        request_timeout: float = 120.0,
        poll_seconds: float = 5.0,
        max_attempts: int = 3,
        max_terminal_attempts: int = 3,
        backoff_seconds: float = 1.0,
        media_transport: str = "data_uri",
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            timeout=request_timeout,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            session=session,
            sleep=sleep,
        )
        if not str(base_url).strip():
            raise ValueError("base_url is required")
        if timeout <= 0 or poll_seconds < 0:
            raise ValueError("timeout must be positive and poll_seconds non-negative")
        if int(max_terminal_attempts) < 1:
            raise ValueError("max_terminal_attempts must be at least one")
        if media_transport not in {"data_uri", "path"}:
            raise ValueError("media_transport must be data_uri or path")
        self.base_url = str(base_url).rstrip("/")
        self.generation_timeout = float(timeout)
        self.poll_seconds = float(poll_seconds)
        self.media_transport = media_transport
        self.max_terminal_attempts = int(max_terminal_attempts)
        self.extra_headers = dict(extra_headers or {})

    @classmethod
    def from_env(cls, **overrides: Any) -> "EchoVideoProvider":
        base_url = os.getenv("ECHO15_R2V_SERVER_URL") or os.getenv("ECHO_R2V_SERVER_URL") or ""
        values = {"base_url": base_url}
        values.update(overrides)
        return cls(**values)

    def _media_value(self, path: str | Path, *, expected: str, label: str) -> tuple[str, dict[str, Any]]:
        source = require_file(path, label=label)
        descriptor = describe_media(source)
        if not descriptor.mime_type.startswith(f"{expected}/"):
            raise ValueError(f"{label} must be {expected} media, got {descriptor.mime_type}")
        value = data_uri(source, mime_type=descriptor.mime_type) if self.media_transport == "data_uri" else str(source)
        return value, descriptor.to_dict()

    def validate_endpoint(
        self,
        *,
        tokenizer_profile_sha256: str,
        schema_version: str = "echo15.r2v.v1",
        max_prompt_chars_exclusive: int | None = None,
        max_prompt_tokens: int = 1024,
        max_memory_slots: int = 7,
        require_audio_memory: bool = False,
    ) -> dict[str, Any]:
        if (
            not isinstance(tokenizer_profile_sha256, str)
            or not tokenizer_profile_sha256.startswith("sha256:")
            or len(tokenizer_profile_sha256) != len("sha256:") + 64
            or any(
                character not in "0123456789abcdef"
                for character in tokenizer_profile_sha256.removeprefix("sha256:")
            )
        ):
            raise ValueError("tokenizer_profile_sha256 must be a lowercase SHA-256 digest")
        response, attempts = self._request(
            "GET", f"{self.base_url}/health", headers=self.extra_headers
        )
        body = _response_json(response)
        inference = body.get("inference")
        workers = inference.get("workers") if isinstance(inference, Mapping) else None
        scheduler = body.get("scheduler")
        scheduler_workers = (
            scheduler.get("gpu_workers") if isinstance(scheduler, Mapping) else None
        )
        advertised_ready_workers = (
            inference.get("ready_workers") if isinstance(inference, Mapping) else None
        )
        unready_states = {
            "starting",
            "resource_error",
            "waiting_resources",
            "failed",
        }
        ready_worker_count = sum(
            1
            for worker in workers or []
            if isinstance(worker, Mapping)
            and worker.get("ready") is not False
            and not worker.get("error")
            and bool(str(worker.get("state") or ""))
            and str(worker.get("state") or "") not in unready_states
        )
        worker_errors = [
            copy.deepcopy(worker.get("error"))
            for worker in workers or []
            if isinstance(worker, Mapping) and worker.get("error")
        ]
        ready = (
            body.get("status") == "ok"
            and isinstance(inference, Mapping)
            and inference.get("enabled") is True
            and isinstance(workers, list)
            and bool(workers)
            and ready_worker_count >= 1
            and (
                advertised_ready_workers is None
                or (
                    type(advertised_ready_workers) is int
                    and advertised_ready_workers >= 1
                    and advertised_ready_workers == ready_worker_count
                )
            )
            and type(scheduler_workers) is int
            and scheduler_workers == len(workers)
            and not worker_errors
        )
        if not ready:
            raise ProviderProtocolError(
                "Echo endpoint is healthy at HTTP level but has no ready inference workers",
                provenance={
                    "endpoint_sha256": _endpoint_sha256(self.base_url),
                    "status": body.get("status"),
                    "inference_enabled": (
                        inference.get("enabled")
                        if isinstance(inference, Mapping)
                        else None
                    ),
                    "worker_count": len(workers) if isinstance(workers, list) else None,
                    "ready_worker_count": ready_worker_count,
                    "advertised_ready_workers": advertised_ready_workers,
                    "scheduler_gpu_workers": scheduler_workers,
                    "worker_errors": worker_errors,
                    "health_response_sha256": sha256_bytes(_response_bytes(response)),
                    "attempts": attempts,
                },
            )
        contract = body.get("r2v_contract")
        contract_advertised = "r2v_contract" in body and contract is not None
        expected = {
            "schema_version": schema_version,
            "prompt_normalization": "reject",
            "max_prompt_chars_exclusive": (
                None if max_prompt_chars_exclusive is None else int(max_prompt_chars_exclusive)
            ),
            "max_prompt_tokens": int(max_prompt_tokens),
            "max_memory_slots": int(max_memory_slots),
            "tokenizer_profile_sha256": tokenizer_profile_sha256,
        }
        if require_audio_memory:
            expected["enable_audio_memory"] = True
            if not contract_advertised:
                raise ProviderProtocolError(
                    "Echo endpoint must advertise an R2V contract for verified audio memory",
                    provenance={
                        "endpoint_sha256": _endpoint_sha256(self.base_url),
                        "expected": expected,
                        "advertised": copy.deepcopy(contract),
                        "attempts": attempts,
                    },
                )
        if contract_advertised and (
            not isinstance(contract, Mapping)
            or any(contract.get(key) != value for key, value in expected.items())
        ):
            raise ProviderProtocolError(
                "Echo endpoint advertises a conflicting R2V contract",
                provenance={
                    "endpoint_sha256": _endpoint_sha256(self.base_url),
                    "expected": expected,
                    "advertised": copy.deepcopy(contract),
                    "attempts": attempts,
                },
            )
        return {
            "endpoint_sha256": _endpoint_sha256(self.base_url),
            "status": "ok",
            "readiness": {
                "ready": True,
                "inference_enabled": True,
                "worker_count": len(workers),
                "ready_worker_count": ready_worker_count,
                "advertised_ready_workers": advertised_ready_workers,
                "scheduler_gpu_workers": scheduler_workers,
            },
            "contract": (
                copy.deepcopy(dict(contract))
                if isinstance(contract, Mapping)
                else None
            ),
            "contract_advertised": contract_advertised,
            "client_enforced_contract": expected,
            "health_response_sha256": sha256_bytes(_response_bytes(response)),
            "attempts": attempts,
        }

    def build_payload(
        self,
        request: VideoGenerationRequest,
        *,
        provider_job_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        for label, value in (
            ("request_id", request.request_id),
            ("work_id", request.work_id),
            ("shot_id", request.shot_id),
        ):
            if not str(value).strip():
                raise ValueError(f"{label} is required")
        if not isinstance(request.prompt, str) or not request.prompt:
            raise ValueError("prompt must be a non-empty string")
        if request.prompt != request.prompt.strip():
            raise ValueError("prompt has leading/trailing whitespace; provider will not normalize it")
        if min(int(request.width), int(request.height), int(request.num_frames)) <= 0:
            raise ValueError("width, height, and num_frames must be positive")
        if len(request.memory_slots) > 7:
            raise ValueError("Echo 1.5 accepts at most seven memory slots")
        payload: dict[str, Any] = {
            "work_id": str(request.work_id),
            "shot_id": str(request.shot_id),
            "job_id": str(provider_job_id or request.request_id),
            "prompt": request.prompt,
            "memory_slots": [],
            "seed": int(request.seed),
            "width": int(request.width),
            "height": int(request.height),
            "num_frames": int(request.num_frames),
        }
        media_provenance: dict[str, Any] = {"condition": None, "memory_slots": []}
        if request.condition_path is not None:
            value, manifest = self._media_value(
                request.condition_path, expected="image", label="condition_path"
            )
            payload["condition_img"] = value
            media_provenance["condition"] = manifest
        for index, raw_slot in enumerate(request.memory_slots):
            if not isinstance(raw_slot, Mapping):
                raise TypeError(f"memory_slots[{index}] must be an object")
            extra = set(raw_slot) - {"image_path", "audio_path", "metadata"}
            if extra:
                raise ValueError(f"memory_slots[{index}] has unknown fields: {sorted(extra)}")
            image_path = raw_slot.get("image_path")
            if not image_path:
                raise ValueError(f"memory_slots[{index}].image_path is required")
            image_value, image_manifest = self._media_value(
                str(image_path), expected="image", label=f"memory_slots[{index}].image_path"
            )
            metadata = raw_slot.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise TypeError(f"memory_slots[{index}].metadata must be an object")
            # Force a canonical JSON serialization now rather than failing after
            # an expensive generation has already been submitted.
            canonical_json(metadata)
            slot: dict[str, Any] = {
                "image_url": image_value,
                "metadata": copy.deepcopy(dict(metadata)),
            }
            audio_manifest = None
            audio_path = raw_slot.get("audio_path")
            if audio_path:
                audio_value, audio_manifest = self._media_value(
                    str(audio_path), expected="audio", label=f"memory_slots[{index}].audio_path"
                )
                slot["audio_url"] = audio_value
            else:
                slot["audio_mode"] = "empty"
            payload["memory_slots"].append(slot)
            media_provenance["memory_slots"].append(
                {
                    "index": index,
                    "image": image_manifest,
                    "audio": audio_manifest,
                    "metadata": copy.deepcopy(dict(metadata)),
                }
            )
        return payload, media_provenance

    def _status_url(self, accepted: Mapping[str, Any]) -> str:
        raw = str(accepted.get("status_url") or "")
        version_id = str(
            accepted.get("version_id") or accepted.get("task_id") or accepted.get("remote_task_id") or ""
        )
        if not raw:
            if not version_id:
                raise ProviderProtocolError("Echo /r2v response has no status_url or version_id")
            return f"{self.base_url}/version/{quote(version_id, safe='')}"
        resolved = urljoin(f"{self.base_url}/", raw)
        parsed = urlparse(resolved)
        base = urlparse(self.base_url)
        if parsed.hostname in {"127.0.0.1", "localhost", "0.0.0.0"} and base.hostname not in {
            "127.0.0.1",
            "localhost",
            "0.0.0.0",
        }:
            parsed = parsed._replace(scheme=base.scheme, netloc=base.netloc)
            resolved = parsed.geturl()
        elif (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
            raise ProviderProtocolError("Echo status_url points at a different origin")
        return resolved

    def _poll(self, accepted: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        last = accepted
        attempts: list[dict[str, Any]] = []
        deadline = time.monotonic() + self.generation_timeout
        status_url: str | None = None
        while True:
            status = str(last.get("status") or "").lower()
            if status in _SUCCESS_STATES | _FAILURE_STATES:
                return last, attempts
            if status_url is None:
                status_url = self._status_url(accepted)
            if time.monotonic() >= deadline:
                raise ProviderHTTPError(
                    "Echo /r2v generation timed out",
                    provenance={"poll_attempts": attempts, "last_response": last},
                )
            if self.poll_seconds:
                self.sleep(self.poll_seconds)
            response, current_attempts = self._request("GET", status_url, headers=self.extra_headers)
            attempts.extend(current_attempts)
            last = _response_json(response)

    def _download_video(self, url: str) -> tuple[bytes, list[dict[str, Any]], str]:
        if url.startswith("data:"):
            return _decode_data_url(url), [], "data_url"
        resolved = urljoin(f"{self.base_url}/", url)
        parsed = urlparse(resolved)
        base = urlparse(self.base_url)
        if parsed.hostname in {"127.0.0.1", "localhost", "0.0.0.0"} and base.hostname not in {
            "127.0.0.1",
            "localhost",
            "0.0.0.0",
        }:
            resolved = parsed._replace(scheme=base.scheme, netloc=base.netloc).geturl()
        response, attempts = self._request(
            "GET", resolved, headers=self.extra_headers, allow_redirects=True
        )
        payload = _response_bytes(response)
        if not payload:
            raise ProviderProtocolError("Echo video download returned empty bytes")
        return payload, attempts, _request_header(response, "content-type")

    @staticmethod
    def _copy_video(source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination.resolve()

    def generate(self, request: VideoGenerationRequest) -> ProviderResult:
        endpoint = f"{self.base_url}/r2v"
        infrastructure_attempts: list[dict[str, Any]] = []
        payload: dict[str, Any] | None = None
        media_provenance: dict[str, Any] | None = None
        response: Any = None
        accepted: dict[str, Any] | None = None
        final: dict[str, Any] | None = None
        submit_attempts: list[dict[str, Any]] = []
        poll_attempts: list[dict[str, Any]] = []
        provider_job_id = ""
        for infrastructure_attempt in range(1, self.max_terminal_attempts + 1):
            # A terminal failed Echo job is immutable under its idempotency key.
            # Replaying that key after a process restart only retrieves the same
            # failure forever.  Derive a bounded sequence of physical job IDs
            # from the one durable logical request ID.  Re-running this loop is
            # deterministic: completed/failed attempts are rediscovered before
            # advancing, while ambiguous transport timeouts keep the same ID.
            provider_job_id = (
                str(request.request_id)
                if infrastructure_attempt == 1
                else f"{request.request_id}-infra{infrastructure_attempt}"
            )
            payload, media_provenance = self.build_payload(
                request, provider_job_id=provider_job_id
            )
            headers = {
                "Content-Type": "application/json",
                "Idempotency-Key": provider_job_id,
                **self.extra_headers,
            }
            response, submit_attempts = self._request(
                "POST",
                endpoint,
                headers=headers,
                json_payload=payload,
            )
            accepted = _response_json(response)
            final, poll_attempts = self._poll(accepted)
            status = str(final.get("status") or "").lower()
            infrastructure_attempts.append(
                {
                    "attempt": infrastructure_attempt,
                    "provider_job_id": provider_job_id,
                    "status": status,
                    "submit_attempts": copy.deepcopy(submit_attempts),
                    "poll_attempts": copy.deepcopy(poll_attempts),
                    "accepted": copy.deepcopy(accepted),
                    "final": copy.deepcopy(final),
                }
            )
            if status in _SUCCESS_STATES:
                break
        if payload is None or media_provenance is None or accepted is None or final is None:
            raise AssertionError("unreachable Echo infrastructure-attempt loop")
        status = str(final.get("status") or "").lower()
        if status not in _SUCCESS_STATES:
            raise ProviderProtocolError(
                "Echo /r2v generation failed after "
                f"{len(infrastructure_attempts)} deterministic infrastructure attempt(s): "
                f"{final.get('error') or status or 'unknown status'}",
                provenance={
                    "logical_request_id": str(request.request_id),
                    "infrastructure_attempts": infrastructure_attempts,
                },
            )
        result = final.get("result") if isinstance(final.get("result"), Mapping) else {}
        uses_memory_audio = any(
            isinstance(slot, Mapping) and bool(slot.get("audio_path"))
            for slot in request.memory_slots
        )
        if uses_memory_audio:
            metadata = result.get("metadata")
            conditioning = (
                metadata.get("conditioning")
                if isinstance(metadata, Mapping)
                else None
            )
            if not isinstance(conditioning, Mapping) or conditioning.get(
                "has_memory_audio"
            ) is not True:
                raise ProviderProtocolError(
                    "Echo succeeded without proving that memory audio reached conditioning",
                    provenance={"accepted": accepted, "final": final},
                )
        local_source = result.get("output_path") or final.get("output_path")
        video_url = (
            result.get("video_url")
            or final.get("video_url")
            or result.get("artifact_url")
            or final.get("artifact_url")
        )
        output_dir = Path(request.output_dir).expanduser().resolve()
        output_path = output_dir / f"{safe_name(request.request_id)}.mp4"
        download_attempts: list[dict[str, Any]] = []
        output_transport = ""
        if local_source and Path(str(local_source)).expanduser().is_file():
            source = Path(str(local_source)).expanduser().resolve()
            output_path = self._copy_video(source, output_path)
            output_transport = "shared_local_path"
        elif isinstance(video_url, str) and video_url:
            raw, download_attempts, _ = self._download_video(video_url)
            output_path = atomic_write_bytes(output_path, raw)
            output_transport = "download_url"
        else:
            raise ProviderProtocolError(
                "successful Echo response has no accessible output_path or video_url",
                provenance={"accepted": accepted, "final": final},
            )
        output_mime = detect_mime(output_path)
        if not output_mime.startswith("video/"):
            output_path.unlink(missing_ok=True)
            raise ProviderProtocolError(f"Echo output is not recognized as video: {output_mime}")
        response_id = _response_id(final) or _response_id(accepted)
        provenance = {
            "provider_kind": "video",
            "adapter": "echo_1_5_r2v",
            "endpoint_sha256": _endpoint_sha256(endpoint),
            "request_id": str(request.request_id),
            "provider_job_id": provider_job_id,
            "request_payload_sha256": sha256_json(payload),
            "prompt_sha256": sha256_text(request.prompt),
            "prompt_transport": "verbatim",
            "media_transport": self.media_transport,
            "media_inputs": media_provenance,
            "submit_response_sha256": sha256_bytes(_response_bytes(response)),
            "accepted_response": copy.deepcopy(accepted),
            "final_response": copy.deepcopy(final),
            "response_id": response_id,
            "output_transport": output_transport,
            "output_sha256": sha256_file(output_path),
            "output_mime_type": output_mime,
            "submit_attempts": submit_attempts,
            "poll_attempts": poll_attempts,
            "infrastructure_attempts": infrastructure_attempts,
            "download_attempts": download_attempts,
            "response_request_id": _request_header(
                response, "x-request-id", "request-id", "x-jdcloud-request-id"
            ),
        }
        return ProviderResult(
            output_path=str(output_path),
            request_payload=_audit_safe_payload(payload),
            request_sha256=sha256_json(payload),
            request_id=str(request.request_id),
            response_id=response_id,
            provenance=provenance,
        )


class ThreadSafeLimiter:
    """Share one in-process concurrency budget across provider instances."""

    def __init__(
        self,
        delegate: Any,
        semaphore: threading.Semaphore | None = None,
        *,
        limit: int | None = None,
    ) -> None:
        if semaphore is not None and limit is not None:
            raise ValueError("provide semaphore or limit, not both")
        if semaphore is None:
            if limit is None or int(limit) < 1:
                raise ValueError("limit must be positive when semaphore is omitted")
            semaphore = threading.BoundedSemaphore(int(limit))
        self.delegate = delegate
        self.semaphore = semaphore

    def _run(self, method: str, *args: Any, **kwargs: Any) -> Any:
        with self.semaphore:
            return getattr(self.delegate, method)(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        return self._run("call", *args, **kwargs)

    def call_json(self, *args: Any, **kwargs: Any) -> Any:
        return self._run("call_json", *args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self._run("generate", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class RoundRobinEchoProvider:
    """Capacity-aware, request-sticky distribution across Echo endpoints.

    New logical requests are assigned to the least-loaded endpoint with a
    rotating tie break, then that route is persisted before the provider is
    called. Replays therefore keep endpoint-local idempotency, while unrelated
    requests do not collide on a hash-selected node and leave other GPUs idle.
    A separate semaphore per endpoint enforces the capacity proven by health.
    """

    def __init__(
        self,
        base_urls: str | Sequence[str],
        *,
        semaphore: threading.Semaphore | None = None,
        limit: int | None = None,
        per_endpoint_limit: int | None = None,
        provider_factory: Callable[..., Any] = EchoVideoProvider,
        **provider_kwargs: Any,
    ) -> None:
        values = (
            [item.strip() for item in base_urls.replace(",", " ").split()]
            if isinstance(base_urls, str)
            else [str(item).strip() for item in base_urls]
        )
        endpoints: list[str] = []
        endpoint_hashes: set[str] = set()
        for value in values:
            endpoint = value.rstrip("/")
            if not endpoint:
                continue
            fingerprint = endpoint_sha256(endpoint)
            if fingerprint not in endpoint_hashes:
                endpoints.append(endpoint)
                endpoint_hashes.add(fingerprint)
        if not endpoints:
            raise ValueError("at least one Echo base URL is required")
        if semaphore is not None and limit is not None:
            raise ValueError("provide semaphore or limit, not both")
        if semaphore is None and limit is not None:
            if int(limit) < 1:
                raise ValueError("limit must be positive")
            semaphore = threading.BoundedSemaphore(int(limit))
        if per_endpoint_limit is not None and int(per_endpoint_limit) < 1:
            raise ValueError("per_endpoint_limit must be positive")
        self.base_urls = tuple(endpoints)
        self.providers = tuple(provider_factory(url, **provider_kwargs) for url in endpoints)
        self.semaphore = semaphore
        self._fixed_endpoint_limit = (
            int(per_endpoint_limit) if per_endpoint_limit is not None else None
        )
        initial_capacity = self._fixed_endpoint_limit or 1
        self._endpoint_capacities = [initial_capacity for _ in endpoints]
        self._endpoint_semaphores = [
            threading.BoundedSemaphore(initial_capacity) for _ in endpoints
        ]
        self._endpoint_pending = [0 for _ in endpoints]
        self._tie_cursor = 0
        self._routing_lock = threading.Lock()
        # ``None`` means endpoint capability preflight has not run.  Audio
        # requests fail closed until this set has been populated from signed
        # health responses; visual requests remain usable on legacy workers.
        self._audio_capable_indices: frozenset[int] | None = None

    @staticmethod
    def _read_route(route_path: Path, request_id: str) -> dict[str, Any] | None:
        if not route_path.is_file():
            return None
        try:
            route = json.loads(route_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderProtocolError(
                f"persisted Echo route is unreadable: {route_path}"
            ) from exc
        if not isinstance(route, Mapping) or route.get("request_id") != request_id:
            raise ProviderProtocolError("persisted Echo route does not match request_id")
        if route.get("schema_version") not in {
            "storymem.echo_route.v1",
            "storymem.echo_route.v2",
            "storymem.echo_route.v3",
        }:
            raise ProviderProtocolError("persisted Echo route has an unknown schema")
        return dict(route)

    def _route_index(self, route: Mapping[str, Any]) -> int:
        endpoint_hash = str(route.get("endpoint_sha256") or "")
        matching = [
            index
            for index, url in enumerate(self.base_urls)
            if _endpoint_sha256(url) == endpoint_hash
        ]
        if len(matching) != 1:
            raise ProviderProtocolError(
                "persisted Echo route endpoint is absent or ambiguous; refusing to drift"
            )
        return matching[0]

    def _least_loaded_index(
        self, eligible_indices: Sequence[int] | None = None
    ) -> int:
        eligible = tuple(
            range(len(self.base_urls))
            if eligible_indices is None
            else sorted({int(index) for index in eligible_indices})
        )
        if not eligible:
            raise ProviderProtocolError(
                "no verified audio-capable Echo endpoint is available"
            )
        count = len(self.base_urls)
        ordered = [
            index
            for offset in range(count)
            for index in [(self._tie_cursor + offset) % count]
            if index in eligible
        ]
        _, index = min(
            enumerate(ordered),
            key=lambda pair: (
                self._endpoint_pending[pair[1]]
                / self._endpoint_capacities[pair[1]],
                pair[0],
            ),
        )
        self._tie_cursor = (index + 1) % count
        return index

    def _choose(
        self,
        request_id: str,
        output_dir: str | Path,
        *,
        requires_audio_memory: bool,
    ) -> tuple[int, Any, Path, dict[str, Any]]:
        if not str(request_id).strip():
            raise ValueError("request_id is required for stable Echo routing")
        route_path = (
            Path(output_dir).expanduser().resolve()
            / f".echo_route_{sha256_text(request_id).removeprefix('sha256:')[:20]}.json"
        )
        route_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = route_path.with_suffix(route_path.suffix + ".lock")
        # The process lock protects load accounting across different request
        # IDs; flock additionally prevents two collector processes from
        # assigning the same durable request differently.
        with self._routing_lock, lock_path.open("a+b") as route_lock:
            fcntl.flock(route_lock.fileno(), fcntl.LOCK_EX)
            try:
                route = self._read_route(route_path, request_id)
                if route is None:
                    if requires_audio_memory:
                        if self._audio_capable_indices is None:
                            raise ProviderProtocolError(
                                "audio Echo routing requires endpoint capability preflight"
                            )
                        eligible_indices: Sequence[int] = tuple(
                            self._audio_capable_indices
                        )
                    else:
                        eligible_indices = tuple(range(len(self.base_urls)))
                    index = self._least_loaded_index(eligible_indices)
                    route = {
                        "schema_version": "storymem.echo_route.v3",
                        "request_id": request_id,
                        "routing": "persistent_least_loaded_round_robin_v1",
                        "requires_audio_memory": requires_audio_memory,
                        "endpoint_sha256": _endpoint_sha256(self.base_urls[index]),
                        "endpoint_count_at_assignment": len(self.base_urls),
                        "endpoint_capacity_at_assignment": self._endpoint_capacities[
                            index
                        ],
                    }
                    atomic_write_json(route_path, route)
                else:
                    index = self._route_index(route)
                    route_requires_audio = bool(
                        route.get("requires_audio_memory", False)
                    )
                    if route_requires_audio != requires_audio_memory:
                        raise ProviderProtocolError(
                            "persisted Echo route capability requirement differs from the request"
                        )
                    if requires_audio_memory and (
                        self._audio_capable_indices is None
                        or index not in self._audio_capable_indices
                    ):
                        raise ProviderProtocolError(
                            "persisted audio Echo route targets an unverified endpoint; refusing to drift"
                        )
                self._endpoint_pending[index] += 1
            finally:
                fcntl.flock(route_lock.fileno(), fcntl.LOCK_UN)
        return index, self.providers[index], route_path, route

    def _release_route(self, index: int) -> None:
        with self._routing_lock:
            if self._endpoint_pending[index] < 1:
                raise RuntimeError("Echo endpoint pending counter underflow")
            self._endpoint_pending[index] -= 1

    def generate(self, request: VideoGenerationRequest) -> ProviderResult:
        requires_audio_memory = any(
            isinstance(slot, Mapping) and bool(slot.get("audio_path"))
            for slot in request.memory_slots
        )
        index, provider, route_path, route = self._choose(
            str(request.request_id),
            request.output_dir,
            requires_audio_memory=requires_audio_memory,
        )

        def invoke() -> ProviderResult:
            result = provider.generate(request)
            if not isinstance(result, ProviderResult):
                return result
            provenance = copy.deepcopy(result.provenance)
            provenance["endpoint_pool"] = {
                "routing": route["routing"],
                "endpoint_index": index,
                "endpoint_sha256": _endpoint_sha256(self.base_urls[index]),
                "endpoint_count": len(self.base_urls),
                "endpoint_capacity": self._endpoint_capacities[index],
                "requires_audio_memory": requires_audio_memory,
                "audio_capability_verified": (
                    self._audio_capable_indices is not None
                    and index in self._audio_capable_indices
                ),
                "route_manifest_sha256": sha256_file(route_path),
            }
            return replace(result, provenance=provenance)

        try:
            with self._endpoint_semaphores[index]:
                if self.semaphore is None:
                    return invoke()
                with self.semaphore:
                    return invoke()
        finally:
            self._release_route(index)

    def validate_endpoints(
        self,
        *,
        tokenizer_profile_sha256: str,
        schema_version: str = "echo15.r2v.v1",
        max_prompt_chars_exclusive: int | None = None,
        max_prompt_tokens: int = 1024,
        max_memory_slots: int = 7,
        require_audio_memory: bool = False,
    ) -> list[dict[str, Any]]:
        """Validate readiness and reject any explicitly conflicting contract.

        Older Echo workers do not advertise ``r2v_contract``.  They are still
        usable because this client performs the same prompt, slot, and media
        checks before submitting a request.  If one or more workers do
        advertise a contract, every advertised value must agree.
        """

        results = []
        for provider in self.providers:
            results.append(
                provider.validate_endpoint(
                    tokenizer_profile_sha256=tokenizer_profile_sha256,
                    schema_version=schema_version,
                    max_prompt_chars_exclusive=max_prompt_chars_exclusive,
                    max_prompt_tokens=max_prompt_tokens,
                    max_memory_slots=max_memory_slots,
                    # A mixed pool is valid: legacy/visual-only endpoints keep
                    # serving visual requests, while audio requests are routed
                    # only to the subset that explicitly advertises support.
                    require_audio_memory=False,
                )
            )
        contracts = {
            canonical_json(
                {
                    key: value
                    for key, value in item["contract"].items()
                    if key != "enable_audio_memory"
                }
            )
            for item in results
            if isinstance(item.get("contract"), Mapping)
        }
        if len(contracts) > 1:
            raise ProviderProtocolError(
                "Echo round-robin nodes advertise different R2V contracts",
                provenance={"endpoints": results},
            )
        audio_capable_indices = frozenset(
            index
            for index, item in enumerate(results)
            if isinstance(item.get("contract"), Mapping)
            and item["contract"].get("enable_audio_memory") is True
        )
        if require_audio_memory and not audio_capable_indices:
            raise ProviderProtocolError(
                "no Echo endpoint advertises verified audio-memory support",
                provenance={"endpoints": results},
            )
        if self._fixed_endpoint_limit is None:
            capacities = []
            for item in results:
                readiness = item.get("readiness")
                ready_workers = (
                    readiness.get("ready_worker_count")
                    if isinstance(readiness, Mapping)
                    else None
                )
                capacities.append(
                    int(ready_workers)
                    if type(ready_workers) is int and ready_workers > 0
                    else 1
                )
            with self._routing_lock:
                if any(self._endpoint_pending):
                    raise RuntimeError(
                        "cannot change Echo endpoint capacities while requests are active"
                    )
                self._endpoint_capacities = capacities
                self._endpoint_semaphores = [
                    threading.BoundedSemaphore(capacity)
                    for capacity in capacities
                ]
                self._audio_capable_indices = audio_capable_indices
        else:
            with self._routing_lock:
                if any(self._endpoint_pending):
                    raise RuntimeError(
                        "cannot change Echo endpoint capabilities while requests are active"
                    )
                self._audio_capable_indices = audio_capable_indices
        return results


__all__ = [
    "EchoVideoProvider",
    "GenericImageProvider",
    "IMAGE_GENERATION_REQUEST_SCHEMA",
    "ImageGenerationRequest",
    "JsonCallResult",
    "ProviderError",
    "ProviderHTTPError",
    "ProviderProtocolError",
    "ProviderResult",
    "RemoteTeacherClient",
    "RoundRobinEchoProvider",
    "SharedRequestRateLimiter",
    "ThreadSafeLimiter",
    "ToolCall",
    "VIDEO_GENERATION_REQUEST_SCHEMA",
    "VideoGenerationRequest",
    "endpoint_sha256",
    "normalize_endpoint_url",
]
