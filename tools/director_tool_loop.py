#!/usr/bin/env python3
"""Closed-source director/tool-loop prototype for JoyAI-Echo planning.

The director model emits JSON tool calls. This script executes local/API-backed
pseudo tools and feeds results back to the director until it returns a final plan.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from PIL import Image, ImageDraw, ImageFont

from resource_pool import resource_slot

DEFAULT_MODEL_SERVICE_BASE_URL = "https://modelservice.jdcloud.com/v1"
DEFAULT_RECAPTION_MODEL = "doubao-seed-2-1-pro-260628"
DEFAULT_IMAGE_MODEL = "doubao-seedream-4-0-250828"
LOGGER = logging.getLogger(__name__)


def _load_local_model_config() -> dict[str, Any]:
    candidates: list[Path] = []
    if os.getenv("STORYMEM_MODEL_CONFIG"):
        candidates.append(Path(os.environ["STORYMEM_MODEL_CONFIG"]))
    candidates.append(Path(__file__).resolve().parents[1] / "Director_Agent" / ".config.local.json")
    for path in candidates:
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


_LOCAL_MODEL_CONFIG = _load_local_model_config()
_LOCAL_CUSTOM_PROVIDER = (_LOCAL_MODEL_CONFIG.get("providers") or {}).get("custom") if isinstance(_LOCAL_MODEL_CONFIG.get("providers"), dict) else {}
_LOCAL_AGENT_DEFAULTS = _LOCAL_MODEL_CONFIG.get("agents", {}).get("defaults", {}) if isinstance(_LOCAL_MODEL_CONFIG.get("agents"), dict) else {}
_LOCAL_API_KEY = str((_LOCAL_CUSTOM_PROVIDER or {}).get("apiKey") or "")
_LOCAL_API_BASE = str((_LOCAL_CUSTOM_PROVIDER or {}).get("apiBase") or "")
_LOCAL_DEFAULT_MODEL = str((_LOCAL_AGENT_DEFAULTS or {}).get("model") or "")


def load_module_constant(module_path: Path, name: str, default: str = "") -> str:
    if not module_path.is_file():
        return default
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        return default
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(getattr(module, name, default))


RECAPTION_API_KEY = os.getenv(
    "RECAPTION_API_KEY",
    os.getenv("API_KEY", _LOCAL_API_KEY),
)
RECAPTION_OPENAI_BASE_URL = os.getenv("RECAPTION_OPENAI_BASE_URL", _LOCAL_API_BASE or DEFAULT_MODEL_SERVICE_BASE_URL)
RECAPTION_URL = os.getenv(
    "RECAPTION_URL",
    f"{RECAPTION_OPENAI_BASE_URL.rstrip('/')}/chat/completions",
)
RECAPTION_MODEL = os.getenv(
    "RECAPTION_MODEL",
    os.getenv("STORYMEM_TEACHER_MODEL", _LOCAL_DEFAULT_MODEL or DEFAULT_RECAPTION_MODEL),
)
RECAPTION_API_STYLE = os.getenv("RECAPTION_API_STYLE", "").strip().lower()

GEN_BASE_URL = os.getenv(
    "BASE_URL",
    _LOCAL_API_BASE or DEFAULT_MODEL_SERVICE_BASE_URL,
)
GEN_API_KEY = os.getenv(
    "GEN_API_KEY",
    os.getenv("API_KEY", _LOCAL_API_KEY),
)
GEN_URL = os.getenv("GEN_URL", f"{GEN_BASE_URL}/images/gemini_flash/generations")
OPENAI_IMAGE_URL = os.getenv("OPENAI_IMAGE_URL", f"{GEN_BASE_URL.rstrip('/')}/images/generations")
SEEDREAM_IMAGE_URL = os.getenv("SEEDREAM_IMAGE_URL", f"{GEN_BASE_URL.rstrip('/')}/imageEdit/generations")
GEN_MODEL = os.getenv("GEN_MODEL", DEFAULT_IMAGE_MODEL)
GEN_API_STYLE = os.getenv("GEN_API_STYLE", "").strip().lower()
SEEDREAM_MIN_PIXELS = int(os.getenv("SEEDREAM_MIN_PIXELS", str(1024 * 1024)))
GPT_IMAGE_SIZES = ["1024x1024", "1024x1536", "1536x1024"]

RATIO_MAP = {
    "1024x1024": "1:1",
    "1152x896": "4:3",
    "896x1152": "3:4",
    "1216x832": "3:2",
    "832x1216": "2:3",
    "1344x768": "16:9",
    "768x1344": "9:16",
    "1536x640": "21:9",
}


def parse_jsonish(text: str | None) -> Any:
    if not text:
        raise ValueError("empty model response")
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[idx:])
                return parsed
            except json.JSONDecodeError:
                continue
        raise


def image_to_b64(path: Path, size: int = 1024, quality: int = 92) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def choose_ratio(width: int, height: int) -> str:
    src = float(width) / float(height)
    best = min(RATIO_MAP.values(), key=lambda ratio: abs(float(ratio.split(":")[0]) / float(ratio.split(":")[1]) - src))
    return best


def _choose_size_from_candidates(width: int, height: int, candidates: list[str]) -> str:
    src = float(width) / float(max(1, height))
    return min(candidates, key=lambda size: abs((float(size.split("x")[0]) / float(size.split("x")[1])) - src))


def choose_gpt_image_size(width: int, height: int) -> str:
    return _choose_size_from_candidates(width, height, GPT_IMAGE_SIZES)


def image_to_data_uri(path: Path, size: int = 1024, quality: int = 95) -> str:
    return f"data:image/jpeg;base64,{image_to_b64(path, size=size, quality=quality)}"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _understanding_api_style() -> str:
    if RECAPTION_API_STYLE:
        return RECAPTION_API_STYLE
    normalized_model = RECAPTION_MODEL.lower().replace("_", "-").replace(" ", "")
    if "chat/completions" in RECAPTION_URL.lower() or "doubao-seed" in normalized_model or "2.5" in normalized_model or normalized_model.startswith("gpt-"):
        return "openai_chat"
    return "gemini"


def _generation_api_style() -> str:
    if GEN_API_STYLE:
        return GEN_API_STYLE
    normalized_model = GEN_MODEL.lower()
    if "seedream" in normalized_model or "doubao-seedream" in normalized_model:
        return "seedream"
    if "gpt-image" in normalized_model:
        return "openai_image"
    return "gemini_flash"


def _parts_to_openai_content(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        if "text" in part:
            content.append({"type": "text", "text": str(part.get("text") or "")})
            continue
        inline = part.get("inlineData") if isinstance(part.get("inlineData"), dict) else part.get("inline_data")
        if not isinstance(inline, dict):
            continue
        mime_type = str(inline.get("mimeType") or inline.get("mime_type") or "application/octet-stream")
        data = str(inline.get("data") or "")
        if not data:
            continue
        data_url = f"data:{mime_type};base64,{data}"
        if mime_type.startswith("image/"):
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        elif mime_type.startswith("video/"):
            content.append({"type": "video_url", "video_url": {"url": data_url}})
        else:
            content.append({"type": "file_url", "file_url": {"url": data_url}})
    return content


def _extract_openai_chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    if not choices:
        raise RuntimeError(f"no choices in chat response: {json.dumps(data, ensure_ascii=False)[:1000]}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text = "\n".join(str(item.get("text")) for item in content if isinstance(item, dict) and item.get("text"))
        if text.strip():
            return text
    raise RuntimeError(f"empty chat content: {json.dumps(data, ensure_ascii=False)[:1000]}")


def _raise_for_status_with_body(resp: requests.Response) -> None:
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        body = resp.text[:1000] if resp.text else ""
        raise requests.HTTPError(f"{exc}; body={body}", response=resp) from exc


def _http_status_code(exc: Exception) -> int | None:
    """Return an HTTP status without mistaking arbitrary digits for a status."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        try:
            return int(status_code)
        except (TypeError, ValueError):
            pass

    message = str(exc)
    match = re.search(r"\b([45]\d{2})\s+(?:client|server)\s+error\b", message, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'["\'](?:code|status_code)["\']\s*:\s*["\']?([45]\d{2})\b', message, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    lowered = message.lower()
    if "too many requests" in lowered or "rate_limit" in lowered or "rate limit" in lowered:
        return 429
    return None


def _response_header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get(name)
        if value is not None:
            return str(value)
    except (AttributeError, TypeError):
        pass
    try:
        for key, value in headers.items():
            if str(key).lower() == name.lower():
                return str(value)
    except (AttributeError, TypeError):
        pass
    return None


def _redact_provenance_value(value: Any) -> str:
    redacted = str(value)
    for secret in {RECAPTION_API_KEY, os.getenv("API_KEY", "")}:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _provenance_endpoint(url: str) -> str:
    """Return an endpoint label that cannot retain URL credentials or query keys."""
    try:
        parsed = urlsplit(str(url))
        hostname = parsed.hostname or ""
        if not hostname:
            return _redact_provenance_value(str(url).split("?", 1)[0].split("#", 1)[0])
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return _redact_provenance_value(urlunsplit((parsed.scheme, netloc, parsed.path, "", "")))
    except (TypeError, ValueError):
        return "unparseable-endpoint"


def _teacher_provenance_path() -> Path | None:
    value = os.getenv("STORYMEM_TEACHER_API_PROVENANCE_PATH", "").strip()
    return Path(value) if value else None


def _append_teacher_api_provenance(record: dict[str, Any]) -> None:
    """Append one compact JSON object under an advisory lock.

    The record deliberately contains only hashes and selected metadata: never
    request/response bodies, headers, or authorization material.
    """
    path = _teacher_provenance_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short provenance append: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _teacher_response_request_id(response: Any) -> str | None:
    for name in (
        "x-request-id",
        "request-id",
        "x-amzn-requestid",
        "x-ms-request-id",
        "x-trace-id",
        "trace-id",
    ):
        value = _response_header(response, name)
        if value:
            return _redact_provenance_value(value)
    return None


def _teacher_attempt_record(
    *,
    attempt: int,
    started_at: str,
    started_monotonic: float,
    trace_id: str,
    endpoint: str,
    model: str,
    style: str,
    request_payload_sha256: str,
    response: Any,
    success: bool,
    error: Exception | None,
) -> dict[str, Any]:
    response_content = getattr(response, "content", None)
    if isinstance(response_content, str):
        response_content = response_content.encode("utf-8")
    response_sha256 = (
        hashlib.sha256(response_content).hexdigest()
        if isinstance(response_content, (bytes, bytearray))
        else None
    )
    raw_status = getattr(response, "status_code", None)
    try:
        http_status = int(raw_status) if raw_status is not None else None
    except (TypeError, ValueError):
        http_status = None
    safe_endpoint = _provenance_endpoint(endpoint)
    return {
        "schema_version": "storymem_teacher_api_attempt_v1",
        "event": "teacher_api_http_attempt",
        "attempt": int(attempt),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "endpoint": safe_endpoint,
        "endpoint_sha256": hashlib.sha256(safe_endpoint.encode("utf-8")).hexdigest(),
        "model": _redact_provenance_value(model),
        "style": _redact_provenance_value(style),
        "request_payload_sha256": request_payload_sha256,
        "http_status": http_status,
        "response_sha256": response_sha256,
        "response_request_id": _teacher_response_request_id(response),
        "latency_ms": round(max(0.0, time.perf_counter() - started_monotonic) * 1000.0, 3),
        "success": bool(success),
        # Never persist exception messages: providers may echo request URLs or
        # credentials in them. The exception class plus status is auditable.
        "error": None if error is None else type(error).__name__,
    }


def _retry_after_seconds(exc: Exception) -> float | None:
    """Parse standard Retry-After headers (seconds or HTTP date)."""
    response = getattr(exc, "response", None)
    retry_after_ms = _response_header(response, "retry-after-ms")
    if retry_after_ms:
        try:
            return max(0.0, float(retry_after_ms) / 1000.0)
        except ValueError:
            pass

    retry_after = _response_header(response, "retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(retry_at.tzinfo)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass

    match = re.search(
        r"retry(?:[_ -]?after)?[\s\"'=:\-]+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|mins|minutes)?",
        str(exc),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    delay = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    if unit in {"ms", "milliseconds"}:
        delay /= 1000.0
    elif unit in {"m", "min", "mins", "minutes"}:
        delay *= 60.0
    return max(0.0, delay)


def _rate_limit_retry_delay(rate_limit_failure: int, exc: Exception) -> float:
    """Compute a capped exponential delay while never undercutting Retry-After."""
    base = max(0.0, float(os.getenv("RECAPTION_429_BACKOFF_SECONDS", "60")))
    multiplier = max(1.0, float(os.getenv("RECAPTION_429_BACKOFF_MULTIPLIER", "2")))
    max_backoff = max(0.0, float(os.getenv("RECAPTION_429_MAX_BACKOFF_SECONDS", "600")))
    jitter = max(0.0, float(os.getenv("RECAPTION_429_JITTER_SECONDS", "5")))
    try:
        exponential = base * (multiplier ** max(0, rate_limit_failure - 1))
    except OverflowError:
        exponential = float("inf")
    if max_backoff > 0:
        exponential = min(exponential, max_backoff)
    server_delay = _retry_after_seconds(exc)
    delay = max(exponential, server_delay or 0.0)
    return delay + (random.uniform(0.0, jitter) if jitter else 0.0)


def _understanding_rate_limit_paths() -> tuple[Path, Path]:
    configured_dir = os.getenv("STORYMEM_UNDERSTANDING_RATE_LIMIT_DIR", "").strip()
    if configured_dir:
        rate_limit_dir = Path(configured_dir)
    else:
        pool_root = Path(os.getenv("STORYMEM_RESOURCE_POOL_DIR", "/tmp/storymem_resource_pools"))
        rate_limit_dir = pool_root / "understanding_rate_limit"
    rate_limit_dir.mkdir(parents=True, exist_ok=True)
    return rate_limit_dir / "state.json", rate_limit_dir / "state.lock"


def _load_understanding_rate_limit_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_understanding_rate_limit_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_timestamp(state: dict[str, Any], key: str) -> float:
    try:
        value = float(state.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def _wait_for_understanding_request_slot() -> dict[str, Any]:
    """Grant HTTP starts at a smooth host-wide rate using an advisory file lock.

    Unlike pre-allocating future tickets, contenders only claim a slot when it
    is actually due. This prevents delayed processes from consuming old tickets
    together and creating a burst. The same state carries a 429 cooldown.
    """
    max_qps = max(0.0, float(os.getenv("STORYMEM_UNDERSTANDING_MAX_QPS", "0")))
    global_cooldown_enabled = max(
        0.0,
        float(os.getenv("RECAPTION_429_GLOBAL_COOLDOWN_SECONDS", "0")),
    ) > 0
    if max_qps <= 0 and not global_cooldown_enabled:
        return {"enabled": False, "wait_sec": 0.0}

    interval = (1.0 / max_qps) if max_qps > 0 else 0.0
    state_path, lock_path = _understanding_rate_limit_paths()
    started_at = time.time()
    while True:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
        wait_seconds = 0.0
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            state = _load_understanding_rate_limit_state(state_path)
            now = time.time()
            next_allowed_at = _state_timestamp(state, "next_allowed_at")
            cooldown_until = _state_timestamp(state, "cooldown_until")
            allowed_at = max(next_allowed_at, cooldown_until)
            if allowed_at <= now:
                state.update(
                    {
                        "version": 1,
                        "max_qps": max_qps,
                        "last_granted_at": now,
                        "last_granted_pid": os.getpid(),
                        "next_allowed_at": now + interval,
                    }
                )
                _write_understanding_rate_limit_state(state_path, state)
                return {
                    "enabled": True,
                    "max_qps": max_qps,
                    "granted_at": now,
                    "wait_sec": max(0.0, now - started_at),
                    "state_path": str(state_path),
                }
            wait_seconds = max(0.001, allowed_at - now)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        time.sleep(wait_seconds)


def _extend_understanding_global_cooldown(delay_seconds: float) -> float:
    """Atomically extend the cooldown observed by all understanding workers."""
    delay_seconds = max(0.0, float(delay_seconds))
    if delay_seconds <= 0:
        return 0.0
    state_path, lock_path = _understanding_rate_limit_paths()
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        state = _load_understanding_rate_limit_state(state_path)
        now = time.time()
        cooldown_until = max(
            _state_timestamp(state, "cooldown_until"),
            now + delay_seconds,
        )
        state.update(
            {
                "version": 1,
                "cooldown_until": cooldown_until,
                "cooldown_set_at": now,
                "cooldown_set_by_pid": os.getpid(),
                "next_allowed_at": max(
                    _state_timestamp(state, "next_allowed_at"),
                    cooldown_until,
                ),
            }
        )
        _write_understanding_rate_limit_state(state_path, state)
        return max(0.0, cooldown_until - now)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def call_understanding(parts: list[dict[str, Any]], *, temperature: float = 0.0, retries: int = 3) -> str:
    if not RECAPTION_API_KEY:
        raise RuntimeError("missing RECAPTION_API_KEY or API_KEY")
    style = _understanding_api_style()
    base_headers = {"Content-Type": "application/json"}
    if RECAPTION_API_KEY:
        base_headers["Authorization"] = f"Bearer {RECAPTION_API_KEY}"
    max_output_tokens = os.getenv("RECAPTION_MAX_OUTPUT_TOKENS") or os.getenv("RECAPTION_MAX_TOKENS")
    if style in {"openai_chat", "chat", "chat_completions", "openai_chat_json"}:
        url = RECAPTION_URL if "chat/completions" in RECAPTION_URL.lower() else f"{RECAPTION_OPENAI_BASE_URL.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": RECAPTION_MODEL,
            "messages": [{"role": "user", "content": _parts_to_openai_content(parts)}],
            "temperature": temperature,
            "stream": False,
        }
        if max_output_tokens:
            payload["max_tokens"] = int(max_output_tokens)
        if _env_flag("RECAPTION_RESPONSE_FORMAT_JSON", True):
            payload["response_format"] = {"type": "json_object"}
    else:
        url = RECAPTION_URL
        payload = {
            "model": RECAPTION_MODEL,
            "stream": False,
            "contents": {"role": "user", "parts": parts},
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        if max_output_tokens:
            payload["generationConfig"]["maxOutputTokens"] = int(max_output_tokens)
    request_payload_sha256 = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    last_error: Exception | None = None
    max_attempts = max(1, int(retries), int(os.getenv("RECAPTION_MAX_ATTEMPTS", "3")))
    # Rate limits are capacity signals, not bad samples, and do not consume the
    # ordinary retry budget. Production generation can explicitly wait forever;
    # one-off callers retain a finite default.
    retry_429_forever = _env_flag("RECAPTION_429_RETRY_FOREVER", False)
    max_429_attempts = max(1, int(os.getenv("RECAPTION_429_MAX_ATTEMPTS", str(max_attempts))))
    ordinary_failures = 0
    rate_limit_failures = 0
    request_attempt = 0
    while True:
        request_attempt += 1
        trace_id = str(uuid.uuid4())
        headers = dict(base_headers)
        headers["Trace-Id"] = trace_id
        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.perf_counter()
        resp: requests.Response | None = None
        try:
            time.sleep(random.uniform(0.05, 0.2))
            with resource_slot(os.getenv("STORYMEM_UNDERSTANDING_POOL_NAME", "understanding")):
                _wait_for_understanding_request_slot()
                resp = requests.post(url, headers=headers, json=payload, timeout=float(os.getenv("RECAPTION_TIMEOUT", os.getenv("RECAPTION_HTTP_TIMEOUT_SECONDS", "180"))), proxies={"http": None, "https": None})
            _raise_for_status_with_body(resp)
            data = resp.json()
            if style in {"openai_chat", "chat", "chat_completions", "openai_chat_json"}:
                result = _extract_openai_chat_text(data)
            else:
                result = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            last_error = exc
            error_response = resp if resp is not None else getattr(exc, "response", None)
            _append_teacher_api_provenance(
                _teacher_attempt_record(
                    attempt=request_attempt,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    trace_id=trace_id,
                    endpoint=url,
                    model=RECAPTION_MODEL,
                    style=style,
                    request_payload_sha256=request_payload_sha256,
                    response=error_response,
                    success=False,
                    error=exc,
                )
            )
            status_code = _http_status_code(exc)
            if status_code == 429:
                rate_limit_failures += 1
                configured_global_cooldown = max(
                    0.0,
                    float(os.getenv("RECAPTION_429_GLOBAL_COOLDOWN_SECONDS", "0")),
                )
                if configured_global_cooldown > 0:
                    server_delay = _retry_after_seconds(exc) or 0.0
                    shared_delay = _extend_understanding_global_cooldown(
                        max(configured_global_cooldown, server_delay)
                    )
                    LOGGER.warning(
                        "understanding API shared 429 cooldown extended; cooldown=%.3fs configured=%.3fs retry_after=%.3fs",
                        shared_delay,
                        configured_global_cooldown,
                        server_delay,
                    )
                if not retry_429_forever and rate_limit_failures >= max_429_attempts:
                    break
                delay = _rate_limit_retry_delay(rate_limit_failures, exc)
                limit_label = "unlimited" if retry_429_forever else str(max_429_attempts)
                LOGGER.warning(
                    "understanding API rate-limited; request_attempt=%d rate_limit_failure=%d/%s retrying_in=%.3fs error=%s",
                    request_attempt,
                    rate_limit_failures,
                    limit_label,
                    delay,
                    str(exc)[:500],
                )
                time.sleep(delay)
                continue

            # Authentication, permission, malformed-request, and other 4xx
            # failures are permanent for an unchanged payload.
            if status_code is not None and 400 <= status_code < 500:
                break

            ordinary_failures += 1
            if ordinary_failures >= max_attempts:
                break
            if status_code is not None and status_code >= 500:
                base = float(os.getenv("RECAPTION_5XX_BACKOFF_SECONDS", "8.0"))
                delay = base * ordinary_failures + random.uniform(0.5, 2.0)
            else:
                delay = float(os.getenv("RECAPTION_BACKOFF_SECONDS", "2.0")) * ordinary_failures + random.uniform(0.1, 1.0)
            LOGGER.warning(
                "understanding API transient failure; request_attempt=%d ordinary_failure=%d/%d retrying_in=%.3fs status=%s error=%s",
                request_attempt,
                ordinary_failures,
                max_attempts,
                delay,
                status_code,
                str(exc)[:500],
            )
            time.sleep(delay)
            continue
        _append_teacher_api_provenance(
            _teacher_attempt_record(
                attempt=request_attempt,
                started_at=started_at,
                started_monotonic=started_monotonic,
                trace_id=trace_id,
                endpoint=url,
                model=RECAPTION_MODEL,
                style=style,
                request_payload_sha256=request_payload_sha256,
                response=resp,
                success=True,
                error=None,
            )
        )
        return result
    raise RuntimeError(f"understanding API failed: {last_error}")


def choose_seedream_size(width: int, height: int, *, model: str = "") -> str:
    model_lower = model.lower()
    model_specific = ""
    if "4-5" in model_lower or "4.5" in model_lower:
        model_specific = os.getenv("SEEDREAM_SIZE_4_5", "2K")
    elif "4-0" in model_lower or "4.0" in model_lower:
        model_specific = os.getenv("SEEDREAM_SIZE_4_0", "1K")
    override = (os.getenv("SEEDREAM_SIZE") or os.getenv("SEEDREAM_SIZE_PRESET") or model_specific or "").strip()
    if override:
        return override
    width = max(2, int(width))
    height = max(2, int(height))
    if width * height < SEEDREAM_MIN_PIXELS:
        scale = (SEEDREAM_MIN_PIXELS / float(width * height)) ** 0.5
        width = int(width * scale + 0.999)
        height = int(height * scale + 0.999)
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    while width * height < SEEDREAM_MIN_PIXELS:
        width += 2
        height += 2
    return f"{width}x{height}"


def _decode_gemini_image_response(res_json: dict[str, Any]) -> Image.Image:
    candidates = res_json.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"no candidates in response: {json.dumps(res_json, ensure_ascii=False)[:1000]}")
    content = candidates[0].get("content") or {}
    parts_out = content.get("parts") or []
    for part in parts_out:
        inline = part.get("inlineData") or part.get("inline_data")
        if isinstance(inline, dict) and inline.get("data"):
            return Image.open(io.BytesIO(base64.b64decode(inline["data"]))).convert("RGB")
    raise RuntimeError(f"no image in response: {json.dumps(res_json, ensure_ascii=False)[:1000]}")


def _decode_openai_image_response(res_json: dict[str, Any]) -> Image.Image:
    data = res_json.get("data") if isinstance(res_json.get("data"), list) else []
    if not data:
        raise RuntimeError(f"no data in image response: {json.dumps(res_json, ensure_ascii=False)[:1000]}")
    first = data[0] if isinstance(data[0], dict) else {}
    b64_value = first.get("b64_json")
    if isinstance(b64_value, str) and b64_value.strip():
        if b64_value.lstrip().startswith("data:") and "," in b64_value:
            b64_value = b64_value.split(",", 1)[-1]
        return Image.open(io.BytesIO(base64.b64decode(b64_value))).convert("RGB")
    url = first.get("url")
    if isinstance(url, str) and url.strip():
        resp = requests.get(url, timeout=float(os.getenv("GEN_IMAGE_DOWNLOAD_TIMEOUT", "300")), proxies={"http": None, "https": None})
        _raise_for_status_with_body(resp)
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    raise RuntimeError(f"no b64_json/url in image response: {json.dumps(first, ensure_ascii=False)[:1000]}")


def call_image_generation(prompt: str, output_path: Path, *, refs: list[Path] | None = None, width: int = 1280, height: int = 736, retries: int = 2) -> dict[str, Any]:
    if not GEN_API_KEY:
        raise RuntimeError("missing generation API key")
    refs = refs or []
    style = _generation_api_style()
    if style in {"seedream", "seedream_image_edit"}:
        ratio = choose_seedream_size(width, height, model=GEN_MODEL)
        payload: dict[str, Any] = {
            "model": GEN_MODEL,
            "prompt": prompt,
            "size": ratio,
            "response_format": os.getenv("SEEDREAM_RESPONSE_FORMAT", "url"),
            "watermark": False,
        }
        optimize_prompt_mode = os.getenv("SEEDREAM_OPTIMIZE_PROMPT_MODE", "standard").strip()
        if optimize_prompt_mode and optimize_prompt_mode.lower() not in {"0", "false", "off", "none", "disabled"}:
            payload["optimize_prompt_options"] = {"mode": optimize_prompt_mode}
        output_format = os.getenv("SEEDREAM_OUTPUT_FORMAT", "").strip()
        if output_format:
            payload["output_format"] = output_format
        if "pro" not in GEN_MODEL.lower():
            payload["sequential_image_generation"] = os.getenv("SEEDREAM_SEQUENTIAL_IMAGE_GENERATION", "disabled")
        if refs:
            payload["image"] = [image_to_data_uri(ref, size=1536, quality=95) for ref in refs]
        url = SEEDREAM_IMAGE_URL
    elif style in {"openai_image", "openai_image_json"}:
        ratio = choose_gpt_image_size(width, height)
        payload = {
            "model": GEN_MODEL,
            "prompt": prompt,
            "size": ratio,
            "quality": os.getenv("GPT_IMAGE_QUALITY", "low"),
            "n": 1,
        }
        if refs:
            image_data = [image_to_data_uri(ref, size=1024, quality=95) for ref in refs]
            payload["image"] = image_data if len(image_data) > 1 else image_data[0]
            payload["input_fidelity"] = os.getenv("GPT_IMAGE_INPUT_FIDELITY", "high")
        url = OPENAI_IMAGE_URL
    else:
        ratio = choose_ratio(width, height)
        parts: list[dict[str, Any]] = []
        for ref in refs:
            parts.append({"inline_data": {"mimeType": "image/jpeg", "data": image_to_b64(ref, size=1024, quality=95)}})
        parts.append({"text": prompt})
        payload = {
            "model": GEN_MODEL,
            "contents": [{"role": "USER", "parts": parts}],
            "generation_config": {
                "response_modalities": ["TEXT", "IMAGE"],
                "image_config": {"aspect_ratio": ratio},
            },
        }
        url = GEN_URL
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GEN_API_KEY}"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    max_attempts = max(int(retries) + 1, int(os.getenv("GEN_MAX_ATTEMPTS", "4")))
    for attempt in range(max_attempts):
        try:
            with resource_slot("image"):
                resp = requests.post(url, headers=headers, json=payload, timeout=float(os.getenv("GEN_TIMEOUT", "360")), proxies={"http": None, "https": None})
            _raise_for_status_with_body(resp)
            res_json = resp.json()
            img = _decode_openai_image_response(res_json) if style in {"seedream", "seedream_image_edit", "openai_image", "openai_image_json"} else _decode_gemini_image_response(res_json)
            if style in {"gemini_flash", "gemini_flash_image", "openai_image", "openai_image_json"} or _env_flag("SEEDREAM_RESIZE_TO_REQUEST", False):
                if img.size != (width, height):
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
            img.save(output_path)
            return {"image_path": str(output_path), "model": GEN_MODEL, "api_style": style, "ratio": ratio, "refs": [str(p) for p in refs]}
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code == 429 or "429" in str(exc) or "rate_limit" in str(exc).lower():
                    time.sleep(float(os.getenv("GEN_429_BACKOFF_SECONDS", "60")))
                else:
                    time.sleep(float(os.getenv("GEN_BACKOFF_SECONDS", "1.5")) * (attempt + 1))
    raise RuntimeError(f"image generation failed: {last_error}")


def make_placeholder_image(output_path: Path, title: str, width: int = 1280, height: int = 736) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), (34, 38, 48))
    draw = ImageDraw.Draw(img)
    text = title[:240]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font = None
    draw.multiline_text((48, 48), text, fill=(230, 235, 245), font=font, spacing=10)
    img.save(output_path)
    return {"image_path": str(output_path), "placeholder": True}


@dataclass
class ToolContext:
    story: str
    workdir: Path
    memory_manifest: list[dict[str, Any]]
    dry_run: bool = False
    width: int = 1280
    height: int = 736


def tool_analyze_memories(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""你是视频生成 memory 质量评估器。请根据 story goal 和 memory manifest，为每条 memory 打分并分类。

story goal:
{ctx.story}

memory manifest:
{json.dumps(ctx.memory_manifest, ensure_ascii=False, indent=2)}

返回 JSON：{{"ranked_memories":[{{"id":"...","score":0-100,"role":"identity|scene|transition|discard","reason":"..."}}],"recommended_keep_ids":["..."],"notes":"..."}}"""
    text = call_understanding([{"text": prompt}], temperature=0.0)
    return parse_jsonish(text)


def tool_generate_next_prompt(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    shot_index = args.get("shot_index", 1)
    selected_memories = args.get("selected_memories", [])
    prompt = f"""你是 JoyAI-Echo 的导演提示词生成器。请为下一 shot 生成完整英文 prompt。
要求包含角色外观、动作、对白/无对白、风格、镜头、背景、音效/BGM。当前先做 video-only，也请保留可选 sound 描述但不要依赖音频生成。

story goal:
{ctx.story}

shot_index: {shot_index}
selected_memories:
{json.dumps(selected_memories, ensure_ascii=False, indent=2)}

返回 JSON：{{"shot_prompt":"...","intent":"...","memory_usage":"...","negative_notes":"..."}}"""
    text = call_understanding([{"text": prompt}], temperature=0.2)
    return parse_jsonish(text)


def tool_plan_scene_prewarm(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    selected_memories = args.get("selected_memories", [])
    prompt = f"""你是电影美术指导。请为新场景预生成一张 establishing keyframe 的图像生成指令。

story goal:
{ctx.story}
selected_memories:
{json.dumps(selected_memories, ensure_ascii=False, indent=2)}

返回 JSON：{{"image_prompt":"...","purpose":"scene|identity|transition","should_generate":true}}"""
    text = call_understanding([{"text": prompt}], temperature=0.2)
    return parse_jsonish(text)


def tool_generate_or_edit_image(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    prompt = str(args.get("prompt", "")).strip()
    purpose = str(args.get("purpose", "scene"))
    refs = [Path(p) for p in args.get("refs", []) if Path(p).is_file()]
    out = ctx.workdir / "generated_assets" / f"{purpose}_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    if ctx.dry_run:
        return make_placeholder_image(out, prompt, width=ctx.width, height=ctx.height)
    return call_image_generation(prompt, out, refs=refs, width=ctx.width, height=ctx.height)


def tool_register_visual_memory(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    image_path = str(args.get("image_path", ""))
    memory_id = str(args.get("memory_id") or f"mem_{len(ctx.memory_manifest):03d}")
    entry = {
        "id": memory_id,
        "type": "visual",
        "source": image_path,
        "tags": args.get("tags", []),
        "description": args.get("description", ""),
        "quality_score": args.get("quality_score"),
    }
    ctx.memory_manifest.append(entry)
    manifest_path = ctx.workdir / "memory_manifest.json"
    manifest_path.write_text(json.dumps(ctx.memory_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"registered": entry, "manifest_path": str(manifest_path)}


def tool_finalize(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    return {"final": True, **args}


TOOLS = {
    "analyze_memories": tool_analyze_memories,
    "generate_next_prompt": tool_generate_next_prompt,
    "plan_scene_prewarm": tool_plan_scene_prewarm,
    "generate_or_edit_image": tool_generate_or_edit_image,
    "register_visual_memory": tool_register_visual_memory,
    "finalize": tool_finalize,
}


def tool_specs() -> list[dict[str, Any]]:
    return [
        {"name": "analyze_memories", "description": "Score/select current visual memories for next shot", "args": {}},
        {"name": "generate_next_prompt", "description": "Create the next JoyAI-Echo shot prompt", "args": {"shot_index": "int", "selected_memories": "list"}},
        {"name": "plan_scene_prewarm", "description": "Create a high-quality keyframe/image generation prompt for a new scene", "args": {"selected_memories": "list"}},
        {"name": "generate_or_edit_image", "description": "Use generation API to create/edit a keyframe image", "args": {"prompt": "str", "purpose": "str", "refs": "list[path]"}},
        {"name": "register_visual_memory", "description": "Register a generated/reference image as visual memory", "args": {"image_path": "path", "memory_id": "str", "tags": "list", "description": "str", "quality_score": "number"}},
        {"name": "finalize", "description": "Return final plan", "args": {"selected_memory_ids": "list", "shot_prompt": "str", "scene_asset": "path", "notes": "str"}},
    ]


def build_director_prompt(ctx: ToolContext, history: list[dict[str, Any]]) -> str:
    return f"""你是一个视频生成导演智能体。你可以选择工具来完成：memory筛选、下一shot prompt生成、场景预热图生成/编辑、注册visual memory、最终输出计划。

规则：
1. 每次只返回 JSON，不要 markdown。
2. 如果需要工具，返回：{{"tool_calls":[{{"name":"tool_name","args":{{...}}}}],"reason":"..."}}
3. 如果完成，调用 finalize 工具或返回：{{"final":{{...}}}}
4. 优先流程：analyze_memories -> generate_next_prompt -> 如需新场景则 plan_scene_prewarm -> generate_or_edit_image -> register_visual_memory -> finalize。
5. 所有生成/编辑图像必须通过 generate_or_edit_image 工具。

story goal:
{ctx.story}

available tools:
{json.dumps(tool_specs(), ensure_ascii=False, indent=2)}

current memory manifest:
{json.dumps(ctx.memory_manifest, ensure_ascii=False, indent=2)}

tool history:
{json.dumps(history, ensure_ascii=False, indent=2)}

请决定下一步。"""


def call_director(ctx: ToolContext, history: list[dict[str, Any]]) -> dict[str, Any]:
    text = call_understanding([{"text": build_director_prompt(ctx, history)}], temperature=0.0)
    parsed = parse_jsonish(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"director returned non-object: {parsed}")
    return parsed


def run_loop(ctx: ToolContext, max_turns: int) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    for turn in range(max_turns):
        decision = call_director(ctx, history)
        record: dict[str, Any] = {"turn": turn, "director": decision, "tool_results": []}
        if "final" in decision:
            history.append(record)
            return {"status": "ok", "final": decision["final"], "history": history, "memory_manifest": ctx.memory_manifest}
        tool_calls = decision.get("tool_calls", [])
        if not tool_calls:
            history.append(record)
            return {"status": "stopped", "reason": "no tool calls", "history": history, "memory_manifest": ctx.memory_manifest}
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {}) or {}
            if name not in TOOLS:
                result = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = TOOLS[name](ctx, args)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            record["tool_results"].append({"name": name, "args": args, "result": result})
            if name == "finalize" and isinstance(result, dict) and result.get("final"):
                history.append(record)
                return {"status": "ok", "final": result, "history": history, "memory_manifest": ctx.memory_manifest}
        history.append(record)
    return {"status": "max_turns", "history": history, "memory_manifest": ctx.memory_manifest}


def load_memory_manifest(path: Path | None) -> list[dict[str, Any]]:
    if path and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("memories"), list):
            return data["memories"]
    return [
        {"id": "identity_hero", "type": "visual", "source": "generated_previous_shot_center", "description": "Main character close-up with stable face and costume", "quality_score": 82, "tags": ["identity", "hero"]},
        {"id": "scene_corridor", "type": "visual", "source": "previous_establishing_shot", "description": "Torch-lit stone corridor with warm magical atmosphere", "quality_score": 76, "tags": ["scene", "lighting"]},
        {"id": "bad_motion", "type": "visual", "source": "blurred_action_frame", "description": "Motion-blurred frame with distorted hand", "quality_score": 34, "tags": ["candidate", "low_quality"]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype closed-source director/tool loop for JoyAI-Echo")
    parser.add_argument("--story", type=str, default="A young wizard explores a hidden torch-lit corridor and transitions into a secret library while preserving identity and cinematic magical realism.")
    parser.add_argument("--memory-manifest", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=Path("inference_result/director_tool_loop"))
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Do not call image generation API; create placeholder images instead")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=736)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(
        story=args.story,
        workdir=args.workdir,
        memory_manifest=load_memory_manifest(args.memory_manifest),
        dry_run=args.dry_run,
        width=args.width,
        height=args.height,
    )
    result = run_loop(ctx, max_turns=max(1, args.max_turns))
    out_path = args.workdir / "director_run.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result.get("status"), "output": str(out_path), "final": result.get("final")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
