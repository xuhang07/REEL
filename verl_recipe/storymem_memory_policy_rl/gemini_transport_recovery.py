"""Keep transient Gemini outages inside the interrupted transport request."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import fcntl
import json
import math
import os
from pathlib import Path
import time
from typing import Any


ENABLED_ENV = "STORYMEM_RL_GEMINI_RECOVER_TRANSIENT"
INITIAL_BACKOFF_ENV = "STORYMEM_RL_GEMINI_RECOVERY_INITIAL_SECONDS"
MAX_BACKOFF_ENV = "STORYMEM_RL_GEMINI_RECOVERY_MAX_SECONDS"
FAST_TIMEOUT_ENV = "STORYMEM_RL_GEMINI_FAST_TIMEOUT_SECONDS"
REVIEW_RECOVERY_ENV = "STORYMEM_RL_GEMINI_RECOVER_REVIEW_ERRORS"


def recover_review_errors() -> bool:
    value = os.getenv(REVIEW_RECOVERY_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{REVIEW_RECOVERY_ENV} must be 0 or 1")
    return value == "1"


def fast_timeout_seconds() -> float | None:
    raw = os.getenv(FAST_TIMEOUT_ENV, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{FAST_TIMEOUT_ENV} must be finite and positive")
    return value


def recovery_settings() -> tuple[bool, float, float]:
    enabled = os.getenv(ENABLED_ENV, "0")
    if enabled not in {"0", "1"}:
        raise ValueError(f"{ENABLED_ENV} must be 0 or 1")
    initial = float(os.getenv(INITIAL_BACKOFF_ENV, "30"))
    maximum = float(os.getenv(MAX_BACKOFF_ENV, "120"))
    if not math.isfinite(initial) or not math.isfinite(maximum) or not 0 < initial <= maximum:
        raise ValueError("Gemini recovery backoff must be finite with 0 < initial <= maximum")
    return enabled == "1", initial, maximum


def _record_event(event: dict[str, Any]) -> None:
    record = {"schema_version": "storymem.gemini-transport-recovery.v1",
              "created_at_unix": time.time(), "pid": os.getpid(), **event}
    print("[gemini-recovery] " + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    root = os.getenv("STORYMEM_RL_OUTPUT_ROOT", "")
    if not root:
        return
    try:
        path = Path(root) / "provider_recovery.jsonl"
        with path.open("a") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as error:
        print(f"[gemini-recovery] audit_write_failed={type(error).__name__}", flush=True)


def call_with_recovery(clients: dict[str, Any], selector: Any, method: str,
                       args: tuple[Any, ...], kwargs: dict[str, Any], *, rate_limiter: Any = None,
                       avoid_model: str | None = None, priority: bool = False) -> Any:
    from . import gemini_reward as legacy
    from storymem_teacher.providers import ProviderHTTPError as TextHTTPError
    from tools.storymem_teacher.providers import ProviderHTTPError as JsonHTTPError

    started = selector.clock()
    failures = deque(maxlen=16)
    failure_count = 0
    fast_timeout = fast_timeout_seconds() if method == "call_json" else None
    while True:
        queued = selector.clock()
        with selector.reserve(avoid_model=avoid_model, priority=priority or bool(failure_count),
                              with_generation=True) as (model, generation):
            route_wait = selector.clock() - queued
            if rate_limiter is not None:
                rate_limiter.acquire()
            attempt_kwargs = dict(kwargs)
            if fast_timeout is not None:
                # Escalate after the two short sequential attempts so that
                # genuinely slow evidence can still finish. Never race scores.
                maximum = float(getattr(clients[model], "timeout"))
                attempt_kwargs["timeout"] = min(maximum, fast_timeout * (1.5 ** failure_count)) if failure_count < 2 else maximum
            attempt_started = selector.clock()
            try:
                result = getattr(clients[model], method)(*args, **attempt_kwargs)
            except (TextHTTPError, JsonHTTPError) as error:
                status, retry_after, error_provenance = legacy._provider_failure_metadata(error)
                if not legacy._retryable_provider_failure(status):
                    raise
                failure_count += 1
                attempts = error_provenance.get("attempts", [])
                read_timeout = (status == 0 and bool(attempts)
                                and attempts[-1].get("error_type") in {"ReadTimeout", "Timeout"})
                # A slow individual review is not a route-wide outage. The
                # old circuit serialized an otherwise healthy route after
                # any read timeout, defeating increased concurrency.
                delay = 0.0 if read_timeout else selector.mark_unavailable(model, retry_after=retry_after)
                failure = {"model": model, "http_status": status, "backoff_seconds": delay,
                           "elapsed_seconds": round(selector.clock() - started, 3),
                           "route_quarantined": not read_timeout,
                           "route_wait_seconds": round(route_wait, 3),
                           "http_seconds": round(selector.clock() - attempt_started, 3)}
                if "timeout" in attempt_kwargs:
                    failure["timeout_seconds"] = attempt_kwargs["timeout"]
                if status == 429 and rate_limiter is not None:
                    failure["shared_cooldown_seconds"] = rate_limiter.extend_cooldown(
                        max(rate_limiter.cooldown_seconds, retry_after or 0.0, delay))
                failures.append(failure)
                _record_event({"event": "waiting", "operation": method,
                               "request_failure_count": failure_count, **failure})
                avoid_model = model
                continue
            except Exception:
                selector.mark_available(model, generation=generation)
                raise
            selector.mark_available(model, generation=generation)
        if not failure_count:
            return result
        recovery = {"failure_count": failure_count, "recent_failures": list(failures),
                    "elapsed_seconds": round(selector.clock() - started, 3), "recovered": True}
        _record_event({"event": "recovered", "operation": method, "model": model,
                       "request_failure_count": failure_count, "elapsed_seconds": recovery["elapsed_seconds"]})
        return replace(result, provenance={**result.provenance, "transport_recovery": recovery})
