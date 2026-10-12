"""Bounded, audited routing between calibrated names for the same Gemini family."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterator


ALIASES_ENV = "STORYMEM_RL_GEMINI_MODEL_ALIASES"
APPROVAL_PATH_ENV = "STORYMEM_RL_GEMINI_ALIAS_APPROVAL_PATH"
APPROVAL_HASH_ENV = "STORYMEM_RL_GEMINI_ALIAS_APPROVAL_SHA256"


def exceeds_parity_limit(value: float, limit: float) -> bool:
    """Ignore binary rounding at an exact threshold, not scoring differences."""
    return value > limit and not math.isclose(value, limit, rel_tol=0.0, abs_tol=1e-12)


def load_model_aliases(primary_model: str) -> tuple[str, ...]:
    from . import gemini_reward as legacy

    raw = os.getenv(ALIASES_ENV, "").strip()
    if not raw:
        return ()
    try:
        values = json.loads(raw)
    except ValueError as error:
        raise legacy.RewardInfrastructureError("Gemini aliases must be a JSON array") from error
    if (not isinstance(values, list) or len(values) != 2
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values) or primary_model not in values):
        raise legacy.RewardInfrastructureError("Gemini alias pool requires two distinct names including the primary")
    return tuple(values)


def validate_pool_approval(config: Any = None, *, primary_model: str | None = None,
                           pool_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    from . import gemini_reward as legacy
    from .online_reward_v2 import validate_acceptance

    raw_path = pool_receipt.get("path", "") if pool_receipt is not None else os.getenv(APPROVAL_PATH_ENV, "")
    expected_hash = pool_receipt.get("sha256", "") if pool_receipt is not None else os.getenv(APPROVAL_HASH_ENV, "")
    path = Path(raw_path).expanduser().resolve()
    if not raw_path or not expected_hash or not path.is_file() or legacy._file_sha256(path) != expected_hash:
        raise legacy.RewardInfrastructureError("Gemini alias pool requires an unchanged pinned approval")
    approval = legacy._strict_json_loads(path.read_text(), label="Gemini alias approval")
    if approval.get("reward_version") == "reel_joint_v1":
        from .reel_joint_reward import validate_acceptance
    expected_primary = config.model if config is not None else primary_model
    models = approval.get("models", [])
    if (approval.get("schema_version") != "storymem.gemini-alias-pool-approval.v1"
            or approval.get("passed") is not True or approval.get("primary_model") != expected_primary
            or not isinstance(models, list) or len(models) != 2
            or any(not isinstance(model, str) or not model.strip() for model in models)
            or len(set(models)) != 2 or expected_primary not in models):
        raise legacy.RewardInfrastructureError("Gemini alias approval does not match the configured primary")
    if config is not None and tuple(models) != getattr(config, "model_aliases", ()):
        raise legacy.RewardInfrastructureError("Gemini alias list differs from its approval")
    judge_config = approval.get("judge_config", {})
    fields = ("url", "api_style", "review_repeats", "max_output_tokens")
    if set(judge_config) != set(fields) or (config is not None and any(
        judge_config[name] != getattr(config, name) for name in fields
    )):
        raise legacy.RewardInfrastructureError("Gemini alias judge settings differ from their approval")
    accepted = approval.get("acceptances", {})
    if set(accepted) != set(models):
        raise legacy.RewardInfrastructureError("each Gemini alias needs its own reward acceptance")
    for model in models:
        record = accepted[model]
        validate_acceptance(SimpleNamespace(model=model, **judge_config),
                            acceptance_path=record["path"], acceptance_sha256=record["sha256"])
    comparison_record = approval.get("comparison", {})
    comparison_path = Path(comparison_record.get("path", ""))
    if not comparison_path.is_file() or legacy._file_sha256(comparison_path) != comparison_record.get("sha256"):
        raise legacy.RewardInfrastructureError("Gemini alias comparison is missing or changed")
    comparison = legacy._strict_json_loads(comparison_path.read_text(), label="Gemini alias comparison")
    criteria_path = Path(comparison.get("criteria_path", ""))
    if not criteria_path.is_file() or legacy._file_sha256(criteria_path) != comparison.get("criteria_sha256"):
        raise legacy.RewardInfrastructureError("Gemini alias comparison criteria changed")
    criteria = legacy._strict_json_loads(criteria_path.read_text(), label="Gemini alias criteria")
    differences = [abs(float(row["third_reward"]) - float(row["flash_reward"]))
                   for row in comparison.get("cases", [])]
    if (comparison.get("passed") is not True or len(differences) != 12
            or any(not math.isfinite(value) for value in differences)
            or exceeds_parity_limit(sum(differences) / len(differences), criteria["maximum_mean_absolute_reward_difference"])
            or exceeds_parity_limit(max(differences), criteria["maximum_absolute_reward_difference"])
            or comparison.get("baseline_acceptance_path") != accepted[expected_primary]["path"]
            or comparison.get("candidate_acceptance_path") != accepted[next(model for model in models if model != expected_primary)]["path"]):
        raise legacy.RewardInfrastructureError("Gemini alias reward parity has not passed")
    result = {"path": str(path), "sha256": expected_hash, "models": models,
              "primary_model": expected_primary}
    if pool_receipt is not None and pool_receipt != result:
        raise legacy.RewardInfrastructureError("Gemini alias receipt differs from its approval")
    return result


class AliasSelector:
    def __init__(self, models: tuple[str, ...], concurrency: int, *, initial_backoff: float = 30.0,
                 max_backoff: float = 120.0):
        if not models or len(set(models)) != len(models) or concurrency < 1:
            raise ValueError("a non-empty unique model list and positive concurrency are required")
        self.models = models
        self.concurrency = concurrency
        self.per_model_limit = math.ceil(concurrency / len(models))
        self.active = {model: 0 for model in models}
        self.cursor = 0
        self.condition = threading.Condition()
        self.clock = time.monotonic
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.cooldown_until = {model: 0.0 for model in models}
        self.consecutive_failures = {model: 0 for model in models}
        self.failure_generation = {model: 0 for model in models}
        self.retry_waiters = 0

    def mark_unavailable(self, model: str, *, retry_after: float | None = None) -> float:
        with self.condition:
            self.consecutive_failures[model] += 1
            self.failure_generation[model] += 1
            delay = min(self.max_backoff, self.initial_backoff * 2 ** min(self.consecutive_failures[model] - 1, 20))
            delay = max(delay, retry_after or 0.0)
            self.cooldown_until[model] = max(self.cooldown_until[model], self.clock() + delay)
            self.condition.notify_all()
            return delay

    def mark_available(self, model: str, *, generation: int | None = None) -> None:
        with self.condition:
            # An older in-flight success must not erase a later failure or
            # its Retry-After deadline. Only a newly admitted probe can heal it.
            if generation is not None and generation != self.failure_generation[model]:
                return
            self.consecutive_failures[model] = 0
            self.cooldown_until[model] = 0.0
            self.condition.notify_all()

    @contextmanager
    def reserve(self, *, avoid_model: str | None = None, priority: bool = False,
                with_generation: bool = False) -> Iterator[Any]:
        with self.condition:
            self.retry_waiters += int(priority)
            if priority:
                self.condition.notify_all()
            try:
                while True:
                    ordered = self.models[self.cursor:] + self.models[:self.cursor]
                    now = self.clock()
                    available = [model for model in ordered if self.active[model] < self.per_model_limit
                                 and self.cooldown_until[model] <= now
                                 and (not self.consecutive_failures[model] or self.active[model] == 0)]
                    if (available and sum(self.active.values()) < self.concurrency
                            and (priority or not self.retry_waiters)):
                        model = min(available, key=lambda m: (m == avoid_model, self.active[m]))
                        self.active[model] += 1
                        generation = self.failure_generation[model]
                        self.cursor = (self.models.index(model) + 1) % len(self.models)
                        break
                    remaining = [deadline - now for deadline in self.cooldown_until.values() if deadline > now]
                    self.condition.wait(timeout=min(remaining) if remaining else None)
            finally:
                self.retry_waiters -= int(priority)
                self.condition.notify_all()
        try:
            yield (model, generation) if with_generation else model
        finally:
            with self.condition:
                self.active[model] -= 1
                self.condition.notify_all()


_SELECTORS: dict[tuple[Any, ...], AliasSelector] = {}
_SELECTOR_LOCK = threading.Lock()


def get_selector(config: Any, approval: dict[str, Any] | None = None) -> AliasSelector:
    from .gemini_transport_recovery import recovery_settings

    settings = recovery_settings()
    models = tuple(approval["models"]) if approval is not None else (config.model,)
    approval_hash = approval["sha256"] if approval is not None else None
    key = (config.url, models, config.concurrency,
           str(config.rate_limit_state_path), approval_hash, settings)
    with _SELECTOR_LOCK:
        if key not in _SELECTORS:
            _SELECTORS[key] = AliasSelector(models, config.concurrency,
                                           initial_backoff=settings[1], max_backoff=settings[2])
        return _SELECTORS[key]


class JsonAliasPool:
    def __init__(self, clients: dict[str, Any], selector: AliasSelector, *, rate_limiter: Any = None):
        if set(clients) != set(selector.models):
            raise ValueError("client models differ from selector models")
        self.clients = clients
        self.selector = selector
        self.rate_limiter = rate_limiter

    @property
    def manages_rate_limit(self) -> bool:
        from .gemini_transport_recovery import recovery_settings
        return self.rate_limiter is not None and recovery_settings()[0]

    def call_json(self, *args: Any, **kwargs: Any) -> Any:
        from .gemini_transport_recovery import call_with_recovery, recovery_settings

        avoid_model = kwargs.pop("_avoid_model", None)
        priority = kwargs.pop("_priority", False)
        if recovery_settings()[0]:
            return call_with_recovery(self.clients, self.selector, "call_json", args, kwargs,
                                      rate_limiter=self.rate_limiter, avoid_model=avoid_model,
                                      priority=priority)
        with self.selector.reserve(avoid_model=avoid_model, priority=priority) as model:
            return self.clients[model].call_json(*args, **kwargs)


class TextAliasPool(JsonAliasPool):
    def __init__(self, clients: dict[str, Any], selector: AliasSelector, *, max_attempts: int):
        super().__init__(clients, selector)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts

    def generate_text(self, **kwargs: Any) -> Any:
        from .gemini_transport_recovery import call_with_recovery, recovery_settings

        if recovery_settings()[0]:
            return call_with_recovery(self.clients, self.selector, "generate_text", (), kwargs)
        from . import gemini_reward as legacy
        from storymem_teacher.providers import ProviderHTTPError as TextHTTPError
        from tools.storymem_teacher.providers import ProviderHTTPError as RewardHTTPError

        failures = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                with self.selector.reserve() as model:
                    result = self.clients[model].generate_text(**kwargs)
            except (TextHTTPError, RewardHTTPError) as error:
                status, _, _ = legacy._provider_failure_metadata(error)
                failures.append({"model": model, "http_status": status})
                if attempt == self.max_attempts or not legacy._retryable_provider_failure(status):
                    raise
                time.sleep(min(4.0, float(2 ** (attempt - 1))))
                continue
            provenance = {**result.provenance, "alias_pool_prior_failures": failures,
                          "alias_pool_attempts": attempt}
            return replace(result, provenance=provenance)
        raise RuntimeError("Gemini alias retry budget exhausted")
