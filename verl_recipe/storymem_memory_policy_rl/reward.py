"""VERL reward entrypoint for the Qwen3.5 Memory Policy online rollout."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

try:  # Normal package import from online_agent_loop.py.
    from .contracts import ALLOWED_DATA_SOURCES
    from .gemini_reward import RewardInfrastructureError, evaluate_episode as _evaluate_v1
except ImportError:  # VERL loads custom reward modules directly from a file path.
    echo_longvideo_root = str(Path(__file__).resolve().parents[2])
    if echo_longvideo_root not in sys.path:
        sys.path.insert(0, echo_longvideo_root)
    # Import through the package name.  Adding this recipe directory itself to
    # sys.path would let its local tools.py shadow the repository tools package.
    from verl_recipe.storymem_memory_policy_rl.contracts import (  # type: ignore
        ALLOWED_DATA_SOURCES,
    )
    from verl_recipe.storymem_memory_policy_rl.gemini_reward import (  # type: ignore
        RewardInfrastructureError,
        evaluate_episode as _evaluate_v1,
    )


def evaluate_episode(trajectory_path: str | Path, *, mode: str) -> dict[str, Any]:
    version = os.getenv("STORYMEM_RL_REWARD_VERSION", "v1").strip()
    from verl_recipe.storymem_memory_policy_rl.policy_profile import validate_reward_profile
    validate_reward_profile(version, mode)
    if version == "reel_joint_v1":
        from verl_recipe.storymem_memory_policy_rl.reel_joint_reward import evaluate_episode as evaluate_joint
        return evaluate_joint(trajectory_path, mode=mode)
    if version == "v1":
        return _evaluate_v1(trajectory_path, mode=mode)
    if version == "single_v2":
        from verl_recipe.storymem_memory_policy_rl.online_reward_v2 import evaluate_episode as evaluate_v2

        return evaluate_v2(trajectory_path, mode=mode)
    if version == "single_memory_v3":
        from verl_recipe.storymem_memory_policy_rl.memory_reward_v3 import evaluate_episode as evaluate_memory

        return evaluate_memory(trajectory_path, mode=mode)
    raise RewardInfrastructureError(f"unsupported STORYMEM_RL_REWARD_VERSION: {version!r}")


def _online_reward(extra_info: Any, kwargs: Mapping[str, Any]) -> float:
    locations: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(extra_info, Mapping):
        locations.append(("extra_info", extra_info))
    locations.append(("kwargs", kwargs))
    for key in ("extra_fields", "tool_extra_info"):
        value = kwargs.get(key)
        if isinstance(value, Mapping):
            locations.append((key, value))

    candidates: list[tuple[str, Any]] = [
        (label, value["online_reward"])
        for label, value in locations
        if "online_reward" in value
    ]
    if not candidates:
        raise ValueError(
            "online_reward is missing; this recipe requires the online Gemini "
            "episode evaluator and has no heuristic fallback"
        )
    normalized: list[tuple[str, float]] = []
    for label, raw in candidates:
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(f"{label}.online_reward must be numeric")
        reward = float(raw)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError(f"{label}.online_reward must be finite and in [0, 1]")
        normalized.append((label, reward))
    first = normalized[0][1]
    if any(value != first for _, value in normalized[1:]):
        raise ValueError("conflicting online_reward values were supplied")
    return first


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Return only the reward produced by the online nine-tool evaluator."""

    del solution_str, ground_truth
    if data_source not in ALLOWED_DATA_SOURCES:
        raise ValueError(
            f"unsupported data_source {data_source!r}; expected one of "
            f"{sorted(ALLOWED_DATA_SOURCES)!r}"
        )
    return _online_reward(extra_info, kwargs)


__all__ = ["RewardInfrastructureError", "compute_score", "evaluate_episode"]
