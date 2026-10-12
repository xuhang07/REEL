"""Pin the policy contract once per process, including Ray rollout workers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TOOLS_ROOT = str(Path(__file__).resolve().parents[2] / "tools")
if _TOOLS_ROOT not in sys.path:
    sys.path.append(_TOOLS_ROOT)

POLICY_PROFILE = os.getenv("STORYMEM_RL_POLICY_PROFILE", "continuity_v4")
if POLICY_PROFILE not in {"continuity_v4", "reel_joint_v1"}:
    raise ValueError(f"Unsupported STORYMEM_RL_POLICY_PROFILE: {POLICY_PROFILE!r}")
PROMPT_AUTHORING = POLICY_PROFILE == "reel_joint_v1"
DIRECT_PROMPT = PROMPT_AUTHORING and os.getenv("STORYMEM_RL_PE_MODE") == "policy_direct"

if DIRECT_PROMPT:
    from tools.reel_direct_prompt import (
        TRAINING_SYSTEM_PROMPT, TRAINING_SYSTEM_PROMPT_VERSION, training_system_prompt_hash,
    )
elif PROMPT_AUTHORING:
    from tools.reel_sft_prompt import (
        TRAINING_SYSTEM_PROMPT, TRAINING_SYSTEM_PROMPT_VERSION, training_system_prompt_hash,
    )
else:
    from tools.qwen35_training_prompt import (
        TRAINING_SYSTEM_PROMPT, TRAINING_SYSTEM_PROMPT_VERSION, training_system_prompt_hash,
    )

from tools.storymem_policy.contract import get_tool_schemas as _schemas, tool_schema_hash as _hash


def get_tool_schemas(tool_names=None):
    return _schemas(tool_names, prompt_authoring=PROMPT_AUTHORING)


def tool_schema_hash(tool_names=None):
    return _hash(tool_names, prompt_authoring=PROMPT_AUTHORING)


def pe_mode_for_profile(mode: str) -> str:
    if mode == "policy_direct":
        if not PROMPT_AUTHORING:
            raise ValueError("policy_direct requires REEL prompt authoring")
        return mode
    if mode not in {"auto", "gemini_rewrite", "passthrough"}:
        raise ValueError("unsupported STORYMEM_RL_PE_MODE")
    if PROMPT_AUTHORING:
        if mode == "passthrough":
            raise ValueError("REEL prompt authoring requires Gemini adapter PE; passthrough is unsupported")
        return "gemini_rewrite"
    return mode


def validate_reward_profile(version: str, mode: str) -> None:
    if PROMPT_AUTHORING:
        if version != "reel_joint_v1" or mode not in {"single", "multi"}:
            raise ValueError("REEL joint policy requires reel_joint_v1 reward in single or multi mode")
    elif version == "reel_joint_v1":
        raise ValueError("REEL joint reward requires the reel_joint_v1 policy profile")
