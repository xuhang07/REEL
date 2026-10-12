"""Single source of truth for the current Memory Policy RL wire contract.

This recipe intentionally supports one protocol generation only.  Callers must
reject any value that differs from these constants; there are no migration,
alias, or legacy-decoding paths in the RL stack.
"""

from __future__ import annotations

from .policy_profile import PROMPT_AUTHORING


AGENT_NAME = "storymem_memory_policy_online_agent"

DATASET_SCHEMA_VERSION = "storymem.memory-policy.verl-episode.v2"
MANIFEST_SCHEMA_VERSION = "storymem.memory-policy.verl-dataset.v2"
# v3 proves token counts against the recursively canonicalized model-visible
# tool schema order used by SFT.  v2 used an equivalent schema in construction
# order and therefore cannot be safely reused by the online runtime.
PROMPT_VALIDATION_SCHEMA_VERSION = "storymem.memory-policy.prompt-length.v3"
SHOT_SPEC_SCHEMA_VERSION = "storymem.memory-policy.verl-shot-spec.v2"
STATE_SEED_SCHEMA_VERSION = "storymem.memory-policy.rl-state-seed.v2"
STATE_SEED_ENCODING = "canonical-json-utf8-v2"
TRAJECTORY_SCHEMA_VERSION = "storymem.memory-policy.rl-rollout.v2"
REWARD_INPUT_SCHEMA_VERSION = "storymem.memory-policy.reward-input.v2"
REVIEW_SCHEMA_VERSION = "storymem.memory-policy.gemini-review.v2"

# Reject old parquet/state seeds even when their row structure looks compatible.
if PROMPT_AUTHORING:
    DATASET_SCHEMA_VERSION = "reel.joint-policy.verl-episode.v1"
    MANIFEST_SCHEMA_VERSION = "reel.joint-policy.verl-dataset.v1"
    PROMPT_VALIDATION_SCHEMA_VERSION = "reel.joint-policy.prompt-length.v1"
    STATE_SEED_SCHEMA_VERSION = "reel.joint-policy.rl-state-seed.v1"

DATA_SOURCE_BY_MODE = {
    "single": "storymem_memory_policy_qwen35_rl_single",
    "multi": "storymem_memory_policy_qwen35_rl_multi",
}
ALLOWED_DATA_SOURCES = frozenset(DATA_SOURCE_BY_MODE.values())


__all__ = [
    "AGENT_NAME",
    "ALLOWED_DATA_SOURCES",
    "DATASET_SCHEMA_VERSION",
    "DATA_SOURCE_BY_MODE",
    "MANIFEST_SCHEMA_VERSION",
    "PROMPT_VALIDATION_SCHEMA_VERSION",
    "REVIEW_SCHEMA_VERSION",
    "REWARD_INPUT_SCHEMA_VERSION",
    "SHOT_SPEC_SCHEMA_VERSION",
    "STATE_SEED_ENCODING",
    "STATE_SEED_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
]
