"""Public, side-effect-free StoryMem memory-policy contract.

Importing this package does not initialize a provider, load media, or mutate
runtime state.  Collection, training export, and inference should share these
exact models, tool schemas, and system prompt.
"""

from .contract import (
    CANONICAL_TOOL_NAMES,
    JSON_SCHEMA_DIALECT,
    TOOL_PROTOCOL_VERSION,
    TOOL_SCHEMAS,
    TOOL_SCHEMA_SHA256,
    canonical_tool_names,
    get_tool_schemas,
    tool_schema_hash,
)
from .models import (
    ConditionBudget,
    FALLBACK_POLICIES,
    IMAGE_GENERATION_PURPOSES,
    MODALITIES,
    SELECTION_MODES,
    SOURCE_KINDS,
    TIMELINE_ROLES,
    VISUAL_MODALITIES,
    EvidenceBinding,
    EvidenceRequirement,
    EvidencePlan,
)
from .prompts import OPEN_MODEL_SYSTEM_PROMPT, SYSTEM_PROMPT


__all__ = [
    "CANONICAL_TOOL_NAMES",
    "ConditionBudget",
    "FALLBACK_POLICIES",
    "IMAGE_GENERATION_PURPOSES",
    "JSON_SCHEMA_DIALECT",
    "MODALITIES",
    "OPEN_MODEL_SYSTEM_PROMPT",
    "SELECTION_MODES",
    "SOURCE_KINDS",
    "SYSTEM_PROMPT",
    "TIMELINE_ROLES",
    "TOOL_PROTOCOL_VERSION",
    "TOOL_SCHEMAS",
    "TOOL_SCHEMA_SHA256",
    "VISUAL_MODALITIES",
    "EvidenceBinding",
    "EvidenceRequirement",
    "EvidencePlan",
    "canonical_tool_names",
    "get_tool_schemas",
    "tool_schema_hash",
]
