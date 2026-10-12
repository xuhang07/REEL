"""Model-visible boundary helpers for the Memory Policy protocol.

Private adapters may retain richer manifests and diagnostics, but only the
small, implementation-neutral projections defined here may cross into model
observations or exported SFT messages.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping


PUBLIC_REPRESENTATION_FIELDS: tuple[str, ...] = (
    "representation_id",
    "modality",
    "mime_type",
    "size_bytes",
    "sha256",
)

# These are semantic effects understood by Memory Policy.  Concrete channel,
# workflow, provider, or model names belong only in private adapter receipts.
NEUTRAL_SEMANTIC_EFFECTS: frozenset[str] = frozenset(
    {
        "exact_opening_state",
        "exact_ending_state",
        "temporal_continuation",
        "composition_anchor",
        "identity_reference",
        "appearance_reference",
        "wardrobe_reference",
        "scene_reference",
        "environment_reference",
        "object_reference",
        "style_reference",
        "interaction_reference",
        "motion_reference",
        "pose_reference",
        "spatial_reference",
        "continuity_reference",
        "voice_reference",
        "audio_reference",
        "visual_reference",
        "soft_audio_reference",
        "soft_visual_reference",
        "generic_reference",
    }
)

NEUTRAL_REASON_CODES_BY_OUTCOME: Mapping[str, frozenset[str]] = {
    "applied_as_requested": frozenset({"condition.applied"}),
    "allowed_fallback": frozenset({"condition.fallback"}),
    "not_applied": frozenset(
        {
            "condition.unsupported",
            "condition.capacity_limited",
            "condition.conflict",
            "condition.invalid_evidence",
            "condition.unavailable",
        }
    ),
}
NEUTRAL_BINDING_REASON_CODES: frozenset[str] = frozenset(
    code
    for codes in NEUTRAL_REASON_CODES_BY_OUTCOME.values()
    for code in codes
)


def public_representation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project a private media manifest onto the public, neutral allowlist."""

    if not isinstance(value, Mapping):
        raise ValueError("representation must be an object")
    projected = {
        key: copy.deepcopy(value[key])
        for key in PUBLIC_REPRESENTATION_FIELDS
        if key in value
    }
    validate_public_representation(projected)
    return projected


def validate_public_representation(value: Any) -> None:
    """Reject non-neutral keys in a model-visible representation object."""

    if not isinstance(value, Mapping):
        raise ValueError("public representation must be an object")
    extra = set(str(key) for key in value) - set(PUBLIC_REPRESENTATION_FIELDS)
    if extra:
        raise ValueError(
            "public representation contains non-neutral fields: "
            + ", ".join(sorted(extra))
        )
    missing = set(PUBLIC_REPRESENTATION_FIELDS) - set(str(key) for key in value)
    if missing:
        raise ValueError(
            "public representation is missing fields: " + ", ".join(sorted(missing))
        )
    if not isinstance(value.get("representation_id"), str) or not value.get(
        "representation_id"
    ):
        raise ValueError("public representation has an invalid representation_id")
    if value.get("modality") not in {"image", "video", "audio", "text"}:
        raise ValueError("public representation has an invalid modality")
    if not isinstance(value.get("mime_type"), str) or not value.get("mime_type"):
        raise ValueError("public representation has an invalid mime_type")
    if type(value.get("size_bytes")) is not int or value["size_bytes"] < 1:
        raise ValueError("public representation has an invalid size_bytes")
    if not isinstance(value.get("sha256"), str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", value["sha256"]
    ) is None:
        raise ValueError("public representation has an invalid sha256")


def validate_neutral_semantic_effect(value: Any) -> str:
    """Return one protocol semantic effect, rejecting adapter-native labels."""

    if not isinstance(value, str) or value not in NEUTRAL_SEMANTIC_EFFECTS:
        raise ValueError("applied effect is not a neutral Memory Policy semantic token")
    return value


def validate_neutral_reason_code(value: Any) -> str:
    """Return one public reason code, rejecting private adapter diagnostics."""

    if not isinstance(value, str) or value not in NEUTRAL_BINDING_REASON_CODES:
        raise ValueError("binding reason_code is not a neutral protocol reason")
    return value


def validate_neutral_binding_outcome(
    *,
    outcome: Any,
    reason_code: Any,
    applied_effect: Any,
    requested_effect: Any,
) -> None:
    """Validate the complete public projection of one adapter binding result."""

    status = str(outcome or "")
    allowed_reasons = NEUTRAL_REASON_CODES_BY_OUTCOME.get(status)
    if allowed_reasons is None:
        raise ValueError("binding outcome is invalid")
    reason = validate_neutral_reason_code(reason_code)
    if reason not in allowed_reasons:
        raise ValueError("binding reason_code is not a neutral protocol reason")
    if status == "not_applied":
        if applied_effect is not None:
            raise ValueError("not_applied must use applied_effect=null")
        return
    effect = validate_neutral_semantic_effect(applied_effect)
    if status == "applied_as_requested" and effect != requested_effect:
        raise ValueError("applied_as_requested must preserve the requested intended effect")


__all__ = [
    "NEUTRAL_BINDING_REASON_CODES",
    "NEUTRAL_REASON_CODES_BY_OUTCOME",
    "NEUTRAL_SEMANTIC_EFFECTS",
    "PUBLIC_REPRESENTATION_FIELDS",
    "public_representation",
    "validate_neutral_binding_outcome",
    "validate_neutral_reason_code",
    "validate_neutral_semantic_effect",
    "validate_public_representation",
]
