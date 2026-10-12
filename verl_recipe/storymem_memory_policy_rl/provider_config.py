"""Strict secret lookup for the Memory Policy RL recipe.

The v2 runtime accepts only its two canonical environment variables. It does
not inspect local provider files or alternate variable names, which keeps the
configuration identical on every Ray node.
"""

from __future__ import annotations

import os


_SECRET_ENVIRONMENTS = frozenset(
    {
        "STORYMEM_RL_GEMINI_API_KEY",
        "STORYMEM_RL_IMAGE_API_KEY",
    }
)


class ProviderConfigError(RuntimeError):
    """The private provider configuration is missing or malformed."""


def private_api_key(environment_name: str) -> str:
    """Read one canonical secret; alternate sources are intentionally rejected."""

    if environment_name not in _SECRET_ENVIRONMENTS:
        raise ProviderConfigError(
            f"unsupported provider secret {environment_name!r}; expected one of "
            f"{sorted(_SECRET_ENVIRONMENTS)!r}"
        )
    value = os.getenv(environment_name, "").strip()
    if not value:
        raise ProviderConfigError(f"{environment_name} must be exported")
    return value


__all__ = [
    "ProviderConfigError",
    "private_api_key",
]
