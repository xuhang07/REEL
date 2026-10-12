"""Model-specific adapters behind the provider-neutral Memory Policy.

The public boundary accepts the user's immutable shot prompt and an accepted
portable evidence plan.  Prompt engineering and conversion into the target
generator's native prompt/condition layout are private adapter work; neither
operation is exposed as a Memory Policy tool or training-trajectory action.
"""

from .echo15 import (
    Echo15AdapterError,
    Echo15BackendAdapter,
    Echo15Capabilities,
    Echo15Compilation,
    Echo15RequestSpec,
)


__all__ = [
    "Echo15AdapterError",
    "Echo15BackendAdapter",
    "Echo15Capabilities",
    "Echo15Compilation",
    "Echo15RequestSpec",
]
