"""Injected infrastructure ports for the provider-neutral Memory Policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class VisualEvidenceCreationRequest:
    """Neutral request understood by an injected visual-evidence adapter.

    ``creation_spec`` contains only creator-affecting parameters. Policy
    bookkeeping such as requirement, subject, evidence, and representation
    IDs remains outside the adapter request.
    """

    request_id: str
    purpose: str
    creation_spec: Mapping[str, Any]
    input_paths: Sequence[Path]
    output_dir: Path


@dataclass(frozen=True)
class VisualEvidenceCreationResult:
    """Result returned by a visual-evidence adapter.

    ``private_receipt`` is persisted for audit but is never copied into a
    model-visible observation.
    """

    request_id: str
    output_path: Path
    private_receipt: Mapping[str, Any]


@runtime_checkable
class VisualEvidenceCreator(Protocol):
    """The only generation capability available inside Memory Policy.

    Implementations must treat ``request.request_id`` as an idempotency key:
    replaying the same request ID after a crash must return the same logical
    result rather than creating and charging for another image.
    """

    def create(
        self, request: VisualEvidenceCreationRequest
    ) -> VisualEvidenceCreationResult | Mapping[str, Any]: ...


__all__ = [
    "VisualEvidenceCreationRequest",
    "VisualEvidenceCreationResult",
    "VisualEvidenceCreator",
]
