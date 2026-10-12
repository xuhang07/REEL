from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import canonical_json, sha256_bytes


class PEManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedSkillBundle:
    route: str
    profile_id: str
    revision: str
    generator_profile_id: str
    resources: tuple[dict[str, Any], ...]
    bundle_sha256: str
    language_policy: str
    validation_contract: dict[str, Any]

    def as_observation(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "pe_profile_id": self.profile_id,
            "pe_revision": self.revision,
            "generator_profile_id": self.generator_profile_id,
            "resolved_resources": [dict(item) for item in self.resources],
            "resolved_bundle_sha256": self.bundle_sha256,
            "language_policy": self.language_policy,
            "validation_contract": dict(self.validation_contract),
        }


class PESkillResolver:
    """Resolve one immutable PE bundle without invoking a model.

    The manifest is the only router.  Callers cannot select a provider, file,
    revision, or arbitrary topic through the public tool arguments.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        # Official Director PE is repository-owned, but intentionally lives
        # outside the StoryMem adapter resource directory.  Keep this root
        # derived from this module (not from manifest input) so a caller cannot
        # redirect a resource entry to an arbitrary filesystem location.
        self.director_pe_root = (
            Path(__file__).resolve().parents[2] / "Director_Agent" / "pe"
        ).resolve()
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise PEManifestError("PE manifest must be an object")
        self.manifest = value
        self._resource_cache: dict[str, dict[str, Any]] = {}
        self._validate_manifest()

    @property
    def manifest_sha256(self) -> str:
        return sha256_bytes(self.manifest_path.read_bytes())

    def profile_locks(self) -> dict[str, Any]:
        """Return public immutable PE/generator profile metadata, never paths."""

        result: dict[str, Any] = {}
        for target in ("image", "video"):
            profile = self.manifest["profiles"][target]
            result[target] = {
                "pe_profile_id": str(profile.get("id") or f"{target}_canonical"),
                "pe_revision": str(
                    profile.get("revision") or self.manifest.get("revision") or ""
                ),
                "generator_profile_id": str(
                    profile.get("generator_profile_id") or f"{target}.current"
                ),
                "language_policy": str(
                    profile.get("language_policy") or "full_caption_locked"
                ),
                "validation_contract": dict(
                    profile.get("validation_contract") or {}
                ),
            }
        return {
            "pe_manifest_sha256": self.manifest_sha256,
            "profiles": result,
        }

    def _validate_manifest(self) -> None:
        resources = self.manifest.get("resources")
        routes = self.manifest.get("routes")
        profiles = self.manifest.get("profiles")
        if not isinstance(resources, dict) or not resources:
            raise PEManifestError("manifest.resources must be a non-empty object")
        if not isinstance(routes, dict):
            raise PEManifestError("manifest.routes must be an object")
        required_routes = {"image.compose", "image.revise", "video.compose", "video.revise"}
        if set(routes) != required_routes:
            raise PEManifestError(
                f"manifest.routes must contain exactly {sorted(required_routes)}"
            )
        if not isinstance(profiles, dict) or set(profiles) != {"image", "video"}:
            raise PEManifestError("manifest.profiles must contain exactly image and video")
        referenced: set[str] = set()
        for route, ids in routes.items():
            if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids):
                raise PEManifestError(f"route {route} must be a non-empty resource-id list")
            referenced.update(ids)
        context = self.manifest.get("context_resources") or {}
        self._collect_context_ids(context, referenced)
        story_ids = self.manifest.get("story_preparation") or []
        if not isinstance(story_ids, list) or not all(isinstance(item, str) for item in story_ids):
            raise PEManifestError("story_preparation must be a resource-id list")
        referenced.update(story_ids)
        unknown = referenced - set(resources)
        if unknown:
            raise PEManifestError(f"manifest references unknown resources: {sorted(unknown)}")
        for resource_id in resources:
            self._load_resource(resource_id)

    def _collect_context_ids(self, value: Any, destination: set[str]) -> None:
        if isinstance(value, str):
            destination.add(value)
        elif isinstance(value, list):
            for item in value:
                self._collect_context_ids(item, destination)
        elif isinstance(value, dict):
            for item in value.values():
                self._collect_context_ids(item, destination)

    def _load_resource(self, resource_id: str) -> dict[str, Any]:
        cached = self._resource_cache.get(resource_id)
        if cached is not None:
            return dict(cached)
        raw = self.manifest["resources"].get(resource_id)
        if not isinstance(raw, dict):
            raise PEManifestError(f"resource {resource_id!r} must be an object")
        relative = raw.get("path")
        if not isinstance(relative, str) or not relative:
            raise PEManifestError(f"resource {resource_id!r} has no path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise PEManifestError(f"resource path is not a safe relative path: {resource_id}")
        source_root = str(raw.get("source_root") or "manifest")
        if source_root == "manifest":
            allowed_root = self.root
        elif source_root == "director_pe":
            allowed_root = self.director_pe_root
        else:
            raise PEManifestError(
                f"resource {resource_id!r} has unsupported source_root {source_root!r}"
            )
        path = (allowed_root / relative_path).resolve()
        try:
            path.relative_to(allowed_root)
        except ValueError as exc:
            raise PEManifestError(f"resource escapes allowed root: {resource_id}") from exc
        if not path.is_file():
            raise PEManifestError(f"resource file is missing: {path}")
        content_bytes = path.read_bytes()
        actual = sha256_bytes(content_bytes)
        expected = str(raw.get("sha256") or "")
        if expected and expected != actual:
            raise PEManifestError(
                f"resource hash mismatch for {resource_id}: expected {expected}, got {actual}"
            )
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PEManifestError(f"resource is not UTF-8: {resource_id}") from exc
        record = {
            "resource_id": resource_id,
            "revision": str(raw.get("revision") or self.manifest.get("revision") or ""),
            "sha256": actual,
            "content": content,
        }
        self._resource_cache[resource_id] = record
        return dict(record)

    @staticmethod
    def _context_resource_ids(
        mapping: dict[str, Any], *, target: str, purpose: str, prompt_context: dict[str, Any]
    ) -> list[str]:
        result: list[str] = []
        image_purpose = mapping.get("image_purpose") if isinstance(mapping, dict) else None
        if target == "image" and isinstance(image_purpose, dict):
            value = image_purpose.get(purpose) or []
            result.extend(value if isinstance(value, list) else [value])
        image_references = (
            mapping.get("image_reference_inputs") if isinstance(mapping, dict) else None
        )
        if target == "image" and isinstance(image_references, dict):
            has_references = prompt_context.get("has_reference_inputs")
            if not isinstance(has_references, bool):
                raise PEManifestError(
                    "image prompt_context.has_reference_inputs must be boolean"
                )
            value = image_references.get("true" if has_references else "false") or []
            result.extend(value if isinstance(value, list) else [value])
        video_condition = mapping.get("video_condition") if isinstance(mapping, dict) else None
        condition = prompt_context.get("condition")
        if target == "video" and isinstance(video_condition, dict):
            key = "none" if condition is None else str(condition.get("relation") or "")
            value = video_condition.get(key) or []
            result.extend(value if isinstance(value, list) else [value])
        return [str(item) for item in result if item]

    def resolve(
        self,
        *,
        target: str,
        operation: str,
        purpose: str,
        prompt_context: dict[str, Any],
    ) -> ResolvedSkillBundle:
        route = f"{target}.{operation}"
        if route not in self.manifest["routes"]:
            raise PEManifestError(f"unsupported PE route: {route}")
        profile = self.manifest["profiles"].get(target)
        if not isinstance(profile, dict):
            raise PEManifestError(f"missing profile for target={target}")
        resource_ids = list(self.manifest["routes"][route])
        resource_ids.extend(
            self._context_resource_ids(
                self.manifest.get("context_resources") or {},
                target=target,
                purpose=purpose,
                prompt_context=prompt_context,
            )
        )
        deduplicated: list[str] = []
        for resource_id in resource_ids:
            if resource_id not in deduplicated:
                deduplicated.append(resource_id)
        resources = tuple(self._load_resource(resource_id) for resource_id in deduplicated)
        bundle_digest_input = [
            {"resource_id": row["resource_id"], "revision": row["revision"], "sha256": row["sha256"]}
            for row in resources
        ]
        return ResolvedSkillBundle(
            route=route,
            profile_id=str(profile.get("id") or f"{target}_canonical"),
            revision=str(profile.get("revision") or self.manifest.get("revision") or ""),
            generator_profile_id=str(profile.get("generator_profile_id") or f"{target}.current"),
            resources=resources,
            bundle_sha256=sha256_bytes(canonical_json(bundle_digest_input).encode("utf-8")),
            language_policy=str(profile.get("language_policy") or "full_caption_locked"),
            validation_contract=dict(profile.get("validation_contract") or {}),
        )

    def story_preparation_bundle(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._load_resource(item) for item in self.manifest.get("story_preparation") or [])
