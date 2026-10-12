from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .pe import PESkillResolver
from .util import (
    atomic_write_json,
    canonical_json,
    safe_name,
    sha256_file,
    sha256_json,
    sha256_text,
    utc_now,
)


STORY_PLAN_SCHEMA_VERSION = "storymem.canonical.story_plan.v1"
SOURCE_ASSET_RECEIPT_SCHEMA_VERSION = "storymem.canonical.source_assets.v1"
SOURCE_ASSET_RECEIPT_AUDIO_SCHEMA_VERSION = "storymem.canonical.source_assets.v2"
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$", re.IGNORECASE)
_STRICT_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SOURCE_ASSET_OBJECT_RE = re.compile(
    r"^objects/(?P<digest>[0-9a-f]{64})\.[a-z0-9]{1,12}$"
)
_SOURCE_ASSET_RECEIPT_FIELDS = frozenset(
    {"schema_version", "source_sha256", "assets", "receipt_sha256"}
)
_SOURCE_ASSET_RECEIPT_ROW_FIELDS = frozenset(
    {
        "asset_id",
        "scheduled_shots",
        "media_type",
        "mime_type",
        "size_bytes",
        "content_sha256",
        "object_relpath",
        "source_transport",
        "source_locator_sha256",
        "declared_sha256",
    }
)
_SOURCE_ASSET_RECEIPT_V2_ROW_FIELDS = frozenset(
    {*_SOURCE_ASSET_RECEIPT_ROW_FIELDS, "declared_asset_type", "speaker_binding"}
)
_SPEAKER_BINDING_FIELDS = frozenset(
    {"basis", "declared_name", "subject_id", "matching_method"}
)
_ASSET_LOCATOR_FIELDS = (
    "url",
    "path",
    "image_path",
    "video_path",
    "audio_path",
    "document_path",
)
_PERSON_TAG_RE = re.compile(r"^ID_[A-Z][A-Z0-9_]*$")
_ENTITY_TAG_RE = re.compile(r"^ENTITY_[1-9][0-9]*$")
_PUBLIC_TAG_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LEGACY_MATERIAL_TAG_RE = re.compile(
    r"^<(?P<kind>id|entity|scene|obj|object|prop|style)_(?P<index>[1-9][0-9]*)>$",
    re.IGNORECASE,
)
_RESERVED_TAG_PREFIXES = (
    "asset_",
    "cand_",
    "ext_",
    "mem_",
    "obs_",
    "repr_",
    "shotreq_",
    "upload_",
    "view_",
)

_EXPLICIT_ENTITY_WORDS = re.compile(
    r"(?ix)(?:"
    r"\b(?:animal|android|automaton|bot|cat|creature|dog|drone|fox|mascot|"
    r"non[- ]?human|puppy|robot)\b|"
    r"动物|安卓人|仿生人|机器人|机械人|非人|宠物|小狗|犬|猫|狐狸|精怪|生物|吉祥物"
    r")"
)
_EXPLICIT_HUMAN_WORDS = re.compile(
    r"(?ix)(?:"
    r"\b(?:adult|archivist|artist|astronaut|botanist|boy|chef|child|conductor|courier|"
    r"dancer|diver|elderly|explorer|farmer|girl|human|inspector|man|musician|neighbor|"
    r"neighbour|person|repairer|student|teenage|teenager|traveler|traveller|woman)\b|"
    r"人类|男人|女人|男孩|女孩|儿童|孩子|少年|少女|老人|学生|舞者|邻居|厨师|农民|"
    r"艺术家|宇航员|航天员|潜水员|检票员|修表匠|植物学家|音乐家|档案员"
    r")"
)
_STORY_PLAN_FIELDS = {
    "schema_version",
    "story_id",
    "title",
    "summary",
    "caption_language",
    "dialogue_language",
    "materials",
    "shots",
    "prompt_state",
}
_SHOT_PLAN_FIELDS = {
    "shot_index",
    "raw_request",
    "direct_continuation",
    "active_material_tags",
    "scene_state",
    "continuity_notes",
}
_GENERATOR_SEGMENT_RE = re.compile(
    r"(?:^|[\n;；])\s*(?:shot|cut|镜头)\s*#?\s*[0-9一二三四五六七八九十]+\s*[:：]",
    re.IGNORECASE,
)
_GENERATOR_TIMECODE_RE = re.compile(
    r"(?:\[\s*)?\d+(?:\.\d+)?\s*(?:s|秒)?\s*[-–—~～至到]\s*"
    r"\d+(?:\.\d+)?\s*(?:s|秒)(?:\s*\])?",
    re.IGNORECASE,
)
_GENERATOR_INTERNAL_RE = re.compile(
    r"(?:\b(?:get_pe_skill|register_prompt|generate_shot|generate_image|memory[_ -]?bank)\b|"
    r"\b(?:reference\s+(?:image\s*)?#?\s*\d+)\b|"
    r"(?:参考图(?:像|片)?\s*[一二三四五六七八九十0-9]+))",
    re.IGNORECASE,
)
_GENERATOR_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"keep\s+the\s+same\s+character\s+identity\s*,\s*wardrobe\s*,\s*"
    r"prop\s+design\s*,\s*scene\s+layout\s*,\s*lighting|"
    r"no\s+subtitles\s*,\s*captions\s*,\s*ui\s+overlays\s*,\s*"
    r"watermarks\s*,\s*logos\s*,\s*(?:or\s+)?readable\s+text|"
    r"visual\s+style\s*:\s*full-frame\s+cinematic\s+scene"
    r")",
    re.IGNORECASE,
)


def _validate_raw_next_shot_request(
    value: str,
    *,
    label: str,
    material_tags: set[str] | None = None,
) -> None:
    """Reject obvious generator captions while preserving ordinary user prose.

    Story PE is deliberately *not* the image/video prompt writer.  These
    markers are narrow, structural signals from our former provider-ready
    caption format; creative camera wording or an explicit user constraint is
    otherwise allowed through unchanged.
    """

    if _GENERATOR_SEGMENT_RE.search(value):
        raise ValueError(f"{label} looks generator-ready: numbered shot/cut segments are forbidden")
    if _GENERATOR_TIMECODE_RE.search(value):
        raise ValueError(f"{label} looks generator-ready: timeline timecodes are forbidden")
    if _GENERATOR_INTERNAL_RE.search(value):
        raise ValueError(f"{label} looks generator-ready: provider/tool reference syntax is forbidden")
    if _GENERATOR_BOILERPLATE_RE.search(value):
        raise ValueError(
            f"{label} looks generator-ready: legacy prompt boilerplate is forbidden"
        )
    if re.search(r"\bno\s+ocr\b", value, re.IGNORECASE):
        raise ValueError(f"{label} looks generator-ready: no-OCR boilerplate is forbidden")
    # The per-shot USER turn must look like an ordinary request.  Canonical
    # registry tags are deliberately hidden until view_material_registry
    # returns them; allowing Story PE to paste ID_A/SCENE_A/etc. into the USER
    # message would leak the very state that the retrieval trajectory is meant
    # to discover.
    for tag in sorted(material_tags or (), key=len, reverse=True):
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(tag)}(?![A-Za-z0-9_])",
            value,
        ):
            raise ValueError(
                f"{label} exposes material tag {tag!r}; use a natural name in the raw request"
            )


@dataclass(frozen=True)
class PreparedStory:
    plan: dict[str, Any]
    provenance: dict[str, Any]


def load_story_source(path: str | Path) -> dict[str, Any]:
    source_path = Path(path).resolve()
    value = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"story source must be a JSON object: {source_path}")
    value = copy.deepcopy(value)
    value["_source_path"] = str(source_path)
    return value


def _source_shots(source: dict[str, Any]) -> list[str]:
    raw = source.get("shots")
    if isinstance(raw, list) and raw:
        result: list[str] = []
        for item in raw:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(str(item.get("raw_request") or item.get("prompt") or item.get("summary") or ""))
        return [item for item in result if item.strip()]
    prompts = source.get("prompts")
    if isinstance(prompts, list):
        return [str(item) for item in prompts if str(item).strip()]
    return []


def _person_tag(index: int) -> str:
    """Return ID_A, ..., ID_Z, ID_AA, ... for a zero-based ordinal."""

    if index < 0:
        raise ValueError("person tag index must be non-negative")
    letters = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"ID_{letters}"


def _entity_tag(index: int) -> str:
    if index < 0:
        raise ValueError("entity tag index must be non-negative")
    return f"ENTITY_{index + 1}"


def _declared_character_category(
    item: dict[str, Any], *, label: str, fail_if_ambiguous: bool = True
) -> str | None:
    """Classify a declared character using ordered, fail-closed evidence.

    Descriptions often mention another subject (for example, a robotics
    student may carry a helper drone), so combining every field into one bag
    of words silently changes the student's taxonomy.  Structured declarations
    therefore win, identity-bearing fields are considered next, and the free
    description is consulted only when neither earlier layer has a signal.
    Conflicting signals inside a layer are always rejected.
    """

    def lexical_signals(values: list[Any]) -> set[str]:
        text = " ".join(str(value or "") for value in values)
        signals: set[str] = set()
        if _EXPLICIT_ENTITY_WORDS.search(text):
            signals.add("entity")
        if _EXPLICIT_HUMAN_WORDS.search(text):
            signals.add("person")
        return signals

    def resolve(signals: set[str], *, layer: str) -> str | None:
        if len(signals) > 1:
            raise ValueError(
                f"{label} has conflicting human and non-human evidence in {layer}"
            )
        return next(iter(signals)) if signals else None

    explicit_signals: set[str] = set()
    explicit_person_values = {
        "person",
        "human",
        "human person",
        "human_person",
        "homo sapiens",
    }
    explicit_entity_values = {
        "entity",
        "nonhuman",
        "non human",
        "non-human",
        "animal",
        "robot",
        "android",
        "creature",
        "mascot",
    }
    for key in ("category", "entity_type", "subject_type", "species"):
        raw = item.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        normalized = re.sub(r"\s+", " ", str(raw).strip().casefold())
        if normalized in explicit_person_values:
            explicit_signals.add("person")
        elif normalized in explicit_entity_values:
            explicit_signals.add("entity")
        else:
            explicit_signals.update(lexical_signals([raw]))
    if "is_human" in item and item.get("is_human") is not None:
        if not isinstance(item["is_human"], bool):
            raise ValueError(f"{label}.is_human must be boolean when declared")
        explicit_signals.add("person" if item["is_human"] else "entity")
    explicit_category = resolve(
        explicit_signals, layer="structured taxonomy declarations"
    )
    if explicit_category is not None:
        return explicit_category

    identity_category = resolve(
        lexical_signals([item.get(key) for key in ("name", "role", "type")]),
        layer="name/role/type",
    )
    if identity_category is not None:
        return identity_category

    description_category = resolve(
        lexical_signals([item.get("description")]),
        layer="description",
    )
    if description_category is not None:
        return description_category

    if fail_if_ambiguous:
        raise ValueError(
            f"{label} does not explicitly establish human versus non-human identity; "
            "set category=person|entity or is_human=true|false"
        )
    return None


def validate_source_character_taxonomy(source: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and count source character declarations without side effects."""

    metadata = source.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object when present")

    category_counts = {"person": 0, "entity": 0}
    source_field_counts: dict[str, dict[str, int]] = {}
    for key in ("primary_character", "secondary_character"):
        item = metadata.get(key)
        if item is None:
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"metadata.{key} must be an object when present")
        category = _declared_character_category(
            dict(item), label=f"metadata.{key}"
        )
        category_counts[category] += 1
        source_field_counts[f"metadata.{key}"] = {
            "person": int(category == "person"),
            "entity": int(category == "entity"),
        }
    return {
        "character_count": sum(category_counts.values()),
        "category_counts": category_counts,
        "source_field_counts": source_field_counts,
    }


def _canonical_character_tag(category: str, ordinal: int) -> str:
    if category == "person":
        return _person_tag(ordinal)
    if category == "entity":
        return _entity_tag(ordinal)
    raise ValueError(f"unsupported character category: {category}")


def _canonicalize_local_materials(
    materials: list[dict[str, Any]], shots: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Canonicalize legacy preapproved tags without a semantic model call.

    Preapproved input is an audited escape hatch, but it must still obey the
    same public-tag protocol as remote story PE.  Mapping is deterministic by
    material category and first appearance.  Opaque legacy placeholders are
    not retained as public aliases; readable non-canonical labels are.
    """

    reserved = {
        str(item.get("tag") or "")
        for item in materials
        if (
            item.get("category") == "person"
            and _PERSON_TAG_RE.fullmatch(str(item.get("tag") or ""))
        )
        or (
            item.get("category") == "entity"
            and _ENTITY_TAG_RE.fullmatch(str(item.get("tag") or ""))
        )
        or (
            item.get("category") not in {"person", "entity"}
            and _PUBLIC_TAG_RE.fullmatch(str(item.get("tag") or ""))
            and not str(item.get("tag") or "").casefold().startswith(
                _RESERVED_TAG_PREFIXES
            )
        )
    }
    counters = {"person": 0, "entity": 0, "scene": 0, "object": 0, "style": 0}
    mapping: dict[str, tuple[str, str]] = {}
    natural_names: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []

    def allocate(category: str) -> str:
        while True:
            ordinal = counters[category]
            counters[category] += 1
            candidate = (
                _person_tag(ordinal)
                if category == "person"
                else _entity_tag(ordinal)
                if category == "entity"
                else f"{category.upper()}_{ordinal + 1}"
            )
            if candidate not in reserved:
                reserved.add(candidate)
                return candidate

    for item in materials:
        row = copy.deepcopy(item)
        category = str(row["category"])
        raw_tag = str(row["tag"])
        valid = (
            bool(_PERSON_TAG_RE.fullmatch(raw_tag))
            if category == "person"
            else bool(_ENTITY_TAG_RE.fullmatch(raw_tag))
            if category == "entity"
            else bool(_PUBLIC_TAG_RE.fullmatch(raw_tag))
            and not _PERSON_TAG_RE.fullmatch(raw_tag)
            and not _ENTITY_TAG_RE.fullmatch(raw_tag)
            and not raw_tag.casefold().startswith(_RESERVED_TAG_PREFIXES)
        )
        prior_mapping = mapping.get(raw_tag)
        if prior_mapping is not None and prior_mapping[0] != category:
            raise ValueError(f"legacy material tag is reused across categories: {raw_tag!r}")
        canonical = raw_tag if valid else (prior_mapping[1] if prior_mapping else None)
        if canonical is None:
            canonical = allocate(category)
        mapping[raw_tag] = (category, canonical)
        row["tag"] = canonical
        aliases = list(row.get("aliases") or [])
        if (
            canonical != raw_tag
            and not _LEGACY_MATERIAL_TAG_RE.fullmatch(raw_tag)
            and raw_tag not in aliases
        ):
            aliases.append(raw_tag)
        row["aliases"] = aliases
        natural_names[raw_tag] = next(
            (
                str(alias).strip()
                for alias in aliases
                if str(alias).strip()
                and str(alias).strip() != raw_tag
                and not _LEGACY_MATERIAL_TAG_RE.fullmatch(str(alias).strip())
            ),
            str(row.get("description") or "").strip(),
        )
        normalized.append(row)

    rewritten_shots = list(shots)
    for raw_tag, (_, canonical) in sorted(mapping.items(), key=lambda row: -len(row[0])):
        natural = natural_names.get(raw_tag) or canonical
        rewritten_shots = [text.replace(raw_tag, natural) for text in rewritten_shots]
    return normalized, rewritten_shots


def local_story_plan(source: dict[str, Any]) -> dict[str, Any]:
    """Normalize an already PE-approved source into the canonical protocol."""
    shots = _source_shots(source)
    if not shots:
        raise ValueError("story source contains no shots/prompts")
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    language = str(metadata.get("caption_language") or source.get("caption_language") or "English")
    materials: list[dict[str, Any]] = []
    for category, key in ((None, "primary_character"), (None, "secondary_character"), ("object", "prop"), ("style", "style")):
        item = metadata.get(key)
        if isinstance(item, dict):
            tag = str(item.get("tag") or "").strip()
            description = str(item.get("description") or item.get("role") or "").strip()
            if not tag or not description:
                continue
            resolved_category = (
                _declared_character_category(item, label=f"metadata.{key}")
                if category is None
                else category
            )
            materials.append(
                {
                    "tag": tag,
                    "category": resolved_category,
                    "description": description,
                    "aliases": [str(item.get("role"))] if item.get("role") else [],
                }
            )
    for item in metadata.get("scenes") or []:
        if isinstance(item, dict):
            tag = str(item.get("tag") or "").strip()
            description = str(item.get("description") or item.get("name") or "").strip()
            if not tag or not description:
                continue
            materials.append(
                {
                    "tag": tag,
                    "category": "scene",
                    "description": description,
                    "aliases": [str(item.get("name"))] if item.get("name") else [],
                }
            )
    materials, shots = _canonicalize_local_materials(materials, shots)
    return validate_story_plan(
        {
            "schema_version": STORY_PLAN_SCHEMA_VERSION,
            "story_id": str(source.get("story_id") or safe_name(Path(source["_source_path"]).stem, "story")),
            "title": str(source.get("title") or "Untitled story"),
            "summary": str(metadata.get("summary") or source.get("title") or ""),
            "caption_language": language,
            "dialogue_language": str(metadata.get("dialogue_language") or ("Mandarin Chinese" if "Chinese" in language else "English")),
            "materials": materials,
            "shots": [
                {
                    "shot_index": index,
                    "raw_request": prompt,
                    "direct_continuation": "continu" in prompt.casefold() or "延续" in prompt,
                    "active_material_tags": [],
                    "scene_state": "",
                    "continuity_notes": "",
                }
                for index, prompt in enumerate(shots, start=1)
            ],
            "prompt_state": {
                "story_state_notes": str(metadata.get("summary") or source.get("title") or ""),
                "material_state_notes": {
                    item["tag"]: item["description"] for item in materials
                },
            },
        }
    )


def validate_story_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("story PE output must be a JSON object")
    plan = copy.deepcopy(value)
    if set(plan) != _STORY_PLAN_FIELDS:
        raise ValueError(
            "story plan must contain exactly "
            f"{sorted(_STORY_PLAN_FIELDS)}; missing={sorted(_STORY_PLAN_FIELDS - set(plan))}, "
            f"extra={sorted(set(plan) - _STORY_PLAN_FIELDS)}"
        )
    if plan["schema_version"] != STORY_PLAN_SCHEMA_VERSION:
        raise ValueError(
            "story plan schema_version must be " + STORY_PLAN_SCHEMA_VERSION
        )
    for field, allow_empty in (("story_id", False), ("title", False), ("summary", True)):
        item = plan[field]
        if (
            not isinstance(item, str)
            or (not allow_empty and not item)
            or item != item.strip()
        ):
            suffix = "a trimmed string" if allow_empty else "a trimmed non-empty string"
            raise ValueError(f"story plan {field} must be {suffix}")
    if safe_name(plan["story_id"], "story") != plan["story_id"]:
        raise ValueError("story plan story_id must already be a canonical filesystem-safe ID")

    materials = plan["materials"]
    if not isinstance(materials, list):
        raise ValueError("story plan materials must be an array")
    normalized_materials: list[dict[str, Any]] = []
    material_tags: set[str] = set()
    material_fields = {"tag", "category", "description", "aliases"}
    for index, item in enumerate(materials):
        if not isinstance(item, dict) or set(item) != material_fields:
            raise ValueError(
                f"story plan materials[{index}] must contain exactly {sorted(material_fields)}"
            )
        tag = item.get("tag")
        description = item.get("description")
        category = item.get("category")
        aliases = item.get("aliases")
        if not isinstance(tag, str) or not tag or tag != tag.strip():
            raise ValueError(f"story plan materials[{index}].tag must be a trimmed non-empty string")
        if tag in material_tags:
            raise ValueError(f"story plan material tag is duplicated: {tag}")
        if category not in {"person", "entity", "scene", "object", "style"}:
            raise ValueError(f"story plan materials[{index}].category is invalid")
        if category == "person" and not _PERSON_TAG_RE.fullmatch(tag):
            raise ValueError(
                f"story plan person tag must be a public ID such as ID_A/ID_B: {tag!r}"
            )
        if category == "entity" and not _ENTITY_TAG_RE.fullmatch(tag):
            raise ValueError(
                f"story plan entity tag must be ENTITY_1, ENTITY_2, ...: {tag!r}"
            )
        if category not in {"person", "entity"} and (
            not _PUBLIC_TAG_RE.fullmatch(tag)
            or _PERSON_TAG_RE.fullmatch(tag)
            or _ENTITY_TAG_RE.fullmatch(tag)
            or tag.casefold().startswith(_RESERVED_TAG_PREFIXES)
        ):
            raise ValueError(
                f"story plan scene/object/style tag must use its own public uppercase namespace: {tag!r}"
            )
        descriptor_category = _declared_character_category(
            {
                "description": description,
                "role": " ".join(
                    str(alias)
                    for alias in (aliases if isinstance(aliases, list) else [])
                ),
            },
            label=f"story plan materials[{index}]",
            fail_if_ambiguous=False,
        )
        if category == "person" and descriptor_category == "entity":
            raise ValueError(
                f"story plan materials[{index}] describes a non-human but uses category=person"
            )
        if category == "entity" and descriptor_category == "person":
            raise ValueError(
                f"story plan materials[{index}] describes a human but uses category=entity"
            )
        if not isinstance(description, str) or not description or description != description.strip():
            raise ValueError(
                f"story plan materials[{index}].description must be a trimmed non-empty string"
            )
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias or alias != alias.strip() for alias in aliases
        ):
            raise ValueError(f"story plan materials[{index}].aliases must contain trimmed strings")
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"story plan materials[{index}].aliases contains duplicates")
        material_tags.add(tag)
        normalized_materials.append(
            {
                "tag": tag,
                "category": category,
                "description": description,
                "aliases": list(aliases),
            }
        )

    shots = plan.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("story plan requires a non-empty shots array")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(shots, start=1):
        if not isinstance(item, dict) or set(item) != _SHOT_PLAN_FIELDS:
            actual = set(item) if isinstance(item, dict) else set()
            raise ValueError(
                f"story plan shots[{index - 1}] must contain exactly "
                f"{sorted(_SHOT_PLAN_FIELDS)}; missing={sorted(_SHOT_PLAN_FIELDS - actual)}, "
                f"extra={sorted(actual - _SHOT_PLAN_FIELDS)}"
            )
        shot_index = item["shot_index"]
        if type(shot_index) is not int or shot_index != index:
            raise ValueError(
                f"story plan shots[{index - 1}].shot_index must be integer {index}"
            )
        raw_request = item.get("raw_request")
        if not isinstance(raw_request, str) or not raw_request.strip():
            raise ValueError(f"story plan shots[{index - 1}].raw_request must be non-empty")
        if raw_request != raw_request.strip():
            raise ValueError(f"story plan shots[{index - 1}].raw_request has surrounding whitespace")
        _validate_raw_next_shot_request(
            raw_request,
            label=f"story plan shots[{index - 1}].raw_request",
            material_tags=material_tags,
        )
        fingerprint = sha256_text(raw_request.casefold())
        if fingerprint in seen:
            raise ValueError("story plan contains duplicate shot requests")
        seen.add(fingerprint)
        direct_continuation = item["direct_continuation"]
        if not isinstance(direct_continuation, bool):
            raise ValueError(
                f"story plan shots[{index - 1}].direct_continuation must be boolean"
            )
        if index == 1 and direct_continuation:
            raise ValueError(
                "story plan first shot cannot be a direct continuation because no "
                "previous accepted tail exists"
            )
        active_tags = item["active_material_tags"]
        if not isinstance(active_tags, list) or any(
            not isinstance(tag, str) or not tag or tag != tag.strip()
            for tag in active_tags
        ):
            raise ValueError(
                f"story plan shots[{index - 1}].active_material_tags must be a string array"
            )
        if len(set(active_tags)) != len(active_tags):
            raise ValueError(
                f"story plan shots[{index - 1}].active_material_tags contains duplicates"
            )
        unknown_tags = sorted(set(active_tags) - material_tags)
        if unknown_tags:
            raise ValueError(
                f"story plan shots[{index - 1}] references unknown material tags: {unknown_tags}"
            )
        active_people = [
            tag
            for tag in active_tags
            if any(
                material["tag"] == tag and material["category"] == "person"
                for material in normalized_materials
            )
        ]
        if len(active_people) > 2:
            raise ValueError(
                f"story plan shots[{index - 1}] has more than two active human "
                "participants, which the frozen Director profile cannot represent"
            )
        scene_state = item["scene_state"]
        continuity_notes = item["continuity_notes"]
        if (
            not isinstance(scene_state, str)
            or scene_state != scene_state.strip()
            or not isinstance(continuity_notes, str)
            or continuity_notes != continuity_notes.strip()
        ):
            raise ValueError(
                f"story plan shots[{index - 1}] scene_state/continuity_notes must be trimmed strings"
            )
        normalized.append(
            {
                "shot_index": index,
                "raw_request": raw_request,
                "direct_continuation": direct_continuation,
                "active_material_tags": list(active_tags),
                "scene_state": scene_state,
                "continuity_notes": continuity_notes,
            }
        )
    language = plan["caption_language"]
    if not isinstance(language, str):
        raise ValueError("caption_language must be a string")
    if language not in {"English", "Simplified Chinese"}:
        raise ValueError("caption_language must be English or Simplified Chinese")
    dialogue_language = plan["dialogue_language"]
    if not isinstance(dialogue_language, str):
        raise ValueError("dialogue_language must be a string")
    expected_dialogue = "English" if language == "English" else "Mandarin Chinese"
    if dialogue_language != expected_dialogue:
        raise ValueError("dialogue_language must align with caption_language")
    prompt_state = plan["prompt_state"]
    prompt_state_fields = {"story_state_notes", "material_state_notes"}
    if not isinstance(prompt_state, dict) or set(prompt_state) != prompt_state_fields:
        raise ValueError(
            "story plan prompt_state must contain exactly story_state_notes and material_state_notes"
        )
    story_state_notes = prompt_state["story_state_notes"]
    material_state_notes = prompt_state["material_state_notes"]
    if not isinstance(story_state_notes, str) or story_state_notes != story_state_notes.strip():
        raise ValueError("prompt_state.story_state_notes must be a trimmed string")
    if not isinstance(material_state_notes, dict) or any(
        not isinstance(key, str)
        or not key
        or key != key.strip()
        or not isinstance(item, str)
        or item != item.strip()
        for key, item in material_state_notes.items()
    ):
        raise ValueError("prompt_state.material_state_notes must map trimmed tags to trimmed strings")
    unknown_state_tags = sorted(set(material_state_notes) - material_tags)
    if unknown_state_tags:
        raise ValueError(
            f"prompt_state.material_state_notes references unknown tags: {unknown_state_tags}"
        )
    normalized_prompt_state = {
        "story_state_notes": story_state_notes,
        "material_state_notes": copy.deepcopy(material_state_notes),
    }
    canonical_json(normalized_prompt_state)
    plan = {
            "schema_version": STORY_PLAN_SCHEMA_VERSION,
            "story_id": plan["story_id"],
            "title": plan["title"],
            "summary": plan["summary"],
            "caption_language": language,
            "dialogue_language": dialogue_language,
            "materials": normalized_materials,
            "shots": normalized,
            "prompt_state": normalized_prompt_state,
        }
    return plan


def _normalized_words(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", str(value or "").casefold()))


def _normalized_phrase(value: Any) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", str(value or "").casefold())
    )


def _source_character_expectations(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic human/entity declarations from structured source.

    The current controlled corpus declares characters in metadata.  The old
    ``<id_N>`` spelling is deliberately ignored as evidence of personhood: the
    accompanying semantic declaration must establish human versus non-human.
    """

    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    counters = {"person": 0, "entity": 0}
    expectations: list[dict[str, Any]] = []
    for key in ("primary_character", "secondary_character"):
        item = metadata.get(key)
        if not isinstance(item, dict) or not str(item.get("tag") or "").strip():
            continue
        category = _declared_character_category(item, label=f"metadata.{key}")
        assert category in counters
        aliases = item.get("aliases") or []
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise ValueError(
                f"metadata.{key}.aliases must be an array of non-empty strings"
            )
        raw_tag = str(item["tag"]).strip()
        if category == "person" and _PERSON_TAG_RE.fullmatch(raw_tag):
            canonical_tag = raw_tag
        elif category == "entity" and _ENTITY_TAG_RE.fullmatch(raw_tag):
            canonical_tag = raw_tag
        else:
            canonical_tag = _canonical_character_tag(category, counters[category])
        counters[category] += 1
        expectations.append(
            {
                "source_field": f"metadata.{key}",
                "source_tag": raw_tag,
                "category": category,
                "canonical_tag": canonical_tag,
                "names": [
                    str(item.get(field) or "").strip()
                    for field in ("name", "role")
                    if str(item.get(field) or "").strip()
                ]
                + [
                    str(alias).strip()
                    for alias in aliases
                ],
                "description": str(item.get("description") or "").strip(),
            }
        )
    return expectations


def _normalized_speaker_name(value: Any) -> str:
    """Normalize a natural speaker label without introducing fuzzy matching."""

    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()


def _resolve_source_speaker_declaration(
    source: Mapping[str, Any], speaker_name: str
) -> dict[str, Any]:
    """Resolve one user-declared speaker to exactly one declared human.

    Voice identity is too consequential for token overlap, substring matching,
    or ordinal guessing.  The user-facing name must exactly equal one declared
    human name, role, or alias after mechanical Unicode/whitespace/case
    normalization.
    """

    normalized = _normalized_speaker_name(speaker_name)
    if not normalized:
        raise ValueError("voice_reference speaker_name must be non-empty")
    matches: dict[str, dict[str, Any]] = {}
    for expected in _source_character_expectations(dict(source)):
        labels = {
            _normalized_speaker_name(value)
            for value in expected.get("names") or []
            if _normalized_speaker_name(value)
        }
        if normalized in labels:
            matches[str(expected["source_field"])] = expected
    if not matches:
        raise ValueError(
            f"voice_reference speaker_name {speaker_name!r} does not exactly match "
            "a declared character name, role, or alias"
        )
    if len(matches) != 1:
        raise ValueError(
            f"voice_reference speaker_name {speaker_name!r} is ambiguous across "
            f"declared characters: {sorted(matches)}"
        )
    resolved = next(iter(matches.values()))
    if resolved.get("category") != "person":
        raise ValueError(
            f"voice_reference speaker_name {speaker_name!r} resolves to a non-human "
            "entity; voice references require a declared human"
        )
    return copy.deepcopy(resolved)


def validate_story_plan_against_source(
    plan: dict[str, Any], source: dict[str, Any]
) -> None:
    """Reject source/plan taxonomy drift before any media call is possible."""

    by_tag = {
        str(item["tag"]): item
        for item in plan.get("materials") or []
        if isinstance(item, dict) and item.get("tag")
    }
    seen_expected: set[str] = set()
    for expected in _source_character_expectations(source):
        tag = str(expected["canonical_tag"])
        if tag in seen_expected:
            raise ValueError(f"source character declarations collide on canonical tag {tag!r}")
        seen_expected.add(tag)
        material = by_tag.get(tag)
        if not material:
            raise ValueError(
                f"story PE omitted {expected['source_field']} canonical material {tag!r}"
            )
        if material.get("category") != expected["category"]:
            raise ValueError(
                f"story PE classified {expected['source_field']} as {material.get('category')!r}; "
                f"expected {expected['category']!r} with tag {tag!r}"
            )

        plan_phrases = {
            _normalized_phrase(value)
            for value in [material.get("description"), *(material.get("aliases") or [])]
            if _normalized_phrase(value)
        }
        plan_text = " ".join(sorted(plan_phrases))
        names = [
            _normalized_phrase(value)
            for value in expected.get("names") or []
            if _normalized_phrase(value)
        ]
        if names and not any(
            name in plan_phrases or name in plan_text for name in names
        ):
            raise ValueError(
                f"story PE material {tag!r} lost the natural name/role for "
                f"{expected['source_field']}"
            )
        source_words = _normalized_words(
            " ".join([*(expected.get("names") or []), expected.get("description") or ""])
        )
        plan_words = _normalized_words(plan_text)
        if source_words and not (source_words & plan_words):
            raise ValueError(
                f"story PE material {tag!r} has no deterministic semantic overlap with "
                f"{expected['source_field']}"
            )

    upload_schedule = source.get("uploaded_assets_by_shot")
    has_voice_reference = isinstance(upload_schedule, Mapping) and any(
        isinstance(raw_items, list)
        and any(
            isinstance(item, Mapping)
            and str(item.get("declared_asset_type") or "").strip().lower()
            == "voice_reference"
            for item in raw_items
        )
        for raw_items in upload_schedule.values()
    )
    shots_by_index = {
        int(item.get("shot_index", -1)): item
        for item in plan.get("shots") or []
        if isinstance(item, Mapping)
    }
    for spec in _source_asset_specs(source) if has_voice_reference else []:
        speaker = spec.get("speaker_declaration") or {}
        subject_id = str(speaker.get("canonical_tag") or "")
        shot_index = int(spec["shot_index"])
        shot = shots_by_index.get(shot_index)
        active_tags = set(shot.get("active_material_tags") or []) if shot else set()
        if subject_id not in active_tags:
            raise ValueError(
                f"voice_reference {spec['asset_id']!r} is scheduled for shot "
                f"{shot_index}, but its speaker {subject_id!r} is not active in that shot"
            )


def _validate_story_plan_for_director_pe(plan: Mapping[str, Any]) -> None:
    """Require an unambiguous hand-off to the frozen Director v7 writer."""

    language = str(plan.get("caption_language") or "")
    materials = {
        str(item.get("tag") or ""): dict(item)
        for item in plan.get("materials") or []
        if isinstance(item, Mapping)
    }
    for index, shot in enumerate(plan.get("shots") or [], start=1):
        if not isinstance(shot, Mapping):
            continue
        raw_request = str(shot.get("raw_request") or "")
        if language == "English":
            silent_count = len(
                re.findall(r"(?:^|\s)No spoken dialogue\.(?=\s|$)", raw_request)
            )
            speaking_count = len(
                re.findall(r'(?:^|\s)Spoken dialogue:\s*["“].+?["”]', raw_request)
            )
        else:
            silent_count = raw_request.count("无对白。")
            speaking_count = len(
                re.findall(r"对白[：:]\s*[“\"].+?[”\"]", raw_request)
            )
        if (
            (silent_count > 0) == (speaking_count > 0)
            or silent_count > 1
            or speaking_count > 1
        ):
            raise ValueError(
                f"story plan shots[{index - 1}].raw_request must contain exactly one "
                "language-matched explicit speech intent: full dialogue or no dialogue"
            )
        active_people = [
            str(tag)
            for tag in shot.get("active_material_tags") or []
            if materials.get(str(tag), {}).get("category") == "person"
        ]
        if speaking_count and not active_people:
            raise ValueError(
                f"story plan shots[{index - 1}] requests spoken dialogue without an "
                "active declared human speaker; this would make Director invent an "
                "unregistered ID_A"
            )


def _json_result(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(value, tuple) and len(value) == 2:
        body, provenance = value
        return dict(body), dict(provenance)
    if isinstance(value, dict) and "value" in value:
        return dict(value["value"]), dict(value.get("provenance") or {})
    if hasattr(value, "value"):
        return dict(value.value), dict(getattr(value, "provenance", {}) or {})
    if isinstance(value, dict):
        return value, {}
    raise TypeError("teacher call_json returned an unsupported value")


def _validate_remote_story_pe_api_receipt(
    value: Any,
    *,
    label: str,
    require_text_hash: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    receipt = copy.deepcopy(dict(value))
    if receipt.get("assistant_source") != "remote_closed_model_api":
        raise ValueError(f"{label} is not a remote closed-model API receipt")
    if receipt.get("api_style") not in {"openai_chat", "gemini"}:
        raise ValueError(f"{label}.api_style is invalid")
    if not isinstance(receipt.get("model"), str) or not receipt["model"].strip():
        raise ValueError(f"{label}.model is missing")
    digest_fields = [
        "endpoint_sha256",
        "request_payload_sha256",
        "response_sha256",
    ]
    if require_text_hash:
        digest_fields.append("assistant_text_sha256")
    for key in digest_fields:
        digest = receipt.get(key)
        if not isinstance(digest, str) or not _STRICT_SHA256_RE.fullmatch(digest):
            raise ValueError(f"{label}.{key} is not a canonical SHA-256 digest")
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(f"{label}.attempts must be a non-empty array")
    for index, attempt in enumerate(attempts, start=1):
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("attempt") != index
            or type(attempt.get("http_status")) is not int
            or not isinstance(attempt.get("trace_id"), str)
            or not attempt["trace_id"].strip()
        ):
            raise ValueError(f"{label}.attempts[{index - 1}] is malformed")
    final_status = attempts[-1].get("http_status")
    if not 200 <= final_status < 300:
        raise ValueError(f"{label} has no terminal successful HTTP attempt")
    final_response_hash = attempts[-1].get("response_sha256")
    if final_response_hash is not None and final_response_hash != receipt["response_sha256"]:
        raise ValueError(f"{label} response hash differs from its final HTTP attempt")
    return receipt


def validate_story_pe_provenance(
    value: Any,
    *,
    expected_story_plan_sha256: str,
    expected_pe_manifest_sha256: str,
    expected_resource_bundle: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate the immutable remote Teacher receipt bound to one story plan."""

    if not isinstance(value, Mapping):
        raise ValueError("story PE provenance must be an object")
    provenance = copy.deepcopy(dict(value))
    if provenance.get("mode") != "remote_model_story_pe":
        raise ValueError("story PE must come from the remote closed-model API")
    if provenance.get("story_plan_sha256") != expected_story_plan_sha256:
        raise ValueError("story PE receipt is not bound to this story plan")
    if not _STRICT_SHA256_RE.fullmatch(expected_story_plan_sha256):
        raise ValueError("expected story plan hash is not a canonical SHA-256 digest")
    if provenance.get("pe_manifest_sha256") != expected_pe_manifest_sha256:
        raise ValueError("story PE receipt uses a different PE manifest")
    if not _STRICT_SHA256_RE.fullmatch(expected_pe_manifest_sha256):
        raise ValueError("expected PE manifest hash is not a canonical SHA-256 digest")

    resources = provenance.get("resource_bundle")
    if not isinstance(resources, list) or not resources:
        raise ValueError("story PE resource_bundle must be a non-empty array")
    normalized_resources: list[dict[str, Any]] = []
    for index, resource in enumerate(resources):
        if not isinstance(resource, Mapping) or set(resource) != {
            "resource_id",
            "revision",
            "sha256",
        }:
            raise ValueError(f"story PE resource_bundle[{index}] has an invalid shape")
        normalized = {
            "resource_id": str(resource["resource_id"]),
            "revision": str(resource["revision"]),
            "sha256": str(resource["sha256"]),
        }
        if (
            not normalized["resource_id"]
            or not normalized["revision"]
            or not _STRICT_SHA256_RE.fullmatch(normalized["sha256"])
        ):
            raise ValueError(f"story PE resource_bundle[{index}] is invalid")
        normalized_resources.append(normalized)
    if resources != normalized_resources:
        raise ValueError("story PE resource_bundle is not canonical")
    if expected_resource_bundle is not None and resources != expected_resource_bundle:
        raise ValueError("story PE resource bundle differs from the frozen resolver")
    if provenance.get("resource_bundle_sha256") != sha256_json(resources):
        raise ValueError("story PE resource bundle hash mismatch")

    semantic_attempts = provenance.get("semantic_attempts")
    if not isinstance(semantic_attempts, list) or not semantic_attempts:
        raise ValueError("story PE semantic_attempts must be a non-empty array")
    for index, attempt in enumerate(semantic_attempts, start=1):
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("attempt") != index
            or attempt.get("status") not in {"accepted", "rejected"}
        ):
            raise ValueError(f"story PE semantic_attempts[{index - 1}] is malformed")
        _validate_remote_story_pe_api_receipt(
            attempt.get("api_provenance"),
            label=f"story PE semantic_attempts[{index - 1}].api_provenance",
            require_text_hash=attempt.get("status") == "accepted",
        )
    accepted = semantic_attempts[-1]
    if accepted.get("status") != "accepted" or any(
        attempt.get("status") == "accepted" for attempt in semantic_attempts[:-1]
    ):
        raise ValueError("story PE must end with exactly one accepted semantic attempt")
    if accepted.get("candidate_sha256") != expected_story_plan_sha256:
        raise ValueError("accepted story PE candidate hash differs from the story plan")
    accepted_api = accepted["api_provenance"]
    for key in (
        "assistant_source",
        "api_style",
        "model",
        "endpoint_sha256",
        "request_payload_sha256",
        "response_sha256",
        "assistant_text_sha256",
        "attempts",
    ):
        if provenance.get(key) != accepted_api.get(key):
            raise ValueError(f"story PE top-level receipt differs from accepted {key}")
    _validate_remote_story_pe_api_receipt(
        provenance,
        label="story PE provenance",
        require_text_hash=True,
    )
    return provenance


def _story_pe_source_brief(source: dict[str, Any]) -> dict[str, Any]:
    """Expose story semantics to PE without provider prompts or media locators."""

    brief: dict[str, Any] = {}
    for key in (
        "story_id",
        "title",
        "summary",
        "synopsis",
        "story",
        "story_outline",
        "narrative",
        "task_type",
        "mode",
        "caption_language",
        "dialogue_language",
    ):
        if key in source:
            brief[key] = copy.deepcopy(source[key])
    brief["requested_shots"] = _source_shots(source)

    metadata = source.get("metadata")
    if isinstance(metadata, dict):
        safe_metadata: dict[str, Any] = {}
        for key in (
            "summary",
            "task_type",
            "mode",
            "num_shots",
            "caption_language",
            "dialogue_language",
        ):
            if key in metadata:
                safe_metadata[key] = copy.deepcopy(metadata[key])

        def semantic_descriptor(value: Any) -> dict[str, Any] | None:
            if not isinstance(value, dict):
                return None
            result = {
                key: copy.deepcopy(value[key])
                for key in (
                    "tag",
                    "tags",
                    "name",
                    "role",
                    "description",
                    "aliases",
                    "category",
                    "entity_type",
                    "subject_type",
                    "species",
                    "is_human",
                )
                if key in value
            }
            return result or None

        for key in ("primary_character", "secondary_character", "prop", "style"):
            descriptor = semantic_descriptor(metadata.get(key))
            if descriptor is not None:
                safe_metadata[key] = descriptor
        scenes = [
            descriptor
            for descriptor in (
                semantic_descriptor(item) for item in metadata.get("scenes") or []
            )
            if descriptor is not None
        ]
        if scenes:
            safe_metadata["scenes"] = scenes
        if safe_metadata:
            brief["metadata"] = safe_metadata

    materials = source.get("materials")
    if isinstance(materials, list):
        brief["materials"] = [
            {
                key: copy.deepcopy(item[key])
                for key in (
                    "tag",
                    "tags",
                    "name",
                    "role",
                    "description",
                    "aliases",
                    "category",
                    "asset_type",
                )
                if key in item
            }
            for item in materials
            if isinstance(item, dict)
        ]

    material_library = source.get("material_library")
    if isinstance(material_library, dict):
        brief["declared_materials"] = [
            {
                key: copy.deepcopy(item[key])
                for key in (
                    "material_id",
                    "asset_type",
                    "tag",
                    "tags",
                    "name",
                    "description",
                )
                if key in item
            }
            for _, item in sorted(material_library.items())
            if isinstance(item, dict)
        ]

    upload_schedule = source.get("uploaded_assets_by_shot")
    if isinstance(upload_schedule, dict):
        declarations: list[dict[str, Any]] = []
        for schedule_key, raw_items in upload_schedule.items():
            if not isinstance(raw_items, list):
                continue
            for offset, raw in enumerate(raw_items, start=1):
                item = raw if isinstance(raw, dict) else {}
                declaration = {
                    "schedule_index": str(schedule_key),
                    "upload_id": str(
                        item.get("upload_id")
                        or f"uploaded_asset_{schedule_key}_{offset}"
                    ),
                }
                for key in (
                    "shot_index",
                    "declared_asset_type",
                    "speaker_name",
                    "user_description",
                    "description",
                    "tags",
                    "condition_eligible",
                    "memory_role",
                ):
                    if key in item:
                        declaration[key] = copy.deepcopy(item[key])
                declarations.append(declaration)
        if declarations:
            brief["uploaded_asset_declarations"] = declarations
    return brief


def _build_story_pe_request(
    *,
    source: dict[str, Any],
    expected_story_id: str,
    requested_shots: list[str],
) -> dict[str, Any]:
    """Build the untrusted data request for the offline story-planner call.

    The planner's behavior lives in the frozen story resource bundle.  This
    object carries only source data, the exact output shape, and mechanically
    checkable preservation constraints; it intentionally contains no Director
    workspace workflow or downstream generator-prompt instructions.
    """

    return {
        "instruction": (
            "Return only one complete story-plan object matching required_output. "
            "Do not repeat this request wrapper, its constraints, or the schema. "
            "In every raw_request use natural names and descriptions only; never "
            "expose material tags such as ID_A, ENTITY_1, SCENE_1, OBJECT_1, or "
            "STYLE_1. Those tags belong only in active_material_tags and prompt_state."
        ),
        "task": (
            "Convert the supplied story material into a coherent sequence of natural "
            "next-shot requests for the offline StoryMem collector."
        ),
        "required_output": {
            "schema_version": STORY_PLAN_SCHEMA_VERSION,
            "story_id": expected_story_id,
            "title": "string",
            "summary": "string",
            "caption_language": "English|Simplified Chinese",
            "dialogue_language": "English|Mandarin Chinese",
            "materials": [
                {
                    "tag": (
                        "public stable tag; humans use ID_A... and non-human "
                        "recurring entities use ENTITY_1..."
                    ),
                    "category": "person|entity|scene|object|style",
                    "description": "stable human/entity/scene/object/style facts",
                    "aliases": [],
                }
            ],
            "shots": [
                {
                    "shot_index": 1,
                    "raw_request": "one natural next-shot request",
                    "direct_continuation": False,
                    "active_material_tags": ["only tags declared in materials"],
                    "scene_state": (
                        "visible scene/time/weather/object state needed for this shot"
                    ),
                    "continuity_notes": (
                        "state transition from the prior shot, or empty"
                    ),
                }
            ],
            "prompt_state": {
                "story_state_notes": "stable story-wide facts only",
                "material_state_notes": {"declared material tag": "stable state"},
            },
        },
        "constraints": {
            "preserve_source_story": True,
            "preserve_shot_count": len(requested_shots),
            "preserve_explicit_audio_intent": True,
            "do_not_force_dialogue": True,
            "resolve_every_shot_speech_intent": (
                "Every raw_request must contain exactly one language-matched marker: "
                "English uses either Spoken dialogue: \"full line\" or No spoken "
                "dialogue.; Simplified Chinese uses either 对白：“完整台词” or 无对白。"
            ),
            "speaking_requires_active_declared_human": True,
            "voice_upload_binding": (
                "A voice_reference is a trusted user declaration for its exact "
                "speaker_name. Include that declared human's canonical material tag "
                "in active_material_tags for the scheduled shot; never infer audio content."
            ),
            "max_visible_human_participants_per_shot": 2,
            "active_person_tags_are_exactly_visible_people": (
                "For person materials, active_material_tags is exactly the set "
                "of people visible in the shot. A speaking person is visible; "
                "do not activate an off-screen person merely because the story "
                "mentions them."
            ),
            "avoid_repeated_beats": True,
            "vary_narrative_function_action_emotion": True,
            "do_not_emit_generator_prompt": True,
            "do_not_infer_media_content_from_upload_declarations": True,
            "subject_taxonomy": (
                "Only actual humans are person and use ID_A, ID_B, ... . "
                "Animals, robots, androids, creatures, mascots, and anthropomorphic "
                "non-humans are entity and use ENTITY_1, ENTITY_2, ... . Preserve "
                "source character order within each namespace; never infer personhood "
                "from a legacy <id_N> tag."
            ),
            "raw_request_visibility": (
                "Use natural entity and place names. Never put a canonical material "
                "tag, upload schedule, memory catalog, or tool instruction in raw_request."
            ),
        },
        "source_story": _story_pe_source_brief(source),
    }


def prepare_story(
    *,
    source: dict[str, Any],
    resolver: PESkillResolver,
    teacher_client: Any,
    cache_path: str | Path,
    mode: str = "always",
    temperature: float = 0.7,
    semantic_max_attempts: int = 3,
) -> PreparedStory:
    """Run story-level PE before any shot trajectory is opened.

    Canonical collection always uses a real closed-model call. Local or
    preapproved plans may be source material, but are never accepted as
    closed-Teacher trajectories.
    """
    if mode != "always":
        raise ValueError("canonical story PE mode must be always")
    if semantic_max_attempts < 1:
        raise ValueError("semantic_max_attempts must be positive")
    # Resolve legacy character taxonomy before opening a provider request.
    # Ambiguous personhood is a source-contract error, not something repeated
    # semantic calls can repair.
    _source_character_expectations(source)
    source_hash = sha256_json({key: value for key, value in source.items() if key != "_source_path"})
    destination = Path(cache_path)
    resources = resolver.story_preparation_bundle()
    resource_bundle = [
        {key: item[key] for key in ("resource_id", "revision", "sha256")}
        for item in resources
    ]
    if destination.is_file():
        cached = json.loads(destination.read_text(encoding="utf-8"))
        if (
            isinstance(cached, dict)
            and cached.get("source_sha256") == source_hash
            and cached.get("pe_manifest_sha256") == resolver.manifest_sha256
        ):
            cached_plan = validate_story_plan(cached["plan"])
            _validate_story_plan_for_director_pe(cached_plan)
            validate_story_plan_against_source(cached_plan, source)
            cached_provenance = validate_story_pe_provenance(
                cached.get("provenance"),
                expected_story_plan_sha256=sha256_json(cached_plan),
                expected_pe_manifest_sha256=resolver.manifest_sha256,
                expected_resource_bundle=resource_bundle,
            )
            return PreparedStory(cached_plan, cached_provenance)
    if mode == "always":
        system = "\n\n".join(item["content"] for item in resources)
        requested_shots = _source_shots(source)
        expected_story_id = safe_name(
            source.get("story_id") or Path(str(source["_source_path"])).stem,
            "story",
        )
        user = _build_story_pe_request(
            source=source,
            expected_story_id=expected_story_id,
            requested_shots=requested_shots,
        )
        semantic_attempts: list[dict[str, Any]] = []
        plan = None
        provenance: dict[str, Any] = {}
        validation_feedback: dict[str, Any] | None = None
        for attempt in range(1, semantic_max_attempts + 1):
            body: dict[str, Any] | None = None
            call_provenance: dict[str, Any] = {}
            visible_user = copy.deepcopy(user)
            if validation_feedback is not None:
                visible_user["private_validation_feedback"] = validation_feedback
                visible_user["instruction"] = (
                    "Return a new complete JSON object that fixes the validation error. "
                    "Do not explain, patch, or refer to the previous response."
                )
            try:
                response = teacher_client.call_json(
                    system_prompt=system,
                    user_content=[{"type": "text", "text": canonical_json(visible_user)}],
                    temperature=temperature,
                )
                body, call_provenance = _json_result(response)
                candidate = validate_story_plan(body)
                _validate_story_plan_for_director_pe(candidate)
                validate_story_plan_against_source(candidate, source)
                if candidate["story_id"] != expected_story_id:
                    raise ValueError(
                        "story PE changed story_id: expected "
                        f"{expected_story_id!r}, got {candidate['story_id']!r}"
                    )
                if len(candidate["shots"]) != len(requested_shots):
                    raise ValueError("story PE changed the requested shot count")
            except Exception as exc:  # semantic/schema failures are retried as fresh calls
                # Transport exhaustion remains an infrastructure failure; its
                # own HTTP retry policy has already run and must not be hidden
                # inside semantic retries.
                if type(exc).__name__ in {"ProviderHTTPError", "TeacherApiCallError"}:
                    raise
                error_record = {
                    "attempt": attempt,
                    "status": "rejected",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "api_provenance": copy.deepcopy(
                        getattr(exc, "provenance", {}) or call_provenance
                    ),
                }
                if isinstance(body, dict):
                    error_record["candidate_sha256"] = sha256_json(body)
                if not error_record["api_provenance"]:
                    raise RuntimeError(
                        "story PE failed before an auditable remote API response was received"
                    ) from exc
                _validate_remote_story_pe_api_receipt(
                    error_record["api_provenance"],
                    label=f"story PE semantic attempt {attempt}",
                    require_text_hash=False,
                )
                semantic_attempts.append(error_record)
                if attempt >= semantic_max_attempts:
                    raise ValueError(
                        f"story PE failed semantic validation after {attempt} attempts: {exc}"
                    ) from exc
                validation_feedback = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "attempt": attempt,
                }
                # Avoid accidentally carrying a prior call's objects into the
                # next exception record.
                continue
            semantic_attempts.append(
                {
                    "attempt": attempt,
                    "status": "accepted",
                    "candidate_sha256": sha256_json(body),
                    "api_provenance": copy.deepcopy(call_provenance),
                }
            )
            plan = candidate
            provenance = call_provenance
            break
        if plan is None:
            raise AssertionError("unreachable story PE semantic retry loop")
        provenance = {
            **provenance,
            "mode": "remote_model_story_pe",
            "semantic_attempts": semantic_attempts,
            "story_plan_sha256": sha256_json(plan),
            "pe_manifest_sha256": resolver.manifest_sha256,
            "resource_bundle": resource_bundle,
            "resource_bundle_sha256": sha256_json(resource_bundle),
        }
        provenance = validate_story_pe_provenance(
            provenance,
            expected_story_plan_sha256=sha256_json(plan),
            expected_pe_manifest_sha256=resolver.manifest_sha256,
            expected_resource_bundle=resource_bundle,
        )
    envelope = {
        "schema_version": STORY_PLAN_SCHEMA_VERSION,
        "source_sha256": source_hash,
        "pe_manifest_sha256": resolver.manifest_sha256,
        "plan": plan,
        "provenance": provenance,
        "created_at": utc_now(),
    }
    atomic_write_json(destination, envelope)
    return PreparedStory(plan, provenance)


def _normalize_declared_sha256(value: Any, *, label: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} must contain exactly 64 hexadecimal characters")
    digest = text.split(":", 1)[-1].lower()
    return "sha256:" + digest


def _resolve_local_asset(source_path: Path, locator: str) -> Path:
    path = Path(locator).expanduser()
    if path.is_absolute():
        return path.resolve()
    first = (source_path.parent / path).resolve()
    second = (source_path.parent.parent / path).resolve()
    return first if first.is_file() else second


def _source_asset_specs(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse upload locators without reading media bytes or invoking a subprocess."""

    source_path = Path(str(source.get("_source_path") or "")).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"story source is missing: {source_path}")
    schedule = source.get("uploaded_assets_by_shot")
    if schedule is None:
        schedule = {}
    if not isinstance(schedule, dict):
        raise ValueError("uploaded_assets_by_shot must be an object")
    source_shot_count = len(_source_shots(source))
    if schedule and source_shot_count < 1:
        raise ValueError(
            "uploaded_assets_by_shot cannot be mapped because the source has no shots"
        )
    parsed_schedule: list[tuple[int, Any, Any]] = []
    for key, raw_items in schedule.items():
        try:
            numeric_key = int(str(key))
        except ValueError as exc:
            raise ValueError(f"upload schedule key must be an integer: {key!r}") from exc
        if isinstance(key, bool) or numeric_key < 0 or str(numeric_key) != str(key):
            raise ValueError(
                f"upload schedule key must be a canonical non-negative integer: {key!r}"
            )
        if not isinstance(raw_items, list):
            raise ValueError(f"upload schedule entry {key!r} must be an array")
        parsed_schedule.append((numeric_key, key, raw_items))

    zero_based = any(numeric_key == 0 for numeric_key, _, _ in parsed_schedule)
    specs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for numeric_key, key, raw_items in sorted(parsed_schedule, key=lambda row: row[0]):
        scheduled_shot_index = numeric_key + 1 if zero_based else numeric_key
        if scheduled_shot_index < 1:
            raise ValueError(f"upload schedule key resolves before shot 1: {key!r}")
        if scheduled_shot_index > source_shot_count:
            raise ValueError(
                "upload schedule key resolves after the last source shot: "
                f"{key!r} -> shot {scheduled_shot_index}, source has "
                f"{source_shot_count} shot(s)"
            )
        for offset, raw in enumerate(raw_items, start=1):
            item = copy.deepcopy(raw) if isinstance(raw, dict) else {"path": str(raw)}
            explicit_shot = item.get("shot_index")
            if explicit_shot is None:
                shot_index = scheduled_shot_index
            else:
                try:
                    shot_index = int(str(explicit_shot))
                except ValueError as exc:
                    raise ValueError("uploaded asset shot_index must be an integer") from exc
                if (
                    isinstance(explicit_shot, bool)
                    or str(shot_index) != str(explicit_shot)
                ):
                    raise ValueError(
                        "uploaded asset shot_index must be a canonical integer"
                    )
            if shot_index < 1:
                raise ValueError("uploaded asset shot_index must be positive")
            if shot_index > source_shot_count:
                raise ValueError(
                    "uploaded asset shot_index exceeds the source shot count: "
                    f"{shot_index} > {source_shot_count}"
                )
            asset_id = safe_name(
                item.get("upload_id") or f"upload_s{shot_index:03d}_{offset:03d}",
                "upload",
            )
            if asset_id in seen_ids:
                raise ValueError(f"duplicate upload_id: {asset_id}")
            seen_ids.add(asset_id)

            locator_values = [
                (field, item.get(field))
                for field in _ASSET_LOCATOR_FIELDS
                if item.get(field) not in (None, "")
            ]
            if len(locator_values) != 1:
                raise ValueError(
                    f"{asset_id} must declare exactly one asset locator field"
                )
            locator_field, locator_value = locator_values[0]
            if not isinstance(locator_value, (str, Path)):
                raise ValueError(f"{asset_id}.{locator_field} must be a path or URL string")
            locator = str(locator_value)
            if not locator or locator != locator.strip():
                raise ValueError(f"{asset_id}.{locator_field} must be a trimmed non-empty string")
            parsed = urlparse(locator)
            is_url = parsed.scheme.lower() in {"http", "https"}
            if is_url:
                if not parsed.netloc:
                    raise ValueError(f"{asset_id} URL must be absolute")
                if parsed.username or parsed.password:
                    raise ValueError(f"{asset_id} URL must not contain inline credentials")
            elif parsed.scheme or parsed.netloc:
                raise ValueError(f"{asset_id} locator must be a local path or HTTP(S) URL")
            expected_sha256 = _normalize_declared_sha256(
                item.get("sha256"), label=f"{asset_id}.sha256"
            )
            if is_url and expected_sha256 is None:
                raise ValueError(
                    f"{asset_id} URL upload requires a declared sha256 content hash"
                )
            resolved_path = None if is_url else _resolve_local_asset(source_path, locator)
            if resolved_path is not None and not resolved_path.is_file():
                raise FileNotFoundError(
                    f"uploaded asset is missing: {locator!r} from {source_path}"
                )
            requested_condition = item.get("condition_eligible", False)
            if not isinstance(requested_condition, bool):
                raise ValueError(f"{asset_id}.condition_eligible must be boolean")
            declared_tags = item.get("tags") or []
            if not isinstance(declared_tags, list) or any(
                not isinstance(tag, str) for tag in declared_tags
            ):
                raise ValueError(f"{asset_id}.tags must be a string array")
            declared_asset_type = str(
                item.get("declared_asset_type") or "reference"
            ).strip().lower()
            is_voice_reference = declared_asset_type == "voice_reference"
            speaker_name = item.get("speaker_name")
            if locator_field == "audio_path" and not is_voice_reference:
                raise ValueError(
                    f"{asset_id}.audio_path requires declared_asset_type=voice_reference"
                )
            if is_voice_reference:
                if locator_field in {"image_path", "video_path", "document_path"}:
                    raise ValueError(
                        f"{asset_id} voice_reference must use audio_path, path, or url"
                    )
                if (
                    not isinstance(speaker_name, str)
                    or not speaker_name
                    or speaker_name != speaker_name.strip()
                ):
                    raise ValueError(
                        f"{asset_id}.speaker_name must be a trimmed non-empty string"
                    )
                if item.get("memory_role") != "voice":
                    raise ValueError(
                        f"{asset_id} voice_reference requires memory_role=voice"
                    )
                if requested_condition:
                    raise ValueError(
                        f"{asset_id} voice_reference cannot be condition_eligible"
                    )
                speaker_declaration = _resolve_source_speaker_declaration(
                    source, speaker_name
                )
            else:
                if speaker_name is not None:
                    raise ValueError(
                        f"{asset_id}.speaker_name is allowed only for voice_reference"
                    )
                speaker_declaration = None
            specs.append(
                {
                    "asset_id": asset_id,
                    "shot_index": shot_index,
                    "item": item,
                    "locator": locator,
                    "locator_sha256": sha256_text(locator),
                    "is_url": is_url,
                    "resolved_path": resolved_path,
                    "expected_sha256": expected_sha256,
                    "declared_asset_type": declared_asset_type,
                    "speaker_declaration": speaker_declaration,
                }
            )
    return specs


def validate_source_asset_locators(source: dict[str, Any]) -> dict[str, Any]:
    """Statically validate upload schema and local existence.

    This function deliberately does not hash/open media, create a directory,
    download a URL, or run ffprobe/pdfinfo.  It is therefore safe for the CLI's
    ``--validate-only`` path and for large launch manifests.
    """

    specs = _source_asset_specs(source)
    locator_counts = {
        "local": sum(not bool(item["is_url"]) for item in specs),
        "url": sum(bool(item["is_url"]) for item in specs),
    }
    return {
        "asset_count": len(specs),
        "scheduled_shot_count": len({int(item["shot_index"]) for item in specs}),
        "locator_type_counts": locator_counts,
    }


def _snapshot_source_asset(path: Path, cache_root: Path, content_sha256: str) -> Path:
    """Copy immutable source bytes into the story-local content-addressed store."""

    suffix = path.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,12}", suffix):
        suffix = ".bin"
    destination = cache_root / "objects" / (
        content_sha256.removeprefix("sha256:") + suffix
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) != content_sha256:
            raise RuntimeError(f"content-addressed asset is corrupt: {destination}")
        return destination.resolve()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, path.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if sha256_file(temporary) != content_sha256:
            raise RuntimeError("source asset changed while it was being snapshotted")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def ingest_source_assets(
    source: dict[str, Any],
    *,
    cache_dir: str | Path,
    download_session: Any | None = None,
) -> tuple[dict[str, Any], dict[int, list[str]]]:
    """Lazily ingest uploads into one story's output-local immutable cache."""

    from .media import download_url, media_manifest, selector_catalog

    specs = _source_asset_specs(source)
    cache_root = Path(cache_dir).expanduser().resolve()
    assets: dict[str, Any] = {}
    uploads_by_shot: dict[int, list[str]] = {}
    for spec in specs:
        item = spec["item"]
        asset_id = str(spec["asset_id"])
        expected_sha256 = spec["expected_sha256"]
        if spec["is_url"]:
            downloaded = download_url(
                str(spec["locator"]),
                cache_root / "url_cache",
                expected_sha256=expected_sha256,
                session=download_session,
            )
            original_path = Path(downloaded["path"])
            source_provenance = {
                key: copy.deepcopy(value)
                for key, value in dict(downloaded["provenance"]).items()
                if key != "path"
            }
        else:
            original_path = Path(spec["resolved_path"])
            source_provenance = {
                "transport": "local_file_snapshot",
                "source_locator_sha256": spec["locator_sha256"],
            }
        content_sha256 = sha256_file(original_path)
        if expected_sha256 is not None and content_sha256 != expected_sha256:
            raise ValueError(f"{asset_id} does not match its declared sha256")
        snapshot_path = _snapshot_source_asset(
            original_path, cache_root, content_sha256
        )
        source_provenance.update(
            {
                "source_locator_sha256": spec["locator_sha256"],
                "declared_sha256": expected_sha256,
                "content_sha256": content_sha256,
            }
        )
        representation_id = f"repr_{asset_id}_original"
        manifest = _plain_media_manifest(
            media_manifest(snapshot_path, representation_id=representation_id)
        )
        selector_info = selector_catalog(snapshot_path)
        declared_asset_type = str(spec["declared_asset_type"])
        is_audio = manifest.get("media_type") == "audio"
        is_voice_reference = declared_asset_type == "voice_reference"
        if is_audio != is_voice_reference:
            raise ValueError(
                f"{asset_id} actual audio MIME and declared_asset_type=voice_reference "
                "must agree in both directions"
            )
        explicitly_condition_typed = declared_asset_type in {
            "interaction_start_frame",
            "full_frame_reference",
            "first_frame",
            "condition_frame",
        }
        requested_condition = item.get("condition_eligible", False)
        condition_eligible = bool(requested_condition and explicitly_condition_typed)
        normalization_reason = (
            "accepted_explicit_complete_frame_type"
            if condition_eligible
            else "forced_false_non_frame_reference_type"
            if requested_condition and not explicitly_condition_typed
            else "not_requested"
        )
        declared_tags = item.get("tags") or []
        assets[asset_id] = {
            "asset_id": asset_id,
            "media_type": manifest.get("media_type"),
            "default_representation_id": representation_id,
            "representations": {representation_id: manifest},
            "user_description": str(
                item.get("user_description") or item.get("description") or ""
            ),
            "declared_tags": list(declared_tags),
            "declared_asset_type": declared_asset_type,
            "condition_eligible": condition_eligible,
            "memory_role": str(item.get("memory_role") or "reference"),
            "selectors": selector_info["selectors"],
            "selector_limits": selector_info["limits"],
            "selector_cache_dir": str(
                (cache_root / "selectors" / asset_id).resolve()
            ),
            "source_provenance": source_provenance,
            "normalization_provenance": {
                "condition_eligible_requested": requested_condition,
                "condition_eligible_effective": condition_eligible,
                "reason": normalization_reason,
            },
        }
        if is_voice_reference:
            assets[asset_id]["speaker_name"] = str(item["speaker_name"])
        uploads_by_shot.setdefault(int(spec["shot_index"]), []).append(asset_id)
    return assets, uploads_by_shot


def build_source_asset_receipt(
    *,
    source: dict[str, Any],
    assets: dict[str, Any],
    uploads_by_shot: dict[int, list[str]],
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Build a deterministic closure over source bytes and any voice binding.

    Image-only stories retain the byte-compatible v1 receipt.  A story with an
    audio upload uses v2 so the trusted speaker declaration and canonical
    subject binding cannot be separated from the bytes later.
    """

    cache_root = Path(cache_dir).expanduser().resolve()
    has_audio = any(
        isinstance(asset, Mapping)
        and any(
            isinstance(representation, Mapping)
            and representation.get("media_type") == "audio"
            for representation in (asset.get("representations") or {}).values()
        )
        for asset in assets.values()
    )
    schema_version = (
        SOURCE_ASSET_RECEIPT_AUDIO_SCHEMA_VERSION
        if has_audio
        else SOURCE_ASSET_RECEIPT_SCHEMA_VERSION
    )
    scheduled: dict[str, list[int]] = {}
    for shot_index, asset_ids in uploads_by_shot.items():
        for asset_id in asset_ids:
            scheduled.setdefault(str(asset_id), []).append(int(shot_index))
    rows: list[dict[str, Any]] = []
    for asset_id, asset in sorted(assets.items()):
        representation_id = str(asset.get("default_representation_id") or "")
        representation = (asset.get("representations") or {}).get(representation_id)
        if not isinstance(representation, dict):
            raise ValueError(f"{asset_id} has no default representation")
        object_path = Path(str(representation.get("path") or "")).resolve()
        try:
            object_relpath = str(object_path.relative_to(cache_root))
        except ValueError as exc:
            raise RuntimeError(
                f"source asset escaped its story-local cache: {object_path}"
            ) from exc
        provenance = asset.get("source_provenance") or {}
        if representation.get("media_type") == "audio":
            binding = asset.get("speaker_binding")
            if not isinstance(binding, Mapping):
                raise ValueError(
                    f"audio source asset {asset_id!r} has no speaker binding"
                )
            if list(asset.get("subject_ids") or []) != [
                str(binding.get("subject_id") or "")
            ]:
                raise ValueError(
                    f"audio source asset {asset_id!r} subject_ids do not match its speaker binding"
                )
            if str(asset.get("speaker_name") or "") != str(
                binding.get("declared_name") or ""
            ):
                raise ValueError(
                    f"audio source asset {asset_id!r} declared speaker changed after binding"
                )
        row = {
            "asset_id": asset_id,
            "scheduled_shots": sorted(scheduled.get(asset_id, [])),
            "media_type": representation.get("media_type"),
            "mime_type": representation.get("mime_type"),
            "size_bytes": representation.get("size_bytes"),
            "content_sha256": representation.get("sha256"),
            "object_relpath": object_relpath,
            "source_transport": provenance.get("transport"),
            "source_locator_sha256": provenance.get("source_locator_sha256"),
            "declared_sha256": provenance.get("declared_sha256"),
        }
        if schema_version == SOURCE_ASSET_RECEIPT_AUDIO_SCHEMA_VERSION:
            row.update(
                {
                    "declared_asset_type": asset.get("declared_asset_type"),
                    "speaker_binding": copy.deepcopy(asset.get("speaker_binding")),
                }
            )
        rows.append(row)
    body = {
        "schema_version": schema_version,
        "source_sha256": sha256_json(
            {key: value for key, value in source.items() if key != "_source_path"}
        ),
        "assets": rows,
    }
    return validate_source_asset_receipt(
        {**body, "receipt_sha256": sha256_json(body)}
    )


def validate_source_asset_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_source_sha256: str | None = None,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the locator-free, deterministic source-asset receipt.

    V1 intentionally carries only content-addressed object paths and hashes of
    original locators.  V2 additionally seals the explicit speaker binding for
    audio.  Requiring exact field sets prevents local paths, signed URLs, or
    untracked semantic claims from entering canonical state and trajectories.
    """

    if not isinstance(receipt, Mapping):
        raise ValueError("source asset receipt must be an object")
    value = copy.deepcopy(dict(receipt))
    if set(value) != _SOURCE_ASSET_RECEIPT_FIELDS:
        raise ValueError("source asset receipt has an incompatible field set")
    schema_version = value.get("schema_version")
    if schema_version not in {
        SOURCE_ASSET_RECEIPT_SCHEMA_VERSION,
        SOURCE_ASSET_RECEIPT_AUDIO_SCHEMA_VERSION,
    }:
        raise ValueError("source asset receipt schema_version is incompatible")
    source_sha256 = value.get("source_sha256")
    if not isinstance(source_sha256, str) or not _STRICT_SHA256_RE.fullmatch(
        source_sha256
    ):
        raise ValueError("source asset receipt source_sha256 is invalid")
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError("source asset receipt is bound to a different source")

    assets = value.get("assets")
    if not isinstance(assets, list):
        raise ValueError("source asset receipt assets must be an array")
    previous_asset_id: str | None = None
    for index, raw_row in enumerate(assets):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"source asset receipt row {index} must be an object")
        row = dict(raw_row)
        expected_row_fields = (
            _SOURCE_ASSET_RECEIPT_V2_ROW_FIELDS
            if schema_version == SOURCE_ASSET_RECEIPT_AUDIO_SCHEMA_VERSION
            else _SOURCE_ASSET_RECEIPT_ROW_FIELDS
        )
        if set(row) != expected_row_fields:
            raise ValueError(
                f"source asset receipt row {index} has an incompatible field set"
            )
        asset_id = row.get("asset_id")
        if (
            not isinstance(asset_id, str)
            or not asset_id
            or safe_name(asset_id, "") != asset_id
        ):
            raise ValueError(f"source asset receipt row {index} has an invalid asset_id")
        if previous_asset_id is not None and asset_id <= previous_asset_id:
            raise ValueError("source asset receipt assets must be uniquely sorted by asset_id")
        previous_asset_id = asset_id

        scheduled_shots = row.get("scheduled_shots")
        if (
            not isinstance(scheduled_shots, list)
            or not scheduled_shots
            or any(type(item) is not int or item < 1 for item in scheduled_shots)
            or scheduled_shots != sorted(set(scheduled_shots))
        ):
            raise ValueError(
                f"source asset receipt row {index} has invalid scheduled_shots"
            )
        media_type = row.get("media_type")
        mime_type = row.get("mime_type")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError(f"source asset receipt row {index} has invalid media_type")
        if (
            not isinstance(mime_type, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*", mime_type)
        ):
            raise ValueError(f"source asset receipt row {index} has invalid mime_type")
        if media_type == "audio" and not mime_type.startswith("audio/"):
            raise ValueError(
                f"source asset receipt row {index} audio media_type has a non-audio MIME"
            )
        size_bytes = row.get("size_bytes")
        if type(size_bytes) is not int or size_bytes < 1:
            raise ValueError(f"source asset receipt row {index} has invalid size_bytes")
        if schema_version == SOURCE_ASSET_RECEIPT_SCHEMA_VERSION:
            if media_type == "audio":
                raise ValueError(
                    "audio source assets require the v2 receipt with explicit speaker binding"
                )
        else:
            declared_asset_type = row.get("declared_asset_type")
            speaker_binding = row.get("speaker_binding")
            if (
                not isinstance(declared_asset_type, str)
                or not declared_asset_type
                or declared_asset_type != declared_asset_type.strip().lower()
            ):
                raise ValueError(
                    f"source asset receipt row {index} has invalid declared_asset_type"
                )
            if (media_type == "audio") != (
                declared_asset_type == "voice_reference"
            ):
                raise ValueError(
                    f"source asset receipt row {index} audio MIME and voice_reference type disagree"
                )
            if media_type == "audio":
                if not isinstance(speaker_binding, Mapping) or set(
                    speaker_binding
                ) != _SPEAKER_BINDING_FIELDS:
                    raise ValueError(
                        f"source asset receipt row {index} has invalid speaker_binding"
                    )
                if speaker_binding.get("basis") != "trusted_user_declared":
                    raise ValueError(
                        f"source asset receipt row {index} has invalid speaker basis"
                    )
                declared_name = speaker_binding.get("declared_name")
                if (
                    not isinstance(declared_name, str)
                    or not declared_name
                    or declared_name != declared_name.strip()
                ):
                    raise ValueError(
                        f"source asset receipt row {index} has invalid declared speaker name"
                    )
                if not _PERSON_TAG_RE.fullmatch(
                    str(speaker_binding.get("subject_id") or "")
                ):
                    raise ValueError(
                        f"source asset receipt row {index} has invalid speaker subject_id"
                    )
                if speaker_binding.get("matching_method") != "exact_normalized_name":
                    raise ValueError(
                        f"source asset receipt row {index} has invalid speaker matching method"
                    )
            elif speaker_binding is not None:
                raise ValueError(
                    f"source asset receipt row {index} non-audio asset cannot bind a speaker"
                )
        content_sha256 = row.get("content_sha256")
        locator_sha256 = row.get("source_locator_sha256")
        declared_sha256 = row.get("declared_sha256")
        if not isinstance(content_sha256, str) or not _STRICT_SHA256_RE.fullmatch(
            content_sha256
        ):
            raise ValueError(
                f"source asset receipt row {index} has invalid content_sha256"
            )
        if not isinstance(locator_sha256, str) or not _STRICT_SHA256_RE.fullmatch(
            locator_sha256
        ):
            raise ValueError(
                f"source asset receipt row {index} has invalid source_locator_sha256"
            )
        if declared_sha256 is not None and (
            not isinstance(declared_sha256, str)
            or not _STRICT_SHA256_RE.fullmatch(declared_sha256)
        ):
            raise ValueError(
                f"source asset receipt row {index} has invalid declared_sha256"
            )
        object_relpath = row.get("object_relpath")
        object_match = (
            _SOURCE_ASSET_OBJECT_RE.fullmatch(object_relpath)
            if isinstance(object_relpath, str)
            else None
        )
        if (
            object_match is None
            or object_match.group("digest") != content_sha256.removeprefix("sha256:")
        ):
            raise ValueError(
                f"source asset receipt row {index} has an invalid content-addressed object_relpath"
            )
        if row.get("source_transport") not in {
            "local_file_snapshot",
            "http_download",
            "https_download",
        }:
            raise ValueError(
                f"source asset receipt row {index} has invalid source_transport"
            )

    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    receipt_sha256 = value.get("receipt_sha256")
    if (
        not isinstance(receipt_sha256, str)
        or not _STRICT_SHA256_RE.fullmatch(receipt_sha256)
        or receipt_sha256 != sha256_json(body)
    ):
        raise ValueError("source asset receipt integrity check failed")
    if (
        expected_receipt_sha256 is not None
        and receipt_sha256 != expected_receipt_sha256
    ):
        raise ValueError("source asset receipt hash differs from its run manifest")
    return value


def persist_source_asset_receipt(
    path: str | Path, receipt: dict[str, Any]
) -> dict[str, Any]:
    """Create an immutable receipt, or verify that an existing one is identical."""

    destination = Path(path)
    receipt = validate_source_asset_receipt(receipt)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != receipt:
            raise RuntimeError(
                "source asset bytes or locator closure changed; refusing to overwrite receipt"
            )
        return existing
    atomic_write_json(destination, receipt)
    return receipt


def bind_assets_to_story_plan(
    *,
    source: dict[str, Any],
    story_plan: dict[str, Any],
    assets: dict[str, Any],
) -> dict[str, Any]:
    """Bind source-local upload tags to canonical story-plan material tags.

    Existing material-library sources use placeholders such as ``<id_1>`` and
    ``<scene_1>``, while story PE intentionally exposes stable public tags such
    as ``ID_A``.  This deterministic post-PE adapter keeps the original tags as
    provenance, maps them through descriptions/aliases and typed ordinals, and
    adds the exact subject IDs required by the Echo memory-slot contract.  It
    never inspects media or invokes a model.
    """

    materials = [
        copy.deepcopy(item)
        for item in story_plan.get("materials") or []
        if isinstance(item, dict) and isinstance(item.get("tag"), str)
    ]
    by_tag = {str(item["tag"]): item for item in materials}

    source_descriptors: dict[str, dict[str, Any]] = {}

    def remember(
        raw_tag: Any,
        *,
        category: str | None,
        texts: list[Any],
    ) -> None:
        tag = str(raw_tag or "").strip()
        if not tag:
            return
        row = source_descriptors.setdefault(tag, {"categories": [], "texts": []})
        if category and category not in row["categories"]:
            row["categories"].append(category)
        for value in texts:
            text = str(value or "").strip()
            if text and text not in row["texts"]:
                row["texts"].append(text)

    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    for key in ("primary_character", "secondary_character"):
        item = metadata.get(key)
        if isinstance(item, dict):
            remember(
                item.get("tag"),
                category=_declared_character_category(
                    item, label=f"metadata.{key}"
                ),
                texts=[item.get("name"), item.get("role"), item.get("description")],
            )
    prop = metadata.get("prop")
    if isinstance(prop, dict):
        remember(
            prop.get("tag"),
            category="object",
            texts=[prop.get("name"), prop.get("role"), prop.get("description")],
        )
    style = metadata.get("style")
    if isinstance(style, dict):
        remember(
            style.get("tag"),
            category="style",
            texts=[style.get("name"), style.get("description")],
        )
    for scene in metadata.get("scenes") or []:
        if isinstance(scene, dict):
            remember(
                scene.get("tag"),
                category="scene",
                texts=[scene.get("name"), scene.get("description")],
            )

    asset_type_categories = {
        "scene_plate": "scene",
        "object_closeup": "object",
        "prop_reference": "object",
        "style_reference": "style",
        # A character reference can depict either a human or a non-human.
        # Its tag's structured source declaration supplies the taxonomy.
        "character_reference": None,
    }
    material_library = source.get("material_library")
    if isinstance(material_library, dict):
        for item in material_library.values():
            if not isinstance(item, dict):
                continue
            tags = list(item.get("tags") or [])
            if item.get("tag") and item.get("tag") not in tags:
                tags.append(item["tag"])
            category = asset_type_categories.get(
                str(item.get("asset_type") or "").strip().lower()
            )
            for tag in tags:
                remember(
                    tag,
                    category=category,
                    texts=[item.get("name"), item.get("description")],
                )

    for asset in assets.values():
        if not isinstance(asset, dict):
            continue
        category = asset_type_categories.get(
            str(asset.get("declared_asset_type") or "").strip().lower()
        )
        for tag in asset.get("declared_tags") or []:
            remember(
                tag,
                category=category,
                texts=[asset.get("user_description")],
            )

    def tokens(values: list[Any]) -> set[str]:
        text = " ".join(str(value or "").casefold() for value in values)
        return set(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text))

    def normalized_phrase(value: Any) -> str:
        return " ".join(
            re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", str(value or "").casefold())
        )

    plan_tokens = {
        str(item["tag"]): tokens(
            [item.get("description"), *(item.get("aliases") or [])]
        )
        for item in materials
    }
    plan_phrases = {
        str(item["tag"]): {
            phrase
            for phrase in (
                normalized_phrase(value)
                for value in [item.get("description"), *(item.get("aliases") or [])]
            )
            if phrase
        }
        for item in materials
    }
    explicit_ordinal_binding = bool(
        source.get("material_tag_order_preserved") is True
        or metadata.get("material_tag_order_preserved") is True
        or (
            isinstance(source.get("story_pe_receipt"), dict)
            and source["story_pe_receipt"].get("material_tag_order_preserved") is True
        )
    )
    resolved_cache: dict[str, tuple[str | None, str]] = {}

    def resolve(raw_tag: str) -> tuple[str | None, str]:
        if raw_tag in resolved_cache:
            return resolved_cache[raw_tag]
        if raw_tag in by_tag:
            result = (raw_tag, "already_canonical")
            resolved_cache[raw_tag] = result
            return result
        descriptor = source_descriptors.get(raw_tag) or {"categories": [], "texts": []}
        declared_categories = set(descriptor.get("categories") or [])
        if len(declared_categories) > 1:
            raise ValueError(
                f"legacy material tag {raw_tag!r} has conflicting source categories"
            )
        legacy_match = _LEGACY_MATERIAL_TAG_RE.fullmatch(raw_tag)
        if legacy_match is not None:
            kind = legacy_match.group("kind").casefold()
            if kind in {"id", "entity"}:
                legacy_category = next(iter(declared_categories), None)
                if legacy_category is not None and legacy_category not in {"person", "entity"}:
                    raise ValueError(
                        f"legacy character tag {raw_tag!r} has invalid source category {legacy_category!r}"
                    )
            else:
                legacy_category = {
                    "scene": "scene",
                    "obj": "object",
                    "object": "object",
                    "prop": "object",
                    "style": "style",
                }[kind]
                if declared_categories and declared_categories != {legacy_category}:
                    raise ValueError(
                        f"legacy material tag {raw_tag!r} has conflicting source categories"
                    )
        else:
            legacy_category = next(iter(declared_categories), None)

        candidates = [
            item
            for item in materials
            if (
                item.get("category") in {"person", "entity"}
                if legacy_match is not None
                and legacy_match.group("kind").casefold() in {"id", "entity"}
                and legacy_category is None
                else legacy_category is None or item.get("category") == legacy_category
            )
        ]
        # For a natural user label, match the label itself against an exact
        # canonical tag/alias.  Do not let the containing asset's shared
        # description turn every arbitrary label into the same material.
        descriptor_texts = (
            descriptor.get("texts", []) if legacy_match is not None else []
        )
        source_phrases = {
            normalized_phrase(value)
            for value in [raw_tag, *descriptor_texts]
            if normalized_phrase(value)
        }
        exact = [
            (
                len(source_phrases & plan_phrases[str(item["tag"])]),
                str(item["tag"]),
            )
            for item in candidates
            if source_phrases & plan_phrases[str(item["tag"])]
        ]
        if exact:
            exact.sort(reverse=True)
            best_exact_count = exact[0][0]
            best_exact = [tag for count, tag in exact if count == best_exact_count]
            if len(best_exact) > 1:
                raise ValueError(
                    f"ambiguous exact material binding for {raw_tag!r}: {sorted(best_exact)}"
                )
            result = (best_exact[0], "exact_descriptor_match")
            resolved_cache[raw_tag] = result
            return result

        source_tokens = tokens([raw_tag, *descriptor_texts])
        ranked: list[tuple[float, str]] = []
        for item in candidates:
            candidate_tag = str(item["tag"])
            candidate_tokens = plan_tokens[candidate_tag]
            overlap = len(source_tokens & candidate_tokens)
            if overlap:
                score = overlap / max(1.0, (len(source_tokens) * len(candidate_tokens)) ** 0.5)
                ranked.append((score, candidate_tag))
        if ranked:
            ranked.sort(reverse=True)
            best_score, best_tag = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            if best_score >= 0.55:
                if len(ranked) > 1 and best_score - second_score < 0.12:
                    tied = [tag for score, tag in ranked if best_score - score < 0.12]
                    raise ValueError(
                        f"ambiguous semantic material binding for {raw_tag!r}: {sorted(tied)}"
                    )
                result = (best_tag, "strong_descriptor_match")
                resolved_cache[raw_tag] = result
                return result

        if legacy_match is not None and explicit_ordinal_binding:
            ordinal = int(legacy_match.group("index")) - 1
            if 0 <= ordinal < len(candidates):
                result = (
                    str(candidates[ordinal]["tag"]),
                    "explicit_typed_ordinal_fallback",
                )
                resolved_cache[raw_tag] = result
                return result
        result = (None, "unresolved_preserved")
        resolved_cache[raw_tag] = result
        return result

    has_voice_reference = any(
        isinstance(asset, Mapping)
        and str(asset.get("declared_asset_type") or "") == "voice_reference"
        for asset in assets.values()
    )
    voice_specs = (
        {
            str(spec["asset_id"]): spec
            for spec in _source_asset_specs(source)
            if spec.get("declared_asset_type") == "voice_reference"
        }
        if has_voice_reference
        else {}
    )
    plan_shots = {
        int(item.get("shot_index", -1)): item
        for item in story_plan.get("shots") or []
        if isinstance(item, Mapping)
    }
    bound = copy.deepcopy(assets)
    for asset_id, asset in bound.items():
        original_tags = list(asset.get("declared_tags") or [])
        canonical_tags: list[str] = []
        unbound_tags: list[str] = []
        bindings: list[dict[str, Any]] = []
        for raw_tag in original_tags:
            canonical_tag, method = resolve(str(raw_tag))
            if canonical_tag is None and _LEGACY_MATERIAL_TAG_RE.fullmatch(str(raw_tag)):
                raise ValueError(
                    f"could not bind legacy material tag {raw_tag!r} to the prepared story plan"
                )
            bindings.append(
                {
                    "source_tag": str(raw_tag),
                    "canonical_tag": canonical_tag,
                    "method": method,
                }
            )
            if canonical_tag and canonical_tag not in canonical_tags:
                canonical_tags.append(canonical_tag)
            elif canonical_tag is None and str(raw_tag) not in unbound_tags:
                # Arbitrary user labels remain useful discovery metadata even
                # when they are not canonical story materials.
                unbound_tags.append(str(raw_tag))
        subject_ids = [
            tag
            for tag in canonical_tags
            if by_tag[tag].get("category") == "person"
        ]
        if str(asset.get("declared_asset_type") or "") == "voice_reference":
            spec = voice_specs.get(str(asset_id))
            if not isinstance(spec, Mapping):
                raise ValueError(
                    f"voice_reference {asset_id!r} has no validated source declaration"
                )
            speaker = spec.get("speaker_declaration")
            if not isinstance(speaker, Mapping):
                raise ValueError(
                    f"voice_reference {asset_id!r} has no resolved speaker declaration"
                )
            subject_id = str(speaker.get("canonical_tag") or "")
            material = by_tag.get(subject_id)
            if not material or material.get("category") != "person":
                raise ValueError(
                    f"voice_reference {asset_id!r} speaker {subject_id!r} is not a "
                    "canonical person in the prepared story plan"
                )
            if unbound_tags or any(tag != subject_id for tag in canonical_tags):
                raise ValueError(
                    f"voice_reference {asset_id!r} tags conflict with its exact "
                    f"speaker binding {subject_id!r}"
                )
            shot_index = int(spec["shot_index"])
            shot = plan_shots.get(shot_index)
            active_tags = set(shot.get("active_material_tags") or []) if shot else set()
            if subject_id not in active_tags:
                raise ValueError(
                    f"voice_reference {asset_id!r} speaker {subject_id!r} is not active "
                    f"in scheduled shot {shot_index}"
                )
            declared_name = str(asset.get("speaker_name") or "")
            canonical_tags = [subject_id]
            unbound_tags = []
            subject_ids = [subject_id]
            asset["speaker_binding"] = {
                "basis": "trusted_user_declared",
                "declared_name": declared_name,
                "subject_id": subject_id,
                "matching_method": "exact_normalized_name",
            }
            asset["transport_eligible"] = True
            asset["audio_quality_status"] = "unassessed"
            asset["transport_description"] = (
                str(asset.get("user_description") or "").strip()
                or f"User-declared voice reference for {declared_name}."
            )
        asset["source_declared_tags"] = original_tags
        asset["declared_tags"] = canonical_tags + unbound_tags
        asset["canonical_material_tags"] = canonical_tags
        asset["subject_ids"] = subject_ids
        asset["tag_binding_provenance"] = bindings
    return bound


def _plain_media_manifest(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError("unsupported media manifest value")
