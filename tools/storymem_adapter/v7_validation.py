"""Deterministic admission checks for the official Director v7 shot format."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


V7_VALIDATOR_ID = "director.v7_cinematic_full.admission.v8.source-cast"
V7_SEMANTIC_CONTRACT_VERSION = "director.v7.raw-intent-preservation.v6"

_SEMANTIC_CATEGORIES = (
    "subject",
    "event",
    "scene",
    "dialogue",
    "style",
    "state_changes",
)

# These words describe the v7 serialization rather than the requested story
# world.  Excluding them keeps the deterministic guard focused on facts that PE
# is forbidden to add, delete, or replace.
_NON_SEMANTIC_ENGLISH = frozenset(
    """
    a an and are as at be been being by chinese continuous described english for free
    from has have he her hers him his i in into is it its of one our ours she
    on or that the their them then this through to under with while who whose
    current previous same video shot shots frame frames camera framing angle
    composition cinematic closeup close up wide medium macro detail focus
    focused focusing slow gentle controlled locked lock off reveal dolly crane
    tracking track lateral overhead eye level viewpoint view shows showing establish
    depict depicts featuring feature involving include includes keep maintain
    preserving preserve exact precise directly natural naturally normal speed
    character characters identity me named people person prior silent take wardrobe wordless
    prop design scene layout lighting subtitles their theirs they them we you your yours
    subtitle captions caption ui overlays overlay watermarks watermark logos
    logo readable text ocr requested request requirement required style
    continue no not never without
    """.split()
)

_SEMANTIC_SYNONYMS = {
    "awning": "awning",
    "awnings": "awning",
    "arrives": "arrive",
    "arrived": "arrive",
    "arriving": "arrive",
    "carried": "carry",
    "carries": "carry",
    "carrying": "carry",
    "closed": "close",
    "closes": "close",
    "closing": "close",
    "continues": "continue",
    "crossed": "cross",
    "crosses": "cross",
    "crossing": "cross",
    "continued": "continue",
    "continuing": "continue",
    "discovers": "discover",
    "discovered": "discover",
    "discovering": "discover",
    "changed": "change",
    "changes": "change",
    "changing": "change",
    "enters": "enter",
    "entered": "enter",
    "entering": "enter",
    "examines": "inspect",
    "examined": "inspect",
    "examining": "inspect",
    "finds": "discover",
    "finding": "discover",
    "holds": "carry",
    "holding": "carry",
    "inspects": "inspect",
    "inspected": "inspect",
    "inspecting": "inspect",
    "looks": "look",
    "looking": "look",
    "walks": "move",
    "walked": "move",
    "walking": "move",
    "steps": "move",
    "stepped": "move",
    "stepping": "move",
    "moves": "move",
    "moving": "move",
    "motion": "move",
    "notices": "discover",
    "noticing": "discover",
    "opened": "open",
    "opens": "open",
    "opening": "open",
    "remained": "stable",
    "remaining": "stable",
    "remains": "stable",
    "shifts": "shift",
    "shifted": "shift",
    "shifting": "shift",
    "speaks": "speak",
    "speaking": "speak",
    "said": "speak",
    "saying": "speak",
    "says": "speak",
    "includes": "include",
    "included": "include",
    "including": "include",
    "preserves": "preserve",
    "preserved": "preserve",
    "preserving": "preserve",
    "presents": "present",
    "presented": "present",
    "presenting": "present",
    "places": "place",
    "placed": "place",
    "placing": "place",
    "edges": "edge",
    "fog": "mist",
    "foggy": "mist",
    "misty": "mist",
    "ropes": "rope",
    "waves": "wave",
    "silhouettes": "silhouette",
    "textured": "texture",
    "textures": "texture",
    "studies": "inspect",
    "studied": "inspect",
    "studying": "inspect",
    "unchanged": "stable",
}

_SCENE_TERMS = frozenset(
    """
    alley archive attic beach bridge cafe castle city corridor courtyard desert
    dock forest garden greenhouse harbor hospital house kitchen laboratory market
    mountain observatory ocean office palace pier platform restaurant river room
    rooftop shop shore station street studio subway temple train village warehouse
    workshop
    """.split()
)
_STYLE_TERMS = frozenset(
    """
    anime cartoon claymation cyberpunk documentary dreamlike dreamy fantasy film
    historical ink wash monochrome naturalism noir painterly photorealistic realism
    realistic retro storybook surreal watercolor
    """.split()
)

# High-confidence world facts which must never be invented by PE.  The v7
# format necessarily adds cinematography prose, so this list intentionally
# excludes camera/lighting vocabulary and focuses on entities and irreversible
# story actions whose appearance is unambiguously a semantic addition.
_SUBJECT_ENTITY_TERMS = frozenset(
    """
    alien animal bear bird cat creature dog dragon horse monster robot soldier
    vampire wolf wizard
    """.split()
)
_SUBJECT_ATTRIBUTE_CARRIERS = frozenset({"appearance", "build", "face"})
_EVENT_ADDITION_TERMS = frozenset(
    """
    attack destroy explode fight kill kidnap murder rescue shoot steal
    """.split()
)
_HIGH_CONFIDENCE_EVENT_TERMS = frozenset(
    """
    arrive attack break burn carry catch close cross destroy discover drink drop
    eat enter exit explode fight follow give hand hold hug inspect kill kidnap
    leave lift look meet move murder open pick place point present read repair
    rescue return run shift shoot sit speak stand steal take touch turn walk
    watch write
    """.split()
)
_OPERATIONAL_REQUIREMENT_TERMS = frozenset(
    """
    appearance authoritative binding camera condition consistent continuity current
    declar direction end ending environment evidence exact flow frame identity layout
    light lighting match matching opening placement pose preceding previous reference
    requirement scale scene spatial start starting state style subject temporal trusted
    upload user visual wardrobe
    """.split()
)
_NEGATION_TERMS = frozenset({"no", "not", "never", "without"})
_NEGATIVE_REQUIREMENT_RE = re.compile(
    r"(?ix)(?:\b(?:no|not|never|without|forbid(?:den)?|prohibit(?:ed)?)\b|"
    r"(?:不得|不能|不可|禁止|不要|没有|无))"
)
_STYLE_MARKERS = re.compile(
    r"(?ix)(?:\b(?:style|look|aesthetic|realism|naturalism|anime|noir|watercolou?r|"
    r"painterly|documentary|cyberpunk|storybook|photorealistic|claymation|"
    r"monochrome|fantasy|historical|retro|film)\b|"
    r"风格|写实|动漫|动画|水彩|赛博朋克|黑色电影|绘本|纪录片|油画|水墨)"
)
_SPEECH_CUE_RE = re.compile(
    r"(?ix)(?:\b(?:say|says|said|speak|speaks|spoke|whisper|whispers|shout|shouts|"
    r"ask|asks|reply|replies|dialogue|spoken\s+line)\b|说|说道|讲话|对白|台词|低语|"
    r"耳语|喊道|问道|回答)"
)
_RAW_QUOTE_RE = re.compile(r'["“]([^"”]+)["”]')
_PERSON_ID_RE = re.compile(r"(?<![A-Za-z0-9_])ID_[A-Z][A-Z0-9_]*(?![A-Za-z0-9_])")
_PERSON_WORD_RE = re.compile(
    r"(?i)\b(?:actor|actress|adult|astronaut|boy|botanist|child|chef|conductor|"
    r"archivist|cartographer|courier|diver|doctor|elder|girl|guard|kid|man|musician|"
    r"person|pilot|robotics\s+student|soldier|student|teacher|teenager|woman|worker)\b"
)
_EXPLICIT_NONHUMAN_IDENTITY_RE = re.compile(
    r"(?ix)(?:"
    r"\b(?:alien|animal|android|automaton|bear|bird|bot|cat|creature|dog|dragon|"
    r"drone|fox|horse|mascot|monster|non[- ]?human|puppy|robot|vampire|wolf)\b|"
    r"动物|安卓人|仿生人|机器人|机械人|非人|宠物|小狗|犬|猫|狐狸|龙|马|熊|鸟|"
    r"怪物|吸血鬼|精怪|生物|吉祥物"
    r")"
)
_EXPLICIT_HUMAN_IDENTITY_RE = re.compile(
    r"(?ix)(?:"
    r"\b(?:actor|actress|adult|archivist|artist|astronaut|botanist|boy|chef|"
    r"child|conductor|courier|dancer|diver|doctor|elder|elderly|explorer|farmer|"
    r"girl|guard|human(?![- ]shaped)|inspector|kid|man|musician|neighbor|neighbour|"
    r"person|pilot|repairer|soldier|student|teacher|teenager|traveler|traveller|"
    r"woman|worker)\b|"
    r"人类|男人|女人|男孩|女孩|儿童|孩子|少年|少女|青年|老人|学生|舞者|邻居|"
    r"厨师|农民|艺术家|宇航员|航天员|潜水员|检票员|修表匠|植物学家|音乐家|"
    r"档案员"
    r")"
)
_PERSON_FREE_RE = re.compile(
    r"(?ix)(?:\bperson[ -]?free\b|\bno\s+(?:people|person|characters?)\b|"
    r"无人|没有人物|纯景)"
)
_PROPER_NAME_EXCLUSIONS = frozenset(
    {
        "A",
        "An",
        "At",
        "English",
        "Keep",
        "No",
        "Only",
        "Preserve",
        "Simplified",
        "The",
        "This",
        "Use",
    }
)

_ROLE_CATEGORIES = {
    "subject": frozenset(
        {
            "identity",
            "appearance",
            "wardrobe",
            "character",
            "person",
            "voice",
            "character_identity",
            "character_appearance",
            "entity_identity",
            "entity_appearance",
        }
    ),
    "event": frozenset(
        {"event", "action", "motion", "interaction", "behavior", "behaviour"}
    ),
    "scene": frozenset({"scene", "environment", "location", "setting"}),
    "dialogue": frozenset({"dialogue", "spoken_dialogue", "speech"}),
    "style": frozenset({"style", "aesthetic", "lighting"}),
    "state_changes": frozenset(
        {
            "object_state",
            "state",
            "state_change",
            "continuity",
            "opening_state",
        }
    ),
}

_TAIL_PREAMBLES = {
    "English": (
        "Use the current image as the first frame of the video, and continue "
        "naturally from the characters, objects, environment, composition, camera "
        "position, lighting, and action state already shown in it."
    ),
    "Simplified Chinese": (
        "以当前图片作为视频首帧，并基于首帧中已有的人物、物体、环境、构图、机位、光线和动作状态自然延续。"
    ),
}

_DECLARATIONS = {
    "English": {
        "prefix": "This video has ",
        "no_ocr": "No OCR or readable text appears anywhere in the frame.",
        "closing": "No on-screen text or subtitles.",
        "speaking": (
            "Only the written spoken dialogue is voiced; no additional human "
            "vocalizations are audible."
        ),
        "silent": (
            "Only environmental ambience is audible; no human vocal sounds are present."
        ),
        "silent_shot": "No character speaks in this shot.",
    },
    "Simplified Chinese": {
        "prefix": "本视频包含",
        "no_ocr": "画面任何位置均不出现OCR或可读文字。",
        "closing": "画面中没有屏幕文字或字幕。",
        "speaking": "只有写出的对白被发声，不出现其他人声。",
        "silent": "只有环境声，不出现任何人声。",
        "silent_shot": "本镜头中没有角色说话。",
    },
}

_NO_CUT_RE = re.compile(
    r"(?ix)(?:"
    r"\b(?:no|without)\s+(?:internal\s+)?cuts?\b|"
    r"\b(?:single|one)\s+(?:continuous\s+)?take\b|"
    r"一镜到底|单一连续镜头|单镜头连续拍摄|不要(?:内部)?切镜|无(?:内部)?剪辑"
    r")"
)
_SILENT_RE = re.compile(
    r"(?ix)(?:"
    r"\bno\s+(?:spoken\s+)?dialogue\b|\b(?:silent|wordless)\b|"
    r"\bperson[ -]?free\b|\bno\s+(?:people|person|characters?)\b|"
    r"无对白|没有对白|不要对白|静默|无声|无人|没有人物|纯景"
    r")"
)
_FOCUSED_EVENT_RE = re.compile(
    r"(?is)\b(?:the\s+)?shot\s+focus(?:es|ed|ing)?\s+on\s+"
    r"(.+?)(?=\s+involving\b|[.!?。！？；;]|$)"
)
_INVOLVING_OBJECT_RE = re.compile(
    r"(?is)\binvolving\s+(.+?)(?=[.!?。！？；;]|$)"
)
_ENGLISH_LANGUAGE_RE = re.compile(
    r"(?i)(?:caption|prompt|dialogue|output)\s+language\s*[:=]?\s*English|"
    r"(?:write|respond|caption|prompt)\s+(?:it\s+)?in\s+English|"
    r"\buse\s+English(?:\s+(?:output|captions?|prompts?))?\b|用英文"
)
_CHINESE_LANGUAGE_RE = re.compile(
    r"(?i)(?:caption|prompt|dialogue|output)\s+language\s*[:=]?\s*"
    r"(?:Simplified\s+Chinese|Chinese|Mandarin)|"
    r"(?:write|respond|caption|prompt)\s+(?:it\s+)?in\s+(?:Chinese|Mandarin)|"
    r"\buse\s+(?:Chinese|Mandarin)(?:\s+(?:output|captions?|prompts?))?\b|"
    r"用中文|简体中文|普通话"
)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _normalize_semantic_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = re.sub(r"\b(?:without\s+changing|remain(?:s|ed|ing)?|unchanged)\b", " stable ", normalized)
    return normalized


def _canonical_term(value: str) -> str:
    term = value.casefold().replace("-", " ").strip()
    if " " in term:
        return " ".join(_canonical_term(part) for part in term.split())
    if term in _SEMANTIC_SYNONYMS:
        return _SEMANTIC_SYNONYMS[term]
    if len(term) > 5 and term.endswith("ies"):
        term = term[:-3] + "y"
    elif len(term) > 5 and term.endswith("ing"):
        term = term[:-3]
        if len(term) > 3 and term[-1:] == term[-2:-1] and not term.endswith("ss"):
            term = term[:-1]
        if term.endswith("v"):
            term += "e"
    elif len(term) > 4 and term.endswith("ed"):
        term = term[:-2]
        if len(term) > 3 and term[-1:] == term[-2:-1] and not term.endswith("ss"):
            term = term[:-1]
        if term.endswith(("clos", "chang", "mov")):
            term += "e"
    elif len(term) > 4 and term.endswith("es") and not term.endswith("sses"):
        term = term[:-2]
    elif len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        term = term[:-1]
    return _SEMANTIC_SYNONYMS.get(term, term)


def _semantic_terms(value: str) -> list[str]:
    normalized = _normalize_semantic_text(value)
    terms: list[str] = []
    for token in re.findall(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*|\d+(?:\.\d+)?", normalized):
        for part in token.replace("'s", "").split("-"):
            if part in _NON_SEMANTIC_ENGLISH:
                continue
            concept = _canonical_term(part)
            if (
                len(concept) > 1
                and concept not in _NON_SEMANTIC_ENGLISH
            ):
                terms.append(concept)

    # Chinese has no reliable word boundary in the standard library.  Preserve
    # explicit semantic runs as overlapping bigrams; this is intentionally
    # conservative (failure is safer than silently accepting a changed fact).
    for run in re.findall(r"[\u3400-\u9fff]+", normalized):
        cleaned = re.sub(
            r"(?:本视频|镜头|画面|当前|保持|相同|没有|不出现|字幕|文字|角色)",
            "",
            run,
        )
        if len(cleaned) == 1:
            terms.append(cleaned)
        else:
            for index in range(len(cleaned) - 1):
                token = cleaned[index : index + 2]
                terms.append(token)
    return terms


def _normalized_dialogue(value: str) -> str:
    value = _normalize_semantic_text(value)
    return re.sub(r"[^\w\u3400-\u9fff]+", " ", value).strip()


def _split_semantic_clauses(raw_prompt: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？；;])\s*", raw_prompt)
        if item.strip()
    ]


def _event_terms(value: str) -> set[str]:
    """Return only unambiguous story actions, never PE/camera instructions."""

    value = re.sub(
        r"(?ix)\bhold(?:s|ing)?\s+(?:a\s+)?(?:quiet\s+)?pause\b",
        " ",
        value,
    )
    terms = set(_semantic_terms(value)).intersection(_HIGH_CONFIDENCE_EVENT_TERMS)
    # In StoryMem prompts ``hand`` overwhelmingly appears in the operational
    # phrase "preserve hand placement", where it is a noun rather than an act.
    terms.discard("hand")
    return terms


def _raw_style_sources(clauses: Sequence[str]) -> list[str]:
    """Extract authored style phrases without swallowing story/PE boilerplate."""

    sources: list[str] = []
    for clause in clauses:
        if not _STYLE_MARKERS.search(clause) or _FOCUSED_EVENT_RE.search(clause):
            continue
        trailing = re.search(
            r"(?is)\b(?:keep|maintain)\b.+?,\s+and\s+(.+)$",
            clause.rstrip(".。"),
        )
        sources.append((trailing.group(1) if trailing else clause).strip())
    return sources


def _requirement_terms(
    description: str,
    *,
    category: str,
    roles: set[str],
) -> set[str]:
    """Project an EvidencePlan description onto high-confidence world facts.

    EvidencePlan continuity rows often contain operational prose such as
    ``preserve end-frame temporal/spatial placement``.  Those words describe
    how the adapter should use evidence; they are not words that the generated
    caption must repeat.  Typed entity, scene, style, and object-state rows are
    authoritative, while event rows contribute only explicit action concepts.
    """

    terms = set(_semantic_terms(description))
    if category == "event":
        return _event_terms(description)
    if category == "state_changes" and not roles.intersection(
        {"object_state", "state", "state_change"}
    ):
        transition_terms = {
            concept
            for transition in _state_transitions(description)
            for side in ("source_concepts", "target_concepts")
            for concept in transition.get(side) or []
        }
        return transition_terms
    terms.difference_update(_OPERATIONAL_REQUIREMENT_TERMS)
    if category == "subject":
        terms.difference_update(_SUBJECT_ATTRIBUTE_CARRIERS)
    return terms


def _state_transitions(value: str) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    patterns = (
        re.compile(
            r"(?is)\bfrom\s+([^,.;!?。；！？]+?)\s+to\s+([^,.;!?。；！？]+)"
        ),
        re.compile(r"从\s*([^，。；！？]+?)\s*(?:到|变为|变成)\s*([^，。；！？]+)"),
        re.compile(r"由\s*([^，。；！？]+?)\s*(?:变为|变成)\s*([^，。；！？]+)"),
    )
    for match in (
        match
        for pattern in patterns
        for match in pattern.finditer(value)
    ):
        source = sorted(set(_semantic_terms(match.group(1))))
        target = sorted(set(_semantic_terms(match.group(2))))
        if source and target:
            transitions.append(
                {
                    "source_concepts": source,
                    "target_concepts": target,
                    "source_text_sha256": _sha256_text(match.group(0)),
                }
            )
    return transitions


def _semantic_obligation(
    *, source: str, source_text: str, terms: Sequence[str]
) -> dict[str, Any] | None:
    concepts = sorted({str(item) for item in terms if str(item)})
    if not concepts:
        return None
    return {
        "source": source,
        "source_sha256": _sha256_text(source_text),
        "source_text": source_text,
        "required_concepts": concepts,
    }


def _infer_raw_subject_count(raw_prompt: str, subject_text: str | None) -> int | None:
    if _PERSON_FREE_RE.search(raw_prompt):
        return 0
    if subject_text and _PERSON_WORD_RE.search(subject_text):
        # Count explicit joined people where possible; otherwise the subject
        # phrase denotes one person.
        people = _PERSON_WORD_RE.findall(subject_text)
        return max(1, len(people))
    return None


def build_semantic_preservation_contract(
    raw_prompt: str,
    *,
    expected_subject_ids: Sequence[str] = (),
    semantic_requirements: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the deterministic raw-intent contract used by official v7 PE.

    The contract is deliberately independent of a generated prompt.  It is
    constructed only from the immutable raw request and the accepted plan's
    immutable requirement projection, then content-addressed before Gemini is
    called.  Every non-empty obligation is checked against the final wire text.
    """

    if not isinstance(raw_prompt, str) or not raw_prompt:
        raise V7PromptValidationError("raw prompt is empty")
    subject_ids = sorted(
        {
            str(item)
            for item in expected_subject_ids
            if isinstance(item, str) and _PERSON_ID_RE.fullmatch(item)
        }
    )
    obligations: dict[str, list[dict[str, Any]]] = {
        category: [] for category in _SEMANTIC_CATEGORIES if category != "dialogue"
    }
    allowed_addition_concepts: dict[str, set[str]] = {
        category: set()
        for category in _SEMANTIC_CATEGORIES
        if category != "dialogue"
    }
    clauses = _split_semantic_clauses(raw_prompt)
    raw_dialogue = [_normalized_dialogue(item) for item in _RAW_QUOTE_RE.findall(raw_prompt)]
    raw_dialogue = [item for item in raw_dialogue if item]
    raw_without_dialogue = _RAW_QUOTE_RE.sub(" ", raw_prompt)
    semantic_clauses = [
        clause
        for clause in clauses
        if not re.search(
            r"(?ix)(?:no\s+(?:subtitles?|captions?|ui\s+overlays?|watermarks?|logos?|"
            r"readable\s+text)|不出现.*(?:字幕|文字|水印|标志))",
            clause,
        )
    ]

    subject_text: str | None = None
    scene_text: str | None = None
    if semantic_clauses:
        first = semantic_clauses[0]
        match = re.search(r"(?is)\bof\s+(.+?)\s+\bin\s+(.+)$", first.rstrip(".。"))
        if match is None:
            leading = re.search(r"(?is)^(.+?)\s+\bin\s+(.+)$", first.rstrip(".。"))
            if leading and (
                _PERSON_WORD_RE.search(leading.group(1))
                or re.search(r"(?<![A-Za-z])[A-Z][a-z]{2,}(?![A-Za-z])", leading.group(1))
            ):
                match = leading
        if match:
            subject_text = match.group(1).strip()
            scene_text = match.group(2).strip()
            obligation = _semantic_obligation(
                source="raw_request.subject",
                source_text=subject_text,
                terms=set(_semantic_terms(subject_text))
                - _SUBJECT_ATTRIBUTE_CARRIERS,
            )
            if obligation is not None:
                obligations["subject"].append(obligation)
            obligation = _semantic_obligation(
                source="raw_request.scene",
                source_text=scene_text,
                terms=_semantic_terms(scene_text),
            )
            if obligation is not None:
                obligations["scene"].append(obligation)

    proper_names = {
        concept
        for item in re.findall(
            r"(?<![A-Za-z])([A-Z][a-z]{2,})(?![A-Za-z])",
            raw_without_dialogue,
        )
        if item not in _PROPER_NAME_EXCLUSIONS
        for concept in _semantic_terms(item)
    }
    if subject_text is None and proper_names and not subject_ids:
        obligation = _semantic_obligation(
            source="raw_request.subject_names",
            source_text=raw_prompt,
            terms=sorted(proper_names),
        )
        if obligation is not None:
            obligations["subject"].append(obligation)

    if subject_ids and proper_names:
        for category_rows in obligations.values():
            for row in category_rows:
                row["required_concepts"] = [
                    concept
                    for concept in row["required_concepts"]
                    if concept not in proper_names
                ]

    all_raw_terms = set(
        _semantic_terms(_RAW_QUOTE_RE.sub(" ", " ".join(semantic_clauses)))
    )
    # A canonical ID is the adapter-approved replacement for a user's name.
    # Do not also demand that the private/display name appear in the wire text.
    if subject_ids:
        all_raw_terms.difference_update(proper_names)

    scene_terms = set(_semantic_terms(scene_text or ""))
    if not scene_terms:
        scene_terms = all_raw_terms.intersection(_SCENE_TERMS)
        obligation = _semantic_obligation(
            source="raw_request.scene_terms",
            source_text=raw_prompt,
            terms=scene_terms,
        )
        if obligation is not None:
            obligations["scene"].append(obligation)

    style_clauses = _raw_style_sources(semantic_clauses)
    style_terms: set[str] = set()
    for clause in style_clauses:
        terms = set(_semantic_terms(clause))
        # Boilerplate continuity nouns do not become style obligations merely
        # because they share the same sentence as the requested visual style.
        terms.difference_update(
            {
                "character",
                "identity",
                "wardrobe",
                "prop",
                "design",
                "scene",
                "layout",
                "stable",
            }
        )
        style_terms.update(terms)
        obligation = _semantic_obligation(
            source="raw_request.style",
            source_text=clause,
            terms=terms,
        )
        if obligation is not None:
            obligations["style"].append(obligation)

    for object_text in _INVOLVING_OBJECT_RE.findall(" ".join(semantic_clauses)):
        obligation = _semantic_obligation(
            source="raw_request.object_state",
            source_text=object_text,
            terms=_semantic_terms(object_text),
        )
        if obligation is not None:
            obligations["state_changes"].append(obligation)

    assigned = set()
    for category in ("subject", "scene", "style"):
        for obligation in obligations[category]:
            assigned.update(obligation["required_concepts"])
    focused_events = _FOCUSED_EVENT_RE.findall(" ".join(semantic_clauses))
    event_source = " ".join(focused_events or semantic_clauses)
    event_terms = _event_terms(event_source) - assigned
    # Speech is checked structurally below; these generic cues are not event
    # anchors in their own right.
    event_terms.difference_update({"speak", "dialogue", "silent", "wordless"})
    if event_terms:
        obligation = _semantic_obligation(
            source="raw_request.event",
            source_text=raw_prompt,
            terms=event_terms,
        )
        if obligation is not None:
            obligations["event"].append(obligation)

    raw_category_concepts = {
        category: {
            concept
            for obligation in obligations[category]
            for concept in obligation["required_concepts"]
        }
        for category in obligations
    }

    for index, requirement in enumerate(semantic_requirements):
        if not isinstance(requirement, Mapping):
            raise V7PromptValidationError("semantic requirement must be an object")
        description = requirement.get("description")
        if not isinstance(description, str) or not description.strip():
            raise V7PromptValidationError("semantic requirement description is empty")
        if re.fullmatch(r"(?i)requirement\s+for\s+binding[_ -]?\d+\.?", description.strip()):
            # Legacy unit fixtures carry no semantic information in this label.
            continue
        roles = {
            str(role).strip().casefold().replace(" ", "_")
            for role in requirement.get("semantic_roles") or []
            if isinstance(role, str) and role.strip()
        }
        categories = [
            category
            for category, category_roles in _ROLE_CATEGORIES.items()
            if roles.intersection(category_roles)
        ]
        authorization_categories = [
            category
            for category, category_roles in _ROLE_CATEGORIES.items()
            if roles.intersection(
                category_roles - {"continuity", "opening_state"}
            )
        ]
        if not categories:
            # Untyped EvidencePlan prose is not a reliable semantic category.
            # The immutable raw request remains authoritative for story facts.
            continue
        for category in categories:
            if category == "dialogue":
                continue
            terms = _requirement_terms(
                description,
                category=category,
                roles=roles,
            )
            # A typed row from the accepted, immutable EvidencePlan may carry a
            # fact visible only in continuity evidence (for example, a robot in
            # the previous tail frame).  Such facts are permitted additions,
            # not obligations: absence from the generated prompt must remain
            # valid.  Untyped rows never reach this branch.
            # Authorization is category-local.  A word that happens to occur
            # elsewhere in the raw request (for example ``amber`` in scene
            # lighting) must not turn a plan-only subject attribute into a
            # required subject fact.
            allowed_terms: set[str] = set()
            if category == "subject":
                subjects = requirement.get("subjects")
                # Normalization converts continuity wording such as "without
                # changing" to ``stable`` before the negative check, while a
                # real exclusion such as "no robot" cannot grant permission.
                effective_negative = bool(
                    _NEGATIVE_REQUIREMENT_RE.search(
                        _normalize_semantic_text(description)
                    )
                )
                if (
                    isinstance(subjects, list)
                    and subjects
                    and authorization_categories == ["subject"]
                    and not effective_negative
                ):
                    allowed_terms = {
                        concept
                        for subject in subjects
                        if isinstance(subject, str) and subject.strip()
                        for concept in _requirement_terms(
                            subject,
                            category="subject",
                            roles=roles,
                        )
                    }
                elif (
                    len(categories) == 1
                    and not _NEGATIVE_REQUIREMENT_RE.search(description)
                    and not subjects
                ):
                    # Backwards-compatible input without the adapter's
                    # structured subject projection is safe only when the
                    # typed row names one semantic category.
                    allowed_terms = set(terms)
            elif (
                len(categories) == 1
                and not _NEGATIVE_REQUIREMENT_RE.search(description)
            ):
                allowed_terms = set(terms)
            allowed_addition_concepts[category].update(
                allowed_terms - raw_category_concepts[category]
            )
            # EvidencePlan text may describe provenance or handling policy.
            # It may strengthen a fact already present in raw intent, but it
            # must never create a new word-level obligation of its own.
            terms.intersection_update(raw_category_concepts[category])
            if len(categories) > 1:
                # A multi-role continuity sentence cannot tell us which noun
                # belongs to which semantic category.  Restrict it to facts
                # already classified from the raw request instead of copying
                # the whole sentence into every category.
                terms.intersection_update(raw_category_concepts[category])
            if subject_ids:
                terms.difference_update(proper_names)
            obligation = _semantic_obligation(
                source=f"accepted_plan.requirements[{index}]",
                source_text=description,
                terms=terms,
            )
            if obligation is not None:
                obligations[category].append(obligation)

    if raw_dialogue:
        dialogue_mode = "exact"
    elif _SILENT_RE.search(raw_prompt) or not _SPEECH_CUE_RE.search(raw_prompt):
        dialogue_mode = "forbidden"
    else:
        dialogue_mode = "required_unspecified"

    state_transitions = _state_transitions(raw_prompt)

    inferred_count = _infer_raw_subject_count(raw_prompt, subject_text)
    expected_count = len(subject_ids) if subject_ids else inferred_count
    body: dict[str, Any] = {
        "schema_version": V7_SEMANTIC_CONTRACT_VERSION,
        "raw_prompt_sha256": _sha256_text(raw_prompt),
        "expected_subject_ids": subject_ids,
        "expected_subject_count": expected_count,
        "dialogue": {
            "mode": dialogue_mode,
            "exact_lines": raw_dialogue,
        },
        "state_transitions": state_transitions,
        "obligations": obligations,
        "allowed_addition_concepts": {
            category: sorted(concepts)
            for category, concepts in allowed_addition_concepts.items()
        },
    }
    body["contract_sha256"] = _sha256_text(_canonical_json(body))
    return body


def _validate_semantic_contract_shape(
    contract: Mapping[str, Any], *, raw_prompt: str
) -> dict[str, Any]:
    try:
        value = json.loads(_canonical_json(contract))
    except (TypeError, ValueError) as exc:
        raise V7PromptValidationError("semantic preservation contract is not JSON-safe") from exc
    if value.get("schema_version") != V7_SEMANTIC_CONTRACT_VERSION:
        raise V7PromptValidationError("semantic preservation contract version is invalid")
    if value.get("raw_prompt_sha256") != _sha256_text(raw_prompt):
        raise V7PromptValidationError("semantic preservation contract is bound to another raw prompt")
    stored = value.pop("contract_sha256", None)
    expected = _sha256_text(_canonical_json(value))
    if stored != expected:
        raise V7PromptValidationError("semantic preservation contract hash is invalid")
    value["contract_sha256"] = stored
    obligations = value.get("obligations")
    if not isinstance(obligations, dict) or set(obligations) != set(_SEMANTIC_CATEGORIES) - {"dialogue"}:
        raise V7PromptValidationError("semantic preservation obligations are incomplete")
    allowed_additions = value.get("allowed_addition_concepts")
    expected_categories = set(_SEMANTIC_CATEGORIES) - {"dialogue"}
    if not isinstance(allowed_additions, dict) or set(allowed_additions) != expected_categories:
        raise V7PromptValidationError(
            "semantic preservation allowed addition concepts are incomplete"
        )
    for concepts in allowed_additions.values():
        if (
            not isinstance(concepts, list)
            or any(not isinstance(concept, str) or not concept for concept in concepts)
            or concepts != sorted(set(concepts))
        ):
            raise V7PromptValidationError(
                "semantic preservation allowed addition concepts are invalid"
            )
    if not isinstance(value.get("state_transitions"), list):
        raise V7PromptValidationError("semantic preservation state transitions are invalid")
    return value


def _candidate_dialogue(text: str) -> list[str]:
    return [
        normalized
        for normalized in (_normalized_dialogue(item) for item in _RAW_QUOTE_RE.findall(text))
        if normalized
    ]


def _nonhuman_identity_bindings(text: str, candidate_ids: set[str]) -> list[str]:
    """Return canonical IDs whose identity anchor declares a non-human.

    ID_X belongs exclusively to people in the official v7 contract.  The
    relative ordering rule avoids treating a person's later prop or companion
    (for example, ``a robotics student with a helper drone``) as that ID's
    identity, while still rejecting ``a humanoid robot shaped like a man``.
    """

    invalid: list[str] = []
    for person_id in sorted(candidate_ids):
        match = re.search(
            rf"(?:{re.escape(person_id)}\s+is\s+|{re.escape(person_id)}是)"
            r"([^.!?。！？；;]+)",
            text,
        )
        if match is None:
            continue
        identity = match.group(1)
        entity = _EXPLICIT_NONHUMAN_IDENTITY_RE.search(identity)
        human = _EXPLICIT_HUMAN_IDENTITY_RE.search(identity)
        if entity is not None and (human is None or entity.start() < human.start()):
            invalid.append(person_id)
    return invalid


def _obligation_concepts(contract: Mapping[str, Any], category: str) -> set[str]:
    return {
        str(concept)
        for row in contract["obligations"][category]
        for concept in row.get("required_concepts") or []
    }


def _violation(code: str, **details: Any) -> dict[str, Any]:
    """Build one JSON-safe deterministic validation violation."""

    value = {"code": code, **details}
    return json.loads(_canonical_json(value))


def _format_violation(message: str) -> dict[str, Any]:
    """Map stable v7 format messages to durable machine-readable codes."""

    exact_codes = {
        "v7 prompt must be one trimmed paragraph": "format.single_paragraph",
        "v7 prompt must not contain Markdown wrappers": "format.markdown_wrapper",
        "v7 prompt must not contain an ellipsis": "format.ellipsis",
        (
            "expected_caption_language must be English, Simplified Chinese, or null"
        ): "language.invalid_lock",
        (
            "temporal continuation requires exactly one language-matched official "
            "tail-frame preamble at the start"
        ): "temporal.preamble_count",
        (
            "tail-frame preamble violates the caption-language lock"
        ): "temporal.preamble_language",
        (
            "tail-frame preamble must precede one complete ordinary prompt"
        ): "temporal.incomplete_prompt",
        (
            "official tail-frame preamble is forbidden without temporal continuation"
        ): "temporal.unexpected_preamble",
        (
            "an explicit no-cut request must not contain cut syntax"
        ): "shots.no_cut_syntax",
        (
            "cut-count prefix violates the caption-language lock"
        ): "shots.prefix_language",
        "v7 shot labels must be exactly sequential shot1..shotN": "shots.labels_sequence",
        "cut-count prefix must be followed immediately by shot1": "shots.prefix_adjacency",
        "v7 prompt contains malformed or extra shot labels": "shots.labels_malformed",
        (
            "v7 prompt mixes fixed declarations from both languages"
        ): "declarations.mixed_languages",
        (
            "silent v7 prompt requires exactly one silence and silent-shot declaration"
        ): "dialogue.silence_declarations",
        (
            "speaking v7 prompt requires exactly one written-dialogue vocal declaration"
        ): "dialogue.vocal_declaration",
        (
            "speaking v7 prompt requires quoted dialogue from a canonical ID"
        ): "dialogue.canonical_quoted_line",
    }
    code = exact_codes.get(message)
    if code is not None:
        return _violation(code)
    match = re.fullmatch(
        r"v7 prompt must contain ([0-9]+)-([0-9]+) characters; got ([0-9]+)",
        message,
    )
    if match:
        return _violation(
            "length.out_of_range",
            expected_minimum=int(match.group(1)),
            expected_maximum=int(match.group(2)),
            actual_count=int(match.group(3)),
        )
    if message.startswith("default v7 output requires the language-matched "):
        return _violation("shots.required_prefix_missing")
    match = re.fullmatch(
        r"segment ([0-9]+) requires exactly one no-OCR declaration", message
    )
    if match:
        return _violation(
            "declarations.no_ocr_count", segment_index=int(match.group(1))
        )
    match = re.fullmatch(
        r"segment ([0-9]+) requires exactly one no-subtitle closing", message
    )
    if match:
        return _violation(
            "declarations.no_subtitle_count", segment_index=int(match.group(1))
        )
    match = re.fullmatch(
        r"segment ([0-9]+) must end with adjacent no-OCR and no-subtitle declarations",
        message,
    )
    if match:
        return _violation(
            "declarations.closing_adjacency", segment_index=int(match.group(1))
        )
    match = re.fullmatch(
        r"v7 prompt requires exactly one (identity|clothing) anchor for "
        r"(ID_[A-Z][A-Z0-9_]*)",
        message,
    )
    if match:
        return _violation(
            f"subject.{match.group(1)}_anchor_count",
            subject_ids=[match.group(2)],
        )
    return _violation("validation.unclassified")


def _high_confidence_additions(
    candidate_terms: set[str], *, raw_terms: set[str], category: str,
    category_required: set[str], allowed_addition_concepts: set[str],
) -> list[str]:
    # A vague request may intentionally leave a category to PE (for example,
    # no location at all).  Addition checks become authoritative when either
    # raw intent or a typed immutable plan row supplies facts for the category.
    if not category_required and not allowed_addition_concepts:
        return []
    guarded = {
        "subject": _SUBJECT_ENTITY_TERMS,
        "event": _EVENT_ADDITION_TERMS,
        # ``room`` and ``film`` are also ordinary v7 composition vocabulary
        # (room tone, film look).  They are too ambiguous to prove invention.
        "scene": _SCENE_TERMS - {"room"},
        "style": _STYLE_TERMS - {"film"},
    }.get(category, frozenset())
    return sorted(
        (candidate_terms & guarded) - (raw_terms | allowed_addition_concepts)
    )


def _without_fixed_declarations(text: str) -> str:
    """Remove mandatory serializer prose before semantic negation checks."""

    cleaned = text
    for declarations in _DECLARATIONS.values():
        for key in ("no_ocr", "closing", "speaking", "silent", "silent_shot"):
            cleaned = cleaned.replace(declarations[key], " ")
    cleaned = re.sub(
        r"(?ix)(?:\bthere\s+is\s+)?\bno\s+background\s+music\b|"
        r"\bwithout\s+background\s+music\b|无背景音乐|没有背景音乐",
        " ",
        cleaned,
    )
    return cleaned


def _negated_required_concepts(text: str, required: set[str]) -> set[str]:
    """Return required concepts occurring in a short English negation scope."""

    negated: set[str] = set()
    # Negation cannot leak across sentence/segment boundaries.  Without this,
    # an audio sentence such as "There is no background music." could negate
    # the subject named at the beginning of the next ``shotN:`` segment.
    for clause in re.split(r"[.!?。！？；;]+", _normalize_semantic_text(text)):
        tokens = re.findall(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*", clause)
        for index, token in enumerate(tokens):
            if token not in _NEGATION_TERMS:
                continue
            for raw in tokens[index + 1 : index + 6]:
                concept = _canonical_term(raw)
                if concept in required:
                    negated.add(concept)
    return negated


def _validate_semantic_preservation(
    text: str,
    *,
    raw_prompt: str,
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    active = _validate_semantic_contract_shape(
        contract or build_semantic_preservation_contract(raw_prompt),
        raw_prompt=raw_prompt,
    )
    semantic_candidate_text = _without_fixed_declarations(text)
    candidate_term_sequence = _semantic_terms(semantic_candidate_text)
    candidate_terms = set(candidate_term_sequence)
    raw_terms = set(_semantic_terms(_RAW_QUOTE_RE.sub(" ", raw_prompt)))
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    violations: list[dict[str, Any]] = []
    for category in _SEMANTIC_CATEGORIES:
        if category == "dialogue":
            continue
        category_missing: list[str] = []
        rows = active["obligations"][category]
        for row in rows:
            missing = [
                concept
                for concept in row["required_concepts"]
                if concept not in candidate_terms
            ]
            category_missing.extend(missing)
        category_missing = sorted(set(category_missing))
        required = _obligation_concepts(active, category)
        added = _high_confidence_additions(
            candidate_terms,
            raw_terms=raw_terms,
            category=category,
            category_required=required,
            allowed_addition_concepts=set(
                active["allowed_addition_concepts"][category]
            ),
        )
        negated = sorted(
            _negated_required_concepts(semantic_candidate_text, required)
            - _negated_required_concepts(raw_prompt, required)
        )
        if category_missing:
            failures.append(f"{category}: missing {', '.join(category_missing)}")
            violations.append(
                _violation(
                    "semantic.missing_concepts",
                    category=category,
                    missing_concepts=category_missing,
                )
            )
        if added:
            failures.append(f"{category}: added {', '.join(added)}")
            violations.append(
                _violation(
                    "semantic.added_concepts",
                    category=category,
                    added_concepts=added,
                )
            )
        if negated:
            failures.append(f"{category}: negated {', '.join(negated)}")
            violations.append(
                _violation(
                    "semantic.negated_concepts",
                    category=category,
                    negated_concepts=negated,
                )
            )
        checks.append(
            {
                "category": category,
                "result": "pass" if not (category_missing or added or negated) else "fail",
                "obligation_count": len(rows),
                "missing_concepts": category_missing,
                "added_concepts": added,
                "negated_concepts": negated,
            }
        )

    expected_ids = set(active.get("expected_subject_ids") or [])
    candidate_ids = set(_PERSON_ID_RE.findall(text))
    expected_count = active.get("expected_subject_count")
    if expected_ids and candidate_ids != expected_ids:
        failures.append(
            "subject: canonical IDs differ "
            f"(expected {sorted(expected_ids)}, got {sorted(candidate_ids)})"
        )
        violations.append(
            _violation(
                "subject.canonical_ids_mismatch",
                expected_subject_ids=sorted(expected_ids),
                actual_subject_ids=sorted(candidate_ids),
            )
        )
    elif isinstance(expected_count, int) and len(candidate_ids) != expected_count:
        failures.append(
            "subject: person count differs "
            f"(expected {expected_count}, got {len(candidate_ids)})"
        )
        violations.append(
            _violation(
                "subject.person_count_mismatch",
                expected_count=expected_count,
                actual_count=len(candidate_ids),
            )
        )
    nonhuman_ids = _nonhuman_identity_bindings(text, candidate_ids)
    if nonhuman_ids:
        failures.append(
            "subject: canonical person IDs assigned to non-human identities "
            f"({', '.join(nonhuman_ids)})"
        )
        violations.append(
            _violation(
                "subject.nonhuman_identity",
                subject_ids=nonhuman_ids,
            )
        )

    dialogue = active["dialogue"]
    candidate_lines = _candidate_dialogue(text)
    mode = dialogue["mode"]
    expected_lines = dialogue["exact_lines"]
    dialogue_ok = True
    if mode == "forbidden" and candidate_lines:
        dialogue_ok = False
        failures.append("dialogue: raw request contains no dialogue but wire prompt adds it")
        violations.append(
            _violation(
                "dialogue.unexpected",
                mode=mode,
                expected_count=0,
                actual_count=len(candidate_lines),
            )
        )
    elif mode == "required_unspecified" and len(candidate_lines) != 1:
        dialogue_ok = False
        failures.append(
            "dialogue: requested speech requires exactly one quoted line; "
            f"got {len(candidate_lines)}"
        )
        violations.append(
            _violation(
                (
                    "dialogue.requested_speech_missing"
                    if not candidate_lines
                    else "dialogue.requested_speech_count"
                ),
                mode=mode,
                expected_count=1,
                actual_count=len(candidate_lines),
            )
        )
    elif mode == "exact" and candidate_lines != expected_lines:
        dialogue_ok = False
        failures.append("dialogue: quoted lines differ from the immutable raw request")
        violations.append(
            _violation(
                "dialogue.exact_lines_mismatch",
                mode=mode,
                expected_count=len(expected_lines),
                actual_count=len(candidate_lines),
            )
        )
    checks.append(
        {
            "category": "dialogue",
            "result": "pass" if dialogue_ok else "fail",
            "mode": mode,
            "expected_line_count": len(expected_lines),
            "candidate_line_count": len(candidate_lines),
        }
    )

    candidate_transitions = _state_transitions(text)
    transition_failures = []
    for transition in active["state_transitions"]:
        source = set(transition.get("source_concepts") or [])
        target = set(transition.get("target_concepts") or [])
        if not any(
            source.issubset(set(candidate.get("source_concepts") or []))
            and target.issubset(set(candidate.get("target_concepts") or []))
            for candidate in candidate_transitions
        ):
            transition_failures.append(
                f"{'+'.join(sorted(source))} -> {'+'.join(sorted(target))}"
            )
    if transition_failures:
        failures.append(
            "state_changes: ordered transition differs ("
            + ", ".join(transition_failures)
            + ")"
        )
        violations.append(
            _violation(
                "state_changes.ordered_transition_missing",
                category="state_changes",
                missing_transitions=transition_failures,
            )
        )

    if failures:
        raise V7PromptValidationError(
            "raw-intent semantic preservation failed: " + "; ".join(failures),
            violations=violations,
        )
    return {
        "schema_version": V7_SEMANTIC_CONTRACT_VERSION,
        "contract_sha256": active["contract_sha256"],
        "raw_prompt_sha256": active["raw_prompt_sha256"],
        "result": "pass",
        "checks": checks,
    }


class V7PromptValidationError(ValueError):
    """A deterministic v7 rejection with secret-free structured reasons."""

    def __init__(
        self,
        message: str,
        *,
        violations: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        normalized = [
            json.loads(_canonical_json(dict(item)))
            for item in violations
            if isinstance(item, Mapping)
            and isinstance(item.get("code"), str)
            and bool(str(item.get("code")).strip())
        ]
        self.violations = normalized or [_violation("validation.unclassified")]
        self.violation_codes = list(
            dict.fromkeys(str(item["code"]) for item in self.violations)
        )


def infer_caption_language(raw_prompt: str) -> str:
    """Infer the official two-way caption-language lock from user text."""

    english = bool(_ENGLISH_LANGUAGE_RE.search(raw_prompt))
    chinese = bool(_CHINESE_LANGUAGE_RE.search(raw_prompt))
    if english != chinese:
        return "English" if english else "Simplified Chinese"
    han_count = len(re.findall(r"[\u3400-\u9fff]", raw_prompt))
    latin_words = len(re.findall(r"[A-Za-z]+", raw_prompt))
    return (
        "Simplified Chinese"
        if han_count >= 2 and han_count >= latin_words
        else "English"
    )


@dataclass(frozen=True, slots=True)
class DirectorV7PromptValidator:
    """Validate format and immutable raw intent before persistence."""

    validation_id: str = V7_VALIDATOR_ID
    min_characters: int = 1500
    max_characters: int | None = 1999
    semantic_contract: Mapping[str, Any] | None = None

    def correction_context(
        self,
        *,
        raw_prompt: str,
        condition_effect: str | None,
        expected_caption_language: str | None,
    ) -> dict[str, Any]:
        language = expected_caption_language or infer_caption_language(raw_prompt)
        semantic_contract = _validate_semantic_contract_shape(
            self.semantic_contract or build_semantic_preservation_contract(raw_prompt),
            raw_prompt=raw_prompt,
        )
        obligation_projection = {
            category: sorted(_obligation_concepts(semantic_contract, category))
            for category in _SEMANTIC_CATEGORIES
            if category != "dialogue"
        }
        transition_projection = [
            {
                "source_concepts": list(transition.get("source_concepts") or []),
                "target_concepts": list(transition.get("target_concepts") or []),
            }
            for transition in semantic_contract["state_transitions"]
        ]
        dialogue_mode = semantic_contract["dialogue"]["mode"]
        return {
            "validator_id": self.validation_id,
            "locked_caption_language": language,
            "condition_effect": condition_effect,
            "explicit_no_cut": bool(_NO_CUT_RE.search(raw_prompt)),
            "dialogue_mode": dialogue_mode,
            "exact_dialogue_lines": list(
                semantic_contract["dialogue"].get("exact_lines") or []
            ),
            "expected_subject_ids": list(
                semantic_contract.get("expected_subject_ids") or []
            ),
            "expected_subject_count": semantic_contract.get(
                "expected_subject_count"
            ),
            "semantic_contract_sha256": semantic_contract["contract_sha256"],
            "semantic_obligations": obligation_projection,
            "state_transitions": transition_projection,
            "required_character_range": [
                self.min_characters,
                self.max_characters,
            ],
            "required_output_checklist": {
                "format": {
                    "single_trimmed_paragraph": True,
                    "markdown_wrappers_forbidden": True,
                    "ellipsis_forbidden": True,
                    "character_range": [
                        self.min_characters,
                        self.max_characters,
                    ],
                    "shot_structure": (
                        "single_continuous_no_cut"
                        if _NO_CUT_RE.search(raw_prompt)
                        else "language_matched_2_or_3_shot_prefix_with_sequential_labels"
                    ),
                },
                "dialogue": {
                    "mode": dialogue_mode,
                    "quoted_line_requirement": (
                        "none"
                        if dialogue_mode == "forbidden"
                        else (
                            "preserve_exact_lines"
                            if dialogue_mode == "exact"
                            else "compose_exactly_one_natural_line"
                        )
                    ),
                    "canonical_id_speaker_required": dialogue_mode != "forbidden",
                    "source_no_exact_dialogue_phrase_is_not_silence": (
                        dialogue_mode == "required_unspecified"
                    ),
                },
                "subjects": {
                    "exact_canonical_ids": list(
                        semantic_contract.get("expected_subject_ids") or []
                    ),
                    "identity_anchor_count_per_id": 1,
                    "clothing_anchor_count_per_id": 1,
                },
                "semantic_preservation": (
                    "include every semantic_obligation and ordered state transition; "
                    "do not add or negate guarded world facts"
                ),
                "per_segment_declarations": {
                    "no_ocr_count": 1,
                    "no_subtitle_closing_count": 1,
                    "must_be_adjacent_at_segment_end": True,
                },
                "privacy": {
                    "registry_or_legacy_material_tags_forbidden": True,
                    "numbered_reference_images_forbidden": True,
                    "private_paths_urls_or_evidence_handles_forbidden": True,
                },
            },
        }

    def validate(
        self,
        text: str,
        *,
        raw_prompt: str,
        condition_effect: str | None,
        expected_caption_language: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(text, str) or not text:
            raise V7PromptValidationError(
                "v7 prompt is empty",
                violations=[_violation("output.empty")],
            )
        errors: list[str] = []
        structured_violations: list[dict[str, Any]] = []
        semantic_error_message: str | None = None
        if text != text.strip() or "\n" in text or "\r" in text:
            errors.append("v7 prompt must be one trimmed paragraph")
        if len(text) < self.min_characters or (
            self.max_characters is not None and len(text) > self.max_characters
        ):
            errors.append(
                f"v7 prompt must contain {self.min_characters}-{self.max_characters} characters; "
                f"got {len(text)}"
            )
        if "```" in text or re.match(r"^\s*(?:#{1,6}|[-*+]\s)", text):
            errors.append("v7 prompt must not contain Markdown wrappers")
        if "..." in text or "…" in text:
            errors.append("v7 prompt must not contain an ellipsis")

        if expected_caption_language not in {
            None,
            "English",
            "Simplified Chinese",
        }:
            errors.append(
                "expected_caption_language must be English, Simplified Chinese, or null"
            )
            raise V7PromptValidationError(
                "; ".join(errors),
                violations=[_format_violation(message) for message in errors],
            )
        expected_language = expected_caption_language or infer_caption_language(raw_prompt)
        tail_matches = [
            (language, preamble)
            for language, preamble in _TAIL_PREAMBLES.items()
            if text.startswith(preamble)
        ]
        embedded_tail_count = sum(
            text.count(preamble) for preamble in _TAIL_PREAMBLES.values()
        )
        ordinary = text
        if condition_effect == "temporal_continuation":
            if len(tail_matches) != 1 or embedded_tail_count != 1:
                errors.append(
                    "temporal continuation requires exactly one language-matched "
                    "official tail-frame preamble at the start"
                )
            if tail_matches:
                tail_language, preamble = tail_matches[0]
                if tail_language != expected_language:
                    errors.append(
                        "tail-frame preamble violates the caption-language lock"
                    )
                ordinary = text[len(preamble) :]
                if ordinary.startswith(" "):
                    ordinary = ordinary[1:]
                if not ordinary or ordinary.startswith(" "):
                    errors.append(
                        "tail-frame preamble must precede one complete ordinary prompt"
                    )
        else:
            if embedded_tail_count:
                errors.append(
                    "official tail-frame preamble is forbidden without temporal continuation"
                )

        no_cut = bool(_NO_CUT_RE.search(raw_prompt))
        english_prefix = re.match(r"^This video has ([23]) shots\. ", ordinary)
        chinese_prefix = re.match(r"^本视频包含([23])个镜头。", ordinary)
        prefix = english_prefix or chinese_prefix
        language = expected_language
        if no_cut:
            if prefix is not None or re.search(r"(?i)\bshot\s*[0-9]+\s*[:：]", ordinary):
                errors.append(
                    "an explicit no-cut request must not contain cut syntax"
                )
            segment_count = 1
            segments = [ordinary]
        else:
            if prefix is None:
                expected = (
                    "This video has N shots."
                    if expected_language == "English"
                    else "本视频包含N个镜头。"
                )
                errors.append(
                    f"default v7 output requires the language-matched {expected} prefix"
                )
                segment_count = 0
                segments = [ordinary]
            else:
                language = (
                    "English" if english_prefix is not None else "Simplified Chinese"
                )
                if language != expected_language:
                    errors.append(
                        "cut-count prefix violates the caption-language lock"
                    )
                segment_count = int(prefix.group(1))
                labels = list(re.finditer(r"shot([1-9][0-9]*): ", ordinary))
                if [int(match.group(1)) for match in labels] != list(
                    range(1, segment_count + 1)
                ):
                    errors.append(
                        "v7 shot labels must be exactly sequential shot1..shotN"
                    )
                if not labels or labels[0].start() != prefix.end():
                    errors.append(
                        "cut-count prefix must be followed immediately by shot1"
                    )
                if len(
                    re.findall(r"(?i)\bshot\s*[0-9]+\s*[:：]", ordinary)
                ) != segment_count:
                    errors.append("v7 prompt contains malformed or extra shot labels")
                segments = (
                    [
                        ordinary[
                            match.end() : (
                                labels[index + 1].start()
                                if index + 1 < len(labels)
                                else len(ordinary)
                            )
                        ]
                        for index, match in enumerate(labels)
                    ]
                    if labels
                    else [ordinary]
                )

        semantic_contract: dict[str, Any] | None = None
        semantic_validation: dict[str, Any] | None = None
        try:
            semantic_contract = _validate_semantic_contract_shape(
                self.semantic_contract
                or build_semantic_preservation_contract(raw_prompt),
                raw_prompt=raw_prompt,
            )
            semantic_validation = _validate_semantic_preservation(
                text,
                raw_prompt=raw_prompt,
                contract=semantic_contract,
            )
        except V7PromptValidationError as exc:
            semantic_error_message = str(exc)
            errors.append(semantic_error_message)
            structured_violations.extend(exc.violations)

        active = _DECLARATIONS[language]
        opposite = _DECLARATIONS[
            "Simplified Chinese" if language == "English" else "English"
        ]
        for declaration in opposite.values():
            if declaration in text:
                errors.append(
                    "v7 prompt mixes fixed declarations from both languages"
                )
                break
        exact_ending = active["no_ocr"] + " " + active["closing"]
        for index, segment in enumerate(segments, start=1):
            if segment.count(active["no_ocr"]) != 1:
                errors.append(
                    f"segment {index} requires exactly one no-OCR declaration"
                )
            if segment.count(active["closing"]) != 1:
                errors.append(
                    f"segment {index} requires exactly one no-subtitle closing"
                )
            if not segment.strip().endswith(exact_ending):
                errors.append(
                    f"segment {index} must end with adjacent no-OCR and no-subtitle declarations"
                )

        dialogue_mode = (
            semantic_contract["dialogue"]["mode"]
            if semantic_contract is not None
            else "required_unspecified"
        )
        silent = dialogue_mode == "forbidden"
        speaking_count = text.count(active["speaking"])
        silent_count = text.count(active["silent"])
        silent_shot_count = text.count(active["silent_shot"])
        if silent:
            if (speaking_count, silent_count, silent_shot_count) != (0, 1, 1):
                errors.append(
                    "silent v7 prompt requires exactly one silence and silent-shot declaration"
                )
        else:
            if (speaking_count, silent_count, silent_shot_count) != (1, 0, 0):
                errors.append(
                    "speaking v7 prompt requires exactly one written-dialogue vocal declaration"
                )
            speech_pattern = (
                r"ID_[A-Z][A-Z0-9_]* says\s*[,，]?\s*[\"“]"
                if language == "English"
                else r"ID_[A-Z][A-Z0-9_]*说[，,]?\s*[\"“]"
            )
            if re.search(speech_pattern, text) is None:
                errors.append(
                    "speaking v7 prompt requires quoted dialogue from a canonical ID"
                )

        person_ids = sorted(
            set(
                re.findall(
                    r"(?<![A-Za-z0-9_])ID_[A-Z][A-Z0-9_]*(?![A-Za-z0-9_])",
                    text,
                )
            )
        )
        if language == "English":
            identity = lambda item: f"{item} is "
            clothing = lambda item: f"{item} wears "
        else:
            identity = lambda item: f"{item}是"
            clothing = lambda item: f"{item}穿着"
        for person_id in person_ids:
            if text.count(identity(person_id)) != 1:
                errors.append(
                    f"v7 prompt requires exactly one identity anchor for {person_id}"
                )
            if text.count(clothing(person_id)) != 1:
                errors.append(
                    f"v7 prompt requires exactly one clothing anchor for {person_id}"
                )

        if errors:
            structured_violations.extend(
                _format_violation(message)
                for message in errors
                if message != semantic_error_message
            )
            raise V7PromptValidationError(
                "; ".join(errors),
                violations=structured_violations,
            )
        assert semantic_validation is not None

        return {
            "validation_id": self.validation_id,
            "caption_language": language,
            "condition_effect": condition_effect,
            "explicit_no_cut": no_cut,
            "explicit_silent": silent,
            "character_count": len(text),
            "internal_segment_count": segment_count,
            "person_ids": person_ids,
            "semantic_preservation": semantic_validation,
        }


__all__ = [
    "DirectorV7PromptValidator",
    "V7PromptValidationError",
    "V7_VALIDATOR_ID",
    "V7_SEMANTIC_CONTRACT_VERSION",
    "build_semantic_preservation_contract",
    "infer_caption_language",
]
