from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path
from typing import Any, Callable


class PromptValidationError(ValueError):
    pass


TOKENIZER_PROFILE_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

# ``transformers`` exposes AutoTokenizer through a process-global lazy module.
# In transformers 4.57.x, resolving that symbol (and the concrete tokenizer
# class selected by ``from_pretrained``) concurrently from several fresh
# threads can observe a partially imported module and raise a spurious
# ``ImportError``.  A canonical collection creates one PromptValidator per
# story, so its first prompt validations otherwise hit precisely that race.
#
# Keep one immutable/read-only tokenizer per canonical local path.  The fast
# path deliberately stays outside the lock; construction and publication are
# serialized, and a failed construction is never inserted into the cache.
_TOKENIZER_CACHE: dict[Path, Any] = {}
_TOKENIZER_LOAD_LOCK = threading.RLock()


def _cached_tokenizer(tokenizer_path: Path) -> Any:
    cached = _TOKENIZER_CACHE.get(tokenizer_path)
    if cached is not None:
        return cached

    with _TOKENIZER_LOAD_LOCK:
        cached = _TOKENIZER_CACHE.get(tokenizer_path)
        if cached is not None:
            return cached

        # Import inside the same lock as from_pretrained: AutoTokenizer and
        # the model-specific tokenizer are both resolved lazily by
        # transformers.  Publish only after the complete load succeeds.
        from transformers import AutoTokenizer

        loaded = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        _TOKENIZER_CACHE[tokenizer_path] = loaded
        return loaded

_TAIL_FRAME_PREAMBLES = {
    "English": (
        "Use the current image as the first frame of the video, and continue "
        "naturally from the characters, objects, environment, composition, camera "
        "position, lighting, and action state already shown in it."
    ),
    "Simplified Chinese": (
        "以当前图片作为视频首帧，并基于首帧中已有的人物、物体、环境、构图、机位、光线和动作状态自然延续。"
    ),
}


def tokenizer_profile_sha256(tokenizer_path: str | Path) -> str:
    """Fingerprint the exact tokenizer files used for prompt admission.

    The same byte-level recipe is implemented by the Echo server health
    contract.  Missing tokenizer metadata is an error, rather than returning
    the SHA-256 of an empty stream and falsely claiming compatibility.
    """

    root = Path(tokenizer_path).expanduser().resolve()
    if not root.is_dir():
        raise PromptValidationError(f"tokenizer path is not a directory: {root}")
    digest = hashlib.sha256()
    found = 0
    for name in TOKENIZER_PROFILE_FILES:
        path = root / name
        if not path.is_file():
            continue
        found += 1
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    if not found:
        raise PromptValidationError(
            f"tokenizer path contains none of the profile files: {root}"
        )
    return "sha256:" + digest.hexdigest()


class PromptValidator:
    """Deterministic validation only; this class never edits prompt text."""

    def __init__(
        self,
        tokenizer_path: str | Path | None = None,
        *,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.tokenizer_path = (
            Path(tokenizer_path).expanduser().resolve() if tokenizer_path else None
        )
        self._token_counter = token_counter
        self._tokenizer: Any = None

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        if self.tokenizer_path is None:
            raise PromptValidationError(
                "video prompt validation requires the exact Echo tokenizer path"
            )
        self._tokenizer = _cached_tokenizer(self.tokenizer_path)
        return self._tokenizer

    def preload(self) -> int:
        """Load and exercise the exact tokenizer before worker threads start."""

        return self.token_count("StoryMem tokenizer startup validation probe.")

    @property
    def tokenizer_profile_sha256(self) -> str | None:
        if self.tokenizer_path is None:
            return None
        return tokenizer_profile_sha256(self.tokenizer_path)

    def token_count(self, text: str) -> int:
        if self._token_counter is not None:
            return int(self._token_counter(text))
        tokenizer = self._load_tokenizer()
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, list):
            raise PromptValidationError("tokenizer returned no input_ids")
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)

    @staticmethod
    def _explicit_no_cut(raw_user_request: str) -> bool:
        return bool(
            re.search(
                r"(?ix)(?:"
                r"\b(?:no|without)\s+(?:internal\s+)?cuts?\b|"
                r"\b(?:single|one)\s+(?:continuous\s+)?take\b|"
                r"一镜到底|单一连续镜头|单镜头连续拍摄|不要(?:内部)?切镜|无(?:内部)?剪辑"
                r")",
                raw_user_request,
            )
        )

    @staticmethod
    def _validate_director_v7_format(
        text: str,
        *,
        raw_user_request: str,
        allowed_person_ids: set[str] | None,
        prompt_context: dict[str, Any] | None,
        expected_caption_language: str | None,
    ) -> dict[str, Any]:
        """Check the exact mechanically decidable parts of official v7.

        Cinematic quality and semantic faithfulness remain the VLM agent's
        responsibility.  This admission check intentionally covers only byte-
        level syntax that the official PE itself calls mandatory.
        """

        condition = (
            prompt_context.get("condition")
            if isinstance(prompt_context, dict)
            else None
        )
        relation = (
            condition.get("relation") if isinstance(condition, dict) else None
        )
        temporal_continuation = relation == "temporal_continuation"
        matched_tail_languages = [
            language
            for language, preamble in _TAIL_FRAME_PREAMBLES.items()
            if text.startswith(preamble)
        ]
        embedded_tail_languages = [
            language
            for language, preamble in _TAIL_FRAME_PREAMBLES.items()
            if preamble in text
        ]
        if temporal_continuation:
            if len(matched_tail_languages) != 1:
                raise PromptValidationError(
                    "official tail-frame continuation requires its exact language-matched "
                    "first-frame preamble at the start"
                )
            tail_language = matched_tail_languages[0]
            if sum(
                text.count(preamble)
                for preamble in _TAIL_FRAME_PREAMBLES.values()
            ) != 1:
                raise PromptValidationError(
                    "official tail-frame continuation requires exactly one first-frame preamble"
                )
            ordinary_text = text[len(_TAIL_FRAME_PREAMBLES[tail_language]) :]
            if ordinary_text.startswith(" "):
                ordinary_text = ordinary_text[1:]
            if not ordinary_text or ordinary_text.startswith(" "):
                raise PromptValidationError(
                    "official tail-frame preamble must be followed by the complete ordinary prompt"
                )
        else:
            if embedded_tail_languages:
                raise PromptValidationError(
                    "official tail-frame preamble is allowed only for temporal_continuation"
                )
            tail_language = None
            ordinary_text = text

        english_prefix = re.match(r"^This video has ([23]) shots\. ", ordinary_text)
        chinese_prefix = re.match(r"^本视频包含([23])个镜头。", ordinary_text)
        if english_prefix and chinese_prefix:  # pragma: no cover - impossible by construction
            raise PromptValidationError("video prompt has conflicting language prefixes")

        declarations = {
            "English": {
                "no_ocr": "No OCR or readable text appears anywhere in the frame.",
                "closing": "No on-screen text or subtitles.",
                "speaking": (
                    "Only the written spoken dialogue is voiced; no additional "
                    "human vocalizations are audible."
                ),
                "silent": (
                    "Only environmental ambience is audible; no human vocal sounds are present."
                ),
                "silent_shot": "No character speaks in this shot.",
            },
            "Simplified Chinese": {
                "no_ocr": "画面任何位置均不出现OCR或可读文字。",
                "closing": "画面中没有屏幕文字或字幕。",
                "speaking": "只有写出的对白被发声，不出现其他人声。",
                "silent": "只有环境声，不出现任何人声。",
                "silent_shot": "本镜头中没有角色说话。",
            },
        }
        prefix = english_prefix or chinese_prefix
        if prefix is not None:
            language = "English" if english_prefix else "Simplified Chinese"
            segment_count = int(prefix.group(1))
            labels = list(re.finditer(r"shot([1-9][0-9]*): ", ordinary_text))
            if [int(item.group(1)) for item in labels] != list(
                range(1, segment_count + 1)
            ):
                raise PromptValidationError(
                    "official v7 shot labels must be exactly sequential shot1..shotN"
                )
            if not labels or labels[0].start() != prefix.end():
                raise PromptValidationError(
                    "official v7 count prefix must be followed immediately by shot1"
                )
            loose_labels = re.findall(
                r"(?i)\bshot\s*[0-9]+\s*[:：]", ordinary_text
            )
            if len(loose_labels) != segment_count:
                raise PromptValidationError(
                    "official v7 prompt contains malformed or extra shot labels"
                )
            segments = [
                ordinary_text[
                    item.end() : (
                        labels[index + 1].start()
                        if index + 1 < len(labels)
                        else len(ordinary_text)
                    )
                ]
                for index, item in enumerate(labels)
            ]
        else:
            if re.search(r"(?i)\bshot\s*[0-9]+\s*[:：]", ordinary_text):
                raise PromptValidationError(
                    "official v7 prompt has shot labels without the exact count prefix"
                )
            if not PromptValidator._explicit_no_cut(raw_user_request):
                raise PromptValidationError(
                    "official v7 prompt requires 2-3 internal segments unless USER explicitly requested no cuts"
                )
            english_markers = sum(
                ordinary_text.count(declarations["English"][key])
                for key in ("no_ocr", "closing")
            )
            chinese_markers = sum(
                ordinary_text.count(declarations["Simplified Chinese"][key])
                for key in ("no_ocr", "closing")
            )
            if bool(english_markers) == bool(chinese_markers):
                raise PromptValidationError(
                    "official v7 no-cut prompt must use one language-matched declaration pair"
                )
            language = "English" if english_markers else "Simplified Chinese"
            segment_count = 1
            segments = [ordinary_text]

        if tail_language is not None and tail_language != language:
            raise PromptValidationError(
                "official tail-frame preamble language differs from the prompt language"
            )
        if expected_caption_language is not None:
            if expected_caption_language not in declarations:
                raise PromptValidationError(
                    "expected_caption_language must be English or Simplified Chinese"
                )
            if language != expected_caption_language:
                raise PromptValidationError(
                    "video prompt language differs from the locked story caption language"
                )

        active = declarations[language]
        opposite = declarations[
            "Simplified Chinese" if language == "English" else "English"
        ]
        for value in opposite.values():
            if value in text:
                raise PromptValidationError(
                    "official v7 prompt mixes fixed declarations from both caption languages"
                )
        for index, segment in enumerate(segments, start=1):
            if segment.count(active["no_ocr"]) != 1 or segment.count(active["closing"]) != 1:
                raise PromptValidationError(
                    f"official v7 segment {index} requires exactly one no-OCR and closing declaration"
                )
            tail = segment.rsplit(active["no_ocr"], 1)[1]
            if tail.strip() != active["closing"]:
                raise PromptValidationError(
                    f"official v7 segment {index} must end with no-OCR immediately followed by its closing sentence"
                )

        vocal_count = text.count(active["speaking"]) + text.count(active["silent"])
        if vocal_count != 1:
            raise PromptValidationError(
                "official v7 prompt requires exactly one language-matched final vocal-output declaration"
            )

        if language == "English":
            raw_silent_count = len(
                re.findall(
                    r"(?:^|\s)No spoken dialogue\.(?=\s|$)", raw_user_request
                )
            )
            raw_dialogue = re.findall(
                r'(?:^|\s)Spoken dialogue:\s*["“](.+?)["”]',
                raw_user_request,
            )
        else:
            raw_silent_count = raw_user_request.count("无对白。")
            raw_dialogue = re.findall(
                r"对白[：:]\s*[“\"](.+?)[”\"]",
                raw_user_request,
            )
        if expected_caption_language is not None and (
            (raw_silent_count > 0) == bool(raw_dialogue)
            or raw_silent_count > 1
            or len(raw_dialogue) > 1
        ):
            raise PromptValidationError(
                "canonical raw request must contain exactly one language-matched "
                "explicit speech intent"
            )
        if raw_silent_count:
            if text.count(active["silent"]) != 1 or active["speaking"] in text:
                raise PromptValidationError(
                    "video prompt vocal output conflicts with the raw no-dialogue intent"
                )
            if text.count(active["silent_shot"]) != 1:
                raise PromptValidationError(
                    "official v7 silent prompt requires exactly one silent-shot declaration"
                )
        elif raw_dialogue:
            if text.count(active["speaking"]) != 1 or active["silent"] in text:
                raise PromptValidationError(
                    "video prompt vocal output conflicts with the raw spoken-dialogue intent"
                )
            if active["silent_shot"] in text:
                raise PromptValidationError(
                    "a speaking video prompt must not contain the silent-shot declaration"
                )
            if raw_dialogue[0] not in text:
                raise PromptValidationError(
                    "video prompt must preserve the complete story-planned spoken dialogue"
                )

        entity_tags = sorted(set(re.findall(r"(?<![A-Za-z0-9_])ENTITY_[1-9][0-9]*(?![A-Za-z0-9_])", text)))
        if entity_tags:
            raise PromptValidationError(
                "non-human ENTITY_N registry tags must not appear in an official v7 generator prompt: "
                + ", ".join(entity_tags)
            )
        used_person_ids = set(
            re.findall(r"(?<![A-Za-z0-9_])ID_[A-Z][A-Z0-9_]*(?![A-Za-z0-9_])", text)
        )
        if len(used_person_ids) > 2:
            raise PromptValidationError(
                "official v7 prompt may contain at most two active human IDs"
            )
        if raw_dialogue and not used_person_ids:
            raise PromptValidationError(
                "a speaking video prompt requires an active declared human ID"
            )
        if allowed_person_ids is not None:
            unknown_ids = sorted(used_person_ids - allowed_person_ids)
            if unknown_ids:
                raise PromptValidationError(
                    "official v7 prompt uses IDs not declared as human people: "
                    + ", ".join(unknown_ids)
                )
            missing_ids = sorted(allowed_person_ids - used_person_ids)
            if missing_ids:
                raise PromptValidationError(
                    "official v7 prompt omits active visible human IDs: "
                    + ", ".join(missing_ids)
                )

        # The official writer makes the ID token operational through formal
        # identity/clothing anchors; merely mentioning ``ID_A`` somewhere in
        # an action sentence is not a valid cast binding.  Story PE defines
        # ``allowed_person_ids`` as the exact visible-person set for this shot,
        # so the checks above and below close both accidental omission and ID
        # invention without trying to judge the semantic prose.
        if language == "English":
            identity_anchor = lambda subject_id: f"{subject_id} is "
            clothing_anchor = lambda subject_id: f"{subject_id} wears "
            voice_anchor = lambda subject_id: f"{subject_id}'s voice is "
            speech_form = lambda subject_id: f"{subject_id} says"
        else:
            identity_anchor = lambda subject_id: f"{subject_id}是"
            clothing_anchor = lambda subject_id: f"{subject_id}穿着"
            voice_anchor = lambda subject_id: f"{subject_id}的声音是"
            speech_form = lambda subject_id: f"{subject_id}说"

        spoken_ids: set[str] = set()
        for subject_id in sorted(used_person_ids):
            if text.count(identity_anchor(subject_id)) != 1:
                raise PromptValidationError(
                    f"official v7 prompt requires exactly one identity anchor for {subject_id}"
                )
            if text.count(clothing_anchor(subject_id)) != 1:
                raise PromptValidationError(
                    f"official v7 prompt requires exactly one clothing anchor for {subject_id}"
                )
            speech_count = text.count(speech_form(subject_id))
            voice_count = text.count(voice_anchor(subject_id))
            if speech_count:
                spoken_ids.add(subject_id)
                if voice_count != 1:
                    raise PromptValidationError(
                        f"official v7 speaking character {subject_id} requires exactly one voice anchor"
                    )
            elif voice_count:
                raise PromptValidationError(
                    f"official v7 non-speaking character {subject_id} must not have a voice anchor"
                )
        if raw_dialogue and not spoken_ids:
            raise PromptValidationError(
                "official v7 spoken dialogue must use a canonical ID_X says form"
            )
        if raw_silent_count and spoken_ids:
            raise PromptValidationError(
                "official v7 silent prompt must not contain an ID_X says form"
            )
        return {
            "format_profile_id": "director.v7_cinematic_full",
            "caption_language": language,
            "internal_segment_count": segment_count,
            "used_person_ids": sorted(used_person_ids),
        }

    def validate(
        self,
        text: Any,
        *,
        target: str,
        contract: dict[str, Any],
        raw_user_request: str,
        allowed_person_ids: set[str] | None = None,
        prompt_context: dict[str, Any] | None = None,
        expected_caption_language: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(text, str) or not text:
            raise PromptValidationError("prompt text must be a non-empty string")
        if text != text.strip():
            raise PromptValidationError("prompt text must not contain leading or trailing whitespace")
        if "\n" in text or "\r" in text:
            raise PromptValidationError("prompt text must be one paragraph")
        if text == raw_user_request:
            raise PromptValidationError("raw USER text cannot be registered as a generator prompt")
        leak_patterns = {
            "internal_handle": re.compile(
                r"(?i)(?<![A-Za-z0-9])(?:"
                r"asset|audio_clip|cand(?:idate)?(?:_image|_video)?|frame|"
                r"imagegroup|videogroup|inspect|mem(?:ory)?|memupd|obs|"
                r"peskill|prompt|providerreq|registry|repr(?:esentation)?|"
                r"search|selection|shotreq|upload|use|view"
                r")_[A-Za-z0-9][A-Za-z0-9_.:-]*"
            ),
            "legacy_handle": re.compile(
                r"(?i)<(?:id|scene|obj|object|prop|style)_[1-9][0-9]*>"
            ),
            "url": re.compile(r"(?i)(?:https?|ftp)://|\bwww\."),
            "data_uri": re.compile(r"(?i)\bdata:[a-z0-9.+-]+/[a-z0-9.+-]+[;,]"),
            "local_path": re.compile(
                r"(?i)(?:^|[\s\"'(])(?:file://|~[/\\]|\.\.?[/\\]|"
                r"/(?:tmp|var|home|root|mnt|pfs|workspace)(?:[/\\]|$)|"
                r"[A-Z]:[/\\])"
            ),
            "numbered_reference": re.compile(
                r"(?ix)(?:"
                r"\breference[\s_-]*(?:image|picture|photo)[\s_-]*(?:\#\s*)?"
                r"(?:[1-9][0-9]*|[１-９][０-９]*)\b|"
                r"\b(?:image|picture|photo)[\s_-]*(?:no\.?\s*|number\s*|\#\s*)?"
                r"(?:[1-9][0-9]*|[１-９][０-９]*)\b|"
                r"\breference\s*\#?\s*(?:[1-9][0-9]*|[１-９][０-９]*)\b|"
                r"\b(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|"
                r"eighth|ninth|tenth|[1-9][0-9]*(?:st|nd|rd|th))\s+reference\s+"
                r"(?:image|picture|photo)\b|"
                r"(?<!构)图\s*(?:[1-9][0-9]*|[１-９][０-９]*|[一二三四五六七八九十百千]+)|"
                r"第\s*(?:[1-9][0-9]*|[１-９][０-９]*|[一二三四五六七八九十百千]+)"
                r"\s*张?\s*(?:参考)?(?:图像|图片|图)|"
                r"参考(?:图像|图片|图)\s*"
                r"(?:[1-9][0-9]*|[１-９][０-９]*|[一二三四五六七八九十百千]+)"
                r")"
            ),
        }
        found = [name for name, pattern in leak_patterns.items() if pattern.search(text)]
        if found:
            raise PromptValidationError(
                f"prompt leaks internal handles, paths, URLs, or numbered reference labels: {found}"
            )
        min_chars = contract.get("min_chars_inclusive")
        if min_chars is not None and len(text) < int(min_chars):
            raise PromptValidationError(
                f"prompt has {len(text)} characters; must be >= {int(min_chars)}"
            )
        max_chars = contract.get("max_chars_exclusive")
        if max_chars is not None and len(text) >= int(max_chars):
            raise PromptValidationError(
                f"prompt has {len(text)} characters; must be < {int(max_chars)}"
            )
        max_tokens = contract.get("max_tokens")
        token_count: int | None = None
        if max_tokens is not None:
            token_count = self.token_count(text)
            if token_count > int(max_tokens):
                raise PromptValidationError(
                    f"prompt has {token_count} tokens; must be <= {int(max_tokens)}"
                )
        format_validation: dict[str, Any] = {}
        format_profile_id = contract.get("format_profile_id")
        if format_profile_id == "director.v7_cinematic_full":
            if target != "video":
                raise PromptValidationError(
                    "director.v7_cinematic_full format is valid only for video prompts"
                )
            format_validation = self._validate_director_v7_format(
                text,
                raw_user_request=raw_user_request,
                allowed_person_ids=allowed_person_ids,
                prompt_context=prompt_context,
                expected_caption_language=expected_caption_language,
            )
        elif format_profile_id is not None:
            raise PromptValidationError(
                f"unsupported prompt format profile: {format_profile_id}"
            )
        return {
            "char_count": len(text),
            "token_count": token_count,
            "min_chars_inclusive": min_chars,
            "max_chars_exclusive": max_chars,
            "max_tokens": max_tokens,
            "tokenizer_profile_sha256": self.tokenizer_profile_sha256 if max_tokens is not None else None,
            **format_validation,
        }
