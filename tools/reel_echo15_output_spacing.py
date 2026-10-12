"""Audited fixed-footer spacing for the independent Echo production backend."""

from __future__ import annotations

import re

from storymem_adapter import prompt_engineering
from storymem_adapter.models import sha256_text


VERSION = "reel.echo15.fixed-footer-spacing.v1"
ORIGINAL_NORMALIZER = prompt_engineering._canonicalize_provider_output
FOOTERS = (
    ("画面任何位置均不出现OCR或可读文字。", "画面中没有屏幕文字或字幕。"),
    ("No OCR or readable text appears anywhere in the frame.", "No on-screen text or subtitles."),
)


def canonicalize_echo15_output(text):
    normalized, upstream = ORIGINAL_NORMALIZER(text)
    insertions = 0
    for no_ocr, closing in FOOTERS:
        pattern = re.escape(no_ocr + closing) + r"(?=\s*(?:shot[1-9][0-9]*: |$))"
        normalized, count = re.subn(pattern, no_ocr + " " + closing, normalized)
        insertions += count
    if not insertions:
        return normalized, upstream
    return normalized, {
        "method": VERSION,
        "changed": normalized != text,
        "raw_text_sha256": sha256_text(text),
        "normalized_text_sha256": sha256_text(normalized),
        "footer_separator_insertions": insertions,
        "upstream": upstream,
    }


def install_echo15_output_spacing():
    if prompt_engineering._canonicalize_provider_output not in (ORIGINAL_NORMALIZER, canonicalize_echo15_output):
        raise RuntimeError("Unexpected PE output normalizer; refusing to replace it")
    prompt_engineering._canonicalize_provider_output = canonicalize_echo15_output
