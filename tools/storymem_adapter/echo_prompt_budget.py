"""Exact, non-truncating admission against Echo's Gemma text encoder."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

from storymem_teacher.validation import PromptValidator

from .v7_validation import V7PromptValidationError

MAX_PROMPT_TOKENS = 1024


class EchoPromptBudget:
    def __init__(self, token_count: Callable[[str], int], tokenizer_sha256: str):
        self.token_count = token_count
        self.tokenizer_sha256 = tokenizer_sha256

    @property
    def contract(self) -> dict:
        return {
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "add_special_tokens": True,
            "truncation": False,
            "tokenizer_profile_sha256": self.tokenizer_sha256,
        }

    def validate(self, text: str) -> dict:
        count = self.token_count(text)
        if count > MAX_PROMPT_TOKENS:
            raise V7PromptValidationError(
                f"Echo Gemma prompt has {count} tokens, exceeding {MAX_PROMPT_TOKENS}. "
                "The complete caption must fit the 1024-token budget from its initial "
                "generation, including special tokens, prefixes and declarations. "
                "This output is rejected; no compression or truncation is allowed.",
                violations=[{
                    "code": "length.encoder_token_limit",
                    "actual_count": count,
                    "expected_maximum": MAX_PROMPT_TOKENS,
                }],
            )
        return {**self.contract, "token_count": count}


@lru_cache(maxsize=4)
def _load_budget(path: Path) -> EchoPromptBudget:
    validator = PromptValidator(path)
    validator.preload()
    return EchoPromptBudget(validator.token_count, validator.tokenizer_profile_sha256)


def get_echo_prompt_budget() -> EchoPromptBudget:
    default = Path(__file__).resolve().parents[2] / "checkpoints" / "gemma-3-12b"
    path = Path(os.getenv("STORYMEM_ECHO_TOKENIZER_PATH", str(default))).expanduser().resolve()
    return _load_budget(path)
