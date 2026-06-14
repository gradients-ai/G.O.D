"""Tokenizer-backed TokenCounter so memory slot budgets are real model tokens.

memory.py stays dependency-free (whitespace counter, fine for tests and as a
fallback). Transformers is imported lazily so pure tests and callers can still
use the whitespace fallback when optional tokenizer dependencies are unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from core.pvp.memory import TokenCounter
from core.pvp.memory import WhitespaceTokenCounter


logger = logging.getLogger(__name__)
AutoTokenizer = None


def _get_auto_tokenizer():
    global AutoTokenizer
    if AutoTokenizer is None:
        from transformers import AutoTokenizer as _AutoTokenizer

        AutoTokenizer = _AutoTokenizer
    return AutoTokenizer


class HFTokenCounter:
    """Count and truncate text in real model tokens via a HuggingFace tokenizer."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, max_tokens: int, keep: Literal["head", "tail"]) -> str:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        kept = ids[:max_tokens] if keep == "head" else ids[len(ids) - max_tokens:]
        return self._tokenizer.decode(kept, skip_special_tokens=True)


def load_token_counter(model_repo: str) -> TokenCounter:
    """Build a tokenizer-backed counter for model_repo; whitespace fallback otherwise.

    Ids that aren't a HuggingFace repo or a local path (test sentinels, Claude
    model ids, …) skip the load entirely so callers never pay a hub lookup, and
    any genuine load failure (gated repo, no network) falls back gracefully.
    """
    if "/" not in model_repo and not os.path.exists(model_repo):
        return WhitespaceTokenCounter()
    try:
        return HFTokenCounter(_get_auto_tokenizer().from_pretrained(model_repo))
    except Exception as exc:
        logger.warning("Tokenizer load failed for %r (%s); using whitespace counter", model_repo, exc)
        return WhitespaceTokenCounter()
