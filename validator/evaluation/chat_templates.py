"""Chat template resolution and the transformers adapter for the chat labelling rule.

Template names mean what they meant in the axolotl-based eval: `tokenizer_default` (also when the task has no name) is
the tokenizer's own template, `chatml` is the ChatML template axolotl bundled (copied byte-identical into
`validator/assets/chat_templates/chatml.jinja`), `tokenizer_default_fallback_<name>` prefers the tokenizer's template and
falls back to the bundled one. Only `chatml` is bundled; other axolotl template names are rejected rather than guessed.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from validator import assets


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


TOKENIZER_DEFAULT_TEMPLATE = "tokenizer_default"
TOKENIZER_DEFAULT_FALLBACK_PREFIX = "tokenizer_default_fallback_"
CHATML_TEMPLATE = "chatml"
BUNDLED_TEMPLATES_DIR = Path(assets.__file__).parent / "chat_templates"
BUNDLED_TEMPLATE_NAMES = frozenset({CHATML_TEMPLATE})


@cache
def bundled_chat_template(name: str) -> str:
    if name not in BUNDLED_TEMPLATE_NAMES:
        raise ValueError(f"Chat template {name!r} is not bundled (bundled: {sorted(BUNDLED_TEMPLATE_NAMES)})")
    return (BUNDLED_TEMPLATES_DIR / f"{name}.jinja").read_text(encoding="utf-8")


def tokenizer_chat_template(tokenizer: PreTrainedTokenizerBase) -> str | None:
    """The tokenizer's own template string.

    A tokenizer may carry named templates (`{"default": ..., "tool_use": ...}`, e.g. Hermes-2-Theta); transformers renders
    the `"default"` one when no template is passed, so that is the one used. (axolotl passed the dict straight back into
    `apply_chat_template`, which raised.)
    """
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        if "default" not in template:
            raise ValueError(f"Tokenizer has named chat templates {sorted(template)} but no 'default' one")
        return template["default"]
    return template


def resolve_chat_template(name: str | None, tokenizer: PreTrainedTokenizerBase) -> str:
    """The jinja chat template a task's `chat_template` name selects for this tokenizer."""
    choice = name or TOKENIZER_DEFAULT_TEMPLATE
    tokenizer_template = tokenizer_chat_template(tokenizer)
    if choice == TOKENIZER_DEFAULT_TEMPLATE:
        if not tokenizer_template:
            raise ValueError(f"Chat template {TOKENIZER_DEFAULT_TEMPLATE!r} requested but the tokenizer has no chat_template")
        return tokenizer_template
    if choice.startswith(TOKENIZER_DEFAULT_FALLBACK_PREFIX):
        if tokenizer_template:
            return tokenizer_template
        choice = choice[len(TOKENIZER_DEFAULT_FALLBACK_PREFIX) :]
    return bundled_chat_template(choice)


class HfChatTokenizer:
    """`chat_labels.ChatTokenizer` over a transformers tokenizer and a resolved template string."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, chat_template: str):
        if not getattr(tokenizer, "is_fast", False):
            raise TypeError(
                f"Chat labelling needs a fast tokenizer for character offsets; got {type(tokenizer).__name__} (is_fast=False)"
            )
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template=self.chat_template,
        )

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(encoded["input_ids"]), [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
