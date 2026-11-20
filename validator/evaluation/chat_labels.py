"""Which tokens of a chat conversation the text eval scores.

This module is the single owner of the chat labelling rule. It is deliberately self-contained (stdlib + pydantic only,
no transformers import, the tokenizer is injected through the `ChatTokenizer` protocol) because the exact same file is
copied verbatim into the miner repos (`scripts/chat_labels.py`) and is meant to be adopted by the quant eval: training
masks and eval masks must agree byte for byte. Keep it importable without transformers and do not add G.O.D imports.

The rule, for every assistant turn i of messages M (after role mapping), with F = render(M):

* the turn's scored text runs from the end of the generation prompt `render(M[:i], add_generation_prompt=True)` (when
  that prompt is a prefix of F and ends before the content) or else from the content start, through the content and the
  template's own turn-end string (`<|im_end|>\\n`, `<end_of_turn>\\n`, `</s>`, ...). An assistant turn with no message
  before it (i = 0) has no prompt to render - real templates index `messages[0]` and raise on an empty list - so its
  span starts at the content;
* the content span is located by re-rendering with the content replaced by a sentinel (common prefix / suffix), both in
  the full render F and in the "as last turn" render E = render(M[:i+1]); the turn-end string is read from E and must
  appear right after the content in F.

Why not "content + first eos within 3 tokens" (axolotl): on base tokenizers the eos is not the template's turn end, so
the end-of-turn token was never scored; and templates that decorate only the last assistant turn (Qwen3's empty think
block) shifted every earlier turn's label span. Locating the span in the full render fixes both.

Input ids are exactly `encode(F, add_special_tokens=False)` - identical to `apply_chat_template(tokenize=True)`.
"""

import json
from enum import Enum
from typing import Protocol

from pydantic import BaseModel
from pydantic import ConfigDict


IGNORE_INDEX = -100
# Invisible separators around a token that never occurs in real data; replaces one assistant turn's content so a
# common prefix / suffix against the real render isolates exactly where the template placed that content.
SENTINEL = "⁣GODSENTINEL⁣"
ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"


class ChatMessage(BaseModel):
    """One conversation turn as handed to the chat template.

    `content=None` is dropped from the rendered dict (axolotl omitted None-valued properties too), numbers are coerced to
    strings so a numeric answer column still renders.
    """

    model_config = ConfigDict(coerce_numbers_to_str=True)

    role: str
    content: str | None = None

    def template_dict(self) -> dict[str, str]:
        return self.model_dump(exclude_none=True)


class ChatTokenizer(Protocol):
    """What the labelling rule needs from a tokenizer + chat template pair."""

    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        """The chat template applied to `messages` (no tokenisation)."""
        ...

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """`text` tokenised without special tokens, with each token's [start, end) character range in `text`."""
        ...


class ChatLabelSkip(str, Enum):
    """Why a row yields no labelled example."""

    EMPTY = "empty"  # no assistant turn, or nothing to score
    RENDER = "render"  # the template does not render this row in a way the rule can locate turns in


class TurnSpan(BaseModel):
    """One assistant turn's scored tokens: input_ids[start:end]."""

    turn_index: int
    start: int
    end: int
    start_from_prompt: bool  # True when the span starts right after the generation prompt, False when at the content
    straddle_start: bool  # the first scored token also covers text before the span (merged across the boundary)
    straddle_end: bool  # the last scored token also covers text after the span


class ChatLabelled(BaseModel):
    input_ids: list[int]
    labels: list[int]
    spans: list[TurnSpan]

    @property
    def straddles(self) -> int:
        return sum(int(span.straddle_start) + int(span.straddle_end) for span in self.spans)


def chat_role_map(user_reference: str | None, assistant_reference: str | None) -> dict[str, str]:
    """Dataset role names -> template role names. Roles not in the map (system, tool, ...) pass through unchanged.

    When both references are the same string the user mapping wins, as it did in axolotl's role table.
    """
    role_map: dict[str, str] = {}
    if assistant_reference is not None:
        role_map[assistant_reference] = ASSISTANT_ROLE
    if user_reference is not None:
        role_map[user_reference] = USER_ROLE
    return role_map


def chat_messages(
    row: dict,
    chat_column: str,
    role_field: str,
    content_field: str,
    role_map: dict[str, str],
) -> list[ChatMessage]:
    """The row's conversation as template messages.

    A JSON-encoded string column is decoded (axolotl accepted those). Malformed rows raise `ValueError` naming the shape
    problem rather than failing somewhere inside the template.
    """
    if chat_column not in row:
        raise ValueError(f"Chat row has no {chat_column!r} column (columns: {sorted(row)})")
    raw_messages = row[chat_column]
    if raw_messages is None:
        raise ValueError(f"Chat row column {chat_column!r} is null")
    if isinstance(raw_messages, str):
        try:
            raw_messages = json.loads(raw_messages)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Chat row column {chat_column!r} is a string that is not JSON: {exc}") from exc
    if not isinstance(raw_messages, list):
        raise ValueError(f"Chat row column {chat_column!r} must be a list of messages, got {type(raw_messages).__name__}")

    messages: list[ChatMessage] = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict):
            raise ValueError(f"Chat message {index} must be a dict, got {type(raw_message).__name__}")
        role = raw_message.get(role_field)
        if role is None:
            raise ValueError(f"Chat message {index} has no {role_field!r} role field (keys: {sorted(raw_message)})")
        messages.append(ChatMessage(role=role_map.get(role, role), content=raw_message.get(content_field)))
    return messages


def label_chat(tokenizer: ChatTokenizer, messages: list[ChatMessage]) -> ChatLabelled | ChatLabelSkip:
    """Input ids and labels for one conversation, every assistant turn scored per the module rule.

    Rendering the conversation itself may raise (a template rejecting the row); the extra renders the rule needs are
    the rule's own business, so a failure there is reported as `ChatLabelSkip.RENDER` instead.
    """
    assistant_turns = [index for index, message in enumerate(messages) if message.role == ASSISTANT_ROLE]
    if not assistant_turns:
        return ChatLabelSkip.EMPTY

    rendered = [message.template_dict() for message in messages]
    full_text = tokenizer.render(rendered, add_generation_prompt=False)
    input_ids, offsets = tokenizer.encode_with_offsets(full_text)
    if len(input_ids) != len(offsets):
        raise ValueError(f"Tokenizer returned {len(input_ids)} ids but {len(offsets)} offsets")

    labels = [IGNORE_INDEX] * len(input_ids)
    spans: list[TurnSpan] = []
    for turn_index in assistant_turns:
        try:
            char_span = _turn_char_span(tokenizer, rendered, turn_index, full_text)
        except Exception:  # noqa: BLE001 - any failure of the rule's auxiliary renders is a per-row skip
            return ChatLabelSkip.RENDER
        if char_span is None:
            return ChatLabelSkip.RENDER
        start_char, end_char, start_from_prompt = char_span
        span = _token_span(offsets, start_char, end_char, turn_index, start_from_prompt)
        if span is None:
            continue
        for position in range(span.start, span.end):
            labels[position] = input_ids[position]
        spans.append(span)

    if not spans:
        return ChatLabelSkip.EMPTY
    return ChatLabelled(input_ids=list(input_ids), labels=labels, spans=spans)


def _turn_char_span(
    tokenizer: ChatTokenizer,
    rendered: list[dict],
    turn_index: int,
    full_text: str,
) -> tuple[int, int, bool] | None:
    """[start, end) characters of `full_text` scored for one assistant turn, and whether it starts at the prompt end."""
    with_sentinel = rendered[:turn_index] + [{**rendered[turn_index], "content": SENTINEL}] + rendered[turn_index + 1 :]
    # Nothing precedes the first message, so there is no generation prompt to start from (see the module docstring).
    prompt_text = tokenizer.render(rendered[:turn_index], add_generation_prompt=True) if turn_index else None
    as_last_text = tokenizer.render(rendered[: turn_index + 1], add_generation_prompt=False)
    full_sentinel = tokenizer.render(with_sentinel, add_generation_prompt=False)
    as_last_sentinel = tokenizer.render(with_sentinel[: turn_index + 1], add_generation_prompt=False)

    content_start_full, content_end_full = _content_span(full_text, full_sentinel)
    content_start_last, content_end_last = _content_span(as_last_text, as_last_sentinel)
    if content_start_full > content_end_full or content_start_last > content_end_last:
        return None
    # The content itself may differ between the two renders (Qwen3 / Nanbeige strip a leading newline only on the
    # as-last path), so only the turn-end string is carried over from the as-last render.
    turn_end = as_last_text[content_end_last:]
    if full_text[content_end_full : content_end_full + len(turn_end)] != turn_end:
        return None
    end_char = content_end_full + len(turn_end)
    if prompt_text is not None and full_text.startswith(prompt_text) and len(prompt_text) <= content_start_full:
        return len(prompt_text), end_char, True
    return content_start_full, end_char, False


def _content_span(text: str, sentinel_text: str) -> tuple[int, int]:
    """Where `text` differs from the same render with one turn's content replaced by the sentinel."""
    prefix = _common_prefix_len(text, sentinel_text)
    suffix = _common_prefix_len(text[::-1], sentinel_text[::-1])
    return prefix, len(text) - suffix


def _common_prefix_len(first: str, second: str) -> int:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return length


def _token_span(
    offsets: list[tuple[int, int]],
    start_char: int,
    end_char: int,
    turn_index: int,
    start_from_prompt: bool,
) -> TurnSpan | None:
    """Contiguous tokens from the first to the last one whose character range overlaps [start_char, end_char).

    Zero-width tokens never decide where a span starts or ends; one between two scored tokens is scored with them.
    A token reaching across a span boundary (BPE merging the prompt's newline with a content newline) is scored and
    reported as a straddle.
    """
    overlapping = [
        position
        for position, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start_char and token_start < end_char
    ]
    if not overlapping:
        return None
    first, last = overlapping[0], overlapping[-1]
    return TurnSpan(
        turn_index=turn_index,
        start=first,
        end=last + 1,
        start_from_prompt=start_from_prompt,
        straddle_start=offsets[first][0] < start_char,
        straddle_end=offsets[last][1] > end_char,
    )
