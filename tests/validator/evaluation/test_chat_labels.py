"""The chat labelling rule on fake templates that reproduce the shapes real chat templates take.

Only imports `chat_labels`, so the miners can copy these tests next to their copy of the module (only the import path changes).
"""

import ast
from pathlib import Path

import pytest

from validator.evaluation import chat_labels
from validator.evaluation.chat_labels import IGNORE_INDEX
from validator.evaluation.chat_labels import SENTINEL
from validator.evaluation.chat_labels import ChatLabelled
from validator.evaluation.chat_labels import ChatLabelSkip
from validator.evaluation.chat_labels import ChatMessage
from validator.evaluation.chat_labels import chat_messages
from validator.evaluation.chat_labels import chat_role_map
from validator.evaluation.chat_labels import label_chat


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK = "<think>\n\n</think>\n\n"
MERGED_NEWLINES_ID = 1_000_000


def _turn(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def _content(message: dict) -> str:
    return message.get("content", "")


def plain_template(messages: list[dict], add_generation_prompt: bool) -> str:
    text = "".join(_turn(m["role"], _content(m)) for m in messages)
    return text + (f"{IM_START}assistant\n" if add_generation_prompt else "")


def qwen3_like_template(messages: list[dict], add_generation_prompt: bool) -> str:
    """Empty think block and leading-newline strip only on a final assistant turn after the last user turn."""
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    parts = []
    for index, message in enumerate(messages):
        content = _content(message)
        if message["role"] == "assistant" and index > last_user and index == len(messages) - 1:
            content = THINK + content.lstrip("\n")
        parts.append(_turn(message["role"], content))
    return "".join(parts) + (f"{IM_START}assistant\n" if add_generation_prompt else "")


def kanana_like_template(messages: list[dict], add_generation_prompt: bool) -> str:
    """Think block in every generation prompt, rendered in a message only when it is the last one."""
    parts = []
    for index, message in enumerate(messages):
        content = _content(message)
        if message["role"] == "assistant" and index == len(messages) - 1:
            content = THINK + content
        parts.append(_turn(message["role"], content))
    return "".join(parts) + (f"{IM_START}assistant\n{THINK}" if add_generation_prompt else "")


def lfm_like_template(messages: list[dict], add_generation_prompt: bool) -> str:
    """Generation prompt ends with an opening think tag that never appears in a rendered message."""
    text = "".join(_turn(m["role"], _content(m)) for m in messages)
    return text + (f"{IM_START}assistant\n<think>" if add_generation_prompt else "")


def eos_on_last_template(messages: list[dict], add_generation_prompt: bool) -> str:
    """A final assistant turn ends differently from earlier ones - the rule cannot carry its turn end over."""
    parts = [_turn(m["role"], _content(m)) for m in messages]
    if messages and messages[-1]["role"] == "assistant" and not add_generation_prompt:
        parts[-1] = f"{IM_START}assistant\n{_content(messages[-1])}{IM_END}<|endoftext|>"
    return "".join(parts) + (f"{IM_START}assistant\n" if add_generation_prompt else "")


def raises_on_empty_template(messages: list[dict], add_generation_prompt: bool) -> str:
    if not messages:
        raise RuntimeError("template indexes messages[0]")
    return plain_template(messages, add_generation_prompt)


class FakeChatTokenizer:
    """One id per character (its code point); optionally fuses "\\n\\n" into one token like BPE vocabularies do."""

    def __init__(self, template, merge_double_newline: bool = False):
        self.template = template
        self.merge_double_newline = merge_double_newline
        self.render_calls = 0

    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        self.render_calls += 1
        return self.template(messages, add_generation_prompt)

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        position = 0
        while position < len(text):
            if self.merge_double_newline and text.startswith("\n\n", position):
                ids.append(MERGED_NEWLINES_ID)
                offsets.append((position, position + 2))
                position += 2
                continue
            ids.append(ord(text[position]))
            offsets.append((position, position + 1))
            position += 1
        return ids, offsets

    def decode(self, ids: list[int]) -> str:
        return "".join("\n\n" if token == MERGED_NEWLINES_ID else chr(token) for token in ids)


def _msgs(*pairs: tuple[str, str | None]) -> list[ChatMessage]:
    return [ChatMessage(role=role, content=content) for role, content in pairs]


TWO_TURNS = _msgs(
    ("user", "What is the capital of France?"),
    ("assistant", "Paris is the capital."),
    ("user", "And Germany?"),
    ("assistant", "Berlin."),
)


def _labelled(tokenizer: FakeChatTokenizer, messages: list[ChatMessage]) -> ChatLabelled:
    result = label_chat(tokenizer, messages)
    assert isinstance(result, ChatLabelled), result
    return result


def _scored_texts(tokenizer: FakeChatTokenizer, result: ChatLabelled) -> list[str]:
    return [tokenizer.decode(result.input_ids[span.start : span.end]) for span in result.spans]


def _assert_labels_follow_spans(result: ChatLabelled) -> None:
    in_span = {k for span in result.spans for k in range(span.start, span.end)}
    for position, (token, label) in enumerate(zip(result.input_ids, result.labels)):
        assert label == (token if position in in_span else IGNORE_INDEX)


def test_plain_template_scores_content_and_turn_end_of_every_assistant_turn():
    tokenizer = FakeChatTokenizer(plain_template)
    result = _labelled(tokenizer, TWO_TURNS)

    assert _scored_texts(tokenizer, result) == [f"Paris is the capital.{IM_END}\n", f"Berlin.{IM_END}\n"]
    assert [span.turn_index for span in result.spans] == [1, 3]
    assert all(span.start_from_prompt for span in result.spans)
    assert result.straddles == 0
    assert result.spans[-1].end == len(result.input_ids)
    _assert_labels_follow_spans(result)


def test_input_ids_are_the_plain_encoding_of_the_full_render():
    tokenizer = FakeChatTokenizer(plain_template)
    result = _labelled(tokenizer, TWO_TURNS)
    full = plain_template([m.template_dict() for m in TWO_TURNS], False)
    assert result.input_ids == tokenizer.encode_with_offsets(full)[0]


def test_qwen3_like_earlier_turn_scores_its_own_content_and_last_turn_scores_the_think_block():
    tokenizer = FakeChatTokenizer(qwen3_like_template)
    result = _labelled(tokenizer, TWO_TURNS)

    assert _scored_texts(tokenizer, result) == [f"Paris is the capital.{IM_END}\n", f"{THINK}Berlin.{IM_END}\n"]
    assert all(span.start_from_prompt for span in result.spans)
    _assert_labels_follow_spans(result)


def test_qwen3_like_leading_newline_strip_on_the_as_last_path_does_not_skip_the_row():
    tokenizer = FakeChatTokenizer(qwen3_like_template)
    messages = _msgs(("user", "Q1"), ("assistant", "\nA1"), ("user", "Q2"), ("assistant", "\nA2"))
    result = _labelled(tokenizer, messages)

    assert _scored_texts(tokenizer, result) == [f"\nA1{IM_END}\n", f"{THINK}A2{IM_END}\n"]


def test_kanana_like_generation_prompt_not_a_prefix_starts_earlier_turns_at_the_content():
    tokenizer = FakeChatTokenizer(kanana_like_template)
    result = _labelled(tokenizer, TWO_TURNS)

    # The last turn's think block is part of its own generation prompt, so it is prompt, not prediction.
    assert _scored_texts(tokenizer, result) == [f"Paris is the capital.{IM_END}\n", f"Berlin.{IM_END}\n"]
    assert [span.start_from_prompt for span in result.spans] == [False, True]


def test_lfm_like_extra_generation_suffix_starts_every_turn_at_the_content():
    tokenizer = FakeChatTokenizer(lfm_like_template)
    result = _labelled(tokenizer, TWO_TURNS)

    assert _scored_texts(tokenizer, result) == [f"Paris is the capital.{IM_END}\n", f"Berlin.{IM_END}\n"]
    assert [span.start_from_prompt for span in result.spans] == [False, False]


def test_turn_end_that_differs_on_the_last_turn_skips_multi_turn_rows_as_render():
    tokenizer = FakeChatTokenizer(eos_on_last_template)
    assert label_chat(tokenizer, TWO_TURNS) is ChatLabelSkip.RENDER


def test_turn_end_that_differs_on_the_last_turn_still_labels_single_turn_rows():
    tokenizer = FakeChatTokenizer(eos_on_last_template)
    result = _labelled(tokenizer, _msgs(("user", "Hi"), ("assistant", "Hello")))
    assert _scored_texts(tokenizer, result) == [f"Hello{IM_END}<|endoftext|>"]


def test_auxiliary_render_failure_is_a_render_skip():
    tokenizer = FakeChatTokenizer(raises_on_empty_template)
    assert label_chat(tokenizer, _msgs(("assistant", "first"), ("user", "hi"))) is ChatLabelSkip.RENDER


def test_full_render_failure_propagates():
    def broken(messages, add_generation_prompt):
        raise RuntimeError("template rejects the conversation")

    with pytest.raises(RuntimeError, match="rejects"):
        label_chat(FakeChatTokenizer(broken), TWO_TURNS)


def test_merged_newline_token_across_the_prompt_boundary_is_scored_and_counted():
    tokenizer = FakeChatTokenizer(plain_template, merge_double_newline=True)
    messages = _msgs(("user", "Q"), ("assistant", "\nA"), ("user", "Q2"), ("assistant", "B"))
    result = _labelled(tokenizer, messages)

    first = result.spans[0]
    assert result.input_ids[first.start] == MERGED_NEWLINES_ID
    assert first.straddle_start and not first.straddle_end
    assert not result.spans[1].straddle_start
    assert result.straddles == 1
    _assert_labels_follow_spans(result)


def test_empty_assistant_content_scores_only_the_turn_end():
    tokenizer = FakeChatTokenizer(plain_template)
    result = _labelled(tokenizer, _msgs(("user", "Q"), ("assistant", ""), ("user", "Q2"), ("assistant", "B")))
    assert _scored_texts(tokenizer, result) == [f"{IM_END}\n", f"B{IM_END}\n"]


def test_no_assistant_turn_is_empty_and_renders_nothing():
    tokenizer = FakeChatTokenizer(plain_template)
    assert label_chat(tokenizer, _msgs(("user", "only a question"))) is ChatLabelSkip.EMPTY
    assert label_chat(tokenizer, _msgs(("system", "s"), ("user", "u"))) is ChatLabelSkip.EMPTY
    assert label_chat(tokenizer, []) is ChatLabelSkip.EMPTY
    assert tokenizer.render_calls == 0


def test_system_and_user_turns_are_never_scored():
    tokenizer = FakeChatTokenizer(plain_template)
    messages = _msgs(("system", "Be terse."), ("user", "Capital of France?"), ("assistant", "Paris."))
    result = _labelled(tokenizer, messages)

    scored = tokenizer.decode([label for label in result.labels if label != IGNORE_INDEX])
    assert scored == f"Paris.{IM_END}\n"
    assert "Be terse." not in scored and "Capital" not in scored


def test_assistant_first_conversation_uses_the_empty_prefix_generation_prompt():
    tokenizer = FakeChatTokenizer(plain_template)
    result = _labelled(tokenizer, _msgs(("assistant", "Hello."), ("user", "Hi")))
    assert _scored_texts(tokenizer, result) == [f"Hello.{IM_END}\n"]
    assert result.spans[0].start_from_prompt


def test_sentinel_content_in_real_data_does_not_break_span_location():
    tokenizer = FakeChatTokenizer(plain_template)
    result = _labelled(tokenizer, _msgs(("user", "Q"), ("assistant", "GODSENTINEL")))
    assert _scored_texts(tokenizer, result) == [f"GODSENTINEL{IM_END}\n"]
    assert SENTINEL not in tokenizer.decode(result.input_ids)


def test_labelling_is_deterministic():
    tokenizer = FakeChatTokenizer(qwen3_like_template, merge_double_newline=True)
    assert label_chat(tokenizer, TWO_TURNS) == label_chat(tokenizer, TWO_TURNS)


def test_role_map_maps_references_and_lets_other_roles_through():
    role_map = chat_role_map("human", "gpt")
    assert role_map == {"gpt": "assistant", "human": "user"}
    row = {
        "conversations": [
            {"from": "system", "value": "S"},
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"},
            {"from": "tool", "value": "T"},
        ]
    }
    messages = chat_messages(row, "conversations", "from", "value", role_map)
    assert [m.role for m in messages] == ["system", "user", "assistant", "tool"]
    assert [m.content for m in messages] == ["S", "Q", "A", "T"]


def test_role_map_user_reference_wins_when_both_references_match():
    assert chat_role_map("x", "x") == {"x": "user"}
    assert chat_role_map(None, "gpt") == {"gpt": "assistant"}


def test_none_content_is_omitted_from_the_template_dict():
    row = {"conversations": [{"from": "user", "value": None}, {"from": "assistant", "value": "A"}]}
    messages = chat_messages(row, "conversations", "from", "value", chat_role_map("user", "assistant"))
    assert messages[0].template_dict() == {"role": "user"}
    assert messages[1].template_dict() == {"role": "assistant", "content": "A"}


def test_numeric_content_is_coerced_to_text():
    row = {"conversations": [{"from": "user", "value": 2}, {"from": "assistant", "value": 4.5}]}
    messages = chat_messages(row, "conversations", "from", "value", {})
    assert [m.content for m in messages] == ["2", "4.5"]


def test_json_string_column_is_decoded():
    row = {"conversations": '[{"from": "user", "value": "Q"}, {"from": "assistant", "value": "A"}]'}
    messages = chat_messages(row, "conversations", "from", "value", {})
    assert [(m.role, m.content) for m in messages] == [("user", "Q"), ("assistant", "A")]


@pytest.mark.parametrize(
    "row, match",
    [
        ({"other": []}, "no 'conversations' column"),
        ({"conversations": None}, "is null"),
        ({"conversations": "not json"}, "not JSON"),
        ({"conversations": {"from": "user"}}, "must be a list"),
        ({"conversations": ["text"]}, "must be a dict"),
        ({"conversations": [{"value": "no role"}]}, "no 'from' role field"),
        ({"conversations": [{"from": None, "value": "null role"}]}, "no 'from' role field"),
    ],
)
def test_malformed_rows_raise_value_error(row, match):
    with pytest.raises(ValueError, match=match):
        chat_messages(row, "conversations", "from", "value", {})


def test_non_text_content_raises_value_error():
    row = {"conversations": [{"from": "user", "value": [{"text": "parts"}]}]}
    with pytest.raises(ValueError):
        chat_messages(row, "conversations", "from", "value", {})


def test_module_stays_self_contained():
    """Miners and the quant eval copy chat_labels.py verbatim; it must import nothing beyond stdlib and pydantic."""
    tree = ast.parse(Path(chat_labels.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"json", "enum", "typing", "pydantic"}, imported
