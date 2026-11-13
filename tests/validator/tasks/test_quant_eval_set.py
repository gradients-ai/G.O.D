"""Quant eval-set tokenisation with a fake QuantTokenizer (no transformers): masking, skips, determinism, format defaults."""

import json

import pytest

from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.training_config import _process_instruct_dataset_fields
from validator.evaluation.chat_labels import ChatLabelled
from validator.evaluation.chat_labels import ChatMessage
from validator.evaluation.chat_labels import label_chat
from validator.evaluation.quant_loss import IGNORE_INDEX
from validator.evaluation.quant_loss import QuantEvalExample
from validator.tasks.quant_eval_set import SkipReason
from validator.tasks.quant_eval_set import build_quant_eval_set
from validator.tasks.quant_eval_set import instruct_prompt_and_completion
from validator.tasks.quant_eval_set import quant_eval_max_sequence_len
from validator.tasks.quant_eval_set import quant_eval_set_jsonl
from validator.tasks.quant_eval_set import tokenize_instruct


BOS = 1
EOS = 2
MAX_LEN = 4096
INSTRUCT = InstructTextDatasetType(field_instruction="instruct", field_output="output")
INSTRUCT_WITH_INPUT = InstructTextDatasetType(
    field_instruction="instruct", field_input="input", field_output="output", field_system="system"
)
CHAT = ChatTemplateDatasetType(chat_column="conversations", chat_role_field="from", chat_content_field="value")


class FakeChatTokenizer:
    """chat_labels.ChatTokenizer: one id per character (ord); a ChatML-like template that emits BOS itself."""

    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        text = "\x01" + "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]


class RejectingChatTokenizer(FakeChatTokenizer):
    """A template that raises on a turn role it does not know, as gemma's does on a system turn."""

    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        if any(m["role"] == "tool" for m in messages):
            raise ValueError("tool turns are not supported")
        return super().render(messages, add_generation_prompt)


class FakeTokenizer:
    """One id per character (ord); BOS on add_special_tokens; chat through FakeChatTokenizer."""

    def __init__(self, vocab_size: int = 256, eos_token_id: int | None = EOS, chat_tokenizer=None):
        self.vocab_size = vocab_size
        self.context_len: int | None = None
        self.bos_token_id = BOS
        self.eos_token_id = eos_token_id
        self.chat_tokenizer = chat_tokenizer or FakeChatTokenizer()
        self.chat_templates: list[str | None] = []

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return ([BOS] if add_special_tokens else []) + [ord(c) for c in text]

    def chat_labeller(self, chat_template: str | None) -> FakeChatTokenizer:
        self.chat_templates.append(chat_template)
        return self.chat_tokenizer


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def _chat_row(*turns: tuple[str, str]) -> dict:
    return {"conversations": [{"from": role, "value": value} for role, value in turns]}


def test_instruct_masks_exactly_the_prompt_and_supervises_completion_and_eos():
    prompt, completion = instruct_prompt_and_completion({"instruct": "Add 2+2", "output": "4"}, INSTRUCT)
    example = tokenize_instruct(FakeTokenizer(), prompt, completion)

    assert prompt == "Add 2+2"
    prompt_ids = [BOS] + _ids(prompt)
    assert example.input_ids == prompt_ids + _ids("4") + [EOS]
    assert example.labels == [IGNORE_INDEX] * len(prompt_ids) + _ids("4") + [EOS]


def test_instruct_without_eos_supervises_completion_only():
    example = tokenize_instruct(FakeTokenizer(eos_token_id=None), "q ", "a")

    assert example.labels == [IGNORE_INDEX] * 3 + _ids("a")


def test_instruct_completion_loses_its_bos_and_keeps_an_eos_it_already_ends_with():
    example = tokenize_instruct(FakeTokenizer(), "q", "a")
    assert example.input_ids == [BOS] + _ids("q") + _ids("a") + [EOS]

    tokenizer = FakeTokenizer()
    tokenizer.encode = lambda text, add_special_tokens: (
        ([BOS] if add_special_tokens else []) + _ids(text) + ([EOS] if text == "a" else [])
    )
    assert tokenize_instruct(tokenizer, "q", "a").input_ids == [BOS] + _ids("q") + _ids("a") + [EOS]


def test_instruct_prompt_with_input_and_system_uses_the_default_format():
    row = {"instruct": "Translate", "input": "hola", "output": "hello", "system": "Be terse"}

    # system_format defaults to "{system}": glued straight onto the turn, as axolotl does in training.
    assert instruct_prompt_and_completion(row, INSTRUCT_WITH_INPUT) == ("Be terseTranslate hola", "hello")


def test_system_format_and_static_system_prompt_follow_axolotl():
    dataset_type = INSTRUCT_WITH_INPUT.model_copy(
        update={"system_format": "### System:\n{system}\n\n", "system_prompt": "static"}
    )
    row = {"instruct": "q", "output": "a"}

    assert instruct_prompt_and_completion(row, dataset_type)[0] == "### System:\nstatic\n\nq"
    assert instruct_prompt_and_completion({**row, "system": "own"}, dataset_type)[0] == "### System:\nown\n\nq"
    assert instruct_prompt_and_completion({**row, "system": ""}, dataset_type)[0] == "q"


def test_unset_field_names_fall_back_to_axolotls_defaults():
    dataset_type = InstructTextDatasetType(field_output="out", format="{instruction}|{input}", no_input_format="{instruction}")

    assert instruct_prompt_and_completion({"instruction": "x", "input": "i", "out": "y"}, dataset_type) == ("x|i", "y")


def test_instruct_prompt_ending_in_whitespace_is_kept():
    dataset_type = INSTRUCT.model_copy(update={"no_input_format": "Q: {instruction}\nA: "})

    assert instruct_prompt_and_completion({"instruct": "x", "output": "y"}, dataset_type) == ("Q: x\nA: ", "y")


def test_empty_input_uses_no_input_format():
    dataset_type = INSTRUCT_WITH_INPUT.model_copy(
        update={"format": "{instruction} | {input}", "no_input_format": "N {instruction}"}
    )

    assert instruct_prompt_and_completion({"instruct": "x", "input": "", "output": "y"}, dataset_type)[0] == "N x"
    assert instruct_prompt_and_completion({"instruct": "x", "input": "i", "output": "y"}, dataset_type)[0] == "x | i"


@pytest.mark.parametrize("dataset_type", [INSTRUCT, INSTRUCT_WITH_INPUT])
def test_format_defaults_match_training_config(dataset_type):
    dumped = {key: value for key, value in dataset_type.model_dump().items() if value is not None}
    trainer_formats = _process_instruct_dataset_fields(dumped)["type"]
    row = {"instruct": "I", "input": "N", "output": "o"}

    expected = trainer_formats["format"].format(instruction="I", input="N")
    assert instruct_prompt_and_completion(row, dataset_type)[0] == expected
    no_input = trainer_formats["no_input_format"].format(instruction="I")
    assert instruct_prompt_and_completion({**row, "input": ""}, dataset_type)[0] == no_input


def _chat_example(row: dict, dataset_type: ChatTemplateDatasetType = CHAT) -> QuantEvalExample | SkipReason:
    build = build_quant_eval_set([row], dataset_type, FakeTokenizer(), MAX_LEN)
    if build.examples:
        return build.examples[0]
    skips = {SkipReason.EMPTY: build.n_skipped_empty, SkipReason.RENDER: build.n_skipped_render}
    return next(reason for reason, count in skips.items() if count)


def _scored_text(example: QuantEvalExample) -> str:
    return "".join(chr(label) if label != IGNORE_INDEX else "_" for label in example.labels)


def test_chat_labels_are_label_chats():
    """The quant eval set and the standard eval share one chat rule: the example is label_chat's output, verbatim."""
    row = _chat_row(("system", "s"), ("user", "hi"), ("assistant", "yo"), ("user", "more"), ("assistant", "ok"))
    messages = [ChatMessage(role=turn["from"], content=turn["value"]) for turn in row["conversations"]]

    expected = label_chat(FakeChatTokenizer(), messages)
    example = _chat_example(row)

    assert isinstance(expected, ChatLabelled)
    assert (example.input_ids, example.labels) == (expected.input_ids, expected.labels)


def test_chat_supervises_every_assistant_turn_through_its_turn_end():
    example = _chat_example(_chat_row(("system", "s"), ("user", "hi"), ("assistant", "yo"), ("user", "c"), ("assistant", "d")))

    full_text = "\x01<system>s</><user>hi</><assistant>yo</><user>c</><assistant>d</>"
    assert example.input_ids == _ids(full_text)
    assert (
        _scored_text(example)
        == "_" * len("\x01<system>s</><user>hi</><assistant>") + "yo</>" + "_" * len("<user>c</><assistant>") + "d</>"
    )


def test_chat_resolves_the_tasks_template_name():
    tokenizer = FakeTokenizer()
    build_quant_eval_set([_chat_row(("user", "a"), ("assistant", "b"))], CHAT, tokenizer, MAX_LEN)
    build_quant_eval_set([], CHAT.model_copy(update={"chat_template": "tokenizer_default"}), tokenizer, MAX_LEN)

    assert tokenizer.chat_templates == ["chatml", "tokenizer_default"]


def test_chat_maps_custom_role_references():
    dataset_type = CHAT.model_copy(update={"chat_user_reference": "human", "chat_assistant_reference": "gpt"})
    example = _chat_example(_chat_row(("human", "a"), ("gpt", "b")), dataset_type)

    assert example.input_ids == _ids("\x01<user>a</><assistant>b</>")
    assert _scored_text(example).endswith("_b</>")


def test_user_last_chat_still_scores_its_earlier_assistant_turns():
    example = _chat_example(_chat_row(("user", "a"), ("assistant", "b"), ("user", "c")))

    assert _scored_text(example) == "_" * len("\x01<user>a</><assistant>") + "b</>" + "_" * len("<user>c</>")


@pytest.mark.parametrize(
    "row",
    [
        _chat_row(("user", "a"), ("user", "c")),
        {"conversations": ""},
        {"conversations": [{"value": "no role"}]},
        {"other": []},
    ],
)
def test_unsupervisable_or_malformed_chat_rows_are_skipped_as_empty(row):
    assert _chat_example(row) == SkipReason.EMPTY


def test_chat_rows_the_template_rejects_are_skipped_as_render():
    build = build_quant_eval_set(
        [_chat_row(("user", "a"), ("tool", "t"), ("assistant", "b")), _chat_row(("user", "a"), ("assistant", "b"))],
        CHAT,
        FakeTokenizer(chat_tokenizer=RejectingChatTokenizer()),
        MAX_LEN,
    )

    assert (len(build.examples), build.n_skipped_render) == (1, 1)


def test_build_skips_and_counts_user_last_empty_and_long_rows():
    rows = [
        {"instruct": "ok", "output": "fine"},
        {"instruct": "empty", "output": ""},
        {"instruct": "long", "output": "x" * 50},
    ]
    build = build_quant_eval_set(rows, INSTRUCT, FakeTokenizer(), max_sequence_len=20)
    chat_build = build_quant_eval_set(
        [_chat_row(("user", "a"), ("assistant", "b")), _chat_row(("user", "a"), ("user", "c"))],
        CHAT,
        FakeTokenizer(),
        MAX_LEN,
    )

    assert (len(build.examples), build.n_rows, build.n_skipped_empty, build.n_skipped_long) == (1, 3, 1, 1)
    assert (len(chat_build.examples), chat_build.n_skipped_empty) == (1, 1)


def test_example_whose_only_label_is_position_zero_is_skipped():
    tokenizer = FakeTokenizer(eos_token_id=None)
    build = build_quant_eval_set([{"instruct": "", "output": "a"}], INSTRUCT, tokenizer, MAX_LEN)

    assert (build.examples, build.n_skipped_empty) == ([], 1)


def test_id_outside_vocab_raises():
    with pytest.raises(ValueError, match="outside the tokenizer vocabulary"):
        build_quant_eval_set([{"instruct": "q", "output": "a"}], INSTRUCT, FakeTokenizer(vocab_size=50), MAX_LEN)


def test_build_is_deterministic_and_every_example_aligns():
    rows = [{"instruct": f"q{i}", "output": f"a{i}"} for i in range(20)]
    chat_rows = [_chat_row(("user", f"u{i}"), ("assistant", f"a{i}")) for i in range(20)]

    for dataset_type, data in ((INSTRUCT, rows), (CHAT, chat_rows)):
        first = build_quant_eval_set(data, dataset_type, FakeTokenizer(), MAX_LEN)
        second = build_quant_eval_set(data, dataset_type, FakeTokenizer(), MAX_LEN)
        assert first == second and len(first.examples) == 20
        for example in first.examples:
            QuantEvalExample.model_validate(example.model_dump())


def test_jsonl_has_one_example_per_line():
    build = build_quant_eval_set([{"instruct": "q", "output": "a"}] * 3, INSTRUCT, FakeTokenizer(), MAX_LEN)
    lines = quant_eval_set_jsonl(build.examples).splitlines()

    assert [QuantEvalExample.model_validate(json.loads(line)) for line in lines] == build.examples


class TestQuantEvalMaxSequenceLen:
    def test_unknown_context_keeps_the_cap(self):
        assert quant_eval_max_sequence_len(None, 4096) == 4096

    def test_long_context_keeps_the_cap(self):
        assert quant_eval_max_sequence_len(131_072, 4096) == 4096

    def test_short_context_leaves_room_for_the_generated_token(self):
        # TinyLlama: 2048 positions, so the scorer's engine holds a 2047-token example plus one.
        assert quant_eval_max_sequence_len(2048, 4096) == 2047

    def test_examples_over_the_context_are_skipped_as_long(self):
        rows = [{"instruct": "q" * 10, "output": "a" * 10}, {"instruct": "q", "output": "a"}]
        tokenizer = FakeTokenizer()
        build = build_quant_eval_set(rows, INSTRUCT, tokenizer, quant_eval_max_sequence_len(12, 4096))

        assert len(build.examples) == 1
        assert build.n_skipped_long == 1
        assert all(len(example.input_ids) + 1 <= 12 for example in build.examples)
