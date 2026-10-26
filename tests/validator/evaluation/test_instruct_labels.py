import pytest
from pydantic import ValidationError

from core.models.dataset_models import InstructTextDatasetType
from core.training_config import _process_instruct_dataset_fields
from validator.evaluation.instruct_labels import COMPLETION_MAX_LENGTH_MULTIPLIER
from validator.evaluation.instruct_labels import InstructFormat
from validator.evaluation.instruct_labels import build_instruct_prompt
from validator.evaluation.instruct_labels import instruct_format
from validator.evaluation.instruct_labels import tokenize_completion_row
from validator.evaluation.instruct_labels import tokenize_instruct_row
from validator.evaluation.text_examples import IGNORE_INDEX
from validator.evaluation.text_examples import TokenizedExample


BOS = 1
EOS = 2
EOS_TEXT = "¤"  # the fake tokenizer encodes this one character to EOS


class FakeTokenizer:
    """Per-character ids, optional auto BOS/EOS, HF-style truncation that reserves room for the special tokens."""

    def __init__(self, add_bos: bool = False, add_eos: bool = False, truncation_side: str = "right",
                 bos_token_id: int | None = BOS, eos_token_id: int | None = EOS):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.truncation_side = truncation_side
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.calls: list[tuple[str, int]] = []

    def char_id(self, char: str) -> int:
        return EOS if char == EOS_TEXT else ord(char) + 10

    def ids(self, text: str) -> list[int]:
        return [self.char_id(c) for c in text]

    def encode_truncated(self, text: str, max_length: int) -> list[int]:
        self.calls.append((text, max_length))
        body = self.ids(text)
        n_special = int(self.add_bos) + int(self.add_eos)
        room = max(max_length - n_special, 0)
        if len(body) > room:
            body = body[:room] if self.truncation_side == "right" else body[len(body) - room :]
        return ([self.bos_token_id] if self.add_bos else []) + body + ([self.eos_token_id] if self.add_eos else [])


TOK = FakeTokenizer()


def _fmt(**overrides) -> InstructFormat:
    dataset_type = InstructTextDatasetType(field_instruction="instruct", field_input="input", field_output="output")
    return instruct_format(dataset_type.model_copy(update=overrides))


def _scored(example: TokenizedExample) -> list[int]:
    return [label for label in example.labels if label != IGNORE_INDEX]


# ---- format resolution: parity with core.training_config + axolotl's dataclass defaults ----

# UserDefinedDatasetConfig defaults (axolotl 0bda5a13 prompt_strategies/user_defined.py) for keys G.O.D strips.
AXOLOTL_DEFAULTS = {
    "system_prompt": "",
    "field_system": "system",
    "field_instruction": "instruction",
    "field_input": "input",
    "field_output": "output",
    "system_format": "{system}",
}


@pytest.mark.parametrize(
    "dataset_type",
    [
        InstructTextDatasetType(field_instruction="instruct", field_input="input", field_output="output"),
        InstructTextDatasetType(field_instruction="instruct", field_output="output"),
        InstructTextDatasetType(
            field_instruction="q",
            field_input="ctx",
            field_output="a",
            field_system="sys",
            system_prompt="Be terse.",
            system_format="<<{system}>>\n",
            format="Q: {instruction}\nI: {input}\nA:",
            no_input_format="Q: {instruction}\nA:",
        ),
    ],
)
def test_format_matches_training_config_processing(dataset_type):
    entry = _process_instruct_dataset_fields({k: v for k, v in dataset_type.model_dump().items() if v is not None})
    assert entry["format"] == "custom"
    effective = {**AXOLOTL_DEFAULTS, **entry["type"]}
    fmt = instruct_format(dataset_type)
    assert fmt.format == effective["format"]
    assert fmt.no_input_format == effective["no_input_format"]
    assert fmt.system_format == effective["system_format"]
    assert fmt.system_prompt == effective["system_prompt"]
    for field in ("field_instruction", "field_input", "field_output", "field_system"):
        assert getattr(fmt, field) == effective[field]
    assert not fmt.is_completion


@pytest.mark.parametrize("field_output", [None, ""])
def test_completion_shape_matches_training_config(field_output):
    dataset_type = InstructTextDatasetType(field_instruction="text", field_output=field_output)
    entry = _process_instruct_dataset_fields({k: v for k, v in dataset_type.model_dump().items() if v is not None})
    assert entry == {"type": "completion", "field": "text"}
    fmt = instruct_format(dataset_type)
    assert fmt.is_completion
    assert fmt.field_instruction == "text"


def test_completion_without_field_instruction_raises():
    with pytest.raises(ValueError):
        instruct_format(InstructTextDatasetType())


def test_explicit_empty_strings_are_kept_not_defaulted():
    # setdefault only fills missing keys, so an explicit "" survives into axolotl.
    fmt = _fmt(system_format="", no_input_format="")
    assert fmt.system_format == ""
    assert fmt.no_input_format == ""


def test_field_system_defaults_to_system_column():
    assert _fmt().field_system == "system"


# ---- prompt construction ----


def test_input_present_uses_format():
    prompt, response = build_instruct_prompt({"instruct": "Translate", "input": "hola", "output": "hello"}, _fmt())
    assert (prompt, response) == ("Translate hola", "hello")


@pytest.mark.parametrize("input_value", ["", None])
def test_falsy_input_uses_no_input_format(input_value):
    prompt, _ = build_instruct_prompt({"instruct": "Add 2+2", "input": input_value, "output": "4"}, _fmt())
    assert prompt == "Add 2+2"


def test_missing_input_column():
    prompt, _ = build_instruct_prompt({"instruct": "Add", "output": "4"}, _fmt())
    assert prompt == "Add"


def test_custom_formats():
    fmt = _fmt(format="Q: {instruction}\nI: {input}\nA:", no_input_format="Q: {instruction}\nA:")
    assert build_instruct_prompt({"instruct": "x", "input": "y", "output": "z"}, fmt)[0] == "Q: x\nI: y\nA:"
    assert build_instruct_prompt({"instruct": "x", "input": "", "output": "z"}, fmt)[0] == "Q: x\nA:"


def test_braces_in_content_are_not_reformatted():
    prompt, _ = build_instruct_prompt({"instruct": "{input} {x}", "input": "{instruction}", "output": "o"}, _fmt())
    assert prompt == "{input} {x} {instruction}"


def test_null_instruction_formats_as_none_string():
    assert build_instruct_prompt({"instruct": None, "input": "", "output": "o"}, _fmt())[0] == "None"


def test_missing_instruction_column_raises_key_error():
    with pytest.raises(KeyError):
        build_instruct_prompt({"input": "", "output": "o"}, _fmt())


@pytest.mark.parametrize("row", [{"instruct": "i", "output": None}, {"instruct": "i", "output": ""}, {"instruct": "i"}])
def test_empty_null_or_missing_output_is_empty_response(row):
    assert build_instruct_prompt(row, _fmt())[1] == ""


def test_system_column_read_even_without_task_field_system():
    fmt = _fmt()
    prompt, _ = build_instruct_prompt({"instruct": "i", "input": "", "output": "o", "system": "SYS "}, fmt)
    assert prompt == "SYS i"


def test_system_prompt_used_only_when_column_absent():
    fmt = _fmt(system_prompt="DEFAULT ")
    assert build_instruct_prompt({"instruct": "i", "output": "o"}, fmt)[0] == "DEFAULT i"
    # axolotl tests column membership, so a present-but-null system column means no system text at all.
    assert build_instruct_prompt({"instruct": "i", "output": "o", "system": None}, fmt)[0] == "i"
    assert build_instruct_prompt({"instruct": "i", "output": "o", "system": ""}, fmt)[0] == "i"
    assert build_instruct_prompt({"instruct": "i", "output": "o", "system": "ROW "}, fmt)[0] == "ROW i"


def test_system_format_applied_and_empty_format_suppresses_system():
    row = {"instruct": "i", "output": "o", "system": "S"}
    assert build_instruct_prompt(row, _fmt(system_format="[{system}]"))[0] == "[S]i"
    assert build_instruct_prompt(row, _fmt(system_format=""))[0] == "i"


def test_custom_field_system():
    fmt = _fmt(field_system="sys")
    row = {"instruct": "i", "output": "o", "sys": "A ", "system": "B "}
    assert build_instruct_prompt(row, fmt)[0] == "A i"


# ---- tokenization ----


def test_prompt_masked_response_scored_with_eos():
    example = tokenize_instruct_row(TOK, {"instruct": "ab", "input": "", "output": "cd"}, _fmt(), 4096)
    assert example.input_ids == TOK.ids("ab") + TOK.ids("cd") + [EOS]
    assert example.labels == [IGNORE_INDEX] * 2 + TOK.ids("cd") + [EOS]
    assert example.attention_mask == [1] * 5
    assert example.n_scored() == 3


def test_tokenizer_called_with_sequence_len_for_both_parts():
    tok = FakeTokenizer()
    tokenize_instruct_row(tok, {"instruct": "ab", "input": "", "output": "cd"}, _fmt(), 123)
    assert tok.calls == [("ab", 123), ("cd", 123)]


def test_no_double_eos_when_response_already_ends_in_eos():
    example = tokenize_instruct_row(TOK, {"instruct": "a", "output": "b" + EOS_TEXT}, _fmt(), 4096)
    assert example.input_ids == TOK.ids("a") + [TOK.char_id("b"), EOS]


def test_auto_eos_tokenizer_gets_no_second_eos_and_prompt_keeps_its_eos():
    tok = FakeTokenizer(add_eos=True)
    example = tokenize_instruct_row(tok, {"instruct": "a", "output": "b"}, _fmt(), 4096)
    # The prompt's auto-EOS stays (add_eos=False only stops appending one more); masked.
    assert example.input_ids == [TOK.char_id("a"), EOS, TOK.char_id("b"), EOS]
    assert example.labels == [IGNORE_INDEX, IGNORE_INDEX, TOK.char_id("b"), EOS]


def test_bos_stripped_from_response_only():
    tok = FakeTokenizer(add_bos=True)
    example = tokenize_instruct_row(tok, {"instruct": "a", "output": "b"}, _fmt(), 4096)
    assert example.input_ids == [BOS, TOK.char_id("a"), TOK.char_id("b"), EOS]
    assert example.labels == [IGNORE_INDEX, IGNORE_INDEX, TOK.char_id("b"), EOS]


def test_eos_not_appended_when_truncated_to_sequence_len():
    example = tokenize_instruct_row(TOK, {"instruct": "a", "output": "bcdef"}, _fmt(), 3)
    assert example.input_ids == TOK.ids("a") + TOK.ids("bcd")
    assert _scored(example) == TOK.ids("bcd")


def test_eos_appended_when_one_short_of_sequence_len():
    example = tokenize_instruct_row(TOK, {"instruct": "a", "output": "bc"}, _fmt(), 3)
    assert _scored(example) == TOK.ids("bc") + [EOS]


def test_prompt_truncated_independently_left_side():
    tok = FakeTokenizer(truncation_side="left")
    example = tokenize_instruct_row(tok, {"instruct": "abcdef", "output": "xyz"}, _fmt(), 4)
    assert example.input_ids == TOK.ids("cdef") + TOK.ids("xyz") + [EOS]
    assert len(example.input_ids) > 4  # longer than sequence_len: the dataset builder drops it


class BosOnlyResponseTokenizer(FakeTokenizer):
    def encode_truncated(self, text: str, max_length: int) -> list[int]:
        return [BOS] if text == "R" else super().encode_truncated(text, max_length)


def test_bos_strip_runs_after_eos_append():
    # A response that encodes to only BOS gains EOS first, then loses BOS: [EOS].
    example = tokenize_instruct_row(BosOnlyResponseTokenizer(), {"instruct": "a", "output": "R"}, _fmt(), 4096)
    assert example.input_ids == [TOK.char_id("a"), EOS]
    assert _scored(example) == [EOS]


def test_empty_response_yields_masked_prompt_only():
    example = tokenize_instruct_row(TOK, {"instruct": "abc", "output": ""}, _fmt(), 4096)
    assert example.input_ids == TOK.ids("abc")
    assert example.n_scored() == 0


def test_empty_prompt_and_response_is_empty_example():
    fmt = _fmt(no_input_format="")
    example = tokenize_instruct_row(TOK, {"instruct": "", "output": ""}, fmt, 4096)
    assert example.input_ids == [] and example.labels == []


def test_tokenizer_returning_nothing_is_empty():
    tok = FakeTokenizer()
    tok.encode_truncated = lambda text, max_length: []
    assert tokenize_instruct_row(tok, {"instruct": "a", "output": "b"}, _fmt(), 4096).input_ids == []


def test_missing_eos_id_raises_instead_of_appending_none():
    tok = FakeTokenizer(eos_token_id=None)
    with pytest.raises(ValueError):
        tokenize_instruct_row(tok, {"instruct": "a", "output": "b"}, _fmt(), 4096)


def test_unicode_and_whitespace_preserved():
    row = {"instruct": "  héllo 🙂 ", "input": "", "output": " ok\n"}
    example = tokenize_instruct_row(TOK, row, _fmt(), 4096)
    assert example.input_ids == TOK.ids("  héllo 🙂 ") + TOK.ids(" ok\n") + [EOS]


# ---- completion ----


def _completion_fmt() -> InstructFormat:
    return instruct_format(InstructTextDatasetType(field_instruction="text"))


def test_completion_whole_text_scored_with_eos_and_bos_kept():
    tok = FakeTokenizer(add_bos=True)
    [example] = tokenize_completion_row(tok, {"text": "abc"}, _completion_fmt(), 4096)
    assert example.input_ids == [BOS] + TOK.ids("abc") + [EOS]
    assert example.labels == example.input_ids


def test_completion_chunks_of_sequence_len():
    chunks = tokenize_completion_row(TOK, {"text": "abcdefg"}, _completion_fmt(), 3)
    ids = TOK.ids("abcdefg") + [EOS]
    assert [c.input_ids for c in chunks] == [ids[0:3], ids[3:6], ids[6:8]]
    assert all(c.labels == c.input_ids for c in chunks)


def test_completion_truncates_at_sixty_four_times_sequence_len():
    tok = FakeTokenizer()
    seq = 2
    cap = seq * COMPLETION_MAX_LENGTH_MULTIPLIER
    chunks = tokenize_completion_row(tok, {"text": "a" * (cap + 10)}, _completion_fmt(), seq)
    assert tok.calls[-1][1] == cap
    assert sum(len(c.input_ids) for c in chunks) == cap  # truncated, so no EOS
    assert EOS not in [t for c in chunks for t in c.input_ids]


@pytest.mark.parametrize("text", ["", None])
def test_completion_empty_text_yields_no_chunk(text):
    assert tokenize_completion_row(TOK, {"text": text}, _completion_fmt(), 16) == []


def test_completion_missing_column_raises():
    with pytest.raises(KeyError):
        tokenize_completion_row(TOK, {"other": "x"}, _completion_fmt(), 16)


def test_build_instruct_prompt_rejects_completion_format():
    with pytest.raises(ValueError):
        build_instruct_prompt({"text": "x"}, _completion_fmt())


# ---- TokenizedExample ----


def test_tokenized_example_rejects_mismatched_labels():
    with pytest.raises(ValidationError):
        TokenizedExample(input_ids=[5, 6], attention_mask=[1, 1], labels=[5, 7])
    with pytest.raises(ValidationError):
        TokenizedExample(input_ids=[5, 6], attention_mask=[1], labels=[5, 6])
    assert TokenizedExample(input_ids=[5, 6], attention_mask=[1, 1], labels=[IGNORE_INDEX, 6]).n_scored() == 1
