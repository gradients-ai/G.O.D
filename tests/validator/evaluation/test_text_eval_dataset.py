import json

import pytest

from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from validator.evaluation import text_eval_dataset
from validator.evaluation.text_eval_config import TextEvalConfig
from validator.evaluation.text_eval_dataset import DropReason
from validator.evaluation.text_eval_dataset import HfTruncatingTokenizer
from validator.evaluation.text_eval_dataset import TextEvalDatasetBuild
from validator.evaluation.text_eval_dataset import build_chat_eval_dataset
from validator.evaluation.text_eval_dataset import build_instruct_eval_dataset
from validator.evaluation.text_eval_dataset import build_text_eval_dataset
from validator.evaluation.text_eval_dataset import load_eval_rows
from validator.evaluation.text_examples import IGNORE_INDEX


EOS = 2


class CharTokenizer:
    bos_token_id = 1
    eos_token_id = EOS

    def encode_truncated(self, text: str, max_length: int) -> list[int]:
        return [ord(c) + 10 for c in text][:max_length]


TOK = CharTokenizer()
INSTRUCT = InstructTextDatasetType(field_instruction="instruct", field_input="input", field_output="output")
COMPLETION = InstructTextDatasetType(field_instruction="text")


def _build(rows, dataset_type=INSTRUCT, **config) -> TextEvalDatasetBuild:
    return build_instruct_eval_dataset(rows, dataset_type, TOK, TextEvalConfig(**config))


def _decode(ids: list[int]) -> str:
    return "".join("|" if t == EOS else chr(t - 10) for t in ids)


def test_drop_reasons_and_counts():
    rows = [
        {"instruct": "keep", "output": "yes"},  # 4 + 4 = 8 tokens, kept
        {"instruct": "", "output": ""},  # nothing at all: EMPTY  (no_input_format "{instruction}" -> "")
        {"instruct": "a", "output": ""},  # one prompt token: SHORT (checked before UNSCORED)
        {"instruct": "only prompt", "output": None},  # all masked: UNSCORED
        {"instruct": "x" * 20, "output": "y"},  # 22 tokens > 16: LONG
    ]
    build = _build(rows, sequence_len=16)
    assert [_decode(e.input_ids) for e in build.examples] == ["keepyes|"]
    assert build.n_rows == 5
    assert build.dropped == {
        DropReason.EMPTY: 1,
        DropReason.SHORT: 1,
        DropReason.LONG: 1,
        DropReason.UNSCORED: 1,
        DropReason.RENDER: 0,
    }
    assert build.straddles == 0


def test_long_rule_is_inclusive_at_sequence_len():
    build = _build([{"instruct": "abc", "output": "d"}], sequence_len=5)  # 3 + 1 + EOS = 5
    assert len(build.examples) == 1
    assert len(build.examples[0].input_ids) == 5


def test_min_sequence_len_two_keeps_two_token_rows():
    build = _build([{"instruct": "", "output": "x"}])  # [x, EOS], both scored
    assert [e.input_ids for e in build.examples] == [[ord("x") + 10, EOS]]


def test_stable_sort_by_length_keeps_file_order_on_ties():
    rows = [
        {"instruct": "long one", "output": "zz"},
        {"instruct": "b", "output": "1"},
        {"instruct": "c", "output": "2"},
        {"instruct": "mid", "output": "3"},
        {"instruct": "a", "output": "0"},
    ]
    build = _build(rows)
    assert [_decode(e.input_ids) for e in build.examples] == ["b1|", "c2|", "a0|", "mid3|", "long onezz|"]


def test_labels_mask_prompt_in_built_examples():
    [example] = _build([{"instruct": "ab", "input": "c", "output": "d"}]).examples
    assert _decode(example.input_ids) == "ab cd|"
    assert example.labels[:4] == [IGNORE_INDEX] * 4
    assert example.labels[4:] == example.input_ids[4:]


def test_completion_rows_are_chunked_and_empty_rows_counted():
    rows = [{"text": "abcdefgh"}, {"text": ""}, {"text": "xyz"}]
    build = _build(rows, dataset_type=COMPLETION, sequence_len=4)
    # "abcdefgh|" -> abcd, efgh, "|" (len 1: SHORT); "xyz|" -> one chunk.
    assert sorted(_decode(e.input_ids) for e in build.examples) == ["abcd", "efgh", "xyz|"]
    assert all(e.labels == e.input_ids for e in build.examples)
    assert build.n_rows == 3
    assert build.dropped[DropReason.EMPTY] == 1
    assert build.dropped[DropReason.SHORT] == 1


class CallableCharTokenizer(CharTokenizer):
    """Just enough of a transformers tokenizer for `HfTruncatingTokenizer` (the dispatcher wraps the real one)."""

    def __call__(self, text, truncation, max_length, padding, return_tensors):
        assert truncation is True and padding is False and return_tensors is None
        return {"input_ids": self.encode_truncated(text, max_length)}


def test_dispatch(monkeypatch):
    rows = [{"instruct": "a", "output": "b"}]
    assert len(build_text_eval_dataset(rows, INSTRUCT, CallableCharTokenizer(), TextEvalConfig()).examples) == 1

    # Chat dispatch is exercised end to end in test_instruct_text_dataset_wiring.py; here only the routing.
    seen = {}
    monkeypatch.setattr(text_eval_dataset, "resolve_chat_template", lambda name, tokenizer: f"template:{name}")
    monkeypatch.setattr(text_eval_dataset, "HfChatTokenizer", lambda tokenizer, template: ("chat", template))
    monkeypatch.setattr(
        text_eval_dataset, "build_chat_eval_dataset", lambda rows, dataset_type, tokenizer, config: seen.update(tok=tokenizer)
    )
    build_text_eval_dataset(rows, ChatTemplateDatasetType(chat_template="chatml"), TOK, TextEvalConfig())
    assert seen["tok"] == ("chat", "template:chatml")

    with pytest.raises(TypeError):
        build_text_eval_dataset(rows, DpoDatasetType(), TOK, TextEvalConfig())


def test_load_eval_rows_reads_json_in_order_with_null_columns(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([
        {"instruct": "first", "output": "1"},
        {"instruct": "second", "input": "ctx", "output": "2"},
    ]))
    rows = load_eval_rows(str(path))
    assert [row["instruct"] for row in rows] == ["first", "second"]
    # pyarrow unifies the schema: the first row gains input=None, which the instruct rule treats as no input.
    assert list(rows)[0]["input"] is None
    build = build_instruct_eval_dataset(rows, INSTRUCT, TOK, TextEvalConfig())
    assert sorted(_decode(e.input_ids) for e in build.examples) == ["first1|", "second ctx2|"]


def test_load_eval_rows_resolves_relative_paths(tmp_path, monkeypatch):
    (tmp_path / "eval.json").write_text(json.dumps([{"instruct": "a", "output": "b"}]))
    monkeypatch.chdir(tmp_path)
    assert len(load_eval_rows("eval.json")) == 1


class ChatMLCharTokenizer:
    """`chat_labels.ChatTokenizer` with one id per character and a ChatML template rendered in Python."""

    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        if any(m.get("content") == "reject me" for m in messages):
            raise RuntimeError("template refuses the conversation")
        text = "".join(f"<s>{m['role']}\n{m.get('content', '')}</s>\n" for m in messages)
        return text + ("<s>assistant\n" if add_generation_prompt else "")

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        return [ord(c) for c in text], [(k, k + 1) for k in range(len(text))]


def _conversation(*turns: tuple[str, str]) -> dict:
    return {"conversations": [{"from": role, "value": value} for role, value in turns]}


def test_chat_build_scores_assistant_turns_counts_drops_and_sorts():
    rows = [
        _conversation(("user", "a longer question"), ("assistant", "yes")),
        _conversation(("user", "only a question")),
        _conversation(("human", "q"), ("gpt", "ok")),  # unknown roles here: no assistant turn
        _conversation(("user", "q"), ("assistant", "ok")),
        _conversation(("user", "x" * 40), ("assistant", "long")),
    ]
    build = build_chat_eval_dataset(rows, ChatTemplateDatasetType(), ChatMLCharTokenizer(), TextEvalConfig(sequence_len=60))
    scored = ["".join(chr(t) for t in e.labels if t != IGNORE_INDEX) for e in build.examples]
    assert scored == ["ok</s>\n", "yes</s>\n"]
    assert all(e.attention_mask == [1] * len(e.input_ids) for e in build.examples)
    assert build.n_rows == 5
    assert build.dropped[DropReason.UNSCORED] == 2
    assert build.dropped[DropReason.LONG] == 1
    assert build.straddles == 0


def test_chat_build_maps_dataset_roles_and_field_names():
    rows = [{"msgs": [{"who": "human", "text": "q"}, {"who": "gpt", "text": "ok"}]}]
    dataset_type = ChatTemplateDatasetType(
        chat_column="msgs", chat_role_field="who", chat_content_field="text",
        chat_user_reference="human", chat_assistant_reference="gpt",
    )
    [example] = build_chat_eval_dataset(rows, dataset_type, ChatMLCharTokenizer(), TextEvalConfig()).examples
    assert "".join(chr(t) for t in example.input_ids) == "<s>user\nq</s>\n<s>assistant\nok</s>\n"


def test_chat_build_fails_when_the_template_refuses_a_row():
    rows = [_conversation(("user", "q"), ("assistant", "ok")), _conversation(("user", "reject me"), ("assistant", "x"))]
    with pytest.raises(RuntimeError, match="refuses"):
        build_chat_eval_dataset(rows, ChatTemplateDatasetType(), ChatMLCharTokenizer(), TextEvalConfig())


def test_load_eval_rows_reads_csv_and_rejects_non_local_formats(tmp_path):
    path = tmp_path / "eval.csv"
    path.write_text("instruct,output\nfirst,1\nsecond,2\n")
    assert [row["instruct"] for row in load_eval_rows(str(path), FileFormat.CSV)] == ["first", "second"]
    with pytest.raises(ValueError, match="local json or csv"):
        load_eval_rows(str(path), FileFormat.HF)


# ---- real tokenizers (need transformers + the offline HF cache; skipped in the dev venv) ----


def _cached_tokenizer(repo: str):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except OSError:
        pytest.skip(f"{repo} tokenizer not in the local HF cache")


SAMPLES = [
    "Paris is the capital.",
    "  leading and trailing whitespace \n",
    "unicode héllo 🙂 中文",
    "x " * 300,
]


@pytest.mark.parametrize("repo", ["Qwen/Qwen2.5-0.5B", "unsloth/Llama-3.2-1B"])
def test_hf_truncating_tokenizer_matches_direct_call(repo):
    tokenizer = _cached_tokenizer(repo)
    adapter = HfTruncatingTokenizer(tokenizer)
    assert adapter.bos_token_id == tokenizer.bos_token_id
    assert adapter.eos_token_id == tokenizer.eos_token_id
    for text in SAMPLES:
        for max_length in (8, 256, 4096):
            expected = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
            assert adapter.encode_truncated(text, max_length) == expected


@pytest.mark.parametrize("repo", ["Qwen/Qwen2.5-0.5B", "unsloth/Llama-3.2-1B"])
def test_real_tokenizer_instruct_row(repo):
    tokenizer = _cached_tokenizer(repo)
    adapter = HfTruncatingTokenizer(tokenizer)
    [example] = build_instruct_eval_dataset(
        [{"instruct": "What is 2+2?", "input": "", "output": "4"}], INSTRUCT, adapter, TextEvalConfig()
    ).examples
    prompt_ids = tokenizer("What is 2+2?")["input_ids"]
    response_ids = tokenizer("4")["input_ids"]
    if response_ids[0] == tokenizer.bos_token_id:
        response_ids = response_ids[1:]
    if response_ids[-1] != tokenizer.eos_token_id:
        response_ids = response_ids + [tokenizer.eos_token_id]
    assert example.input_ids == prompt_ids + response_ids
    assert example.labels == [IGNORE_INDEX] * len(prompt_ids) + response_ids
