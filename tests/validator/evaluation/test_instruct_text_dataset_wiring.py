"""The instruct/chat evaluator's dataset wiring: what `_load_evaluation_dataset` hands the Trainer, and the sequence_len ladder.

A fake transformers tokenizer (one id per character, ChatML rendered in Python) stands in for the real one, so the test pins
the evaluator <-> builder plumbing rather than tokenisation (the golden equivalence test covers that on real tokenizers).
Importing the evaluator needs torch/transformers/accelerate, so this runs under the golden venv and is skipped in the dev venv.
"""

from types import SimpleNamespace

import pytest

from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from validator.evaluation.text_eval_config import TextEvalConfig
from validator.evaluation.text_examples import IGNORE_INDEX


pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")
instruct_text = pytest.importorskip("validator.evaluation.evaluators.instruct_text")
common = pytest.importorskip("validator.evaluation.common")


EOS = 2
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
INSTRUCT = InstructTextDatasetType(field_instruction="instruct", field_input="input", field_output="output")
CHAT = ChatTemplateDatasetType(chat_template="tokenizer_default")


class FakeHfTokenizer:
    """The slice of a transformers fast tokenizer the eval data path calls."""

    is_fast = True
    bos_token_id = None
    eos_token_id = EOS
    pad_token_id = EOS
    name_or_path = "fake/tokenizer"
    chat_template = "{# fake: rendered by apply_chat_template below #}"

    def __init__(self, model_max_length: int = 1_000_000_000):
        self.model_max_length = model_max_length

    def __call__(self, text, truncation=False, max_length=None, padding=False, return_tensors=None,
                 add_special_tokens=True, return_offsets_mapping=False):
        ids = [ord(char) + 10 for char in text]
        if return_offsets_mapping:
            return {"input_ids": ids, "offset_mapping": [(k, k + 1) for k in range(len(text))]}
        return {"input_ids": ids[:max_length] if truncation else ids}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, chat_template):
        assert tokenize is False and chat_template == self.chat_template
        if not messages:
            raise IndexError("list index out of range")  # transformers indexes conversation[0]
        text = "".join(f"{IM_START}{m['role']}\n{m.get('content', '')}{IM_END}\n" for m in messages)
        return text + (f"{IM_START}assistant\n" if add_generation_prompt else "")

    def decode(self, ids):
        return "".join("|" if token == EOS else chr(token - 10) for token in ids)


def _model(max_position_embeddings: int | None):
    return SimpleNamespace(config=SimpleNamespace(max_position_embeddings=max_position_embeddings))


@pytest.fixture
def rows(monkeypatch):
    """Rows served in place of the downloaded eval file; records the (path, format) the evaluator asked for."""
    served: dict = {"rows": [], "calls": []}

    def fake_load_eval_rows(dataset_path, file_format):
        served["calls"].append((dataset_path, file_format))
        return list(served["rows"])

    monkeypatch.setattr(instruct_text, "load_eval_rows", fake_load_eval_rows)
    return served


def test_instruct_rows_come_back_as_sorted_dicts_with_the_three_keys(rows):
    rows["rows"] = [
        {"instruct": "longer prompt", "input": None, "output": "yes"},
        {"instruct": "q", "input": None, "output": "a"},
        {"instruct": "masked only", "input": None, "output": None},
    ]
    tokenizer = FakeHfTokenizer()
    dataset = instruct_text._load_evaluation_dataset(TextEvalConfig(), INSTRUCT, "/data/eval.json", tokenizer, FileFormat.JSON)

    assert rows["calls"] == [("/data/eval.json", FileFormat.JSON)]
    assert all(isinstance(sample, dict) and set(sample) == {"input_ids", "attention_mask", "labels"} for sample in dataset)
    assert [tokenizer.decode(sample["input_ids"]) for sample in dataset] == ["qa|", "longer promptyes|"]
    first = dataset[0]
    assert first["attention_mask"] == [1, 1, 1]
    assert first["labels"] == [IGNORE_INDEX, ord("a") + 10, EOS]


def test_chat_rows_score_every_assistant_turn_and_its_turn_end(rows):
    rows["rows"] = [
        {"conversations": [
            {"from": "user", "value": "Hi"},
            {"from": "assistant", "value": "Hello"},
            {"from": "user", "value": "Bye"},
            {"from": "assistant", "value": "Ciao"},
        ]},
        {"conversations": [{"from": "user", "value": "only a question"}]},
    ]
    tokenizer = FakeHfTokenizer()
    [sample] = instruct_text._load_evaluation_dataset(TextEvalConfig(), CHAT, "/data/eval.json", tokenizer)

    scored = tokenizer.decode([label for label in sample["labels"] if label != IGNORE_INDEX])
    assert scored == f"Hello{IM_END}\nCiao{IM_END}\n"
    assert tokenizer.decode(sample["input_ids"]).startswith(f"{IM_START}user\nHi{IM_END}\n")


def test_sequence_len_ladder_retries_until_rows_fit(rows, monkeypatch):
    rows["rows"] = [{"instruct": "x" * 9, "input": None, "output": "y"}]  # 9 + 1 + EOS = 11 tokens
    seen: list[int] = []
    real_load = instruct_text._load_evaluation_dataset

    def spy(config, *args):
        seen.append(config.sequence_len)
        return real_load(config, *args)

    monkeypatch.setattr(instruct_text, "_load_evaluation_dataset", spy)
    dataset = instruct_text._load_evaluation_dataset_with_sequence_retries(
        TextEvalConfig(sequence_len=4), INSTRUCT, "/data/eval.json", FakeHfTokenizer(), _model(32)
    )
    # Ladder 4 -> 8 -> 16 -> 32; the first candidate with a surviving row wins.
    assert seen == [4, 8, 16]
    assert len(dataset) == 1 and len(dataset[0]["input_ids"]) == 11


def test_sequence_len_ladder_raises_with_the_candidates_when_nothing_fits(rows):
    rows["rows"] = [{"instruct": "x" * 40, "input": None, "output": "y"}]
    with pytest.raises(ValueError, match=r"\[4, 8, 16\]"):
        instruct_text._load_evaluation_dataset_with_sequence_retries(
            TextEvalConfig(sequence_len=4), INSTRUCT, "/data/eval.json", FakeHfTokenizer(model_max_length=16), _model(4096)
        )


@pytest.mark.parametrize("max_position_embeddings, expected", [(40960, 4096), (8192, 4096), (4000, 2000), (None, 4096)])
def test_eval_config_halves_the_window_for_short_context_models(max_position_embeddings, expected):
    config = common.text_eval_config_for_finetuned_model(_model(max_position_embeddings))
    assert config.sequence_len == expected
    assert config.starting_batch_size == 1


def test_finetuned_model_entry_point_rejects_non_text_dataset_types():
    args = SimpleNamespace(dataset_type=DpoDatasetType(), dataset="/data/eval.json", file_format=FileFormat.JSON)
    with pytest.raises(TypeError, match="instruct or chat"):
        instruct_text.evaluate_finetuned_model(args, _model(4096), FakeHfTokenizer())
