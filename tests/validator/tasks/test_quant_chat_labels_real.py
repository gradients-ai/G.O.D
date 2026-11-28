"""Quant chat labels on real tokenizers equal the text eval's axolotl labels (offline HF cache, CPU only).

The golden fixture holds rows with the input ids and labels the text eval's axolotl `chat_template` tokenisation
produced for them (Qwen/Qwen3-8B-Base, `tokenizer_default`). Skipped where transformers is not installed or the
tokenizer is not in the local HF cache; run with transformers installed and HF_HUB_OFFLINE=1.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from core.models.dataset_models import ChatTemplateDatasetType
from validator.evaluation.chat_labels import IGNORE_INDEX
from validator.tasks.quant_eval_set import SkipReason
from validator.tasks.quant_eval_set import build_quant_eval_set
from validator.tasks.quant_eval_set import tokenize_chat


transformers = pytest.importorskip("transformers")
quant_tokenizer = pytest.importorskip("validator.tasks.quant_tokenizer")


GOLDEN_PATH = Path(__file__).parent / "fixtures" / "axolotl_chat_labels_qwen3.json"
QWEN25_REPO = "Qwen/Qwen2.5-0.5B"
TURNS = [
    ("system", "Be terse."),
    ("user", "Capital of France?"),
    ("assistant", "Paris."),
    ("user", "Germany?"),
    ("assistant", "Berlin."),
]
ROW = {"conversations": [{"from": role, "value": value} for role, value in TURNS]}


class GoldenCase(BaseModel):
    name: str
    row: dict
    input_ids: list[int]
    labels: list[int]


class Golden(BaseModel):
    repo: str
    chat_template: str
    cases: list[GoldenCase]


GOLDEN = Golden.model_validate_json(GOLDEN_PATH.read_text())


def _load(repo: str) -> "quant_tokenizer.HfQuantTokenizer":
    try:
        return quant_tokenizer.HfQuantTokenizer(repo)
    except Exception as exc:  # noqa: BLE001 - not cached / incomplete snapshot
        pytest.skip(f"{repo} not usable from the local HF cache: {type(exc).__name__}")


@pytest.fixture(scope="module")
def golden_tokenizer():
    return _load(GOLDEN.repo)


@pytest.mark.parametrize("case", GOLDEN.cases, ids=[case.name for case in GOLDEN.cases])
def test_chat_labels_equal_axolotls(golden_tokenizer, case):
    dataset_type = ChatTemplateDatasetType(chat_template=GOLDEN.chat_template)
    labeller = golden_tokenizer.chat_labeller(GOLDEN.chat_template)

    example = tokenize_chat(labeller, case.row, dataset_type, golden_tokenizer.eos_token_id)

    assert not isinstance(example, SkipReason)
    assert (example.input_ids, example.labels) == (case.input_ids, case.labels)


def test_the_eval_set_keeps_every_golden_row_with_a_scored_token(golden_tokenizer):
    dataset_type = ChatTemplateDatasetType(chat_template=GOLDEN.chat_template)

    build = build_quant_eval_set([case.row for case in GOLDEN.cases], dataset_type, golden_tokenizer, 65536)

    scored = [case for case in GOLDEN.cases if any(label != IGNORE_INDEX for label in case.labels[1:])]
    assert [(e.input_ids, e.labels) for e in build.examples] == [(c.input_ids, c.labels) for c in scored]
    assert build.n_skipped_empty == len(GOLDEN.cases) - len(scored)


def test_qwen3_base_scores_the_answer_but_not_the_template_think_block_or_turn_end(golden_tokenizer):
    case = next(case for case in GOLDEN.cases if case.name == "single_turn")
    hf = golden_tokenizer._tokenizer

    scored = hf.decode([token for token, label in zip(case.input_ids, case.labels) if label != IGNORE_INDEX])
    assert "<think>" in hf.decode(case.input_ids)
    assert scored == case.row["conversations"][-1]["value"].split("</think>")[-1].strip()


def test_chatml_on_qwen_base_scores_both_turns_without_a_turn_end():
    """Qwen base's eos is <|endoftext|>, which ChatML never emits, so no turn end is scored."""
    tokenizer = _load(QWEN25_REPO)
    build = build_quant_eval_set([ROW], ChatTemplateDatasetType(chat_template="chatml"), tokenizer, 4096)
    example = build.examples[0]
    hf = tokenizer._tokenizer

    scored = [hf.decode(token) for token, label in zip(example.input_ids, example.labels) if label != IGNORE_INDEX]
    assert "".join(scored) == "Paris.Berlin."
