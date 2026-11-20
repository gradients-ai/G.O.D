"""Quant chat labels on a real tokenizer equal chat_labels.label_chat's (offline HF cache, CPU only).

Skipped where transformers is not installed; run with transformers installed and HF_HUB_OFFLINE=1.
"""

import pytest

from core.models.dataset_models import ChatTemplateDatasetType
from validator.evaluation.chat_labels import IGNORE_INDEX
from validator.evaluation.chat_labels import ChatLabelled
from validator.evaluation.chat_labels import ChatMessage
from validator.evaluation.chat_labels import label_chat
from validator.evaluation.chat_templates import HfChatTokenizer
from validator.evaluation.chat_templates import resolve_chat_template
from validator.tasks.quant_eval_set import build_quant_eval_set


transformers = pytest.importorskip("transformers")
quant_tokenizer = pytest.importorskip("validator.tasks.quant_tokenizer")


REPO = "Qwen/Qwen2.5-0.5B"
TURNS = [
    ("system", "Be terse."),
    ("user", "Capital of France?"),
    ("assistant", "Paris."),
    ("user", "Germany?"),
    ("assistant", "Berlin."),
]
ROW = {"conversations": [{"from": role, "value": value} for role, value in TURNS]}


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return quant_tokenizer.HfQuantTokenizer(REPO)
    except Exception as exc:  # noqa: BLE001 - not cached / incomplete snapshot
        pytest.skip(f"{REPO} not usable from the local HF cache: {type(exc).__name__}")


@pytest.mark.parametrize("chat_template", ["chatml", "tokenizer_default"])
def test_quant_chat_example_equals_label_chat(tokenizer, chat_template):
    hf = tokenizer._tokenizer
    adapter = HfChatTokenizer(hf, resolve_chat_template(chat_template, hf))
    expected = label_chat(adapter, [ChatMessage(role=role, content=value) for role, value in TURNS])

    build = build_quant_eval_set([ROW], ChatTemplateDatasetType(chat_template=chat_template), tokenizer, 4096)

    assert isinstance(expected, ChatLabelled)
    assert [(e.input_ids, e.labels) for e in build.examples] == [(expected.input_ids, expected.labels)]


def test_chatml_on_qwen_base_scores_both_turns_through_im_end(tokenizer):
    build = build_quant_eval_set([ROW], ChatTemplateDatasetType(chat_template="chatml"), tokenizer, 4096)
    example = build.examples[0]
    hf = tokenizer._tokenizer

    scored = [hf.decode(token) for token, label in zip(example.input_ids, example.labels) if label != IGNORE_INDEX]
    assert "".join(scored) == "Paris.<|im_end|>\nBerlin.<|im_end|>\n"
