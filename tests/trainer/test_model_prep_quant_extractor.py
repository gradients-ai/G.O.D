"""Model-prep text extractor for quant tasks: chosen from the records, since a quant task carries instruct or chat data.

stats.py imports torch, so this runs only with --run-gpu (uv run --extra dev --extra gpu pytest --run-gpu ...).
"""

import pytest

from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskType


pytestmark = pytest.mark.gpu

CHAT_RECORDS = [{"conversations": [{"from": "user", "value": "hi"}, {"from": "assistant", "value": "yo"}]}]
INSTRUCT_RECORDS = [{"instruct": "q", "output": "a"}]


@pytest.fixture
def stats():
    pytest.importorskip("torch", reason="model-prep stats require torch")
    pytest.importorskip("transformers", reason="model-prep stats require transformers")
    from trainer.model_prep import stats

    return stats


@pytest.mark.parametrize("task_type", sorted(QUANT_TASK_TYPES))
def test_quant_chat_records_use_the_chat_extractor(stats, task_type):
    extractor = stats.text_extractor_for(task_type, CHAT_RECORDS)

    assert extractor is stats._extract_chat_texts
    assert extractor(CHAT_RECORDS) == [("hi", "yo")]


@pytest.mark.parametrize("task_type", sorted(QUANT_TASK_TYPES))
@pytest.mark.parametrize("records", [INSTRUCT_RECORDS, []])
def test_quant_instruct_records_use_the_instruct_extractor(stats, task_type, records):
    assert stats.text_extractor_for(task_type, records) is stats._extract_instruct_texts


def test_chat_and_instruct_tasks_keep_their_extractors(stats):
    assert stats.text_extractor_for(TaskType.CHATTASK, INSTRUCT_RECORDS) is stats._extract_chat_texts
    assert stats.text_extractor_for(TaskType.INSTRUCTTEXTTASK, CHAT_RECORDS) is stats._extract_instruct_texts
