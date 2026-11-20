"""Every TaskType must be routed by each type dispatch, so a new type cannot silently fall through."""

import pytest
from pydantic import ValidationError

from core.models.task_models import TEXT_TASK_TYPES
from core.models.task_models import TaskType
from core.models.task_models import higher_is_better
from validator.db.sql.tasks import _get_specific_query_for_task_type
from validator.scoring.models import MinerResultsImage
from validator.scoring.models import MinerResultsText
from validator.scoring.tasks import _create_failed_miner_result
from validator.tasks.prep.constants import MODEL_PREP_ENABLED_BY_TASK_TYPE


ALL_TASK_TYPES = list(TaskType)


def test_every_type_is_text_or_image():
    assert TEXT_TASK_TYPES | {TaskType.IMAGETASK} == set(TaskType)
    assert TaskType.IMAGETASK not in TEXT_TASK_TYPES


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_miner_results_text_accepts_exactly_text_types(task_type):
    kwargs = dict(hotkey="hk", test_loss=1.0, synth_loss=1.0, is_finetune=True, task_type=task_type)
    if task_type in TEXT_TASK_TYPES:
        assert MinerResultsText(**kwargs).task_type == task_type
    else:
        with pytest.raises(ValidationError):
            MinerResultsText(**kwargs)


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_failed_miner_result_class_matches_type(task_type):
    result = _create_failed_miner_result("hk", "x", task_type)
    expected = MinerResultsText if task_type in TEXT_TASK_TYPES else MinerResultsImage
    assert type(result) is expected


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_batch_sql_query_exists_for_type(task_type):
    query = _get_specific_query_for_task_type(task_type.value)
    assert isinstance(query, str) and query.strip()


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_model_prep_flag_exists_for_type(task_type):
    assert task_type in MODEL_PREP_ENABLED_BY_TASK_TYPE


@pytest.mark.parametrize("task_type", ALL_TASK_TYPES)
def test_higher_is_better_only_for_reward_types(task_type):
    assert higher_is_better(task_type) is (task_type in {TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK})

