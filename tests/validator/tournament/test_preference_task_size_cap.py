from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import validator.tournament.constants as t_cst
from core.models.task_models import TaskType
from validator.tournament import task_creator


class SizeRecorder:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, _keypair, smallest_size_b: float = 0.1, largest_size_b: float = 12.0):
        self.calls.append({"smallest": smallest_size_b, "largest": largest_size_b})
        return MagicMock()


@pytest.fixture
def sizes(monkeypatch) -> SizeRecorder:
    recorder = SizeRecorder()
    monkeypatch.setattr(task_creator, "_get_text_models", recorder)
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", MagicMock())
    monkeypatch.setattr(task_creator, "_get_dpo_datasets", MagicMock())
    monkeypatch.setattr(task_creator, "_create_task_by_type", AsyncMock(return_value=SimpleNamespace(task_id="new")))
    return recorder


def _task(task_type: TaskType, params_b: float) -> SimpleNamespace:
    return SimpleNamespace(
        task_id="old",
        task_type=task_type,
        model_params_count=int(params_b * t_cst.MODEL_PARAMS_TO_BILLIONS),
    )


@pytest.mark.parametrize("task_type", [TaskType.DPOTASK, TaskType.GRPOTASK])
@pytest.mark.parametrize("params_b", [7.5, 8.0, 12.0])
async def test_preference_replacement_never_draws_above_the_cap(sizes, task_type, params_b):
    await task_creator.create_new_task_of_same_type(_task(task_type, params_b), MagicMock())

    (call,) = sizes.calls
    assert call["largest"] <= t_cst.PREFERENCE_TASK_MAX_SIZE_B
    assert call["smallest"] <= call["largest"]


@pytest.mark.parametrize("task_type", [TaskType.DPOTASK, TaskType.GRPOTASK])
async def test_preference_replacement_of_an_unsized_task_is_capped(sizes, task_type):
    await task_creator.create_new_task_of_same_type(_task(task_type, 0), MagicMock())

    (call,) = sizes.calls
    assert call["largest"] == min(t_cst.DEFAULT_MODEL_MAX_SIZE_B, t_cst.PREFERENCE_TASK_MAX_SIZE_B)


async def test_instruct_replacement_keeps_its_size_band(sizes):
    await task_creator.create_new_task_of_same_type(_task(TaskType.INSTRUCTTEXTTASK, 10.0), MagicMock())

    (call,) = sizes.calls
    assert call["largest"] == pytest.approx(10.0 * t_cst.MODEL_SIZE_RANGE_MULTIPLIER_MAX)
    assert call["smallest"] == pytest.approx(10.0 * t_cst.MODEL_SIZE_RANGE_MULTIPLIER_MIN)


async def test_boss_round_preference_tasks_draw_from_the_capped_pool(sizes, monkeypatch):
    continuous = SimpleNamespace(task_id="cont", task_type=TaskType.CHATTASK)
    monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_creator, "_get_existing_tasks", AsyncMock(return_value=[continuous]))
    monkeypatch.setattr(task_creator.t_cst, "is_continuous_sft_task", lambda task: task.task_id == "cont")
    monkeypatch.setattr(task_creator.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=None))
    created = AsyncMock(side_effect=lambda task_type, *_args, **_kwargs: SimpleNamespace(task_id=task_type.value))
    monkeypatch.setattr(task_creator, "_create_single_new_text_task", created)

    tasks = await task_creator._create_seeded_text_boss_round_tasks("tourn", "round", MagicMock())

    assert [task.task_id for task in tasks][0] == "cont"
    assert sizes.calls == [{"smallest": 0.1, "largest": t_cst.PREFERENCE_TASK_MAX_SIZE_B}]
