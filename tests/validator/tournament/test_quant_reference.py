"""Which bf16 model, and which stored reference key, a quant row is normalised by."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskType
from validator.tournament import quant_reference


def _task(start_point=TrainingStartPoint.DEFAULT):
    return SimpleNamespace(
        task_id="task",
        task_type=TaskType.QUANTFREEFORMTASK,
        training_start_point=start_point,
        model_id="org/common",
        augmented_model_id=None,
    )


@pytest.mark.parametrize(
    ("start_point", "expected"),
    [
        (TrainingStartPoint.OWN_R1_MODEL, "miner"),
        (TrainingStartPoint.OWN_UPSTREAM_MODEL, "miner"),
        (TrainingStartPoint.DEFAULT, ""),
        (TrainingStartPoint.PREVIOUS_WINNER, ""),
        (TrainingStartPoint.CONTINUOUS_SFT, ""),
    ],
)
def test_reference_hotkey_is_the_miners_only_on_an_own_model_task(start_point, expected) -> None:
    assert quant_reference.reference_hotkey_for(_task(start_point), "miner") == expected


def test_is_own_model_start_point() -> None:
    assert quant_reference.is_own_model_start_point(TrainingStartPoint.OWN_R1_MODEL)
    assert quant_reference.is_own_model_start_point(TrainingStartPoint.OWN_UPSTREAM_MODEL)
    assert not quant_reference.is_own_model_start_point(TrainingStartPoint.DEFAULT)
    assert not quant_reference.is_own_model_start_point(None)


class TestQuantReferenceModel:
    @pytest.fixture
    def starting_repos(self, monkeypatch):
        repos: dict[str, str | None] = {}
        monkeypatch.setattr(
            quant_reference.task_sql, "get_starting_model_repo", AsyncMock(side_effect=lambda _t, hotkey, _db: repos.get(hotkey))
        )
        return repos

    async def test_own_model_task_uses_the_miners_starting_model(self, starting_repos) -> None:
        starting_repos["miner"] = "org/miner-r1"

        task = _task(TrainingStartPoint.OWN_R1_MODEL)
        assert await quant_reference.quant_reference_model(task, "miner", psql_db=None) == "org/miner-r1"

    async def test_upstream_own_model_task_uses_the_miners_starting_model(self, starting_repos) -> None:
        starting_repos["miner"] = "org/miner-instruct-output"

        task = _task(TrainingStartPoint.OWN_UPSTREAM_MODEL)
        assert await quant_reference.quant_reference_model(task, "miner", psql_db=None) == "org/miner-instruct-output"
        assert await quant_reference.quant_reference_model(task, "other", psql_db=None) is None

    async def test_own_model_miner_without_a_starting_model_has_no_reference(self, starting_repos) -> None:
        task = _task(TrainingStartPoint.OWN_R1_MODEL)

        assert await quant_reference.quant_reference_model(task, "miner", psql_db=None) is None

    async def test_other_tasks_keep_the_common_input_whatever_the_hotkey(self, starting_repos) -> None:
        starting_repos["miner"] = "org/miner-r1"
        task = _task()
        task.augmented_model_id = "org/augmented"

        assert await quant_reference.quant_reference_model(task, "miner", psql_db=None) == "org/augmented"
