"""A quant row that raises while being scheduled must not hold the text and image rows behind it; a text or image row that
raises behaves as it always did (logged, retried in place)."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.models.task_models import TaskType
from validator.tournament import orchestrator
from validator.tournament.seed_tasks import INPUT_SMALL_SEED_TRAINING_RANK
from validator.tournament.seed_tasks import NON_SEED_TRAINING_RANK


CONFIG = SimpleNamespace(psql_db=None)


def _row(task_type: TaskType, hotkey: str = "5Miner") -> SimpleNamespace:
    task = SimpleNamespace(task_id=uuid.uuid4(), task_type=task_type, hours_to_complete=2.0)
    return SimpleNamespace(
        task=task,
        hotkey=hotkey,
        n_training_attempts=0,
        training_repo="https://github.com/x/repo",
        training_commit_hash="c" * 40,
        github_token=None,
        requested_datasets=None,
    )


class Scheduling:
    def __init__(self, monkeypatch) -> None:
        self.ranks: dict[str, int] = {}
        self.reference_raises: set[uuid.UUID] = set()
        self.build_raises: set[uuid.UUID] = set()
        self.built: list[uuid.UUID] = []
        self.sleeps: list[float] = []
        self.on_sleep = None

        async def reference_ready(task, hotkey, config):
            if task.task_id in self.reference_raises:
                raise RuntimeError("reference lookup blew up")
            return True

        async def create_request(task, *_args):
            if task.task_id in self.build_raises:
                raise RuntimeError("presign failed")
            self.built.append(task.task_id)
            return SimpleNamespace()

        async def sleep(seconds):
            self.sleeps.append(seconds)
            if self.on_sleep is not None:
                self.on_sleep()

        async def training_ranks(task_ids, _db):
            return {task_id: self.ranks.get(task_id, NON_SEED_TRAINING_RANK) for task_id in task_ids}

        monkeypatch.setattr(orchestrator, "training_ranks", training_ranks)
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value="tourn"))
        monkeypatch.setattr(orchestrator, "_reference_ready", reference_ready)
        monkeypatch.setattr(orchestrator, "get_task_gpu_requirement", lambda task: SimpleNamespace(value="h100"))
        monkeypatch.setattr(orchestrator, "_update_all_trainers_gpu_availability", AsyncMock())
        monkeypatch.setattr(orchestrator, "_check_suitable_gpus", AsyncMock(return_value=("1.2.3.4", [0], "NVLINK")))
        monkeypatch.setattr(orchestrator, "_create_training_request", create_request)
        monkeypatch.setattr(orchestrator, "start_training_task", AsyncMock(return_value=True))
        monkeypatch.setattr(orchestrator.tournament_sql, "update_tournament_task_training_status", AsyncMock())
        monkeypatch.setattr(orchestrator.tournament_sql, "update_gpu_availability", AsyncMock())
        monkeypatch.setattr(orchestrator.gpu_cost_sql, "start_cost_run", AsyncMock())
        monkeypatch.setattr(orchestrator.asyncio, "sleep", sleep)


@pytest.fixture
def scheduling(monkeypatch) -> Scheduling:
    return Scheduling(monkeypatch)


class TestAQuantRowThatRaises:
    async def test_a_raising_reference_check_does_not_abort_the_cycle(self, scheduling):
        quant, text = _row(TaskType.QUANTFIXEDTASK), _row(TaskType.INSTRUCTTEXTTASK)
        scheduling.reference_raises.add(quant.task.task_id)
        pending = [text, quant]  # the last row is taken first

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.built == [text.task.task_id]
        assert pending == []

    async def test_a_raising_request_build_pops_the_row_and_the_next_is_scheduled(self, scheduling):
        quant, text = _row(TaskType.QUANTFIXEDTASK), _row(TaskType.INSTRUCTTEXTTASK)
        scheduling.build_raises.add(quant.task.task_id)
        pending = [text, quant]

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.built == [text.task.task_id]
        assert pending == []
        assert orchestrator.cst.TRAINING_START_RETRY_INTERVAL not in scheduling.sleeps

    async def test_an_instruct_seed_row_of_a_quant_tournament_is_treated_the_same(self, scheduling):
        seed, text = _row(TaskType.INSTRUCTTEXTTASK), _row(TaskType.INSTRUCTTEXTTASK)
        scheduling.ranks[str(seed.task.task_id)] = INPUT_SMALL_SEED_TRAINING_RANK
        scheduling.build_raises.add(seed.task.task_id)
        pending = [text, seed]

        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.built == [text.task.task_id]
        assert pending == []


class TestATextRowThatRaises:
    async def test_a_raising_request_build_is_retried_in_place_with_the_same_sleep(self, scheduling):
        text, other = _row(TaskType.INSTRUCTTEXTTASK), _row(TaskType.IMAGETASK)
        scheduling.build_raises.add(text.task.task_id)
        pending = [other, text]
        scheduling.on_sleep = scheduling.build_raises.clear  # the failure clears after the first retry sleep
        await orchestrator.schedule_tasks_for_training(pending, CONFIG)

        assert scheduling.sleeps.count(orchestrator.cst.TRAINING_START_RETRY_INTERVAL) == 1
        assert scheduling.built == [text.task.task_id, other.task.task_id]

    async def test_a_raising_reference_check_on_a_text_row_still_propagates(self, scheduling):
        text = _row(TaskType.INSTRUCTTEXTTASK)
        scheduling.reference_raises.add(text.task.task_id)

        with pytest.raises(RuntimeError, match="reference lookup blew up"):
            await orchestrator.schedule_tasks_for_training([text], CONFIG)
