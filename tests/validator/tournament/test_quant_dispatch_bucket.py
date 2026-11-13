"""Tournament tasks reach training through three buckets: image, quant tournament (its round tasks and its seed tasks,
instruct seeds included) and everything else. Each bucket is fed only while it has fewer than PENDING_QUEUE_THRESHOLD_PER_TYPE
pending assignments, so a full bucket never blocks another one."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tournament.constants as t_cst
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.tournament import orchestrator
from validator.tournament.models import TournamentType


CONFIG = SimpleNamespace(psql_db=None)
THRESHOLD = t_cst.PENDING_QUEUE_THRESHOLD_PER_TYPE


def _task(task_type: TaskType) -> SimpleNamespace:
    return SimpleNamespace(task_id=uuid.uuid4(), task_type=task_type)


class Dispatch:
    """The orchestrator's SQL seams: which tournament each task belongs to, what is pending and what is waiting."""

    def __init__(self, monkeypatch) -> None:
        self.types: dict[str, TournamentType | None] = {}
        self.pending: list[SimpleNamespace] = []
        self.waiting: list[SimpleNamespace] = []
        self.processed: list[tuple[list, int]] = []

        async def tasks_with_status(status, psql_db, tournament_filter="all", benchmark_filter="include", **_):
            if status == TaskStatus.LOOKING_FOR_NODES and tournament_filter == "only":
                return list(self.waiting)
            return []

        async def process(tasks, config, priority):
            if priority == 2:  # tournament tasks; the organic and benchmark passes are not under test
                self.processed.append((list(tasks), priority))

        monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_training_tasks", AsyncMock(side_effect=self._pending))
        monkeypatch.setattr(orchestrator.tournament_sql, "get_tournament_type_for_task", AsyncMock(side_effect=self._type))
        monkeypatch.setattr(orchestrator.task_sql, "get_tasks_with_status", tasks_with_status)
        monkeypatch.setattr(orchestrator, "_process_tasks_for_training", process)

    async def _pending(self, psql_db, status):
        return list(self.pending)

    async def _type(self, task_id, psql_db):
        return self.types.get(task_id)

    def task(self, task_type: TaskType, tournament_type: TournamentType | None) -> SimpleNamespace:
        task = _task(task_type)
        self.types[str(task.task_id)] = tournament_type
        return task

    def add_pending(self, count: int, task_type: TaskType, tournament_type: TournamentType | None) -> None:
        for _ in range(count):
            self.pending.append(SimpleNamespace(task=self.task(task_type, tournament_type)))

    async def run(self) -> dict[str, list]:
        """Run one fetch cycle; the tournament tasks handed to training, by bucket."""
        await orchestrator._fetch_tournament_tasks_ready_to_train(CONFIG)
        buckets = {"text": [], "image": [], "quant": []}
        for tasks, priority in self.processed:
            if priority != 2:
                continue
            kinds = {self._bucket(task) for task in tasks}
            assert len(kinds) == 1, "one call must not mix buckets"
            buckets[kinds.pop()] += tasks
        return buckets

    def _bucket(self, task) -> str:
        if task.task_type == TaskType.IMAGETASK:
            return "image"
        return "quant" if self.types[str(task.task_id)] == TournamentType.QUANT else "text"


@pytest.fixture
def dispatch(monkeypatch):
    return Dispatch(monkeypatch)


async def test_a_full_text_bucket_does_not_block_a_quant_tournament_task(dispatch):
    dispatch.add_pending(THRESHOLD, TaskType.INSTRUCTTEXTTASK, TournamentType.TEXT)
    text = dispatch.task(TaskType.INSTRUCTTEXTTASK, TournamentType.TEXT)
    quant = dispatch.task(TaskType.QUANTFIXEDTASK, TournamentType.QUANT)
    dispatch.waiting = [text, quant]

    buckets = await dispatch.run()

    assert buckets["quant"] == [quant]
    assert buckets["text"] == []


async def test_a_full_quant_bucket_does_not_block_text(dispatch):
    dispatch.add_pending(THRESHOLD, TaskType.QUANTFREEFORMTASK, TournamentType.QUANT)
    text = dispatch.task(TaskType.INSTRUCTTEXTTASK, TournamentType.TEXT)
    quant = dispatch.task(TaskType.QUANTFIXEDTASK, TournamentType.QUANT)
    dispatch.waiting = [text, quant]

    buckets = await dispatch.run()

    assert buckets["text"] == [text]
    assert buckets["quant"] == []


async def test_quant_seed_tasks_go_to_the_quant_bucket_instruct_seeds_included(dispatch):
    instruct_seed = dispatch.task(TaskType.INSTRUCTTEXTTASK, TournamentType.QUANT)
    compression_seed = dispatch.task(TaskType.QUANTFREEFORMTASK, TournamentType.QUANT)
    text = dispatch.task(TaskType.INSTRUCTTEXTTASK, TournamentType.TEXT)
    dispatch.waiting = [instruct_seed, compression_seed, text]

    buckets = await dispatch.run()

    assert buckets["quant"] == [instruct_seed, compression_seed]
    assert buckets["text"] == [text]


async def test_pending_quant_assignments_are_counted_by_their_tournament_not_their_task_type(dispatch):
    dispatch.add_pending(THRESHOLD, TaskType.INSTRUCTTEXTTASK, TournamentType.QUANT)  # instruct seeds of a quant tournament
    text = dispatch.task(TaskType.INSTRUCTTEXTTASK, TournamentType.TEXT)
    quant = dispatch.task(TaskType.QUANTFIXEDTASK, TournamentType.QUANT)
    dispatch.waiting = [text, quant]

    buckets = await dispatch.run()

    assert buckets["text"] == [text] and buckets["quant"] == []


async def test_a_full_image_bucket_blocks_only_image(dispatch):
    dispatch.add_pending(THRESHOLD, TaskType.IMAGETASK, TournamentType.IMAGE)
    image = dispatch.task(TaskType.IMAGETASK, TournamentType.IMAGE)
    text = dispatch.task(TaskType.INSTRUCTTEXTTASK, TournamentType.TEXT)
    quant = dispatch.task(TaskType.QUANTFIXEDTASK, TournamentType.QUANT)
    dispatch.waiting = [image, text, quant]

    buckets = await dispatch.run()

    assert buckets == {"text": [text], "image": [], "quant": [quant]}


async def test_without_a_quant_tournament_text_and_image_buckets_are_mains(dispatch):
    dispatch.add_pending(THRESHOLD - 1, TaskType.INSTRUCTTEXTTASK, TournamentType.TEXT)
    dispatch.add_pending(THRESHOLD, TaskType.IMAGETASK, TournamentType.IMAGE)
    tasks = [
        dispatch.task(TaskType.INSTRUCTTEXTTASK, TournamentType.TEXT),
        dispatch.task(TaskType.DPOTASK, TournamentType.TEXT),
        dispatch.task(TaskType.GRPOTASK, None),
        dispatch.task(TaskType.IMAGETASK, TournamentType.IMAGE),
    ]
    dispatch.waiting = tasks

    buckets = await dispatch.run()

    assert buckets["text"] == tasks[:3]  # the text bucket (below the threshold) takes every non-image task
    assert buckets["image"] == []  # the image bucket is full
    assert buckets["quant"] == []
    assert [priority for _, priority in dispatch.processed] == [2]


async def test_the_quant_bucket_is_fed_in_its_own_call_at_tournament_priority(dispatch):
    quant = dispatch.task(TaskType.QUANTFIXEDTASK, TournamentType.QUANT)
    dispatch.waiting = [quant]

    await dispatch.run()

    assert dispatch.processed == [([quant], 2)]
