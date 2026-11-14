"""_prep_task: a prep that cannot fetch its data (DataFetchError) leaves the task PENDING as it was, backs off and alerts once;
any other prep failure still ends PREP_TASK_FAILURE."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.models.task_models import TaskStatus
from validator.lifecycle import constants as lifecycle_cst
from validator.lifecycle import tasks as lifecycle_tasks
from validator.tasks.datasets.fetch import DataFetchError
from validator.tasks.datasets.preparation import NearDuplicateDatasetError
from validator.tasks.models import InstructTextRawTask


CONFIG = SimpleNamespace(psql_db="db", keypair=None)


def _task() -> InstructTextRawTask:
    return InstructTextRawTask(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.PENDING,
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 14),
        field_instruction="q",
        field_output="a",
    )


@pytest.fixture
def seams(monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    stored: list = []

    async def update_task(task, psql_db):
        stored.append(task.model_copy(deep=True))
        return task

    async def prep_mutating_then(task, *args):
        task.model_id = "org/yarn-extended"  # prep mutates the task before it fails
        raise seams_ns.error

    seams_ns = SimpleNamespace(stored=stored, clock=clock, error=DataFetchError("could not download chunk: 404"))
    prep = AsyncMock(side_effect=prep_mutating_then)
    seams_ns.prep = prep
    seams_ns.notify = AsyncMock()
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task", update_task)
    monkeypatch.setattr(lifecycle_tasks, "get_task_config", lambda task: SimpleNamespace(task_prep_function=prep))
    monkeypatch.setattr(lifecycle_tasks, "notify_evaluation_exception", seams_ns.notify)
    monkeypatch.setattr(lifecycle_tasks.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(lifecycle_tasks, "_PREP_FETCH_BACKOFF", {})
    return seams_ns


async def test_a_fetch_failure_leaves_the_task_pending_as_it_was_before_prep(seams):
    task = _task()

    await lifecycle_tasks._prep_task(task, CONFIG)

    assert [row.status for row in seams.stored] == [TaskStatus.PREPARING_DATA, TaskStatus.PENDING]
    assert seams.stored[-1].model_id == "org/base"  # the pre-prep task, not prep's half-mutated copy
    backoff = lifecycle_tasks._PREP_FETCH_BACKOFF[str(task.task_id)]
    assert backoff.failures == 1 and backoff.retry_at == 1000.0 + lifecycle_cst.PREP_FETCH_RETRY_BASE_SECONDS
    seams.notify.assert_not_awaited()


async def test_repeated_fetch_failures_alert_exactly_once(seams):
    task = _task()

    for _ in range(lifecycle_cst.PREP_FETCH_FAILURES_BEFORE_ALERT + 3):
        await lifecycle_tasks._prep_task(_task().model_copy(update={"task_id": task.task_id}), CONFIG)

    seams.notify.assert_awaited_once()
    kwargs = seams.notify.await_args.kwargs
    assert kwargs["task_id"] == str(task.task_id) and isinstance(kwargs["error"], DataFetchError)
    assert "not failed or replaced" in kwargs["context"]
    assert all(row.status != TaskStatus.PREP_TASK_FAILURE for row in seams.stored)


def test_the_backoff_doubles_from_the_base_and_is_capped():
    delays = [lifecycle_tasks.prep_fetch_retry_delay_seconds(n) for n in range(1, 12)]

    assert delays[:3] == [60, 120, 240]
    assert max(delays) == delays[-1] == lifecycle_cst.PREP_FETCH_RETRY_MAX_SECONDS
    assert delays == sorted(delays)


async def test_a_backing_off_task_is_not_prepped_until_its_backoff_has_passed(seams, monkeypatch):
    task = _task()
    monkeypatch.setattr(lifecycle_tasks, "clean_all_hf_datasets_cache", lambda: None)

    async def pending(status, psql_db):
        return [_task().model_copy(update={"task_id": task.task_id})]

    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "get_tasks_with_status", pending)

    await lifecycle_tasks._processing_pending_tasks(CONFIG)
    assert seams.prep.await_count == 1
    seams.clock.now += lifecycle_cst.PREP_FETCH_RETRY_BASE_SECONDS - 1
    await lifecycle_tasks._processing_pending_tasks(CONFIG)
    assert seams.prep.await_count == 1  # still backing off
    seams.clock.now += 1
    await lifecycle_tasks._processing_pending_tasks(CONFIG)
    assert seams.prep.await_count == 2
    assert lifecycle_tasks._PREP_FETCH_BACKOFF[str(task.task_id)].failures == 2


async def test_a_deterministic_rejection_still_ends_prep_task_failure(seams):
    task = _task()
    await lifecycle_tasks._prep_task(task, CONFIG)  # one fetch failure first
    seams.error = NearDuplicateDatasetError("near-duplicate rate 0.40 >= 0.15")

    await lifecycle_tasks._prep_task(_task().model_copy(update={"task_id": task.task_id}), CONFIG)

    assert seams.stored[-1].status == TaskStatus.PREP_TASK_FAILURE
    assert str(task.task_id) not in lifecycle_tasks._PREP_FETCH_BACKOFF
    seams.notify.assert_not_awaited()


async def test_a_successful_prep_clears_the_backoff(seams):
    task = _task()
    await lifecycle_tasks._prep_task(task, CONFIG)
    seams.prep.side_effect = lambda prepped, *args: prepped

    await lifecycle_tasks._prep_task(_task().model_copy(update={"task_id": task.task_id}), CONFIG)

    assert seams.stored[-1].status in (TaskStatus.LOOKING_FOR_NODES, TaskStatus.AWAITING_MODEL_PREP)
    assert str(task.task_id) not in lifecycle_tasks._PREP_FETCH_BACKOFF
