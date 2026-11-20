"""_prep_task: a compression task whose prep produced no calibration set is stored as PREP_TASK_FAILURE and alerts."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.models.dataset_models import InstructTextDatasetType
from core.models.task_models import TaskStatus
from validator.lifecycle import tasks as lifecycle_tasks
from validator.tasks.datasets import preparation
from validator.tasks.datasets.calibration import CalibrationUnavailableError
from validator.tasks.models import QuantFreeformRawTask


CONFIG = SimpleNamespace(psql_db="db", keypair=None)


def _task() -> QuantFreeformRawTask:
    return QuantFreeformRawTask(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.PENDING,
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 14),
        dataset_type=InstructTextDatasetType(field_instruction="q", field_output="a"),
    )


@pytest.fixture
def seams(monkeypatch):
    stored: list[str] = []

    async def update_task(task, psql_db):
        stored.append(task.status)
        return task

    prep = AsyncMock()
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task", update_task)
    monkeypatch.setattr(lifecycle_tasks, "get_task_config", lambda task: SimpleNamespace(task_prep_function=prep))
    notify = AsyncMock()
    monkeypatch.setattr(lifecycle_tasks, "notify_evaluation_exception", notify)
    return SimpleNamespace(stored=stored, prep=prep, notify=notify)


async def test_a_calibration_failure_stores_prep_task_failure_and_alerts(seams):
    task = _task()
    seams.prep.side_effect = CalibrationUnavailableError("only 3/9 rows survive")

    await lifecycle_tasks._prep_task(task, CONFIG)

    assert seams.stored == [TaskStatus.PREPARING_DATA, TaskStatus.PREP_TASK_FAILURE]
    seams.notify.assert_awaited_once()
    kwargs = seams.notify.await_args.kwargs
    assert kwargs["task_id"] == str(task.task_id) and kwargs["task_type"] == task.task_type
    assert "no calibration set" in kwargs["context"] and "not released to miners" in kwargs["context"]
    assert isinstance(kwargs["error"], CalibrationUnavailableError)


async def test_another_prep_failure_stores_prep_task_failure_without_the_alert(seams):
    seams.prep.side_effect = ValueError("dataset too small")

    await lifecycle_tasks._prep_task(_task(), CONFIG)

    assert seams.stored == [TaskStatus.PREPARING_DATA, TaskStatus.PREP_TASK_FAILURE]
    seams.notify.assert_not_awaited()


async def test_a_prep_that_returns_a_task_without_calibration_cannot_be_stored_as_ready_for_miners(seams, monkeypatch):
    """The data layer is the last line: update_task refuses LOOKING_FOR_NODES, and the failure write still succeeds."""
    real_sql = lifecycle_tasks.tasks_sql
    task = _task()
    seams.prep.side_effect = lambda *args: task  # prep "succeeds" with no calibration set
    monkeypatch.setattr(real_sql, "get_task", AsyncMock(return_value=None))
    stored: list[str] = []

    async def guarded_update(updated, psql_db):
        real_sql._reject_quant_without_calibration(updated)
        stored.append(updated.status)

    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task", guarded_update)

    await lifecycle_tasks._prep_task(task, CONFIG)

    assert stored == [TaskStatus.PREPARING_DATA, TaskStatus.PREP_TASK_FAILURE]
    seams.notify.assert_not_awaited()  # a plain ValueError from the guard, not a calibration prep failure


async def test_a_given_split_without_calibration_rows_ends_prep_task_failure_with_the_alert(seams):
    """The real prep function: a non-continuous compression task given a split with no calibration rows fails before any IO."""
    task = _task()
    task.training_data, task.test_data = "s3://bucket/train.json", "s3://bucket/test.json"
    seams.prep.side_effect = preparation.prepare_text_task

    await lifecycle_tasks._prep_task(task, CONFIG)

    assert seams.stored == [TaskStatus.PREPARING_DATA, TaskStatus.PREP_TASK_FAILURE]
    seams.notify.assert_awaited_once()
    assert "input model was trained on" in str(seams.notify.await_args.kwargs["error"])
