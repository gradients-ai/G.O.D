from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.tournament import orchestrator
from validator.tournament import task_creator


def _task(task_id: str, hours: float):
    return SimpleNamespace(task_id=task_id, hours_to_complete=hours, status=TaskStatus.PENDING)


async def test_draw_keeps_natural_hours_and_retires_longer_tasks(monkeypatch):
    long_task = _task("long", 4.0)
    short_task = _task("short", 2.5)
    create = AsyncMock(side_effect=[long_task, short_task])
    update = AsyncMock()
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", create)
    monkeypatch.setattr(task_creator.task_sql, "update_task", update)

    task = await task_creator._draw_round_one_text_task_within_hours(MagicMock(), None, None)

    assert task is short_task
    assert short_task.hours_to_complete == 2.5
    assert long_task.status == TaskStatus.FAILURE
    assert update.await_args.args[0] is long_task


async def test_draw_raises_when_every_task_is_too_long(monkeypatch):
    monkeypatch.setattr(task_creator.t_cst, "ROUND_ONE_TEXT_DURATION_DRAW_ATTEMPTS", 2)
    draws = [_task("a", 3.0), _task("b", 3.25)]
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", AsyncMock(side_effect=draws))
    monkeypatch.setattr(task_creator.task_sql, "update_task", AsyncMock())

    with pytest.raises(RuntimeError, match="within 2.5h"):
        await task_creator._draw_round_one_text_task_within_hours(MagicMock(), None, None)

    assert all(task.status == TaskStatus.FAILURE for task in draws)


def _measured_task(hours: float):
    return SimpleNamespace(
        task_id="task",
        hours_to_complete=hours,
        baseline_stats=object(),
        task_type=TaskType.INSTRUCTTEXTTASK,
        model_id="model",
        model_params_count=1_000_000_000,
        training_start_point=None,
        ds="dataset",
        status=TaskStatus.PREPARING_DATA,
        termination_at=None,
    )


async def test_post_prep_over_limit_replaces_without_shortening(monkeypatch):
    task = _measured_task(1.5)
    monkeypatch.setattr(orchestrator, "compute_hours_from_baseline_stats", lambda *args, **kwargs: 4.0)
    monkeypatch.setattr(orchestrator, "_max_training_hours_for_task", AsyncMock(return_value=2.5))
    update = AsyncMock()
    monkeypatch.setattr(orchestrator.task_sql, "update_task", update)

    replaced = await orchestrator._apply_measured_training_hours(task, MagicMock())

    assert replaced is True
    assert task.hours_to_complete == 1.5
    assert task.status == TaskStatus.PREP_TASK_FAILURE
    update.assert_awaited_once()


async def test_post_prep_within_limit_keeps_measured_hours(monkeypatch):
    task = _measured_task(1.5)
    monkeypatch.setattr(orchestrator, "compute_hours_from_baseline_stats", lambda *args, **kwargs: 2.25)
    monkeypatch.setattr(orchestrator, "_max_training_hours_for_task", AsyncMock(return_value=2.5))
    monkeypatch.setattr(orchestrator.task_sql, "update_task", AsyncMock())

    replaced = await orchestrator._apply_measured_training_hours(task, MagicMock())

    assert replaced is False
    assert task.hours_to_complete == 2.25
    assert task.status == TaskStatus.PREPARING_DATA
