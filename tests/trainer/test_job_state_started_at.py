"""Tests for refreshing training started_at when the train container starts."""

from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace

import pytest

from core.models.task_models import TaskStatus
from trainer import job_state


@pytest.mark.asyncio
async def test_update_training_started_at_refreshes_clock(monkeypatch: pytest.MonkeyPatch):
    accept_time = datetime.utcnow() - timedelta(hours=1)
    task = SimpleNamespace(status=TaskStatus.TRAINING, started_at=accept_time)
    saved = {"n": 0}

    monkeypatch.setattr(job_state, "load_task_history", lambda: None)
    monkeypatch.setattr(job_state, "get_task", lambda task_id, hotkey: task)

    async def _save():
        saved["n"] += 1

    monkeypatch.setattr(job_state, "save_task_history", _save)

    container_start = datetime.utcnow()
    await job_state.update_training_started_at("task-1", "hotkey-a", started_at=container_start)

    assert task.started_at == container_start
    assert task.started_at > accept_time
    assert saved["n"] == 1


@pytest.mark.asyncio
async def test_update_training_started_at_skips_non_training(monkeypatch: pytest.MonkeyPatch):
    accept_time = datetime.utcnow() - timedelta(hours=1)
    task = SimpleNamespace(status=TaskStatus.FAILURE, started_at=accept_time)
    saved = {"n": 0}

    monkeypatch.setattr(job_state, "load_task_history", lambda: None)
    monkeypatch.setattr(job_state, "get_task", lambda task_id, hotkey: task)

    async def _save():
        saved["n"] += 1

    monkeypatch.setattr(job_state, "save_task_history", _save)

    await job_state.update_training_started_at("task-1", "hotkey-a", started_at=datetime.utcnow())

    assert task.started_at == accept_time
    assert saved["n"] == 0
