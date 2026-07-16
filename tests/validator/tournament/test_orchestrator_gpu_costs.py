from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from validator.tournament import orchestrator


def test_gpu_cost_run_keys_are_stable_per_work_identity():
    assert orchestrator._training_cost_run_key("task-1", "hotkey-1", 2) == "training:task-1:hotkey-1:2"
    assert orchestrator._prep_cost_run_key("task-1") == "prep:task-1:task"
    assert orchestrator._prep_cost_run_key("task-1", "hotkey-1") == "prep:task-1:miner:hotkey-1"


@pytest.mark.asyncio
async def test_third_prep_failure_sends_one_notification(monkeypatch):
    finish = AsyncMock(
        return_value={
            "task_id": "task-1",
            "prep_failure_count": 3,
            "metadata": {"prep_identity": "miner:hotkey-1"},
        }
    )
    notify = AsyncMock()
    monkeypatch.setattr(orchestrator.gpu_cost_sql, "finish_cost_run", finish)
    monkeypatch.setattr(orchestrator, "notify_model_prep_failure_limit", notify)
    config = SimpleNamespace(psql_db=object(), discord_url="https://discord.invalid/webhook")

    await orchestrator._finish_prep_cost_run("prep:task-1:miner:hotkey-1", False, config)

    finish.assert_awaited_once()
    notify.assert_awaited_once_with(
        task_id="task-1",
        prep_identity="miner:hotkey-1",
        discord_url=config.discord_url,
    )


@pytest.mark.asyncio
async def test_duplicate_or_non_third_prep_finish_does_not_notify(monkeypatch):
    finish = AsyncMock(side_effect=[None, {"task_id": "task-1", "prep_failure_count": 2, "metadata": {}}])
    notify = AsyncMock()
    monkeypatch.setattr(orchestrator.gpu_cost_sql, "finish_cost_run", finish)
    monkeypatch.setattr(orchestrator, "notify_model_prep_failure_limit", notify)
    config = SimpleNamespace(psql_db=object(), discord_url=None)

    await orchestrator._finish_prep_cost_run("prep:task-1:task", False, config)
    await orchestrator._finish_prep_cost_run("prep:task-1:task", False, config)

    notify.assert_not_awaited()
