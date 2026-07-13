from types import SimpleNamespace

import pytest

from validator.lifecycle import tasks as lifecycle_tasks


@pytest.mark.asyncio
async def test_pending_evaluation_row_is_not_picked_when_basilica_is_full(monkeypatch):
    status_updates = []
    evaluations = []

    async def rows_by_status(_task_id, status, _psql_db):
        if status == "pending":
            return [{"hotkey": "hotkey-1", "evaluation_status": "pending"}]
        return []

    async def no_capacity():
        return False

    async def update_status(*args):
        status_updates.append(args)

    async def evaluate(*args):
        evaluations.append(args)

    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "get_task_evaluations_by_status", rows_by_status)
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task_evaluations_status", update_status)
    monkeypatch.setattr(lifecycle_tasks, "basilica_deployment_capacity_available", no_capacity)
    monkeypatch.setattr(lifecycle_tasks, "evaluate_and_score_hotkeys", evaluate)

    task = SimpleNamespace(task_id="task-1", task_type=None)
    config = SimpleNamespace(psql_db=object())

    await lifecycle_tasks._evaluate_pending_pairs_for_task(task, num_gpus=1, config=config)

    assert status_updates == []
    assert evaluations == []


@pytest.mark.asyncio
async def test_finalized_task_triggers_drained_basilica_cleanup(monkeypatch):
    calls: list[str] = []

    async def no_rows(*_args, **_kwargs):
        return []

    async def finalize(*_args, **_kwargs):
        return True

    async def no_active_evaluations(*_args, **_kwargs):
        return 0

    async def cleanup():
        calls.append("cleanup")

    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "get_task_evaluations_by_status", no_rows)
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "count_task_evaluations_by_status", no_active_evaluations)
    monkeypatch.setattr(lifecycle_tasks, "_finalize_task_status_from_evaluations", finalize)
    monkeypatch.setattr(lifecycle_tasks, "cleanup_all_basilica_deployments", cleanup)

    task = SimpleNamespace(task_id="task-1", task_type=None)
    config = SimpleNamespace(psql_db=object())

    await lifecycle_tasks._evaluate_pending_pairs_for_task(task, num_gpus=1, config=config)

    assert calls == ["cleanup"]


@pytest.mark.asyncio
async def test_finalized_task_skips_basilica_cleanup_with_active_evaluations(monkeypatch):
    calls: list[str] = []

    async def no_rows(*_args, **_kwargs):
        return []

    async def finalize(*_args, **_kwargs):
        return True

    async def active_evaluations(*_args, **_kwargs):
        return 2

    async def cleanup():
        calls.append("cleanup")

    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "get_task_evaluations_by_status", no_rows)
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "count_task_evaluations_by_status", active_evaluations)
    monkeypatch.setattr(lifecycle_tasks, "_finalize_task_status_from_evaluations", finalize)
    monkeypatch.setattr(lifecycle_tasks, "cleanup_all_basilica_deployments", cleanup)

    task = SimpleNamespace(task_id="task-1", task_type=None)
    config = SimpleNamespace(psql_db=object())

    await lifecycle_tasks._evaluate_pending_pairs_for_task(task, num_gpus=1, config=config)

    assert calls == []


@pytest.mark.asyncio
async def test_unfinalized_task_does_not_check_drained_basilica_cleanup(monkeypatch):
    calls: list[str] = []

    async def no_rows(*_args, **_kwargs):
        return []

    async def finalize(*_args, **_kwargs):
        return False

    async def count_active(*_args, **_kwargs):
        calls.append("count")
        return 0

    async def cleanup():
        calls.append("cleanup")

    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "get_task_evaluations_by_status", no_rows)
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "count_task_evaluations_by_status", count_active)
    monkeypatch.setattr(lifecycle_tasks, "_finalize_task_status_from_evaluations", finalize)
    monkeypatch.setattr(lifecycle_tasks, "cleanup_all_basilica_deployments", cleanup)

    task = SimpleNamespace(task_id="task-1", task_type=None)
    config = SimpleNamespace(psql_db=object())

    await lifecycle_tasks._evaluate_pending_pairs_for_task(task, num_gpus=1, config=config)

    assert calls == []
