"""Lifecycle: an evaluation error that is not the miner's fault re-queues the rows, counts, alerts, then holds them."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

import validator.lifecycle.constants as lifecycle_cst
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.evaluation.errors import DeploymentNotReadyError
from validator.evaluation.errors import EvaluationCapacityUnavailable
from validator.evaluation.errors import EvaluationInfraError
from validator.evaluation.errors import EvaluationRetryableError
from validator.lifecycle import eval_infra_errors
from validator.lifecycle import tasks as lifecycle_tasks
from validator.tasks.models import InstructTextRawTask


CONFIG = SimpleNamespace(psql_db="db", discord_url=None)
HELD = lifecycle_cst.EVALUATION_STATUS_INFRA_HELD
CAP = lifecycle_cst.MAX_EVAL_INFRA_ATTEMPTS


def _task() -> InstructTextRawTask:
    return InstructTextRawTask(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.EVALUATING,
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=4.0,
        created_at=datetime(2026, 11, 10),
        field_instruction="q",
        field_output="a",
    )


def _quant_task() -> SimpleNamespace:
    return SimpleNamespace(task_id=uuid.uuid4(), task_type=TaskType.QUANTFIXEDTASK, status=TaskStatus.EVALUATING)


@pytest.fixture
def rows(monkeypatch):
    """The evaluations table for one task: hotkey -> [status, infra_error_attempts]; .alerts records each alert context."""
    state = SimpleNamespace(rows={"A": ["evaluating", 0], "B": ["evaluating", 0]}, alerts=[])

    async def record(task_id, hotkeys, max_attempts, held_status, psql_db):
        counts = {}
        for hotkey in hotkeys:
            row = state.rows[hotkey]
            row[1] += 1
            row[0] = held_status if row[1] >= max_attempts else "pending"
            counts[hotkey] = row[1]
        return counts

    async def set_status(task_id, hotkeys, status, psql_db):
        for hotkey in hotkeys:
            state.rows[hotkey][0] = status

    async def notify(config, **kwargs):
        state.alerts.append(kwargs)

    monkeypatch.setattr(eval_infra_errors.tasks_sql, "record_task_evaluation_infra_errors", record)
    monkeypatch.setattr(eval_infra_errors, "notify_evaluation_exception", notify)
    monkeypatch.setattr(eval_infra_errors, "task_deployment_ids_for_hotkeys", AsyncMock(return_value=["runpod:dep"]))
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task_evaluations_status", set_status)
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "clear_task_evaluation_deployments", AsyncMock())
    monkeypatch.setattr(lifecycle_tasks.boss_seed_eval, "retry_seed_time_exception", AsyncMock(return_value=False))
    return state


def _evaluation_raises(monkeypatch, error: BaseException) -> None:
    monkeypatch.setattr(lifecycle_tasks, "evaluate_and_score_hotkeys", AsyncMock(side_effect=error))


@pytest.mark.parametrize(
    "error",
    [
        EvaluationInfraError("Evaluation failed after 1 attempts: ReadTimeout (no message)"),
        EvaluationRetryableError("Failed to verify deletion for failed deployment dep"),
        httpx.ReadTimeout(""),  # escaped evaluation entirely
        KeyError("bug in our code"),
    ],
)
async def test_quant_errors_that_are_not_the_miners_never_fail_a_row(monkeypatch, rows, error) -> None:
    _evaluation_raises(monkeypatch, error)
    task = _quant_task()

    for attempt in range(1, CAP + 1):
        await lifecycle_tasks._evaluate_and_update_hotkeys(task, ["A", "B"], 1, CONFIG)
        expected = HELD if attempt == CAP else "pending"
        assert rows.rows == {"A": [expected, attempt], "B": [expected, attempt]}

    contexts = [alert["context"] for alert in rows.alerts]
    assert len(contexts) == 2, contexts  # the first occurrence and the hold, nothing in between
    assert "NOT the miner's fault: re-queued" in contexts[0]
    assert "rows HELD" in contexts[1] and "infra_error_attempts=0" in contexts[1]
    assert "SET evaluation_status='failure'" in contexts[1]
    assert all(alert["deployment_ids"] == ["runpod:dep"] for alert in rows.alerts)
    assert type(error).__name__ in eval_infra_errors.describe_error(rows.alerts[0]["error"])


async def test_a_text_task_follows_mains_handling(monkeypatch, rows) -> None:
    notify = AsyncMock()
    monkeypatch.setattr(lifecycle_tasks, "notify_evaluation_exception", notify)
    monkeypatch.setattr(lifecycle_tasks, "task_deployment_ids_for_hotkeys", AsyncMock(return_value=[]))

    _evaluation_raises(monkeypatch, EvaluationInfraError("transient"))
    await lifecycle_tasks._evaluate_and_update_hotkeys(_task(), ["A"], 1, CONFIG)
    assert rows.rows["A"] == ["pending", 0]  # retryable: re-queued, never counted

    _evaluation_raises(monkeypatch, KeyError("bug"))
    await lifecycle_tasks._evaluate_and_update_hotkeys(_task(), ["B"], 1, CONFIG)
    assert rows.rows["B"] == ["failure", 0]  # any other exception fails the row and notifies
    notify.assert_awaited_once()
    assert rows.alerts == []


@pytest.mark.parametrize("error", [EvaluationCapacityUnavailable("no GPUs"), DeploymentNotReadyError("not ready in 1800s")])
async def test_capacity_waits_are_not_counted(monkeypatch, rows, error) -> None:
    _evaluation_raises(monkeypatch, error)

    await lifecycle_tasks._evaluate_and_update_hotkeys(_task(), ["A"], 1, CONFIG)

    assert rows.rows["A"] == ["pending", 0]
    assert rows.alerts == []


async def test_scored_failures_still_fail_their_rows(monkeypatch, rows) -> None:
    outcome = SimpleNamespace(evaluated=["A", "B"], failed=["B"], deferred=[])
    monkeypatch.setattr(lifecycle_tasks, "evaluate_and_score_hotkeys", AsyncMock(return_value=outcome))

    await lifecycle_tasks._evaluate_and_update_hotkeys(_task(), ["A", "B"], 1, CONFIG)

    assert rows.rows == {"A": ["success", 0], "B": ["failure", 0]}


@pytest.fixture
def finalize(monkeypatch):
    seams = SimpleNamespace(update_task=AsyncMock(), finalize=AsyncMock())
    monkeypatch.setattr(lifecycle_tasks.tournament_sql, "get_training_status_for_task", AsyncMock(return_value={}))
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task", seams.update_task)
    monkeypatch.setattr(lifecycle_tasks, "finalize_task_scores_from_raw_losses", seams.finalize)
    return seams


@pytest.mark.parametrize("other_status", ["failure", "success"])
async def test_a_held_row_keeps_the_task_from_finalising(monkeypatch, finalize, other_status) -> None:
    rows = [{"hotkey": "A", "evaluation_status": HELD}, {"hotkey": "B", "evaluation_status": other_status}]
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "get_task_evaluation_rows", AsyncMock(return_value=rows))
    task = _task()

    assert await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG) is False

    assert task.status == TaskStatus.EVALUATING
    finalize.update_task.assert_not_awaited()
    finalize.finalize.assert_not_awaited()


def test_requeue_sql_selects_the_held_rows() -> None:
    sql = eval_infra_errors.requeue_sql("t-1")

    assert sql == (
        "UPDATE evaluations SET evaluation_status='pending', infra_error_attempts=0 "
        "WHERE task_id='t-1' AND evaluation_status='infra_held';"
    )


def test_fail_sql_selects_the_held_rows() -> None:
    sql = eval_infra_errors.fail_sql("t-1")

    assert sql == "UPDATE evaluations SET evaluation_status='failure' WHERE task_id='t-1' AND evaluation_status='infra_held';"


def test_reference_requeue_sql_names_few_rows_and_scopes_many() -> None:
    assert eval_infra_errors.reference_requeue_sql("t-1", ["A", "B"]) == (
        "UPDATE evaluations SET evaluation_status='pending' WHERE task_id='t-1' AND hotkey IN ('A', 'B');"
    )
    many = eval_infra_errors.reference_requeue_sql("t-1", [f"hk{i}" for i in range(50)])
    assert many == "UPDATE evaluations SET evaluation_status='pending' WHERE task_id='t-1' AND evaluation_status='success';"
    assert eval_infra_errors.reference_requeue_sql("t-1", []) == many


async def test_the_held_alert_for_fifty_hotkeys_fits_the_channel(monkeypatch) -> None:
    hotkeys = [f"5{'H' * 46}{i:02d}" for i in range(50)]
    state = SimpleNamespace(rows={hotkey: ["evaluating", CAP - 1] for hotkey in hotkeys}, posted=[])

    async def record(task_id, keys, max_attempts, held_status, psql_db):
        return {key: CAP for key in keys}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json, timeout):
            state.posted.append(json["content"])

    monkeypatch.setattr(eval_infra_errors.tasks_sql, "record_task_evaluation_infra_errors", record)
    monkeypatch.setattr(
        eval_infra_errors, "task_deployment_ids_for_hotkeys", AsyncMock(return_value=[f"runpod:{'d' * 30}{i}" for i in range(50)])
    )
    monkeypatch.setattr(httpx, "AsyncClient", Client)

    await eval_infra_errors.record_infra_error(
        _task(), hotkeys, EvaluationInfraError("x" * 5000), SimpleNamespace(psql_db="db", discord_url="https://discord.invalid/hook")
    )

    assert len(state.posted) == 1
    content = state.posted[0]
    assert len(content) <= 2000
    assert "evaluation_status='infra_held'" in content and "(+38 more)" in content
