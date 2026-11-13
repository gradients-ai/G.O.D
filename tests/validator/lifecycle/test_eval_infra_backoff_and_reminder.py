"""Lifecycle: infra-error re-queue backoff, the status guard on the count, the held-row reminder, and which errors count."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.db.sql.tasks as tasks_sql
import validator.lifecycle.constants as lifecycle_cst
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.evaluation.errors import DeploymentNotReadyError
from validator.evaluation.errors import EvaluationCapacityUnavailable
from validator.lifecycle import eval_infra_errors
from validator.lifecycle import tasks as lifecycle_tasks


CONFIG = SimpleNamespace(psql_db="db", discord_url=None)
HELD = lifecycle_cst.EVALUATION_STATUS_INFRA_HELD
CAP = lifecycle_cst.MAX_EVAL_INFRA_ATTEMPTS


def _quant_task() -> SimpleNamespace:
    return SimpleNamespace(task_id=uuid.uuid4(), task_type=TaskType.QUANTFIXEDTASK, status=TaskStatus.EVALUATING)


def _text_task() -> SimpleNamespace:
    return SimpleNamespace(task_id=uuid.uuid4(), task_type=TaskType.INSTRUCTTEXTTASK, status=TaskStatus.EVALUATING)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(eval_infra_errors.time, "monotonic", fake)
    monkeypatch.setattr(eval_infra_errors, "_REQUEUED_AT", {})
    monkeypatch.setattr(eval_infra_errors, "_HELD_ALERTED_AT", {})
    return fake


@pytest.fixture
def table(monkeypatch, clock):
    """hotkey -> [status, attempts]; the fake honours the same status guard as the SQL."""
    state = SimpleNamespace(rows={"A": ["evaluating", 0], "B": ["success", 0]}, alerts=[])

    async def record(task_id, hotkeys, max_attempts, held_status, psql_db):
        counts = {}
        for hotkey in hotkeys:
            row = state.rows[hotkey]
            if row[0] not in ("pending", "evaluating"):
                continue
            row[1] += 1
            row[0] = held_status if row[1] >= max_attempts else "pending"
            counts[hotkey] = row[1]
        return counts

    async def notify(config, **kwargs):
        state.alerts.append(kwargs)

    async def set_status(task_id, hotkeys, status, psql_db):
        for hotkey in hotkeys:
            state.rows[hotkey][0] = status

    monkeypatch.setattr(eval_infra_errors.tasks_sql, "record_task_evaluation_infra_errors", record)
    monkeypatch.setattr(eval_infra_errors, "notify_evaluation_exception", notify)
    monkeypatch.setattr(eval_infra_errors, "task_deployment_ids_for_hotkeys", AsyncMock(return_value=[]))
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task_evaluations_status", set_status)
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "clear_task_evaluation_deployments", AsyncMock())
    monkeypatch.setattr(lifecycle_tasks.boss_seed_eval, "retry_seed_time_exception", AsyncMock(return_value=False))
    return state


def _evaluation_raises(monkeypatch, error: BaseException) -> None:
    monkeypatch.setattr(lifecycle_tasks, "evaluate_and_score_hotkeys", AsyncMock(side_effect=error))


async def test_a_quant_pod_that_never_becomes_ready_is_counted_and_held(monkeypatch, table) -> None:
    table.rows = {"A": ["evaluating", 0]}
    _evaluation_raises(monkeypatch, DeploymentNotReadyError("not ready in 1800s"))
    task = _quant_task()

    for attempt in range(1, CAP + 1):
        await lifecycle_tasks._evaluate_and_update_hotkeys(task, ["A"], 1, CONFIG)
        assert table.rows["A"] == [HELD if attempt == CAP else "pending", attempt]

    assert len(table.alerts) == 2  # the first occurrence and the hold


async def test_quant_capacity_unavailable_is_still_not_counted(monkeypatch, table) -> None:
    table.rows = {"A": ["evaluating", 0]}
    _evaluation_raises(monkeypatch, EvaluationCapacityUnavailable("no GPUs"))

    await lifecycle_tasks._evaluate_and_update_hotkeys(_quant_task(), ["A"], 1, CONFIG)

    assert table.rows["A"] == ["pending", 0]
    assert table.alerts == []
    lifecycle_tasks.tasks_sql.clear_task_evaluation_deployments.assert_awaited_once()


async def test_a_text_pod_that_is_not_ready_behaves_as_on_main(monkeypatch, table) -> None:
    table.rows = {"A": ["evaluating", 0]}
    _evaluation_raises(monkeypatch, DeploymentNotReadyError("not ready in 1800s"))

    await lifecycle_tasks._evaluate_and_update_hotkeys(_text_task(), ["A"], 1, CONFIG)

    assert table.rows["A"] == ["pending", 0]
    assert table.alerts == []
    lifecycle_tasks.tasks_sql.clear_task_evaluation_deployments.assert_awaited_once()


async def test_settled_rows_are_not_reported_as_requeued(monkeypatch, table) -> None:
    alerts = table.alerts
    task = _quant_task()

    await eval_infra_errors.record_infra_error(task, ["A", "B"], RuntimeError("late failure"), CONFIG)

    assert table.rows == {"A": ["pending", 1], "B": ["success", 0]}
    assert [alert["hotkeys"] for alert in alerts] == [["A"]]
    assert eval_infra_errors.requeue_backing_off(task.task_id, "A")
    assert not eval_infra_errors.requeue_backing_off(task.task_id, "B")


async def test_when_every_row_is_settled_nothing_is_reported(table) -> None:
    table.rows = {"B": ["success", 0]}

    await eval_infra_errors.record_infra_error(_quant_task(), ["B"], RuntimeError("late failure"), CONFIG)

    assert table.rows == {"B": ["success", 0]}
    assert table.alerts == []


async def test_the_count_sql_only_touches_unsettled_rows() -> None:
    queries = []

    class Connection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def fetch(self, query, *args):
            queries.append(query)
            return []

    class Db:
        async def connection(self):
            return Connection()

    await tasks_sql.record_task_evaluation_infra_errors(uuid.uuid4(), ["A"], CAP, HELD, Db())

    assert "evaluation_status IN ('pending', 'evaluating')" in queries[0]


@pytest.fixture
def dispatch(monkeypatch):
    """_evaluate_pending_pairs_for_task with the DB reads faked: the hotkeys it launched an evaluation for."""
    state = SimpleNamespace(pending=["A", "B"], launched=[], finalised=0)

    async def by_status(task_id, status, psql_db):
        return [{"hotkey": hotkey} for hotkey in state.pending] if status == "pending" else []

    async def evaluate(task, hotkeys, num_gpus, config):
        state.launched.append(list(hotkeys))

    async def finalise(task, config):
        state.finalised += 1
        return False

    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "get_task_evaluations_by_status", by_status)
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task_evaluations_status", AsyncMock())
    monkeypatch.setattr(lifecycle_tasks, "_evaluate_and_update_hotkeys", evaluate)
    monkeypatch.setattr(lifecycle_tasks, "_finalize_task_status_from_evaluations", finalise)
    return state


async def test_a_requeued_row_waits_out_the_backoff_then_dispatches(table, dispatch, clock) -> None:
    task = _quant_task()
    table.rows = {"A": ["evaluating", 0], "B": ["pending", 0]}
    await eval_infra_errors.record_infra_error(task, ["A"], RuntimeError("hub down"), CONFIG)

    await lifecycle_tasks._evaluate_pending_pairs_for_task(task, 1, CONFIG)
    assert dispatch.launched == [["B"]]  # A is in backoff: still pending, and the task is not finalised around it

    clock.now += lifecycle_cst.EVAL_INFRA_REQUEUE_BACKOFF_SECONDS - 1
    dispatch.launched.clear()
    await lifecycle_tasks._evaluate_pending_pairs_for_task(task, 1, CONFIG)
    assert dispatch.launched == [["B"]]

    clock.now += 1
    dispatch.launched.clear()
    await lifecycle_tasks._evaluate_pending_pairs_for_task(task, 1, CONFIG)
    assert dispatch.launched == [["A"], ["B"]]


async def test_rows_never_errored_dispatch_as_before(dispatch, clock) -> None:
    await lifecycle_tasks._evaluate_pending_pairs_for_task(_text_task(), 1, CONFIG)

    assert dispatch.launched == [["A"], ["B"]]


async def test_the_held_reminder_waits_one_interval_then_fires_once_per_interval(table, clock) -> None:
    task = _quant_task()
    interval = lifecycle_cst.EVAL_INFRA_HELD_REMINDER_SECONDS

    await eval_infra_errors.remind_held_rows(task, ["A"], CONFIG)  # first sight: the hold already alerted
    assert table.alerts == []

    clock.now += interval - 1
    await eval_infra_errors.remind_held_rows(task, ["A"], CONFIG)
    assert table.alerts == []

    clock.now += 1
    await eval_infra_errors.remind_held_rows(task, ["A"], CONFIG)
    assert len(table.alerts) == 1
    context = table.alerts[0]["context"]
    assert eval_infra_errors.requeue_sql(str(task.task_id)) in context
    assert eval_infra_errors.fail_sql(str(task.task_id)) in context
    assert table.alerts[0]["hotkeys"] == ["A"]

    clock.now += interval - 1
    await eval_infra_errors.remind_held_rows(task, ["A"], CONFIG)
    assert len(table.alerts) == 1

    clock.now += 1
    await eval_infra_errors.remind_held_rows(task, ["A"], CONFIG)
    assert len(table.alerts) == 2


async def test_the_hold_starts_the_reminder_clock(table, clock) -> None:
    table.rows = {"A": ["evaluating", CAP - 1]}
    task = _quant_task()
    await eval_infra_errors.record_infra_error(task, ["A"], RuntimeError("x"), CONFIG)
    assert len(table.alerts) == 1  # the hold alert

    clock.now += lifecycle_cst.EVAL_INFRA_HELD_REMINDER_SECONDS - 1
    await eval_infra_errors.remind_held_rows(task, ["A"], CONFIG)
    assert len(table.alerts) == 1
    clock.now += 1
    await eval_infra_errors.remind_held_rows(task, ["A"], CONFIG)
    assert len(table.alerts) == 2


async def test_finalisation_of_a_task_with_held_rows_sends_the_reminder_hook(monkeypatch) -> None:
    rows = [{"hotkey": "A", "evaluation_status": HELD}, {"hotkey": "B", "evaluation_status": "success"}]
    monkeypatch.setattr(lifecycle_tasks.tournament_sql, "get_training_status_for_task", AsyncMock(return_value={}))
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "get_task_evaluation_rows", AsyncMock(return_value=rows))
    remind = AsyncMock()
    monkeypatch.setattr(lifecycle_tasks, "remind_held_rows", remind)
    task = _quant_task()

    assert await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG) is False

    remind.assert_awaited_once_with(task, ["A"], CONFIG)
