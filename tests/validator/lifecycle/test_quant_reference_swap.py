"""Lifecycle handling of a quant task with no bf16 reference model: it swaps; other task types finalize."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.models.dataset_models import ChatTemplateDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from validator.lifecycle import boss_seed_eval
from validator.lifecycle import quant_reference
from validator.lifecycle import tasks as lifecycle_tasks
from validator.lifecycle.quant_reference import ensure_quant_references
from validator.scoring.quant_ranking import QuantReferenceUnavailable
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament.models import SeedRole
from validator.tournament.models import TournamentSeedTask


CONFIG = SimpleNamespace(psql_db="db")


def _fields() -> dict:
    return dict(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.EVALUATING,
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=4.0,
        created_at=datetime(2026, 10, 19),
    )


def _freeform(n_eval_attempts: int = 0) -> QuantFreeformRawTask:
    return QuantFreeformRawTask(
        **_fields(),
        dataset_type=ChatTemplateDatasetType(chat_template="tokenizer_default"),
        n_eval_attempts=n_eval_attempts,
    )


def _fixed() -> QuantFixedRawTask:
    return QuantFixedRawTask(
        **_fields(),
        dataset_type=ChatTemplateDatasetType(chat_template="tokenizer_default"),
        quant_spec=QuantFormat.W4A16_SYM_G128.spec(),
    )


def _instruct() -> InstructTextRawTask:
    return InstructTextRawTask(**_fields(), field_instruction="q", field_output="a")


@pytest.fixture
def lifecycle(monkeypatch):
    """Training done and evaluation succeeded for the one miner."""
    rows = [{"hotkey": "miner", "evaluation_status": "success"}]
    seams = SimpleNamespace(set_status=AsyncMock(), clear=AsyncMock(), update_task=AsyncMock(), finalize=AsyncMock())
    monkeypatch.setattr(
        lifecycle_tasks.tournament_sql,
        "get_training_status_for_task",
        AsyncMock(return_value={"miner": "success"}),
    )
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "get_task_evaluation_rows", AsyncMock(return_value=rows))
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task_evaluations_status", seams.set_status)
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "clear_task_evaluation_deployments", seams.clear)
    monkeypatch.setattr(lifecycle_tasks.tasks_sql, "update_task", seams.update_task)
    monkeypatch.setattr(lifecycle_tasks, "finalize_task_scores_from_raw_losses", seams.finalize)
    # Per-row references are covered by test_quant_reference_measurement.py.
    monkeypatch.setattr(lifecycle_tasks, "ensure_quant_references", AsyncMock(return_value=True))
    monkeypatch.setattr(boss_seed_eval.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=None))  # not a seed
    seams.notify = AsyncMock()
    monkeypatch.setattr(boss_seed_eval, "notify_evaluation_exception", seams.notify)
    boss_seed_eval._REFERENCE_HELD_ALERTED.clear()
    return seams


async def test_no_reference_model_swaps_the_task(monkeypatch, lifecycle) -> None:
    # The real check: with no reference model no deployment could measure the reference.
    monkeypatch.setattr(lifecycle_tasks, "ensure_quant_references", ensure_quant_references)
    monkeypatch.setattr(quant_reference, "quant_reference_model", AsyncMock(return_value=None))
    task = _freeform(n_eval_attempts=0)

    done = await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG)

    assert done is True
    lifecycle.finalize.assert_not_awaited()
    lifecycle.set_status.assert_not_awaited()
    assert task.status == TaskStatus.PREP_TASK_FAILURE
    assert task.n_eval_attempts == 1
    lifecycle.update_task.assert_awaited_once_with(task, "db")


@pytest.mark.parametrize("make_task", [_fixed, _instruct])
async def test_other_tasks_finalize(lifecycle, make_task) -> None:
    task = make_task()

    done = await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG)

    assert done is True
    lifecycle.set_status.assert_not_awaited()
    lifecycle.finalize.assert_awaited_once()
    assert task.status == TaskStatus.SUCCESS


def _attached_boss_seed(task: QuantFreeformRawTask) -> TournamentSeedTask:
    return TournamentSeedTask(
        tournament_id="tourn", role=SeedRole.BOSS_CHAINED_FREEFORM, task_id=task.task_id, attached_round_id="round-4"
    )


async def test_a_held_boss_seed_alert_carries_the_remedy(monkeypatch, lifecycle) -> None:
    task = _freeform()
    monkeypatch.setattr(lifecycle_tasks, "ensure_quant_references", AsyncMock(side_effect=QuantReferenceUnavailable("none")))
    monkeypatch.setattr(boss_seed_eval.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=_attached_boss_seed(task)))
    monkeypatch.setattr(boss_seed_eval, "rows_missing_a_reference", AsyncMock(return_value=["miner"]))

    assert await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG) is False
    assert await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG) is False

    assert task.status == TaskStatus.EVALUATING
    lifecycle.update_task.assert_not_awaited()
    lifecycle.notify.assert_awaited_once()  # alerted once
    context = lifecycle.notify.await_args.kwargs["context"]
    remedy = f"UPDATE evaluations SET evaluation_status='pending' WHERE task_id='{task.task_id}' AND hotkey IN ('miner');"
    assert remedy in context


async def test_a_failing_hold_never_replaces_a_boss_seed(monkeypatch, lifecycle) -> None:
    task = _freeform()
    monkeypatch.setattr(lifecycle_tasks, "ensure_quant_references", AsyncMock(side_effect=QuantReferenceUnavailable("none")))
    monkeypatch.setattr(boss_seed_eval.seeds_sql, "get_seed_task_by_task_id", AsyncMock(side_effect=ConnectionError("db down")))

    assert await lifecycle_tasks._finalize_task_status_from_evaluations(task, CONFIG) is False

    assert task.status == TaskStatus.EVALUATING
    assert task.n_eval_attempts == 0
    lifecycle.update_task.assert_not_awaited()
