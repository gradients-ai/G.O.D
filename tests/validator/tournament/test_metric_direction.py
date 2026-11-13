"""Sort directions: rewards descend, every loss ascends, in the audit SQL and the synthetic-vs-tournament comparison."""

from types import SimpleNamespace

import pytest

import validator.tournament.performance_calculator as pc
from core.models.task_models import TaskType
from validator.db.sql.auditing import _rank_order_sql


TASK_ID = "11111111-2222-3333-4444-555555555555"
SYNTH_TASK_ID = "66666666-7777-8888-9999-000000000000"
HIGHER_IS_BETTER = (TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK)
LOWER_IS_BETTER = (
    TaskType.INSTRUCTTEXTTASK,
    TaskType.CHATTASK,
    TaskType.DPOTASK,
    TaskType.QUANTFIXEDTASK,
    TaskType.QUANTFREEFORMTASK,
)


@pytest.mark.parametrize("task_type", HIGHER_IS_BETTER)
def test_rank_order_sql_descends_for_rewards(task_type):
    assert _rank_order_sql(task_type) == "tn.test_loss DESC NULLS LAST"


@pytest.mark.parametrize("task_type", LOWER_IS_BETTER)
def test_rank_order_sql_ascends_for_losses(task_type):
    assert _rank_order_sql(task_type) == "tn.test_loss ASC NULLS LAST"


def _wire_performance(monkeypatch, task_type: TaskType) -> None:
    async def fake_pairs(tournament_id, psql_db):
        return [
            SimpleNamespace(
                tournament_task_id=TASK_ID, synthetic_task_id=SYNTH_TASK_ID, winner_hotkey="winner", task_type=task_type.value
            )
        ]

    async def fake_scores(task_ids, psql_db):
        return {
            TASK_ID: [SimpleNamespace(hotkey="winner", test_loss=2.0)],
            SYNTH_TASK_ID: [SimpleNamespace(hotkey=hotkey, test_loss=loss) for hotkey, loss in (("a", 1.0), ("b", 4.0))],
        }

    monkeypatch.setattr(pc, "get_boss_round_winner_task_pairs", fake_pairs)
    monkeypatch.setattr(pc, "get_task_scores_batch", fake_scores)


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", HIGHER_IS_BETTER)
async def test_performance_calculator_takes_max_synthetic_for_rewards(monkeypatch, task_type):
    _wire_performance(monkeypatch, task_type)

    [data] = await pc.get_tournament_performance_data("t", psql_db=None)

    assert data.best_synthetic_score == 4.0
    assert data.performance_difference == pytest.approx((4.0 - 2.0) / 4.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", LOWER_IS_BETTER)
async def test_performance_calculator_takes_min_synthetic_for_losses(monkeypatch, task_type):
    _wire_performance(monkeypatch, task_type)

    [data] = await pc.get_tournament_performance_data("t", psql_db=None)

    assert data.best_synthetic_score == 1.0
    assert data.performance_difference == pytest.approx((2.0 - 1.0) / 1.0)
