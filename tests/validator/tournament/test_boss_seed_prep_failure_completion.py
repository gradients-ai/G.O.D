"""A prep-failed boss-round seed is never replaced: the boss's entries are trained upfront and never retrained.

With a usable challenger the slot goes to the challenger; with none the task still completes and the boss keeps it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.tournament import tournament_manager
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentSeedTask


TID = "tourn_test"


@pytest.fixture
def env(monkeypatch):
    task_id = uuid4()
    alerts = AsyncMock()
    replace = AsyncMock(return_value=uuid4())
    monkeypatch.setattr(tournament_manager, "alert_seed_issue", alerts)
    monkeypatch.setattr(tournament_manager, "replace_tournament_task", replace)
    return SimpleNamespace(
        task_id=task_id, alerts=alerts, replace=replace, monkeypatch=monkeypatch, config=SimpleNamespace(psql_db=None)
    )


def _set(env, seed_role, usable_challenger):
    seed = (
        None
        if seed_role is None
        else TournamentSeedTask(tournament_id=TID, role=seed_role, attempt=1, task_id=env.task_id, status=SeedStatus.TRAINING)
    )
    boss_side = seed_role if usable_challenger else None
    env.monkeypatch.setattr(tournament_manager.seeds_sql, "boss_side_seed_failure", AsyncMock(return_value=boss_side))
    env.monkeypatch.setattr(tournament_manager.seeds_sql, "get_seed_task_by_task_id", AsyncMock(return_value=seed))
    env.monkeypatch.setattr(tournament_manager.seeds_sql, "boss_compression_entry_failed", AsyncMock(return_value=False))


async def _complete(env):
    return await tournament_manager.is_tourn_task_completed(
        SimpleNamespace(tournament_id=TID, round_id="round_4", task_id=env.task_id, group_id=None, pair_id=None),
        SimpleNamespace(status=TaskStatus.PREP_TASK_FAILURE.value, task_id=env.task_id, task_type=TaskType.QUANTFIXEDTASK),
        env.config,
        final_round=True,
    )


@pytest.mark.parametrize("role", [SeedRole.BOSS_CHAINED_FIXED, SeedRole.BOSS_CHAINED_FREEFORM])
async def test_no_usable_challenger_completes_unreplaced_and_the_boss_keeps_the_slot(env, role):
    _set(env, role, usable_challenger=False)

    completed, reason = await _complete(env)

    assert completed is True
    assert "boss keeps the slot" in reason
    env.replace.assert_not_awaited()
    env.alerts.assert_awaited_once()
    assert "stays with the boss" in env.alerts.await_args.args[1]


async def test_a_usable_challenger_still_takes_the_slot(env):
    _set(env, SeedRole.BOSS_CHAINED_FIXED, usable_challenger=True)

    completed, reason = await _complete(env)

    assert completed is True
    assert "challenger takes the slot" in reason
    env.replace.assert_not_awaited()
    env.alerts.assert_awaited_once()
    assert "goes to the challenger" in env.alerts.await_args.args[1]


async def test_a_non_seed_quant_task_is_still_replaced(env):
    _set(env, None, usable_challenger=False)

    completed, reason = await _complete(env)

    assert completed is False
    assert "Replaced with a new task" in reason
    env.replace.assert_awaited_once()
    env.alerts.assert_not_awaited()
