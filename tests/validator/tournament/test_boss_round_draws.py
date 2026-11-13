"""Quant boss-round draws and the built size of the quant boss round.

A draw is zero decided examples: every held-out example landed inside the tie dead zone, so nothing
separated the two models. On a fixed quant task that is a non-win within both crown allowances (crown.py, tested in
test_crown.py and test_quant_boss_crown_rule.py); a quant boss round never adds replacement tasks.
"""


import numpy as np
import pytest

from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.tournament import constants as t_cst
from validator.tournament import round_results
from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.thresholds import compare_paired_losses


async def _async(value):
    return value


async def _async_none(**_kwargs):
    return None


BOSS = "boss_hotkey"
CHALLENGER = "challenger_hotkey"
QUANT_BOSS_ROUND_SIZE = t_cst.FINAL_ROUND_QUANT_TASKS  # four


def _round_tasks(n: int) -> list[TournamentTask]:
    return [
        TournamentTask(
            tournament_id="tourn_1",
            round_id="tourn_1_round_004",
            task_id=f"00000000-0000-0000-0000-{i:012d}",
            pair_id="tourn_1_round_004_pair_001",
        )
        for i in range(n)
    ]


def test_all_examples_inside_dead_zone_is_a_draw():
    """The trigger condition: no example separated the two models by more than the dead zone."""
    boss = np.random.default_rng(7).uniform(0.02, 0.03, 800)
    challenger = boss - (t_cst.BOSS_ROUND_TIE_DEADZONE_NATS / 2)

    result = compare_paired_losses(list(boss), list(challenger))

    assert result.n_decided == 0
    assert result.is_draw is True
    assert result.challenger_won is False


@pytest.mark.parametrize(
    "tournament_type,expected",
    [
        (TournamentType.QUANT, 4),
        (TournamentType.TEXT, t_cst.FINAL_ROUND_TEXT_TASKS),
        # Image keeps deriving the bar from its own count, so it gets 0.
        (TournamentType.IMAGE, 0),
    ],
)
def test_expected_boss_round_task_count(tournament_type, expected):
    assert t_cst.expected_boss_round_task_count(tournament_type) == expected


def test_freeform_can_never_draw():
    """Freeform is decided on effective bytes, never on the paired gate: only fixed can draw."""
    assert TaskType.QUANTFREEFORMTASK not in t_cst.PAIRED_BOSS_ROUND_TASK_TYPES


def _final_round() -> TournamentRoundData:
    return TournamentRoundData(
        round_id="tourn_1_round_004",
        tournament_id="tourn_1",
        round_number=4,
        round_type=RoundType.KNOCKOUT,
        is_final_round=True,
        status=RoundStatus.PENDING,
    )


@pytest.mark.asyncio
async def test_orphaned_task_defers_the_round_instead_of_scoring_it(monkeypatch):
    """A boss-round task with no assigned nodes never ran, so it must not be scored against anyone.

    One gets here when a task is created but the round never makes it back to PENDING - a crash, or a
    failure partway through creating a batch. The challenger is not charged a loss for it;
    the round is deferred and sent back for node assignment.
    """
    round_tasks = _round_tasks(QUANT_BOSS_ROUND_SIZE + 1)
    orphan_id = round_tasks[-1].task_id

    async def fake_get_tournament(_tournament_id, _psql_db):
        return TournamentData(tournament_id="tourn_1", tournament_type=TournamentType.QUANT)

    async def fake_count_hotkeys_assigned_to_task(task_id, _psql_db):
        return 0 if task_id == orphan_id else 2

    monkeypatch.setattr(round_results, "get_tournament", fake_get_tournament)
    monkeypatch.setattr(round_results, "count_hotkeys_assigned_to_task", fake_count_hotkeys_assigned_to_task)

    outcome = await round_results.get_knockout_winners(_final_round(), round_tasks, psql_db=None, config=None)

    assert outcome.is_deferred is True
    assert outcome.unassigned_task_ids == [orphan_id]
    assert outcome.winners == []


