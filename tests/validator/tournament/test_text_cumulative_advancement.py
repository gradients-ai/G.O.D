"""Seeded text rounds 2-3: advance by average rank over every round played so far, among the remaining miners.

Round 1 advances on its own task. The boss has a result on every round's seed task but never takes a slot.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tournament.round_results as rr
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tournament import constants as t_cst
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType


TID = "t"
TASKS = {
    1: "11111111-2222-3333-4444-555555555555",
    2: "21111111-2222-3333-4444-555555555555",
    3: "31111111-2222-3333-4444-555555555555",
}


def _round(number: int) -> TournamentRoundData:
    return TournamentRoundData(round_id=f"r{number}", tournament_id=TID, round_number=number, round_type=RoundType.GROUP)


def _result(hotkey: str, loss: float) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=TaskType.INSTRUCTTEXTTASK
    )


@pytest.fixture
def tournament(monkeypatch):
    """Fake a seeded text tournament: per round, hotkeys best first (the boss is added with the best loss on every task)
    and the group's members."""

    def wire(orders: dict[int, list[str]], members: dict[int, list[str]]):
        results = {
            TASKS[n]: [_result(EMISSION_BURN_HOTKEY, 0.01)] + [_result(hk, 1.0 + i / 10) for i, hk in enumerate(order)]
            for n, order in orders.items()
        }

        async def fake_results(task_id, _db):
            return [r.model_copy() for r in results[str(task_id)]]

        async def fake_rounds(_tid, _db):
            return [_round(n) for n in sorted(members)]

        async def fake_tasks(round_id, _db):
            number = int(round_id.removeprefix("r"))
            return [TournamentTask(tournament_id=TID, round_id=round_id, task_id=TASKS[number], group_id=f"g{number}")]

        async def fake_members(group_id, _db):
            return [SimpleNamespace(hotkey=hk) for hk in members[int(group_id.removeprefix("g"))]]

        monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
        monkeypatch.setattr(rr, "get_task", AsyncMock(return_value=SimpleNamespace(task_type=TaskType.INSTRUCTTEXTTASK)))
        monkeypatch.setattr(rr, "get_tournament_rounds", fake_rounds)
        monkeypatch.setattr(rr, "get_tournament_tasks", fake_tasks)
        monkeypatch.setattr(rr, "get_tournament_group_members", fake_members)
        monkeypatch.setattr(
            rr, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id=TID, tournament_type=TournamentType.TEXT))
        )
        monkeypatch.setattr(rr, "is_seeded_text_tournament", AsyncMock(return_value=True))

    return wire


async def _winners(number: int):
    tasks = await rr.get_tournament_tasks(f"r{number}", None)
    return await rr.get_group_winners(_round(number), tasks, psql_db=None)


def test_round_advance_sizes():
    assert [spec.advance for spec in t_cst.TEXT_ROUND_PLAN] == [8, 4, 1]


@pytest.mark.asyncio
async def test_round_two_cuts_by_average_rank_not_by_the_round_two_task(tournament):
    miners = list("ABCDEF")
    # Round-2 task order alone would advance E, F, A, B; average rank over R1+R2 advances A, B, C, E.
    tournament(
        {1: miners, 2: ["E", "F", "A", "B", "C", "D"]},
        {1: miners, 2: miners},
    )
    # Average ranks: A (1+3)/2=2, B (2+4)/2=3, C (3+5)/2=4, D (4+6)/2=5, E (5+1)/2=3, F (6+2)/2=4.
    round_two_alone = await rr._quant_ordered_hotkeys(await rr.get_tournament_tasks("r2", None), None, cohort=set(miners))
    assert round_two_alone[:4] == ["E", "F", "A", "B"]

    winners = await _winners(2)

    # E ties B on average rank 3 and won round 2, so E ranks ahead of B; F ties C on 4 and was better in round 2.
    assert winners == ["A", "E", "B", "F"]


@pytest.mark.asyncio
async def test_round_three_picks_the_best_average_over_all_three_rounds(tournament):
    cohort = ["A", "B", "C", "D"]
    # Round 3's task alone favours D; averages: A (1+3+2)/3=2, B (2+1+4)/3=2.33, C (3+2+3)/3=2.67, D (4+4+1)/3=3.
    tournament(
        {1: cohort, 2: ["B", "C", "A", "D"], 3: ["D", "A", "C", "B"]},
        {1: cohort, 2: cohort, 3: cohort},
    )

    assert await _winners(3) == ["A"]


@pytest.mark.asyncio
async def test_a_tie_on_average_rank_goes_to_the_better_most_recent_round(tournament):
    # A: R1 1, R2 2. B: R1 2, R2 1. Both average 1.5; B won the latest round.
    tournament({1: ["A", "B"], 2: ["B", "A"]}, {1: ["A", "B"], 2: ["A", "B"]})

    assert await rr.planned_round_ordered_hotkeys(_round(2), await rr.get_tournament_tasks("r2", None), None) == ["B", "A"]


@pytest.mark.asyncio
async def test_a_miner_with_no_result_in_a_round_ranks_last_on_it(tournament):
    # B has no R2 result: it ranks 3 there, so (1+3)/2=2.0 trails A's (2+1)/2=1.5 although B led R1.
    tournament({1: ["B", "A"], 2: ["A"]}, {1: ["A", "B"], 2: ["A", "B"]})

    assert await _winners(2) == ["A", "B"]
    ordered = await rr.planned_round_ordered_hotkeys(_round(2), await rr.get_tournament_tasks("r2", None), None)
    assert ordered == ["A", "B"]


@pytest.mark.asyncio
async def test_the_boss_is_never_ranked_into_a_slot_or_into_the_miners_ranks(tournament):
    miners = ["A", "B", "C", "D", "E"]
    # The boss beats every miner on every task. Ranks among the miners are unchanged by it.
    tournament({1: miners, 2: miners}, {1: miners, 2: miners})

    winners = await _winners(2)

    assert winners == ["A", "B", "C", "D"]
    assert EMISSION_BURN_HOTKEY not in winners
    assert await _winners(1) == miners


@pytest.mark.asyncio
async def test_round_one_advances_on_its_own_task(tournament):
    miners = [f"m{i}" for i in range(10)]
    tournament({1: miners}, {1: miners})

    assert await _winners(1) == miners[:8]


@pytest.mark.asyncio
async def test_third_place_is_the_runner_up_of_the_cumulative_standings(tournament):
    cohort = ["A", "B", "C"]
    # Round 3's task alone ranks C, B, A; cumulative: A (1+1+3)/3, B (2+2+2)/3, C (3+3+1)/3. Challenger A, third B.
    tournament({1: cohort, 2: cohort, 3: ["C", "B", "A"]}, {1: cohort, 2: cohort, 3: cohort})

    assert await rr.get_planned_runner_up(TournamentData(tournament_id=TID, tournament_type=TournamentType.TEXT), "A", None) == "B"
