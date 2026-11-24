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
from validator.tournament.models import TrainingStatus


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
    and the group's members. ``losses`` optionally gives explicit per-round losses (hotkey -> loss) instead of the
    default 1.0 + i/10 derived from the order."""

    def wire(
        orders: dict[int, list[str]],
        members: dict[int, list[str]],
        losses: dict[int, dict[str, float]] | None = None,
    ):
        losses = losses or {}
        results = {
            TASKS[n]: [_result(EMISSION_BURN_HOTKEY, 0.01)]
            + [_result(hk, losses[n][hk] if n in losses else 1.0 + i / 10) for i, hk in enumerate(order)]
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
    round_two = ["E", "F", "A", "B", "C", "D"]
    # Round-2 task order alone would advance E, F, A, B; average rank over R1+R2 advances A, B, E, C.
    tournament(
        {1: miners, 2: round_two},
        {1: miners, 2: miners},
        losses={
            1: {"A": 1.0, "B": 1.1, "C": 1.2, "D": 1.3, "E": 2.0, "F": 2.1},
            2: {"E": 1.0, "F": 1.1, "A": 1.2, "B": 1.3, "C": 1.4, "D": 1.5},
        },
    )
    # Average ranks: A (1+3)/2=2, B (2+4)/2=3, C (3+5)/2=4, D (4+6)/2=5, E (5+1)/2=3, F (6+2)/2=4.
    round_two_alone = await rr._quant_ordered_hotkeys(await rr.get_tournament_tasks("r2", None), None, cohort=set(miners))
    assert round_two_alone[:4] == ["E", "F", "A", "B"]

    ordered = await rr.planned_round_ordered_hotkeys(_round(2), await rr.get_tournament_tasks("r2", None), None)

    # Ties on average rank go to the smaller summed gap to the best (R1 best 1.0, R2 best 1.0), not to the round-2 winner:
    # B (0.1+0.3)=0.4 beats E (1.0+0.0)=1.0 at rank 3, and C (0.2+0.4)=0.6 beats F (1.1+0.1)=1.2 at rank 4.
    assert ordered[:4] == ["A", "B", "E", "C"]
    # C cannot catch the leader over a third round, so the round-3 viability cut drops it from those four.
    assert await _winners(2) == ["A", "B", "E"]


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
async def test_a_tie_on_average_rank_goes_to_the_smaller_summed_gap(tournament):
    # A: R1 1, R2 2. B: R1 2, R2 1. Both average 1.5. B won the latest round but A has the smaller summed gap:
    # A (0 + (1.1-1.0)/1.0)=0.1, B ((2.0-1.0)/1.0 + 0)=1.0.
    tournament(
        {1: ["A", "B"], 2: ["B", "A"]},
        {1: ["A", "B"], 2: ["A", "B"]},
        losses={1: {"A": 1.0, "B": 2.0}, 2: {"B": 1.0, "A": 1.1}},
    )

    assert await rr.planned_round_ordered_hotkeys(_round(2), await rr.get_tournament_tasks("r2", None), None) == ["A", "B"]


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

    # Same order in both rounds: after the top-4 cut only the leader can still finish top.
    assert winners == ["A"]
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

    text = TournamentData(tournament_id=TID, tournament_type=TournamentType.TEXT)
    assert await rr.get_planned_runner_up(text, "A", None) == "B"


@pytest.mark.asyncio
async def test_round_two_drops_advancing_miners_that_can_no_longer_finish_top(tournament):
    miners = list("ABCDE")
    # R1 ranks A1 B2 C3 D4 E5, R2 ranks A1 C2 E3 D4 B5: sums A2 C5 B7 D8 E8.
    # D and E tie on 8 for the fourth slot; after the top-4 cut
    # only A can still finish top or joint top over three rounds.
    tournament({1: miners, 2: ["A", "C", "E", "D", "B"]}, {1: miners, 2: miners})
    ordered = await rr.planned_round_ordered_hotkeys(_round(2), await rr.get_tournament_tasks("r2", None), None)
    assert ordered[:3] == ["A", "C", "B"]

    assert await _winners(2) == ["A"]


@pytest.mark.asyncio
async def test_round_two_keeps_every_miner_that_can_still_finish_top(tournament):
    miners = list("ABCDEF")
    # R1 ranks A1 B2 C3 D4, R2 ranks B1 A2 D3 C4: sums A3 B3 C7 D7. Only A and B can still finish top or joint top.
    tournament({1: miners, 2: ["B", "A", "D", "C", "E", "F"]}, {1: miners, 2: miners})

    assert sorted(await _winners(2)) == ["A", "B"]


@pytest.mark.asyncio
async def test_round_one_and_three_are_not_viability_filtered(tournament, monkeypatch):
    miners = list("ABCDEFGHI")
    tournament({1: miners, 2: miners, 3: miners}, {1: miners, 2: miners, 3: miners})
    cut = AsyncMock(side_effect=lambda *args: args[2])
    monkeypatch.setattr(rr, "round3_viable_winners", cut)

    assert await _winners(1) == miners[:8]
    assert await _winners(3) == ["A"]
    cut.assert_not_awaited()
    await _winners(2)
    cut.assert_awaited_once()


@pytest.mark.asyncio
async def test_round_two_cuts_to_the_top_four_before_viability(tournament):
    miners = list("ABCDE")
    # Every miner sums 6 over R1+R2, so all five are level and all stay viable; the top-4 cut alone removes one.
    tournament({1: miners, 2: list(reversed(miners))}, {1: miners, 2: miners})

    winners = await _winners(2)

    assert len(winners) == 4 and set(winners) < set(miners)


@pytest.mark.asyncio
async def test_a_round_two_field_smaller_than_four_all_advance_when_all_are_viable(tournament):
    miners = list("ABC")
    tournament({1: miners, 2: list(reversed(miners))}, {1: miners, 2: miners})

    assert sorted(await _winners(2)) == miners


@pytest.mark.asyncio
async def test_the_third_place_fallback_uses_the_round_two_advance_count(tournament, monkeypatch):
    miners = list("ABCDE")
    tournament({1: miners, 2: miners, 3: ["A"]}, {1: miners, 2: miners, 3: ["A"]})
    advance_set = AsyncMock(return_value=["A", "B"])
    monkeypatch.setattr(rr, "_round_two_advance_set_ordered_hotkeys", advance_set)

    await rr.get_planned_runner_up(TournamentData(tournament_id=TID, tournament_type=TournamentType.TEXT), "A", None)

    assert advance_set.await_args.args[2] == next(s.advance for s in t_cst.TEXT_ROUND_PLAN if s.round_number == 2) == 4


@pytest.mark.asyncio
async def test_a_lone_round_three_survivor_whose_training_failed_still_advances_and_the_round_is_not_refused(
    tournament, monkeypatch
):
    # Round 3's only challenger has no result on the round's task (it ranks last there) but is still the challenger; the
    # no-score guard accepts the round because that training failed, so the tournament moves on to the crown check.
    tournament({1: list("ABCD"), 2: list("ABCD"), 3: []}, {1: list("ABCD"), 2: list("ABCD"), 3: ["A"]})
    monkeypatch.setattr(rr, "get_training_status_for_task_and_hotkeys", AsyncMock(return_value={"A": TrainingStatus.FAILURE}))

    assert await _winners(3) == ["A"]
    assert await rr.find_groups_with_no_valid_scores(_round(3), None) == []


@pytest.mark.asyncio
async def test_a_lone_round_three_survivor_with_no_evaluation_is_still_refused(tournament, monkeypatch):
    tournament({1: list("ABCD"), 2: list("ABCD"), 3: []}, {1: list("ABCD"), 2: list("ABCD"), 3: ["A"]})
    monkeypatch.setattr(rr, "get_training_status_for_task_and_hotkeys", AsyncMock(return_value={"A": TrainingStatus.SUCCESS}))

    assert [group.group_id for group in await rr.find_groups_with_no_valid_scores(_round(3), None)] == ["g3"]
