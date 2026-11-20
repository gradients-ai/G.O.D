"""Compression rounds 2-3: advance by mean rank over rounds 1..N within the surviving cohort."""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tournament.round_results as rr
from core.models.task_models import TaskType
from validator.scoring.models import MinerResultsText
from validator.tournament import constants as t_cst
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType


ROUND1_TASK = "11111111-2222-3333-4444-555555555555"
ROUND2_TASK = "21111111-2222-3333-4444-555555555555"
ROUND3_TASK = "31111111-2222-3333-4444-555555555555"
TOURNAMENT_ID = "t"


def _instruct(hotkey: str, loss: float) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=TaskType.INSTRUCTTEXTTASK
    )


def _freeform(hotkey: str, total_bytes: int) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=1.0,
        synth_loss=1.0,
        is_finetune=True,
        task_type=TaskType.QUANTFREEFORMTASK,
        artifact_bytes=total_bytes,
        decoder_bytes=0,
        passed_bar=True,
        reference_loss=1.0,
    )


def _fixed(hotkey: str, loss: float) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=TaskType.QUANTFIXEDTASK, reference_loss=1.0
    )


def _fixed_fail(hotkey: str) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=math.nan,
        synth_loss=math.nan,
        is_finetune=True,
        task_type=TaskType.QUANTFIXEDTASK,
        quant_failure="did not load",
    )


def _round(number: int, is_final: bool = False) -> TournamentRoundData:
    return TournamentRoundData(
        round_id=f"r{number}",
        tournament_id=TOURNAMENT_ID,
        round_number=number,
        round_type=RoundType.GROUP,
        is_final_round=is_final,
    )


def _task(round_number: int, task_id: str) -> TournamentTask:
    return TournamentTask(tournament_id=TOURNAMENT_ID, round_id=f"r{round_number}", task_id=task_id, group_id=f"g{round_number}")


@pytest.fixture
def tournament(monkeypatch):
    """Fake a tournament DB: results per task, cohort per round (round number -> hotkeys)."""

    def _wire(results: dict[str, list[MinerResultsText]], cohorts: dict[int, list[str]]):
        task_ids = {1: ROUND1_TASK, 2: ROUND2_TASK, 3: ROUND3_TASK}

        async def fake_results(task_id, psql_db):
            return [result.model_copy() for result in results[task_id]]

        async def fake_rounds(tournament_id, psql_db):
            return [_round(n) for n in sorted(cohorts)]

        async def fake_tasks(round_id, psql_db):
            number = int(round_id.removeprefix("r"))
            return [_task(number, task_ids[number])]

        async def fake_members(group_id, psql_db):
            return [SimpleNamespace(hotkey=hotkey) for hotkey in cohorts[int(group_id.removeprefix("g"))]]

        monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
        monkeypatch.setattr(rr, "get_task", AsyncMock(return_value=SimpleNamespace(task_type=TaskType.INSTRUCTTEXTTASK)))
        monkeypatch.setattr(rr, "get_tournament_rounds", fake_rounds)
        monkeypatch.setattr(rr, "get_tournament_tasks", fake_tasks)
        monkeypatch.setattr(rr, "get_tournament_group_members", fake_members)
        monkeypatch.setattr(
            rr, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id="t", tournament_type=TournamentType.QUANT))
        )

    return _wire


def test_round_advance_sizes():
    assert t_cst.QUANT_ROUND_TWO_ADVANCE == 4
    assert t_cst.QUANT_ROUND_THREE_ADVANCE == 1


@pytest.mark.asyncio
async def test_ranks_are_taken_within_the_cohort_not_the_full_field(tournament):
    # E won R1 outright but was eliminated. Against the full field A=2/B=3 on R1; within the cohort A=1/B=2.
    # R2 reverses A and B, so with cohort ranks both average 1.5 and B wins on the most recent round's rank (R2).
    tournament(
        {
            ROUND1_TASK: [
                _instruct("E", 0.5),
                _instruct("A", 1.0),
                _instruct("B", 1.1),
                _instruct("C", 1.2),
                _instruct("D", 1.3),
            ],
            ROUND2_TASK: [_freeform("B", 100), _freeform("A", 200), _freeform("C", 300), _freeform("D", 400)],
        },
        {1: ["E", "A", "B", "C", "D"], 2: ["A", "B", "C", "D"]},
    )

    ordered = await rr._quant_cumulative_ordered_hotkeys(TOURNAMENT_ID, _round(2), psql_db=None)

    assert "E" not in ordered
    standings = await rr._small_tournament_standings(
        [_task(1, ROUND1_TASK), _task(2, ROUND2_TASK)], None, {"A", "B", "C", "D"}, recent_round_id="r2", seed_round_id="r2"
    )
    by_hotkey = {s.hotkey: s for s in standings}
    assert (by_hotkey["A"].recent_round_rank, by_hotkey["B"].recent_round_rank) == (2.0, 1.0)
    assert by_hotkey["A"].average_rank == by_hotkey["B"].average_rank == 1.5
    assert ordered[:2] == ["B", "A"]


@pytest.mark.asyncio
async def test_round_two_advances_the_top_four_by_mean_rank_over_rounds_one_and_two(tournament):
    hotkeys = [f"h{i}" for i in range(8)]
    # R1 order h0..h7; R2 order reversed for the bottom half so mean ranks differ.
    round1 = [_instruct(hk, 1.0 + i / 10) for i, hk in enumerate(hotkeys)]
    round2_order = ["h1", "h0", "h2", "h3", "h7", "h6", "h5", "h4"]
    round2 = [_freeform(hk, 100 * (i + 1)) for i, hk in enumerate(round2_order)]
    tournament({ROUND1_TASK: round1, ROUND2_TASK: round2}, {1: hotkeys, 2: hotkeys})
    tasks = [_task(2, ROUND2_TASK)]

    top_four = await rr.quant_cumulative_winners(TOURNAMENT_ID, _round(2), t_cst.QUANT_ROUND_TWO_ADVANCE, None)
    winners = await rr.get_group_winners(_round(2), tasks, psql_db=None)

    assert top_four == ["h1", "h0", "h2", "h3"]  # h0 and h1 tie on mean rank; h1 won the latest round
    # h2 (3,3) and h3 (4,4) trail h0 (1,2) by too much to tie after R3, so the viability cut drops them.
    assert winners == ["h1", "h0"]


@pytest.mark.asyncio
async def test_round_three_advances_one_by_mean_rank_over_rounds_one_to_three(tournament):
    cohort = ["A", "B", "C", "D"]
    # Mean ranks: A (1+3+2)/3=2, B (2+1+1)/3=1.33, C (3+2+4)/3=3, D (4+4+3)/3=3.67
    tournament(
        {
            ROUND1_TASK: [_instruct("A", 1.0), _instruct("B", 1.1), _instruct("C", 1.2), _instruct("D", 1.3)],
            ROUND2_TASK: [_freeform("B", 100), _freeform("C", 200), _freeform("A", 300), _freeform("D", 400)],
            ROUND3_TASK: [_fixed("B", 1.0), _fixed("A", 1.1), _fixed("D", 1.2), _fixed("C", 1.3)],
        },
        {1: cohort, 2: cohort, 3: cohort},
    )

    winners = await rr.get_group_winners(_round(3), [_task(3, ROUND3_TASK)], psql_db=None)

    assert winners == ["B"]


@pytest.mark.asyncio
async def test_a_cohort_member_absent_from_a_round_ranks_last_on_it(tournament):
    # B has no R2 result at all: it ranks 3 (len+1) there, so its mean (1+3)/2 loses to A's (2+1)/2... A=1.5, B=2.0.
    tournament(
        {
            ROUND1_TASK: [_instruct("B", 1.0), _instruct("A", 1.1)],
            ROUND2_TASK: [_freeform("A", 100)],
        },
        {1: ["A", "B"], 2: ["A", "B"]},
    )
    standings = await rr._small_tournament_standings(
        [_task(1, ROUND1_TASK), _task(2, ROUND2_TASK)], None, {"A", "B"}, recent_round_id="r2", seed_round_id="r2"
    )

    by_hotkey = {s.hotkey: s for s in standings}
    assert by_hotkey["B"].total_rank == 1 + 2  # len(ranked)+1 on R2
    assert by_hotkey["B"].has_error
    assert [s.hotkey for s in standings] == ["A", "B"]


@pytest.mark.asyncio
async def test_a_scored_fail_ranks_last_on_its_task(tournament):
    cohort = ["A", "B", "C"]
    tournament(
        {
            ROUND1_TASK: [_instruct("A", 1.0), _instruct("B", 1.1), _instruct("C", 1.2)],
            ROUND2_TASK: [_freeform("B", 100), _freeform("A", 200), _freeform("C", 300)],
            ROUND3_TASK: [_fixed("B", 1.0), _fixed("C", 1.1), _fixed_fail("A")],
        },
        {1: cohort, 2: cohort, 3: cohort},
    )
    standings = await rr._small_tournament_standings(
        [_task(1, ROUND1_TASK), _task(2, ROUND2_TASK), _task(3, ROUND3_TASK)],
        None,
        set(cohort),
        recent_round_id="r3",
        seed_round_id="r3",
    )

    a = next(s for s in standings if s.hotkey == "A")
    assert a.total_rank == 1 + 2 + 3
    assert [s.hotkey for s in standings][0] == "B"


@pytest.mark.asyncio
async def test_mean_rank_tie_goes_to_the_better_rank_in_the_most_recent_round(tournament):
    # A: R1 1, R2 2. B: R1 2, R2 1. Both mean 1.5; B was better in the latest round (R2).
    tournament(
        {
            ROUND1_TASK: [_instruct("A", 1.0), _instruct("B", 1.1)],
            ROUND2_TASK: [_freeform("B", 100), _freeform("A", 200)],
        },
        {1: ["A", "B"], 2: ["A", "B"]},
    )

    assert await rr._quant_cumulative_ordered_hotkeys(TOURNAMENT_ID, _round(2), None) == ["B", "A"]


@pytest.mark.asyncio
async def test_gap_then_seeded_shuffle_break_ties_left_after_the_recent_round_rank(tournament):
    # Both tasks in one round (a single-round call): the recent-round rank is the mean rank, so it ties and the summed
    # gap decides: B's gaps sum to 0.1, A's to 1.0.
    cohort = {"A", "B"}
    tournament(
        {
            ROUND1_TASK: [_instruct("A", 1.0), _instruct("B", 1.1)],
            ROUND2_TASK: [_freeform("B", 100), _freeform("A", 200)],
        },
        {1: ["A", "B"], 2: ["A", "B"]},
    )
    tasks = [_task(2, ROUND1_TASK), _task(2, ROUND2_TASK)]
    standings = await rr._small_tournament_standings(tasks, None, cohort, seed_round_id="r2")
    assert [s.hotkey for s in standings] == ["B", "A"]

    # Equal summed gaps (1.0 each): the per-round seeded shuffle decides.
    tournament(
        {
            ROUND1_TASK: [_instruct("A", 1.0), _instruct("B", 2.0)],
            ROUND2_TASK: [_freeform("B", 100), _freeform("A", 200)],
        },
        {1: ["A", "B"], 2: ["A", "B"]},
    )
    standings = await rr._small_tournament_standings(tasks, None, cohort, seed_round_id="r2")
    shuffle = rr.seeded_order(cohort, rr.tiebreak_seed("r2"))
    assert [s.hotkey for s in standings] == sorted(cohort, key=shuffle.__getitem__)


@pytest.mark.asyncio
async def test_runner_up_is_best_non_challenger_of_the_pre_boss_group(tournament, monkeypatch):
    cohort = ["A", "B", "C", "D"]
    tournament(
        {
            ROUND1_TASK: [_instruct("A", 1.0), _instruct("B", 1.1), _instruct("C", 1.2), _instruct("D", 1.3)],
            ROUND2_TASK: [_freeform("A", 100), _freeform("B", 200), _freeform("C", 300), _freeform("D", 400)],
            ROUND3_TASK: [_fixed("A", 1.0), _fixed("B", 1.1), _fixed("C", 1.2), _fixed("D", 1.3)],
        },
        {1: cohort, 2: cohort, 3: cohort},
    )
    final = _round(4, is_final=True).model_copy(update={"round_type": RoundType.KNOCKOUT})

    assert await rr.get_quant_pre_boss_runner_up(TOURNAMENT_ID, final, "A", None) == "B"
    assert await rr.get_quant_pre_boss_runner_up(TOURNAMENT_ID, final, "B", None) == "A"


@pytest.mark.asyncio
async def test_runner_up_is_none_for_a_knockout_pre_boss_round(tournament, monkeypatch):
    tournament({ROUND1_TASK: [_instruct("A", 1.0)]}, {1: ["A"]})

    async def knockout_rounds(tournament_id, psql_db):
        return [_round(3).model_copy(update={"round_type": RoundType.KNOCKOUT})]

    monkeypatch.setattr(rr, "get_tournament_rounds", knockout_rounds)
    final = _round(4, is_final=True)

    assert await rr.get_quant_pre_boss_runner_up(TOURNAMENT_ID, final, "A", None) is None
