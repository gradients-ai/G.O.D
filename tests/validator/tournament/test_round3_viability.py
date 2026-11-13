"""After the R2 cut, drop miners that cannot finish top or joint top after R3; a lone survivor still plays R3."""

from itertools import permutations
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import validator.tournament.round_results as rr
import validator.tournament.tournament_manager as tm
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tournament.models import FirstTwoRoundRanks
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType


ROUND1_TASK = "11111111-2222-3333-4444-555555555555"
ROUND2_TASK = "21111111-2222-3333-4444-555555555555"
ROUND3_TASK = "31111111-2222-3333-4444-555555555555"
TOURNAMENT_ID = "t"


def _viable(pairs: list[tuple[int, int]]) -> list[bool]:
    ranks = {f"m{i}": FirstTwoRoundRanks(round1=round1, round2=round2) for i, (round1, round2) in enumerate(pairs)}
    return [verdict.viable for verdict in rr.round3_viability_verdicts(ranks)]


@pytest.mark.parametrize(
    ("pairs", "index"),
    [
        ([(4, 4), (1, 1), (2, 2), (3, 3)], 0),
        ([(3, 4), (1, 1), (2, 2), (4, 3)], 0),
        ([(4, 3), (1, 1), (2, 2), (3, 4)], 0),
    ],
)
def test_members_far_behind_are_dropped(pairs, index):
    assert _viable(pairs)[index] is False


@pytest.mark.parametrize(
    ("pairs", "index"),
    [
        ([(1, 2), (2, 4), (3, 1), (4, 3)], 1),
        ([(1, 2), (2, 4), (3, 3), (4, 1)], 2),
    ],
)
def test_members_who_can_still_tie_are_kept(pairs, index):
    assert _viable(pairs)[index] is True


def test_a_single_member_is_viable():
    assert _viable([(1, 1)]) == [True]


def test_the_leader_is_always_viable():
    assert all(_viable([(1, 1), (2, 2), (3, 3), (4, 4)])[:1])


def test_best_total_member_is_always_viable_for_every_first_two_round_rank_assignment():
    for round1 in permutations(range(1, 5)):
        for round2 in permutations(range(1, 5)):
            pairs = list(zip(round1, round2, strict=True))
            viable = _viable(pairs)
            sums = [a + b for a, b in pairs]
            assert viable[sums.index(min(sums))]
            for flag, total in zip(viable, sums, strict=True):
                if total >= 7:
                    assert not flag, (pairs, total)


def test_six_is_viable_only_when_no_rival_is_too_far_ahead():
    assert _viable([(1, 2), (2, 4), (3, 3), (4, 1)])[2] is True  # (3,3): the leader's best case is 2+4=6, a tie
    assert _viable([(1, 1), (2, 2), (3, 3), (4, 4)])[2] is False  # (3,3) behind a 2-sum leader


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


def _round(number: int, round_type: RoundType = RoundType.GROUP) -> TournamentRoundData:
    return TournamentRoundData(
        round_id=f"r{number}", tournament_id=TOURNAMENT_ID, round_number=number, round_type=round_type
    )


@pytest.fixture
def two_rounds(monkeypatch):
    """R1 over a field of 8 (round-1 group), R2 over the same 8, R3 (when its cohort is given) over its cohort in that
    order; cohorts per round number."""

    def _wire(round1_order: list[str], round2_order: list[str], cohorts: dict[int, list[str]]):
        results = {
            ROUND1_TASK: [_instruct(h, 1.0 + i / 10) for i, h in enumerate(round1_order)],
            ROUND2_TASK: [_freeform(h, 100 * (i + 1)) for i, h in enumerate(round2_order)],
            ROUND3_TASK: [_freeform(h, 100 * (i + 1)) for i, h in enumerate(cohorts.get(3, []))],
        }
        task_ids = {1: ROUND1_TASK, 2: ROUND2_TASK, 3: ROUND3_TASK}

        async def fake_results(task_id, psql_db):
            return [result.model_copy() for result in results[task_id]]

        async def fake_rounds(tournament_id, psql_db):
            return [_round(n) for n in sorted(cohorts)]

        async def fake_tasks(round_id, psql_db):
            number = int(round_id.removeprefix("r"))
            return [
                TournamentTask(tournament_id=TOURNAMENT_ID, round_id=round_id, task_id=task_ids[number], group_id=f"g{number}")
            ]

        async def fake_members(group_id, psql_db):
            return [SimpleNamespace(hotkey=h) for h in cohorts[int(group_id.removeprefix("g"))]]

        monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
        monkeypatch.setattr(rr, "get_task", lambda *a, **k: _async(SimpleNamespace(task_type=TaskType.INSTRUCTTEXTTASK)))
        monkeypatch.setattr(rr, "get_tournament_rounds", fake_rounds)
        monkeypatch.setattr(rr, "get_tournament_tasks", fake_tasks)
        monkeypatch.setattr(rr, "get_tournament_group_members", fake_members)
        monkeypatch.setattr(
            rr, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id="t", tournament_type=TournamentType.QUANT))
        )

    return _wire


async def _async(value):
    return value


ALL = ["A", "B", "C", "D", "E", "F", "G", "H"]


@pytest.mark.asyncio
async def test_round3_keeps_everyone_when_nobody_is_out_of_reach(two_rounds):
    # Cohort A,B,C,D ranks (R1,R2): A(1,2) B(2,1) C(3,4) D(4,3): sums 3,3,7,7; C and D cannot tie.
    two_rounds(["A", "B", "C", "D", *"EFGH"], ["B", "A", "D", "C", *"EFGH"], {1: ALL, 2: ALL})
    kept = await rr.round3_viable_winners(TOURNAMENT_ID, _round(2), ["A", "B", "C", "D"], None)
    assert kept == ["A", "B"]


@pytest.mark.asyncio
async def test_fixed_point_reranks_within_the_reduced_cohort_down_to_a_lone_survivor(two_rounds):
    # A(1,1)=2, B(2,2)=4, C(3,3)=6, D(4,4)=8. A dominates: B (4) can reach joint top only if A gets rank 4 -> 6,
    # B rank 1 -> 5: B ahead. So B stays; the pass after dropping C and D re-ranks A,B alone: A(1,1) B(2,2).
    # With two left A=2+r, B=4+r': B ties only if A takes 2 and B 1: 4 vs 5 no -> A 4, B 5. B dropped.
    two_rounds(["A", "B", "C", "D", *"EFGH"], ["A", "B", "C", "D", *"EFGH"], {1: ALL, 2: ALL})
    kept = await rr.round3_viable_winners(TOURNAMENT_ID, _round(2), ["A", "B", "C", "D"], None)
    assert kept == ["A"]


@pytest.mark.asyncio
async def test_round_two_winners_are_viability_filtered(two_rounds):
    two_rounds(["A", "B", "C", "D", *"EFGH"], ["A", "B", "C", "D", *"EFGH"], {1: ALL, 2: ALL})
    winners = await rr.get_group_winners(
        _round(2),
        [TournamentTask(tournament_id=TOURNAMENT_ID, round_id="r2", task_id=ROUND2_TASK, group_id="g2")],
        None,
    )
    assert winners == ["A"]


@pytest.mark.asyncio
async def test_third_place_after_a_lone_round_three_survivor_comes_from_first_two_round_standings_among_the_advanced(
    two_rounds,
):
    """Same order in R1 and R2: the cut keeps A alone, A plays round 3 alone and the boss in round 4. Third place is B, the
    best of the round-2 advance set after A, for the boss-round path and the early stop after round 3 alike."""
    two_rounds(["A", "B", "C", "D", *"EFGH"], ["A", "B", "C", "D", *"EFGH"], {1: ALL, 2: ALL, 3: ["A"]})

    final = _round(4, RoundType.KNOCKOUT).model_copy(update={"is_final_round": True})
    assert await rr.get_quant_pre_boss_runner_up(TOURNAMENT_ID, final, "A", None) == "B"
    assert await rr.get_quant_round_runner_up(TOURNAMENT_ID, _round(3), "A", None) == "B"


@pytest.fixture
def inserted(monkeypatch) -> dict:
    captured: dict = {}

    async def fake_node(hotkey, psql_db):
        return SimpleNamespace(hotkey=hotkey)

    async def fake_insert_round(round_data, psql_db):
        captured["round"] = round_data

    async def fake_noop(*args, **kwargs):
        return None

    async def fake_insert_pairs(round_id, pairs, psql_db):
        captured["pairs"] = pairs

    monkeypatch.setattr(tm, "get_node_by_hotkey", fake_node)
    monkeypatch.setattr(tm, "insert_tournament_round", fake_insert_round)
    monkeypatch.setattr(tm, "insert_tournament_groups_with_members", fake_noop)
    monkeypatch.setattr(tm, "insert_tournament_pairs", fake_insert_pairs)
    return captured


async def _next_round(winners: list[str], completed: TournamentRoundData) -> None:
    tournament = TournamentData(tournament_id="t", tournament_type=TournamentType.QUANT)
    await tm.create_next_round(tournament, completed, winners, MagicMock(), psql_db=None)


@pytest.mark.asyncio
async def test_lone_survivor_after_round_two_still_plays_round_three(inserted):
    """Its round-3 result against the boss's entry is one of the crown's head-to-heads, so round 3 is never skipped."""
    await _next_round(["A"], _round(2))

    assert inserted["round"].round_number == 3
    assert inserted["round"].round_type == RoundType.GROUP
    assert inserted["round"].is_final_round is False
    assert "pairs" not in inserted


@pytest.mark.asyncio
async def test_the_round_three_winner_meets_the_boss_in_round_four(inserted):
    await _next_round(["A"], _round(3))

    assert inserted["round"].round_number == 4
    assert inserted["round"].round_type == RoundType.KNOCKOUT and inserted["round"].is_final_round is True
    assert [sorted(pair) for pair in inserted["pairs"]] == [sorted(["A", EMISSION_BURN_HOTKEY])]


@pytest.mark.asyncio
async def test_reduced_cohort_after_round_two_still_plays_round3(inserted):
    await _next_round(["A", "B"], _round(2))

    assert inserted["round"].round_number == 3
    assert inserted["round"].round_type == RoundType.GROUP
    assert inserted["round"].is_final_round is False
