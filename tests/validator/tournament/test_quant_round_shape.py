"""Quant tournament shape: round 1 group of 8 -> round 2 group of 4 -> round 3 group of 1 -> boss round."""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import validator.tournament.round_results as rr
import validator.tournament.tournament_manager as tm
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tournament import constants as t_cst
from validator.tournament.models import GroupRound
from validator.tournament.models import KnockoutRound
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType


FIXED_TASK_ID = "11111111-2222-3333-4444-555555555555"
FREEFORM_TASK_ID = "21111111-2222-3333-4444-555555555555"
INF = float("inf")


def _nodes(hotkeys: list[str]) -> list[SimpleNamespace]:
    return [SimpleNamespace(hotkey=hotkey) for hotkey in hotkeys]


def _hotkeys(n: int) -> list[str]:
    return [f"hk_{i:03d}" for i in range(n)]


def _organise(n: int, tournament_type: TournamentType, round_number: int):
    return tm.organise_tournament_round(
        _nodes(_hotkeys(n)), MagicMock(), tournament_type, round_id="r", round_number=round_number
    )


def test_quant_round_two_is_one_group_of_everyone():
    result = _organise(8, TournamentType.QUANT, round_number=2)

    assert isinstance(result, GroupRound)
    assert result.round_number == 2
    assert len(result.groups) == 1
    assert sorted(result.groups[0].member_ids) == _hotkeys(8)


def test_image_round_two_is_still_a_knockout():
    assert isinstance(_organise(8, TournamentType.IMAGE, round_number=2), KnockoutRound)


def test_quant_round_three_is_one_group_of_the_survivors():
    result = _organise(4, TournamentType.QUANT, round_number=3)

    assert isinstance(result, GroupRound)
    assert len(result.groups) == 1
    assert sorted(result.groups[0].member_ids) == _hotkeys(4)


@pytest.fixture
def inserted(monkeypatch) -> dict:
    """Stub create_next_round's DB calls; return what it inserted."""
    captured: dict = {}

    async def fake_node(hotkey, psql_db):
        return SimpleNamespace(hotkey=hotkey)

    async def fake_insert_round(round_data, psql_db):
        captured["round"] = round_data

    async def fake_insert_groups(round_id, round_structure, psql_db):
        captured["structure"] = round_structure

    async def fake_insert_pairs(round_id, pairs, psql_db):
        captured["pairs"] = pairs

    monkeypatch.setattr(tm, "get_node_by_hotkey", fake_node)
    monkeypatch.setattr(tm, "insert_tournament_round", fake_insert_round)
    monkeypatch.setattr(tm, "insert_tournament_groups_with_members", fake_insert_groups)
    monkeypatch.setattr(tm, "insert_tournament_pairs", fake_insert_pairs)
    return captured


def _completed(round_number: int, round_type: RoundType) -> TournamentRoundData:
    return TournamentRoundData(round_id="r", tournament_id="t", round_number=round_number, round_type=round_type)


async def _next_round(winners: list[str], completed: TournamentRoundData) -> None:
    tournament = TournamentData(tournament_id="t", tournament_type=TournamentType.QUANT)
    await tm.create_next_round(tournament, completed, winners, MagicMock(), psql_db=None)


@pytest.mark.asyncio
async def test_round_one_winners_build_a_non_final_round_two_group(inserted):
    await _next_round(_hotkeys(8), _completed(1, RoundType.GROUP))

    assert inserted["round"].round_number == 2
    assert inserted["round"].round_type == RoundType.GROUP
    assert inserted["round"].is_final_round is False
    (group,) = inserted["structure"].groups
    assert sorted(group.member_ids) == _hotkeys(8)


@pytest.mark.asyncio
async def test_round_two_never_gets_the_burn_hotkey(inserted):
    await _next_round([*_hotkeys(7), EMISSION_BURN_HOTKEY], _completed(1, RoundType.GROUP))

    (group,) = inserted["structure"].groups
    assert EMISSION_BURN_HOTKEY not in group.member_ids
    assert sorted(group.member_ids) == _hotkeys(7)


@pytest.mark.asyncio
async def test_a_lone_challenger_still_plays_round_two(inserted):
    await _next_round(_hotkeys(1), _completed(1, RoundType.GROUP))

    assert inserted["round"].is_final_round is False
    assert isinstance(inserted["structure"], GroupRound)
    (group,) = inserted["structure"].groups
    assert group.member_ids == _hotkeys(1)


@pytest.mark.asyncio
async def test_round_two_winners_build_a_non_final_round_three_group(inserted):
    await _next_round([*_hotkeys(4), EMISSION_BURN_HOTKEY], _completed(2, RoundType.GROUP))

    assert inserted["round"].round_type == RoundType.GROUP
    assert inserted["round"].is_final_round is False
    (group,) = inserted["structure"].groups
    assert sorted(group.member_ids) == _hotkeys(4)


@pytest.mark.asyncio
async def test_round_three_winner_meets_the_burn_hotkey_in_the_boss_round(inserted):
    await _next_round(_hotkeys(1), _completed(3, RoundType.GROUP))

    assert inserted["round"].round_number == 4
    assert inserted["round"].round_type == RoundType.KNOCKOUT
    assert inserted["round"].is_final_round is True
    assert [sorted(pair) for pair in inserted["pairs"]] == [sorted([_hotkeys(1)[0], EMISSION_BURN_HOTKEY])]


def _fixed(hotkey: str, loss: float) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=TaskType.QUANTFIXEDTASK, reference_loss=1.0
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


def _scored_fail(hotkey: str, task_type: TaskType) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=math.nan,
        synth_loss=math.nan,
        is_finetune=True,
        task_type=task_type,
        quant_failure="did not load",
    )


@pytest.fixture
def round_two(monkeypatch):
    """Serve a two-quant-task R2 through the real ranking; return the completed round and its tasks."""

    def _wire(fixed_results, freeform_results):
        results = {FIXED_TASK_ID: fixed_results, FREEFORM_TASK_ID: freeform_results}

        async def fake_results(task_id, psql_db):
            return [result.model_copy() for result in results[task_id]]

        async def fake_task(task_id, psql_db):
            return SimpleNamespace(task_type=TaskType.QUANTFIXEDTASK)

        monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
        monkeypatch.setattr(rr, "get_task", fake_task)
        monkeypatch.setattr(
            rr, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id="t", tournament_type=TournamentType.QUANT))
        )
        completed = TournamentRoundData(round_id="r2", tournament_id="t", round_number=2, round_type=RoundType.GROUP)
        tasks = [
            TournamentTask(tournament_id="t", round_id="r2", task_id=task_id, group_id="g")
            for task_id in (FIXED_TASK_ID, FREEFORM_TASK_ID)
        ]
        return completed, tasks

    return _wire


@pytest.mark.asyncio
async def test_a_scored_fail_ranks_last_on_its_task(round_two):
    # D is best on freeform but scored-failed fixed: average rank 2.5 against A's 2.0 from two seconds.
    _, tasks = round_two(
        [_fixed("B", 1.0), _fixed("A", 1.1), _fixed("C", 1.2), _scored_fail("D", TaskType.QUANTFIXEDTASK)],
        [_freeform("D", 100), _freeform("A", 200), _freeform("C", 300), _freeform("B", 400)],
    )

    ordered = await rr._quant_ordered_hotkeys(tasks, psql_db=None)

    assert ordered.index("A") < ordered.index("D")


@pytest.mark.asyncio
async def test_single_task_round_one_advances_the_top_eight_by_rank(monkeypatch):
    hotkeys = _hotkeys(12)
    losses = {hotkey: 1.0 + i / 100 for i, hotkey in enumerate(reversed(hotkeys))}

    async def fake_results(task_id, psql_db):
        return [
            MinerResultsText(hotkey=hk, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=TaskType.INSTRUCTTEXTTASK)
            for hk, loss in losses.items()
        ]

    monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
    monkeypatch.setattr(rr, "get_task", AsyncMock(return_value=SimpleNamespace(task_type=TaskType.INSTRUCTTEXTTASK)))
    monkeypatch.setattr(
        rr, "get_tournament", AsyncMock(return_value=TournamentData(tournament_id="t", tournament_type=TournamentType.QUANT))
    )
    task = TournamentTask(tournament_id="t", round_id="r", task_id=FIXED_TASK_ID, group_id="g")

    winners = await rr.get_group_winners(_completed(1, RoundType.GROUP), [task], psql_db=None)

    assert winners == list(reversed(hotkeys))[: t_cst.QUANT_ROUND_ONE_ADVANCE]


def test_infinite_metrics_map_to_infinite_gaps():
    assert rr._match_gaps({"A": 2.0, "B": INF}, is_higher_better=False) == {"A": 0.0, "B": INF}
    assert rr._match_gaps({"A": INF, "B": INF}, is_higher_better=False) == {"A": INF, "B": INF}


@pytest.mark.asyncio
async def test_round_two_entrants_fill_round_one_advance_slots(monkeypatch):
    ordered = _hotkeys(12)
    monkeypatch.setattr(rr, "get_tournament_tasks", AsyncMock(return_value=[]))
    monkeypatch.setattr(rr, "_quant_ordered_hotkeys", AsyncMock(return_value=ordered))
    tournament = TournamentData(tournament_id="t", tournament_type=TournamentType.QUANT)

    selected = await rr.select_round2_entrants(tournament, _completed(1, RoundType.GROUP), {"hk_001"}, psql_db=None)

    assert len(selected) == t_cst.QUANT_ROUND_ONE_ADVANCE
    assert selected == [hotkey for hotkey in ordered if hotkey != "hk_001"][: t_cst.QUANT_ROUND_ONE_ADVANCE]
