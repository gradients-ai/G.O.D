"""Tests for identifying a clean 3rd place from the pre-boss round.

Text/image: the loser of the last single-pair knockout before the boss round.
Must return None (no valid #3, keep top-2-only payout) on any ambiguity -
missing pre-boss round, wrong round shape, ties at the cutoff.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentPairData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.participants import get_pre_boss_knockout_loser


CHALLENGER = "5EgpWgYvChallenger"
LOSER = "5D7iEJm5Loser"
OTHER = "5FOther5FOther5F"


def _final_round(round_number: int = 3) -> TournamentRoundData:
    return TournamentRoundData(
        round_id="final_round",
        tournament_id="tourn_test",
        round_number=round_number,
        round_type=RoundType.KNOCKOUT,
        is_final_round=True,
        status=RoundStatus.COMPLETED,
    )


def _round(round_number: int, round_type: RoundType, is_final_round: bool = False) -> TournamentRoundData:
    return TournamentRoundData(
        round_id=f"round_{round_number}",
        tournament_id="tourn_test",
        round_number=round_number,
        round_type=round_type,
        is_final_round=is_final_round,
        status=RoundStatus.COMPLETED,
    )


# --- get_pre_boss_knockout_loser (TEXT / IMAGE) ---


PARTICIPANTS_MODULE = "validator.tournament.participants"


def _tournament(tournament_type: TournamentType) -> TournamentData:
    return TournamentData(tournament_id="tourn_test", tournament_type=tournament_type)


def _pair(winner: str, loser: str) -> TournamentPairData:
    return TournamentPairData(pair_id="p1", round_id="round_2", hotkey1=winner, hotkey2=loser, winner_hotkey=winner)


def _pair_task(task_id: str = "task-1", pair_id: str = "p1") -> TournamentTask:
    return TournamentTask(tournament_id="tourn_test", round_id="round_2", task_id=task_id, pair_id=pair_id)


@pytest.fixture
def knockout_loser_mocks():
    with (
        patch(f"{PARTICIPANTS_MODULE}.get_tournament_rounds", new_callable=AsyncMock) as rounds,
        patch(f"{PARTICIPANTS_MODULE}.get_tournament_pairs", new_callable=AsyncMock) as pairs,
        patch(f"{PARTICIPANTS_MODULE}.get_tournament_tasks", new_callable=AsyncMock) as tasks,
        patch(f"{PARTICIPANTS_MODULE}._resolve_knockout_task_winner", new_callable=AsyncMock) as task_winner,
    ):
        tasks.return_value = [_pair_task()]
        yield rounds, pairs, tasks, task_winner


class TestGetPreBossKnockoutLoser:
    async def test_image_clean_single_pair_returns_loser(self, knockout_loser_mocks):
        """The check is purely structural.

        The pair's winner is resolved from its task (via ``_resolve_knockout_task_winner``),
        not read off ``tournament_pairs.winner_hotkey`` - nothing ever writes that column.
        """
        rounds, pairs, _tasks, task_winner = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]
        pairs.return_value = [_pair(CHALLENGER, LOSER)]
        task_winner.return_value = CHALLENGER

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result == LOSER

    async def test_text_clean_single_pair_returns_loser(self, knockout_loser_mocks):
        """Text rounds are checked structurally too: the pre-boss task draws its model like any
        other knockout task, so there is no model pin left to key off."""
        rounds, pairs, tasks, task_winner = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]
        pairs.return_value = [_pair(CHALLENGER, LOSER)]
        tasks.return_value = [_pair_task()]
        task_winner.return_value = CHALLENGER

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.TEXT), _final_round(), CHALLENGER, AsyncMock())

        assert result == LOSER

    async def test_no_decided_task_winner_returns_none(self, knockout_loser_mocks):
        """Neither side had a resolvable task winner (e.g. both evaluations failed)."""
        rounds, pairs, _tasks, task_winner = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]
        pairs.return_value = [_pair(CHALLENGER, LOSER)]
        task_winner.return_value = None

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_multi_pair_pre_boss_round_returns_none(self, knockout_loser_mocks):
        """GROUP-shaped small-tournament pre-boss variant (or any multi-pair round) is not
        the clean single head-to-head the payout rule requires."""
        rounds, pairs, _tasks, _task_winner = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]
        pairs.return_value = [_pair(CHALLENGER, LOSER), _pair(OTHER, "5GAnother")]

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_missing_pre_boss_round_returns_none(self, knockout_loser_mocks):
        rounds, _pairs, _tasks, _task_winner = knockout_loser_mocks
        rounds.return_value = [_round(1, RoundType.KNOCKOUT)]

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_pre_boss_round_wrong_type_returns_none(self, knockout_loser_mocks):
        rounds, _pairs, _tasks, _task_winner = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.GROUP)]

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_loser_colliding_with_challenger_returns_none(self, knockout_loser_mocks):
        """Data inconsistency: the pair's non-winner is somehow the resolved challenger."""
        rounds, pairs, _tasks, task_winner = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]
        pairs.return_value = [_pair(LOSER, CHALLENGER)]  # winner=LOSER, other side=CHALLENGER
        task_winner.return_value = LOSER

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result is None
