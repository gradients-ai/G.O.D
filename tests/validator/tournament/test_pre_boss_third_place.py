"""Tests for identifying a clean 3rd place from the pre-boss round.

Text/image: the loser of the last single-pair knockout before the boss round.
Environment: the 2nd-best non-advancer in the pre-boss group that produced the
boss-round challenger. Both must return None (no valid #3, keep top-2-only payout)
on any ambiguity - missing pre-boss round, wrong round shape, ties at the cutoff.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.constants import PRE_BOSS_MODEL
from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentPairData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.participants import get_pre_boss_knockout_loser
from validator.tournament.round_results import get_pre_boss_group_runner_up


CHALLENGER = "5EgpWgYvChallenger"
LOSER = "5D7iEJm5Loser"
OTHER = "5FOther5FOther5F"
BOSS = EMISSION_BURN_HOTKEY


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


def _pre_boss_task(model_id: str = PRE_BOSS_MODEL, task_type: TaskType = TaskType.INSTRUCTTEXTTASK):
    task = MagicMock()
    task.task_type = task_type
    task.model_id = model_id
    return task


@pytest.fixture
def knockout_loser_mocks():
    with (
        patch(f"{PARTICIPANTS_MODULE}.get_tournament_rounds", new_callable=AsyncMock) as rounds,
        patch(f"{PARTICIPANTS_MODULE}.get_tournament_pairs", new_callable=AsyncMock) as pairs,
        patch(f"{PARTICIPANTS_MODULE}.get_tournament_tasks", new_callable=AsyncMock) as tasks,
        patch(f"{PARTICIPANTS_MODULE}.get_task", new_callable=AsyncMock) as task,
    ):
        yield rounds, pairs, tasks, task


class TestGetPreBossKnockoutLoser:
    async def test_image_clean_single_pair_returns_loser(self, knockout_loser_mocks):
        """IMAGE has no PRE_BOSS_MODEL signal - structural check alone is enough."""
        rounds, pairs, _tasks, _task = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]
        pairs.return_value = [_pair(CHALLENGER, LOSER)]

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result == LOSER

    async def test_text_requires_pre_boss_model_pin(self, knockout_loser_mocks):
        rounds, pairs, tasks, task = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]
        pairs.return_value = [_pair(CHALLENGER, LOSER)]
        tasks.return_value = [TournamentTask(tournament_id="tourn_test", round_id="round_2", task_id="task-1")]
        task.return_value = _pre_boss_task()

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.TEXT), _final_round(), CHALLENGER, AsyncMock())

        assert result == LOSER

    async def test_text_without_pre_boss_model_returns_none(self, knockout_loser_mocks):
        rounds, pairs, tasks, task = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]
        pairs.return_value = [_pair(CHALLENGER, LOSER)]
        tasks.return_value = [TournamentTask(tournament_id="tourn_test", round_id="round_2", task_id="task-1")]
        task.return_value = _pre_boss_task(model_id="Qwen/SomeOtherModel")

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.TEXT), _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_multi_pair_pre_boss_round_returns_none(self, knockout_loser_mocks):
        """GROUP-shaped small-tournament pre-boss variant (or any multi-pair round) is not
        the clean single head-to-head the payout rule requires."""
        rounds, pairs, _tasks, _task = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]
        pairs.return_value = [_pair(CHALLENGER, LOSER), _pair(OTHER, "5GAnother")]

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_missing_pre_boss_round_returns_none(self, knockout_loser_mocks):
        rounds, _pairs, _tasks, _task = knockout_loser_mocks
        rounds.return_value = [_round(1, RoundType.KNOCKOUT)]

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_pre_boss_round_wrong_type_returns_none(self, knockout_loser_mocks):
        rounds, _pairs, _tasks, _task = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.GROUP)]

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_loser_colliding_with_challenger_returns_none(self, knockout_loser_mocks):
        """Data inconsistency: the pair's non-winner is somehow the resolved challenger."""
        rounds, pairs, _tasks, _task = knockout_loser_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]
        pairs.return_value = [_pair(LOSER, CHALLENGER)]  # winner=LOSER, other side=CHALLENGER

        result = await get_pre_boss_knockout_loser(_tournament(TournamentType.IMAGE), _final_round(), CHALLENGER, AsyncMock())

        assert result is None


# --- get_pre_boss_group_runner_up (ENVIRONMENT) ---


ROUND_RESULTS_MODULE = "validator.tournament.round_results"


def _env_task(group_id: str, task_id: str) -> TournamentTask:
    return TournamentTask(tournament_id="tourn_test", round_id="round_2", task_id=task_id, group_id=group_id)


def _members(*hotkeys: str):
    return [AsyncMock(hotkey=hk) for hk in hotkeys]


def _scored(scores: dict[str, float]):
    return [AsyncMock(hotkey=hk, adjusted_loss=score) for hk, score in scores.items()]


@pytest.fixture
def runner_up_mocks():
    with (
        patch(f"{ROUND_RESULTS_MODULE}.get_tournament_rounds", new_callable=AsyncMock) as rounds,
        patch(f"{ROUND_RESULTS_MODULE}.get_tournament_tasks", new_callable=AsyncMock) as tasks,
        patch(f"{ROUND_RESULTS_MODULE}.get_tournament_group_members", new_callable=AsyncMock) as members,
        patch(f"{ROUND_RESULTS_MODULE}.get_task_results_for_ranking", new_callable=AsyncMock) as results,
        patch(f"{ROUND_RESULTS_MODULE}.calculate_miner_ranking_and_scores") as ranked,
    ):
        results.return_value = [object()]
        yield rounds, tasks, members, results, ranked


class TestGetPreBossGroupRunnerUp:
    async def test_clean_runner_up_returned(self, runner_up_mocks):
        rounds, tasks, members, _results, ranked = runner_up_mocks
        rounds.return_value = [_round(2, RoundType.GROUP)]
        tasks.return_value = [_env_task("g1", "t1")]
        members.return_value = _members(CHALLENGER, LOSER, BOSS)
        ranked.return_value = _scored({CHALLENGER: 1.0, LOSER: 0.5, BOSS: 0.0})

        result = await get_pre_boss_group_runner_up("tourn_test", _final_round(), CHALLENGER, AsyncMock())

        assert result == LOSER

    async def test_boss_placeholder_excluded_even_if_it_would_rank_second(self, runner_up_mocks):
        rounds, tasks, members, _results, ranked = runner_up_mocks
        rounds.return_value = [_round(2, RoundType.GROUP)]
        tasks.return_value = [_env_task("g1", "t1")]
        members.return_value = _members(CHALLENGER, BOSS, OTHER)
        ranked.return_value = _scored({CHALLENGER: 1.0, BOSS: 0.5, OTHER: 0.1})

        result = await get_pre_boss_group_runner_up("tourn_test", _final_round(), CHALLENGER, AsyncMock())

        assert result == OTHER

    async def test_no_non_boss_alternative_returns_none(self, runner_up_mocks):
        rounds, tasks, members, _results, ranked = runner_up_mocks
        rounds.return_value = [_round(2, RoundType.GROUP)]
        tasks.return_value = [_env_task("g1", "t1")]
        members.return_value = _members(CHALLENGER, BOSS)
        ranked.return_value = _scored({CHALLENGER: 1.0, BOSS: 0.5})

        result = await get_pre_boss_group_runner_up("tourn_test", _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_tie_at_cutoff_returns_none(self, runner_up_mocks):
        rounds, tasks, members, _results, ranked = runner_up_mocks
        rounds.return_value = [_round(2, RoundType.GROUP)]
        tasks.return_value = [_env_task("g1", "t1")]
        members.return_value = _members(CHALLENGER, LOSER, OTHER)
        ranked.return_value = _scored({CHALLENGER: 1.0, LOSER: 0.5, OTHER: 0.5})

        result = await get_pre_boss_group_runner_up("tourn_test", _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_challenger_not_in_any_pre_boss_group_returns_none(self, runner_up_mocks):
        rounds, tasks, members, _results, _ranked = runner_up_mocks
        rounds.return_value = [_round(2, RoundType.GROUP)]
        tasks.return_value = [_env_task("g1", "t1")]
        members.return_value = _members(OTHER, LOSER)

        result = await get_pre_boss_group_runner_up("tourn_test", _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_missing_pre_boss_round_returns_none(self, runner_up_mocks):
        rounds, _tasks, _members, _results, _ranked = runner_up_mocks
        rounds.return_value = [_round(1, RoundType.GROUP)]

        result = await get_pre_boss_group_runner_up("tourn_test", _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_pre_boss_round_wrong_type_returns_none(self, runner_up_mocks):
        rounds, _tasks, _members, _results, _ranked = runner_up_mocks
        rounds.return_value = [_round(2, RoundType.KNOCKOUT)]

        result = await get_pre_boss_group_runner_up("tourn_test", _final_round(), CHALLENGER, AsyncMock())

        assert result is None

    async def test_cross_group_absolute_highest_wins_even_from_a_different_group(self, runner_up_mocks):
        """Regression for the live case (tourn_31f2e0fe36783f71_20260831): the boss's own
        group can have the weakest real field (everyone there loses to the boss gate) while
        a completely different group has the strongest real competition. 3rd place must come
        from wherever the single best non-boss, non-challenger score actually is, not be
        artificially confined to the challenger's own group."""
        rounds, tasks, members, results, ranked = runner_up_mocks
        rounds.return_value = [_round(2, RoundType.GROUP)]
        tasks.return_value = [_env_task("g1", "t1"), _env_task("g2", "t2")]
        # g1: boss's group - weak real field, all beaten by the boss gate.
        # g2: challenger's group - stronger real field, produces the highest overall non-boss score.
        members.side_effect = [_members(BOSS, "5GWeak1", "5GWeak2"), _members(CHALLENGER, LOSER, OTHER)]
        ranked.side_effect = [
            _scored({BOSS: 0.625, "5GWeak1": 0.375, "5GWeak2": 0.125}),
            _scored({CHALLENGER: 0.5625, LOSER: 0.5, OTHER: 0.4375}),
        ]

        result = await get_pre_boss_group_runner_up("tourn_test", _final_round(), CHALLENGER, AsyncMock())

        assert result == LOSER

    async def test_tie_across_different_groups_returns_none(self, runner_up_mocks):
        rounds, tasks, members, results, ranked = runner_up_mocks
        rounds.return_value = [_round(2, RoundType.GROUP)]
        tasks.return_value = [_env_task("g1", "t1"), _env_task("g2", "t2")]
        members.side_effect = [_members(BOSS, "5GWeak1"), _members(CHALLENGER, LOSER)]
        ranked.side_effect = [
            _scored({BOSS: 0.9, "5GWeak1": 0.5}),
            _scored({CHALLENGER: 1.0, LOSER: 0.5}),
        ]

        result = await get_pre_boss_group_runner_up("tourn_test", _final_round(), CHALLENGER, AsyncMock())

        assert result is None
