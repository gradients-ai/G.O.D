"""The boss must be beaten to be got past, in whichever group it was drawn into.

Regression cover for tourn_10592fcefa2f37ad_20260810 round 1 group 1: three of the four
challengers failed training, and the lone survivor advanced on 0.0 having lost both
environments to the boss (78-94 clobber, 11-18 othello). The old rule only let the boss retain
when the round had a single group, so with three groups it did not apply and the survivor went
through on attrition rather than merit.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.round_results import get_environment_group_winners


MODULE = "validator.tournament.round_results"

CHALLENGER = "5EgpWgYvChallenger"
OTHER = "5D7iEJm5Other"
BOSS = EMISSION_BURN_HOTKEY


def _round() -> TournamentRoundData:
    return TournamentRoundData(
        round_id="tourn_test_round_001",
        tournament_id="tourn_test",
        round_number=1,
        round_type=RoundType.GROUP,
        is_final_round=False,
        status=RoundStatus.COMPLETED,
    )


def _task(group_id: str, task_id: str) -> TournamentTask:
    return TournamentTask(
        tournament_id="tourn_test", round_id="tourn_test_round_001", task_id=task_id, group_id=group_id
    )


def _members(*hotkeys: str):
    return [AsyncMock(hotkey=hk) for hk in hotkeys]


def _scored(scores: dict[str, float]):
    """Ranked results as calculate_miner_ranking_and_scores would return them."""
    return [AsyncMock(hotkey=hk, adjusted_loss=score) for hk, score in scores.items()]


@pytest.fixture
def env_group():
    with (
        patch(f"{MODULE}.get_tournament_group_members", new_callable=AsyncMock) as members,
        patch(f"{MODULE}.get_task_results_for_ranking", new_callable=AsyncMock) as results,
        patch(f"{MODULE}.calculate_miner_ranking_and_scores") as ranked,
    ):
        results.return_value = [object()]
        yield members, results, ranked


class TestBossRetains:
    async def test_lone_survivor_beaten_by_boss_does_not_advance(self, env_group):
        """The production case: one challenger left, it lost to the boss, it stays out."""
        members, _results, ranked = env_group
        members.return_value = _members(CHALLENGER)
        ranked.return_value = _scored({BOSS: 1.0, CHALLENGER: 0.0})

        winners = await get_environment_group_winners(_round(), [_task("g1", "t1")], AsyncMock(), None)

        assert winners == []

    async def test_boss_retains_on_a_tie(self, env_group):
        members, _results, ranked = env_group
        members.return_value = _members(CHALLENGER)
        ranked.return_value = _scored({BOSS: 0.5, CHALLENGER: 0.5})

        assert await get_environment_group_winners(_round(), [_task("g1", "t1")], AsyncMock(), None) == []

    async def test_challenger_that_beats_the_boss_advances(self, env_group):
        members, _results, ranked = env_group
        members.return_value = _members(CHALLENGER)
        ranked.return_value = _scored({CHALLENGER: 1.0, BOSS: 0.0})

        assert await get_environment_group_winners(_round(), [_task("g1", "t1")], AsyncMock(), None) == [CHALLENGER]

    async def test_multi_group_round_still_gates_the_boss_group(self, env_group):
        """Three groups, boss drawn into the first. Previously single_group=False skipped the
        gate entirely and the beaten challenger advanced alongside the other groups' winners."""
        members, _results, ranked = env_group
        members.side_effect = [_members(CHALLENGER), _members("5AAA", "5BBB"), _members("5CCC", "5DDD")]
        ranked.side_effect = [
            _scored({BOSS: 1.0, CHALLENGER: 0.0}),   # boss group: nobody advances
            _scored({"5AAA": 0.9, "5BBB": 0.1}),
            _scored({"5CCC": 0.8, "5DDD": 0.2}),
        ]

        winners = await get_environment_group_winners(
            _round(), [_task("g1", "t1"), _task("g2", "t2"), _task("g3", "t3")], AsyncMock(), None
        )

        assert winners == ["5AAA", "5CCC"]
        assert CHALLENGER not in winners

    async def test_groups_without_the_boss_are_unaffected(self, env_group):
        """boss_score is None where the boss never played, so the gate cannot suppress them."""
        members, _results, ranked = env_group
        members.return_value = _members(CHALLENGER, OTHER)
        ranked.return_value = _scored({CHALLENGER: 0.667, OTHER: 0.5})

        winners = await get_environment_group_winners(_round(), [_task("g2", "t2")], AsyncMock(), None)

        assert winners == [CHALLENGER]

    async def test_ties_at_the_cutoff_still_advance_together(self, env_group):
        """Group 3 tonight advanced two miners tied on 0.833; that must not regress."""
        members, _results, ranked = env_group
        members.return_value = _members("5AAA", "5BBB", "5CCC", "5DDD")
        ranked.return_value = _scored({"5AAA": 0.833, "5BBB": 0.833, "5CCC": 0.167, "5DDD": 0.167})

        winners = await get_environment_group_winners(_round(), [_task("g3", "t3")], AsyncMock(), None)

        assert sorted(winners) == ["5AAA", "5BBB"]
