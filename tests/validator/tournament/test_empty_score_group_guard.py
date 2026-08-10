"""Tests for the zero-score group guard that blocks tournament advancement.

Regression cover for tourn_10592fcefa2f37ad_20260810: two of three round-1 groups lost their
evaluation data after miners were re-trained, scored 0/NaN across the board, and were silently
skipped by the winner functions. The round then advanced with one challenger instead of three.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np
import pytest

from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.round_results import find_groups_with_no_valid_scores


MODULE = "validator.tournament.round_results"


def _round(round_type: RoundType = RoundType.GROUP) -> TournamentRoundData:
    return TournamentRoundData(
        round_id="tourn_test_round_001",
        tournament_id="tourn_test",
        round_number=1,
        round_type=round_type,
        is_final_round=False,
        status=RoundStatus.COMPLETED,
    )


def _task(group_id: str, task_id: str) -> TournamentTask:
    return TournamentTask(
        tournament_id="tourn_test",
        round_id="tourn_test_round_001",
        task_id=task_id,
        group_id=group_id,
    )


def _member(hotkey: str):
    member = AsyncMock()
    member.hotkey = hotkey
    return member


def _result(hotkey: str, adjusted_loss):
    result = AsyncMock()
    result.hotkey = hotkey
    result.adjusted_loss = adjusted_loss
    return result


@pytest.fixture
def patched():
    """Patch the three DB reads find_groups_with_no_valid_scores depends on."""
    with (
        patch(f"{MODULE}.get_tournament_tasks", new_callable=AsyncMock) as tasks,
        patch(f"{MODULE}.get_tournament_group_members", new_callable=AsyncMock) as members,
        patch(f"{MODULE}.get_task_results_for_ranking", new_callable=AsyncMock) as results,
        patch(f"{MODULE}.calculate_miner_ranking_and_scores") as ranked,
    ):
        members.return_value = [_member("5A"), _member("5B")]
        yield tasks, members, results, ranked


class TestFindGroupsWithNoValidScores:
    async def test_healthy_group_is_not_flagged(self, patched):
        tasks, _members, results, ranked = patched
        tasks.return_value = [_task("group_001", "task-1")]
        results.return_value = [object()]
        ranked.return_value = [_result("5A", 0.4), _result("5B", 0.6)]

        assert await find_groups_with_no_valid_scores(_round(), AsyncMock()) == []

    async def test_group_with_no_scored_rows_is_flagged(self, patched):
        # get_task_results_for_ranking drops NaN/None test_loss, so a wiped group arrives empty.
        tasks, _members, results, ranked = patched
        tasks.return_value = [_task("group_002", "task-2")]
        results.return_value = []
        ranked.return_value = []

        flagged = await find_groups_with_no_valid_scores(_round(), AsyncMock())

        assert len(flagged) == 1
        assert flagged[0].group_id == "group_002"
        assert flagged[0].task_id == "task-2"
        assert flagged[0].participants == 2

    async def test_group_scored_entirely_nan_is_flagged(self, patched):
        # Rows survive ranking but carry no usable loss — the shape task_nodes was left in.
        tasks, _members, results, ranked = patched
        tasks.return_value = [_task("group_003", "task-3")]
        results.return_value = [object()]
        ranked.return_value = [_result("5A", np.nan), _result("5B", None)]

        flagged = await find_groups_with_no_valid_scores(_round(), AsyncMock())

        assert [group.group_id for group in flagged] == ["group_003"]

    async def test_flags_only_the_broken_groups_in_a_mixed_round(self, patched):
        """The exact production shape: group 1 healthy, groups 2 and 3 wiped."""
        tasks, _members, results, ranked = patched
        tasks.return_value = [
            _task("group_001", "task-1"),
            _task("group_002", "task-2"),
            _task("group_003", "task-3"),
        ]
        results.side_effect = [[object()], [], []]
        ranked.side_effect = [[_result("5A", 0.4), _result("5B", 0.6)]]

        flagged = await find_groups_with_no_valid_scores(_round(), AsyncMock())

        assert [group.group_id for group in flagged] == ["group_002", "group_003"]

    async def test_knockout_rounds_are_not_covered(self, patched):
        tasks, _members, results, ranked = patched
        tasks.return_value = [_task("group_001", "task-1")]
        results.return_value = []
        ranked.return_value = []

        assert await find_groups_with_no_valid_scores(_round(RoundType.KNOCKOUT), AsyncMock()) == []
        tasks.assert_not_called()

    async def test_group_with_no_members_is_skipped(self, patched):
        # An empty group has nothing to lose, so it is not evidence of missing eval data.
        tasks, members, results, ranked = patched
        tasks.return_value = [_task("group_001", "task-1")]
        members.return_value = []
        results.return_value = []
        ranked.return_value = []

        assert await find_groups_with_no_valid_scores(_round(), AsyncMock()) == []
        results.assert_not_called()
