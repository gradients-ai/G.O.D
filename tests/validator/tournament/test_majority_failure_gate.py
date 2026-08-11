"""Tests for the majority-training-failure manual review gate.

Regression cover for env task 46e18d06 (tourn_10592fcefa2f37ad_20260810_round_001): 3 of its
5 trainings failed legitimately, so is_tourn_task_completed returned False forever on a task
whose status was SUCCESS. The round could never complete and the Discord warning re-fired
every ~60s. The gate keeps the block deliberate but gives a human a way to clear it.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from core.models.task_models import TaskStatus
from validator.tournament import tournament_manager
from validator.tournament.models import TaskFailureReviewStatus
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentTaskFailureReview


MODULE = "validator.tournament.tournament_manager"

TASK_ID = "46e18d06-f1de-40c6-b39d-54107d3cb0c6"
MOSTLY_FAILED = {"5CAMXmFr": "failure", "5CkiAXSu": "failure", "5EqKHiZG": "failure", "5EgpWgYv": "success", "5GU4Xkd3": "success"}
MOSTLY_PASSED = {"5A": "success", "5B": "success", "5C": "failure"}


def _task() -> TournamentTask:
    return TournamentTask(
        tournament_id="tourn_test",
        round_id="tourn_test_round_001",
        task_id=TASK_ID,
        group_id="tourn_test_round_001_group_001",
    )


def _review(status: TaskFailureReviewStatus) -> TournamentTaskFailureReview:
    return TournamentTaskFailureReview(
        task_id=TASK_ID,
        tournament_id="tourn_test",
        round_id="tourn_test_round_001",
        status=status,
        failed_hotkeys=["5CAMXmFr", "5CkiAXSu", "5EqKHiZG"],
        total_trainings=5,
    )


@pytest.fixture
def gate():
    with (
        patch(f"{MODULE}.get_training_status_for_task", new_callable=AsyncMock) as trainings,
        patch(f"{MODULE}.task_failure_reviews_sql") as reviews,
        patch(f"{MODULE}._notify_discord", new_callable=AsyncMock) as notify,
    ):
        reviews.get_task_failure_review = AsyncMock(return_value=None)
        reviews.insert_task_failure_review = AsyncMock(return_value=True)
        yield trainings, reviews, notify


class TestMajorityFailureGate:
    async def test_blocks_and_opens_a_review_when_majority_failed(self, gate):
        trainings, reviews, notify = gate
        trainings.return_value = MOSTLY_FAILED

        blocked = await tournament_manager._majority_failure_blocks_completion(_task(), MagicMock())

        assert blocked is True
        reviews.insert_task_failure_review.assert_awaited_once()
        opened = reviews.insert_task_failure_review.await_args.args[0]
        assert sorted(opened.failed_hotkeys) == ["5CAMXmFr", "5CkiAXSu", "5EqKHiZG"]
        assert opened.total_trainings == 5
        notify.assert_awaited_once()

    async def test_approved_review_lets_the_task_through(self, gate):
        trainings, reviews, notify = gate
        trainings.return_value = MOSTLY_FAILED
        reviews.get_task_failure_review = AsyncMock(return_value=_review(TaskFailureReviewStatus.APPROVED))

        blocked = await tournament_manager._majority_failure_blocks_completion(_task(), MagicMock())

        assert blocked is False
        reviews.insert_task_failure_review.assert_not_awaited()
        notify.assert_not_awaited()

    async def test_pending_review_keeps_blocking_without_re_alerting(self, gate):
        """The gate is re-checked every cycle; only the cycle that opened it should ping."""
        trainings, reviews, notify = gate
        trainings.return_value = MOSTLY_FAILED
        reviews.get_task_failure_review = AsyncMock(return_value=_review(TaskFailureReviewStatus.PENDING_REVIEW))
        reviews.insert_task_failure_review = AsyncMock(return_value=False)  # row already exists

        blocked = await tournament_manager._majority_failure_blocks_completion(_task(), MagicMock())

        assert blocked is True
        notify.assert_not_awaited()

    async def test_does_not_block_below_threshold(self, gate):
        trainings, reviews, notify = gate
        trainings.return_value = MOSTLY_PASSED

        blocked = await tournament_manager._majority_failure_blocks_completion(_task(), MagicMock())

        assert blocked is False
        reviews.get_task_failure_review.assert_not_awaited()
        notify.assert_not_awaited()


class TestIsTournTaskCompleted:
    async def test_success_task_is_held_while_review_pending(self, gate):
        trainings, _reviews, _notify = gate
        trainings.return_value = MOSTLY_FAILED
        task_obj = MagicMock(status=TaskStatus.SUCCESS.value, task_id=TASK_ID)

        completed, reason = await tournament_manager.is_tourn_task_completed(_task(), task_obj, MagicMock())

        assert completed is False
        assert "manual review" in reason

    async def test_success_task_completes_once_approved(self, gate):
        trainings, reviews, _notify = gate
        trainings.return_value = MOSTLY_FAILED
        reviews.get_task_failure_review = AsyncMock(return_value=_review(TaskFailureReviewStatus.APPROVED))
        task_obj = MagicMock(status=TaskStatus.SUCCESS.value, task_id=TASK_ID)

        completed, reason = await tournament_manager.is_tourn_task_completed(_task(), task_obj, MagicMock())

        assert completed is True
        assert reason == "Task completed successfully"
