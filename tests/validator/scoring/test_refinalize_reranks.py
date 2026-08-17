"""Re-finalizing a task must re-rank every valid miner, not just freshly evaluated ones.

Regression cover for the bug where `calculate_miner_ranking_and_scores` skips any result that
already carries a `score_reason`, combined with `_result_from_persisted_row` carrying the previous
ranking reason forward. On a partial re-evaluation + re-finalize that crowned whichever hotkey was
re-evaluated (its reason had been cleared), regardless of its actual loss.
"""

from types import SimpleNamespace

from core.models.task_models import TaskType
from validator.scoring import constants as scoring_cst
from validator.scoring.tasks import BOTTOM_PENALTY_REASON_PREFIX
from validator.scoring.tasks import RANKED_BELOW_REASON_PREFIX
from validator.scoring.tasks import RANKED_FIRST_REASON_PREFIX
from validator.scoring.tasks import _is_ranking_derived_reason
from validator.scoring.tasks import _result_from_persisted_row
from validator.scoring.tasks import calculate_miner_ranking_and_scores


def _image_task():
    return SimpleNamespace(task_type=TaskType.IMAGETASK, task_id=None)


def _row(test_loss, score_reason):
    return {"test_loss": test_loss, "synth_loss": test_loss, "score_reason": score_reason}


class TestIsRankingDerivedReason:
    def test_ranking_reasons_match(self) -> None:
        assert _is_ranking_derived_reason(f"{RANKED_FIRST_REASON_PREFIX}test_loss")
        assert _is_ranking_derived_reason(f"{RANKED_BELOW_REASON_PREFIX}test_loss")
        assert _is_ranking_derived_reason(f"{BOTTOM_PENALTY_REASON_PREFIX}test_loss")

    def test_failure_and_empty_reasons_do_not_match(self) -> None:
        assert not _is_ranking_derived_reason(None)
        assert not _is_ranking_derived_reason("")
        assert not _is_ranking_derived_reason("Non-finetuned submission")
        assert not _is_ranking_derived_reason("Evaluation failed: boom")
        assert not _is_ranking_derived_reason("Invalid test loss")


class TestResultFromPersistedRow:
    def test_ranking_reason_is_dropped_for_valid_loss(self) -> None:
        result = _result_from_persisted_row(
            _image_task(), "hk", _row(0.03, f"{RANKED_FIRST_REASON_PREFIX}test_loss")
        )
        assert result.score_reason is None
        assert result.is_finetune is True
        assert result.test_loss == 0.03

    def test_exclusion_reason_is_preserved_for_valid_loss(self) -> None:
        result = _result_from_persisted_row(_image_task(), "hk", _row(0.0, "Non-finetuned submission"))
        assert result.score_reason == "Non-finetuned submission"

    def test_missing_loss_stays_failed(self) -> None:
        result = _result_from_persisted_row(_image_task(), "hk", _row(None, "Evaluation failed: boom"))
        assert result.is_finetune is False
        assert result.score_reason == "Evaluation failed: boom"


class TestReFinalizeReranksAll:
    def test_lowest_loss_wins_after_refinalize(self) -> None:
        # Simulate re-finalize: every miner already carries a persisted ranking reason from a prior
        # finalize, and the mid-pack miner "cleared" was just re-evaluated (reason back to None).
        persisted = {
            "best": _row(0.010, f"{RANKED_FIRST_REASON_PREFIX}test_loss"),
            "second": _row(0.020, f"{RANKED_BELOW_REASON_PREFIX}test_loss"),
            "cleared": _row(0.030, None),
            "worst": _row(0.040, f"{RANKED_BELOW_REASON_PREFIX}test_loss"),
        }
        task = _image_task()
        results = [_result_from_persisted_row(task, hk, row) for hk, row in persisted.items()]

        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores(results)}

        assert scored["best"].score == scoring_cst.FIRST_PLACE_SCORE
        assert scored["best"].score_reason == f"{RANKED_FIRST_REASON_PREFIX}test_loss"
        # The re-evaluated mid-pack miner must NOT be crowned just because its reason was cleared.
        assert scored["cleared"].score != scoring_cst.FIRST_PLACE_SCORE
        assert scored["cleared"].score_reason == f"{RANKED_BELOW_REASON_PREFIX}test_loss"

    def test_non_finetune_stays_excluded_after_refinalize(self) -> None:
        # A base-model (non-finetune) submission can have a very low loss but must never win; its
        # exclusion reason is preserved on reload, so it is skipped by the ranking.
        persisted = {
            "real_winner": _row(0.050, f"{RANKED_FIRST_REASON_PREFIX}test_loss"),
            "base_model": _row(0.000, "Non-finetuned submission"),
        }
        task = _image_task()
        results = [_result_from_persisted_row(task, hk, row) for hk, row in persisted.items()]

        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores(results)}

        assert scored["real_winner"].score == scoring_cst.FIRST_PLACE_SCORE
        assert scored["base_model"].score != scoring_cst.FIRST_PLACE_SCORE
        assert scored["base_model"].score_reason == "Non-finetuned submission"
