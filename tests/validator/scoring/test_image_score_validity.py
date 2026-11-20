"""An image entry that is not a valid fine-tune never holds a valid score.

The path under test is evaluation result -> persisted raw row -> finalize / get_task_results_for_ranking ->
calculate_miner_ranking_and_scores, with in-memory fakes in place of the database.
"""

import math
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from core.models.payload_models import DiffusionLosses
from core.models.payload_models import EvaluationResultImage
from core.models.task_models import TaskType
from validator.db import constants as db_cst
from validator.scoring import constants as scoring_cst
from validator.scoring import tasks as scoring_tasks
from validator.scoring.tasks import INVALID_TEST_LOSS_REASON
from validator.scoring.tasks import NON_FINETUNED_REASON
from validator.scoring.tasks import RANKED_FIRST_REASON_PREFIX
from validator.scoring.tasks import _calculate_weighted_loss_for_image_eval
from validator.scoring.tasks import _create_failed_miner_result
from validator.scoring.tasks import _image_miner_result
from validator.scoring.tasks import calculate_miner_ranking_and_scores
from validator.tournament import task_results


TASK_ID = uuid.uuid4()
TASK = SimpleNamespace(task_id=TASK_ID, task_type=TaskType.IMAGETASK)


def _eval(text_guided: list[float], no_text: list[float], is_finetune: bool = True) -> EvaluationResultImage:
    return EvaluationResultImage(
        eval_loss=DiffusionLosses(text_guided_losses=text_guided, no_text_losses=no_text), is_finetune=is_finetune
    )


class FakeDb:
    """task_nodes rows keyed by hotkey, written by the real persist/finalize code through patched SQL calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rows: dict[str, dict] = {}

        async def set_task_node_losses(task_id, hotkey, test_loss, synth_loss, psql_db, score_reason=None, **kwargs):
            self.rows[hotkey] = {
                db_cst.HOTKEY: hotkey,
                db_cst.TEST_LOSS: test_loss,
                db_cst.SYNTH_LOSS: synth_loss,
                db_cst.SCORE_REASON: score_reason,
                db_cst.TASK_NODE_QUALITY_SCORE: None,
                db_cst.PER_EXAMPLE_LOSSES: kwargs["per_example_losses"],
            }

        async def set_task_node_quality_score(
            task_id, hotkey, quality_score, test_loss, synth_loss, psql_db, score_reason=None, **kwargs
        ):
            row = self.rows[hotkey]
            row[db_cst.TASK_NODE_QUALITY_SCORE] = quality_score
            row[db_cst.TEST_LOSS] = None if test_loss is None or math.isnan(test_loss) else test_loss
            row[db_cst.SCORE_REASON] = score_reason

        async def get_task_node_losses(task_id, psql_db):
            return list(self.rows.values())

        async def get_all_scores_and_losses_for_task(task_id, psql_db):
            return [row for row in self.rows.values() if row[db_cst.TASK_NODE_QUALITY_SCORE] is not None]

        async def get_task(task_id, psql_db):
            return TASK

        async def never(*args, **kwargs):
            return False

        async def no_tournament_task(*args, **kwargs):
            return None

        monkeypatch.setattr(scoring_tasks, "set_task_node_losses", set_task_node_losses)
        monkeypatch.setattr(scoring_tasks, "set_task_node_quality_score", set_task_node_quality_score)
        monkeypatch.setattr(scoring_tasks, "get_task_node_losses", get_task_node_losses)
        monkeypatch.setattr(scoring_tasks, "is_paired_comparison_task", never)
        monkeypatch.setattr(scoring_tasks.tournament_sql, "get_tournament_task_by_task_id", no_tournament_task)
        monkeypatch.setattr(task_results, "get_all_scores_and_losses_for_task", get_all_scores_and_losses_for_task)
        monkeypatch.setattr(task_results, "get_task", get_task)

    async def persist(self, entries: dict[str, EvaluationResultImage]) -> None:
        results = [_image_miner_result(hotkey, result, None) for hotkey, result in entries.items()]
        await scoring_tasks._persist_raw_task_results(TASK, results, "db")

    async def ranking(self) -> dict[str, float]:
        """hotkey -> adjusted_loss for everything tournament ranking would treat as a valid score."""
        ranked = calculate_miner_ranking_and_scores(await task_results.get_task_results_for_ranking(str(TASK_ID), "db"))
        return {r.hotkey: r.adjusted_loss for r in ranked if r.adjusted_loss is not None and not np.isnan(r.adjusted_loss)}


class TestWeightedLoss:
    def test_normal_entry_is_the_half_half_mean(self) -> None:
        loss = _calculate_weighted_loss_for_image_eval(_eval([0.2, 0.4, 0.6], [0.1, 0.1, 0.4]))
        assert loss == pytest.approx(0.5 * 0.4 + 0.5 * 0.2)

    @pytest.mark.parametrize(
        "text_guided, no_text", [([], []), ([0.2], []), ([], [0.2])], ids=["both-empty", "no-text-empty", "text-guided-empty"]
    )
    def test_empty_vector_is_not_a_loss(self, text_guided: list[float], no_text: list[float]) -> None:
        assert math.isnan(_calculate_weighted_loss_for_image_eval(_eval(text_guided, no_text)))

    def test_plain_float_loss_is_not_a_loss(self) -> None:
        assert math.isnan(_calculate_weighted_loss_for_image_eval(EvaluationResultImage(eval_loss=0.3, is_finetune=True)))


class TestImageMinerResult:
    def test_normal_entry_is_valid(self) -> None:
        result = _image_miner_result("hk", _eval([0.2, 0.4], [0.1, 0.3]), None)
        assert result.is_finetune is True
        assert result.test_loss == pytest.approx(0.25)
        assert result.synth_loss == result.test_loss
        assert result.score_reason is None

    def test_base_model_stub_has_no_loss(self) -> None:
        result = _image_miner_result("hk", _eval([0], [0], is_finetune=False), None)
        assert result.is_finetune is False
        assert math.isnan(result.test_loss)
        assert result.score_reason == NON_FINETUNED_REASON

    def test_empty_vectors_have_no_loss(self) -> None:
        result = _image_miner_result("hk", _eval([], []), None)
        assert math.isnan(result.test_loss)
        assert result.score_reason == INVALID_TEST_LOSS_REASON

    def test_unknown_finetune_status_is_not_a_fine_tune(self) -> None:
        unknown = EvaluationResultImage(eval_loss=DiffusionLosses(text_guided_losses=[0.1], no_text_losses=[0.1]))
        result = _image_miner_result("hk", unknown, None)
        assert result.is_finetune is False
        assert math.isnan(result.test_loss)


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_non_finetune_entry_is_persisted_as_no_loss(self, monkeypatch) -> None:
        db = FakeDb(monkeypatch)
        await db.persist({"base": _eval([0], [0], is_finetune=False)})
        row = db.rows["base"]
        assert row[db_cst.TEST_LOSS] is None
        assert row[db_cst.SYNTH_LOSS] is None
        assert row[db_cst.SCORE_REASON] == NON_FINETUNED_REASON

    @pytest.mark.asyncio
    async def test_non_finetune_never_outranks_a_real_entry(self, monkeypatch) -> None:
        db = FakeDb(monkeypatch)
        await db.persist({"base": _eval([0], [0], is_finetune=False), "real": _eval([0.5], [0.3]), "other": _eval([0.9], [0.7])})
        finalized = await scoring_tasks.finalize_task_scores_from_raw_losses(
            TASK, ["base", "real", "other"], SimpleNamespace(psql_db="db")
        )
        by_hotkey = {r.hotkey: r for r in finalized}

        assert by_hotkey["real"].score == scoring_cst.FIRST_PLACE_SCORE
        assert by_hotkey["real"].score_reason.startswith(RANKED_FIRST_REASON_PREFIX)
        assert math.isnan(by_hotkey["base"].test_loss)
        assert by_hotkey["base"].score_reason == NON_FINETUNED_REASON
        assert by_hotkey["base"].score < by_hotkey["real"].score
        ranking = await db.ranking()
        assert "base" not in ranking
        assert sorted(ranking, key=ranking.get) == ["real", "other"]
        assert ranking == {"real": pytest.approx(0.4), "other": pytest.approx(0.8)}

    @pytest.mark.asyncio
    async def test_empty_vector_entry_is_not_a_valid_score(self, monkeypatch) -> None:
        db = FakeDb(monkeypatch)
        await db.persist({"empty": _eval([], []), "real": _eval([0.5], [0.3])})
        await scoring_tasks.finalize_task_scores_from_raw_losses(TASK, ["empty", "real"], SimpleNamespace(psql_db="db"))
        assert db.rows["empty"][db_cst.TEST_LOSS] is None
        assert await db.ranking() == {"real": pytest.approx(0.4)}

    @pytest.mark.asyncio
    async def test_only_non_finetune_entries_leave_no_valid_score(self, monkeypatch) -> None:
        db = FakeDb(monkeypatch)
        await db.persist({"base": _eval([0], [0], is_finetune=False), "empty": _eval([], [])})
        await scoring_tasks.finalize_task_scores_from_raw_losses(TASK, ["base", "empty"], SimpleNamespace(psql_db="db"))
        assert await db.ranking() == {}

    def test_failed_evaluation_has_no_loss(self) -> None:
        result = _create_failed_miner_result("hk", "Evaluation failed: boom", TaskType.IMAGETASK)
        assert math.isnan(result.test_loss)
        assert result.is_finetune is False
