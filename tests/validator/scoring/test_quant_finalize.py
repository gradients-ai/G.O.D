"""Quant finalize: where the freeform bar comes from, the reference exclusion, and the scored-fail carriers."""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.db.constants as db_cst
from core.constants.environments import TrainingStartPoint
from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantEvalGpu
from core.models.task_models import TaskType
from validator.scoring import constants as scoring_cst
from validator.scoring import tasks as scoring_tasks
from validator.scoring.models import MinerResultsText
from validator.scoring.models import QuantSkuReference
from validator.scoring.quant_ranking import QuantReferenceUnavailable
from validator.scoring.tasks import _evaluate_submissions
from validator.scoring.tasks import _result_from_persisted_row
from validator.scoring.tasks import _update_scores as _real_update_scores
from validator.scoring.tasks import calculate_miner_ranking_and_scores
from validator.scoring.tasks import finalize_task_scores_from_raw_losses
from validator.tasks.models import freeform_penalty_policy
from validator.tournament import task_results


CONFIG = SimpleNamespace(psql_db=None)
SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")
PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")


def _freeform_task(training_start_point=TrainingStartPoint.DEFAULT, bar_cap=0.05, penalty_knee=None, penalty_curvature=None):
    return SimpleNamespace(
        task_id="task-1",
        task_type=TaskType.QUANTFREEFORMTASK,
        training_start_point=training_start_point,
        bar_cap=bar_cap,
        penalty_knee=penalty_knee,
        penalty_curvature=penalty_curvature,
    )


def _penalised_task():
    return _freeform_task(bar_cap=0.08, penalty_knee=0.04, penalty_curvature=2500.0)


def _reference(gpu, loss, hotkey=""):
    return QuantSkuReference(
        task_id="00000000-0000-0000-0000-000000000001",
        hotkey=hotkey,
        sku=gpu.reference_key,
        model="org/base",
        loss=loss,
        per_example_losses=[loss],
        eval_set_fingerprint="fp",
        eval_gpu=gpu,
    )


def _row(hotkey, test_loss, artifact_bytes=1000, decoder_bytes=0, quant_failure=None, passed_bar=None, gpu=SXM):
    return {
        db_cst.HOTKEY: hotkey,
        db_cst.TEST_LOSS: test_loss,
        db_cst.SYNTH_LOSS: test_loss,
        db_cst.SCORE_REASON: None,
        db_cst.ARTIFACT_BYTES: artifact_bytes,
        db_cst.DECODER_BYTES: decoder_bytes,
        db_cst.PASSED_BAR: passed_bar,
        db_cst.QUANT_FAILURE: quant_failure,
        db_cst.EVAL_GPU: gpu.model_dump() if gpu else None,
    }


@pytest.fixture
def persisted(monkeypatch):
    """Patch the raw-loss and reference reads and the score write; returns a setter for rows and references."""
    rows: list[dict] = []
    references: dict = {}
    monkeypatch.setattr(scoring_tasks, "get_task_node_losses", AsyncMock(side_effect=lambda *_: rows))
    monkeypatch.setattr(scoring_tasks, "get_quant_sku_references", AsyncMock(side_effect=lambda *_: references))
    monkeypatch.setattr(scoring_tasks, "_update_scores", AsyncMock())

    def install(new_rows, sku_references=(_reference(SXM, 2.0),)):
        rows.extend(new_rows)
        references.update({reference.key: reference for reference in sku_references})

    return install


class TestFreeformFinalizeBar:
    async def test_each_row_is_held_to_its_own_skus_reference(self, persisted) -> None:
        # SXM reference 2.0 -> bar 2.1, PCIe reference 2.04 -> bar 2.142: the same 2.12 fails one and passes the other.
        persisted(
            [_row("near", 2.1, artifact_bytes=5000), _row("far", 2.2, artifact_bytes=10), _row("pcie", 2.12, gpu=PCIE)],
            [_reference(SXM, 2.0), _reference(PCIE, 2.04)],
        )
        results = await finalize_task_scores_from_raw_losses(_freeform_task(), ["near", "far", "pcie"], CONFIG)

        by_hotkey = {r.hotkey: r for r in results}
        assert by_hotkey["near"].passed_bar is True
        assert by_hotkey["pcie"].passed_bar is True
        assert by_hotkey["pcie"].reference_loss == 2.04
        assert by_hotkey["pcie"].score == scoring_cst.FIRST_PLACE_SCORE
        assert by_hotkey["far"].passed_bar is False

    async def test_own_model_rows_are_normalised_by_their_own_miners_reference(self, persisted) -> None:
        # Same raw loss and bytes, but "a" was quantised from a model whose bf16 loss is 2.0 and "b" from one at 2.5:
        # a's bar is 2.1 (2.2 fails), b's is 2.625 (2.2 passes).
        persisted([_row("a", 2.2), _row("b", 2.2)], [_reference(SXM, 2.0, "a"), _reference(SXM, 2.5, "b")])
        task = _freeform_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL)
        results = await finalize_task_scores_from_raw_losses(task, ["a", "b"], CONFIG)

        by_hotkey = {r.hotkey: r for r in results}
        assert by_hotkey["a"].reference_loss == 2.0 and by_hotkey["a"].passed_bar is False
        assert by_hotkey["b"].reference_loss == 2.5 and by_hotkey["b"].passed_bar is True

    async def test_own_model_row_without_its_own_reference_does_not_borrow_another_miners(self, persisted) -> None:
        persisted([_row("a", 2.0), _row("b", 2.0)], [_reference(SXM, 2.0, "a")])
        task = _freeform_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL)
        with pytest.raises(QuantReferenceUnavailable):
            await finalize_task_scores_from_raw_losses(task, ["a", "b"], CONFIG)

    @pytest.mark.parametrize("gpu", [PCIE, None], ids=["uncached-sku", "no-eval-gpu"])
    async def test_raises_for_a_row_without_a_reference_on_its_sku(self, persisted, gpu) -> None:
        persisted([_row("cached", 2.0), _row("miner", 2.0, gpu=gpu)])
        with pytest.raises(QuantReferenceUnavailable):
            await finalize_task_scores_from_raw_losses(_freeform_task(), ["cached", "miner"], CONFIG)


class TestFreeformFinalizePenalty:
    async def test_persisted_rows_price_bytes_like_the_live_path(self, persisted) -> None:
        # Reference 2.0: loss 2.1 is r = 1.05 (0.01 over the knee), loss 2.2 is r = 1.10 (over the cap).
        persisted(
            [_row("knee", 2.0, artifact_bytes=1000), _row("past", 2.1, artifact_bytes=900), _row("out", 2.2, artifact_bytes=1)]
        )
        results = await finalize_task_scores_from_raw_losses(_penalised_task(), ["knee", "past", "out"], CONFIG)
        by_hotkey = {r.hotkey: r for r in results}

        live = [
            MinerResultsText(
                hotkey="past",
                test_loss=2.1,
                synth_loss=2.1,
                is_finetune=True,
                task_type=TaskType.QUANTFREEFORMTASK,
                artifact_bytes=900,
                decoder_bytes=0,
                reference_loss=2.0,
                eval_gpu=SXM,
            )
        ]
        live_past = calculate_miner_ranking_and_scores(live, policy=freeform_penalty_policy(_penalised_task()))[0]

        assert by_hotkey["past"].rank_metric == live_past.rank_metric == 900 * math.exp(by_hotkey["past"].freeform_penalty)
        assert by_hotkey["knee"].freeform_penalty == 0.0
        assert by_hotkey["out"].freeform_penalty is None and by_hotkey["out"].passed_bar is False
        assert [by_hotkey[hk].rank for hk in ("knee", "past", "out")] == [1, 2, 3]  # 1000 < 900 * 1.284

    async def test_scores_are_written_with_the_computed_penalty(self, persisted, monkeypatch) -> None:
        persisted([_row("past", 2.1, artifact_bytes=900), _row("out", 2.2)])
        monkeypatch.setattr(scoring_tasks, "_update_scores", _real_update_scores)
        write = AsyncMock()
        monkeypatch.setattr(scoring_tasks, "set_task_node_quality_score", write)

        await finalize_task_scores_from_raw_losses(_penalised_task(), ["past", "out"], CONFIG)

        written = {call.kwargs["hotkey"]: call.kwargs["freeform_penalty"] for call in write.await_args_list}
        assert written["past"] == pytest.approx(2500.0 * (2.1 / 2.0 - 1.04) ** 2)
        assert written["out"] is None

    async def test_task_without_penalty_reranks_under_the_cliff_at_its_bar_cap(self, persisted) -> None:
        persisted([_row("in", 2.098, artifact_bytes=900), _row("out", 2.12, artifact_bytes=1)])
        results = await finalize_task_scores_from_raw_losses(_freeform_task(), ["in", "out"], CONFIG)
        by_hotkey = {r.hotkey: r for r in results}

        assert by_hotkey["in"].passed_bar is True and by_hotkey["in"].freeform_penalty == 0.0
        assert by_hotkey["in"].rank == 1
        assert by_hotkey["out"].passed_bar is False and by_hotkey["out"].freeform_penalty is None


class TestResultFromPersistedRow:
    def test_rebuilds_scored_fail_from_quant_failure(self) -> None:
        row = _row("hk", None, artifact_bytes=123, decoder_bytes=4, quant_failure="does not load")
        result = _result_from_persisted_row(_freeform_task(), "hk", row)

        assert result.is_finetune is True
        assert math.isnan(result.test_loss)
        assert result.quant_failure == "does not load"
        assert (result.artifact_bytes, result.decoder_bytes) == (123, 4)

    def test_carries_bytes_and_passed_bar(self) -> None:
        row = _row("hk", 2.0, artifact_bytes=700, decoder_bytes=50, passed_bar=True)
        result = _result_from_persisted_row(_freeform_task(), "hk", row)

        assert (result.artifact_bytes, result.decoder_bytes, result.passed_bar) == (700, 50, True)
        assert result.total_bytes == 750

    def test_missing_row_is_still_a_no_show(self) -> None:
        result = _result_from_persisted_row(_freeform_task(), "hk", None)
        assert result.is_finetune is False


class TestGetTaskResultsForRanking:
    @pytest.fixture
    def scores(self, monkeypatch):
        def install(task_type, rows):
            monkeypatch.setattr(task_results, "get_all_scores_and_losses_for_task", AsyncMock(return_value=rows))
            task = SimpleNamespace(task_type=task_type, training_start_point=TrainingStartPoint.DEFAULT)
            monkeypatch.setattr(task_results, "get_task", AsyncMock(return_value=task))
            monkeypatch.setattr(
                task_results, "get_quant_sku_references", AsyncMock(return_value={_reference(SXM, 2.0).key: _reference(SXM, 2.0)})
            )

        return install

    async def test_keeps_quant_scored_fails(self, scores) -> None:
        scores(
            TaskType.QUANTFREEFORMTASK,
            [_row("pass", 2.0, passed_bar=True), _row("fail", None, quant_failure="does not load", passed_bar=False)],
        )
        results = {r.hotkey: r for r in await task_results.get_task_results_for_ranking("task-1", None)}

        assert results["fail"].quant_failure == "does not load"
        assert results["fail"].is_finetune is True
        assert (results["pass"].passed_bar, results["pass"].artifact_bytes) == (True, 1000)
        assert (results["pass"].eval_gpu, results["pass"].reference_loss) == (SXM, 2.0)

    async def test_drops_null_loss_rows_for_non_quant(self, scores) -> None:
        scores(TaskType.INSTRUCTTEXTTASK, [_row("ok", 0.1), _row("missing", None)])
        results = await task_results.get_task_results_for_ranking("task-1", None)
        assert [r.hotkey for r in results] == ["ok"]


class TestBaseModelShortcut:
    @staticmethod
    def _task(task_type):
        return SimpleNamespace(task_type=task_type, augmented_model_id=None, model_id="org/base")

    async def test_quant_base_model_is_a_scored_fail(self) -> None:
        results = await _evaluate_submissions(self._task(TaskType.QUANTFIXEDTASK), ["org/base"], num_gpus=1)
        assert results["org/base"] == EvaluationResultText(
            is_finetune=True, eval_loss=None, quant_failure="submission is the uncompressed base model"
        )

    async def test_instruct_base_model_is_non_finetuned(self) -> None:
        results = await _evaluate_submissions(self._task(TaskType.INSTRUCTTEXTTASK), ["org/base"], num_gpus=1)
        assert results["org/base"] == EvaluationResultText(is_finetune=False, eval_loss=0.0)

    async def test_quant_freeform_without_decoded_model_is_an_evaluation_error(self) -> None:
        task = self._task(TaskType.QUANTFREEFORMTASK)
        task.test_data = "https://bucket/test.jsonl"

        results = await _evaluate_submissions(task, ["org/miner"], num_gpus=1)

        assert isinstance(results["org/miner"], Exception)
