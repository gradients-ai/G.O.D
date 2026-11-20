"""No infrastructure failure is scored as the miner's: process_miners_pool re-queues it, scored fails stay scored."""

import http.client
import math
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import requests
from huggingface_hub.errors import RepositoryNotFoundError
from minio.error import S3Error

import validator.infrastructure.service_constants as service_cst
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantFreeformPenalty
from core.models.task_models import TaskType
from validator.evaluation.errors import EvaluationInfraError
from validator.evaluation.errors import RepoEvaluationFailed
from validator.evaluation.quant_artifacts import HubArtifactListing
from validator.scoring import tasks as scoring_tasks
from validator.scoring.models import MinerResultsText
from validator.scoring.quant_ranking import SCORED_FAIL_REASON_PREFIX
from validator.scoring.tasks import FREEFORM_OVER_CAP_RANKING_TYPE
from validator.scoring.tasks import RANKED_FIRST_REASON_PREFIX
from validator.scoring.tasks import _create_failed_miner_result
from validator.scoring.tasks import calculate_miner_ranking_and_scores
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFreeformRawTask


CONFIG = SimpleNamespace(psql_db="db")
REPO = f"{service_cst.RAYONLABS_HF_USERNAME}/repo-a"


def _fields() -> dict:
    return dict(
        task_id=uuid.uuid4(),
        is_organic=False,
        status="evaluating",
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=4.0,
        created_at=datetime(2026, 11, 10),
        test_data="https://s3/test",
    )


def _instruct_task() -> InstructTextRawTask:
    return InstructTextRawTask(**_fields(), field_instruction="q", field_output="a")


def _freeform_task() -> QuantFreeformRawTask:
    return QuantFreeformRawTask(**_fields(), dataset_type=ChatTemplateDatasetType(chat_template="tokenizer_default"))


def _requests_response(status: int) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    return response


class _HfApi:
    """repo_info raises the class-level error when one is set."""

    error: BaseException | None = None

    def __init__(self, token=None):
        pass

    def repo_info(self, repo, timeout):
        if _HfApi.error is not None:
            raise _HfApi.error


@pytest.fixture
def pool(monkeypatch):
    """process_miners_pool with one miner "A" and _evaluate_submissions stubbed to return state.results."""
    _HfApi.error = None
    state = SimpleNamespace(results={}, notify=AsyncMock())

    async def fake_evaluate(**_kwargs):
        if isinstance(state.results, BaseException):
            raise state.results
        return state.results

    monkeypatch.setattr(scoring_tasks, "get_expected_repo_name", AsyncMock(return_value="repo-a"))
    monkeypatch.setattr(scoring_tasks, "HfApi", _HfApi)
    monkeypatch.setattr(scoring_tasks, "_evaluate_submissions", fake_evaluate)
    monkeypatch.setattr(scoring_tasks, "get_task_node_losses", AsyncMock(return_value=[]))
    monkeypatch.setattr(scoring_tasks, "notify_evaluation_exception", state.notify)
    monkeypatch.setattr(scoring_tasks, "task_deployment_ids_for_hotkeys", AsyncMock(return_value=[]))
    monkeypatch.setattr(scoring_tasks, "decoded_model_urls_for", AsyncMock(return_value={REPO: {"index.json": "u"}}))
    return state


async def _run(task) -> dict[str, MinerResultsText]:
    return {r.hotkey: r for r in await scoring_tasks.process_miners_pool([SimpleNamespace(hotkey="A")], task, CONFIG, 1)}


class TestRequeuedNeverScored:
    @pytest.mark.parametrize(
        "error",
        [
            EvaluationInfraError("Evaluation failed after 1 attempts: ReadTimeout (no message)"),
            Exception("Evaluation failed to complete"),
        ],
    )
    async def test_a_result_that_is_not_the_evaluators_verdict_is_requeued(self, pool, error) -> None:
        pool.results = {REPO: error}

        with pytest.raises(EvaluationInfraError, match="not the miner's fault"):
            await _run(_freeform_task())

    async def test_an_exception_out_of_the_evaluation_call_is_requeued_with_its_type(self, pool) -> None:
        pool.results = httpx.ReadTimeout("")

        with pytest.raises(EvaluationInfraError, match="ReadTimeout"):
            await _run(_freeform_task())

    async def test_object_storage_failure_listing_decoded_models_is_requeued(self, pool, monkeypatch) -> None:
        s3_error = S3Error("InternalError", "boom", "r", "q", "h", SimpleNamespace(status=500))
        monkeypatch.setattr(scoring_tasks, "decoded_model_urls_for", AsyncMock(side_effect=s3_error))

        with pytest.raises(EvaluationInfraError, match="S3Error"):
            await _run(_freeform_task())

    @pytest.mark.parametrize(
        "error",
        [httpx.ReadTimeout(""), requests.ConnectionError("reset"), RuntimeError("401 from our own token")],
    )
    async def test_a_hub_that_cannot_answer_repo_info_is_requeued(self, pool, error) -> None:
        _HfApi.error = error

        with pytest.raises(EvaluationInfraError, match=type(error).__name__):
            await _run(_freeform_task())

    async def test_a_hub_disconnect_listing_artifact_bytes_is_requeued(self, pool, monkeypatch) -> None:
        # A missing byte count is a scored fail for an under-cap freeform row, so hub trouble here must re-queue.
        pool.results = {REPO: EvaluationResultText(is_finetune=True, eval_loss=1.9)}

        def disconnected(repo, token=None):
            raise http.client.RemoteDisconnected("Remote end closed connection without response")

        monkeypatch.setattr(scoring_tasks, "artifact_listing_from_hub", disconnected)

        with pytest.raises(EvaluationInfraError, match="RemoteDisconnected"):
            await _run(_freeform_task())

    async def test_a_missing_expected_repo_name_is_ours(self, pool, monkeypatch) -> None:
        monkeypatch.setattr(scoring_tasks, "get_expected_repo_name", AsyncMock(return_value=None))

        with pytest.raises(EvaluationInfraError, match="No expected repo name"):
            await _run(_freeform_task())


class TestScoredFailsUnchanged:
    async def test_a_repo_the_hub_says_does_not_exist_is_scored(self, pool) -> None:
        _HfApi.error = RepositoryNotFoundError("404 Client Error", response=_requests_response(404))

        results = await _run(_instruct_task())

        assert results["A"].score_reason == "Model repo not found on HuggingFace"
        assert results["A"].is_finetune is False

    async def test_the_evaluators_error_about_the_repo_is_scored(self, pool) -> None:
        pool.results = {REPO: RepoEvaluationFailed("OSError: not a valid checkpoint")}

        results = await _run(_instruct_task())

        assert results["A"].score_reason == "Evaluation failed: OSError: not a valid checkpoint"
        assert math.isnan(results["A"].test_loss)

    async def test_a_quant_failure_is_scored(self, pool, monkeypatch) -> None:
        pool.results = {REPO: EvaluationResultText(is_finetune=True, eval_loss=None, quant_failure="non-finite loss")}
        listing = HubArtifactListing(bytes=10, files=[], revision="r")
        monkeypatch.setattr(scoring_tasks, "artifact_listing_from_hub", lambda repo, token=None: listing)

        results = await _run(_freeform_task())

        assert results["A"].quant_failure == "non-finite loss"
        assert math.isnan(results["A"].test_loss)


class TestTextTasksKeepScoredFailures:
    async def test_a_result_that_is_not_a_score_is_a_failed_miner(self, pool) -> None:
        pool.results = {REPO: Exception("Evaluation failed to complete")}

        results = await _run(_instruct_task())

        assert results["A"].score_reason == "Evaluation failed: Evaluation failed to complete"

    async def test_an_exception_out_of_the_evaluation_call_fails_the_batch_miners(self, pool) -> None:
        pool.results = httpx.ReadTimeout("")

        results = await _run(_instruct_task())

        assert results["A"].is_finetune is False
        assert results["A"].score_reason.startswith("Evaluation failed")

    async def test_a_hub_that_cannot_answer_repo_info_scores_the_miner_zero(self, pool) -> None:
        _HfApi.error = httpx.ReadTimeout("")

        results = await _run(_instruct_task())

        assert results["A"].score_reason == "Model repo not found on HuggingFace"

    async def test_a_missing_expected_repo_name_is_a_failed_miner(self, pool, monkeypatch) -> None:
        monkeypatch.setattr(scoring_tasks, "get_expected_repo_name", AsyncMock(return_value=None))

        results = await _run(_instruct_task())

        assert results["A"].score_reason == "No expected repo name found"


def _freeform_row(hotkey: str, loss: float) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=loss,
        synth_loss=loss,
        is_finetune=True,
        task_type=TaskType.QUANTFREEFORMTASK,
        artifact_bytes=1000,
        decoder_bytes=0,
        reference_loss=2.0,
    )


class TestFreeformRankingReason:
    POLICY = QuantFreeformPenalty.cliff(1.05)  # bar = 2.1 on a 2.0 reference

    def test_over_cap_row_first_only_because_the_other_failed_says_so(self) -> None:
        results = [
            _freeform_row("over", 2.5),
            _create_failed_miner_result("failed", "Evaluation failed: OSError: bad", TaskType.QUANTFREEFORMTASK),
        ]

        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores(results, policy=self.POLICY)}

        assert scored["over"].passed_bar is False
        assert scored["over"].score_reason == f"{RANKED_FIRST_REASON_PREFIX}{FREEFORM_OVER_CAP_RANKING_TYPE}"
        assert scored["failed"].score_reason == "Evaluation failed: OSError: bad"

    def test_under_cap_row_keeps_the_penalised_bytes_reason(self) -> None:
        scored = calculate_miner_ranking_and_scores([_freeform_row("under", 2.0)], policy=self.POLICY)

        assert scored[0].score_reason == f"{RANKED_FIRST_REASON_PREFIX}quant penalised bytes under cap"

    def test_scored_fail_keeps_its_reason(self) -> None:
        fail = _freeform_row("fail", math.nan)
        fail.quant_failure = "structural: missing [x]"

        scored = {r.hotkey: r for r in calculate_miner_ranking_and_scores([_freeform_row("under", 2.0), fail], self.POLICY)}

        assert scored["fail"].score_reason == f"{SCORED_FAIL_REASON_PREFIX}structural: missing [x]"
