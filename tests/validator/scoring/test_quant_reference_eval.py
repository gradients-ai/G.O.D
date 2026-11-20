"""Quant freeform scoring against the bf16 reference: only decoded repos are evaluated, each with its reference model,
and every row's bar is the reference on its own SKU."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.db.constants as db_cst
import validator.infrastructure.service_constants as service_cst
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.payload_models import DockerEvaluationResults
from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantEvalGpu
from validator.evaluation.errors import EvaluationInfraError
from validator.scoring import constants as scoring_cst
from validator.scoring import tasks as scoring_tasks
from validator.scoring.models import QuantSkuReference
from validator.tasks.models import QuantFreeformRawTask


TASK_ID = uuid.uuid4()
CONFIG = SimpleNamespace(psql_db="db")
REPO_NAMES = {"miner": "miner-repo", "undecoded": "undecoded-repo"}
REFERENCE_REPO = f"{service_cst.RAYONLABS_HF_USERNAME}/reference-repo"
MINER_REPO = f"{service_cst.RAYONLABS_HF_USERNAME}/miner-repo"
UNDECODED_REPO = f"{service_cst.RAYONLABS_HF_USERNAME}/undecoded-repo"
SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")
PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")


def _freeform_task() -> QuantFreeformRawTask:
    return QuantFreeformRawTask(
        task_id=TASK_ID,
        is_organic=False,
        status="evaluating",
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=4.0,
        created_at=datetime(2026, 10, 19),
        test_data="https://s3/test",
        dataset_type=ChatTemplateDatasetType(chat_template="tokenizer_default"),
    )


@pytest.fixture
def evaluator(monkeypatch):
    """process_miners_pool seams up to a stubbed run_evaluation_basilica_quant; returns its captured kwargs."""
    calls: list[dict] = []

    async def fake_run(**kwargs):
        calls.append(kwargs)
        results = {repo: EvaluationResultText(is_finetune=True, eval_loss=2.0) for repo in kwargs["models"]}
        return DockerEvaluationResults(results=results, base_model_params_count=1234)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, repo, timeout):
            return None

    lookups: list[dict] = []

    async def decoded_urls(task_id, repos, bucket):
        lookups.append(repos)
        return {MINER_REPO: {"index.json": "https://s3/i"}} if "miner" in repos else {}

    monkeypatch.setattr(scoring_tasks, "run_evaluation_basilica_quant", fake_run)
    monkeypatch.setattr(scoring_tasks, "get_task_node_losses", AsyncMock(return_value=[]))
    monkeypatch.setattr(scoring_tasks, "get_expected_repo_name", AsyncMock(side_effect=lambda _t, hk, _db: REPO_NAMES[hk]))
    monkeypatch.setattr(scoring_tasks, "HfApi", FakeHfApi)
    monkeypatch.setattr(scoring_tasks, "decoded_model_urls_for", decoded_urls)
    monkeypatch.setattr(scoring_tasks, "artifact_bytes_from_hub", lambda repo, token=None: 1000)
    monkeypatch.setattr(scoring_tasks, "notify_evaluation_exception", AsyncMock())
    monkeypatch.setattr(scoring_tasks, "task_deployment_ids_for_hotkeys", AsyncMock(return_value=[]))
    monkeypatch.setattr(scoring_tasks, "quant_reference_model", AsyncMock(return_value=REFERENCE_REPO))
    monkeypatch.setattr(scoring_tasks, "get_quant_sku_references", AsyncMock(return_value={}))
    return SimpleNamespace(calls=calls, lookups=lookups)


async def test_only_decoded_repos_reach_the_evaluator_with_their_reference_model(evaluator) -> None:
    pool = [SimpleNamespace(hotkey="miner")]

    results = {r.hotkey: r for r in await scoring_tasks.process_miners_pool(pool, _freeform_task(), CONFIG, 1)}

    (call,) = evaluator.calls
    assert call["models"] == [MINER_REPO]
    assert call["reference_models"] == {MINER_REPO: REFERENCE_REPO}
    assert results["miner"].test_loss == 2.0


async def test_a_missing_decoded_model_is_requeued_never_a_scored_fail(evaluator) -> None:
    pool = [SimpleNamespace(hotkey=hotkey) for hotkey in ("miner", "undecoded")]

    # The decode upload is ours: a missing decoded model is an evaluation error for the whole call, not the miner's fail.
    with pytest.raises(EvaluationInfraError, match="undecoded: EvaluationInfraError: decoded model not available"):
        await scoring_tasks.process_miners_pool(pool, _freeform_task(), CONFIG, 1)

    assert [call["models"] for call in evaluator.calls] == [[MINER_REPO]]


def _reference(gpu, loss):
    return QuantSkuReference(
        task_id=TASK_ID,
        sku=gpu.reference_key,
        model=REFERENCE_REPO,
        loss=loss,
        per_example_losses=[loss],
        eval_set_fingerprint="fp",
        eval_gpu=gpu,
    )


def _row(hotkey, test_loss, artifact_bytes, gpu=SXM):
    return {
        db_cst.HOTKEY: hotkey,
        db_cst.TEST_LOSS: test_loss,
        db_cst.SYNTH_LOSS: test_loss,
        db_cst.SCORE_REASON: None,
        db_cst.ARTIFACT_BYTES: artifact_bytes,
        db_cst.DECODER_BYTES: 0,
        db_cst.PASSED_BAR: None,
        db_cst.QUANT_FAILURE: None,
        db_cst.EVAL_GPU: gpu.model_dump(),
    }


async def test_finalize_ranks_each_row_against_the_reference_on_its_own_sku(monkeypatch) -> None:
    # The reference scored 2.0 on SXM (bar 2.1) and 2.03 on PCIe (bar 2.1315). "small" passes and is smallest;
    # "over" misses the bar; "pcie_exact" equals the PCIe reference (ratio 1.0, an honest pass); "pcie_pass"
    # would miss the SXM bar but passes its own.
    rows = [
        _row("small", 2.09, artifact_bytes=4_000),
        _row("over", 2.2, artifact_bytes=1_000),
        _row("pcie_exact", 2.03, artifact_bytes=16_000, gpu=PCIE),
        _row("pcie_pass", 2.12, artifact_bytes=8_000, gpu=PCIE),
    ]
    references = {("", gpu.reference_key): _reference(gpu, loss) for gpu, loss in ((SXM, 2.0), (PCIE, 2.03))}
    monkeypatch.setattr(scoring_tasks, "get_task_node_losses", AsyncMock(return_value=rows))
    monkeypatch.setattr(scoring_tasks, "get_quant_sku_references", AsyncMock(return_value=references))
    monkeypatch.setattr(scoring_tasks, "_update_scores", AsyncMock())
    hotkeys = ["small", "over", "pcie_exact", "pcie_pass"]

    results = {r.hotkey: r for r in await scoring_tasks.finalize_task_scores_from_raw_losses(_freeform_task(), hotkeys, CONFIG)}

    assert results["small"].passed_bar is True
    assert results["small"].rank == 1
    assert results["small"].score == scoring_cst.FIRST_PLACE_SCORE
    assert results["over"].passed_bar is False
    assert (results["pcie_pass"].passed_bar, results["pcie_pass"].rank) == (True, 2)
    assert (results["pcie_exact"].passed_bar, results["pcie_exact"].quant_failure) == (True, None)
