"""Quant eval wiring: dispatch from _evaluate_submissions, persisted scored fails, bytes, and the failed list."""

import math
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

import validator.db.constants as db_cst
import validator.infrastructure.service_constants as service_cst
from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import DockerEvaluationResults
from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantEvalGpu
from core.models.quant_models import QuantFormat
from core.models.quant_models import QuantFreeformPenalty
from core.models.quant_models import QuantReferenceScore
from core.models.task_models import TaskType
from validator.evaluation.errors import EvaluationInfraError
from validator.evaluation.quant_artifacts import HubArtifactListing
from validator.scoring import tasks as scoring_tasks
from validator.scoring.models import HubArtifactFile
from validator.scoring.models import MinerResultsText
from validator.scoring.models import QuantSkuReference
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament.models import QuantDecodeRow
from validator.tournament.quant_reference import NO_REFERENCE_MODEL_FAILURE


TASK_ID = uuid.uuid4()
CONFIG = SimpleNamespace(psql_db="db")
REPO_A = f"{service_cst.RAYONLABS_HF_USERNAME}/repo-a"
REPO_B = f"{service_cst.RAYONLABS_HF_USERNAME}/repo-b"
PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")
HUB_FILE = HubArtifactFile(name="model.bin", bytes=123_456, sha256="a" * 64, sha256_kind="sha256")
MINER = {"org/miner": "miner"}
AB = {"org/a": "a", "org/b": "b"}


def _task_fields() -> dict:
    return {
        "task_id": TASK_ID,
        "is_organic": False,
        "status": "evaluating",
        "model_id": "org/base",
        "ds": "org/dataset",
        "account_id": uuid.uuid4(),
        "hours_to_complete": 2.0,
        "created_at": datetime(2026, 10, 19),
        "test_data": "https://bucket/test.jsonl",
        "dataset_type": InstructTextDatasetType(field_instruction="instruction", field_output="output"),
    }


def _fixed_task() -> QuantFixedRawTask:
    return QuantFixedRawTask(**_task_fields(), quant_spec=QuantFormat.W4A8_FP8_G128.spec())


def _freeform_task(**overrides) -> QuantFreeformRawTask:
    return QuantFreeformRawTask(**{**_task_fields(), **overrides})


def _cached(gpu: QuantEvalGpu, loss: float, model: str = "org/base", hotkey: str = "") -> QuantSkuReference:
    return QuantSkuReference(
        task_id=TASK_ID,
        hotkey=hotkey,
        sku=gpu.reference_key,
        model=model,
        loss=loss,
        per_example_losses=[loss, loss],
        eval_set_fingerprint="fp",
        eval_gpu=gpu,
    )


@pytest.fixture
def references(monkeypatch):
    """The per-SKU reference cache: pre-fill .cached; .stored records add_quant_sku_reference (first writer wins)."""
    state = SimpleNamespace(cached={}, stored=[], alerts=[])

    async def add(reference, _db):
        state.stored.append(reference)
        return state.cached.setdefault(reference.key, reference)

    async def alert(_config, **kwargs):
        state.alerts.append(kwargs)

    monkeypatch.setattr(scoring_tasks, "get_quant_sku_references", AsyncMock(side_effect=lambda *_: dict(state.cached)))
    monkeypatch.setattr(scoring_tasks, "add_quant_sku_reference", add)
    monkeypatch.setattr(scoring_tasks, "notify_evaluation_exception", alert)
    monkeypatch.setattr(
        scoring_tasks,
        "quant_reference_model",
        AsyncMock(side_effect=lambda task, _hotkey, _db: task.augmented_model_id or task.model_id),
    )
    return state


class _Calls(list):
    """The captured dispatch kwargs, carrying the fake's knobs as .state."""


@pytest.fixture
def gpu_runner(monkeypatch, references):
    """Fake run_evaluation_basilica_quant; returns the captured kwargs per call. .measured sets what it returns
    as each repo's same-deployment reference."""
    calls = _Calls()

    async def fake_run(**kwargs):
        calls.append(kwargs)
        reference = getattr(gpu_runner_state, "measured", None)
        results = {
            repo: EvaluationResultText(is_finetune=True, eval_loss=1.5, quant_reference=reference) for repo in kwargs["models"]
        }
        return DockerEvaluationResults(results=results, base_model_params_count=1234)

    gpu_runner_state = SimpleNamespace()
    monkeypatch.setattr(scoring_tasks, "run_evaluation_basilica_quant", fake_run)
    # Every fixed repo here passes the structural pre-check (test_quant_structural_precheck covers it).
    monkeypatch.setattr(scoring_tasks, "expected_tensor_set_for", AsyncMock(return_value=None))
    monkeypatch.setattr(scoring_tasks, "fixed_submission_failure", AsyncMock(return_value=None))
    calls.state = gpu_runner_state
    return calls


class TestEvaluateSubmissions:
    async def test_fixed_task_dispatches_spec_and_format(self, gpu_runner) -> None:
        task = _fixed_task()

        results = await scoring_tasks._evaluate_submissions(task, ["org/miner"], num_gpus=1, config=CONFIG, repo_to_hotkey=MINER)

        (call,) = gpu_runner
        assert call["reference_models"] == {"org/miner": "org/base"}
        assert call["cached_references"] == {"org/miner": {}}
        assert call["quant_spec"] == task.quant_spec
        assert call["quant_format"] == QuantFormat.W4A8_FP8_G128
        assert call["dataset_type"] is task.dataset_type
        assert call["original_model"] == task.model_id
        assert call["psql_db"] == "db"
        assert results == {"org/miner": EvaluationResultText(is_finetune=True, eval_loss=1.5)}
        assert task.model_params_count == 1234

    async def test_original_model_prefers_augmented_model(self, gpu_runner) -> None:
        task = _fixed_task()
        task.augmented_model_id = "org/augmented"

        await scoring_tasks._evaluate_submissions(task, ["org/miner"], num_gpus=1, config=CONFIG, repo_to_hotkey=MINER)

        assert gpu_runner[0]["original_model"] == "org/augmented"
        assert gpu_runner[0]["reference_models"] == {"org/miner": "org/augmented"}

    async def test_the_calibration_set_is_never_dispatched_for_scoring(self, gpu_runner) -> None:
        calibration_url = "https://bucket/calibration.jsonl"
        task = _freeform_task(calibration_data=calibration_url, calibration_fingerprint="fp-cal", calibration_n_examples=250)
        urls = {"org/a": {"index.json": "https://s3/i"}}

        await scoring_tasks._evaluate_submissions(
            task, ["org/a"], num_gpus=1, config=CONFIG, decoded_model_urls=urls, repo_to_hotkey=AB
        )

        (call,) = gpu_runner
        assert call["dataset"] == task.test_data
        assert calibration_url not in {value for value in call.values() if isinstance(value, str)}

    async def test_freeform_without_decoded_models_never_reaches_gpu(self, gpu_runner) -> None:
        results = await scoring_tasks._evaluate_submissions(
            _freeform_task(), ["org/a", "org/b"], num_gpus=1, config=CONFIG, decoded_model_urls={}, repo_to_hotkey=AB
        )

        assert gpu_runner == []
        assert set(results) == {"org/a", "org/b"}
        assert all(isinstance(result, Exception) for result in results.values())

    async def test_freeform_sends_only_decoded_repos(self, gpu_runner) -> None:
        urls = {"org/a": {"index.json": "https://s3/i"}}

        results = await scoring_tasks._evaluate_submissions(
            _freeform_task(), ["org/a", "org/b"], num_gpus=1, config=CONFIG, decoded_model_urls=urls, repo_to_hotkey=AB
        )

        (call,) = gpu_runner
        assert call["models"] == ["org/a"]
        assert call["quant_spec"] is None and call["quant_format"] is None
        assert call["decoded_model_urls"] == urls
        assert results["org/a"].eval_loss == 1.5
        assert isinstance(results["org/b"], Exception)


SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132, sku="H100 SXM")


def _measured(gpu: QuantEvalGpu, loss: float, model: str = "org/base") -> QuantReferenceScore:
    return QuantReferenceScore(
        model=model, eval_loss=loss, per_example_losses=[loss, loss], eval_set_fingerprint="fp", eval_gpu=gpu
    )


class TestSkuReferenceCache:
    async def test_cached_skus_reach_the_dispatch(self, gpu_runner, references) -> None:
        references.cached[("", SXM.reference_key)] = _cached(SXM, 1.4)

        await scoring_tasks._evaluate_submissions(_fixed_task(), ["org/miner"], num_gpus=1, config=CONFIG, repo_to_hotkey=MINER)

        assert gpu_runner[0]["cached_references"] == {"org/miner": {"H100 SXM": 1.4}}

    async def test_a_same_deployment_reference_is_cached(self, gpu_runner, references) -> None:
        gpu_runner.state.measured = _measured(PCIE, 1.41)

        await scoring_tasks._evaluate_submissions(_fixed_task(), ["org/miner"], num_gpus=1, config=CONFIG, repo_to_hotkey=MINER)

        assert references.cached[("", "H100 PCIe")].loss == 1.41
        assert references.cached[("", "H100 PCIe")].model == "org/base"
        assert references.alerts == []

    async def test_first_writer_wins_and_an_identical_remeasurement_is_quiet(self, gpu_runner, references) -> None:
        references.cached[("", PCIE.reference_key)] = _cached(PCIE, 1.41)
        gpu_runner.state.measured = _measured(PCIE, 1.41)

        await scoring_tasks._evaluate_submissions(_fixed_task(), ["org/miner"], num_gpus=1, config=CONFIG, repo_to_hotkey=MINER)

        assert len(references.stored) == 1
        assert references.alerts == []

    async def test_a_mismatched_remeasurement_keeps_the_cached_row_and_alerts(self, gpu_runner, references) -> None:
        references.cached[("", PCIE.reference_key)] = _cached(PCIE, 1.41)
        gpu_runner.state.measured = _measured(PCIE, 1.4100001)

        await scoring_tasks._evaluate_submissions(_fixed_task(), ["org/miner"], num_gpus=1, config=CONFIG, repo_to_hotkey=MINER)

        assert references.cached[("", "H100 PCIe")].loss == 1.41
        assert [alert["context"] for alert in references.alerts] == ["Quant reference re-measurement mismatch"]

    async def test_no_reference_model_is_an_error_and_never_dispatched(self, gpu_runner, references, monkeypatch) -> None:
        monkeypatch.setattr(scoring_tasks, "quant_reference_model", AsyncMock(return_value=None))

        results = await scoring_tasks._evaluate_submissions(
            _fixed_task(), ["org/miner"], num_gpus=1, config=CONFIG, repo_to_hotkey=MINER
        )

        # A task-common input with no reference model is our problem: an evaluation error, never the miner's scored fail.
        assert gpu_runner == []
        assert isinstance(results["org/miner"], EvaluationInfraError)
        assert "no bf16 reference model" in str(results["org/miner"])


class TestOwnModelReferences:
    """OWN_R1_MODEL: each repo gets its own miner's reference model, cache and storage key."""

    @pytest.fixture
    def own_models(self, monkeypatch, references):
        models = {"a": "org/a-r1", "b": "org/b-r1"}
        monkeypatch.setattr(
            scoring_tasks, "quant_reference_model", AsyncMock(side_effect=lambda _task, hotkey, _db: models.get(hotkey))
        )
        return models

    async def test_each_repo_gets_its_own_reference_model_and_cached_losses(self, gpu_runner, references, own_models) -> None:
        references.cached[("a", SXM.reference_key)] = _cached(SXM, 1.4, "org/a-r1", hotkey="a")
        references.cached[("b", SXM.reference_key)] = _cached(SXM, 1.9, "org/b-r1", hotkey="b")
        references.cached[("", PCIE.reference_key)] = _cached(PCIE, 7.0)  # a task-common row is not an own-model one
        task = _freeform_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL)

        await scoring_tasks._evaluate_submissions(
            task, ["org/a", "org/b"], num_gpus=1, config=CONFIG, decoded_model_urls={"org/a": {}, "org/b": {}}, repo_to_hotkey=AB
        )

        (call,) = gpu_runner
        assert call["reference_models"] == {"org/a": "org/a-r1", "org/b": "org/b-r1"}
        assert call["cached_references"] == {"org/a": {"H100 SXM": 1.4}, "org/b": {"H100 SXM": 1.9}}

    async def test_a_miner_without_a_reference_model_is_its_scored_fail_the_others_still_run(
        self, gpu_runner, own_models
    ) -> None:
        task = _freeform_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL)
        repo_to_hotkey = {**AB, "org/c": "c"}
        urls = {repo: {} for repo in repo_to_hotkey}

        results = await scoring_tasks._evaluate_submissions(
            task, list(repo_to_hotkey), num_gpus=1, config=CONFIG, decoded_model_urls=urls, repo_to_hotkey=repo_to_hotkey
        )

        # An own-model miner with no starting model is unscoreable on its own account, as finalize records it.
        assert sorted(gpu_runner[0]["models"]) == ["org/a", "org/b"]
        assert results["org/c"].quant_failure == NO_REFERENCE_MODEL_FAILURE

    async def test_a_measured_reference_is_stored_under_its_miners_hotkey(self, gpu_runner, references, own_models) -> None:
        gpu_runner.state.measured = _measured(PCIE, 1.41, "org/a-r1")
        task = _freeform_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL)

        await scoring_tasks._evaluate_submissions(
            task, ["org/a"], num_gpus=1, config=CONFIG, decoded_model_urls={"org/a": {}}, repo_to_hotkey=AB
        )

        assert list(references.cached) == [("a", "H100 PCIe")]
        assert references.stored[0].hotkey == "a"

    async def test_a_task_common_reference_is_stored_under_the_empty_hotkey(self, gpu_runner, references) -> None:
        gpu_runner.state.measured = _measured(PCIE, 1.41)

        await scoring_tasks._evaluate_submissions(_fixed_task(), ["org/miner"], num_gpus=1, config=CONFIG, repo_to_hotkey=MINER)

        assert list(references.cached) == [("", "H100 PCIe")]


class TestProcessMinersPool:
    @pytest.fixture
    def pool(self, monkeypatch):
        """Freeform pool: A is evaluated, B has a persisted decode-job scored fail."""
        persisted = [
            {db_cst.HOTKEY: "A", db_cst.DECODER_BYTES: 4096, db_cst.QUANT_FAILURE: None},
            {db_cst.HOTKEY: "B", db_cst.ARTIFACT_BYTES: 77, db_cst.DECODER_BYTES: 9, db_cst.QUANT_FAILURE: "decode timeout"},
        ]
        names = {"A": "repo-a", "B": "repo-b"}
        repo_info_calls: list[str] = []
        evaluated: list[list[str]] = []

        class FakeHfApi:
            def __init__(self, token=None):
                pass

            def repo_info(self, repo, timeout):
                repo_info_calls.append(repo)

        async def fake_evaluate(task, submission_repos, num_gpus, dataset_type, config, decoded_model_urls, repo_to_hotkey):
            evaluated.append(submission_repos)
            assert repo_to_hotkey == {REPO_A: "A"}
            assert decoded_model_urls == {REPO_A: {"index.json": "https://s3/i"}}
            return {REPO_A: EvaluationResultText(is_finetune=True, eval_loss=1.9, eval_gpu=PCIE)}

        monkeypatch.setattr(scoring_tasks, "get_task_node_losses", AsyncMock(return_value=persisted))
        monkeypatch.setattr(scoring_tasks, "get_expected_repo_name", AsyncMock(side_effect=lambda _t, hk, _db: names[hk]))
        monkeypatch.setattr(scoring_tasks, "HfApi", FakeHfApi)
        monkeypatch.setattr(
            scoring_tasks,
            "decoded_model_urls_for",
            AsyncMock(side_effect=lambda _t, repos, bucket: {repo: {"index.json": "https://s3/i"} for repo in repos.values()}),
        )
        monkeypatch.setattr(scoring_tasks, "_evaluate_submissions", fake_evaluate)
        listing_calls: list[str] = []

        def fake_listing(repo, token=None):
            listing_calls.append(repo)
            return HubArtifactListing(bytes=123_456, files=[HUB_FILE], revision="rev-1")

        def no_second_call(repo, token=None):
            raise AssertionError("freeform scoring must use the single hub listing")

        monkeypatch.setattr(scoring_tasks, "artifact_listing_from_hub", fake_listing)
        monkeypatch.setattr(scoring_tasks, "artifact_bytes_from_hub", no_second_call)
        return SimpleNamespace(repo_info_calls=repo_info_calls, evaluated=evaluated, listing_calls=listing_calls)

    async def test_persisted_scored_fail_skips_hub_and_gpu(self, pool) -> None:
        miners = [SimpleNamespace(hotkey="A"), SimpleNamespace(hotkey="B")]

        results = {r.hotkey: r for r in await scoring_tasks.process_miners_pool(miners, _freeform_task(), CONFIG, 1)}

        assert pool.repo_info_calls == [REPO_A]
        assert pool.evaluated == [[REPO_A]]
        b = results["B"]
        assert (b.is_finetune, b.quant_failure, b.artifact_bytes, b.decoder_bytes) == (True, "decode timeout", 77, 9)
        assert math.isnan(b.test_loss)
        assert b.submission.repo == REPO_B

    async def test_evaluated_result_carries_hub_and_decode_job_bytes(self, pool) -> None:
        miners = [SimpleNamespace(hotkey="A"), SimpleNamespace(hotkey="B")]

        results = {r.hotkey: r for r in await scoring_tasks.process_miners_pool(miners, _freeform_task(), CONFIG, 1)}

        a = results["A"]
        assert a.test_loss == 1.9
        assert a.artifact_bytes == 123_456
        assert a.decoder_bytes == 4096
        assert a.quant_failure is None
        assert a.eval_gpu == PCIE

    async def test_one_hub_listing_per_evaluated_miner_feeds_files_and_revision(self, pool) -> None:
        miners = [SimpleNamespace(hotkey="A"), SimpleNamespace(hotkey="B")]

        results = {r.hotkey: r for r in await scoring_tasks.process_miners_pool(miners, _freeform_task(), CONFIG, 1)}

        assert pool.listing_calls == [REPO_A]
        assert results["A"].artifact_files == [HUB_FILE]
        assert results["A"].artifact_revision == "rev-1"
        assert results["B"].artifact_files is None

    async def test_failed_listing_falls_back_to_the_bytes_call(self, pool, monkeypatch) -> None:
        def broken(repo, token=None):
            raise RuntimeError("hub down")

        monkeypatch.setattr(scoring_tasks, "artifact_listing_from_hub", broken)
        monkeypatch.setattr(scoring_tasks, "artifact_bytes_from_hub", lambda repo, token=None: 99)

        results = {
            r.hotkey: r
            for r in await scoring_tasks.process_miners_pool([SimpleNamespace(hotkey="A")], _freeform_task(), CONFIG, 1)
        }

        assert results["A"].artifact_bytes == 99
        assert results["A"].artifact_files is None


def _result(hotkey, test_loss=np.nan, is_finetune=True, quant_failure=None) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=test_loss,
        synth_loss=test_loss,
        is_finetune=is_finetune,
        task_type=TaskType.QUANTFREEFORMTASK,
        quant_failure=quant_failure,
        reference_loss=2.0,
    )


async def test_scored_fail_is_evaluated_not_failed(monkeypatch) -> None:
    pool_results = [_result("ok", 1.5), _result("scored", quant_failure="does not load"), _result("error", is_finetune=False)]
    nodes = [SimpleNamespace(hotkey=r.hotkey) for r in pool_results]
    monkeypatch.setattr(scoring_tasks, "get_nodes_assigned_to_task", AsyncMock(return_value=nodes))
    monkeypatch.setattr(scoring_tasks, "process_miners_pool", AsyncMock(return_value=pool_results))
    monkeypatch.setattr(scoring_tasks, "_persist_raw_task_results", AsyncMock())
    monkeypatch.setattr(scoring_tasks, "_undecoded_freeform_hotkeys", AsyncMock(return_value=[]))

    outcome = await scoring_tasks.evaluate_and_score_hotkeys(_freeform_task(), ["ok", "scored", "error"], 1, CONFIG)

    assert outcome.evaluated == ["ok", "scored", "error"]
    assert outcome.failed == ["error"]


def _decode_row(hotkey, decoder_sha256=None, quant_failure=None) -> QuantDecodeRow:
    return QuantDecodeRow(
        hotkey=hotkey, artifact_bytes=None, decoder_bytes=None, decoder_sha256=decoder_sha256, quant_failure=quant_failure
    )


@pytest.fixture
def decode_gate(monkeypatch):
    """Stub the decode rows and the decoded-index lookup; .pool records what process_miners_pool was handed."""
    state = SimpleNamespace(rows={}, indexed=set(), pool=[], tournament_id="tourn")

    async def decoded_urls(task_id, hotkey_to_repo, bucket):
        return {repo: {"index.json": "https://s3/i"} for repo in hotkey_to_repo.values() if repo in state.indexed}

    async def pool(miners, task, config, num_gpus, dataset_type):
        state.pool.append([m.hotkey for m in miners])
        return [_result(m.hotkey, 1.5) for m in miners]

    monkeypatch.setattr(
        scoring_tasks.tournament_sql, "get_tournament_id_by_task_id", AsyncMock(side_effect=lambda *_: state.tournament_id)
    )
    monkeypatch.setattr(scoring_tasks, "get_quant_decode_rows", AsyncMock(side_effect=lambda *_: state.rows))
    monkeypatch.setattr(scoring_tasks, "decoded_model_urls_for", decoded_urls)
    monkeypatch.setattr(
        scoring_tasks,
        "get_nodes_assigned_to_task",
        AsyncMock(side_effect=lambda *_: [SimpleNamespace(hotkey=h) for h in ("ready", "undecoded", "no_index", "scored")]),
    )
    monkeypatch.setattr(scoring_tasks, "process_miners_pool", pool)
    monkeypatch.setattr(scoring_tasks, "_persist_raw_task_results", AsyncMock())
    return state


async def test_hotkeys_without_a_settled_decode_or_decoded_index_stay_pending(decode_gate) -> None:
    decode_gate.rows = {
        "ready": _decode_row("ready", decoder_sha256="sha"),
        "no_index": _decode_row("no_index", decoder_sha256="sha"),
        "scored": _decode_row("scored", quant_failure="does not load"),
    }
    decode_gate.indexed = {"ready"}

    outcome = await scoring_tasks.evaluate_and_score_hotkeys(
        _freeform_task(), ["ready", "undecoded", "no_index", "scored"], 1, CONFIG
    )

    assert decode_gate.pool == [["ready", "scored"]]
    assert outcome.evaluated == ["ready", "scored"]
    assert outcome.failed == []
    assert sorted(outcome.deferred) == ["no_index", "undecoded"]


async def test_all_hotkeys_undecoded_never_reaches_the_evaluator(decode_gate) -> None:
    outcome = await scoring_tasks.evaluate_and_score_hotkeys(_freeform_task(), ["undecoded"], 1, CONFIG)

    assert decode_gate.pool == []
    assert outcome.evaluated == []
    assert outcome.deferred == ["undecoded"]


async def test_organic_freeform_hotkeys_are_not_gated_on_a_decode(decode_gate) -> None:
    decode_gate.tournament_id = None

    outcome = await scoring_tasks.evaluate_and_score_hotkeys(_freeform_task(), ["undecoded"], 1, CONFIG)

    assert decode_gate.pool == [["undecoded"]]
    assert outcome.deferred == []


async def test_fixed_task_hotkeys_are_not_gated_on_a_decode(decode_gate) -> None:
    outcome = await scoring_tasks.evaluate_and_score_hotkeys(_fixed_task(), ["undecoded"], 1, CONFIG)

    assert decode_gate.pool == [["undecoded"]]
    assert outcome.deferred == []


def test_dataset_type_for_quant_task_is_the_tasks_own() -> None:
    task = _fixed_task()
    assert scoring_tasks._get_dataset_type(task) is task.dataset_type


class TestScoredFailsWithoutValidResults:
    def test_scored_fails_keep_zero_score_and_no_rank(self) -> None:
        scored_fail = _result("scored", quant_failure="does not load")
        results = scoring_tasks.calculate_miner_ranking_and_scores([scored_fail], policy=QuantFreeformPenalty.cliff(1.05))

        assert results[0].score == 0.0
        assert results[0].rank is None

    def test_pass_missing_bytes_records_true_bar_verdict(self) -> None:
        passing = _result("pass", 2.0)
        results = scoring_tasks.calculate_miner_ranking_and_scores(
            [passing, _result("other", 2.05)], policy=QuantFreeformPenalty.cliff(1.05)
        )

        assert passing.passed_bar is True
        assert all(r.score_reason.startswith("Scored fail: ") for r in results)
