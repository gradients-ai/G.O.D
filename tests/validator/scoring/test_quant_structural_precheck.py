"""Fixed-track structural pre-check at the scoring call site, and token-authenticated hub reads."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import DockerEvaluationResults
from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantFormat
from validator.evaluation.basilica import EvaluationRetryableError
from validator.evaluation.quant_structure import StructuralCheckUnavailable
from validator.evaluation.quant_structure import TensorSet
from validator.scoring import constants as scoring_cst
from validator.scoring import tasks as scoring_tasks
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament.quant_reference import NO_REFERENCE_MODEL_FAILURE


EXPECTED = TensorSet(tensors={})
TOKEN = "hf_test_token"


def _task_fields() -> dict:
    return {
        "task_id": uuid.uuid4(),
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


def _fixed_task(**overrides) -> QuantFixedRawTask:
    return QuantFixedRawTask(**{**_task_fields(), "quant_spec": QuantFormat.W4A16_SYM_G128.spec(), **overrides})


@pytest.fixture
def harness(monkeypatch):
    """Fake GPU dispatch and structural check. verdicts: repo -> None (pass), a reason, or an exception."""
    dispatched: list[list[str]] = []
    expected_for: list[str] = []
    checked: list[str] = []
    verdicts: dict[str, str | Exception | None] = {}

    async def fake_run(**kwargs):
        dispatched.append(kwargs["models"])
        results = {repo: EvaluationResultText(is_finetune=True, eval_loss=1.5) for repo in kwargs["models"]}
        return DockerEvaluationResults(results=results, base_model_params_count=1)

    async def fake_expected(base_repo, spec):
        expected_for.append(base_repo)
        return EXPECTED

    async def fake_failure(repo, expected):
        assert expected is EXPECTED
        checked.append(repo)
        verdict = verdicts.get(repo)
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    monkeypatch.setattr(scoring_tasks, "run_evaluation_basilica_quant", fake_run)
    monkeypatch.setattr(scoring_tasks, "expected_tensor_set_for", fake_expected)
    monkeypatch.setattr(scoring_tasks, "fixed_submission_failure", fake_failure)
    return dispatched, expected_for, checked, verdicts


@pytest.fixture
def no_backoff(monkeypatch) -> list[float]:
    """Record the structural-check retry sleeps instead of waiting them out."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(scoring_tasks.asyncio, "sleep", fake_sleep)
    return sleeps


async def _evaluate(task, repos):
    return await scoring_tasks._evaluate_quant_submissions(
        task, repos, {}, num_gpus=1, config=None, decoded_model_urls=None, repo_to_hotkey={}
    )


class TestFixedPrecheck:
    async def test_mismatch_is_a_scored_fail_and_never_dispatched(self, harness) -> None:
        dispatched, expected_for, checked, verdicts = harness
        verdicts["org/bad"] = "structural: extra [x]"

        results = await _evaluate(_fixed_task(), ["org/good", "org/bad"])

        assert expected_for == ["org/base"]
        assert checked == ["org/good", "org/bad"]
        assert dispatched == [["org/good"]]
        assert results["org/bad"] == EvaluationResultText(is_finetune=True, eval_loss=None, quant_failure="structural: extra [x]")
        assert results["org/good"].eval_loss == 1.5

    async def test_expected_set_comes_from_the_augmented_input(self, harness) -> None:
        _, expected_for, _, _ = harness

        await _evaluate(_fixed_task(augmented_model_id="org/augmented"), ["org/a", "org/b"])

        assert expected_for == ["org/augmented"]

    async def test_unreadable_repo_requeues_the_evaluation_and_dispatches_nothing(self, harness, no_backoff) -> None:
        # A hub outage is never the miner's fault (an HF RemoteDisconnected must not be scored as a failure).
        dispatched, _, checked, verdicts = harness
        verdicts["org/flaky"] = StructuralCheckUnavailable("hub down")

        with pytest.raises(EvaluationRetryableError, match="org/flaky"):
            await _evaluate(_fixed_task(), ["org/good", "org/flaky"])

        assert dispatched == []
        assert checked.count("org/flaky") == scoring_cst.STRUCTURAL_CHECK_READ_ATTEMPTS

    async def test_transient_hub_error_is_retried_in_place(self, harness, no_backoff, monkeypatch) -> None:
        dispatched, _, checked, _ = harness
        calls = {"org/blip": 0}

        async def blip_once(repo, expected):
            checked.append(repo)
            if repo == "org/blip":
                calls[repo] += 1
                if calls[repo] == 1:
                    raise StructuralCheckUnavailable("connection aborted")
            return None

        monkeypatch.setattr(scoring_tasks, "fixed_submission_failure", blip_once)

        results = await _evaluate(_fixed_task(), ["org/good", "org/blip"])

        assert calls["org/blip"] == 2
        assert dispatched == [["org/good", "org/blip"]]
        assert results["org/blip"].eval_loss == 1.5
        assert len(no_backoff) == 1

    async def test_unreadable_base_requeues_the_evaluation(self, harness, no_backoff, monkeypatch) -> None:
        dispatched, _, checked, _ = harness

        async def base_down(base_repo, spec):
            raise StructuralCheckUnavailable("hub down")

        monkeypatch.setattr(scoring_tasks, "expected_tensor_set_for", base_down)

        with pytest.raises(EvaluationRetryableError, match="org/base"):
            await _evaluate(_fixed_task(), ["org/a", "org/b"])

        assert dispatched == [] and checked == []

    async def test_unsupported_base_is_an_evaluation_error_not_a_scored_fail(self, harness, monkeypatch, caplog) -> None:
        dispatched, _, checked, _ = harness

        async def bad_base(base_repo, spec):
            raise ValueError("expected one embedding with 100 rows, got []")

        monkeypatch.setattr(scoring_tasks, "expected_tensor_set_for", bad_base)

        results = await _evaluate(_fixed_task(), ["org/a", "org/b"])

        assert dispatched == [] and checked == []
        assert all(isinstance(result, Exception) and "org/base" in str(result) for result in results.values())
        assert not any(isinstance(result, EvaluationResultText) for result in results.values())

    async def test_all_failing_dispatches_nothing(self, harness) -> None:
        dispatched, _, _, verdicts = harness
        verdicts["org/a"] = "structural: missing [y]"

        results = await _evaluate(_fixed_task(), ["org/a"])

        assert dispatched == []
        assert results["org/a"].quant_failure == "structural: missing [y]"


class TestOwnModelPrecheck:
    async def test_each_repo_is_checked_against_its_own_reference_model(self, harness, monkeypatch) -> None:
        dispatched, expected_for, checked, _ = harness
        own_models = {"a": "org/a-r1", "b": "org/b-r1"}

        async def fake_reference_model(task, hotkey, psql_db):
            return own_models.get(hotkey)

        monkeypatch.setattr(scoring_tasks, "quant_reference_model", fake_reference_model)
        monkeypatch.setattr(scoring_tasks, "get_quant_sku_references", AsyncMock(return_value={}))
        task = _fixed_task(training_start_point=TrainingStartPoint.OWN_R1_MODEL)

        results = await scoring_tasks._evaluate_quant_submissions(
            task,
            ["org/a-q", "org/b-q", "org/c-q"],
            {},
            num_gpus=1,
            config=SimpleNamespace(psql_db=None),
            decoded_model_urls=None,
            repo_to_hotkey={"org/a-q": "a", "org/b-q": "b", "org/c-q": "c"},
        )

        assert sorted(expected_for) == ["org/a-r1", "org/b-r1"]
        assert checked == ["org/a-q", "org/b-q"]
        assert dispatched == [["org/a-q", "org/b-q"]]
        assert results["org/c-q"].quant_failure == NO_REFERENCE_MODEL_FAILURE


async def test_freeform_tasks_never_call_the_check(harness) -> None:
    dispatched, expected_for, checked, _ = harness
    task = QuantFreeformRawTask(**_task_fields())

    await scoring_tasks._evaluate_quant_submissions(
        task, ["org/a"], {}, num_gpus=1, config=None, decoded_model_urls={"org/a": {"index.json": "u"}}, repo_to_hotkey={}
    )

    assert expected_for == [] and checked == []
    assert dispatched == [["org/a"]]


class TestHubTokens:
    def test_artifact_bytes_pass_the_token(self, monkeypatch) -> None:
        seen = {}

        def fake_bytes(repo, token=None):
            seen["token"] = token
            return 7

        monkeypatch.setattr(scoring_tasks, "HUGGINGFACE_TOKEN", TOKEN)
        monkeypatch.setattr(scoring_tasks, "artifact_bytes_from_hub", fake_bytes)

        assert scoring_tasks._hub_artifact_bytes("org/repo") == 7
        assert seen["token"] == TOKEN

    async def test_repo_info_is_token_authenticated(self, monkeypatch) -> None:
        tokens: list[str | None] = []

        class FakeHfApi:
            def __init__(self, token=None):
                tokens.append(token)

            def repo_info(self, repo, timeout):
                raise RepositoryNotFoundError("stop here")

        async def expected_name(task_id, hotkey, psql_db):
            return "repo"

        async def no_rows(task_id, psql_db):
            return []

        monkeypatch.setattr(scoring_tasks, "HUGGINGFACE_TOKEN", TOKEN)
        monkeypatch.setattr(scoring_tasks, "HfApi", FakeHfApi)
        monkeypatch.setattr(scoring_tasks, "get_expected_repo_name", expected_name)
        monkeypatch.setattr(scoring_tasks, "get_task_node_losses", no_rows)

        task = _fixed_task()
        results = await scoring_tasks.process_miners_pool(
            [SimpleNamespace(hotkey="hk")], task, SimpleNamespace(psql_db=None), num_gpus=1
        )

        assert tokens == [TOKEN]
        assert results[0].score_reason == "Model repo not found on HuggingFace"
