"""Whose fault an evaluation failure is: the infra predicate, error descriptions, and the evaluator/backends channels."""

import http.client
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import asyncpg
import httpx
import pytest
import requests
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.errors import RepositoryNotFoundError
from minio.error import S3Error
from minio.error import ServerError

from core.models.payload_models import EvaluationResultText
from validator.evaluation import basilica
from validator.evaluation import docker_evaluation
from validator.evaluation import runpod
from validator.evaluation.errors import EvaluationCapacityUnavailable
from validator.evaluation.errors import EvaluationInfraError
from validator.evaluation.errors import EvaluationRetryableError
from validator.evaluation.errors import RepoEvaluationFailed
from validator.evaluation.errors import as_infra_error
from validator.evaluation.errors import describe_error
from validator.evaluation.errors import evaluation_error_result
from validator.evaluation.errors import is_infra_error
from validator.evaluation.result_processing import process_evaluation_results


REPO_ROOT = Path(__file__).resolve().parents[3]


def _requests_response(status: int) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    return response


def _httpx_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://dstack/api/project/main/runs/get")
    return httpx.HTTPStatusError("err", request=request, response=httpx.Response(status, request=request))


def _s3_error(status: int, code: str) -> S3Error:
    return S3Error(code, "msg", "res", "req", "host", SimpleNamespace(status=status), bucket_name="b", object_name="o")


def _chained(outer: Exception, inner: BaseException) -> Exception:
    try:
        try:
            raise inner
        except BaseException as e:
            raise outer from e
    except Exception as wrapped:
        return wrapped


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ReadTimeout(""), True),
        (httpx.ConnectError("refused"), True),
        (http.client.RemoteDisconnected("Remote end closed connection without response"), True),
        (_chained(OSError("could not load safetensors metadata"), http.client.RemoteDisconnected("closed")), True),
        (_chained(RuntimeError("wrapped"), httpx.ReadTimeout("")), True),
        (HfHubHTTPError("503 Server Error", response=_requests_response(503)), True),
        (HfHubHTTPError("429 Too Many Requests", response=_requests_response(429)), True),
        (RepositoryNotFoundError("404 Client Error", response=_requests_response(404)), False),
        (requests.ConnectionError("reset"), True),
        (_httpx_status_error(502), True),
        (_httpx_status_error(400), False),
        (_s3_error(500, "InternalError"), True),
        (_s3_error(503, "SlowDown"), True),
        (_s3_error(404, "NoSuchKey"), False),
        (ServerError("bad gateway", 502), True),
        (urllib.error.URLError("Name or service not known"), True),
        (urllib.error.HTTPError("https://s3/x", 403, "Forbidden", None, None), False),
        (asyncpg.exceptions.ConnectionDoesNotExistError("connection was closed"), True),
        (TimeoutError(), True),
        (EvaluationCapacityUnavailable("no GPUs"), True),
        (ValueError("structural mismatch"), False),
        (RuntimeError("Runpod deployment ended in failed"), False),
    ],
)
def test_infra_error_classification(exc, expected):
    assert is_infra_error(exc) is expected


def test_an_empty_message_keeps_its_type():
    assert describe_error(httpx.ReadTimeout("")) == "ReadTimeout (no message)"
    assert describe_error(ValueError("bad")) == "ValueError: bad"
    infra = as_infra_error(httpx.ReadTimeout(""), "get_run")
    assert isinstance(infra, EvaluationInfraError) and isinstance(infra, EvaluationRetryableError)
    assert str(infra) == "get_run: ReadTimeout (no message)"


def test_errors_module_imports_no_third_party_package():
    # The evaluator images import it; the image evaluator carries no core/validator stack and no DB/S3 client.
    script = (
        "import sys; import validator.evaluation.errors; "
        "print(sorted(m for m in ('httpx', 'requests', 'asyncpg', 'minio', 'urllib3', 'huggingface_hub', 'pydantic', "
        "'core', 'validator.evaluation.constants') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


class TestResultProcessing:
    def test_per_repo_error_string_is_the_repos_failure_and_error_entry_is_infra(self):
        processed = process_evaluation_results(
            {
                "org/bad": "OSError: not a valid checkpoint",
                "org/hub": evaluation_error_result(http.client.RemoteDisconnected("closed")),
                "org/ok": {"is_finetune": True, "eval_loss": 1.5},
                "org/dead": EvaluationInfraError("deployment died"),
            }
        )
        assert isinstance(processed.results["org/bad"], RepoEvaluationFailed)
        assert str(processed.results["org/bad"]) == "OSError: not a valid checkpoint"
        assert isinstance(processed.results["org/hub"], EvaluationInfraError)
        assert "RemoteDisconnected: closed" in str(processed.results["org/hub"])
        assert isinstance(processed.results["org/ok"], EvaluationResultText)
        assert isinstance(processed.results["org/dead"], EvaluationInfraError)

    def test_deployment_failure_and_missing_entry_are_infra(self):
        collected = docker_evaluation._collect_quant_repo_evaluation_results(
            ["org/failed", "org/missing", "org/raised"],
            {
                "org/failed": "Evaluation failed after 1 attempts: ",
                "org/missing": {"org/other": {"eval_loss": 1.0}, "org/another": {"eval_loss": 2.0}},
                "org/raised": httpx.ReadTimeout(""),
            },
        )
        assert all(isinstance(collected[repo], EvaluationInfraError) for repo in ("org/failed", "org/missing", "org/raised"))
        assert "ReadTimeout (no message)" in str(collected["org/raised"])


async def test_quant_evaluator_error_strings_are_evaluation_errors(monkeypatch):
    """The quant evaluator turns every submission fault into quant_failure, so a per-repo error string is ours."""

    async def fake_state(*_args):
        return {}, {}

    async def fake_remote(**kwargs):
        return {
            "org/err": {"org/err": "LocalEntryNotFoundError: cannot reach the hub"},
            "org/scored": {"org/scored": {"is_finetune": True, "eval_loss": None, "quant_failure": "structural: missing [x]"}},
        }

    monkeypatch.setattr(docker_evaluation, "load_eval_pair_state_for_models", fake_state)
    monkeypatch.setattr(docker_evaluation, "run_remote_eval_repos", fake_remote)

    processed = await docker_evaluation.run_evaluation_basilica_quant(
        dataset="https://s3/test",
        models=["org/err", "org/scored"],
        original_model="org/base",
        dataset_type=SimpleNamespace(model_dump_json=lambda: "{}"),
        file_format=docker_evaluation.FileFormat.JSON,
        num_gpus=1,
        quant_spec=None,
        quant_format=None,
        task_id=None,
        psql_db=None,
    )

    assert isinstance(processed.results["org/err"], EvaluationInfraError)
    assert "cannot reach the hub" in str(processed.results["org/err"])
    assert processed.results["org/scored"].quant_failure == "structural: missing [x]"


def _no_sleep_factory():
    async def no_sleep(_seconds):
        return None

    return no_sleep


class TestRunpodAttempts:
    @pytest.fixture
    def runpod_run(self, monkeypatch):
        monkeypatch.setenv("DSTACK_TOKEN", "secret")
        monkeypatch.setattr(runpod, "DstackClient", lambda: SimpleNamespace())
        monkeypatch.setattr(runpod.asyncio, "sleep", _no_sleep_factory())
        monkeypatch.setattr(runpod.vcst, "EVAL_BASILICA_MAX_RETRIES", 1)

        def run(error: BaseException):
            async def failing_create(**_kwargs):
                raise error

            monkeypatch.setattr(runpod, "_create_or_recover_run", failing_create)
            return runpod._run_single_runpod_eval_repo(
                repo="org/model",
                model_name="org/base",
                task_type="InstructTextDatasetType",
                image="image",
                source="source",
                env={},
                gpu_count=1,
                gpu_models=["H100"],
                instance_types=None,
                task_id=None,
                psql_db=None,
                repo_to_hotkey={},
                hotkey=None,
                existing_deployment_name=None,
                local_logging=True,
                deployment_id_persister=None,
                reserve_deployment_id=False,
                requeue_infra_errors=True,
            )

        return run

    async def test_empty_message_read_timeout_is_requeued_with_its_type(self, runpod_run):
        # dstack get_run timed out with str() == ''; the reason must still name the exception type.
        with pytest.raises(EvaluationInfraError, match="ReadTimeout"):
            await runpod_run(httpx.ReadTimeout(""))

    async def test_dstack_5xx_is_requeued(self, runpod_run):
        with pytest.raises(EvaluationInfraError, match="HTTPStatusError"):
            await runpod_run(_httpx_status_error(503))

    async def test_exhausted_attempts_are_an_infra_error_not_a_failure_string(self, runpod_run):
        with pytest.raises(EvaluationInfraError, match="after 1 attempts: RuntimeError: ended in failed"):
            await runpod_run(RuntimeError("ended in failed"))


class TestBasilicaAttempts:
    @pytest.fixture
    def basilica_run(self, monkeypatch):
        monkeypatch.setattr(basilica.asyncio, "sleep", _no_sleep_factory())

        def run():
            return basilica._run_single_basilica_eval_repo(
                repo="org/model",
                model_name="org/base",
                task_type="InstructTextDatasetType",
                image="image",
                source="source",
                env={},
                gpu_count=1,
                gpu_models=["H100"],
                min_gpu_memory_gb=None,
                task_id=None,
                psql_db=None,
                repo_to_hotkey={},
                local_logging=True,
                requeue_infra_errors=True,
            )

        return run

    async def test_exhausted_attempts_are_an_infra_error_with_the_type(self, basilica_run, monkeypatch):
        async def failing_deploy(**_kwargs):
            raise httpx.ReadTimeout("")

        monkeypatch.setattr(basilica, "_deploy_basilica_eval_repo", failing_deploy)

        with pytest.raises(EvaluationInfraError, match="ReadTimeout"):
            await basilica_run()

    async def test_a_poll_timeout_is_an_infra_error(self, basilica_run, monkeypatch):
        deployment = SimpleNamespace(name="dep", url="http://dep")

        async def deploy(**_kwargs):
            return SimpleNamespace(), deployment, "dep"

        async def timed_out(*_args, **_kwargs):
            return "Timed out waiting for result after 28800s"

        async def deleted(*_args, **_kwargs):
            return True

        monkeypatch.setattr(basilica, "_deploy_basilica_eval_repo", deploy)
        monkeypatch.setattr(basilica, "_poll_basilica_result", timed_out)
        monkeypatch.setattr(basilica, "_delete_eval_deployment", deleted)
        monkeypatch.setattr(basilica, "_fetch_attempt_logs", deleted)

        with pytest.raises(EvaluationInfraError, match="Timed out waiting for result"):
            await basilica_run()

    async def test_an_exception_escaping_the_runner_is_raised_as_infra(self, monkeypatch):
        async def broken(**_kwargs):
            raise KeyError("bug")

        monkeypatch.setattr(basilica, "_run_single_basilica_eval_repo", broken)

        with pytest.raises(EvaluationInfraError, match="KeyError"):
            await basilica.run_basilica_eval_repos(
                repos=["org/model"],
                model_name="org/base",
                task_type="x",
                image="image",
                source="source",
                build_env_for_repo=lambda _repo: {},
                gpu_count=1,
                gpu_models=["H100"],
                min_gpu_memory_gb=None,
                task_id=None,
                psql_db=None,
                repo_to_hotkey={},
                persist_deployment_ids=False,
                requeue_infra_errors=True,
            )


class TestTextEvaluationsKeepScoredFailures:
    """Without requeue_infra_errors (every non-quant evaluation) a failed deployment is the repo's failure string."""

    async def test_runpod_exhausted_attempts_return_a_failure_string(self, monkeypatch):
        monkeypatch.setenv("DSTACK_TOKEN", "secret")
        monkeypatch.setattr(runpod, "DstackClient", lambda: SimpleNamespace())
        monkeypatch.setattr(runpod.asyncio, "sleep", _no_sleep_factory())
        monkeypatch.setattr(runpod.vcst, "EVAL_BASILICA_MAX_RETRIES", 1)

        async def failing_create(**_kwargs):
            raise httpx.ReadTimeout("")

        monkeypatch.setattr(runpod, "_create_or_recover_run", failing_create)
        result = await runpod._run_single_runpod_eval_repo(
            repo="org/model",
            model_name="org/base",
            task_type="InstructTextDatasetType",
            image="image",
            source="source",
            env={},
            gpu_count=1,
            task_id=None,
            psql_db=None,
            repo_to_hotkey={},
            hotkey=None,
            existing_deployment_name=None,
            local_logging=True,
            deployment_id_persister=None,
            reserve_deployment_id=False,
        )
        assert isinstance(result, str)
        assert result.startswith("Evaluation failed after 1 attempts")

    async def test_basilica_escaping_exception_is_a_failure_string(self, monkeypatch):
        async def broken(**_kwargs):
            raise KeyError("bug")

        monkeypatch.setattr(basilica, "_run_single_basilica_eval_repo", broken)
        results = await basilica.run_basilica_eval_repos(
            repos=["org/model"],
            model_name="org/base",
            task_type="x",
            image="image",
            source="source",
            build_env_for_repo=lambda _repo: {},
            gpu_count=1,
            gpu_models=["A100"],
            min_gpu_memory_gb=None,
            task_id=None,
            psql_db=None,
            repo_to_hotkey={},
            persist_deployment_ids=False,
        )
        assert results == {"org/model": "Evaluation failed: 'bug'"}
