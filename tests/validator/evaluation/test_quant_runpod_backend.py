from types import SimpleNamespace

import httpx
import pytest

from validator.evaluation import basilica
from validator.evaluation import remote
from validator.evaluation import runpod
from validator.evaluation.basilica import EvaluationCapacityUnavailable


def test_service_plan_requests_the_given_gpu_models():
    payload = runpod._build_service_plan(
        run_name="god-eval-abc",
        image="eval:test",
        source="print('runner')",
        env={},
        gpu_count=1,
        gpu_models=["H100"],
    )

    assert payload["plan"]["run_spec"]["configuration"]["resources"]["gpu"]["name"] == ["H100"]


def test_service_plan_pins_instance_types_when_given():
    payload = runpod._build_service_plan(
        run_name="god-eval-abc",
        image="eval:test",
        source="print('runner')",
        env={},
        gpu_count=1,
        gpu_models=["H100"],
        instance_types=["NVIDIA H100 80GB HBM3"],
    )

    assert payload["plan"]["run_spec"]["configuration"]["instance_types"] == ["NVIDIA H100 80GB HBM3"]


@pytest.mark.asyncio
async def test_run_runpod_eval_repos_passes_gpu_models_through(monkeypatch):
    captured = []

    async def fake_run_single(**kwargs):
        captured.append(kwargs["gpu_models"])
        return {"eval_loss": 1.0}

    monkeypatch.setattr(runpod, "_run_single_runpod_eval_repo", fake_run_single)

    results = await runpod.run_runpod_eval_repos(
        repos=["org/a", "org/b"],
        model_name="org/base",
        task_type="InstructTextDatasetType",
        image="eval:test",
        source="print('runner')",
        build_env_for_repo=lambda repo: {"MODELS": repo},
        gpu_count=1,
        gpu_models=["H100"],
        task_id=None,
        psql_db=None,
        repo_to_hotkey={},
        persist_deployment_ids=False,
    )

    assert captured == [["H100"], ["H100"]]
    assert results == {"org/a": {"eval_loss": 1.0}, "org/b": {"eval_loss": 1.0}}


@pytest.mark.asyncio
@pytest.mark.parametrize("instance_types", [None, ["NVIDIA H100 80GB HBM3"]], ids=["unpinned", "pinned"])
async def test_run_runpod_eval_repos_passes_instance_types_through(monkeypatch, instance_types):
    captured = []

    async def fake_run_single(**kwargs):
        captured.append(kwargs["instance_types"])
        return {"eval_loss": 1.0}

    monkeypatch.setattr(runpod, "_run_single_runpod_eval_repo", fake_run_single)
    pin = {} if instance_types is None else {"runpod_instance_types": instance_types}

    await runpod.run_runpod_eval_repos(
        repos=["org/a"],
        model_name="org/base",
        task_type="InstructTextDatasetType",
        image="eval:test",
        source="print('runner')",
        build_env_for_repo=lambda repo: {"MODELS": repo},
        gpu_count=1,
        gpu_models=["H100"],
        task_id=None,
        psql_db=None,
        repo_to_hotkey={},
        persist_deployment_ids=False,
        **pin,
    )

    assert captured == [instance_types]


@pytest.mark.asyncio
async def test_basilica_backend_drops_runpod_instance_types(monkeypatch):
    captured = {}

    async def fake_run_basilica_eval_repos(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.delenv("EVAL_BACKEND", raising=False)
    monkeypatch.setattr(basilica, "run_basilica_eval_repos", fake_run_basilica_eval_repos)

    await remote.run_remote_eval_repos(repos=["org/a"], runpod_instance_types=["NVIDIA H100 80GB HBM3"])

    assert captured == {"repos": ["org/a"]}


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://dstack/api/project/main/runs/get")
    return httpx.HTTPStatusError("err", request=request, response=httpx.Response(status_code, request=request))


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ReadTimeout(""), True),
        (httpx.ConnectError(""), True),
        (httpx.RemoteProtocolError(""), True),
        (TimeoutError(), True),
        (_http_status_error(502), True),
        (_http_status_error(429), True),
        (_http_status_error(400), False),
        (_http_status_error(404), False),
        (RuntimeError("Runpod deployment ended in failed"), False),
    ],
)
def test_transient_dstack_error_classification(exc, expected):
    assert runpod._is_transient_dstack_error(exc) is expected


def _waiting_deployment(run: dict) -> runpod.RunpodDeployment:
    return runpod.RunpodDeployment(
        run_name="god-eval-abc",
        deployment_id="runpod:god-eval-abc",
        url=None,
        created_at=runpod.datetime.datetime.now(runpod.datetime.timezone.utc),
        control_token="token",
        run=run,
    )


def _quiet_ctx() -> SimpleNamespace:
    noop = lambda *_args, **_kwargs: None  # noqa: E731
    return SimpleNamespace(
        repo="org/model", eval_logger=SimpleNamespace(info=noop, warning=noop), requeue_infra_errors=True
    )


@pytest.mark.asyncio
async def test_wait_ready_keeps_polling_through_transient_dstack_errors(monkeypatch):
    # A run still waiting for capacity: dstack times out twice, then reports no offers. The timeouts must not
    # end the wait; only the run's real state may.
    calls = {"n": 0}
    no_offers = {"status": "submitted", "latest_job_submission": {"status_message": "no offers"}}

    class Client:
        async def get_run(self, _run_name):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise httpx.ReadTimeout("")
            return no_offers

        def resolve_service_url(self, _run):
            return None

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(runpod.asyncio, "sleep", no_sleep)

    with pytest.raises(EvaluationCapacityUnavailable):
        await runpod._wait_ready(Client(), _waiting_deployment({"status": "submitted"}), _quiet_ctx())
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_wait_ready_still_raises_non_transient_dstack_errors(monkeypatch):
    class Client:
        async def get_run(self, _run_name):
            raise _http_status_error(400)

        def resolve_service_url(self, _run):
            return None

    with pytest.raises(httpx.HTTPStatusError):
        await runpod._wait_ready(Client(), _waiting_deployment({"status": "submitted"}), _quiet_ctx())
