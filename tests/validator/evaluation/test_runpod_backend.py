import importlib
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from validator.evaluation import remote
from validator.evaluation import runpod
from validator.evaluation.basilica import EvaluationCapacityUnavailable
from validator.evaluation.docker_evaluation import _effective_remote_file_format
from core.models.dataset_models import FileFormat
from validator.infrastructure.dstack_client import DstackClient
from validator.infrastructure.dstack_client import DstackConfig
from validator.infrastructure.dstack_client import _decode_log_message


def test_runpod_name_and_persisted_id_are_dstack_safe():
    run_name = runpod.generate_runpod_run_name()

    assert run_name.startswith("god-eval-")
    assert len(run_name) <= 63
    assert run_name.replace("-", "").isalnum()
    assert run_name == run_name.lower()
    assert runpod.decode_runpod_deployment_id(runpod.encode_runpod_deployment_id(run_name)) == run_name


def test_service_plan_uses_all_regions_and_fixed_a100_by_default(monkeypatch):
    monkeypatch.delenv("EVAL_RUNPOD_REGIONS", raising=False)
    monkeypatch.delenv("EVAL_RUNPOD_GATEWAY", raising=False)
    monkeypatch.delenv("EVAL_RUNPOD_DISK_SIZE", raising=False)

    payload = runpod._build_service_plan(
        run_name="god-eval-abc",
        image="eval:test",
        source="print('runner')",
        env={"MODELS": "org/model"},
        gpu_count=2,
    )
    config = payload["plan"]["run_spec"]["configuration"]

    assert config["type"] == "service"
    assert config["port"] == 8000
    assert config["auth"] is False
    assert "entrypoint" not in config
    assert config["probes"] == [{"type": "http", "url": "/health"}]
    assert config["resources"]["gpu"] == {
        "name": ["A100"],
        "count": {"min": 2, "max": 2},
    }
    assert config["resources"]["disk"]["size"] == "200GB"
    assert config["max_duration"] == 7200
    assert "regions" not in config


def test_service_plan_only_limits_regions_when_configured(monkeypatch):
    monkeypatch.setenv("EVAL_RUNPOD_REGIONS", "US-CA-2, EUR-IS-1")

    payload = runpod._build_service_plan(
        run_name="god-eval-abc",
        image="eval:test",
        source="print('runner')",
        env={},
        gpu_count=1,
    )

    assert payload["plan"]["run_spec"]["configuration"]["regions"] == ["US-CA-2", "EUR-IS-1"]


def test_eval_backend_defaults_to_basilica_and_accepts_runpod(monkeypatch):
    monkeypatch.delenv("EVAL_BACKEND", raising=False)
    assert remote.get_eval_backend() == remote.EvalBackend.BASILICA

    monkeypatch.setenv("EVAL_BACKEND", "runpod")
    assert remote.get_eval_backend() == remote.EvalBackend.RUNPOD


def test_s3_transport_uses_downloaded_file_parser_format():
    assert (
        _effective_remote_file_format(
            "https://bucket.example/test_data.json?signature=abc",
            FileFormat.S3,
        )
        == FileFormat.JSON
    )
    assert _effective_remote_file_format("org/dataset", FileFormat.HF) == FileFormat.HF


def test_control_token_is_stable_for_resume(monkeypatch):
    monkeypatch.setenv("DSTACK_TOKEN", "secret")

    assert runpod._control_token("god-eval-abc") == runpod._control_token("god-eval-abc")
    assert runpod._control_token("god-eval-abc") != runpod._control_token("god-eval-def")


def test_old_dstack_base64_logs_are_decoded():
    assert _decode_log_message("W2V2YWxfcnVubmVyXSBvawo=") == "[eval_runner] ok"


def test_old_dstack_resource_not_exists_response_counts_as_not_found():
    request = httpx.Request("POST", "https://dstack.example/runs/get")
    response = httpx.Response(
        400,
        request=request,
        json={"detail": [{"msg": "Run not found", "code": "resource_not_exists"}]},
    )

    assert runpod._is_not_found(
        httpx.HTTPStatusError("bad request", request=request, response=response)
    )


def test_runner_accepts_dstack_prefixed_public_paths(monkeypatch):
    monkeypatch.setenv("EVAL_RUNNER_COMMAND", json.dumps(["python", "-c", "pass"]))
    monkeypatch.setenv("EVAL_RUNNER_RESULT_PATH", "/tmp/result.json")
    sys.modules.pop("validator.evaluation.remote_runner", None)
    runner = importlib.import_module("validator.evaluation.remote_runner")

    assert (
        runner._normalized_request_path("/proxy/services/main/god-eval-abc/health")
        == "/health"
    )
    assert (
        runner._normalized_request_path("/proxy/services/main/god-eval-abc/result")
        == "/result"
    )
    assert (
        runner._normalized_request_path(
            "/proxy/services/main/god-eval-abc/v1/chat/completions"
        )
        == "/v1/chat/completions"
    )


@pytest.mark.asyncio
async def test_no_offers_maps_to_capacity_error():
    failed_run = {
        "status": "failed",
        "latest_job_submission": {"status_message": "no offers"},
    }

    class Client:
        async def get_run(self, _run_name):
            return failed_run

        def resolve_service_url(self, _run):
            return None

    deployment = runpod.RunpodDeployment(
        run_name="god-eval-abc",
        deployment_id="runpod:god-eval-abc",
        url=None,
        created_at=runpod.datetime.datetime.now(runpod.datetime.timezone.utc),
        control_token="token",
        run=failed_run,
    )
    ctx = SimpleNamespace(
        repo="org/model",
        eval_logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )

    with pytest.raises(EvaluationCapacityUnavailable):
        await runpod._wait_ready(Client(), deployment, ctx)


@pytest.mark.asyncio
async def test_dstack_stop_and_delete_use_batch_run_names_schema(monkeypatch):
    client = DstackClient(
        DstackConfig(url="https://dstack.example", token="token", project="project"),
    )
    calls = []

    async def fake_post(url, payload):
        calls.append((url, payload))
        return SimpleNamespace()

    monkeypatch.setattr(client, "_post", fake_post)

    await client.stop_run("god-eval-abc")
    await client.delete_run("god-eval-abc")

    assert calls == [
        (
            "https://dstack.example/api/project/project/runs/stop",
            {"runs_names": ["god-eval-abc"], "abort": True},
        ),
        (
            "https://dstack.example/api/project/project/runs/delete",
            {"runs_names": ["god-eval-abc"]},
        ),
    ]
