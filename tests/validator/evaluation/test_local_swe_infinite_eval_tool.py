import os

import pytest

from ops.tools.evaluation import local_swe_infinite_eval
from validator.evaluation.evaluators import swe


@pytest.mark.asyncio
async def test_resolve_eval_selection_is_deterministic_for_seed(monkeypatch):
    args = local_swe_infinite_eval.parse_args(
        [
            "--seed",
            "123",
            "--num-seeds",
            "6",
            "--task-id-min",
            "1",
            "--task-id-max",
            "2000",
        ]
    )
    eval_config = local_swe_infinite_eval.build_swe_eval_config(args)
    task_selection = local_swe_infinite_eval.build_task_selection_override(args)
    first = await local_swe_infinite_eval.resolve_eval_selection(args, eval_config, task_selection)
    second = await local_swe_infinite_eval.resolve_eval_selection(args, eval_config, task_selection)

    task_ids = [task_id for _seed, task_id in first.eval_list]
    assert first == second
    assert len(task_ids) == 6
    assert len(set(task_ids)) == 6
    assert sum(task_id in set(swe.SWE_VETTED_TASK_IDS) for task_id in task_ids) == 3


@pytest.mark.asyncio
async def test_resolve_eval_selection_preserves_explicit_task_ids(monkeypatch):
    args = local_swe_infinite_eval.parse_args(["--seed", "42", "--task-id", "7", "83", "45"])

    selection = await local_swe_infinite_eval.resolve_eval_selection(args)

    assert [task_id for _seed, task_id in selection.eval_list] == [7, 83, 45]
    assert selection.explicit_task_ids == [7, 83, 45]
    assert selection.num_seeds == 3
    assert selection.task_id_min == 7
    assert selection.task_id_max == 83


def test_resolve_model_base_url_uses_host_docker_internal_for_local_docker_swe():
    args = local_swe_infinite_eval.parse_args(["--sglang-port", "30000"])

    assert (
        local_swe_infinite_eval.resolve_model_base_url(args, swe_runs_in_docker=True)
        == "http://host.docker.internal:30000/v1"
    )


def test_resolve_model_base_url_respects_explicit_override():
    args = local_swe_infinite_eval.parse_args(["--model-base-url", "https://model.example"])

    assert local_swe_infinite_eval.resolve_model_base_url(args, swe_runs_in_docker=True) == "https://model.example/v1"


def test_build_swe_docker_command_uses_configured_image_and_host_gateway():
    args = local_swe_infinite_eval.parse_args(
        [
            "--swe-image",
            "gradientsio/swe-infinite:v1",
            "--swe-host",
            "127.0.0.1",
            "--swe-port",
            "9001",
            "--swe-container-port",
            "8000",
            "--container-model-host",
            "host.docker.internal",
        ]
    )

    command = local_swe_infinite_eval.build_swe_docker_command(args, "swe-test")

    assert command == [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        "swe-test",
        "-p",
        "127.0.0.1:9001:8000",
        "--add-host",
        "host.docker.internal:host-gateway",
        "gradientsio/swe-infinite:v1",
    ]


@pytest.mark.asyncio
async def test_dry_run_does_not_launch_services(monkeypatch, tmp_path):
    async def exploding_start_sglang(*_args, **_kwargs):
        raise AssertionError("dry run should not start SGLang")

    async def exploding_start_swe(*_args, **_kwargs):
        raise AssertionError("dry run should not start SWE server")

    monkeypatch.setattr(local_swe_infinite_eval, "_start_or_reuse_sglang", exploding_start_sglang)
    monkeypatch.setattr(local_swe_infinite_eval, "start_swe_server", exploding_start_swe)
    args = local_swe_infinite_eval.parse_args(
        [
            "--env-file",
            str(tmp_path / "missing.env"),
            "--seed",
            "777",
            "--num-seeds",
            "4",
            "--task-id-min",
            "1",
            "--task-id-max",
            "2000",
            "--dry-run",
        ]
    )

    first = await local_swe_infinite_eval.run(args)
    second = await local_swe_infinite_eval.run(args)

    assert first["evaluations"] == second["evaluations"]
    assert first["avg_score"] is None
    assert os.environ.get("SWE_INFINITE_TASK_ID_MAX") is None
