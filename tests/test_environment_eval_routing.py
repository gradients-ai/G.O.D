from uuid import uuid4

import pytest

from core.constants import VALIDATOR_DOCKER_IMAGE_ENV
from core.constants import EnvironmentName
from core.models.payload_models import DockerEvaluationResults
from core.models.payload_models import EvaluationResultText
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import FileFormat
from validator.evaluation import docker_evaluation
from validator.evaluation import local_evaluation


@pytest.mark.asyncio
async def test_basilica_2048_routes_to_individual_env_image(monkeypatch):
    captured = {}

    async def fake_db_read_with_retry(coro_factory, op_name):
        return {}, {}

    def fake_create_runner_source(command, result_path):
        captured["command"] = command
        captured["result_path"] = result_path
        return "runner-source"

    async def fake_run_basilica_eval_repos(**kwargs):
        captured["image"] = kwargs["image"]
        captured["env"] = kwargs["build_env_for_repo"]("org/repo")
        return {"org/repo": {"org/repo": {"is_finetune": True, "eval_loss": 2048.0}}}

    monkeypatch.setattr(docker_evaluation, "_db_read_with_retry", fake_db_read_with_retry)
    monkeypatch.setattr(docker_evaluation, "create_basilica_eval_runner_source", fake_create_runner_source)
    monkeypatch.setattr(docker_evaluation, "run_basilica_eval_repos", fake_run_basilica_eval_repos)

    result = await docker_evaluation.run_evaluation_basilica_text(
        dataset="environment-task",
        models=["org/repo"],
        original_model="Qwen/Qwen2.5-7B-Instruct",
        dataset_type=EnvironmentDatasetType(environment_names=[EnvironmentName.TWENTY_FORTY_EIGHT]),
        file_format=FileFormat.JSON,
        num_gpus=1,
        task_id=uuid4(),
        psql_db=object(),
    )

    assert captured["image"] == VALIDATOR_DOCKER_IMAGE_ENV
    assert captured["command"] == ["python", "-m", "validator.evaluation.individual"]
    assert captured["env"]["ENVIRONMENT_NAME"] == EnvironmentName.TWENTY_FORTY_EIGHT.value
    assert captured["env"]["MODELS"] == "org/repo"
    assert "ENV_SERVER_CMD" not in captured["env"]
    assert result.results["org/repo"].eval_loss == 2048.0


@pytest.mark.asyncio
async def test_basilica_direct_pvp_environment_rejected(monkeypatch):
    async def fake_db_read_with_retry(coro_factory, op_name):
        return {}, {}

    monkeypatch.setattr(docker_evaluation, "_db_read_with_retry", fake_db_read_with_retry)

    with pytest.raises(ValueError, match="does not support PvP env"):
        await docker_evaluation.run_evaluation_basilica_text(
            dataset="environment-task",
            models=["org/repo"],
            original_model="Qwen/Qwen2.5-7B-Instruct",
            dataset_type=EnvironmentDatasetType(environment_names=[EnvironmentName.LIARS_DICE]),
            file_format=FileFormat.JSON,
            num_gpus=1,
            task_id=uuid4(),
            psql_db=object(),
        )


@pytest.mark.asyncio
async def test_local_2048_routes_to_individual_open_spiel(monkeypatch):
    captured = {}

    async def fake_individual_eval(models, original_model, dataset_type, gpu_id=0, eval_seed=None):
        captured.update(
            {
                "models": models,
                "original_model": original_model,
                "dataset_type": dataset_type,
                "gpu_id": gpu_id,
                "eval_seed": eval_seed,
            }
        )
        return DockerEvaluationResults(
            results={"org/repo": EvaluationResultText(is_finetune=True, eval_loss=512.0)}
        )

    monkeypatch.setattr(
        local_evaluation,
        "run_evaluation_local_individual_open_spiel",
        fake_individual_eval,
    )

    result = await local_evaluation.run_evaluation_docker_text(
        dataset="environment-task",
        models=["org/repo"],
        original_model="Qwen/Qwen2.5-7B-Instruct",
        dataset_type=EnvironmentDatasetType(environment_names=[EnvironmentName.TWENTY_FORTY_EIGHT]),
        file_format=FileFormat.JSON,
        gpu_ids=[3],
        eval_seed=123,
    )

    assert captured["models"] == ["org/repo"]
    assert captured["dataset_type"].environment_names == [EnvironmentName.TWENTY_FORTY_EIGHT]
    assert captured["gpu_id"] == 3
    assert captured["eval_seed"] == 123
    assert result.results["org/repo"].eval_loss == 512.0


@pytest.mark.asyncio
async def test_local_direct_pvp_environment_rejected():
    with pytest.raises(ValueError, match="does not support PvP env"):
        await local_evaluation.run_evaluation_docker_text(
            dataset="environment-task",
            models=["org/repo"],
            original_model="Qwen/Qwen2.5-7B-Instruct",
            dataset_type=EnvironmentDatasetType(environment_names=[EnvironmentName.LIARS_DICE]),
            file_format=FileFormat.JSON,
            gpu_ids=[0],
        )
