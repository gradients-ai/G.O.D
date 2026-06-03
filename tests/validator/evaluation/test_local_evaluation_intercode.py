import pytest

from core.constants.environments import EnvironmentName
from core.models.dataset_models import EnvironmentDatasetType
from core.models.dataset_models import FileFormat
from validator.evaluation import local_evaluation


@pytest.mark.asyncio
async def test_run_evaluation_docker_text_routes_intercode_to_local_intercode(monkeypatch):
    sentinel = object()
    captured = {}

    async def fake_intercode_runner(
        models,
        original_model,
        dataset_type,
        file_format,
        gpu_id=0,
        eval_seed=None,
    ):
        captured.update(
            {
                "models": models,
                "original_model": original_model,
                "dataset_type": dataset_type,
                "file_format": file_format,
                "gpu_id": gpu_id,
                "eval_seed": eval_seed,
            }
        )
        return sentinel

    async def fake_environment_runner(*args, **kwargs):
        raise AssertionError("InterCode should not use the generic local environment runner")

    monkeypatch.setattr(local_evaluation, "run_evaluation_local_intercode", fake_intercode_runner)
    monkeypatch.setattr(local_evaluation, "run_evaluation_local_environment", fake_environment_runner)

    dataset_type = EnvironmentDatasetType(environment_names=[EnvironmentName.INTERCODE])
    result = await local_evaluation.run_evaluation_docker_text(
        dataset="/tmp/unused.json",
        models=["org/model"],
        original_model="base/model",
        dataset_type=dataset_type,
        file_format=FileFormat.JSON,
        gpu_ids=[3],
        eval_seed=42,
    )

    assert result is sentinel
    assert captured == {
        "models": ["org/model"],
        "original_model": "base/model",
        "dataset_type": dataset_type,
        "file_format": FileFormat.JSON,
        "gpu_id": 3,
        "eval_seed": 42,
    }
