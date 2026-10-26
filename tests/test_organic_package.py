import subprocess
import sys
import zipfile

import pytest

from core.downloads import _validate_public_url
from validator.infrastructure.organic_dstack import OrganicRunSpec
from validator.infrastructure.organic_dstack import build_run_plan
from validator.infrastructure.organic_dstack import validate_digest_pinned_image
from validator.tasks.datasets import constants as dataset_constants
from validator.tasks.datasets.preparation import unzip_to_temp_path


def test_organic_imports_do_not_require_chain_packages():
    code = """
import importlib.abc
import sys

class BlockChain(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("fiber", "bittensor_wallet", "substrateinterface")
        if any(fullname == name or fullname.startswith(f"{name}.") for name in blocked):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockChain())
import core.models.payload_models
import validator.infrastructure.dstack_client
import validator.infrastructure.organic_dstack
import validator.tasks.requests
import validator.scoring.tasks
import validator.scoring.organic
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_build_run_plan_requires_an_image_digest():
    spec = OrganicRunSpec(
        task_id="task-id",
        model="org/model",
        task_type="InstructTextTask",
        hours_to_complete=1,
        docker_image="registry.example/trainer:latest",
        gpu_name="H200",
        gpu_count=1,
        regions=("us",),
        expected_repo_name="organic_task-id",
        training_data="https://objects.example/train.json",
    )

    with pytest.raises(ValueError, match="pinned"):
        build_run_plan(spec)


def test_build_run_plan_is_explicit_and_chain_free():
    digest = "registry.example/trainer@sha256:" + ("a" * 64)
    spec = OrganicRunSpec(
        task_id="task-id",
        model="org/model",
        task_type="InstructTextTask",
        hours_to_complete=1.5,
        docker_image=digest,
        gpu_name="H200",
        gpu_count=2,
        regions=("us", "ca"),
        expected_repo_name="organic_task-id",
        training_data="https://objects.example/train.json",
        dataset_type={"field_instruction": "instruction", "field_output": "output"},
        environment={"HUGGINGFACE_TOKEN": "secret"},
    )

    plan = build_run_plan(spec, run_name="organic-run")
    configuration = plan["plan"]["run_spec"]["configuration"]

    assert validate_digest_pinned_image(digest) == digest
    assert plan["plan"]["run_spec"]["run_name"] == "organic-run"
    assert configuration["image"] == digest
    assert configuration["resources"]["gpu"]["count"] == {"min": 2, "max": 2}
    assert configuration["env"]["EXPECTED_REPO_NAME"] == "organic_task-id"
    assert configuration["max_duration"] == 9000


def test_image_archive_rejects_path_traversal(tmp_path, monkeypatch):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside.txt", "not allowed")
    monkeypatch.setattr(dataset_constants, "TEMP_PATH_FOR_IMAGES", str(tmp_path / "extract"))

    with pytest.raises(ValueError, match="escapes"):
        unzip_to_temp_path(str(archive))

    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.asyncio
async def test_download_guard_rejects_metadata_address():
    with pytest.raises(ValueError, match="non-public"):
        await _validate_public_url("http://169.254.169.254/latest/meta-data")


@pytest.mark.asyncio
async def test_evaluate_organic_repo_is_database_free(monkeypatch):
    from core.models.payload_models import EvaluationResultText
    from validator.scoring import organic

    calls = {}

    async def fake_evaluate(task, repos, num_gpus, dataset_type, config=None):
        calls.update(repos=repos, num_gpus=num_gpus, config=config)
        return {repos[0]: EvaluationResultText(is_finetune=True, eval_loss=0.25)}

    monkeypatch.setattr(organic, "_evaluate_submissions", fake_evaluate)
    monkeypatch.setattr(organic, "_get_dataset_type", lambda task: None)

    result = await organic.evaluate_organic_repo(object(), "org/organic_1", num_gpus=2)

    assert result["eval_loss"] == 0.25
    assert calls == {"repos": ["org/organic_1"], "num_gpus": 2, "config": None}


@pytest.mark.asyncio
async def test_evaluate_organic_repo_rejects_unchanged_base_model(monkeypatch):
    from core.models.payload_models import EvaluationResultText
    from validator.scoring import organic

    async def fake_evaluate(task, repos, num_gpus, dataset_type, config=None):
        return {repos[0]: EvaluationResultText(is_finetune=False, eval_loss=0.0)}

    monkeypatch.setattr(organic, "_evaluate_submissions", fake_evaluate)
    monkeypatch.setattr(organic, "_get_dataset_type", lambda task: None)

    class Task:
        model_id = "org/base"

    with pytest.raises(ValueError, match="not a finetune"):
        await organic.evaluate_organic_repo(Task(), "org/base", num_gpus=1)
