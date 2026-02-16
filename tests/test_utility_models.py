"""Tests for core.models.utility_models — Pydantic model validation and behavior."""

import uuid

import pytest
from pydantic import ValidationError

from core.models.utility_models import (
    DpoDatasetType,
    FileFormat,
    GPUInfo,
    GPUType,
    GrpoDatasetType,
    ImageModelType,
    ImageTextPair,
    Job,
    JobStatus,
    Message,
    MinerSubmission,
    MinerTaskResult,
    RewardFunction,
    Role,
    TaskStatus,
    TaskType,
    TextJob,
    DiffusionJob,
    TrainingStatus,
    WinningSubmission,
)


class TestFileFormatEnum:
    """Tests for FileFormat enum values."""

    def test_all_formats(self):
        assert FileFormat.CSV.value == "csv"
        assert FileFormat.JSON.value == "json"
        assert FileFormat.HF.value == "hf"
        assert FileFormat.S3.value == "s3"


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self):
        assert JobStatus.QUEUED.value == "Queued"
        assert JobStatus.RUNNING.value == "Running"
        assert JobStatus.COMPLETED.value == "Completed"
        assert JobStatus.FAILED.value == "Failed"
        assert JobStatus.NOT_FOUND.value == "Not Found"


class TestTaskStatus:
    """Tests for TaskStatus enum completeness."""

    def test_has_all_expected_statuses(self):
        expected = {
            "pending", "preparing_data", "prep_task_failure",
            "looking_for_nodes", "failure_finding_nodes", "delayed",
            "ready", "training", "preevaluation", "evaluating",
            "success", "failure",
        }
        actual = {s.value for s in TaskStatus}
        assert actual == expected


class TestTaskType:
    """Tests for TaskType enum."""

    def test_hashable(self):
        """TaskType must be hashable for use in sets/dicts."""
        s = {TaskType.GRPOTASK, TaskType.DPOTASK, TaskType.GRPOTASK}
        assert len(s) == 2

    def test_all_types(self):
        assert len(TaskType) == 6


class TestRewardFunction:
    """Tests for RewardFunction model."""

    def test_valid_reward_function(self):
        rf = RewardFunction(
            reward_func="def reward(completions, **kwargs):\n    return [1.0] * len(completions)",
            reward_weight=1.0,
        )
        assert rf.reward_weight == 1.0
        assert rf.reward_id is None

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError):
            RewardFunction(reward_func="def f(): pass", reward_weight=-0.5)

    def test_zero_weight_allowed(self):
        rf = RewardFunction(reward_func="def f(): pass", reward_weight=0.0)
        assert rf.reward_weight == 0.0


class TestDpoDatasetType:
    """Tests for DpoDatasetType defaults."""

    def test_defaults(self):
        dt = DpoDatasetType()
        assert dt.prompt_format == "{prompt}"
        assert dt.chosen_format == "{chosen}"
        assert dt.rejected_format == "{rejected}"
        assert dt.field_prompt is None


class TestGrpoDatasetType:
    """Tests for GrpoDatasetType."""

    def test_defaults(self):
        dt = GrpoDatasetType()
        assert dt.field_prompt is None
        assert dt.reward_functions == []
        assert dt.extra_column is None


class TestJob:
    """Tests for Job model."""

    def test_auto_generates_uuid(self):
        job = Job(model="meta-llama/Llama-2-7b")
        assert job.job_id is not None
        uuid.UUID(job.job_id)  # should not raise

    def test_default_status_is_queued(self):
        job = Job(model="test")
        assert job.status == JobStatus.QUEUED


class TestMessage:
    """Tests for Message model."""

    def test_valid_message(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="invalid", content="test")


class TestMinerSubmission:
    """Tests for MinerSubmission model."""

    def test_optional_hash(self):
        sub = MinerSubmission(repo="org/model")
        assert sub.model_hash is None

    def test_with_hash(self):
        sub = MinerSubmission(repo="org/model", model_hash="abc123")
        assert sub.model_hash == "abc123"


class TestWinningSubmission:
    """Tests for WinningSubmission model."""

    def test_valid_submission(self):
        ws = WinningSubmission(hotkey="5abc", score=0.95, model_repo="org/model")
        assert ws.score == 0.95


class TestImageModelType:
    """Tests for ImageModelType enum."""

    def test_all_types(self):
        assert ImageModelType.FLUX.value == "flux"
        assert ImageModelType.SDXL.value == "sdxl"


class TestDiffusionJob:
    """Tests for DiffusionJob model."""

    def test_valid_diffusion_job(self):
        job = DiffusionJob(
            model="stabilityai/sdxl",
            dataset_zip="https://example.com/data.zip",
            model_type=ImageModelType.SDXL,
        )
        assert job.model_type == ImageModelType.SDXL

    def test_empty_dataset_zip_rejected(self):
        with pytest.raises(ValidationError):
            DiffusionJob(model="test", dataset_zip="")


class TestGPUInfo:
    """Tests for GPUInfo model."""

    def test_valid_gpu(self):
        gpu = GPUInfo(gpu_id=0, gpu_type=GPUType.H100, vram_gb=80, available=True)
        assert gpu.available is True
        assert gpu.used_until is None
