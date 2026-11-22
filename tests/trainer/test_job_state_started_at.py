"""Tests for refreshing training started_at when the train container starts."""

import json
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pytest

from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestText
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from trainer import job_state


def _job(status: TaskStatus, started_at: datetime) -> TrainerTaskLog:
    return TrainerTaskLog(
        training_data=TrainRequestText(
            model="org/model",
            task_id="task-1",
            hours_to_complete=1.0,
            expected_repo_name="repo-1",
            dataset="s3://bucket/train.json",
            file_format=FileFormat.S3,
            dataset_type=InstructTextDatasetType(),
            task_type=TaskType.INSTRUCTTEXTTASK,
        ),
        github_repo="https://github.com/org/repo",
        gpu_ids=[0],
        hotkey="hotkey-a",
        status=status,
        started_at=started_at,
    )


@pytest.fixture
def history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = tmp_path / "task_history.json"
    monkeypatch.setattr(job_state, "TASK_HISTORY_FILE", path)
    monkeypatch.setattr(job_state, "task_history", [])

    def write(job: TrainerTaskLog) -> None:
        path.write_text(json.dumps([job.model_dump(mode="json")]))

    return write


def _stored_started_at() -> datetime:
    return job_state.read_persisted_jobs()[0].started_at


@pytest.mark.asyncio
async def test_update_training_started_at_refreshes_clock(history):
    accept_time = datetime.utcnow() - timedelta(hours=1)
    history(_job(TaskStatus.TRAINING, accept_time))

    container_start = datetime.utcnow()
    await job_state.update_training_started_at("task-1", "hotkey-a", started_at=container_start)

    assert _stored_started_at() == container_start
    assert _stored_started_at() > accept_time


@pytest.mark.asyncio
async def test_update_training_started_at_skips_non_training(history):
    accept_time = datetime.utcnow() - timedelta(hours=1)
    history(_job(TaskStatus.FAILURE, accept_time))

    await job_state.update_training_started_at("task-1", "hotkey-a", started_at=datetime.utcnow())

    assert _stored_started_at() == accept_time
