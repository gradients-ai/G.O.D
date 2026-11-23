"""On startup the trainer removes training containers that mount the whole checkpoints volume, and fails their jobs."""

from unittest.mock import MagicMock

import pytest


pytest.importorskip("pynvml")  # trainer.asgi starts the GPU monitor, which needs it

import trainer.constants as cst
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestText
from core.models.task_models import TaskStatus
from trainer import asgi


WHOLE_VOLUME = {"Type": "volume", "Name": cst.CHECKPOINTS_VOLUME_NAME, "Destination": "/app/checkpoints"}
OWN_JOB_DIR = {"Type": "bind", "Source": "/volumes/checkpoints/_data/task-1/hk-new", "Destination": "/app/checkpoints"}
CACHE = {"Type": "volume", "Name": cst.CACHE_VOLUME_NAME, "Destination": cst.CACHE_ROOT_PATH}


def _container(name: str, mounts: list[dict]) -> MagicMock:
    container = MagicMock(status="running", attrs={"Mounts": mounts})
    container.name = name
    return container


def _job(hotkey: str, container_name: str) -> TrainerTaskLog:
    training_data = TrainRequestText(
        model="org/model",
        task_id="task-1",
        hours_to_complete=1.0,
        expected_repo_name=f"repo-{hotkey}",
        dataset="s3://bucket/train.json",
        file_format=FileFormat.S3,
        dataset_type=InstructTextDatasetType(),
    )
    return TrainerTaskLog(
        training_data=training_data,
        github_repo="https://github.com/org/repo",
        gpu_ids=[0],
        hotkey=hotkey,
        status=TaskStatus.TRAINING,
        container_name=container_name,
    )


async def test_tracked_container_with_the_whole_volume_is_removed_and_its_job_failed(monkeypatch):
    old_text = _container("text-trainer-old", [WHOLE_VOLUME, CACHE])
    old_image = _container("image-trainer-old", [WHOLE_VOLUME, CACHE])
    current = _container("text-trainer-new", [OWN_JOB_DIR, CACHE])
    model_prep = _container("model-prep-1", [CACHE])
    unrelated = _container("postgres", [WHOLE_VOLUME])
    client = MagicMock()
    client.containers.list.return_value = [old_text, old_image, current, model_prep, unrelated]
    jobs = [_job("hk-old", "text-trainer-old"), _job("hk-old-image", "image-trainer-old"), _job("hk-new", "text-trainer-new")]
    failed: list[tuple[str, str, bool]] = []

    async def complete(task_id: str, hotkey: str, success: bool) -> None:
        failed.append((task_id, hotkey, success))

    async def log(task_id: str, hotkey: str, message: str) -> None:
        return None

    monkeypatch.setattr(asgi.docker, "from_env", lambda: client)
    monkeypatch.setattr(asgi, "get_running_jobs", lambda: jobs)
    monkeypatch.setattr(asgi, "complete_task", complete)
    monkeypatch.setattr(asgi, "log_task", log)

    await asgi.cleanup_orphaned_trainer_state()

    for removed in (old_text, old_image):
        removed.kill.assert_called_once()
        removed.remove.assert_called_once_with(force=True, v=True)
    for kept in (current, unrelated):
        kept.kill.assert_not_called()
        kept.remove.assert_not_called()
    model_prep.remove.assert_called_once()  # untracked, as before
    assert sorted(failed) == [("task-1", "hk-old", False), ("task-1", "hk-old-image", False)]
