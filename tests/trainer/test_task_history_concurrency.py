"""task_history.json stays consistent when the API loop, the cleanup thread and other processes update it at once."""

import asyncio
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import ModelPrepJob
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestText
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from trainer import cleanup
from trainer import job_state


# How long the cleanup's write waits for a concurrent start to land first. Without a lock across read-modify-write,
# the start lands well inside it; with one, the start is blocked and the wait just runs out.
CONCURRENT_COMMIT_WAIT_SECONDS = 1.0


class _StopLoop(Exception):
    pass


def _request(task_id: str) -> TrainerProxyRequest:
    return TrainerProxyRequest(
        training_data=TrainRequestText(
            model="org/model",
            task_id=task_id,
            hours_to_complete=1.0,
            expected_repo_name=f"repo-{task_id}",
            dataset="s3://bucket/train.json",
            file_format=FileFormat.S3,
            dataset_type=InstructTextDatasetType(),
            task_type=TaskType.INSTRUCTTEXTTASK,
        ),
        github_repo="https://github.com/org/repo",
        gpu_ids=[1],
        hotkey="hk",
    )


def _stale_prep() -> ModelPrepJob:
    return ModelPrepJob(
        task_id="stale-prep",
        model_id="org/model",
        gpu_ids=[0],
        status=TaskStatus.TRAINING,
        started_at=datetime.utcnow() - timedelta(days=1),
    )


@pytest.fixture
def history(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "task_history.json"
    monkeypatch.setattr(job_state, "TASK_HISTORY_FILE", path)
    saved = list(job_state.task_history)
    job_state.task_history.clear()
    yield path
    job_state.task_history[:] = saved


def _persisted_keys() -> set[str]:
    return {
        job.task_id if isinstance(job, ModelPrepJob) else job.training_data.task_id for job in job_state.read_persisted_jobs()
    }


async def _run_one_cleanup_tick(monkeypatch) -> None:
    monkeypatch.setattr(cleanup, "docker", MagicMock())
    monkeypatch.setattr(cleanup, "prune_build_leftovers", lambda: None)

    async def stop(_seconds):
        raise _StopLoop

    monkeypatch.setattr(cleanup.asyncio, "sleep", stop)
    with pytest.raises(_StopLoop):
        await cleanup.periodically_cleanup_tasks_and_cache()


def test_cleanup_timeout_save_keeps_a_job_started_meanwhile(history: Path, monkeypatch):
    history.write_text(json.dumps([_stale_prep().model_dump(mode="json")]))
    job_state.load_task_history()

    original_write = job_state._atomic_write_task_history
    started_written = threading.Event()
    starter: list[threading.Thread] = []

    def start_new_job() -> None:
        asyncio.run(job_state.start_task(_request("new-job")))

    def write(data: str) -> None:
        if threading.current_thread() in starter:
            original_write(data)
            started_written.set()
            return
        if not starter:
            # The cleanup is about to persist its timeout mark: the API accepts a new job right now.
            starter.append(threading.Thread(target=start_new_job))
            starter[0].start()
            started_written.wait(CONCURRENT_COMMIT_WAIT_SECONDS)
        original_write(data)

    monkeypatch.setattr(job_state, "_atomic_write_task_history", write)

    cleanup_thread = threading.Thread(target=lambda: asyncio.run(_run_one_cleanup_tick(monkeypatch)))
    cleanup_thread.start()
    cleanup_thread.join()
    for thread in starter:
        thread.join()

    assert starter, "cleanup never persisted the timeout"
    assert _persisted_keys() == {"stale-prep", "new-job"}
    jobs = {getattr(j, "task_id", None) or j.training_data.task_id: j for j in job_state.read_persisted_jobs()}
    assert jobs["stale-prep"].status == TaskStatus.FAILURE
    assert jobs["new-job"].status == TaskStatus.TRAINING


_CHILD_WRITER = """
import sys
from pathlib import Path

from core.models.payload_models import ModelPrepJob
from core.models.task_models import TaskStatus
from trainer import job_state

job_state.TASK_HISTORY_FILE = Path(sys.argv[1])
print("ready", flush=True)
job = ModelPrepJob(task_id="child-job", model_id="org/model", gpu_ids=[0], status=TaskStatus.TRAINING)
job_state.update_history_now(lambda: job_state.task_history.append(job))
"""

# How long the parent keeps the lock after the child process is ready to write.
CHILD_BLOCK_SECONDS = 1.0


def test_another_process_waits_for_the_history_lock(history: Path):
    history.write_text("[]")
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_WRITER, str(history)],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        with job_state._history_lock():
            job_state.load_task_history()
            job_state.task_history.append(_stale_prep())
            time.sleep(CHILD_BLOCK_SECONDS)
            assert child.poll() is None, "the other process wrote while this one held the lock"
            job_state._atomic_write_task_history(job_state._serialize_history())
        assert child.wait(timeout=60) == 0
    finally:
        child.kill()

    assert _persisted_keys() == {"stale-prep", "child-job"}
