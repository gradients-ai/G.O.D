import asyncio
import json
import os
import tempfile
import threading
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from pydantic import TypeAdapter

from core.logging import get_logger
from core.models.payload_models import ModelPrepJob
from core.models.payload_models import TrainerJob
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.task_models import TaskStatus
from trainer import constants as cst


logger = get_logger(__name__)

task_history: list[TrainerJob] = []
TASK_HISTORY_FILE = Path(cst.TASKS_FILE_PATH)
_task_lock = asyncio.Lock()
_task_file_lock = threading.Lock()
_TASK_HISTORY_READ_RETRIES = 3
_TASK_HISTORY_RETRY_DELAY_SECONDS = 0.5
_job_adapter = TypeAdapter(TrainerTaskLog | ModelPrepJob)


# ---------------------------------------------------------------------------
# Training job helpers
# ---------------------------------------------------------------------------

async def start_task(task: TrainerProxyRequest) -> tuple[str, str]:
    async with _task_lock:
        return await _start_task_unlocked(task)


async def _start_task_unlocked(task: TrainerProxyRequest) -> tuple[str, str]:
    load_task_history()

    task_id = task.training_data.task_id
    hotkey = task.hotkey

    existing_task = get_task(task_id, hotkey)
    if existing_task:
        if existing_task.status == TaskStatus.TRAINING:
            raise ValueError(f"Task {task_id} for hotkey {hotkey} is already training")
        existing_task.logs.clear()
        existing_task.status = TaskStatus.TRAINING
        existing_task.started_at = datetime.utcnow()
        existing_task.finished_at = None
        existing_task.gpu_ids = task.gpu_ids
        await save_task_history()
        return task_id, hotkey

    log_entry = TrainerTaskLog(
        **task.dict(),
        status=TaskStatus.TRAINING,
        started_at=datetime.utcnow(),
        finished_at=None,
    )
    task_history.append(log_entry)
    await save_task_history()
    return log_entry.training_data.task_id, log_entry.hotkey


async def complete_task(task_id: str, hotkey: str, success: bool = True):
    async with _task_lock:
        load_task_history()

        task = get_task(task_id, hotkey)
        if task is None:
            return
        task.status = TaskStatus.SUCCESS if success else TaskStatus.FAILURE
        task.finished_at = datetime.utcnow()
        await save_task_history()


def get_task(task_id: str, hotkey: str) -> TrainerTaskLog | None:
    for job in task_history:
        if isinstance(job, TrainerTaskLog) and job.training_data.task_id == task_id and job.hotkey == hotkey:
            return job
    return None


async def log_task(task_id: str, hotkey: str, message: str):
    async with _task_lock:
        load_task_history()

        task = get_task(task_id, hotkey)
        if task:
            timestamped_message = f"[{datetime.utcnow().isoformat()}] {message}"
            task.logs.append(timestamped_message)
            await save_task_history()


async def update_wandb_url(task_id: str, hotkey: str, wandb_url: str):
    async with _task_lock:
        load_task_history()

        task = get_task(task_id, hotkey)
        if task:
            task.wandb_url = wandb_url
            await save_task_history()
            logger.info(f"Updated wandb_url for task {task_id}: {wandb_url}")
        else:
            logger.warning(f"Task not found for task_id={task_id} and hotkey={hotkey}")


async def update_container_name(task_id: str, hotkey: str, container_name: str):
    async with _task_lock:
        load_task_history()

        task = get_task(task_id, hotkey)
        if task:
            task.container_name = container_name
            await save_task_history()
            logger.info(f"Updated container_name for task {task_id}: {container_name}")
        else:
            logger.warning(f"Task not found for task_id={task_id} and hotkey={hotkey}")


# ---------------------------------------------------------------------------
# Model prep job helpers
# ---------------------------------------------------------------------------

async def _start_model_prep_unlocked(task_id: str, model_id: str, gpu_ids: list[int], hotkey: str | None = None) -> ModelPrepJob:
    load_task_history()

    now = datetime.utcnow()
    for existing in task_history:
        if isinstance(existing, ModelPrepJob) and existing.task_id == task_id and existing.status == TaskStatus.TRAINING:
            # For per-miner preps, only mark the same hotkey's job as failed
            if hotkey and existing.hotkey and existing.hotkey != hotkey:
                continue
            logger.warning(
                "model_prep retry: marking previous TRAINING entry for task_id=%s gpu_ids=%s as FAILURE",
                task_id,
                existing.gpu_ids,
            )
            existing.status = TaskStatus.FAILURE
            existing.finished_at = now

    job = ModelPrepJob(
        task_id=task_id,
        model_id=model_id,
        gpu_ids=gpu_ids,
        hotkey=hotkey,
        status=TaskStatus.TRAINING,
        started_at=now,
    )
    task_history.append(job)
    await save_task_history()
    return job


async def complete_model_prep(task_id: str, success: bool = True, result=None, hotkey: str | None = None):
    async with _task_lock:
        load_task_history()

        job = get_model_prep_job(task_id, hotkey)
        if job is None:
            return
        job.status = TaskStatus.SUCCESS if success else TaskStatus.FAILURE
        job.finished_at = datetime.utcnow()
        if result is not None:
            job.result = result
        await save_task_history()


def get_model_prep_job(task_id: str, hotkey: str | None = None) -> ModelPrepJob | None:
    # Prefer the currently-running entry so complete_model_prep targets the
    # active job, not an older entry with the same task_id from a prior retry.
    fallback: ModelPrepJob | None = None
    for job in task_history:
        if isinstance(job, ModelPrepJob) and job.task_id == task_id:
            if hotkey and job.hotkey and job.hotkey != hotkey:
                continue
            if job.status == TaskStatus.TRAINING:
                return job
            if fallback is None:
                fallback = job
    return fallback


# ---------------------------------------------------------------------------
# Shared queries
# ---------------------------------------------------------------------------

def get_running_jobs() -> list[TrainerJob]:
    load_task_history()
    return [j for j in task_history if j.status == TaskStatus.TRAINING]


def get_recent_tasks(hours: float = 1.0) -> list[TrainerJob]:
    load_task_history()
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    recent = [
        job
        for job in task_history
        if (job.started_at and job.started_at >= cutoff) or (job.finished_at and job.finished_at >= cutoff)
    ]

    recent.sort(key=lambda j: max(j.finished_at or datetime.min, j.started_at or datetime.min), reverse=True)
    return recent


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

async def save_task_history():
    data = json.dumps([t.model_dump() for t in task_history], indent=2, default=str)
    await asyncio.to_thread(_atomic_write_task_history, data)


def load_task_history():
    global task_history
    if TASK_HISTORY_FILE.exists():
        for attempt in range(_TASK_HISTORY_READ_RETRIES):
            try:
                data = _read_task_history()
                task_history.clear()
                task_history.extend(_job_adapter.validate_python(item) for item in data)
                return
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < _TASK_HISTORY_READ_RETRIES - 1:
                    time.sleep(_TASK_HISTORY_RETRY_DELAY_SECONDS)
                    continue
                logger.error(f"Failed to load task history from {TASK_HISTORY_FILE}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading task history: {e}")
                return


def _read_task_history() -> list[dict]:
    with _task_file_lock:
        with open(TASK_HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read()

    if not content.strip():
        raise json.JSONDecodeError("Empty task history file", content, 0)

    return json.loads(content)


def _atomic_write_task_history(data: str) -> None:
    TASK_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    with _task_file_lock:
        fd, temp_path = tempfile.mkstemp(
            prefix=f"{TASK_HISTORY_FILE.name}.",
            suffix=".tmp",
            dir=str(TASK_HISTORY_FILE.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(data)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(temp_path, TASK_HISTORY_FILE)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
