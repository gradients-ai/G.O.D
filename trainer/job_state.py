import asyncio
import fcntl
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter

from core.logging import get_logger
from core.models.payload_models import ModelPrepJob
from core.models.payload_models import TrainerJob
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.quant_artifact_models import QuantDecodeResult
from core.models.task_models import TaskStatus
from trainer import constants as cst


logger = get_logger(__name__)

T = TypeVar("T")

task_history: list[TrainerJob] = []
TASK_HISTORY_FILE = Path(cst.TASKS_FILE_PATH)
# Trainings failed because their GPU throttled. complete_task will not later record these as success.
_gpu_throttle_failures: set[tuple[str, str]] = set()
# Orders history updates from the API loop's coroutines; the endpoints also hold it across their GPU checks.
_task_lock = asyncio.Lock()
# Guards the in-memory list and the file within this process. _history_lock adds a cross-process flock to it.
_task_file_lock = threading.RLock()
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
    """Caller holds _task_lock (the endpoint checks GPUs under it first)."""
    task_id = task.training_data.task_id
    hotkey = task.hotkey

    def start() -> tuple[str, str]:
        existing_task = get_task(task_id, hotkey)
        if existing_task:
            if existing_task.status == TaskStatus.TRAINING:
                raise ValueError(f"Task {task_id} for hotkey {hotkey} is already training")
            existing_task.logs.clear()
            existing_task.status = TaskStatus.TRAINING
            existing_task.started_at = datetime.utcnow()
            existing_task.finished_at = None
            existing_task.gpu_ids = task.gpu_ids
            return task_id, hotkey

        log_entry = TrainerTaskLog(
            **task.dict(),
            status=TaskStatus.TRAINING,
            started_at=datetime.utcnow(),
            finished_at=None,
        )
        task_history.append(log_entry)
        return log_entry.training_data.task_id, log_entry.hotkey

    return await asyncio.to_thread(update_history_now, start)


def note_gpu_throttle_failure(task_id: str, hotkey: str) -> None:
    _gpu_throttle_failures.add((task_id, hotkey))


def is_gpu_throttle_failure(task_id: str, hotkey: str) -> bool:
    return (task_id, hotkey) in _gpu_throttle_failures


async def complete_task(task_id: str, hotkey: str, success: bool = True):
    def complete() -> None:
        task = get_task(task_id, hotkey)
        if task is None:
            return
        succeeded = success and not is_gpu_throttle_failure(task_id, hotkey)
        task.status = TaskStatus.SUCCESS if succeeded else TaskStatus.FAILURE
        task.finished_at = datetime.utcnow()

    await update_history(complete)


def get_task(task_id: str, hotkey: str) -> TrainerTaskLog | None:
    for job in task_history:
        if isinstance(job, TrainerTaskLog) and job.training_data.task_id == task_id and job.hotkey == hotkey:
            return job
    return None


async def log_task(task_id: str, hotkey: str, message: str):
    def append_log() -> None:
        task = get_task(task_id, hotkey)
        if task:
            task.logs.append(f"[{datetime.utcnow().isoformat()}] {message}")

    await update_history(append_log)


async def _update_training_job(task_id: str, hotkey: str, apply: Callable[[TrainerTaskLog], None]) -> bool:
    """Apply a change to one training job and persist it. False when the job is unknown."""

    def update() -> bool:
        task = get_task(task_id, hotkey)
        if task is None:
            return False
        apply(task)
        return True

    found = await update_history(update)
    if not found:
        logger.warning(f"Task not found for task_id={task_id} and hotkey={hotkey}")
    return found


async def update_wandb_url(task_id: str, hotkey: str, wandb_url: str):
    def apply(task: TrainerTaskLog) -> None:
        task.wandb_url = wandb_url

    if await _update_training_job(task_id, hotkey, apply):
        logger.info(f"Updated wandb_url for task {task_id}: {wandb_url}")


async def update_container_name(task_id: str, hotkey: str, container_name: str):
    def apply(task: TrainerTaskLog) -> None:
        task.container_name = container_name

    if await _update_training_job(task_id, hotkey, apply):
        logger.info(f"Updated container_name for task {task_id}: {container_name}")


async def update_quant_decode_result(task_id: str, hotkey: str, result: QuantDecodeResult):
    def apply(task: TrainerTaskLog) -> None:
        task.quant_decode = result

    if await _update_training_job(task_id, hotkey, apply):
        logger.info(f"Updated quant_decode for task {task_id}: ok={result.ok}")


async def update_training_started_at(task_id: str, hotkey: str, started_at: datetime | None = None) -> None:
    """Refresh started_at when the training container actually starts.

    Job accept sets an initial started_at (before download/image build). Cleanup's stale-timeout
    clock uses started_at, so without this refresh it can mark FAILURE while container.wait is
    still inside hours_to_complete. Align the clock with the container.wait budget.
    """

    def refresh() -> TrainerTaskLog | None:
        task = get_task(task_id, hotkey)
        if task is None:
            logger.warning(f"Task not found for task_id={task_id} and hotkey={hotkey}")
            return None
        if task.status != TaskStatus.TRAINING:
            logger.warning(f"Refusing to refresh started_at for task {task_id} hotkey={hotkey}: status={task.status}")
            return None
        task.started_at = started_at or datetime.utcnow()
        return task

    task = await update_history(refresh)
    if task is not None:
        logger.info(f"Updated started_at for task {task_id} to {task.started_at.isoformat()}")


def fail_overdue_jobs(timeout_for: Callable[[TrainerJob], timedelta]) -> list[TrainerJob]:
    """Mark every TRAINING job past started_at + timeout_for(job) as FAILURE. Blocking; callable from any thread."""

    def fail_overdue() -> list[TrainerJob]:
        now = datetime.utcnow()
        overdue = [
            job
            for job in task_history
            if job.status == TaskStatus.TRAINING and job.started_at and now > job.started_at + timeout_for(job)
        ]
        for job in overdue:
            job.status = TaskStatus.FAILURE
            job.finished_at = now
            job.logs.append(f"[{now.isoformat()}] Job marked as FAILED due to timeout.")
        return overdue

    return update_history_now(fail_overdue)


# ---------------------------------------------------------------------------
# Model prep job helpers
# ---------------------------------------------------------------------------


async def _start_model_prep_unlocked(task_id: str, model_id: str, gpu_ids: list[int], hotkey: str | None = None) -> ModelPrepJob:
    """Caller holds _task_lock (the endpoint checks GPUs under it first)."""

    def start() -> ModelPrepJob:
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
        return job

    return await asyncio.to_thread(update_history_now, start)


async def complete_model_prep(task_id: str, success: bool = True, result=None, hotkey: str | None = None):
    def complete() -> None:
        job = get_model_prep_job(task_id, hotkey)
        if job is None:
            return
        job.status = TaskStatus.SUCCESS if success else TaskStatus.FAILURE
        job.finished_at = datetime.utcnow()
        if result is not None:
            job.result = result

    await update_history(complete)


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


def _history_payload(job: TrainerJob) -> dict:
    """Serialize a job for task_history.json.

    Training jobs carry the full baseline stats only so the container can read them
    at start. The history file does not need that payload. Model prep jobs keep it:
    the validator reads result.baseline_stats back from this file after a restart.
    """
    if isinstance(job, TrainerTaskLog) and job.training_data.baseline_stats is not None:
        job.training_data.baseline_stats = None
    payload = job.model_dump()
    if isinstance(job, TrainerTaskLog):
        training_data = payload.get("training_data")
        if isinstance(training_data, dict):
            training_data.pop("baseline_stats", None)
    return payload


async def update_history(mutate: Callable[[], T]) -> T:
    """Reload, mutate and persist the history as one step; see update_history_now."""
    async with _task_lock:
        return await asyncio.to_thread(update_history_now, mutate)


def update_history_now(mutate: Callable[[], T]) -> T:
    """Reload the history from disk, apply mutate to the in-memory list, then write it back, all under the history lock.

    Every change to the history goes through here, so no writer saves a stale copy over a newer file: not the API
    loop, not the cleanup thread, not another process such as trainer/quant_decode/rerun.py. Blocking; run it in a
    worker thread from async code.
    """
    with _history_lock():
        load_task_history()
        try:
            result = mutate()
            _atomic_write_task_history(_serialize_history())
        except BaseException:
            load_task_history()  # memory must not keep a change the file lacks
            raise
        return result


def _serialize_history() -> str:
    return json.dumps([_history_payload(t) for t in task_history], indent=2, default=str)


@contextmanager
def _history_lock() -> Iterator[None]:
    """Exclusive hold on the history across this process's threads and across processes (flock on a side file).

    Not re-entrant across the flock: never call update_history_now from inside a mutate. Readers need only
    _task_file_lock, since writes replace the file atomically.
    """
    with _task_file_lock:
        lock_path = TASK_HISTORY_FILE.with_name(TASK_HISTORY_FILE.name + cst.TASK_HISTORY_LOCK_SUFFIX)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield


def _load_history_items() -> list[TrainerJob] | None:
    """Parse the history file. None means missing or unreadable; the caller must not clear memory."""
    if not TASK_HISTORY_FILE.exists():
        return None
    for attempt in range(_TASK_HISTORY_READ_RETRIES):
        try:
            data = _read_task_history()
            return [_job_adapter.validate_python(item) for item in data]
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < _TASK_HISTORY_READ_RETRIES - 1:
                time.sleep(_TASK_HISTORY_RETRY_DELAY_SECONDS)
                continue
            logger.error(f"Failed to load task history from {TASK_HISTORY_FILE}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading task history: {e}")
            return None
    return None


def load_task_history():
    """Replace the in-memory list with the jobs on disk, in one step so other threads never see it half-filled.

    Every change is persisted before the history lock is released (update_history_now), so memory holds no job the
    file lacks and a reload drops nothing.
    """
    with _task_file_lock:
        items = _load_history_items()
        if items is None:
            return
        task_history[:] = items


def read_persisted_jobs() -> list[TrainerJob]:
    """Jobs on disk. Does not replace the in-memory task list."""
    return _load_history_items() or []


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
