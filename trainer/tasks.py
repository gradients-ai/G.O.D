import asyncio
import json
import os
import tempfile
import threading
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.utility_models import TaskStatus
from trainer import constants as cst
from validator.utils.logging import get_logger


logger = get_logger(__name__)

task_history: list[TrainerTaskLog] = []
TASK_HISTORY_FILE = Path(cst.TASKS_FILE_PATH)
_task_lock = asyncio.Lock()
_task_file_lock = threading.Lock()
_TASK_HISTORY_READ_RETRIES = 3
_TASK_HISTORY_RETRY_DELAY_SECONDS = 0.5


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
    for task in task_history:
        if task.training_data.task_id == task_id and task.hotkey == hotkey:
            return task
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


def get_running_tasks() -> list[TrainerTaskLog]:
    load_task_history()
    return [t for t in task_history if t.status == TaskStatus.TRAINING]


def get_recent_tasks(hours: float = 1.0) -> list[TrainerTaskLog]:
    load_task_history()
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    recent_tasks = [
        task
        for task in task_history
        if (task.started_at and task.started_at >= cutoff) or (task.finished_at and task.finished_at >= cutoff)
    ]

    recent_tasks.sort(key=lambda t: max(t.finished_at or datetime.min, t.started_at or datetime.min), reverse=True)

    return recent_tasks


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
                task_history.extend(TrainerTaskLog(**item) for item in data)
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
