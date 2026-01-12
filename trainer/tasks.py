import asyncio
import json
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import aiofiles

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.utility_models import TaskStatus
from trainer import constants as cst
from validator.utils.logging import get_logger


logger = get_logger(__name__)

task_history: list[TrainerTaskLog] = []
TASK_HISTORY_FILE = Path(cst.TASKS_FILE_PATH)
_task_lock = asyncio.Lock()


async def start_task(task: TrainerProxyRequest) -> tuple[str, str]:
    async with _task_lock:
        load_task_history()
        
        task_id = task.training_data.task_id
        hotkey = task.hotkey

        existing_task = get_task(task_id, hotkey)
        if existing_task:
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
    async with aiofiles.open(TASK_HISTORY_FILE, "w") as f:
        data = json.dumps([t.model_dump() for t in task_history], indent=2, default=str)
        await f.write(data)


def load_task_history():
    global task_history
    if TASK_HISTORY_FILE.exists():
        try:
            with open(TASK_HISTORY_FILE, "r") as f:
                data = json.load(f)
                task_history.clear()
                task_history.extend(TrainerTaskLog(**item) for item in data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load task history from {TASK_HISTORY_FILE}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading task history: {e}")
