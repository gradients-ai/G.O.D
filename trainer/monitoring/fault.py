"""Fail trainings that are sitting on a GPU the throttle monitor just withdrew."""

import asyncio

import docker

from core.logging import get_logger
from core.models.payload_models import TrainerTaskLog
from core.models.task_models import TaskStatus
from trainer import job_state
from trainer.monitoring.throttle import GPU_THROTTLE_FAILURE_MARKER


logger = get_logger(__name__)


def _kill_container(name: str) -> None:
    try:
        container = docker.from_env().containers.get(name)
        container.reload()
        if container.status == "running":
            container.kill()
    except Exception as e:
        logger.warning(f"Failed to kill container {name} after GPU throttle: {e}")


async def abort_trainings_on_gpu(gpu_index: int, reason: str) -> None:
    """Mark every training on this GPU as failed and stop its container.

    The failure log carries GPU_THROTTLE_FAILURE_MARKER so the orchestrator can
    put the training back to pending without counting an n_training_attempts.
    """
    victims = [
        job
        for job in list(job_state.task_history)
        if isinstance(job, TrainerTaskLog) and job.status == TaskStatus.TRAINING and gpu_index in job.gpu_ids
    ]
    for job in victims:
        task_id = job.training_data.task_id
        hotkey = job.hotkey
        job_state.note_gpu_throttle_failure(task_id, hotkey)
        await job_state.log_task(
            task_id,
            hotkey,
            f"{GPU_THROTTLE_FAILURE_MARKER} GPU {gpu_index} fault ({reason}); "
            "failing this training so it can be rescheduled on a healthy GPU",
        )
        if job.container_name:
            await asyncio.to_thread(_kill_container, job.container_name)
        _cancel_active_training(task_id, hotkey)
        await job_state.complete_task(task_id, hotkey, success=False)
        logger.warning(f"Failed training {task_id} hotkey {hotkey} because GPU {gpu_index} is throttling")


def _cancel_active_training(task_id: str, hotkey: str) -> None:
    # Imported lazily: endpoints imports the training runtime, which imports this module's callers.
    from trainer.endpoints import cancel_active_training

    cancel_active_training(task_id, hotkey)
