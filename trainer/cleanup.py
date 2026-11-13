import asyncio
import os
import threading
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import docker
from dateutil.parser import isoparse
from docker.models.containers import Container

from core.constants.quant import DECODE_TIMEOUT_SECONDS
from core.logging import get_all_context_tags
from core.logging import get_logger
from core.logging import stream_container_logs
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestQuant
from core.models.task_models import TaskStatus
from trainer import constants as cst
from trainer.job_state import save_task_history
from trainer.job_state import task_history


logger = get_logger(__name__)

# Extra stale-deadline slack for a freeform job that decodes after its upload: build + run caps plus transfers.
# Lives here, not in trainer/constants.py, because the cache-cleanup image ships trainer/ without core/.
QUANT_DECODE_GRACE_MINUTES = 2 * DECODE_TIMEOUT_SECONDS // 60 + 30


def start_cleanup_loop_in_thread():
    def run():
        asyncio.run(periodically_cleanup_tasks_and_cache())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


def _decodes_after_upload(job: TrainerTaskLog) -> bool:
    """A freeform job with a decode request keeps TRAINING through the sandbox run after its upload."""
    return isinstance(job.training_data, TrainRequestQuant) and job.training_data.decode is not None


def _is_trainer_container(container: Container) -> bool:
    return (container.name or "").startswith(cst.TRAINER_CONTAINER_NAME_PREFIXES)


def remove_stale_trainer_containers(client: docker.DockerClient) -> None:
    """Remove this trainer's own containers that have been created or exited for over STALE_CONTAINER_MAX_AGE_HOURS.

    Only names under TRAINER_CONTAINER_NAME_PREFIXES are touched, never another service's containers on the host.
    Each goes with its anonymous volumes (v=True); named volumes such as cache and checkpoints are never removed, and
    there is no host-wide volume prune.
    """
    now = datetime.now(timezone.utc)
    containers_to_remove = []
    for container in client.containers.list(all=True):
        if container.status not in ("created", "exited") or not _is_trainer_container(container):
            continue
        try:
            created_str = container.attrs.get("Created", "")
            if created_str:
                age_hours = (now - isoparse(created_str)).total_seconds() / 3600
                if age_hours > cst.STALE_CONTAINER_MAX_AGE_HOURS:
                    containers_to_remove.append(container)
        except Exception as e:
            logger.warning(f"Could not parse creation time for {container.name}: {e}")
            containers_to_remove.append(container)

    if containers_to_remove:
        logger.info(f"Cleaning up {len(containers_to_remove)} stopped/created trainer containers...")
    for container in containers_to_remove:
        try:
            container.remove(force=True, v=True)
            logger.debug(f"Removed container: {container.name or container.id[:12]}")
        except Exception as e:
            logger.warning(f"Failed to remove container {container.name or container.id[:12]}: {e}")
    logger.info("Trainer container cleanup completed.")


async def periodically_cleanup_tasks_and_cache(poll_interval_seconds: int = 600):
    while True:
        if len(task_history) > 0:
            now = datetime.utcnow()
            for job in task_history:
                if job.status != TaskStatus.TRAINING or not job.started_at:
                    continue

                # Training jobs refresh started_at when the train container starts (see
                # update_training_started_at), so this deadline tracks container.wait rather than
                # download/image-build time. Status-only: does not kill the container.
                if isinstance(job, TrainerTaskLog):
                    timeout = timedelta(hours=job.training_data.hours_to_complete) + timedelta(
                        minutes=cst.STALE_TASK_GRACE_MINUTES
                    )
                    if _decodes_after_upload(job):
                        timeout += timedelta(minutes=QUANT_DECODE_GRACE_MINUTES)
                else:
                    timeout = timedelta(minutes=cst.MODEL_PREP_TIMEOUT_MINUTES)

                deadline = job.started_at + timeout
                if now > deadline:
                    job.status = TaskStatus.FAILURE
                    job.finished_at = now
                    job.logs.append(f"[{now.isoformat()}] Job marked as FAILED due to timeout.")
                    await save_task_history()

            client = docker.from_env()
            abs_task_path = Path(cst.TASKS_FILE_PATH).resolve()

            if abs_task_path.exists():
                logger.info("Starting cleanup container...")

                container = client.containers.run(
                    image=cst.CACHE_CLEANER_DOCKER_IMAGE,
                    volumes={
                        cst.CHECKPOINTS_VOLUME_NAME: {"bind": "/checkpoints", "mode": "rw"},
                        cst.CACHE_VOLUME_NAME: {"bind": "/cache", "mode": "rw"},
                        str(abs_task_path): {"bind": "/app/trainer/task_history.json", "mode": "ro"},
                    },
                    environment={"MODEL_HASH_SALT": os.environ.get("MODEL_HASH_SALT", "")},
                    remove=True,
                    detach=True,
                )

                asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))

                logger.info("Cleanup container finished.")

        try:
            remove_stale_trainer_containers(docker.from_env())
        except Exception as e:
            logger.error(f"Error during container cleanup: {e}")

        await asyncio.sleep(poll_interval_seconds)
