import asyncio
import os
import threading
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import docker
from dateutil.parser import isoparse

from core.logging import get_all_context_tags
from core.logging import get_logger
from core.logging import stream_container_logs
from core.models.payload_models import TrainerTaskLog
from core.models.task_models import TaskStatus
from trainer import constants as cst
from trainer.job_state import save_task_history
from trainer.job_state import task_history


logger = get_logger(__name__)


def start_cleanup_loop_in_thread():
    def run():
        asyncio.run(periodically_cleanup_tasks_and_cache())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


async def periodically_cleanup_tasks_and_cache(poll_interval_seconds: int = 600):
    while True:
        if len(task_history) > 0:
            now = datetime.utcnow()
            for job in task_history:
                if job.status != TaskStatus.TRAINING or not job.started_at:
                    continue

                if isinstance(job, TrainerTaskLog):
                    timeout = timedelta(hours=job.training_data.hours_to_complete) + timedelta(minutes=cst.STALE_TASK_GRACE_MINUTES)
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

                log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))

                logger.info("Cleanup container finished.")

        try:
            client = docker.from_env()
            all_containers = client.containers.list(all=True)
            
            containers_to_remove = []
            for container in all_containers:
                if container.status in ['created', 'exited']:
                    try:
                        created_str = container.attrs.get('Created', '')
                        if created_str:
                            created_dt = isoparse(created_str)
                            now = datetime.now(timezone.utc)
                            age_hours = (now - created_dt).total_seconds() / 3600
                            
                            if age_hours > 1:
                                containers_to_remove.append(container)
                    except Exception as e:
                        logger.warning(f"Could not parse creation time for {container.name}: {e}")
                        containers_to_remove.append(container)
            
            if containers_to_remove:
                logger.info(f"Cleaning up {len(containers_to_remove)} stopped/created containers...")
                for container in containers_to_remove:
                    try:
                        container.remove(force=True, v=True)
                        logger.debug(f"Removed container: {container.name or container.id[:12]}")
                    except Exception as e:
                        logger.warning(f"Failed to remove container {container.name or container.id[:12]}: {e}")
            
            try:
                prune_result = client.volumes.prune()
                if prune_result.get('SpaceReclaimed', 0) > 0:
                    logger.info(f"Pruned volumes: {prune_result}")
            except Exception as e:
                logger.warning(f"Failed to prune volumes: {e}")
                
            logger.info("Container and volume cleanup completed.")
        except Exception as e:
            logger.error(f"Error during container cleanup: {e}")

        await asyncio.sleep(poll_interval_seconds)
