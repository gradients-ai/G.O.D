import asyncio
import threading
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import docker
from dateutil.parser import isoparse

from core.models.utility_models import TaskStatus
from trainer import constants as cst
from trainer.tasks import save_task_history
from trainer.tasks import task_history
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import get_logger
from validator.utils.logging import stream_container_logs


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
            for task in task_history:
                if task.status != TaskStatus.TRAINING or not task.started_at:
                    continue

                timeout = timedelta(hours=task.training_data.hours_to_complete) + timedelta(minutes=cst.STALE_TASK_GRACE_MINUTES)
                deadline = task.started_at + timeout

                if now > deadline:
                    task.status = TaskStatus.FAILURE
                    task.finished_at = now
                    task.logs.append(f"[{now.isoformat()}] Task marked as FAILED due to timeout.")
                    await save_task_history()

            client = docker.from_env()
            abs_task_path = Path(cst.TASKS_FILE_PATH).resolve()

            if abs_task_path.exists():
                logger.info("Starting cleanup container...")

                container = client.containers.run(
                    image=cst.CACHE_CLEANER_DOCKER_IMAGE,
                    volumes={
                        cst.VOLUME_NAMES[0]: {"bind": "/checkpoints", "mode": "rw"},
                        cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"},
                        str(abs_task_path): {"bind": "/app/trainer/task_history.json", "mode": "ro"},
                    },
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
