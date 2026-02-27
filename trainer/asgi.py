import asyncio
from datetime import datetime

import docker
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trainer.endpoints import factory_router
from trainer.tasks import complete_task
from trainer.tasks import get_running_tasks
from trainer.tasks import log_task
from trainer.utils.cleanup_loop import start_cleanup_loop_in_thread
from validator.utils.logging import get_logger


load_dotenv(".trainer.env")

logger = get_logger(__name__)


def _list_running_trainer_containers() -> list:
    client = docker.from_env()
    containers = client.containers.list()
    return [c for c in containers if c.name.startswith("text-trainer-") or c.name.startswith("image-trainer-")]


def _remove_container(container) -> None:
    container.reload()
    if container.status == "running":
        container.kill()
    container.remove(force=True)


async def cleanup_orphaned_trainer_state():
    running_tasks = get_running_tasks()
    tracked_container_names = {getattr(t, "container_name", None) for t in running_tasks if getattr(t, "container_name", None)}

    running_containers = await asyncio.to_thread(_list_running_trainer_containers)
    running_container_names = {c.name for c in running_containers}

    orphan_containers = [c for c in running_containers if c.name not in tracked_container_names]
    for container in orphan_containers:
        try:
            await asyncio.to_thread(_remove_container, container)
            logger.warning(f"Removed orphaned trainer container {container.name}")
        except Exception as e:
            logger.warning(f"Failed to remove orphaned trainer container {container.name}: {e}")

    active_container_names = running_container_names - {c.name for c in orphan_containers}
    stale_tasks = [
        t
        for t in running_tasks
        if not getattr(t, "container_name", None) or getattr(t, "container_name", None) not in active_container_names
    ]
    now = datetime.utcnow().isoformat()
    for task in stale_tasks:
        await log_task(
            task.training_data.task_id,
            task.hotkey,
            f"[STARTUP_RECOVERY] Marking stale training task as failure at {now}.",
        )
        await complete_task(task.training_data.task_id, task.hotkey, success=False)


def factory() -> FastAPI:
    logger.debug("Entering factory function")
    app = FastAPI()
    app.include_router(factory_router())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        await cleanup_orphaned_trainer_state()
        logger.info("Starting async cleanup loop in a background thread")
        start_cleanup_loop_in_thread()

    return app


app = factory()

if __name__ == "__main__":
    logger.info("Starting trainer")
    uvicorn.run(app, host="0.0.0.0", port=8001)
