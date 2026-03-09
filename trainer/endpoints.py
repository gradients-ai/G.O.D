import asyncio
import os

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.utility_models import GPUInfo
from trainer import constants as cst
from trainer.image_manager import start_training_task
from trainer.tasks import complete_task
from trainer.tasks import get_recent_tasks
from trainer.tasks import get_task
from trainer.tasks import load_task_history
from trainer.tasks import log_task
from trainer.tasks import _start_task_unlocked
from trainer.tasks import _task_lock
from trainer.utils.trainer_logging import logger
from trainer.utils.misc import are_gpus_available
from trainer.utils.misc import clone_repo
from trainer.utils.misc import get_gpu_info
from validator.core.constants import GET_GPU_AVAILABILITY_ENDPOINT
from validator.core.constants import GET_RECENT_TASKS_ENDPOINT
from validator.core.constants import PROXY_TRAINING_IMAGE_ENDPOINT
from validator.core.constants import TASK_DETAILS_ENDPOINT


load_task_history()
_active_tasks: dict[tuple[str, str], asyncio.Task] = {}


async def _remove_active_task(task_key: tuple[str, str], bg_task: asyncio.Task) -> None:
    async with _task_lock:
        if _active_tasks.get(task_key) is bg_task:
            _active_tasks.pop(task_key, None)


async def _run_training_with_clone(req: TrainerProxyRequest) -> None:
    task_id = req.training_data.task_id
    hotkey = req.hotkey
    try:
        local_repo_path = await asyncio.to_thread(
            clone_repo,
            repo_url=req.github_repo,
            parent_dir=cst.TEMP_REPO_PATH,
            branch=req.github_branch,
            commit_hash=req.github_commit_hash,
        )
    except Exception as e:
        await log_task(task_id, hotkey, f"Failed to clone repo: {str(e)}")
        await complete_task(task_id, hotkey, success=False)
        logger.exception("Repository clone failed before training start", extra={"task_id": task_id, "hotkey": hotkey})
        return

    logger.info(
        f"Repo {req.github_repo} cloned to {local_repo_path}",
        extra={"task_id": task_id, "hotkey": hotkey, "model": req.training_data.model},
    )
    await start_training_task(req, local_repo_path)


async def verify_orchestrator_ip(request: Request):
    """Verify request comes from orchestrator IP"""
    client_ip = request.client.host
    allowed_ips_str = os.getenv("ORCHESTRATOR_IPS", os.getenv("ORCHESTRATOR_IP", "185.141.218.122"))
    allowed_ips = [ip.strip() for ip in allowed_ips_str.split(",")]
    allowed_ips.append("127.0.0.1")  # Always allow localhost

    if client_ip not in allowed_ips:
        raise HTTPException(status_code=403, detail="Access forbidden")
    return client_ip
    

async def start_training(req: TrainerProxyRequest) -> JSONResponse:
    task_key = (req.training_data.task_id, req.hotkey)
    bg_task = None
    async with _task_lock:
        existing_bg_task = _active_tasks.get(task_key)
        if existing_bg_task and not existing_bg_task.done():
            raise HTTPException(
                status_code=409,
                detail=f"Task {req.training_data.task_id} for hotkey {req.hotkey} is already running.",
            )
        if not await asyncio.to_thread(are_gpus_available, req.gpu_ids):
            raise HTTPException(
                status_code=409,
                detail="GPU conflict detected. Requested GPUs are already in use by running training tasks.",
            )
        try:
            await _start_task_unlocked(req)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        bg_task = asyncio.create_task(_run_training_with_clone(req))
        _active_tasks[task_key] = bg_task

    bg_task.add_done_callback(lambda finished_task: asyncio.create_task(_remove_active_task(task_key, finished_task)))

    return {"message": "Started Training!", "task_id": req.training_data.task_id}


async def get_available_gpus() -> list[GPUInfo]:
    gpu_info = await get_gpu_info()
    return gpu_info


async def get_task_details(task_id: str, hotkey: str) -> TrainerTaskLog:
    task = get_task(task_id, hotkey)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' and hotkey '{hotkey}' not found.")
    return task


async def get_recent_tasks_list(hours: int) -> list[TrainerTaskLog]:
    tasks = get_recent_tasks(hours)
    if not tasks:
        raise HTTPException(status_code=404, detail=f"Tasks not found in the last {hours} hours.")
    return tasks


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Proxy Trainer"])
    router.add_api_route(
        PROXY_TRAINING_IMAGE_ENDPOINT, start_training, methods=["POST"], dependencies=[Depends(verify_orchestrator_ip)]
    )
    router.add_api_route(
        GET_GPU_AVAILABILITY_ENDPOINT, get_available_gpus, methods=["GET"], dependencies=[Depends(verify_orchestrator_ip)]
    )
    router.add_api_route(
        GET_RECENT_TASKS_ENDPOINT, get_recent_tasks_list, methods=["GET"], dependencies=[Depends(verify_orchestrator_ip)]
    )
    router.add_api_route(TASK_DETAILS_ENDPOINT, get_task_details, methods=["GET"], dependencies=[Depends(verify_orchestrator_ip)])
    return router
