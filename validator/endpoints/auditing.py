from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from loguru import logger
from pydantic import BaseModel  # noqa

from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.models import AnyTypeTask
from validator.core.models import AnyTypeTaskWithHotkeyDetails
from validator.db.sql.auditing import get_latest_scores_url
from validator.db.sql.auditing import get_recent_tasks
from validator.db.sql.auditing import get_recent_tasks_for_hotkey
from validator.db.sql.auditing import get_task_with_hotkey_details


router = APIRouter(tags=["auditing"])


@router.get("/auditing/tasks")
async def audit_recent_tasks_endpoint(
    limit: int = 100, page: int = 1, config: Config = Depends(get_config)
) -> list[AnyTypeTask]:
    return await get_recent_tasks(None, limit=limit, page=page, config=config)


@router.get("/auditing/tasks/hotkey/{hotkey}")
async def audit_recent_tasks_for_hotkey_endpoint(
    hotkey: str, limit: int = 100, page: int = 1, config: Config = Depends(get_config)
) -> list[AnyTypeTaskWithHotkeyDetails]:
    return await get_recent_tasks_for_hotkey(hotkey, limit=limit, page=page, config=config)


@router.get("/auditing/tasks/{task_id}")
async def audit_task_details_endpoint(
    task_id: str, config: Config = Depends(get_config)
) -> AnyTypeTaskWithHotkeyDetails:
    logger.info(f"Getting task details for task {task_id}")
    return await get_task_with_hotkey_details(task_id, config)


class ScoresUrlResponse(BaseModel):
    url: str


@router.get("/auditing/scores-url")
async def audit_latest_scores_url_endpoint(config: Config = Depends(get_config)) -> ScoresUrlResponse:
    """
    Get the scores url for when I last set weights, to prove I did it right
    """
    url = await get_latest_scores_url(config)
    if url is None:
        raise HTTPException(status_code=400, detail="No scores url found... sorry :/")
    return ScoresUrlResponse(url=url)


def factory_router():
    return router
