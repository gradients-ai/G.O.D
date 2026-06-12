from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from loguru import logger
from pydantic import BaseModel  # noqa

from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.models import AnyTypeTask
from validator.core.models import AnyTypeTaskWithHotkeyDetails
from core.models.tournament_models import DupRelationship
from core.models.tournament_models import TournamentDedupReview
from validator.db.sql.auditing import get_latest_scores_url
from validator.db.sql.auditing import get_recent_tasks
from validator.db.sql.auditing import get_recent_tasks_for_hotkey
from validator.db.sql.auditing import get_task_with_hotkey_details
from validator.db.sql.dedup import get_resolved_dedup_reviews
from validator.utils.minio import async_minio_client


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


@router.get("/auditing/dedup")
async def audit_dedup_reviews_endpoint(
    limit: int = Query(100, ge=1, le=200),
    page: int = Query(1, ge=1),
    config: Config = Depends(get_config),
) -> list[TournamentDedupReview]:
    """Confirmed functional-duplicate eliminations, for public transparency.

    Returns the reasoning, the full report URL, and the PUBLIC re-uploaded copies of the
    offending repos (not the miners' original private repo names) so anyone can clone them
    and re-run the de-duplication check themselves.
    """
    reviews = await get_resolved_dedup_reviews(config.psql_db, limit=limit, page=page)
    # report_url is stored as a presigned S3 URL that expires (~7 days); re-sign on read so
    # the public audit link doesn't go dead after the original signature lapses.
    for review in reviews:
        # DISTINCT verdict reasons describe how non-flagged miners' repos differ (their
        # training innovations) — keep them in the DB but out of the public payload.
        review.pair_verdicts = [v for v in review.pair_verdicts if v.relationship != DupRelationship.DISTINCT]
        if review.report_url:
            fresh_url = await async_minio_client.get_new_presigned_url(review.report_url)
            if fresh_url:
                review.report_url = fresh_url
    return reviews


def factory_router():
    return router
