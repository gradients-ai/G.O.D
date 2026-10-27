from contextlib import asynccontextmanager
from typing import Annotated
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi import Path
from fastapi import Query
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError

from subnet_api import models
from subnet_api.settings import Settings
from subnet_api.validator_client import ValidatorClient


SS58 = Annotated[str, Path(pattern=r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")]
TASK_ID = Annotated[str, Path(pattern=r"^[0-9a-fA-F-]{36}$")]
TOURNAMENT_ID = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
LIMIT = Annotated[int, Query(ge=1, le=100)]
PAGE = Annotated[int, Query(ge=1, le=1000)]

AnyAuditTask = (
    models.InstructTextAuditTask
    | models.ChatAuditTask
    | models.DpoAuditTask
    | models.GrpoAuditTask
    | models.ImageAuditTask
    | models.EnvAuditTask
)
AnyAuditTaskWithHotkeyDetails = (
    models.InstructTextAuditTaskWithHotkeyDetails
    | models.ChatAuditTaskWithHotkeyDetails
    | models.DpoAuditTaskWithHotkeyDetails
    | models.GrpoAuditTaskWithHotkeyDetails
    | models.ImageAuditTaskWithHotkeyDetails
    | models.EnvAuditTaskWithHotkeyDetails
)


def _validator(request: Request) -> ValidatorClient:
    return request.app.state.validator


def _tasks(body: Any, *, with_hotkeys: bool) -> JSONResponse:
    if not isinstance(body, list):
        body = []
    return JSONResponse([models.audit_task(task, with_hotkeys=with_hotkeys) for task in body])


def create_app(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            base_url=settings.validator_url,
            headers={"Authorization": f"Bearer {settings.validator_api_key}"},
            transport=transport,
        ) as http:
            app.state.validator = ValidatorClient(http, timeout=settings.request_timeout, cache_seconds=settings.cache_seconds)
            yield

    app = FastAPI(
        title="Gradients SN56 API",
        description="Public, read-only data for subnet 56: auditing, tournaments and emissions.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET"])

    @app.exception_handler(ValidationError)
    async def unexpected_validator_response(request: Request, exc: ValidationError) -> JSONResponse:
        logger.warning(f"Validator response for {request.url.path} did not match the public model: {exc.error_count()} errors")
        return JSONResponse({"detail": "Unexpected validator response"}, status_code=502)

    slow = settings.slow_request_timeout

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/auditing/tasks", tags=["auditing"], response_model=list[AnyAuditTask])
    async def audit_tasks(request: Request, limit: LIMIT = 100, page: PAGE = 1):
        body = await _validator(request).get("/auditing/tasks", {"limit": limit, "page": page})
        return _tasks(body, with_hotkeys=False)

    @app.get("/auditing/tasks/hotkey/{hotkey}", tags=["auditing"], response_model=list[AnyAuditTaskWithHotkeyDetails])
    async def audit_tasks_for_hotkey(request: Request, hotkey: SS58, limit: LIMIT = 100, page: PAGE = 1):
        body = await _validator(request).get(f"/auditing/tasks/hotkey/{hotkey}", {"limit": limit, "page": page})
        return _tasks(body, with_hotkeys=True)

    @app.get("/auditing/tasks/{task_id}", tags=["auditing"], response_model=AnyAuditTaskWithHotkeyDetails)
    async def audit_task(request: Request, task_id: TASK_ID):
        body = await _validator(request).get(f"/auditing/tasks/{task_id}")
        return JSONResponse(models.audit_task(body, with_hotkeys=True))

    @app.get("/auditing/scores-url", tags=["auditing"])
    async def audit_scores_url(request: Request) -> models.ScoresUrlResponse:
        return models.ScoresUrlResponse.model_validate(await _validator(request).get("/auditing/scores-url"))

    @app.get("/tournament/balance/{coldkey}", tags=["tournaments"])
    async def tournament_balance(request: Request, coldkey: SS58) -> models.TournamentBalanceResponse:
        body = await _validator(request).get(f"/transfer/balance/{coldkey}")
        return models.TournamentBalanceResponse.model_validate(body)

    @app.get("/tournament/fees", tags=["tournaments"])
    async def tournament_fees(request: Request) -> models.TournamentFeesResponse:
        return models.TournamentFeesResponse.model_validate(await _validator(request).get("/tournament/fees"))

    @app.get("/tournament/latest/details", tags=["tournaments"])
    async def latest_tournaments(request: Request) -> models.LatestTournamentsDetailsPublicResponse:
        body = await _validator(request).get("/v1/tournaments/latest/details", timeout=slow)
        return models.LatestTournamentsDetailsPublicResponse.model_validate(body)

    @app.get("/tournament/{tournament_id}/details", tags=["tournaments"])
    async def tournament_details(request: Request, tournament_id: TOURNAMENT_ID) -> models.TournamentDetailsPublicResponse:
        body = await _validator(request).get(f"/v1/tournaments/{tournament_id}/details", timeout=slow)
        return models.TournamentDetailsPublicResponse.model_validate(body)

    @app.get("/v1/performance/latest-tournament-weights", tags=["performance"])
    async def latest_tournament_weights(request: Request) -> models.TournamentWeightsResponse:
        body = await _validator(request).get("/v1/performance/latest-tournament-weights", timeout=slow)
        return models.TournamentWeightsResponse.model_validate(body)

    @app.get("/v1/performance/weight-projection", tags=["performance"])
    async def weight_projection(
        request: Request,
        percentage_improvement: Annotated[float, Query(ge=0, le=1000)],
    ) -> models.WeightProjectionResponse:
        body = await _validator(request).get(
            "/v1/performance/weight-projection", {"percentage_improvement": percentage_improvement}, timeout=slow
        )
        return models.WeightProjectionResponse.model_validate(body)

    @app.get("/v1/performance/weight-projection-static", tags=["performance"])
    async def weight_projection_static(request: Request) -> models.MultiWeightProjectionResponse:
        body = await _validator(request).get("/v1/performance/weight-projection-static", timeout=slow)
        return models.MultiWeightProjectionResponse.model_validate(body)

    @app.get("/v1/performance/last-boss-battle", tags=["performance"])
    async def last_boss_battle(request: Request) -> models.BossBattleResponse:
        body = await _validator(request).get("/v1/performance/last-boss-battle", timeout=slow)
        return models.BossBattleResponse.model_validate(body)

    return app


def app() -> FastAPI:
    """Uvicorn factory: `uvicorn subnet_api.app:app --factory`."""
    return create_app(Settings.from_env())
