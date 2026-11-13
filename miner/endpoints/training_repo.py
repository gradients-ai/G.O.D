import os

from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import verify_get_request

from core.models.payload_models import TrainingRepoResponse
from core.models.tournament_models import TournamentType


def entered_tournaments() -> set[str]:
    return {name.strip().lower() for name in os.getenv("MINER_TOURNAMENT_TYPES", "").split(",") if name.strip()}


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    if task_type.value not in entered_tournaments():
        raise HTTPException(status_code=404, detail="not entering this tournament")
    return TrainingRepoResponse(
        github_repo="https://github.com/rayonlabs/G.O.D",
        commit_hash="5f161f642cd578b829e72dedd8444a491b9bbca3",
        github_token=None,  # Optional, only if the repository is private. Leave as None if not using a private repository.
        requested_datasets=None,
    )


def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/training_repo/{task_type}",
        get_training_repo,
        tags=["Subnet"],
        methods=["GET"],
        response_model=TrainingRepoResponse,
        summary="Get Training Repo",
        description="Retrieve the training repository and commit hash for the tournament.",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    return router
