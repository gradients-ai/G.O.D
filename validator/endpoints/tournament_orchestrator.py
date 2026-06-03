import httpx
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Response

from core.logging import get_logger
from validator.app.config import Config
from validator.app.dependencies import get_api_key
from validator.app.dependencies import get_config
from validator.db.sql import tournaments as tournament_sql
from validator.tournament.models import TrainerInfo
from validator.tournament.orchestrator import fetch_trainer_gpus


logger = get_logger(__name__)


# Endpoint constants
ADD_TRAINER_ENDPOINT = "/v1/tournament_orchestrator/add_trainer"
REMOVE_TRAINER_ENDPOINT = "/v1/tournament_orchestrator/remove_trainer/{trainer_ip}"
GET_TRAINERS_ENDPOINT = "/v1/tournament_orchestrator/trainers"
GET_TRAINING_STATS_ENDPOINT = "/v1/tournament_orchestrator/training_stats"


async def add_trainer(
    trainer_ip: str,
    config: Config = Depends(get_config),
) -> Response:
    """
    Add a trainer to the tournament system by fetching its GPU availability
    and storing the information in the database.
    """
    try:
        gpu_infos = await fetch_trainer_gpus(trainer_ip)
        await tournament_sql.add_trainer_gpus(trainer_ip, gpu_infos, config.psql_db)

        logger.info(f"Successfully added trainer {trainer_ip} with {len(gpu_infos)} GPUs")
        return Response(content="Trainer added successfully", status_code=200)

    except httpx.RequestError as e:
        logger.error(f"Failed to contact trainer {trainer_ip}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Failed to contact trainer: {str(e)}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Trainer {trainer_ip} returned error status: {e.response.status_code}")
        raise HTTPException(status_code=502, detail=f"Trainer returned error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Unexpected error adding trainer {trainer_ip}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def remove_trainer(
    trainer_ip: str,
    config: Config = Depends(get_config),
) -> Response:
    """
    Remove a trainer from the tournament system by deleting all its GPU information.
    """
    try:
        await tournament_sql.remove_trainer(trainer_ip, config.psql_db)
        logger.info(f"Successfully removed trainer {trainer_ip}")
        return Response(content="Trainer removed successfully", status_code=200)

    except Exception as e:
        logger.error(f"Error removing trainer {trainer_ip}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def get_trainers(
    config: Config = Depends(get_config),
) -> list[TrainerInfo]:
    """
    Get all trainers and their GPU information from the tournament system.
    """
    try:
        trainers = await tournament_sql.get_trainers(config.psql_db)
        logger.info(f"Retrieved {len(trainers)} trainers")
        return trainers

    except Exception as e:
        logger.error(f"Error retrieving trainers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def get_training_stats(
    config: Config = Depends(get_config),
) -> dict:
    """
    Get statistics about tournament training status for monitoring.
    """
    try:
        stats = await tournament_sql.get_tournament_training_stats(config.psql_db)
        logger.info(f"Retrieved tournament training stats: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Error retrieving training stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Tournament Orchestrator"], dependencies=[Depends(get_api_key)])
    router.add_api_route(ADD_TRAINER_ENDPOINT, add_trainer, methods=["POST"])
    router.add_api_route(REMOVE_TRAINER_ENDPOINT, remove_trainer, methods=["DELETE"])
    router.add_api_route(GET_TRAINERS_ENDPOINT, get_trainers, methods=["GET"])
    router.add_api_route(GET_TRAINING_STATS_ENDPOINT, get_training_stats, methods=["GET"])
    return router
