"""
Transfer balance endpoints for coldkey balance information
"""

from typing import List

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from validator.core.config import Config
from validator.core.dependencies import get_api_key
from validator.core.dependencies import get_config
from validator.core.transfer_models import BalanceEventResponse
from validator.core.transfer_models import ColdkeyBalanceResponse
from validator.core.transfer_models import TournamentFeesResponse
from validator.db.sql.transfers import get_balance_events_by_coldkey
from validator.db.sql.transfers import get_balance_events_by_tournament
from validator.db.sql.transfers import get_coldkey_balance_by_address
from validator.db.sql.transfers import refund_tournament_participants
from validator.tournament import constants as t_cst
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def get_coldkey_balance(coldkey: str, config: Config = Depends(get_config)) -> ColdkeyBalanceResponse:
    """
    Get balance information for a specific coldkey address

    Args:
        coldkey: Coldkey SS58 address
        config: Validator configuration

    Returns:
        ColdkeyBalanceResponse: Balance information for the coldkey

    Raises:
        HTTPException: If coldkey not found
    """
    logger.info(f"Getting balance for coldkey: {coldkey}")

    balance = await get_coldkey_balance_by_address(config.psql_db, coldkey)

    if not balance:
        raise HTTPException(status_code=404, detail=f"Coldkey balance not found for address: {coldkey}")

    response = ColdkeyBalanceResponse(
        coldkey=balance.coldkey,
        balance_rao=balance.balance_rao,
        total_sent_rao=balance.total_sent_rao,
        transfer_count=balance.transfer_count,
        last_transfer_at=balance.last_transfer_at,
        created_at=balance.created_at,
        updated_at=balance.updated_at,
    )

    logger.info(f"Returning balance for {coldkey}: {balance.balance_rao:,} RAO")
    return response


async def get_tournament_fees() -> TournamentFeesResponse:
    """
    Get tournament participation fees for all tournament types

    Returns:
        TournamentFeesResponse: Fee information for text, image, and environment tournaments
    """
    logger.info("Getting tournament participation fees")

    response = TournamentFeesResponse(
        text_tournament_fee_rao=t_cst.TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO,
        image_tournament_fee_rao=t_cst.TOURNAMENT_IMAGE_PARTICIPATION_FEE_RAO,
        environment_tournament_fee_rao=t_cst.TOURNAMENT_ENVIRONMENT_PARTICIPATION_FEE_RAO,
    )

    logger.info(
        f"Returning tournament fees - Text: {response.text_tournament_fee_rao:,} RAO, "
        f"Image: {response.image_tournament_fee_rao:,} RAO, "
        f"Environment: {response.environment_tournament_fee_rao:,} RAO"
    )
    return response


async def get_balance_events_by_coldkey_endpoint(
    coldkey: str, config: Config = Depends(get_config), limit: int = 100
) -> List[BalanceEventResponse]:
    """
    Get balance events for a specific coldkey

    Args:
        coldkey: Coldkey SS58 address
        config: Validator configuration
        limit: Maximum number of events to return

    Returns:
        List of BalanceEventResponse objects
    """
    logger.info(f"Getting balance events for coldkey: {coldkey}")

    events = await get_balance_events_by_coldkey(config.psql_db, coldkey, limit)

    response = [
        BalanceEventResponse(
            id=event.id,
            tournament_id=event.tournament_id,
            coldkey=event.coldkey,
            event_type=event.event_type,
            amount_rao=event.amount_rao,
            description=event.description,
            created_at=event.created_at,
            updated_at=event.updated_at,
        )
        for event in events
    ]

    logger.info(f"Returning {len(response)} balance events for {coldkey}")
    return response


async def get_balance_events_by_tournament_endpoint(
    tournament_id: str, config: Config = Depends(get_config), limit: int = 100
) -> List[BalanceEventResponse]:
    """
    Get balance events for a specific tournament

    Args:
        tournament_id: Tournament ID
        config: Validator configuration
        limit: Maximum number of events to return

    Returns:
        List of BalanceEventResponse objects
    """
    logger.info(f"Getting balance events for tournament: {tournament_id}")

    events = await get_balance_events_by_tournament(config.psql_db, tournament_id, limit)

    response = [
        BalanceEventResponse(
            id=event.id,
            tournament_id=event.tournament_id,
            coldkey=event.coldkey,
            event_type=event.event_type,
            amount_rao=event.amount_rao,
            description=event.description,
            created_at=event.created_at,
            updated_at=event.updated_at,
        )
        for event in events
    ]

    logger.info(f"Returning {len(response)} balance events for tournament {tournament_id}")
    return response


async def refund_tournament_endpoint(tournament_id: str, config: Config = Depends(get_config)) -> dict:
    """
    Refund all participation fees for a tournament

    Args:
        tournament_id: Tournament ID to refund
        config: Validator configuration

    Returns:
        Dictionary with refund results
    """
    logger.info(f"Processing refunds for tournament: {tournament_id}")

    refund_count = await refund_tournament_participants(config.psql_db, tournament_id)

    response = {
        "tournament_id": tournament_id,
        "refunds_processed": refund_count,
        "message": f"Processed {refund_count} refunds for tournament {tournament_id}",
    }

    logger.info(f"Refunded {refund_count} participants for tournament {tournament_id}")
    return response


def factory_router() -> APIRouter:
    """Factory function to create the transfer balances router"""
    router = APIRouter()

    router.add_api_route(
        "/transfer/balance/{coldkey}",
        get_coldkey_balance,
        methods=["GET"],
        response_model=ColdkeyBalanceResponse,
        summary="Get coldkey balance information",
        description="Get balance and transfer information for a specific coldkey address",
        tags=["transfers"],
    )

    router.add_api_route(
        "/tournament/fees",
        get_tournament_fees,
        methods=["GET"],
        response_model=TournamentFeesResponse,
        summary="Get tournament participation fees",
        description="Get participation fees for text, image, and environment tournaments",
        tags=["tournaments"],
    )

    router.add_api_route(
        "/balance/events/coldkey/{coldkey}",
        get_balance_events_by_coldkey_endpoint,
        methods=["GET"],
        response_model=List[BalanceEventResponse],
        summary="Get balance events for coldkey",
        description="Get all balance events for a specific coldkey address",
        tags=["balance-events"],
    )

    router.add_api_route(
        "/balance/events/tournament/{tournament_id}",
        get_balance_events_by_tournament_endpoint,
        methods=["GET"],
        response_model=List[BalanceEventResponse],
        summary="Get balance events for tournament",
        description="Get all balance events for a specific tournament",
        tags=["balance-events"],
    )

    router.add_api_route(
        "/tournament/{tournament_id}/refund",
        refund_tournament_endpoint,
        methods=["POST"],
        summary="Refund tournament participants",
        description="Refund all participation fees for a tournament",
        tags=["tournaments"],
        dependencies=[Depends(get_api_key)],
        include_in_schema=False,
    )

    return router
