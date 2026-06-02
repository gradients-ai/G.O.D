from fastapi import APIRouter
from fastapi import Depends

import validator.scoring.constants as cts
from validator.app.config import Config
from validator.app.dependencies import get_config
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.scoring.tournaments import get_tournament_weights_from_data
from validator.scoring.weights import build_tournament_audit_data
from validator.scoring.weights import get_tournament_burn_details
from validator.tournament.models import BossBattleResponse
from validator.tournament.models import MultiWeightProjectionResponse
from validator.tournament.models import TournamentBurnData
from validator.tournament.models import TournamentType
from validator.tournament.models import TournamentWeightsResponse
from validator.tournament.models import WeightProjectionResponse
from validator.tournament.performance_calculator import calculate_boss_round_performance_differences
from validator.tournament.performance_utils import calculate_scaled_weights
from validator.tournament.performance_utils import calculate_tournament_projection
from validator.tournament.performance_utils import get_top_ranked_miners


router = APIRouter(tags=["Performance Data"])


@router.get("/v1/performance/latest-tournament-weights")
async def get_latest_tournament_weights(config: Config = Depends(get_config)) -> TournamentWeightsResponse:
    burn_data: TournamentBurnData = await get_tournament_burn_details(config.psql_db)

    tournament_audit_data = await build_tournament_audit_data(config.psql_db)

    text_tournament_weights, image_tournament_weights, environment_tournament_weights = get_tournament_weights_from_data(
        tournament_audit_data.text_tournament_data, 
        tournament_audit_data.image_tournament_data,
        tournament_audit_data.environment_tournament_data
    )

    text_base_winner_hotkey = None
    if tournament_audit_data.text_tournament_data:
        text_base_winner_hotkey = tournament_audit_data.text_tournament_data.base_winner_hotkey

    image_base_winner_hotkey = None
    if tournament_audit_data.image_tournament_data:
        image_base_winner_hotkey = tournament_audit_data.image_tournament_data.base_winner_hotkey

    environment_base_winner_hotkey = None
    if tournament_audit_data.environment_tournament_data:
        environment_base_winner_hotkey = tournament_audit_data.environment_tournament_data.base_winner_hotkey

    (
        scaled_text_tournament_weight,
        scaled_text_base_weight,
        scaled_image_tournament_weight,
        scaled_image_base_weight,
        scaled_environment_tournament_weight,
        scaled_environment_base_weight,
        scaled_burn_weight,
        text_winner_hotkey,
        image_winner_hotkey,
        environment_winner_hotkey,
    ) = calculate_scaled_weights(tournament_audit_data)

    text_distributed = 0.0
    for hotkey, weight in text_tournament_weights.items():
        if hotkey == text_winner_hotkey:
            text_distributed += weight * scaled_text_tournament_weight
        else:
            text_distributed += weight * scaled_text_base_weight

    image_distributed = 0.0
    for hotkey, weight in image_tournament_weights.items():
        if hotkey == image_winner_hotkey:
            image_distributed += weight * scaled_image_tournament_weight
        else:
            image_distributed += weight * scaled_image_base_weight

    environment_distributed = 0.0
    for hotkey, weight in environment_tournament_weights.items():
        if hotkey == environment_winner_hotkey:
            environment_distributed += weight * scaled_environment_tournament_weight
        else:
            environment_distributed += weight * scaled_environment_base_weight

    text_undistributed = scaled_text_tournament_weight - text_distributed
    image_undistributed = scaled_image_tournament_weight - image_distributed
    environment_undistributed = scaled_environment_tournament_weight - environment_distributed
    total_undistributed = text_undistributed + image_undistributed + environment_undistributed

    total_burn_weight = scaled_burn_weight + total_undistributed

    adjusted_burn_data = TournamentBurnData(
        text_performance_diff=burn_data.text_performance_diff,
        image_performance_diff=burn_data.image_performance_diff,
        environment_performance_diff=burn_data.environment_performance_diff,
        text_burn_proportion=burn_data.text_burn_proportion,
        image_burn_proportion=burn_data.image_burn_proportion,
        environment_burn_proportion=burn_data.environment_burn_proportion,
        text_tournament_weight=burn_data.text_tournament_weight,
        image_tournament_weight=burn_data.image_tournament_weight,
        environment_tournament_weight=burn_data.environment_tournament_weight,
        burn_weight=total_burn_weight,
    )

    text_top_miners = get_top_ranked_miners(
        text_tournament_weights,
        text_base_winner_hotkey,
        limit=5,
        scaled_tournament_weight=scaled_text_tournament_weight,
        scaled_base_weight=scaled_text_base_weight,
        winner_hotkey=text_winner_hotkey,
    )
    image_top_miners = get_top_ranked_miners(
        image_tournament_weights,
        image_base_winner_hotkey,
        limit=5,
        scaled_tournament_weight=scaled_image_tournament_weight,
        scaled_base_weight=scaled_image_base_weight,
        winner_hotkey=image_winner_hotkey,
    )
    environment_top_miners = get_top_ranked_miners(
        environment_tournament_weights,
        environment_base_winner_hotkey,
        limit=5,
        scaled_tournament_weight=scaled_environment_tournament_weight,
        scaled_base_weight=scaled_environment_base_weight,
        winner_hotkey=environment_winner_hotkey,
    )

    return TournamentWeightsResponse(
        burn_data=adjusted_burn_data,
        text_top_miners=text_top_miners,
        image_top_miners=image_top_miners,
        environment_top_miners=environment_top_miners,
    )


@router.get("/v1/performance/weight-projection")
async def get_weight_projection(
    percentage_improvement: float,
    environment_win_pct: float = 80.0,
    config: Config = Depends(get_config),
) -> WeightProjectionResponse:
    text_projection = await calculate_tournament_projection(
        config.psql_db,
        TournamentType.TEXT,
        percentage_improvement,
        cts.TOURNAMENT_TEXT_WEIGHT,
        cts.MAX_TEXT_TOURNAMENT_WEIGHT,
    )

    image_projection = await calculate_tournament_projection(
        config.psql_db,
        TournamentType.IMAGE,
        percentage_improvement,
        cts.TOURNAMENT_IMAGE_WEIGHT,
        cts.MAX_IMAGE_TOURNAMENT_WEIGHT,
    )

    # Environment tournaments are PvP: the input is a win rate (%), not a score improvement.
    environment_projection = await calculate_tournament_projection(
        config.psql_db,
        TournamentType.ENVIRONMENT,
        percentage_improvement,
        cts.TOURNAMENT_ENVIRONMENT_WEIGHT,
        cts.MAX_ENVIRONMENT_TOURNAMENT_WEIGHT,
        win_pct=environment_win_pct / 100.0,
    )

    return WeightProjectionResponse(
        percentage_improvement=percentage_improvement,
        text_projection=text_projection,
        image_projection=image_projection,
        environment_projection=environment_projection,
    )


@router.get("/v1/performance/weight-projection-static")
async def get_weight_projection_static(
    config: Config = Depends(get_config),
) -> MultiWeightProjectionResponse:
    percentage_improvements = [5.0, 10.0, 15.0, 20.0]

    projections = []
    for percentage_improvement in percentage_improvements:
        text_projection = await calculate_tournament_projection(
            config.psql_db,
            TournamentType.TEXT,
            percentage_improvement,
            cts.TOURNAMENT_TEXT_WEIGHT,
            cts.MAX_TEXT_TOURNAMENT_WEIGHT,
        )

        image_projection = await calculate_tournament_projection(
            config.psql_db,
            TournamentType.IMAGE,
            percentage_improvement,
            cts.TOURNAMENT_IMAGE_WEIGHT,
            cts.MAX_IMAGE_TOURNAMENT_WEIGHT,
        )

        environment_projection = await calculate_tournament_projection(
            config.psql_db,
            TournamentType.ENVIRONMENT,
            percentage_improvement,
            cts.TOURNAMENT_ENVIRONMENT_WEIGHT,
            cts.MAX_ENVIRONMENT_TOURNAMENT_WEIGHT,
        )

        projections.append(
            WeightProjectionResponse(
                percentage_improvement=percentage_improvement,
                text_projection=text_projection,
                image_projection=image_projection,
                environment_projection=environment_projection,
            )
        )

    return MultiWeightProjectionResponse(projections=projections)


@router.get("/v1/performance/last-boss-battle")
async def get_last_boss_battle(
    config: Config = Depends(get_config),
) -> BossBattleResponse:
    # Get latest completed tournaments
    latest_text_tournament = await get_latest_completed_tournament(config.psql_db, TournamentType.TEXT)
    latest_image_tournament = await get_latest_completed_tournament(config.psql_db, TournamentType.IMAGE)
    latest_environment_tournament = await get_latest_completed_tournament(config.psql_db, TournamentType.ENVIRONMENT)

    # Calculate boss round performance differences
    text_performance_differences = []
    text_tournament_id = None
    if latest_text_tournament:
        text_tournament_id = latest_text_tournament.tournament_id
        text_performance_differences = await calculate_boss_round_performance_differences(text_tournament_id, config.psql_db)

    image_performance_differences = []
    image_tournament_id = None
    if latest_image_tournament:
        image_tournament_id = latest_image_tournament.tournament_id
        image_performance_differences = await calculate_boss_round_performance_differences(image_tournament_id, config.psql_db)

    environment_performance_differences = []
    environment_tournament_id = None
    if latest_environment_tournament:
        environment_tournament_id = latest_environment_tournament.tournament_id
        environment_performance_differences = await calculate_boss_round_performance_differences(
            environment_tournament_id, config.psql_db
        )

    return BossBattleResponse(
        text_tournament_id=text_tournament_id,
        text_performance_differences=text_performance_differences,
        image_tournament_id=image_tournament_id,
        image_performance_differences=image_performance_differences,
        environment_tournament_id=environment_tournament_id,
        environment_performance_differences=environment_performance_differences,
    )


def factory_router():
    return router
