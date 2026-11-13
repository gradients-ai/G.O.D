import math

from fastapi import APIRouter
from fastapi import Depends

import validator.scoring.constants as cts
from validator.app.config import Config
from validator.app.dependencies import get_config
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.scoring.emission_balance import get_dynamic_base_weights
from validator.scoring.emission_balance import max_weight_by_type
from validator.scoring.tournaments import get_quant_tournament_weights_from_data
from validator.scoring.tournaments import get_tournament_weights_from_data
from validator.scoring.weights import build_tournament_audit_data
from validator.scoring.weights import get_tournament_burn_details
from validator.tournament.models import BossBattleResponse
from validator.tournament.models import MultiWeightProjectionResponse
from validator.tournament.models import TaskPerformanceDifference
from validator.tournament.models import TournamentBurnData
from validator.tournament.models import TournamentProjection
from validator.tournament.models import TournamentType
from validator.tournament.models import TournamentWeightsResponse
from validator.tournament.models import WeightProjectionResponse
from validator.tournament.performance_calculator import calculate_boss_round_performance_differences
from validator.tournament.performance_utils import calculate_scaled_weights
from validator.tournament.performance_utils import calculate_tournament_projection
from validator.tournament.performance_utils import get_top_ranked_miners
from validator.tournament.round_results import get_real_tournament_winner


router = APIRouter(tags=["Performance Data"])


def _without_quant_boost(projection: TournamentProjection) -> TournamentProjection:
    # The quant champion gets no boost (the weight itself is forced to base in calculate_tournament_weight);
    # keep the reported boost in line with it.
    return projection.model_copy(update={"emission_boost": 0.0})


def _json_safe_differences(differences: list[TaskPerformanceDifference]) -> list[TaskPerformanceDifference]:
    """A failed-but-scored entry has an infinite loss, and inf/nan are not valid JSON: report those scores as null."""
    return [
        row.model_copy(
            update={
                "boss_score": row.boss_score if row.boss_score is None or math.isfinite(row.boss_score) else None,
                "challenger_score": (
                    row.challenger_score if row.challenger_score is None or math.isfinite(row.challenger_score) else None
                ),
                "performance_difference": (
                    row.performance_difference
                    if row.performance_difference is None or math.isfinite(row.performance_difference)
                    else None
                ),
            }
        )
        for row in differences
    ]


@router.get("/v1/performance/latest-tournament-weights")
async def get_latest_tournament_weights(config: Config = Depends(get_config)) -> TournamentWeightsResponse:
    burn_data: TournamentBurnData = await get_tournament_burn_details(config.psql_db)

    tournament_audit_data = await build_tournament_audit_data(config.psql_db)

    text_tournament_weights, image_tournament_weights = get_tournament_weights_from_data(
        tournament_audit_data.text_tournament_data,
        tournament_audit_data.image_tournament_data,
    )

    text_base_winner_hotkey = None
    if tournament_audit_data.text_tournament_data:
        text_base_winner_hotkey = tournament_audit_data.text_tournament_data.base_winner_hotkey

    image_base_winner_hotkey = None
    if tournament_audit_data.image_tournament_data:
        image_base_winner_hotkey = tournament_audit_data.image_tournament_data.base_winner_hotkey

    quant_tournament_weights = get_quant_tournament_weights_from_data(tournament_audit_data.quant_tournament_data)
    quant_base_winner_hotkey = None
    if tournament_audit_data.quant_tournament_data:
        quant_base_winner_hotkey = tournament_audit_data.quant_tournament_data.base_winner_hotkey

    (
        scaled_text_tournament_weight,
        scaled_text_base_weight,
        scaled_image_tournament_weight,
        scaled_image_base_weight,
        scaled_burn_weight,
        text_winner_hotkey,
        image_winner_hotkey,
    ) = calculate_scaled_weights(tournament_audit_data)

    participation_total = len(tournament_audit_data.participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT
    scale_factor = 1.0 - participation_total if participation_total > 0 else 1.0
    scaled_quant_tournament_weight = tournament_audit_data.quant_tournament_weight * scale_factor
    scaled_quant_base_weight = tournament_audit_data.quant_base_weight * scale_factor
    quant_winner_hotkey = get_real_tournament_winner(tournament_audit_data.quant_tournament_data)

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

    quant_distributed = 0.0
    for hotkey, weight in quant_tournament_weights.items():
        if hotkey == quant_winner_hotkey:
            quant_distributed += weight * scaled_quant_tournament_weight
        else:
            quant_distributed += weight * scaled_quant_base_weight

    text_undistributed = scaled_text_tournament_weight - text_distributed
    image_undistributed = scaled_image_tournament_weight - image_distributed
    quant_undistributed = scaled_quant_tournament_weight - quant_distributed
    total_undistributed = text_undistributed + image_undistributed + quant_undistributed

    total_burn_weight = scaled_burn_weight + total_undistributed

    adjusted_burn_data = TournamentBurnData(
        text_performance_diff=burn_data.text_performance_diff,
        image_performance_diff=burn_data.image_performance_diff,
        text_burn_proportion=burn_data.text_burn_proportion,
        image_burn_proportion=burn_data.image_burn_proportion,
        text_tournament_weight=burn_data.text_tournament_weight,
        image_tournament_weight=burn_data.image_tournament_weight,
        burn_weight=total_burn_weight,
        quant_performance_diff=burn_data.quant_performance_diff,
        quant_burn_proportion=burn_data.quant_burn_proportion,
        quant_tournament_weight=burn_data.quant_tournament_weight,
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
    quant_top_miners = get_top_ranked_miners(
        quant_tournament_weights,
        quant_base_winner_hotkey,
        limit=5,
        scaled_tournament_weight=scaled_quant_tournament_weight,
        scaled_base_weight=scaled_quant_base_weight,
        winner_hotkey=quant_winner_hotkey,
    )

    return TournamentWeightsResponse(
        burn_data=adjusted_burn_data,
        text_top_miners=text_top_miners,
        image_top_miners=image_top_miners,
        quant_top_miners=quant_top_miners,
    )


@router.get("/v1/performance/weight-projection")
async def get_weight_projection(
    percentage_improvement: float,
    config: Config = Depends(get_config),
) -> WeightProjectionResponse:
    # Project against the live participation-driven split rather than the static anchors, so a
    # miner choosing a track sees the base they would actually compete for.
    base_weights = await get_dynamic_base_weights(config.psql_db)
    max_weights = max_weight_by_type()

    text_projection = await calculate_tournament_projection(
        config.psql_db,
        TournamentType.TEXT,
        percentage_improvement,
        base_weights[TournamentType.TEXT],
        max_weights[TournamentType.TEXT],
    )

    quant_projection = await calculate_tournament_projection(
        config.psql_db,
        TournamentType.QUANT,
        percentage_improvement,
        base_weights[TournamentType.QUANT],
        max_weights[TournamentType.QUANT],
    )

    image_projection = await calculate_tournament_projection(
        config.psql_db,
        TournamentType.IMAGE,
        percentage_improvement,
        base_weights[TournamentType.IMAGE],
        max_weights[TournamentType.IMAGE],
    )

    return WeightProjectionResponse(
        percentage_improvement=percentage_improvement,
        text_projection=text_projection,
        image_projection=image_projection,
        quant_projection=_without_quant_boost(quant_projection),
    )


@router.get("/v1/performance/weight-projection-static")
async def get_weight_projection_static(
    config: Config = Depends(get_config),
) -> MultiWeightProjectionResponse:
    percentage_improvements = [5.0, 10.0, 15.0, 20.0]

    # Resolved once: the split is the same for every projected improvement level.
    base_weights = await get_dynamic_base_weights(config.psql_db)
    max_weights = max_weight_by_type()

    projections = []
    for percentage_improvement in percentage_improvements:
        text_projection = await calculate_tournament_projection(
            config.psql_db,
            TournamentType.TEXT,
            percentage_improvement,
            base_weights[TournamentType.TEXT],
            max_weights[TournamentType.TEXT],
        )

        quant_projection = await calculate_tournament_projection(
            config.psql_db,
            TournamentType.QUANT,
            percentage_improvement,
            base_weights[TournamentType.QUANT],
            max_weights[TournamentType.QUANT],
        )

        image_projection = await calculate_tournament_projection(
            config.psql_db,
            TournamentType.IMAGE,
            percentage_improvement,
            base_weights[TournamentType.IMAGE],
            max_weights[TournamentType.IMAGE],
        )

        projections.append(
            WeightProjectionResponse(
                percentage_improvement=percentage_improvement,
                text_projection=text_projection,
                image_projection=image_projection,
                quant_projection=_without_quant_boost(quant_projection),
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
    latest_quant_tournament = await get_latest_completed_tournament(config.psql_db, TournamentType.QUANT)

    # Calculate boss round performance differences
    text_performance_differences = []
    text_tournament_id = None
    if latest_text_tournament:
        text_tournament_id = latest_text_tournament.tournament_id
        text_performance_differences = _json_safe_differences(
            await calculate_boss_round_performance_differences(text_tournament_id, config.psql_db)
        )

    image_performance_differences = []
    image_tournament_id = None
    if latest_image_tournament:
        image_tournament_id = latest_image_tournament.tournament_id
        image_performance_differences = _json_safe_differences(
            await calculate_boss_round_performance_differences(image_tournament_id, config.psql_db)
        )

    quant_performance_differences = []
    quant_tournament_id = None
    if latest_quant_tournament:
        quant_tournament_id = latest_quant_tournament.tournament_id
        quant_performance_differences = _json_safe_differences(
            await calculate_boss_round_performance_differences(quant_tournament_id, config.psql_db)
        )

    return BossBattleResponse(
        text_tournament_id=text_tournament_id,
        text_performance_differences=text_performance_differences,
        image_tournament_id=image_tournament_id,
        image_performance_differences=image_performance_differences,
        quant_tournament_id=quant_tournament_id,
        quant_performance_differences=quant_performance_differences,
    )


def factory_router():
    return router
