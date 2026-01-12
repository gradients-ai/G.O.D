from datetime import datetime
from datetime import timezone

import validator.core.constants as cts
from core.models.tournament_models import MinerEmissionWeight
from core.models.tournament_models import TournamentAuditData
from core.models.tournament_models import TournamentProjection
from core.models.tournament_models import TournamentType
from core.models.tournament_models import WeightProjection
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.core.weight_setting import calculate_emission_boost_from_perf
from validator.core.weight_setting import calculate_hybrid_decays
from validator.core.weight_setting import calculate_tournament_weight_with_decay
from validator.db.sql.tournaments import count_champion_consecutive_wins
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_where_champion_first_won
from validator.tournament.utils import get_real_tournament_winner


def calculate_scaled_weights(
    tournament_audit_data: TournamentAuditData,
) -> tuple[float, float, float, float, float, str | None, str | None]:
    """
    Calculate scaled weights and winner hotkeys from tournament audit data.
    Uses the same logic as get_node_weights_from_tournament_audit_data in weight_setting.py.

    Returns:
        Tuple of (scaled_text_tournament_weight, scaled_text_base_weight,
                 scaled_image_tournament_weight, scaled_image_base_weight,
                 scaled_burn_weight, text_winner_hotkey, image_winner_hotkey)
    """
    participants: list[str] = tournament_audit_data.participants
    participation_total: float = len(participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT
    scale_factor: float = 1.0 - participation_total if participation_total > 0 else 1.0

    scaled_text_tournament_weight: float = tournament_audit_data.text_tournament_weight * scale_factor
    scaled_image_tournament_weight: float = tournament_audit_data.image_tournament_weight * scale_factor
    scaled_burn_weight: float = tournament_audit_data.burn_weight * scale_factor

    scaled_text_base_weight: float = cts.TOURNAMENT_TEXT_WEIGHT * scale_factor
    scaled_image_base_weight: float = cts.TOURNAMENT_IMAGE_WEIGHT * scale_factor

    text_winner_hotkey = get_real_tournament_winner(tournament_audit_data.text_tournament_data)
    image_winner_hotkey = get_real_tournament_winner(tournament_audit_data.image_tournament_data)

    return (
        scaled_text_tournament_weight,
        scaled_text_base_weight,
        scaled_image_tournament_weight,
        scaled_image_base_weight,
        scaled_burn_weight,
        text_winner_hotkey,
        image_winner_hotkey,
    )


def get_top_ranked_miners(
    weights: dict[str, float],
    base_winner_hotkey: str | None = None,
    limit: int = 5,
    scaled_tournament_weight: float | None = None,
    scaled_base_weight: float | None = None,
    winner_hotkey: str | None = None,
) -> list[MinerEmissionWeight]:
    real_hotkey_weights = {}
    for hotkey, base_weight in weights.items():
        if hotkey == EMISSION_BURN_HOTKEY and base_winner_hotkey:
            real_hotkey = base_winner_hotkey
        else:
            real_hotkey = hotkey

        if scaled_tournament_weight is not None and scaled_base_weight is not None:
            if real_hotkey == winner_hotkey:
                final_weight = base_weight * scaled_tournament_weight
            else:
                final_weight = base_weight * scaled_base_weight
        else:
            final_weight = base_weight
        
        real_hotkey_weights[real_hotkey] = final_weight

    sorted_miners = sorted(real_hotkey_weights.items(), key=lambda x: x[1], reverse=True)[:limit]

    return [
        MinerEmissionWeight(hotkey=hotkey, rank=idx + 1, weight=weight) for idx, (hotkey, weight) in enumerate(sorted_miners)
    ]


async def calculate_innovation_incentive_for_new_champion(
    psql_db,
    tournament_type: TournamentType,
    base_weight: float,
    max_weight: float,
) -> float:
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    current_champion = get_real_tournament_winner(latest_tournament) if latest_tournament else None

    if not current_champion or not latest_tournament:
        return 0.0

    current_perf_diff = latest_tournament.winning_performance_difference
    if current_perf_diff is None:
        current_perf_diff = 0.0

    current_emission_boost = calculate_emission_boost_from_perf(current_perf_diff)
    consecutive_wins = await count_champion_consecutive_wins(psql_db, tournament_type, current_champion)
    first_win_tournament = await get_tournament_where_champion_first_won(psql_db, tournament_type, current_champion)

    if not first_win_tournament or not first_win_tournament.updated_at:
        return 0.0

    current_old_decay, current_new_decay, current_apply_hybrid = calculate_hybrid_decays(
        first_championship_time=first_win_tournament.updated_at,
        consecutive_wins=consecutive_wins,
        current_time=latest_tournament.updated_at if latest_tournament.updated_at else None,
    )

    current_final_weight = calculate_tournament_weight_with_decay(
        tournament_type=tournament_type,
        base_weight=base_weight,
        emission_boost=current_emission_boost,
        old_decay=current_old_decay,
        new_decay=current_new_decay,
        apply_hybrid=current_apply_hybrid,
        max_weight=max_weight,
    )

    innovation_incentive = max(0.0, base_weight - current_final_weight)
    return innovation_incentive


async def calculate_tournament_projection(
    psql_db,
    tournament_type: TournamentType,
    percentage_improvement: float,
    base_weight: float,
    max_weight: float,
) -> TournamentProjection:
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    current_champion = get_real_tournament_winner(latest_tournament) if latest_tournament else None

    current_champion_decay = 0.0
    if current_champion and latest_tournament:
        consecutive_wins = await count_champion_consecutive_wins(psql_db, tournament_type, current_champion)
        first_win_tournament = await get_tournament_where_champion_first_won(psql_db, tournament_type, current_champion)
        if first_win_tournament and first_win_tournament.updated_at:
            _, new_decay, _ = calculate_hybrid_decays(
                first_win_tournament.updated_at,
                consecutive_wins,
                datetime.now(timezone.utc),
            )
            current_champion_decay = new_decay

    performance_diff = percentage_improvement / 100.0
    emission_boost_from_perf = calculate_emission_boost_from_perf(performance_diff)
    innovation_incentive = await calculate_innovation_incentive_for_new_champion(
        psql_db, tournament_type, base_weight, max_weight
    )
    total_emission_boost = emission_boost_from_perf + innovation_incentive

    initial_weight = calculate_tournament_weight_with_decay(
        tournament_type=tournament_type,
        base_weight=base_weight,
        emission_boost=total_emission_boost,
        old_decay=0.0,
        new_decay=0.0,
        apply_hybrid=False,
        max_weight=max_weight,
    )

    projection_days = [1, 7, 30, 90]

    projections = []
    for days in projection_days:
        new_decay = days * cts.EMISSION_DAILY_TIME_DECAY_RATE

        future_weight = calculate_tournament_weight_with_decay(
            tournament_type=tournament_type,
            base_weight=base_weight,
            emission_boost=total_emission_boost,
            old_decay=0.0,
            new_decay=new_decay,
            apply_hybrid=False,
            max_weight=max_weight,
        )

        cumulative_alpha = days * cts.DAILY_ALPHA_TO_MINERS * (initial_weight + future_weight) / 2.0

        projections.append(WeightProjection(days=days, weight=future_weight, total_alpha=cumulative_alpha))

    return TournamentProjection(
        tournament_type=tournament_type.value,
        current_champion_decay=current_champion_decay,
        initial_weight=initial_weight,
        projections=projections,
    )

