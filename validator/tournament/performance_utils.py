from datetime import datetime
from datetime import timezone

import validator.scoring.constants as cts
from validator.db.sql.tournaments import count_champion_consecutive_wins
from validator.db.sql.tournaments import get_active_tournament_participants
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_participants
from validator.db.sql.tournaments import get_tournament_where_champion_first_won
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.tournaments import exponential_decline_mapping
from validator.scoring.weights import calculate_emission_boost_from_perf
from validator.scoring.weights import calculate_env_perf_diff_from_win_pct
from validator.scoring.weights import calculate_hybrid_decays
from validator.scoring.weights import calculate_tournament_weight_with_decay
from validator.scoring.weights import emission_time_decay_fraction
from validator.tournament import constants as t_cst
from validator.tournament.models import MinerEmissionWeight
from validator.tournament.models import TournamentAuditData
from validator.tournament.models import TournamentProjection
from validator.tournament.models import TournamentType
from validator.tournament.models import WeightProjection
from validator.tournament.round_results import get_real_tournament_winner


def calculate_scaled_weights(
    tournament_audit_data: TournamentAuditData,
) -> tuple[float, float, float, float, float, float, float, str | None, str | None, str | None]:
    """
    Calculate scaled weights and winner hotkeys from tournament audit data.
    Uses the same logic as get_node_weights_from_tournament_audit_data in weight_setting.py.

    Returns:
        Tuple of (scaled_text_tournament_weight, scaled_text_base_weight,
                 scaled_image_tournament_weight, scaled_image_base_weight,
                 scaled_environment_tournament_weight, scaled_environment_base_weight,
                 scaled_burn_weight, text_winner_hotkey, image_winner_hotkey, environment_winner_hotkey)
    """
    participants: list[str] = tournament_audit_data.participants
    participation_total: float = len(participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT
    scale_factor: float = 1.0 - participation_total if participation_total > 0 else 1.0

    scaled_text_tournament_weight: float = tournament_audit_data.text_tournament_weight * scale_factor
    scaled_image_tournament_weight: float = tournament_audit_data.image_tournament_weight * scale_factor
    scaled_environment_tournament_weight: float = tournament_audit_data.environment_tournament_weight * scale_factor
    scaled_burn_weight: float = tournament_audit_data.burn_weight * scale_factor

    scaled_text_base_weight: float = cts.TOURNAMENT_TEXT_WEIGHT * scale_factor
    scaled_image_base_weight: float = cts.TOURNAMENT_IMAGE_WEIGHT * scale_factor
    scaled_environment_base_weight: float = cts.TOURNAMENT_ENVIRONMENT_WEIGHT * scale_factor

    text_winner_hotkey = get_real_tournament_winner(tournament_audit_data.text_tournament_data)
    image_winner_hotkey = get_real_tournament_winner(tournament_audit_data.image_tournament_data)
    environment_winner_hotkey = get_real_tournament_winner(tournament_audit_data.environment_tournament_data)

    return (
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

    all_sorted_miners = sorted(real_hotkey_weights.items(), key=lambda x: x[1], reverse=True)

    if winner_hotkey and winner_hotkey in real_hotkey_weights:
        winner_weight = real_hotkey_weights[winner_hotkey]
        rest = [(hk, w) for hk, w in all_sorted_miners if hk != winner_hotkey]
        sorted_miners = [(winner_hotkey, winner_weight)] + rest[: limit - 1]
    else:
        sorted_miners = all_sorted_miners[:limit]

    return [
        MinerEmissionWeight(hotkey=hotkey, rank=idx + 1, weight=weight)
        for idx, (hotkey, weight) in enumerate(sorted_miners)
    ]


async def calculate_tournament_projection(
    psql_db,
    tournament_type: TournamentType,
    percentage_improvement: float,
    base_weight: float,
    max_weight: float,
    win_pct: float | None = None,
) -> TournamentProjection:
    """
    Project a challenger's emission for a tournament type.

    For TEXT/IMAGE the input is a percentage score improvement over the boss.
    For ENVIRONMENT the input is a win rate (``win_pct``, 0.0-1.0) mapped through
    the PvP emission curve.

    A challenger only becomes champion if they exceed the dethrone threshold;
    below it the boss defends and the challenger projects as the 2nd-place
    runner-up (base-pool share, no champion boost, no time decay).
    """
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    current_champion = get_real_tournament_winner(latest_tournament) if latest_tournament else None

    current_champion_decay = 0.0
    consecutive_wins = 0
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

    # Determine the challenger's performance diff and whether it dethrones the boss.
    if tournament_type == TournamentType.ENVIRONMENT:
        effective_win_pct = win_pct if win_pct is not None else percentage_improvement / 100.0
        performance_diff = calculate_env_perf_diff_from_win_pct(effective_win_pct)
        dethrone_threshold = cts.PVP_WIN_PCT_THRESHOLD
        dethrones = effective_win_pct >= cts.PVP_WIN_PCT_THRESHOLD
    else:
        performance_diff = percentage_improvement / 100.0
        # Match the boss-round crowning margin so the projection agrees with reality.
        dethrone_threshold = t_cst.BOSS_ROUND_WIN_MARGIN
        dethrones = performance_diff > dethrone_threshold

    # Tournament-internal share by rank and the participation scale factor.
    num_participants = 0
    if latest_tournament:
        participants = await get_tournament_participants(latest_tournament.tournament_id, psql_db)
        num_participants = len(participants)
    winner_share = exponential_decline_mapping(max(num_participants, 1), 1)

    active_participants = await get_active_tournament_participants(psql_db)
    participation_total = len(active_participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT
    scale_factor = 1.0 - participation_total if participation_total > 0 else 1.0

    projection_days = [7, 30, 90, 180]

    if dethrones:
        emission_boost = calculate_emission_boost_from_perf(performance_diff)

        raw_initial_weight = calculate_tournament_weight_with_decay(
            tournament_type=tournament_type,
            base_weight=base_weight,
            emission_boost=emission_boost,
            old_decay=0.0,
            new_decay=0.0,
            apply_hybrid=False,
            max_weight=max_weight,
        )
        # Winner's actual emission weight = winner_share * tournament_weight * scale_factor
        initial_weight = winner_share * raw_initial_weight * scale_factor

        projections = []
        for days in projection_days:
            new_decay = emission_time_decay_fraction(days)

            raw_future_weight = calculate_tournament_weight_with_decay(
                tournament_type=tournament_type,
                base_weight=base_weight,
                emission_boost=emission_boost,
                old_decay=0.0,
                new_decay=new_decay,
                apply_hybrid=False,
                max_weight=max_weight,
            )

            weight = winner_share * raw_future_weight * scale_factor
            cumulative_alpha = days * cts.DAILY_ALPHA_TO_MINERS * (initial_weight + weight) / 2.0

            projections.append(WeightProjection(days=days, weight=weight, total_alpha=cumulative_alpha))

        placement = "champion"
    else:
        # Below the dethrone threshold the boss defends; the challenger places 2nd
        # and earns the runner-up share of the base pool (no champion boost, no decay).
        emission_boost = 0.0
        runner_up_share = exponential_decline_mapping(max(num_participants, 1), 2)
        runner_up_weight = runner_up_share * base_weight * scale_factor
        initial_weight = runner_up_weight

        projections = [
            WeightProjection(
                days=days,
                weight=runner_up_weight,
                total_alpha=days * cts.DAILY_ALPHA_TO_MINERS * runner_up_weight,
            )
            for days in projection_days
        ]

        placement = "runner_up"

    return TournamentProjection(
        tournament_type=tournament_type.value,
        current_champion_decay=current_champion_decay,
        initial_weight=initial_weight,
        projections=projections,
        placement=placement,
        dethrone_threshold=dethrone_threshold,
        emission_boost=emission_boost,
    )
