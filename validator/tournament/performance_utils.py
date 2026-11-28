import validator.scoring.constants as cts
from validator.db.sql.tournaments import count_champion_defences
from validator.db.sql.tournaments import get_active_tournament_participants
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_participants
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.tournaments import first_place_share
from validator.scoring.tournaments import place_share
from validator.tournament import constants as t_cst
from validator.tournament.models import MinerEmissionWeight
from validator.tournament.models import TournamentAuditData
from validator.tournament.models import TournamentProjection
from validator.tournament.models import TournamentType
from validator.tournament.models import WeightProjection
from validator.tournament.round_results import get_real_tournament_winner


def calculate_scaled_weights(
    tournament_audit_data: TournamentAuditData,
) -> tuple[float, float, float, str | None, str | None]:
    """
    Calculate scaled pools and winner hotkeys from tournament audit data.
    Uses the same logic as get_node_weights_from_tournament_audit_data in weights.py.

    Returns:
        Tuple of (scaled_text_tournament_weight, scaled_image_tournament_weight,
                 scaled_burn_weight, text_winner_hotkey, image_winner_hotkey)
    """
    participants: list[str] = tournament_audit_data.participants
    participation_total: float = len(participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT
    scale_factor: float = 1.0 - participation_total if participation_total > 0 else 1.0

    scaled_text_tournament_weight: float = tournament_audit_data.text_tournament_weight * scale_factor
    scaled_image_tournament_weight: float = tournament_audit_data.image_tournament_weight * scale_factor
    scaled_burn_weight: float = tournament_audit_data.burn_weight * scale_factor

    text_winner_hotkey = get_real_tournament_winner(tournament_audit_data.text_tournament_data)
    image_winner_hotkey = get_real_tournament_winner(tournament_audit_data.image_tournament_data)

    return (
        scaled_text_tournament_weight,
        scaled_image_tournament_weight,
        scaled_burn_weight,
        text_winner_hotkey,
        image_winner_hotkey,
    )


def get_top_ranked_miners(
    weights: dict[str, float],
    base_winner_hotkey: str | None = None,
    limit: int = 5,
    scaled_tournament_weight: float | None = None,
    winner_hotkey: str | None = None,
) -> list[MinerEmissionWeight]:
    real_hotkey_weights = {}
    for hotkey, base_weight in weights.items():
        if hotkey == EMISSION_BURN_HOTKEY and base_winner_hotkey:
            real_hotkey = base_winner_hotkey
        else:
            real_hotkey = hotkey

        if scaled_tournament_weight is not None:
            final_weight = base_weight * scaled_tournament_weight
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
) -> TournamentProjection:
    """
    Project a challenger's emission for a tournament type.

    The input is a percentage score improvement over the boss.

    A challenger only becomes champion if they exceed the dethrone threshold;
    below it the boss defends and the challenger projects as the 2nd-place
    runner-up (a base-pool share, earning only until the next tournament's results
    replace the standings). A new champion starts on the crown share and, assuming it
    defends every weekly tournament, follows first_place_share down as its defences
    add up; the projected weight is the one in force at the end of each horizon.
    """
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)

    # Determine the challenger's performance diff and whether it dethrones the boss.
    performance_diff = percentage_improvement / 100.0
    # Match the boss-round crowning margin so the projection agrees with reality.
    dethrone_threshold = t_cst.BOSS_ROUND_WIN_MARGIN
    dethrones = performance_diff > dethrone_threshold

    # Tournament-internal share by rank and the participation scale factor.
    num_participants = 0
    current_defences = 0
    if latest_tournament:
        participants = await get_tournament_participants(latest_tournament.tournament_id, psql_db)
        num_participants = len(participants)
        current_defences = await count_champion_defences(psql_db, latest_tournament.tournament_id)
    paid_ranks = min(max(num_participants, 1), cts.TOURNAMENT_PAID_RANKS)

    active_participants = await get_active_tournament_participants(psql_db)
    participation_total = len(active_participants) * cts.TOURNAMENT_PARTICIPATION_WEIGHT
    scale_factor = 1.0 - participation_total if participation_total > 0 else 1.0

    projection_days = [7, 30, 90, 180]

    if dethrones:
        # The crown tournament is defence 0; each weekly tournament defended after it is one more.
        def champion_weight(day: int) -> float:
            return place_share(paid_ranks, 1, first_place_share(day // 7)) * base_weight * scale_factor

        initial_weight = champion_weight(0)

        projections = [
            WeightProjection(
                days=days,
                weight=champion_weight(days - 1),
                total_alpha=sum(cts.DAILY_ALPHA_TO_MINERS * champion_weight(day) for day in range(days)),
            )
            for days in projection_days
        ]

        placement = "champion"
    else:
        # Below the dethrone threshold the boss defends; the challenger places 2nd
        # and earns the runner-up share of the base pool only
        # until the next tournament's results replace the standings.
        # The boss defends once more, so 1st takes the share of its next defence.
        runner_up_share = place_share(paid_ranks, 2, first_place_share(current_defences + 1))
        runner_up_weight = runner_up_share * base_weight * scale_factor
        initial_weight = runner_up_weight

        projections = [
            WeightProjection(
                days=days,
                weight=runner_up_weight if days <= t_cst.RUNNER_UP_EMISSION_DAYS else 0.0,
                total_alpha=min(days, t_cst.RUNNER_UP_EMISSION_DAYS) * cts.DAILY_ALPHA_TO_MINERS * runner_up_weight,
            )
            for days in projection_days
        ]

        placement = "runner_up"

    return TournamentProjection(
        tournament_type=tournament_type.value,
        current_champion_decay=0.0,  # no more time decay; kept for API stability
        initial_weight=initial_weight,
        projections=projections,
        placement=placement,
        dethrone_threshold=dethrone_threshold,
    )
