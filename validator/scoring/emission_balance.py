"""
Participation-driven emission base-split auto-balancer.

Historically the text / image / env emission bases were hand-picked each week. This module
derives them automatically from how many miners actually entered the last few tournaments of
each type, so the split follows demand: a type that is oversubscribed relative to its anchor
share sheds weight (down to a floor) and hands it to a starved type, driving miners toward the
under-contested modality.

Neutral point is the anchor split (cts.TOURNAMENT_*_WEIGHT). The single knob is
cts.EMISSION_BALANCE_ALPHA — the rate of change: 0.0 reproduces the static anchors, larger
values react harder to a given imbalance.

    wᵢ ∝ anchorᵢ · (anchor_shareᵢ / entrant_shareᵢ) ** alpha
       → floor at EMISSION_BALANCE_FLOOR
       → borrow from burn so the pool may grow up to 1.0
       → cap each type at EMISSION_MAX_*_TOURNAMENT_WEIGHT (uniform 0.50)
"""

from statistics import mean

import validator.scoring.constants as cts
from core.logging import get_logger
from core.models.tournament_models import TournamentType
from validator.db.sql.tournaments import get_recent_completed_entrant_counts


logger = get_logger(__name__)

_TYPES: tuple[TournamentType, ...] = (TournamentType.TEXT, TournamentType.IMAGE, TournamentType.ENVIRONMENT)


def _anchors() -> dict[TournamentType, float]:
    return {
        TournamentType.TEXT: cts.TOURNAMENT_TEXT_WEIGHT,
        TournamentType.IMAGE: cts.TOURNAMENT_IMAGE_WEIGHT,
        TournamentType.ENVIRONMENT: cts.TOURNAMENT_ENVIRONMENT_WEIGHT,
    }


def _cap(tournament_type: TournamentType) -> float:
    return {
        TournamentType.TEXT: cts.MAX_TEXT_TOURNAMENT_WEIGHT,
        TournamentType.IMAGE: cts.MAX_IMAGE_TOURNAMENT_WEIGHT,
        TournamentType.ENVIRONMENT: cts.MAX_ENVIRONMENT_TOURNAMENT_WEIGHT,
    }[tournament_type]


def compute_dynamic_base_weights(
    entrant_counts: dict[TournamentType, float | None],
) -> dict[TournamentType, float]:
    """Pure allocator: map rolling entrant counts per type to base emission weights.

    A ``None`` (or non-positive) count means that type has no usable history; it is treated as
    average-subscribed so it stays at its anchor. With balanced participation every type lands
    exactly on its anchor.
    """
    anchor = _anchors()
    pool = sum(anchor.values())
    anchor_share = {t: anchor[t] / pool for t in _TYPES}

    # Each type only needs its own count relative to its anchor share; the shared normaliser
    # cancels out. k = n / anchor_share is a type's "subscription pressure"; missing types
    # inherit the average pressure so they neither gain nor lose.
    pressures_known: dict[TournamentType, float] = {}
    for t in _TYPES:
        n = entrant_counts.get(t)
        if n is not None and n > 0:
            pressures_known[t] = n / anchor_share[t]
    avg_pressure = mean(pressures_known.values()) if pressures_known else 1.0
    pressure = {t: pressures_known.get(t, avg_pressure) for t in _TYPES}

    alpha = cts.EMISSION_BALANCE_ALPHA
    desirability = {t: anchor[t] * (1.0 / pressure[t]) ** alpha for t in _TYPES}
    total_desirability = sum(desirability.values())
    weights = {t: pool * desirability[t] / total_desirability for t in _TYPES}

    # Floor — raising a type to the floor lets the total grow (borrow from burn), up to 1.0.
    floor = cts.EMISSION_BALANCE_FLOOR
    weights = {t: max(floor, weights[t]) for t in _TYPES}
    total = sum(weights.values())
    if total > 1.0:
        over = total - 1.0
        headroom = {t: weights[t] - floor for t in _TYPES}
        headroom_sum = sum(headroom.values())
        if headroom_sum > 0:
            weights = {t: weights[t] - over * headroom[t] / headroom_sum for t in _TYPES}

    # Uniform hard cap; anything shaved off falls through to burn.
    weights = {t: min(_cap(t), weights[t]) for t in _TYPES}
    return weights


async def get_dynamic_base_weights(psql_db) -> dict[TournamentType, float]:
    """Fetch rolling entrant counts per type and compute the dynamic base split."""
    counts: dict[TournamentType, float | None] = {}
    for tournament_type in _TYPES:
        recent = await get_recent_completed_entrant_counts(
            psql_db,
            tournament_type,
            limit=cts.EMISSION_BALANCE_ROLLING_WINDOW,
            min_entrants=cts.EMISSION_MIN_ENTRANTS_FOR_BALANCE,
        )
        counts[tournament_type] = mean(recent) if recent else None

    weights = compute_dynamic_base_weights(counts)
    logger.info(
        "Dynamic emission base split (rolling-%d entrants %s): text=%.4f image=%.4f env=%.4f",
        cts.EMISSION_BALANCE_ROLLING_WINDOW,
        {t.value: counts[t] for t in _TYPES},
        weights[TournamentType.TEXT],
        weights[TournamentType.IMAGE],
        weights[TournamentType.ENVIRONMENT],
    )
    return weights
