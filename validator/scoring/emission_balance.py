"""
Participation-driven emission split: guaranteed minimums plus a balanced flex pool.

Each type gets a fixed guaranteed minimum (cts.TOURNAMENT_*_WEIGHT). Only the remaining
cts.EMISSION_FLEX_POOL moves with participation: it is split from how many miners actually
entered the last few tournaments of each type, so a type that is oversubscribed relative to its
minimums' ratio cedes flex to a starved one, driving miners toward the under-contested modality.
The swing is therefore bounded by construction: a type's base always lies in
[minimum, minimum + flex], and the bases always sum to exactly 1.0.

The single knob is cts.EMISSION_BALANCE_ALPHA — the rate of change: 0.0 splits the flex pool at
the minimums' ratio, larger values react harder to a given imbalance.

    flex_shareᵢ ∝ minimum_shareᵢ · (minimum_shareᵢ / entrant_shareᵢ) ** alpha
    wᵢ = minimumᵢ + EMISSION_FLEX_POOL · flex_shareᵢ
"""

from statistics import mean

import validator.scoring.constants as cts
from core.logging import get_logger
from core.models.tournament_models import TournamentType
from validator.db.sql.tournaments import get_recent_completed_entrant_counts


logger = get_logger(__name__)

_TYPES: tuple[TournamentType, ...] = (TournamentType.TEXT, TournamentType.IMAGE)


def _minimums() -> dict[TournamentType, float]:
    return {
        TournamentType.TEXT: cts.TOURNAMENT_TEXT_WEIGHT,
        TournamentType.IMAGE: cts.TOURNAMENT_IMAGE_WEIGHT,
    }


def compute_dynamic_base_weights(
    entrant_counts: dict[TournamentType, float | None],
) -> dict[TournamentType, float]:
    """Pure allocator: map rolling entrant counts per type to base emission weights.

    A ``None`` (or non-positive) count means that type has no usable history; it is treated as
    average-subscribed so it neither gains nor loses flex. With balanced participation the flex
    pool splits at the minimums' ratio.
    """
    minimum = _minimums()
    minimum_total = sum(minimum.values())
    minimum_share = {t: minimum[t] / minimum_total for t in _TYPES}

    # Each type only needs its own count relative to its minimum share; the shared normaliser
    # cancels out. k = n / minimum_share is a type's "subscription pressure"; missing types
    # inherit the average pressure so they neither gain nor lose.
    pressures_known: dict[TournamentType, float] = {}
    for t in _TYPES:
        n = entrant_counts.get(t)
        if n is not None and n > 0:
            pressures_known[t] = n / minimum_share[t]
    avg_pressure = mean(pressures_known.values()) if pressures_known else 1.0
    pressure = {t: pressures_known.get(t, avg_pressure) for t in _TYPES}

    alpha = cts.EMISSION_BALANCE_ALPHA
    desirability = {t: minimum_share[t] * (1.0 / pressure[t]) ** alpha for t in _TYPES}
    total_desirability = sum(desirability.values())
    return {t: minimum[t] + cts.EMISSION_FLEX_POOL * desirability[t] / total_desirability for t in _TYPES}


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
        "Dynamic emission base split (rolling-%d entrants %s): text=%.4f image=%.4f",
        cts.EMISSION_BALANCE_ROLLING_WINDOW,
        {t.value: counts[t] for t in _TYPES},
        weights[TournamentType.TEXT],
        weights[TournamentType.IMAGE],
    )
    return weights
