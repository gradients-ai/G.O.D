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

The compression share phases in over cts.QUANT_EMISSION_RAMP_DAYS from cts.QUANT_EMISSION_RAMP_START: its base is
scaled by the ramp factor and the rest goes to text and image in proportion to their minimums. Every type's
minimum and cap are phased the same way, so each base stays within its bounds; at a full ramp the split is
exactly the one above.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from statistics import mean

import validator.scoring.constants as cts
from core.logging import get_logger
from core.models.tournament_models import TournamentType
from validator.db.sql.tournaments import get_recent_completed_entrant_counts


logger = get_logger(__name__)

_TYPES: tuple[TournamentType, ...] = (TournamentType.TEXT, TournamentType.QUANT, TournamentType.IMAGE)


def _minimums() -> dict[TournamentType, float]:
    return {
        TournamentType.TEXT: cts.TOURNAMENT_TEXT_WEIGHT,
        TournamentType.QUANT: cts.TOURNAMENT_QUANT_WEIGHT,
        TournamentType.IMAGE: cts.TOURNAMENT_IMAGE_WEIGHT,
    }


def quant_ramp_factor(now: datetime) -> float:
    """Fraction of its base share compression has earned at ``now``: 0 before the start (or with none set), 1 after."""
    if cts.QUANT_EMISSION_RAMP_START is None:
        return 0.0
    if cts.QUANT_EMISSION_RAMP_START.tzinfo is None:
        raise ValueError("QUANT_EMISSION_RAMP_START must be timezone-aware")
    elapsed = (now - cts.QUANT_EMISSION_RAMP_START) / timedelta(days=cts.QUANT_EMISSION_RAMP_DAYS)
    return min(max(elapsed, 0.0), 1.0)


def _phase_in_quant(by_type: dict[TournamentType, float], ramp: float) -> dict[TournamentType, float]:
    """Scale quant's value by ``ramp`` and hand the rest to text and image in proportion to their minimums."""
    minimum = _minimums()
    unearned = (1.0 - ramp) * by_type[TournamentType.QUANT]
    text_image_minimum = minimum[TournamentType.TEXT] + minimum[TournamentType.IMAGE]
    return {
        TournamentType.TEXT: by_type[TournamentType.TEXT] + unearned * minimum[TournamentType.TEXT] / text_image_minimum,
        TournamentType.QUANT: ramp * by_type[TournamentType.QUANT],
        TournamentType.IMAGE: by_type[TournamentType.IMAGE] + unearned * minimum[TournamentType.IMAGE] / text_image_minimum,
    }


def compute_dynamic_base_weights(
    entrant_counts: dict[TournamentType, float | None],
    now: datetime,
) -> dict[TournamentType, float]:
    """Pure allocator: map rolling entrant counts per type to base emission weights at ``now``.

    A ``None`` (or non-positive) count means that type has no usable history; it is treated as
    average-subscribed so it neither gains nor loses flex. With balanced participation the flex
    pool splits at the minimums' ratio. Quant's share is then phased in by ``quant_ramp_factor``.
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
    weights = {t: minimum[t] + cts.EMISSION_FLEX_POOL * desirability[t] / total_desirability for t in _TYPES}
    return _phase_in_quant(weights, quant_ramp_factor(now))


def max_weight_by_type(now: datetime) -> dict[TournamentType, float]:
    """Upper bound of each type's pool at ``now``: its minimum plus the whole flex pool, phased in like the bases."""
    caps = {
        TournamentType.TEXT: cts.MAX_TEXT_TOURNAMENT_WEIGHT,
        TournamentType.QUANT: cts.MAX_QUANT_TOURNAMENT_WEIGHT,
        TournamentType.IMAGE: cts.MAX_IMAGE_TOURNAMENT_WEIGHT,
    }
    return _phase_in_quant(caps, quant_ramp_factor(now))


async def get_dynamic_base_weights(psql_db, now: datetime | None = None) -> dict[TournamentType, float]:
    """Fetch rolling entrant counts per type and compute the dynamic base split at ``now`` (default: the current time)."""
    if now is None:
        now = datetime.now(timezone.utc)
    counts: dict[TournamentType, float | None] = {}
    for tournament_type in _TYPES:
        recent = await get_recent_completed_entrant_counts(
            psql_db,
            tournament_type,
            limit=cts.EMISSION_BALANCE_ROLLING_WINDOW,
            min_entrants=cts.EMISSION_MIN_ENTRANTS_FOR_BALANCE,
        )
        counts[tournament_type] = mean(recent) if recent else None

    weights = compute_dynamic_base_weights(counts, now)
    logger.info(
        "Dynamic emission base split (rolling-%d entrants %s, quant ramp %.4f): text=%.4f quant=%.4f image=%.4f",
        cts.EMISSION_BALANCE_ROLLING_WINDOW,
        {t.value: counts[t] for t in _TYPES},
        quant_ramp_factor(now),
        weights[TournamentType.TEXT],
        weights[TournamentType.QUANT],
        weights[TournamentType.IMAGE],
    )
    return weights
