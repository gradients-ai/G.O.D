"""
Participation-driven emission split: guaranteed minimums plus a balanced flex pool.

Each type gets a fixed guaranteed minimum (cts.TOURNAMENT_*_WEIGHT). Only the remaining
cts.EMISSION_FLEX_POOL moves with participation: it is split from how many miners actually
entered the last few tournaments of each type, on raw counts: a type with more contenders cedes
flex to one with fewer, driving miners toward the under-contested modality. The minimums are
floors only; they do not set what participation counts as balanced.
The swing is therefore bounded by construction: a type's base always lies in
[minimum, minimum + flex], and the bases always sum to exactly 1.0.

The single knob is cts.EMISSION_BALANCE_ALPHA — the rate of change: 0.0 splits the flex pool
equally, larger values react harder to a given imbalance.

    flex_shareᵢ ∝ (1 / contendersᵢ) ** alpha
    wᵢ = minimumᵢ + EMISSION_FLEX_POOL · flex_shareᵢ

Counts are contenders, not sign-ups: a challenger counts only if it ranked strictly above the boss in
round 1 (round_results.count_round_one_above_boss). When the round-1 ranking cannot be computed the
tournament falls back to the hotkeys that completed a round-1 task.
"""

from statistics import mean

import validator.scoring.constants as cts
from core.logging import get_logger
from core.models.tournament_models import TournamentType
from validator.db.sql.tournaments import count_round_one_completers
from validator.db.sql.tournaments import get_recent_completed_tournament_ids
from validator.tournament.round_results import count_round_one_above_boss


logger = get_logger(__name__)

# Round-1 results of a completed tournament no longer change, so its contender count is computed once per process.
_CONTENDERS_BY_TOURNAMENT: dict[str, int] = {}

_TYPES: tuple[TournamentType, ...] = (TournamentType.TEXT, TournamentType.QUANT, TournamentType.IMAGE)


def _minimums() -> dict[TournamentType, float]:
    return {
        TournamentType.TEXT: cts.TOURNAMENT_TEXT_WEIGHT,
        TournamentType.QUANT: cts.TOURNAMENT_QUANT_WEIGHT,
        TournamentType.IMAGE: cts.TOURNAMENT_IMAGE_WEIGHT,
    }


def compute_dynamic_base_weights(entrant_counts: dict[TournamentType, float | None]) -> dict[TournamentType, float]:
    """Pure allocator: map rolling contender counts per type to base emission weights.

    The flex pool is split on raw counts: equal counts split it equally, and a type with fewer contenders gets more.
    A ``None`` (or non-positive) count means that type has no usable history; it is given the average count of the
    others so it neither gains nor loses flex.
    """
    minimum = _minimums()
    known = {t: n for t in _TYPES if (n := entrant_counts.get(t)) is not None and n > 0}
    average = mean(known.values()) if known else 1.0
    counts = {t: known.get(t, average) for t in _TYPES}

    desirability = {t: (1.0 / counts[t]) ** cts.EMISSION_BALANCE_ALPHA for t in _TYPES}
    total_desirability = sum(desirability.values())
    return {t: minimum[t] + cts.EMISSION_FLEX_POOL * desirability[t] / total_desirability for t in _TYPES}


def max_weight_by_type() -> dict[TournamentType, float]:
    """Upper bound of each type's pool (its minimum plus the whole flex pool)."""
    return {
        TournamentType.TEXT: cts.MAX_TEXT_TOURNAMENT_WEIGHT,
        TournamentType.QUANT: cts.MAX_QUANT_TOURNAMENT_WEIGHT,
        TournamentType.IMAGE: cts.MAX_IMAGE_TOURNAMENT_WEIGHT,
    }


async def tournament_contenders(psql_db, tournament_id: str) -> int:
    """Contenders of one completed tournament: challengers ranked strictly above the boss in round 1.

    Falls back, uncached, to the round-1 completers when the ranking cannot be computed."""
    if tournament_id in _CONTENDERS_BY_TOURNAMENT:
        return _CONTENDERS_BY_TOURNAMENT[tournament_id]
    try:
        above_boss = await count_round_one_above_boss(tournament_id, psql_db)
    except Exception as e:
        logger.warning(f"Tournament {tournament_id}: round-1 ranks against the boss unavailable ({type(e).__name__}: {e})")
        above_boss = None
    if above_boss is None:
        completers = await count_round_one_completers(psql_db, tournament_id)
        logger.info(f"Tournament {tournament_id}: {completers} round-1 completers counted as contenders")
        return completers
    _CONTENDERS_BY_TOURNAMENT[tournament_id] = above_boss
    return above_boss


async def get_recent_contender_counts(psql_db, tournament_type: TournamentType) -> list[int]:
    """Contender counts of the last EMISSION_BALANCE_ROLLING_WINDOW completed tournaments of this type with at least
    EMISSION_MIN_ENTRANTS_FOR_BALANCE contenders, newest first. Tournaments below the minimum are skipped, so a reseeded
    or near-empty tournament can't distort the split, and older ones fill the window."""
    counts: list[int] = []
    tournament_ids = await get_recent_completed_tournament_ids(
        psql_db, tournament_type, limit=cts.EMISSION_CONTENDER_SCAN_LIMIT
    )
    for tournament_id in tournament_ids:
        contenders = await tournament_contenders(psql_db, tournament_id)
        if contenders >= cts.EMISSION_MIN_ENTRANTS_FOR_BALANCE:
            counts.append(contenders)
        if len(counts) == cts.EMISSION_BALANCE_ROLLING_WINDOW:
            break
    return counts


async def get_dynamic_base_weights(psql_db) -> dict[TournamentType, float]:
    """Fetch rolling contender counts per type and compute the dynamic base split."""
    counts: dict[TournamentType, float | None] = {}
    for tournament_type in _TYPES:
        recent = await get_recent_contender_counts(psql_db, tournament_type)
        counts[tournament_type] = mean(recent) if recent else None

    weights = compute_dynamic_base_weights(counts)
    logger.info(
        "Dynamic emission base split (rolling-%d contenders %s): text=%.4f quant=%.4f image=%.4f",
        cts.EMISSION_BALANCE_ROLLING_WINDOW,
        {t.value: counts[t] for t in _TYPES},
        weights[TournamentType.TEXT],
        weights[TournamentType.QUANT],
        weights[TournamentType.IMAGE],
    )
    return weights
