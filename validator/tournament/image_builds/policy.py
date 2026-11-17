"""Pure decisions for the build worker: retries, builder choice, waiting, and the prune window."""

from datetime import datetime
from datetime import timedelta

from validator.tournament.image_builds import constants as cst


def next_attempt_at(failure_class: str, attempts_made: int, now: datetime) -> datetime | None:
    """When to retry after a failed attempt, or None when the build should be marked failed."""
    max_attempts = cst.MAX_ATTEMPTS_BY_FAILURE_CLASS.get(failure_class, 2)
    if attempts_made >= max_attempts:
        return None
    backoff = cst.RETRY_BACKOFF_SECONDS[min(attempts_made - 1, len(cst.RETRY_BACKOFF_SECONDS) - 1)]
    return now + timedelta(seconds=backoff)


def choose_builder(
    free_slots: dict[str, int],
    tournament_load: dict[str, int],
    avoid: str | None,
) -> str | None:
    """Pick a builder with a free slot.

    Prefer the builder already building this tournament (BuildKit shares base pulls and identical steps between
    concurrent builds on one daemon, and identical layers then dedupe in R2), then the one with the most free
    slots. A retry avoids the builder of the previous attempt when another one is free.
    """
    candidates = [address for address, slots in free_slots.items() if slots > 0]
    if not candidates:
        return None
    preferred = [address for address in candidates if address != avoid] or candidates
    return max(preferred, key=lambda address: (tournament_load.get(address, 0), free_slots[address], address))


def should_wait_for_image(build_created_at: datetime, now: datetime) -> bool:
    return now - build_created_at < timedelta(seconds=cst.MAX_WAIT_FOR_IMAGE_SECONDS)


def in_prune_window(now: datetime) -> bool:
    """Friday from PRUNE_START_HOUR_UTC through Sunday (tournaments are due by Friday 14:00 UTC)."""
    weekday = now.weekday()
    if weekday == cst.PRUNE_START_WEEKDAY:
        return now.hour >= cst.PRUNE_START_HOUR_UTC
    return weekday > cst.PRUNE_START_WEEKDAY
