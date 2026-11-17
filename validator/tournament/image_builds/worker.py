"""Validator loop that runs prepared image builds on idle trainers (or a Shadeform fallback) and prunes R2.

Each cycle: poll in-flight attempts -> dispatch due queued builds -> fallback builder tick -> (hourly) prune.
All state lives in tournament_image_builds, so the loop is safe to restart at any point.
"""

import asyncio
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import httpx

from core.image_registry import load_registry_config
from core.image_registry import validate_digest
from core.logging import get_logger
from core.models.payload_models import ImageBuilderStatus
from core.models.payload_models import ImageBuildRequest
from validator.db.sql import image_builds as image_build_sql
from validator.db.sql import tournaments as tournament_sql
from validator.tournament.image_builds import builders
from validator.tournament.image_builds import constants as cst
from validator.tournament.image_builds.gate import image_builds_enabled
from validator.tournament.image_builds.models import ImageBuild
from validator.tournament.image_builds.policy import choose_builder
from validator.tournament.image_builds.policy import in_prune_window
from validator.tournament.image_builds.policy import next_attempt_at
from validator.tournament.image_builds.registry_prune import delete_tournament_repositories
from validator.tournament.image_builds.registry_prune import garbage_collect_blobs
from validator.tournament.image_builds.shadeform import FallbackBuilderManager


logger = get_logger(__name__)

ALLOW_BUSY_TRAINERS_ENV = "IMAGE_BUILDS_ALLOW_BUSY_TRAINERS"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def builder_free_slots(status: ImageBuilderStatus | None, is_trainer: bool, allow_busy: bool) -> int:
    """Slots the validator may use on a builder. Trainers running GPU jobs are skipped unless allowed:
    a build's disk/network load would slow the training next to it."""
    if status is None or not status.enabled:
        return 0
    if is_trainer and status.gpu_jobs_running > 0 and not allow_busy:
        return 0
    return max(0, status.free_slots)


async def _fail_attempt(build: ImageBuild, failure_class: str, error: str, config) -> None:
    retry_at = next_attempt_at(failure_class, build.attempts, _now())
    recorded = await image_build_sql.record_failure(
        build.build_id, build.attempts, failure_class, error, retry_at, config.psql_db
    )
    if recorded:
        outcome = f"retry at {retry_at:%H:%M:%S}" if retry_at else "giving up; training will build on the trainer"
        logger.warning(
            f"Image build {build.build_id} ({build.hotkey[:8]} {build.variant.value}) attempt {build.attempts} "
            f"on {build.builder} failed [{failure_class}]: {outcome}"
        )


async def poll_in_flight(config) -> dict[str, dict[str, int]]:
    """Advance every building row. Returns {builder: {tournament_id: active builds}} for placement."""
    load: dict[str, dict[str, int]] = {}
    now = _now()
    for build in await image_build_sql.list_building(config.psql_db):
        if not build.builder:
            await _fail_attempt(build, "lease_expired", "Building row without a builder", config)
            continue
        try:
            status = await builders.poll(build.builder, build.build_id, build.attempts)
        except (httpx.HTTPError, ValueError) as exc:
            if build.lease_expires_at and build.lease_expires_at < now:
                await _fail_attempt(build, "lease_expired", f"Builder {build.builder} unreachable: {exc}", config)
            else:
                load.setdefault(build.builder, {}).setdefault(build.tournament_id, 0)
                load[build.builder][build.tournament_id] += 1
            continue

        if status is None:
            await _fail_attempt(build, "cancelled", f"Builder {build.builder} no longer knows this job (restart?)", config)
        elif status.state == "ready":
            try:
                digest = validate_digest(status.digest or "")
            except ValueError:
                await _fail_attempt(build, "infrastructure", f"Builder reported an invalid digest {status.digest!r}", config)
                continue
            if await image_build_sql.mark_ready(
                build.build_id,
                build.attempts,
                digest,
                status.resolved_commit,
                status.dockerfile_path,
                status.size_bytes,
                config.psql_db,
            ):
                size = f"{status.size_bytes / 1e9:.1f} GB" if status.size_bytes else "unknown size"
                logger.info(f"Image ready for {build.hotkey[:8]} {build.variant.value}: {build.repository}@{digest} ({size})")
        elif status.state == "failed":
            await _fail_attempt(build, status.failure_class or "infrastructure", status.error or "", config)
        elif status.started_at and now - status.started_at > timedelta(seconds=cst.ATTEMPT_HARD_TIMEOUT_SECONDS):
            await _fail_attempt(build, "timeout", "Attempt exceeded the validator's hard timeout", config)
        else:
            await image_build_sql.renew_lease(build.build_id, build.attempts, cst.LEASE_SECONDS, config.psql_db)
            load.setdefault(build.builder, {}).setdefault(build.tournament_id, 0)
            load[build.builder][build.tournament_id] += 1
    return load


async def _builder_candidates(config, fallback: FallbackBuilderManager) -> dict[str, tuple[ImageBuilderStatus | None, bool]]:
    trainers = await tournament_sql.get_trainers(config.psql_db)
    addresses = {trainer.trainer_ip: True for trainer in trainers}
    for address in fallback.addresses:
        addresses.setdefault(address, False)
    statuses = await asyncio.gather(*(builders.fetch_status(address) for address in addresses))
    return {address: (status, is_trainer) for (address, is_trainer), status in zip(addresses.items(), statuses)}


async def dispatch(config, load: dict[str, dict[str, int]], fallback: FallbackBuilderManager) -> int:
    """Start due queued builds on builders with free slots. Returns free trainer slots left (for the fallback)."""
    in_flight = sum(count for per in load.values() for count in per.values())
    budget = cst.GLOBAL_MAX_CONCURRENT_BUILDS - in_flight
    if budget <= 0:
        # Capped by policy, not by capacity: do not let the fallback builder think trainers are full.
        return cst.GLOBAL_MAX_CONCURRENT_BUILDS
    queued = await image_build_sql.list_due_queued(budget, config.psql_db)
    if not queued:
        return 0
    allow_busy = os.getenv(ALLOW_BUSY_TRAINERS_ENV, "false").lower() == "true"
    candidates = await _builder_candidates(config, fallback)
    free = {
        address: builder_free_slots(status, is_trainer, allow_busy) for address, (status, is_trainer) in candidates.items()
    }

    for build in queued:
        previous_builder = build.builder
        tournament_load = {address: per.get(build.tournament_id, 0) for address, per in load.items()}
        address = choose_builder(free, tournament_load, avoid=previous_builder)
        if address is None:
            break
        expected_attempts = build.attempts
        if not await image_build_sql.start_attempt(
            build.build_id, expected_attempts, address, cst.LEASE_SECONDS, config.psql_db
        ):
            continue
        attempt = expected_attempts + 1
        build.attempts = attempt
        repo_info = await tournament_sql.get_tournament_training_repo_and_commit(
            build.hotkey, build.tournament_id, config.psql_db
        )
        # The participant token only applies to the participant's own repo (backup repos are public mirrors).
        token = repo_info.github_token if repo_info.training_repo == build.repo else None
        request = ImageBuildRequest(
            build_id=build.build_id,
            attempt=attempt,
            repo=build.repo,
            commit=None if build.commit_ref == "HEAD" else build.commit_ref,
            github_token=token,
            variant=build.variant.value,
            repository=build.repository,
            tag=build.tag,
        )
        try:
            await builders.submit(address, request)
        except builders.BuilderRejected as exc:
            logger.info(f"Builder {address} rejected build {build.build_id}: {exc}")
            await image_build_sql.undo_unsent_attempt(
                build.build_id, attempt, _now() + timedelta(seconds=30), config.psql_db
            )
            free[address] = 0
            continue
        except builders.BuilderOutcomeUnknown as exc:
            # Keep the attempt; polling either finds the job or gets a 404 and retries.
            logger.warning(f"Unknown outcome dispatching build {build.build_id} to {address}: {exc}")
        free[address] -= 1
        load.setdefault(address, {}).setdefault(build.tournament_id, 0)
        load[address][build.tournament_id] += 1
        logger.info(
            f"Dispatched image build {build.build_id} ({build.hotkey[:8]} {build.variant.value}) "
            f"attempt {attempt} to {address}"
        )
    return sum(slots for address, slots in free.items() if candidates[address][1])


class _PruneState:
    def __init__(self):
        self.last_run: datetime | None = None
        self.last_gc: datetime | None = None


async def prune_if_due(config, state: _PruneState) -> None:
    now = _now()
    if state.last_run and now - state.last_run < timedelta(seconds=cst.PRUNE_CHECK_INTERVAL_SECONDS):
        return
    state.last_run = now
    if not in_prune_window(now):
        return
    registry_config = load_registry_config()
    if registry_config is None:
        logger.warning("IMAGE_REGISTRY_* not set on the validator; skipping prepared image pruning")
        return
    pruned = 0
    for tournament_id in await image_build_sql.tournaments_to_prune(config.psql_db):
        deleted = await asyncio.to_thread(delete_tournament_repositories, registry_config, tournament_id)
        await image_build_sql.mark_tournament_pruned(tournament_id, config.psql_db)
        logger.info(f"Pruned prepared images of tournament {tournament_id} ({deleted} registry objects)")
        pruned += 1
    # GC after pruning, and at least weekly: failed builds can leave uploaded blobs that no repository links.
    if pruned or state.last_gc is None or now - state.last_gc > timedelta(days=6):
        orphans, freed = await asyncio.to_thread(garbage_collect_blobs, registry_config)
        state.last_gc = now
        logger.info(f"Image registry GC removed {orphans} unreferenced blobs ({freed / 1e9:.1f} GB)")


async def image_build_worker_cycle(config) -> None:
    fallback = FallbackBuilderManager()
    prune_state = _PruneState()
    while True:
        try:
            if image_builds_enabled():
                load = await poll_in_flight(config)
                trainer_free_slots = await dispatch(config, load, fallback)
                queued_due, building, oldest = await image_build_sql.queue_summary(config.psql_db)
                await fallback.tick(queued_due, building, oldest, trainer_free_slots)
                await prune_if_due(config, prune_state)
        except Exception:
            logger.exception("Image build worker cycle failed")
        await asyncio.sleep(cst.WORKER_INTERVAL_SECONDS)
