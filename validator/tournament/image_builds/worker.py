import asyncio
import os
import socket
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from core.logging import get_logger
from validator.db.sql import image_builds as image_build_sql
from validator.tournament.image_builds.depot import DepotClient
from validator.tournament.image_builds.depot import classify_failure
from validator.tournament.models import ImageBuild


logger = get_logger(__name__)
_project_locks: dict[tuple[str, str], asyncio.Lock] = {}


async def _ensure_project(build: ImageBuild, depot: DepotClient, config) -> str:
    key = (build.tournament_id, build.hotkey)
    lock = _project_locks.setdefault(key, asyncio.Lock())
    async with lock:
        project_id = await image_build_sql.get_submission_project_id(*key, config.psql_db)
        if project_id:
            return project_id
        name = f"god-{build.tournament_id[-12:]}-{build.hotkey[:8]}".lower().replace("_", "-")
        project_id = await depot.create_project(name)
        await image_build_sql.set_submission_project_id(*key, project_id, config.psql_db)
        return project_id


async def _confirm_publication(depot: DepotClient, project_id: str, tag: str) -> tuple[str, str]:
    for _ in range(10):
        image = await depot.find_image(project_id, tag)
        if image:
            return image
        await asyncio.sleep(2)
    raise RuntimeError("Depot build completed but registry publication was not confirmed")


async def _run_one(build: ImageBuild, depot: DepotClient, config) -> None:
    timeout_seconds = int(os.getenv("DEPOT_BUILD_ATTEMPT_TIMEOUT_SECONDS", "1800"))
    max_attempts = int(os.getenv("DEPOT_BUILD_MAX_ATTEMPTS", "2"))
    tag = f"{build.tournament_id[-8:]}-{build.hotkey[:8]}-{build.variant.value}-a{build.current_attempt}".lower().replace(
        "_", "-"
    )
    project_id = build.depot_project_id
    try:
        project_id = await _ensure_project(build, depot, config)
        depot_build_id, build_token = await depot.create_build(project_id)
        await image_build_sql.create_attempt(build.build_id, build.current_attempt, depot_build_id, config.psql_db)
        result = await depot.run_build(
            project_id=project_id,
            build_id=depot_build_id,
            build_token=build_token,
            context=build.build_context,
            dockerfile=build.dockerfile_path,
            platform=build.platform,
            tag=tag,
            github_token=build.github_token,
            timeout_seconds=timeout_seconds,
        )
        log_ref = f"https://depot.dev/orgs/{depot.organization_id}/projects/{project_id}/builds/{depot_build_id}"
        if result.returncode:
            if result.timed_out:
                raise TimeoutError(result.output)
            raise RuntimeError(result.output[-4000:])
        await image_build_sql.mark_publishing(build.build_id, config.psql_db)
        pull_ref, digest = await _confirm_publication(depot, project_id, tag)
        await image_build_sql.mark_ready(
            build.build_id,
            build.current_attempt,
            pull_ref,
            digest,
            log_ref,
            config.psql_db,
        )
        logger.info(f"Image build ready: {build.stable_build_id} -> {pull_ref}@{digest}")
        return
    except Exception as exc:
        error = str(exc)
        failure_class = classify_failure(error, isinstance(exc, TimeoutError))
        if project_id:
            try:
                await depot.delete_image(project_id, tag)
            except Exception:
                logger.warning(f"Could not remove partial Depot image {project_id}:{tag}")
        retry_at = None
        if build.current_attempt < max_attempts:
            retry_at = datetime.now(timezone.utc) + timedelta(seconds=int(os.getenv("DEPOT_BUILD_RETRY_DELAY_SECONDS", "15")))
        await image_build_sql.fail_or_retry(
            build.build_id,
            build.current_attempt,
            error,
            failure_class,
            retry_at,
            config.psql_db,
        )
        logger.error(
            f"Image build {build.stable_build_id} attempt {build.current_attempt} failed "
            f"({failure_class}); retry={retry_at is not None}"
        )


async def image_build_worker_cycle(config) -> None:
    if os.getenv("DEPOT_IMAGE_BUILDS_ENABLED", "false").lower() != "true":
        logger.info("Depot tournament image builds are disabled")
        while True:
            await asyncio.sleep(60)

    try:
        depot = DepotClient()
    except Exception:
        logger.exception("Depot image builds enabled but client configuration is invalid")
        while True:
            await asyncio.sleep(60)

    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    concurrency = int(os.getenv("DEPOT_BUILD_CONCURRENCY", "10"))
    timeout_seconds = int(os.getenv("DEPOT_BUILD_ATTEMPT_TIMEOUT_SECONDS", "1800"))
    max_attempts = int(os.getenv("DEPOT_BUILD_MAX_ATTEMPTS", "2"))
    stale_builds = await image_build_sql.recover_stale_builds(timeout_seconds, max_attempts, config.psql_db)
    await asyncio.gather(
        *(depot.cancel_build(project_id, build_id) for project_id, build_id in stale_builds),
        return_exceptions=True,
    )
    cleanup_counter = 0
    while True:
        try:
            builds = await image_build_sql.claim_builds(worker_id, concurrency, config.psql_db)
            if builds:
                await asyncio.gather(*(_run_one(build, depot, config) for build in builds))
            else:
                await asyncio.sleep(5)
            cleanup_counter += 1
            if cleanup_counter >= 120:
                cleanup_counter = 0
                due = await image_build_sql.list_projects_due_for_cleanup(
                    int(os.getenv("DEPOT_IMAGE_RETENTION_DAYS", "7")), config.psql_db
                )
                by_batch: dict[str, list[str]] = {}
                for batch_id, project_id in due:
                    by_batch.setdefault(batch_id, []).append(project_id)
                for batch_id, project_ids in by_batch.items():
                    for project_id in set(project_ids):
                        await depot.delete_project(project_id)
                    await image_build_sql.mark_batch_cleaned(batch_id, config.psql_db)
        except Exception:
            logger.exception("Image build worker cycle failed")
            await asyncio.sleep(10)
