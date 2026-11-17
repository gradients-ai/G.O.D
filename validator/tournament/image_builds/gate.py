"""Connect build rows to tournament activation and to training scheduling."""

import os
from datetime import datetime
from datetime import timezone

from core.image_registry import repository_name
from core.logging import get_logger
from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.db.sql import image_builds as image_build_sql
from validator.db.sql import tournaments as tournament_sql
from validator.tournament.image_builds import constants as cst
from validator.tournament.image_builds.models import ImageBuildStatus
from validator.tournament.image_builds.models import ImageBuildVariant
from validator.tournament.image_builds.policy import should_wait_for_image
from validator.tournament.models import TournamentTaskTraining


logger = get_logger(__name__)

TOURNAMENT_PRIORITY = 2


def image_builds_enabled() -> bool:
    return os.getenv(cst.ENABLED_ENV, "false").lower() == "true"


def variant_for_task(task) -> ImageBuildVariant:
    if task.task_type == TaskType.IMAGETASK:
        return ImageBuildVariant.IMAGE_FLUX if task.model_type in cst.FLUX_MODEL_TYPES else ImageBuildVariant.IMAGE_TOOLKIT
    return ImageBuildVariant.TEXT


def variants_for_tournament(tournament_type: TournamentType) -> tuple[ImageBuildVariant, ...]:
    if tournament_type == TournamentType.IMAGE:
        return (ImageBuildVariant.IMAGE_TOOLKIT,)
    if tournament_type == TournamentType.TEXT:
        return (ImageBuildVariant.TEXT,)
    return ()


async def _ensure(tournament_id: str, hotkey: str, variant: ImageBuildVariant, repo: str, commit: str | None, psql_db):
    return await image_build_sql.ensure_build(
        tournament_id=tournament_id,
        hotkey=hotkey,
        variant=variant,
        repo=repo,
        commit_ref=commit or "HEAD",
        repository=repository_name(tournament_id, hotkey, variant.value),
        psql_db=psql_db,
    )


async def enqueue_tournament_builds(tournament_id: str, config) -> int:
    """Queue one build per participant up front so images are ready by the time round-1 tasks finish prep.

    Never raises: activation must not depend on image builds.
    """
    if not image_builds_enabled():
        return 0
    try:
        tournament = await tournament_sql.get_tournament(tournament_id, config.psql_db)
        if tournament is None:
            return 0
        variants = variants_for_tournament(tournament.tournament_type)
        count = 0
        for participant in await tournament_sql.get_tournament_participants(tournament_id, config.psql_db):
            repo_info = await tournament_sql.get_tournament_training_repo_and_commit(
                participant.hotkey, tournament_id, config.psql_db
            )
            if not repo_info.training_repo:
                continue
            for variant in variants:
                await _ensure(
                    tournament_id,
                    participant.hotkey,
                    variant,
                    repo_info.training_repo,
                    repo_info.training_commit_hash,
                    config.psql_db,
                )
                count += 1
        logger.info(f"Queued {count} prepared image builds for tournament {tournament_id}")
        return count
    except Exception:
        logger.exception(f"Could not queue prepared image builds for tournament {tournament_id}")
        return 0


async def attach_prepared_images(pending: list[TournamentTaskTraining], config) -> list[TournamentTaskTraining]:
    """Attach ready images to tournament trainings and hold back trainings whose image is still building.

    A build that failed, or one still not ready after MAX_WAIT_FOR_IMAGE_SECONDS, does not block: the training is
    scheduled without an image and the trainer builds the repo itself, as before this feature.
    """
    if not image_builds_enabled():
        return pending
    schedulable = []
    held_back = 0
    now = datetime.now(timezone.utc)
    for training in pending:
        try:
            if training.priority != TOURNAMENT_PRIORITY or not training.training_repo:
                schedulable.append(training)
                continue
            tournament_id = await tournament_sql.get_tournament_id_by_task_id(str(training.task.task_id), config.psql_db)
            if not tournament_id:
                schedulable.append(training)
                continue
            build = await _ensure(
                tournament_id,
                training.hotkey,
                variant_for_task(training.task),
                training.training_repo,
                training.training_commit_hash,
                config.psql_db,
            )
            if build.status == ImageBuildStatus.READY:
                training.prepared_image = build.prepared_image()
                schedulable.append(training)
            elif build.status == ImageBuildStatus.FAILED or not should_wait_for_image(build.created_at, now):
                if build.status != ImageBuildStatus.FAILED:
                    logger.warning(f"Image for {training.hotkey} not ready after the wait limit; building on the trainer")
                schedulable.append(training)
            else:
                held_back += 1
        except Exception:
            logger.exception(f"Prepared image lookup failed for {training.hotkey}; building on the trainer")
            schedulable.append(training)
    if held_back:
        logger.info(f"Holding back {held_back} tournament trainings until their prepared images are built")
    return schedulable
