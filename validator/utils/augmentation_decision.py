"""Functions for deciding model prep augmentation config."""

import random

import validator.core.constants as vcst
from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskType
from validator.db.database import PSQLDB
from validator.db.sql.tasks import get_expected_repo_name
from validator.db.sql.tasks import get_task
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.utils.logging import get_logger


logger = get_logger(__name__)

def weighted_choice(
    weights: dict[AugmentationType, float] | dict[AugmentationScope, float],
    rng: random.Random,
) -> AugmentationType | AugmentationScope:
    """Pick an enum member from a weighted dict, normalising weights at runtime."""
    keys = list(weights.keys())
    vals = list(weights.values())
    total = sum(vals)
    normalised = [v / total for v in vals]
    return rng.choices(keys, weights=normalised, k=1)[0]


def seeded_intensity(aug_type: AugmentationType, rng: random.Random) -> float:
    """Return a random intensity for each augmentation type, driven by the seeded RNG."""
    low, high = vcst.AUGMENTATION_INTENSITY_RANGES.get(aug_type, (0.01, 0.01))
    return rng.uniform(low, high)


def _augmentation_enabled(task_type: TaskType) -> bool:
    if task_type == TaskType.IMAGETASK:
        return vcst.AUGMENTATION_ENABLED_IMAGE
    if task_type == TaskType.ENVIRONMENTTASK:
        return vcst.AUGMENTATION_ENABLED_ENV
    return vcst.AUGMENTATION_ENABLED_TEXT


def _is_text_task(task_type: TaskType) -> bool:
    return task_type not in (TaskType.IMAGETASK, TaskType.ENVIRONMENTTASK)


async def maybe_get_augmentation_config(task_type: TaskType, psql_db: PSQLDB | None = None) -> AugmentationConfig | None:
    """Randomly decide whether to augment a model and return the full config.

    All random choices after the initial coin flip are driven by a single seed,
    so the config is fully reproducible from {seed}.
    """
    if not _augmentation_enabled(task_type):
        return None

    if random.random() >= vcst.AUGMENTATION_PROBABILITY:
        return None

    if (
        psql_db is not None
        and _is_text_task(task_type)
        and random.random() < vcst.BOSS_BASE_MODEL_PROBABILITY
    ):
        boss_base_config = await maybe_get_boss_base_model_config(task_type, psql_db)
        if boss_base_config is not None:
            return boss_base_config

    seed = random.randint(0, 2**32 - 1)
    rng = random.Random(seed)

    aug_type: AugmentationType = weighted_choice(vcst.AUGMENTATION_TYPE_WEIGHTS, rng)
    scope: AugmentationScope = weighted_choice(vcst.AUGMENTATION_SCOPE_WEIGHTS, rng)
    intensity = seeded_intensity(aug_type, rng)

    return AugmentationConfig(
        aug_type=aug_type,
        scope=scope,
        seed=seed,
        intensity=intensity,
    )


async def maybe_get_boss_base_model_config(
    task_type: TaskType,
    psql_db: PSQLDB,
) -> AugmentationConfig | None:
    """Return a prior boss-round model config for the same text task type, if available."""
    if not _is_text_task(task_type):
        return None

    latest_tournament = await get_latest_completed_tournament(psql_db, TournamentType.TEXT)
    if latest_tournament is None:
        logger.info("No completed text tournament found for previous-model augmentation")
        return None

    rounds = await get_tournament_rounds(latest_tournament.tournament_id, psql_db)
    final_round = next((round_data for round_data in rounds if round_data.is_final_round), None)
    if final_round is None:
        logger.info(f"No final round found for tournament {latest_tournament.tournament_id}")
        return None

    source_candidates: list[tuple[str, str, str]] = []
    for tournament_task in await get_tournament_tasks(final_round.round_id, psql_db):
        source_task = await get_task(tournament_task.task_id, psql_db)
        if source_task is None or source_task.task_type != task_type:
            continue

        expected_repo_name = await get_expected_repo_name(
            source_task.task_id,
            vcst.EMISSION_BURN_HOTKEY,
            psql_db,
        )
        if expected_repo_name:
            source_repo = f"{vcst.RAYONLABS_HF_USERNAME}/{expected_repo_name}"
            source_candidates.append((str(source_task.task_id), source_task.model_id, source_repo))

    if not source_candidates:
        logger.info(
            f"No boss submission repo found in tournament {latest_tournament.tournament_id} "
            f"for task type {task_type.value}"
        )
        return None

    source_task_id, source_base_model_id, source_model_repo = random.choice(source_candidates)
    seed = random.randint(0, 2**32 - 1)
    return AugmentationConfig(
        aug_type=AugmentationType.BOSS_BASE_MODEL,
        seed=seed,
        source_model_repo=source_model_repo,
        source_task_id=source_task_id,
        source_base_model_id=source_base_model_id,
    )
