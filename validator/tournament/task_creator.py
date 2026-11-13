import asyncio
import random
from typing import AsyncGenerator
from uuid import UUID

from fiber.chain.models import Node

from core.constants.credentials import RAYONLABS_HF_USERNAME
from core.constants.quant import QUANT_MODEL_SIZE_B
from core.logging import get_logger
from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.app.config import Config
from validator.db.database import PSQLDB
from validator.db.sql import quant_sources as quant_sources_sql
from validator.db.sql import submissions_and_scoring as submissions_and_scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.db.sql.continuous_sft import warn_orphaned_continuous_sft_state
from validator.db.sql.tournaments import add_tournament_tasks
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import is_planned_image_tournament
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.tasks import calculate_miner_ranking_and_scores
from validator.tasks.models import RawTask
from validator.tasks.quant_input_repo import full_model_rejection
from validator.tasks.quant_shapes import filter_quant_compatible
from validator.tasks.requests import get_model_num_params
from validator.tasks.synthetics.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
from validator.tasks.synthetics.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO
from validator.tasks.synthetics.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
from validator.tasks.synthetics.diffusion import create_synthetic_image_task
from validator.tasks.synthetics.scheduler import _get_dpo_datasets
from validator.tasks.synthetics.scheduler import _get_image_models
from validator.tasks.synthetics.scheduler import _get_instruct_text_datasets
from validator.tasks.synthetics.scheduler import _get_text_models
from validator.tasks.synthetics.scheduler import create_chained_quant_task
from validator.tasks.synthetics.scheduler import create_continuous_sft_task
from validator.tasks.synthetics.scheduler import create_synthetic_dpo_task
from validator.tasks.synthetics.scheduler import create_synthetic_grpo_task
from validator.tasks.synthetics.scheduler import create_synthetic_instruct_text_task
from validator.tasks.synthetics.scheduler import draw_quant_format
from validator.tasks.synthetics.scheduler import recreate_chained_quant_task
from validator.tasks.synthetics.scheduler import repoint_chained_quant_task
from validator.tournament import constants as t_cst
from validator.tournament import quant_sources
from validator.tournament.chaining import is_chained
from validator.tournament.crown import is_text_big_task
from validator.tournament.gpu_requirements import get_task_gpu_requirement
from validator.tournament.models import BOSS_ROUND_SEED_ROLES
from validator.tournament.models import QUANT_SLOT_BY_SEED_ROLE
from validator.tournament.models import QUANT_SLOT_SPECS
from validator.tournament.models import RETIRED_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import UNCREATED_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import GroupRound
from validator.tournament.models import ImageRoundSpec
from validator.tournament.models import ImageTaskSpec
from validator.tournament.models import KnockoutRound
from validator.tournament.models import QuantRoundSpec
from validator.tournament.models import Round
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TextRoundSpec
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentTask
from validator.tournament.task_results import get_task_results_for_ranking


logger = get_logger(__name__)


class MissingBossRoundSeeds(RuntimeError):
    """A round cannot attach: these seed roles have no live seed (a re-draw is pending)."""

    def __init__(self, tournament_id: str, roles: list[str]) -> None:
        self.tournament_id = tournament_id
        self.roles = roles
        super().__init__(f"Tournament {tournament_id}: seed(s) {roles} are not live; cannot attach the round")




def is_single_group_text_round_one(round_data: GroupRound) -> bool:
    """Whether a text group round is round 1 with everyone in one group.

    Text R1 always uses this format (see organise_tournament_round). The round-1
    guard is load-bearing: a later text round can still be a single group, and must
    keep getting one task per group rather than SMALL_TOURNAMENT_GROUP_TASKS.
    """
    return round_data.round_number == 1 and len(round_data.groups) == 1


def is_single_group_image_round_one(round_data: GroupRound) -> bool:
    """Whether this is the universal three-task image opening round."""
    return round_data.round_number == 1 and len(round_data.groups) == 1


async def create_text_tournament_tasks(
    round_data: Round,
    tournament_id: str,
    config: Config,
    is_final_round: bool = False,
) -> list[str]:
    round_id = round_data.round_id
    seeded = await seeds_sql.is_seeded_text_tournament(tournament_id, config.psql_db)
    if seeded and isinstance(round_data, GroupRound):
        logger.info(f"Attaching text round {round_data.round_number}'s seed task")
        tasks = await _attach_text_round_seed(round_data, tournament_id, config)
    elif seeded and is_final_round:
        tasks = await _create_seeded_text_boss_round_tasks(tournament_id, round_id, config)
    elif isinstance(round_data, GroupRound):
        num_groups = len(round_data.groups)
        logger.info(f"Creating text tournament for {num_groups} groups (1 task per group)")
        tasks = await _create_group_text_tasks(round_data, tournament_id, config, is_final_round)
    elif is_final_round:
        logger.info(
            f"Creating final text tournament boss round: {t_cst.FINAL_ROUND_TEXT_TASKS} tasks "
            f"({t_cst.FINAL_ROUND_TEXT_TASK_DISTRIBUTION} + "
            f"{t_cst.FINAL_ROUND_CONTINUOUS_SFT_TASKS} continuous-SFT)"
        )
        tasks = await _create_new_text_boss_round_tasks(tournament_id, round_id, config)
    else:
        num_pairs = len(round_data.pairs)
        logger.info(f"Creating text tournament for {num_pairs} knockout pairs (probability-based)")
        tasks = await _create_probability_based_text_tasks(round_data, tournament_id, config)

    return [str(task.task_id) for task in tasks]


def text_round_spec(round_number: int) -> TextRoundSpec:
    """The TEXT_ROUND_PLAN entry of a miner round."""
    for spec in t_cst.TEXT_ROUND_PLAN:
        if spec.round_number == round_number:
            return spec
    raise ValueError(f"Text round {round_number} is not a miner round (TEXT_ROUND_PLAN has {len(t_cst.TEXT_ROUND_PLAN)})")


async def draw_text_round_task(spec: TextRoundSpec, config: Config, datasets: AsyncGenerator | None = None) -> RawTask:
    """One ordinary instruct task for a miner round of a seeded text tournament, drawn like that round's seed task: a model in
    the round's size band from the quant-compatible pool, KL off, augmentation rolled, a calibration set carved.

    Used when the round's seed role failed: the round plays a task of its own, which the boss has no result on.
    datasets: the dataset generator to draw from (default: the instruct pool).
    """
    models = filter_quant_compatible(_get_text_models(config.keypair, *spec.model_size_b))
    return await create_synthetic_instruct_text_task(
        config,
        models,
        datasets if datasets is not None else _get_instruct_text_datasets(config.keypair),
        enable_kl=False,
        allow_augmentation=True,
        carve_calibration=True,
    )


async def _role_is_failed(tournament_id: str, role: SeedRole, psql_db: PSQLDB) -> bool:
    """Whether the latest attempt of the role is `failed` (out of replacements). A role being replaced, or with no row, is not."""
    latest = next((row for row in await seeds_sql.get_latest_seed_tasks(tournament_id, psql_db) if row.role == role), None)
    return latest is not None and latest.status == SeedStatus.FAILED


async def _attach_text_round_seed(round_data: GroupRound, tournament_id: str, config: Config) -> list[RawTask]:
    """Rounds 1-3 of a seeded text tournament: the round's one task is the seed task the boss trained at tournament start.

    The seed is registered under the round's group (no task is drawn) and marked attached, which also releases its
    finalisation. Idempotent: a round that already has its task returns it. Raises while the role has no live seed (a
    replacement is pending), so the round stays PENDING and is retried. A role that is `failed` (out of replacements) does
    not block the round: it draws one ordinary instruct task (draw_text_round_task) with no seed row and no boss assignment,
    so the boss has no result on it and the challenger takes it with a valid score.
    """
    round_id = round_data.round_id
    spec = text_round_spec(round_data.round_number)
    group_id = f"{round_id}_group_001"
    existing = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
    if existing:
        logger.info(f"Round {round_data.round_number} already has its text task, skipping task creation")
        held = await _get_existing_tasks(existing, config)
        # A stop between registration and the marker leaves the seed unmarked; the marker is idempotent.
        for task in held:
            if await seeds_sql.get_seed_task_by_task_id(task.task_id, config.psql_db) is not None:
                await seeds_sql.mark_seed_attached(task.task_id, round_id, config.psql_db)
        return held

    seed = next(
        (row for row in await seeds_sql.get_active_seed_tasks(tournament_id, config.psql_db) if row.role == spec.seed_role), None
    )
    if seed is None:
        if not await _role_is_failed(tournament_id, spec.seed_role, config.psql_db):
            raise MissingBossRoundSeeds(tournament_id, [spec.seed_role.value])
        task = await draw_text_round_task(spec, config)
        await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
        logger.warning(f"Round {round_id}: seed {spec.seed_role.value} failed; drew its own task {task.task_id} (no boss result)")
        return [task]
    task = await task_sql.get_task(seed.task_id, config.psql_db)
    if task is None:
        raise RuntimeError(f"Tournament {tournament_id}: seed {spec.seed_role.value} task {seed.task_id} is gone")
    await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
    # Durable "ever attached" marker (idempotent): from here on the seed is only replaced for a prep failure.
    await seeds_sql.mark_seed_attached(task.task_id, round_id, config.psql_db)
    logger.info(f"Round {round_id}: attached seed {spec.seed_role.value} task {task.task_id}")
    return [task]


async def _create_seeded_text_boss_round_tasks(tournament_id: str, round_id: str, config: Config) -> list[RawTask]:
    """Stage 1 of a seeded text tournament's boss round: the continuous seed attached, one GRPO and one DPO task drawn.

    The boss already trained the continuous seed at tournament start; the challenger gets its training row from the seed
    loop once the seed is attached. GRPO and DPO are trained by both players. The big task is not created here: it is added
    only if the challenger can still be crowned once these three are scored (create_seeded_text_big_task).

    Idempotent by what the round already holds: a second pass, after a restart or after the big task was added, never
    creates a second continuous, GRPO or DPO task. Raises MissingBossRoundSeeds while the continuous seed has no live row,
    before anything is drawn, so the round stays PENDING and is retried. A continuous seed that is `failed` (out of
    replacements) does not block the round: the round draws its own continuous task on the single lineage, registered under
    the pair like the GRPO and DPO tasks, with no seed row (the boss has no upfront result on it; both players train it here).
    """
    pair_id = f"{round_id}_pair_001"
    existing_pair_tasks = await _get_existing_tasks_by_identifier(round_id, config, pair_id=pair_id)
    existing = await _get_existing_tasks(existing_pair_tasks, config)

    continuous = next((task for task in existing if t_cst.is_continuous_sft_task(task)), None)
    if continuous is None:
        seed = next(
            (
                row
                for row in await seeds_sql.get_active_seed_tasks(tournament_id, config.psql_db)
                if row.role == SeedRole.TEXT_CONTINUOUS
            ),
            None,
        )
        if seed is not None:
            continuous = await task_sql.get_task(seed.task_id, config.psql_db)
            if continuous is None:
                raise RuntimeError(f"Tournament {tournament_id}: seed text_continuous task {seed.task_id} is gone")
            await _create_and_register_tournament_task(continuous, tournament_id, round_id, config, pair_id=pair_id)
            logger.info(f"Boss round {round_id}: attached seed {SeedRole.TEXT_CONTINUOUS.value} task {continuous.task_id}")
        elif await _role_is_failed(tournament_id, SeedRole.TEXT_CONTINUOUS, config.psql_db):
            if len(t_cst.CONTINUOUS_SFT_LINEAGES) != 1:
                raise ValueError(
                    f"the boss round needs exactly one continuous-SFT lineage, found {sorted(t_cst.CONTINUOUS_SFT_LINEAGES)}"
                )
            lineage, seed_model = next(iter(t_cst.CONTINUOUS_SFT_LINEAGES.items()))
            continuous = await _create_continuous_sft_boss_task(tournament_id, round_id, pair_id, config, lineage, seed_model)
            logger.warning(
                f"Boss round {round_id}: seed {SeedRole.TEXT_CONTINUOUS.value} failed; drew its own continuous task "
                f"{continuous.task_id} (both players train it in this round)"
            )
        else:
            raise MissingBossRoundSeeds(tournament_id, [SeedRole.TEXT_CONTINUOUS.value])
    # Durable "ever attached" marker (idempotent): from here on the seed is only replaced for a prep failure. A continuous
    # task the round drew itself has no seed row and gets no marker.
    if await seeds_sql.get_seed_task_by_task_id(continuous.task_id, config.psql_db) is not None:
        await seeds_sql.mark_seed_attached(continuous.task_id, round_id, config.psql_db)
    tasks = [continuous]

    standard_models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)
    for task_type in t_cst.TEXT_BOSS_ROUND_STAGE_ONE_TASK_TYPES:
        present = [task for task in existing if task.task_type == task_type]
        if present:
            tasks.extend(present)
            continue
        logger.info(f"Creating the {task_type.value} task of boss round {round_id}")
        task = await _create_single_new_text_task(
            task_type, tournament_id, round_id, pair_id, config, standard_models, instruct_datasets, dpo_datasets
        )
        # _create_single_new_text_task swallows its errors and returns None. Raise so the round stays PENDING and the next
        # pass draws only what is still missing.
        if task is None:
            raise RuntimeError(f"Failed to create the {task_type.value} task of boss round {round_id}")
        tasks.append(task)
    return tasks


async def create_seeded_text_big_task(tournament_id: str, round_id: str, config: Config) -> RawTask:
    """Stage 2 of a seeded text tournament's boss round: the big task, an instruct task on a 30-72B model.

    Made the way the unseeded text boss round makes its large-instruct slot: drawn from the large-instruct pool, KL on and no
    augmentation (republishing a 70B copy during model-prep exceeds the prep timeout). Idempotent: a round that already
    holds an instruct task returns it. Raises when the task cannot be created.
    """
    pair_id = f"{round_id}_pair_001"
    existing = await _get_existing_tasks(await _get_existing_tasks_by_identifier(round_id, config, pair_id=pair_id), config)
    held = next((task for task in existing if is_text_big_task(task.task_type)), None)
    if held is not None:
        logger.info(f"Boss round {round_id} already holds its big task {held.task_id}")
        return held

    large_instruct_models = _get_text_models(
        config.keypair,
        smallest_size_b=t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B,
        largest_size_b=t_cst.BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B,
    )
    task = await _create_single_new_text_task(
        TaskType.INSTRUCTTEXTTASK,
        tournament_id,
        round_id,
        pair_id,
        config,
        large_instruct_models,
        _get_instruct_text_datasets(config.keypair),
        _get_dpo_datasets(config.keypair),
        allow_augmentation=False,
    )
    if task is None:
        raise RuntimeError(f"Failed to create the big task of boss round {round_id}")
    logger.info(f"Created the big task {task.task_id} of boss round {round_id}")
    return task


async def swap_seed_task_registration(original_task_id: str, new_task: RawTask, tournament_id: str, config: Config) -> str | None:
    """Move an attached seed's round registration from a task nobody trained on to its replacement.

    The new task takes the old one's round, group and pair; every node assigned to the old task is assigned to it with
    its own expected repo, and the old task's registration and node assignments are dropped (the task row is kept).
    Safe to repeat after an interruption. Returns the round id, or None when the old task was not registered anywhere.
    """
    original = await tournament_sql.get_tournament_task_by_task_id(original_task_id, config.psql_db)
    if original is None:
        return None
    nodes = await task_sql.get_nodes_assigned_to_task(original_task_id, config.psql_db)
    if await tournament_sql.get_tournament_task_by_task_id(str(new_task.task_id), config.psql_db) is None:
        await _create_and_register_tournament_task(
            new_task, tournament_id, original.round_id, config, group_id=original.group_id, pair_id=original.pair_id
        )
    already_assigned = {node.hotkey for node in await task_sql.get_nodes_assigned_to_task(new_task.task_id, config.psql_db)}
    for node in nodes:
        if node.hotkey in already_assigned:
            continue
        await task_sql.assign_node_to_task(new_task.task_id, node, config.psql_db)
        await task_sql.set_expected_repo_name(
            new_task.task_id, node, config.psql_db, f"tournament-{tournament_id}-{new_task.task_id}-{node.hotkey[:8]}"
        )
    await task_sql.detach_task_from_tournament(original_task_id, tournament_id, config.psql_db)
    logger.info(f"Round {original.round_id}: seed task {original_task_id} swapped for {new_task.task_id}")
    return original.round_id


async def create_image_tournament_tasks(
    round_data: Round, tournament_id: str, config: Config, is_final_round: bool = False,
) -> list[str]:
    round_id = round_data.round_id
    image_models = _get_image_models(config.keypair)
    tasks = []
    planned = await is_planned_image_tournament(tournament_id, config.psql_db)

    if planned and isinstance(round_data, GroupRound):
        tasks = await _create_planned_image_round_task(round_data, tournament_id, config)
    elif planned and is_final_round:
        specs = await _planned_image_boss_round_specs(tournament_id, round_id, config)
        logger.info(
            "Creating the image tournament boss round: "
            + ", ".join(f"{model_type.value} ({training_mode.value})" for model_type, training_mode in specs)
        )
        tasks = await _create_new_image_boss_round_tasks(tournament_id, round_id, config, specs)
        # The crown needs every boss-round task: a round short of one stays pending and the next pass creates the rest.
        if len(tasks) != t_cst.IMAGE_BOSS_ROUND_TASKS:
            raise RuntimeError(f"Image boss round {round_id} holds {len(tasks)} of its {t_cst.IMAGE_BOSS_ROUND_TASKS} tasks")
    elif planned:
        raise ValueError(f"Image tournament round {round_id} is neither a group round nor the boss round")
    elif isinstance(round_data, GroupRound):
        tasks = await _create_group_image_tasks(round_data, tournament_id, config, image_models)
    elif is_final_round:
        tasks = await _create_new_image_boss_round_tasks(tournament_id, round_id, config, t_cst.FINAL_ROUND_IMAGE_TASK_SPECS)
    else:
        training_mode = None
        if round_data.round_number == 2:
            round_two_model_type, training_mode = random.choice(t_cst.ROUND_TWO_IMAGE_TASK_SPECS)
            logger.info(
                f"Round 2 image tournament model type: {round_two_model_type.value} "
                f"({training_mode.value})"
            )
            image_models = _image_models_of_type(config, round_two_model_type)
        tasks = await _create_knockout_image_tasks(
            round_data, tournament_id, config, image_models, training_mode=training_mode
        )

    return [str(task.task_id) for task in tasks]


def image_round_spec(round_number: int) -> ImageRoundSpec:
    """The IMAGE_ROUND_PLAN entry of a miner round."""
    for spec in t_cst.IMAGE_ROUND_PLAN:
        if spec.round_number == round_number:
            return spec
    raise ValueError(f"Image round {round_number} is not a miner round (IMAGE_ROUND_PLAN has {len(t_cst.IMAGE_ROUND_PLAN)})")


async def _create_planned_image_round_task(round_data: GroupRound, tournament_id: str, config: Config) -> list[RawTask]:
    """R1-R3 of a planned image tournament: the round's one task, a random draw from the (model type, training mode) slots
    IMAGE_ROUND_PLAN gives the round, registered to the round's single group. Idempotent: a group that already has its task
    returns it."""
    if len(round_data.groups) != 1:
        raise ValueError(f"Image round {round_data.round_id} has {len(round_data.groups)} groups; a planned round has one")
    spec = image_round_spec(round_data.round_number)
    group_id = f"{round_data.round_id}_group_001"
    existing = await _get_existing_tasks_by_identifier(round_data.round_id, config, group_id=group_id)
    if existing:
        logger.info(f"Round {round_data.round_number} already has its image task, skipping task creation")
        return await _get_existing_tasks(existing, config)

    model_type, training_mode = random.choice(spec.task_specs)
    logger.info(f"Creating the {model_type.value} ({training_mode.value}) task of image round {round_data.round_number}")
    task = await _create_single_image_task_with_retry(
        config,
        _image_models_of_type(config, model_type),
        0,
        0,
        max_num_prompts=spec.max_num_prompts,
        training_mode=training_mode,
    )
    await _create_and_register_tournament_task(task, tournament_id, round_data.round_id, config, group_id=group_id)
    return [task]


async def _create_group_image_tasks(
    round_data: GroupRound, tournament_id: str, config: Config, image_models: list
) -> list[RawTask]:
    is_round_one = is_single_group_image_round_one(round_data)
    tasks_per_group = t_cst.SMALL_TOURNAMENT_GROUP_TASKS if is_round_one else t_cst.IMAGE_TASKS_PER_GROUP

    num_groups = len(round_data.groups)
    logger.info(f"Creating image tournament for {num_groups} groups ({tasks_per_group} per group)")
    tasks = []

    for i, group in enumerate(round_data.groups):
        group_tasks = await _create_single_group_image_tasks(
            group,
            i,
            tournament_id,
            round_data.round_id,
            config,
            image_models,
            tasks_per_group,
            image_model_types=t_cst.ROUND_ONE_IMAGE_MODEL_TYPES if is_round_one else None,
            max_num_prompts=t_cst.ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS if is_round_one else None,
            training_mode=t_cst.ROUND_ONE_IMAGE_TRAINING_MODE if is_round_one else None,
        )
        tasks.extend(group_tasks)

    return tasks


async def _create_single_group_image_tasks(
    group,
    group_index: int,
    tournament_id: str,
    round_id: str,
    config: Config,
    image_models: list,
    tasks_per_group: int,
    image_model_types: tuple[ImageModelType, ...] | None = None,
    max_num_prompts: int | None = None,
    training_mode: ImageTrainingMode | None = None,
) -> list[RawTask]:
    group_id = f"{round_id}_group_{group_index + 1:03d}"
    logger.info(f"  Group {group_index + 1} ({len(group.member_ids)} members):")

    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
    existing_count = len(existing_tasks)

    if existing_count >= tasks_per_group:
        logger.info(f"    Group {group_index + 1} already has {existing_count} task(s), skipping task creation")
        return await _get_existing_tasks(existing_tasks, config)

    created: list[RawTask] = await _get_existing_tasks(existing_tasks, config)
    for task_index in range(existing_count, tasks_per_group):
        logger.info(f"    Group {group_index + 1} has {len(created)}/{tasks_per_group} task(s), creating 1 more")
        task_models = (
            _image_models_of_type(config, image_model_types[task_index])
            if image_model_types
            else image_models
        )
        task = await _create_single_image_task_with_retry(
            config,
            task_models,
            task_index,
            group_index,
            max_num_prompts=max_num_prompts,
            training_mode=training_mode,
        )
        await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
        created.append(task)

    return created


async def _create_knockout_image_tasks(
    round_data: KnockoutRound,
    tournament_id: str,
    config: Config,
    image_models: list,
    training_mode: ImageTrainingMode | None = None,
) -> list[RawTask]:
    num_pairs = len(round_data.pairs)
    logger.info(f"Creating image tournament for {num_pairs} knockout pairs ({t_cst.KNOCKOUT_PAIR_TASKS} per pair)")
    tasks = []

    for i, pair in enumerate(round_data.pairs):
        pair_tasks = await _create_single_knockout_image_task(
            pair, i, tournament_id, round_data.round_id, config, image_models, training_mode=training_mode
        )
        tasks.extend(pair_tasks)

    return tasks


async def _create_single_knockout_image_task(
    pair,
    pair_index: int,
    tournament_id: str,
    round_id: str,
    config: Config,
    image_models: list,
    training_mode: ImageTrainingMode | None = None,
) -> list[RawTask]:
    pair_id = f"{round_id}_pair_{pair_index + 1:03d}"
    logger.info(f"  Pair {pair_index + 1} ({pair[0]} vs {pair[1]}):")

    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, pair_id=pair_id)
    existing_count = len(existing_tasks)

    if existing_tasks:
        if existing_count > t_cst.KNOCKOUT_PAIR_TASKS:
            logger.warning(
                f"   Pair {pair_index + 1} has {existing_count} tasks when it should only have {t_cst.KNOCKOUT_PAIR_TASKS}!"
            )
        logger.info(f"    Pair {pair_index + 1} already has {existing_count} task(s), skipping task creation")
        return await _get_existing_tasks(existing_tasks, config)

    logger.info(f"    Pair {pair_index + 1} has no tasks, creating {t_cst.KNOCKOUT_PAIR_TASKS}")
    task = await _create_single_image_task_with_retry(
        config, image_models, 0, pair_index, training_mode=training_mode
    )
    await _create_and_register_tournament_task(
        task, tournament_id, round_id, config, pair_id=pair_id
    )
    return [task]


async def _create_single_image_task_with_retry(
    config: Config,
    image_models: list,
    task_num: int,
    group_index: int = None,
    is_final: bool = False,
    max_num_prompts: int | None = None,
    training_mode: ImageTrainingMode | None = None,
) -> RawTask:
    context = f"final image task {task_num + 1}" if is_final else f"image task {task_num + 1} for group {group_index + 1}"
    for attempt in range(1, t_cst.IMAGE_TASK_CREATION_MAX_ATTEMPTS + 1):
        try:
            return await create_synthetic_image_task(
                config,
                image_models,
                max_num_prompts=max_num_prompts,
                training_mode=training_mode,
            )
        except Exception as e:
            if attempt == t_cst.IMAGE_TASK_CREATION_MAX_ATTEMPTS:
                raise RuntimeError(f"Failed to create {context} after {attempt} attempts") from e
            logger.warning(f"Failed to create {context} (attempt {attempt}): {e}. Retrying...")
            await asyncio.sleep(t_cst.IMAGE_TASK_CREATION_RETRY_DELAY_SECONDS)


async def _image_models_of_type(config: Config, model_type: ImageModelType):
    """Yield only models matching model_type from the image pool."""
    async for model in _get_image_models(config.keypair):
        if model.model_type == model_type:
            yield model


async def _create_task_by_type(
    task_type: TaskType,
    config: Config,
    models: list,
    instruct_datasets: list,
    dpo_datasets: list,
    allow_augmentation: bool = True,
    training_mode: ImageTrainingMode | None = None,
    max_num_prompts: int | None = None,
) -> RawTask:
    """Create a synthetic task of the specified type.

    allow_augmentation is instruct-only: the large boss-round instruct slot never republishes a
    70B copy during model-prep.
    """
    if task_type == TaskType.IMAGETASK:
        return await create_synthetic_image_task(
            config, models, max_num_prompts=max_num_prompts, training_mode=training_mode
        )
    elif task_type == TaskType.INSTRUCTTEXTTASK:
        return await create_synthetic_instruct_text_task(
            config,
            models,
            instruct_datasets,
            enable_kl=True,
            allow_augmentation=allow_augmentation,
        )
    elif task_type == TaskType.DPOTASK:
        return await create_synthetic_dpo_task(config, models, dpo_datasets)
    elif task_type == TaskType.GRPOTASK:
        return await create_synthetic_grpo_task(config, models, instruct_datasets)
    else:
        # Default to instruct text task
        return await create_synthetic_instruct_text_task(
            config,
            models,
            instruct_datasets,
            enable_kl=True,
            allow_augmentation=allow_augmentation,
        )


async def _get_existing_tasks(existing_tournament_tasks: list, config: Config) -> list[RawTask]:
    tasks = []
    for task in existing_tournament_tasks:
        task_obj = await task_sql.get_task(task.task_id, config.psql_db)
        if task_obj:
            tasks.append(task_obj)
    return tasks


async def _get_existing_tasks_by_identifier(
    round_id: str, config: Config, group_id: str | None = None, pair_id: str | None = None
) -> list:
    """Get existing tournament tasks filtered by group_id or pair_id."""
    existing_tasks = await get_tournament_tasks(round_id, config.psql_db)
    if group_id:
        return [task for task in existing_tasks if task.group_id == group_id]
    elif pair_id:
        return [task for task in existing_tasks if task.pair_id == pair_id]
    return existing_tasks


async def _create_and_register_tournament_task(
    task: RawTask,
    tournament_id: str,
    round_id: str,
    config: Config,
    group_id: str | None = None,
    pair_id: str | None = None,
) -> None:
    """Create a TournamentTask, register it in the database, and log the creation."""
    tournament_task = TournamentTask(
        tournament_id=tournament_id,
        round_id=round_id,
        task_id=task.task_id,
        group_id=group_id,
        pair_id=pair_id,
    )
    await add_tournament_tasks([tournament_task], config.psql_db)
    gpu_req = get_task_gpu_requirement(task)

    # Format log message based on task type
    if task.task_type == TaskType.IMAGETASK:
        logger.info(f"Image: {task.task_id} - Model: {task.model_id} - GPU: {gpu_req}")
    else:
        dataset_info = f" - Dataset: {task.ds}" if hasattr(task, 'ds') and task.ds else ""
        duration_info = (
            f" - Duration: {task.hours_to_complete} hours"
            if hasattr(task, "hours_to_complete") and task.hours_to_complete
            else ""
        )
        task_type_info = f"{task.task_type.value}: " if hasattr(task.task_type, 'value') else ""
        logger.info(f"{task_type_info}{task.task_id} - Model: {task.model_id}{dataset_info} - GPU: {gpu_req}{duration_info}")


async def _create_group_text_tasks(
    round_data: GroupRound, tournament_id: str, config: Config, is_final_round: bool
) -> list[RawTask]:
    # Text round 1: a single group plays SMALL_TOURNAMENT_GROUP_TASKS instruct matches
    # (rather than one). It keeps the broader dataset range, but all R1 models stay <=4B.
    is_multi_task = is_single_group_text_round_one(round_data)
    tasks_per_group = t_cst.SMALL_TOURNAMENT_GROUP_TASKS if is_multi_task else t_cst.TEXT_TASKS_PER_GROUP

    models = _get_text_models(config.keypair, smallest_size_b=0.1, largest_size_b=4.0)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    tasks = []
    for i, group in enumerate(round_data.groups):
        logger.info(f"  Group {i + 1} ({len(group.member_ids)} members): creating {tasks_per_group} instruct task(s)")
        group_tasks = await _create_single_group_text_tasks(
            group,
            i,
            tournament_id,
            round_data.round_id,
            config,
            models,
            instruct_datasets,
            dpo_datasets,
            tasks_per_group,
            enable_kl=round_data.round_number != 1,
        )
        tasks.extend(group_tasks)

    return tasks


async def _create_single_group_text_tasks(
    group,
    group_index: int,
    tournament_id: str,
    round_id: str,
    config: Config,
    models: list,
    instruct_datasets: list,
    dpo_datasets: list,
    tasks_per_group: int,
    enable_kl: bool,
) -> list[RawTask]:
    group_id = f"{round_id}_group_{group_index + 1:03d}"

    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
    existing_count = len(existing_tasks)

    if existing_count >= tasks_per_group:
        logger.info(f"    Group {group_index + 1} already has {existing_count} task(s), skipping task creation")
        return await _get_existing_tasks(existing_tasks, config)

    created: list[RawTask] = await _get_existing_tasks(existing_tasks, config)
    for task_index in range(existing_count, tasks_per_group):
        logger.info(f"    Group {group_index + 1} has {len(created)}/{tasks_per_group} task(s), creating 1 more")
        if round_id.endswith("_round_001"):
            task = await _draw_round_one_text_task_within_hours(
                config,
                models,
                instruct_datasets,
                enable_kl=enable_kl,
            )
        else:
            task = await create_synthetic_instruct_text_task(
                config,
                models,
                instruct_datasets,
                enable_kl=enable_kl,
            )
        await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
        created.append(task)

    return created


async def _create_probability_based_text_tasks(
    round_data: KnockoutRound, tournament_id: str, config: Config
) -> list[RawTask]:
    num_tasks = len(round_data.pairs)
    models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    text_total = (
        PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
        + PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
        + PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO
    )
    instruct_prob = PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT / text_total
    dpo_prob = PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO / text_total

    tasks = []
    for i in range(num_tasks):
        pair = round_data.pairs[i]
        logger.info(f"  Pair {i + 1} ({pair[0]} vs {pair[1]}):")
        pair_id = f"{round_data.round_id}_pair_{i + 1:03d}"

        existing_tasks = await _get_existing_tasks_by_identifier(round_data.round_id, config, pair_id=pair_id)
        existing_count = len(existing_tasks)

        if existing_tasks:
            if existing_count > t_cst.KNOCKOUT_PAIR_TASKS:
                logger.warning(
                    f"   Pair {i + 1} has {existing_count} tasks when it should only have {t_cst.KNOCKOUT_PAIR_TASKS}!"
                )
            logger.info(f"    Pair {i + 1} already has {existing_count} task(s), skipping task creation")
            pair_task_objects = await _get_existing_tasks(existing_tasks, config)
            tasks.extend(pair_task_objects)
            continue

        logger.info(f"    Pair {i + 1} has no tasks, creating {t_cst.KNOCKOUT_PAIR_TASKS}")
        task = await _create_single_probability_task(
            config, models, instruct_datasets, dpo_datasets, instruct_prob, dpo_prob
        )

        await _create_and_register_tournament_task(
            task, tournament_id, round_data.round_id, config, pair_id=pair_id
        )
        tasks.append(task)
    return tasks


async def _create_single_probability_task(
    config: Config, models: list, instruct_datasets: list, dpo_datasets: list, instruct_prob: float, dpo_prob: float
) -> RawTask:
    rand_val = random.random()
    if rand_val < instruct_prob:
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets, enable_kl=True)
    elif rand_val < (instruct_prob + dpo_prob):
        return await create_synthetic_dpo_task(config, models, dpo_datasets)
    else:
        return await create_synthetic_grpo_task(config, models, instruct_datasets)


async def create_new_task_of_same_type(task: RawTask, config: Config, max_num_prompts: int | None = None) -> RawTask:
    """A new task of the same type (an image task keeps its family and training mode). max_num_prompts is image-only: the
    cap on the new task's image pairs."""
    if task.task_type == TaskType.IMAGETASK:
        model_type = getattr(task, "model_type", None)
        models = _image_models_of_type(config, model_type) if model_type is not None else _get_image_models(config.keypair)
        return await _create_task_by_type(
            task.task_type,
            config,
            models,
            [],
            [],
            training_mode=getattr(task, "training_mode", None),
            max_num_prompts=max_num_prompts,
        )

    if task.task_type in QUANT_TASK_TYPES:
        if not is_chained(task):
            raise ValueError(f"Quant task {task.task_id} is not chained; every tournament quant task is chained on a seed")
        # Pinned to its upstream and data (a compression task to its continuous chunk): the replacement is an identical held copy.
        return await recreate_chained_quant_task(config, task)

    model_params_b = int(task.model_params_count / t_cst.MODEL_PARAMS_TO_BILLIONS)

    # Handle case where model params is 0 or very small
    if model_params_b < t_cst.DEFAULT_MODEL_MIN_SIZE_B:
        logger.warning(
            f"Original task has very small model params ({task.model_params_count}), "
            f"using default range {t_cst.DEFAULT_MODEL_MIN_SIZE_B}-"
            f"{t_cst.DEFAULT_MODEL_MAX_SIZE_B}B"
        )
        models = _get_text_models(
            config.keypair, smallest_size_b=t_cst.DEFAULT_MODEL_MIN_SIZE_B, largest_size_b=t_cst.DEFAULT_MODEL_MAX_SIZE_B
        )
    else:
        models = _get_text_models(
            config.keypair,
            smallest_size_b=model_params_b * t_cst.MODEL_SIZE_RANGE_MULTIPLIER_MIN,
            largest_size_b=model_params_b * t_cst.MODEL_SIZE_RANGE_MULTIPLIER_MAX,
        )
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    return await _create_task_by_type(
        task.task_type,
        config,
        models,
        instruct_datasets,
        dpo_datasets,
    )


def _task_params_count(task) -> int:
    """Persisted param count, or a Hugging Face lookup when prep never wrote one (still 0)."""
    count = int(getattr(task, "model_params_count", 0) or 0)
    if count:
        return count
    model_id = getattr(task, "model_id", None)
    if not model_id:
        return 0
    try:
        fetched = get_model_num_params(model_id)
    except Exception:
        logger.warning(f"Could not fetch param count for {model_id} during replacement routing")
        return 0
    return int(fetched or 0)


def _is_boss_round_large_instruct_task(task, is_final_round: bool) -> bool:
    """The boss-round instruct slot that was forced onto the 30B–72B pool."""
    if not is_final_round or task.task_type != TaskType.INSTRUCTTEXTTASK:
        return False
    params_b = _task_params_count(task) / t_cst.MODEL_PARAMS_TO_BILLIONS
    return t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B <= params_b <= t_cst.BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B


def _is_round_one_group_text_task(task: RawTask, round_id: str, group_id: str | None, pair_id: str | None) -> bool:
    """Return True when task should follow round-1 group text constraints."""
    return (
        task.task_type == TaskType.INSTRUCTTEXTTASK
        and group_id is not None
        and pair_id is None
        and round_id.endswith("_round_001")
    )


def _round_one_text_hours_too_long(hours: float) -> bool:
    return float(hours) > t_cst.ROUND_ONE_TEXT_MAX_TRAINING_HOURS


async def _retire_unused_round_one_text_task(task: RawTask, config: Config) -> None:
    """Keep a too-long draw out of the pending queue. It was already inserted by task creation."""
    logger.info(
        f"Round-1 text task {task.task_id} is {task.hours_to_complete}h "
        f"> {t_cst.ROUND_ONE_TEXT_MAX_TRAINING_HOURS}h; retiring it and drawing another"
    )
    task.status = TaskStatus.FAILURE
    await task_sql.update_task(task, config.psql_db)


async def _draw_round_one_text_task_within_hours(
    config: Config,
    models,
    instruct_datasets,
    enable_kl: bool = False,
) -> RawTask:
    """Create an R1 instruct task with its natural hours, redrawing while those hours exceed the limit.

    Hours are never shortened. A draw over ROUND_ONE_TEXT_MAX_TRAINING_HOURS is marked failed
    so the pending-task lifecycle does not prep it, then another task is created until one fits.
    """
    attempt = 0
    while True:
        attempt += 1
        task = await create_synthetic_instruct_text_task(
            config,
            models,
            instruct_datasets,
            enable_kl=enable_kl,
        )
        if not _round_one_text_hours_too_long(float(task.hours_to_complete)):
            return task
        await _retire_unused_round_one_text_task(task, config)
        logger.info(f"Round-1 text duration draw {attempt} was {task.hours_to_complete}h")


async def _create_round_one_group_text_replacement_task(config: Config) -> RawTask:
    """
    Create a replacement task that matches round-1 group text constraints:
    - small text model pool (0.1B-4.0B)
    - broad dataset bins (same as R1 creation)
    - natural hours; redraw while the draw is longer than ROUND_ONE_TEXT_MAX_TRAINING_HOURS
    """
    models = _get_text_models(config.keypair, smallest_size_b=0.1, largest_size_b=4.0)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    return await _draw_round_one_text_task_within_hours(
        config,
        models,
        instruct_datasets,
        enable_kl=False,
    )


async def _create_boss_round_large_instruct_replacement_task(config: Config) -> RawTask:
    """Redraw the boss-round large instruct slot from the 30B–72B pool.

    Does not pin the failed model: that model is why we are replacing. Augmentation stays off so
    model-prep does not upload a full 70B copy.
    """
    models = _get_text_models(
        config.keypair,
        smallest_size_b=t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B,
        largest_size_b=t_cst.BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B,
    )
    return await create_synthetic_instruct_text_task(
        config,
        models,
        _get_instruct_text_datasets(config.keypair),
        enable_kl=True,
        allow_augmentation=False,
    )


async def _create_new_text_boss_round_tasks(tournament_id: str, round_id: str, config: Config) -> list[RawTask]:
    """Create boss round text tasks using new synthetic tasks."""
    pair_id = f"{round_id}_pair_001"

    existing_pair_tasks = await _get_existing_tasks_by_identifier(round_id, config, pair_id=pair_id)
    existing_count = len(existing_pair_tasks)

    if existing_count >= t_cst.FINAL_ROUND_TEXT_TASKS:
        logger.info(f"Final round already has {existing_count} tasks, skipping task creation")
        return await _get_existing_tasks(existing_pair_tasks, config)

    logger.info("Creating boss round text tasks using new synthetic tasks")

    standard_models = _get_text_models(config.keypair)
    large_instruct_models = _get_text_models(
        config.keypair,
        smallest_size_b=t_cst.BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B,
        largest_size_b=t_cst.BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B,
    )
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    existing_task_type_counts = {}
    existing_continuous_sft_lineages: set[str] = set()
    tasks = []

    for task in existing_pair_tasks:
        task_obj = await task_sql.get_task(task.task_id, config.psql_db)
        if not task_obj:
            continue
        if t_cst.is_continuous_sft_task(task_obj):
            lineage = t_cst.continuous_sft_lineage_from_ds(task_obj.ds)
            if lineage:
                existing_continuous_sft_lineages.add(lineage)
        else:
            task_type_value = task_obj.task_type.value if hasattr(task_obj.task_type, "value") else task_obj.task_type
            existing_task_type_counts[task_type_value] = existing_task_type_counts.get(task_type_value, 0) + 1
        tasks.append(task_obj)

    instruct_target = t_cst.FINAL_ROUND_TEXT_TASK_DISTRIBUTION.get(TaskType.INSTRUCTTEXTTASK, 0)
    for task_type, target_count in t_cst.FINAL_ROUND_TEXT_TASK_DISTRIBUTION.items():
        already = existing_task_type_counts.get(task_type.value, 0)
        for slot_index in range(already, target_count):
            # The last instruct-text slot is always a large (30B+) model, not the usual
            # standard/big-pool probability draw.
            # Augmentation is forced off: republishing a 70B copy during model-prep
            # regularly exceeds the 90-minute prep timeout.
            is_forced_large_instruct = task_type == TaskType.INSTRUCTTEXTTASK and slot_index == instruct_target - 1
            models = large_instruct_models if is_forced_large_instruct else standard_models
            task = await _create_single_new_text_task(
                task_type,
                tournament_id,
                round_id,
                pair_id,
                config,
                models,
                instruct_datasets,
                dpo_datasets,
                allow_augmentation=not is_forced_large_instruct,
            )
            if task:
                tasks.append(task)

    # Surface any state row whose lineage was renamed/removed (its accumulated chain is now orphaned).
    await warn_orphaned_continuous_sft_state(set(t_cst.CONTINUOUS_SFT_LINEAGES), config.psql_db)

    # One continuous-SFT task per lineage; skip lineages already created.
    for lineage, seed_model in t_cst.CONTINUOUS_SFT_LINEAGES.items():
        if lineage in existing_continuous_sft_lineages:
            continue
        task = await _create_continuous_sft_boss_task(tournament_id, round_id, pair_id, config, lineage, seed_model)
        tasks.append(task)

    return tasks


async def create_boss_round_decider_tasks(
    tournament_id: str,
    round_id: str,
    config: Config,
    decider_task_types: list[TaskType],
) -> list[RawTask]:
    """Add one replacement task per drawn boss-round task.

    Additive, not a replacement in the replace_tournament_task sense: the drawn task keeps its
    training, its eval rows and its per-example loss vectors, which are the only data that can tell
    us whether the tie dead zone is set too wide. It is simply excluded from the dethrone tally, and
    these tasks restore the count of decided results the tally runs over.

    Each decider mirrors the type that drew, so the boss round keeps the mix it was built with -
    except a drawn continuous-SFT task, which the caller maps to a plain instruct task on a fresh
    dataset pull. A drawn lineage has already cleared its own dethrone gate, so a second chunk of it
    would buy nothing while putting two tasks on one sequential train_index.

    Nodes are deliberately not copied from the drawn task: the round returns to PENDING and
    assign_nodes_to_tournament_tasks assigns both sides afresh, which gives each decider its own
    expected_repo_name. Copying the drawn task's name would point evaluation at the model that was
    already trained and uploaded for it.

    All-or-nothing. Each decider is registered in tournament_tasks as it is created, so a failure
    partway through would otherwise leave the round holding tasks nobody will ever assign nodes to -
    and since the decider cap keys off the round's task count, that surplus permanently blocks the
    retry from completing the batch, leaving a task that never ran to be scored against somebody.
    On failure the partial batch is deleted so the count is restored and the next cycle starts clean.
    """
    standard_models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)
    pair_id = f"{round_id}_pair_001"

    tasks: list[RawTask] = []
    try:
        for task_type in decider_task_types:
            models = standard_models
            logger.info(f"Creating {task_type.value} boss-round decider")
            task = await _create_single_new_text_task(
                task_type, tournament_id, round_id, pair_id, config, models, instruct_datasets, dpo_datasets
            )
            # _create_single_new_text_task swallows its errors and returns None. A missing decider
            # would leave the round one decided result short and resolve it on a smaller field than
            # intended, so raise instead and let the next cycle retry the whole batch.
            if task is None:
                raise RuntimeError(f"Failed to create {task_type.value} boss-round decider for round {round_id}")
            tasks.append(task)
    except Exception:
        for created in tasks:
            try:
                # Cascades to the tournament_tasks row, which is what restores the count.
                await task_sql.delete_task(created.task_id, config.psql_db)
                logger.info(f"Rolled back partially-created decider task {created.task_id}")
            except Exception as cleanup_error:
                logger.error(
                    f"Could not roll back decider task {created.task_id} for round {round_id}: {cleanup_error}. "
                    f"The round now holds a task that will never be assigned nodes - resolution will "
                    f"refuse to advance until it is removed.",
                    exc_info=True,
                )
        raise

    logger.info(f"Created {len(tasks)} boss-round decider task(s) for round {round_id}")
    return tasks


async def _create_continuous_sft_boss_task(
    tournament_id: str, round_id: str, pair_id: str, config: Config, lineage: str, seed_model: str
) -> RawTask:
    """Create + register one lineage's continuous-SFT boss task. Kept out of
    _create_single_new_text_task because it uses no random model/dataset pool.

    Raises on failure rather than swallowing: a dropped lineage silently weakens the boss-round
    dethrone gate (challenger must win ALL continuous-SFT tasks). Propagating keeps the round PENDING
    so the next cycle retries, and boss-round creation is idempotent (already-created lineages skip).
    create_continuous_sft_task already retries transient content-service failures via retry_with_backoff.
    """
    try:
        task = await create_continuous_sft_task(config, lineage, seed_model)
        await _create_and_register_tournament_task(task, tournament_id, round_id, config, pair_id=pair_id)
        return task
    except Exception as e:
        logger.error(f"Failed to create continuous-SFT boss task for lineage {lineage}: {e}", exc_info=True)
        raise


async def _create_single_new_text_task(
    task_type: TaskType,
    tournament_id: str,
    round_id: str,
    pair_id: str,
    config: Config,
    models: list,
    instruct_datasets: list,
    dpo_datasets: list,
    allow_augmentation: bool = True,
) -> RawTask | None:
    """Create a single new synthetic text task of a specific type."""
    try:
        if task_type not in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK]:
            logger.error(f"Unknown task type {task_type} for boss round text task")
            return None

        task = await _create_task_by_type(
            task_type,
            config,
            models,
            instruct_datasets,
            dpo_datasets,
            allow_augmentation=allow_augmentation,
        )
        await _create_and_register_tournament_task(
            task, tournament_id, round_id, config, pair_id=pair_id
        )
        return task
    except Exception as e:
        logger.error(f"Failed to create boss round {task_type.value} task: {e}", exc_info=True)
        return None


async def _planned_image_boss_round_specs(tournament_id: str, round_id: str, config: Config) -> tuple[ImageTaskSpec, ...]:
    """The slots of a planned image tournament's boss round: IMAGE_BOSS_ROUND_TASKS_PER_MODE of each training mode, drawn
    at random from the FINAL_ROUND_IMAGE_TASK_SPECS slots other than the one round 3 played. Slots the round already holds a
    task on are kept, so a rerun draws only for the tasks still missing."""
    final_miner_round = t_cst.IMAGE_ROUND_PLAN[-1].round_number
    round_three = next(
        (r for r in await get_tournament_rounds(tournament_id, config.psql_db) if r.round_number == final_miner_round), None
    )
    round_three_tasks = await get_tournament_tasks(round_three.round_id, config.psql_db) if round_three else []
    if not round_three_tasks:
        raise ValueError(f"Tournament {tournament_id} has no round {final_miner_round} task to draw its boss round around")
    round_three_task = await task_sql.get_task(round_three_tasks[0].task_id, config.psql_db)
    round_three_spec = (round_three_task.model_type, round_three_task.training_mode)

    existing = await _get_existing_tasks(
        await _get_existing_tasks_by_identifier(round_id, config, pair_id=f"{round_id}_pair_001"), config
    )
    specs = [(task.model_type, task.training_mode) for task in existing]
    for training_mode, count in t_cst.IMAGE_BOSS_ROUND_TASKS_PER_MODE.items():
        open_slots = [
            spec
            for spec in t_cst.FINAL_ROUND_IMAGE_TASK_SPECS
            if spec[1] == training_mode and spec != round_three_spec and spec not in specs
        ]
        missing = count - sum(1 for spec in specs if spec[1] == training_mode)
        specs.extend(random.sample(open_slots, max(missing, 0)))
    return tuple(specs)


async def _create_new_image_boss_round_tasks(
    tournament_id: str,
    round_id: str,
    config: Config,
    specs: tuple[ImageTaskSpec, ...],
) -> list[RawTask]:
    """Create the boss round's image tasks, one per (model type, training mode) slot of specs, using new synthetic tasks.
    Idempotent per slot."""
    pair_id = f"{round_id}_pair_001"
    total_tasks = len(specs)

    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, pair_id=pair_id)
    existing_count = len(existing_tasks)

    if existing_count >= total_tasks:
        logger.info(f"Final round already has {existing_count} tasks, skipping task creation")
        return await _get_existing_tasks(existing_tasks, config)

    logger.info("Creating boss round image tasks using new synthetic tasks")

    existing_task_objects = await _get_existing_tasks(existing_tasks, config)
    tasks = list(existing_task_objects)
    filled_specs = {
        (getattr(task, "model_type", None), getattr(task, "training_mode", None))
        for task in existing_task_objects
    }

    for index, (model_type, training_mode) in enumerate(specs):
        if len(tasks) >= total_tasks:
            break
        if (model_type, training_mode) in filled_specs:
            continue
        model_gen = _image_models_of_type(config, model_type)
        try:
            task = await _create_single_image_task_with_retry(
                config,
                model_gen,
                index,
                is_final=True,
                training_mode=training_mode,
            )
            await _create_and_register_tournament_task(task, tournament_id, round_id, config, pair_id=pair_id)
            tasks.append(task)
            filled_specs.add((model_type, training_mode))
        except Exception as e:
            logger.error(
                f"Failed to create final {model_type.value} {training_mode.value} task: {e}",
                exc_info=True,
            )

    return tasks


async def _image_replacement_max_num_prompts(task: RawTask, tournament_id: str, round_id: str, config: Config) -> int | None:
    """The pair cap a replacement image task is drawn with, the same as its round's tasks: ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS
    for a round-1 task, none for any other round or task type."""
    if task.task_type != TaskType.IMAGETASK:
        return None
    rounds = await get_tournament_rounds(tournament_id, config.psql_db)
    round_number = next((r.round_number for r in rounds if r.round_id == round_id), None)
    return t_cst.ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS if round_number == 1 else None


async def replace_tournament_task(
    original_task_id: str,
    tournament_id: str,
    round_id: str,
    group_id: str | None,
    pair_id: str | None,
    config: Config,
    is_final_round: bool = False,
) -> str:
    logger.info(f"Starting task replacement for task {original_task_id}")
    logger.info(f"Tournament: {tournament_id}, Round: {round_id}, Group: {group_id}, Pair: {pair_id}")

    original_task_obj = await task_sql.get_task(original_task_id, config.psql_db)
    if not original_task_obj:
        logger.error(f"Could not find original task {original_task_id}")
        raise ValueError(f"Original task {original_task_id} not found")

    logger.info(f"Found original task - Type: {original_task_obj.task_type}, Status: {original_task_obj.status}")
    logger.info(f"Original task model params: {original_task_obj.model_params_count}")

    is_quant_task = original_task_obj.task_type in QUANT_TASK_TYPES
    # A miner-round task of a seeded text tournament: replaced by an ordinary instruct task of the round's own, which the boss
    # never trained, so the boss's node is not copied onto it (a seed whose role failed no longer has the boss's run behind it).
    is_seeded_text_round_task = (
        not is_quant_task and group_id is not None and await seeds_sql.is_seeded_text_tournament(tournament_id, config.psql_db)
    )

    # A quant task whose slot was already moved to another task (an earlier pass stopped before finishing) is completed with
    # that task; nothing new is created.
    adopted = await _live_quant_replacement(original_task_id, config) if is_quant_task else None
    resumed = adopted is not None  # the new task is one an earlier pass created: it may be registered or assigned already

    try:
        if adopted is not None:
            logger.warning(f"Task {original_task_id} was already replaced by {adopted.task_id}; completing that replacement")
            new_task = adopted
        elif is_quant_task:
            new_task = await create_new_task_of_same_type(original_task_obj, config)
        elif is_seeded_text_round_task:
            round_number = next(
                round_data.round_number
                for round_data in await get_tournament_rounds(tournament_id, config.psql_db)
                if round_data.round_id == round_id
            )
            logger.info(f"Replacing the task of seeded text round {round_number} with a new task drawn like its seed")
            new_task = await draw_text_round_task(text_round_spec(round_number), config)
        elif _is_round_one_group_text_task(original_task_obj, round_id, group_id, pair_id):
            logger.info(
                "Detected round-1 group text task replacement; redrawing a small-model task "
                f"whose natural hours are within {t_cst.ROUND_ONE_TEXT_MAX_TRAINING_HOURS}h"
            )
            new_task = await _create_round_one_group_text_replacement_task(config)
        elif t_cst.is_continuous_sft_task(original_task_obj):
            # Same lineage, same carried base model; the content service re-materializes the chunk
            # at fresh S3 URLs. Without this branch, create_new_task_of_same_type has no CHATTASK
            # route and would fall through to a random-model instruct task, dropping the lineage
            # from the boss round and weakening the win-all-continuous-SFT dethrone gate.
            lineage = t_cst.continuous_sft_lineage_from_ds(original_task_obj.ds)
            seed_model = t_cst.continuous_sft_seed_repo(lineage)
            if not lineage or not seed_model:
                raise ValueError(
                    f"Cannot replace continuous-SFT task {original_task_id}: unknown lineage in ds {original_task_obj.ds!r}"
                )
            logger.info(f"Detected continuous-SFT task replacement; recreating lineage {lineage}")
            new_task = await create_continuous_sft_task(config, lineage, seed_model)
        elif _is_boss_round_large_instruct_task(original_task_obj, is_final_round):
            # The 30B–72B instruct slot is identified by size, not model id: pinning the failed
            # repo would retry the model that just failed prep. Redraw from the large pool and
            # keep augmentation off (a 70B republish blows the model-prep timeout).
            logger.info("Detected boss-round large instruct replacement; redrawing from the large-instruct pool")
            new_task = await _create_boss_round_large_instruct_replacement_task(config)
        else:
            new_task = await create_new_task_of_same_type(
                original_task_obj,
                config,
                max_num_prompts=await _image_replacement_max_num_prompts(original_task_obj, tournament_id, round_id, config),
            )
        logger.info(f"Successfully created new task {new_task.task_id} of type {new_task.task_type}")
    except Exception as e:
        logger.error(f"Failed to create new task of type {original_task_obj.task_type}: {str(e)}", exc_info=True)
        raise

    if is_quant_task and adopted is None:
        # A task of a bound quant tournament keeps its place and its link: the replacement takes the slot's source row, the
        # replaced task's row is superseded in the same step. A replacement that cannot be linked is deleted, and the task
        # that holds the slot (a pass that got there first) is the one completed.
        original_source = await quant_sources_sql.get_quant_task_source(original_task_id, config.psql_db)
        if original_source is not None:
            linked = await quant_sources.record_new_quant_task(
                original_source.model_copy(
                    update={
                        "quant_task_id": UUID(str(new_task.task_id)),
                        "source_model_repo": None,
                        "source_is_fallback": False,
                        "superseded": False,
                    }
                ),
                config,
                replacing_task_id=UUID(str(original_task_id)),
            )
            if linked.quant_task_id != UUID(str(new_task.task_id)):
                holder = await task_sql.get_task(linked.quant_task_id, config.psql_db)
                if holder is None:
                    raise RuntimeError(f"Slot {linked.slot.value} holds task {linked.quant_task_id}, which is gone")
                logger.warning(f"Slot {linked.slot.value} already has task {holder.task_id}; completing that replacement")
                new_task, resumed = holder, True

    try:
        # A task an earlier pass created may be registered already; a fresh one never is.
        registered = False
        if resumed:
            registered = await tournament_sql.get_tournament_task_by_task_id(str(new_task.task_id), config.psql_db) is not None
        if not registered:
            await _create_and_register_tournament_task(
                new_task, tournament_id, round_id, config, group_id=group_id, pair_id=pair_id
            )
            logger.info(f"Created replacement task {new_task.task_id} for round {round_id}")
    except Exception as e:
        logger.error(f"Failed to add tournament task to database: {str(e)}", exc_info=True)
        raise

    if is_quant_task:
        await _repoint_chained_dependents(original_task_obj, new_task, round_id, config)

    original_assigned_nodes = await task_sql.get_nodes_assigned_to_task(original_task_id, config.psql_db)
    already_assigned: set[str] = set()
    if resumed:
        already_assigned = {node.hotkey for node in await task_sql.get_nodes_assigned_to_task(new_task.task_id, config.psql_db)}
    # A miner-round task of a quant tournament (its round seed's role failed) has no boss run behind its replacement either.
    is_quant_round_task = is_quant_task and group_id is not None
    for node in original_assigned_nodes:
        if (is_seeded_text_round_task or is_quant_round_task) and node.hotkey == EMISSION_BURN_HOTKEY:
            continue
        if node.hotkey not in already_assigned:
            await task_sql.assign_node_to_task(new_task.task_id, node, config.psql_db)

        original_expected_repo_name = await task_sql.get_expected_repo_name(original_task_id, node.hotkey, config.psql_db)
        if original_expected_repo_name:
            await task_sql.set_expected_repo_name(new_task.task_id, node, config.psql_db, original_expected_repo_name)
            logger.info(
                f"Copied node {node.hotkey} with expected_repo_name "
                f"{original_expected_repo_name} to replacement task {new_task.task_id}"
            )
        else:
            logger.warning(f"No expected repo name found for node {node.hotkey} in original task {original_task_id}")
        if is_quant_task:
            await _copy_own_model_state(original_task_id, new_task.task_id, node, config)

    await task_sql.detach_task_from_tournament(original_task_id, tournament_id, config.psql_db)
    logger.info(f"Detached original task {original_task_id} from tournament {tournament_id}; task row kept.")

    return new_task.task_id


async def _live_quant_replacement(original_task_id: str, config: Config) -> RawTask | None:
    """The task that already took a replaced quant task's slot, or None.

    The original's source row is superseded once a replacement was recorded; the slot's live row then names that task."""
    original_source = await quant_sources_sql.get_quant_task_source(original_task_id, config.psql_db)
    if original_source is None or not original_source.superseded:
        return None
    live = await quant_sources_sql.get_live_quant_task_source(
        original_source.quant_tournament_id, original_source.slot, config.psql_db
    )
    if live is None or str(live.quant_task_id) == str(original_task_id):
        return None
    return await task_sql.get_task(live.quant_task_id, config.psql_db)


async def create_quant_tournament_tasks(
    round_data: Round,
    tournament_id: str,
    config: Config,
    is_final_round: bool = False,
) -> list[str]:
    round_id = round_data.round_id
    if isinstance(round_data, GroupRound):
        logger.info(f"Creating quant tournament round {round_data.round_number}: one quant task on a seed's output")
        tasks = await _create_seed_input_round_task(round_data, tournament_id, config)
    elif is_final_round:
        logger.info(
            f"Attaching the final quant tournament boss round: {t_cst.FINAL_ROUND_QUANT_TASKS} seed tasks "
            f"({t_cst.FINAL_ROUND_QUANT_TASK_DISTRIBUTION})"
        )
        tasks = await _create_quant_boss_round_tasks(tournament_id, round_id, config)
    else:
        raise ValueError(f"Quant tournament round {round_id} is neither a group round nor the boss round")

    return [str(task.task_id) for task in tasks]


def quant_round_spec(round_number: int) -> QuantRoundSpec:
    """The QUANT_ROUND_PLAN entry of a miner round."""
    for spec in t_cst.QUANT_ROUND_PLAN:
        if spec.round_number == round_number:
            return spec
    raise ValueError(f"Quant round {round_number} is not a miner round (QUANT_ROUND_PLAN has {len(t_cst.QUANT_ROUND_PLAN)})")


async def _create_seed_input_round_task(round_data: GroupRound, tournament_id: str, config: Config) -> list[RawTask]:
    """R1-R3: the round's one quant task is the round's quant seed (QUANT_ROUND_PLAN: round 1 quant_r1 on the text boss's
    text_r1 model, round 2 quant_r2 on its text_r2 model, round 3 quant_r3 on its text_r3 model).

    The quant boss trains the seed as soon as its source model lands; the round registers the seed task under its group and
    marks it attached, so the round's miners train the same task and the boss's entry is ranked with theirs. Idempotent: a
    round that already has its task returns it (and sets a marker a stop left unset). Raises MissingBossRoundSeeds while the
    role has no live seed and is not `failed` (the tournament is not bound yet, or a replacement is pending), so the round
    stays PENDING and is retried. A `failed` role (out of replacements) does not block the round: it plays a task of its own
    with no boss entry (_create_round_task_without_boss).
    """
    round_id = round_data.round_id
    spec = quant_round_spec(round_data.round_number)
    group_id = f"{round_id}_group_001"
    existing = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
    if existing:
        logger.info(f"Round {round_data.round_number} already has its quant task, skipping task creation")
        held = await _get_existing_tasks(existing, config)
        for task in held:
            if await seeds_sql.get_seed_task_by_task_id(task.task_id, config.psql_db) is not None:
                await seeds_sql.mark_seed_attached(task.task_id, round_id, config.psql_db)
        return held

    seed_role = QUANT_SLOT_SPECS[spec.slot].quant_seed_role
    seed = next(
        (row for row in await seeds_sql.get_active_seed_tasks(tournament_id, config.psql_db) if row.role == seed_role), None
    )
    if seed is None:
        if not await _role_is_failed(tournament_id, seed_role, config.psql_db):
            raise MissingBossRoundSeeds(tournament_id, [seed_role.value])
        return [await _create_round_task_without_boss(round_data, tournament_id, config)]
    task = await task_sql.get_task(seed.task_id, config.psql_db)
    if task is None:
        raise RuntimeError(f"Tournament {tournament_id}: seed {seed_role.value} task {seed.task_id} is gone")
    await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
    # Durable "ever attached" marker (idempotent): from here on the seed is only replaced for a prep failure.
    await seeds_sql.mark_seed_attached(task.task_id, round_id, config.psql_db)
    logger.info(f"Round {round_id}: attached seed {seed_role.value} task {task.task_id}")
    return [task]


async def _create_round_task_without_boss(round_data: GroupRound, tournament_id: str, config: Config) -> RawTask:
    """The task of a quant round whose seed role is `failed`: a new held quant task on the round's text seed, with no seed
    row and no boss assignment, so the boss has no entry on it. It supersedes the failed seed's source row for the slot.

    Idempotent: a live source row of the slot whose task is not a seed (a stop between creation and registration) is
    registered instead of creating another task. The fixed format is the round's pinned one, else drawn from a round-seeded
    rng.
    """
    round_id = round_data.round_id
    spec = quant_round_spec(round_data.round_number)
    group_id = f"{round_id}_group_001"
    binding = await quant_sources.bind_quant_source_tournament(tournament_id, config)
    if binding is None:
        raise MissingBossRoundSeeds(tournament_id, [spec.input_role.value])
    task = None
    live = await quant_sources_sql.get_live_quant_task_source(tournament_id, spec.slot, config.psql_db)
    replacing_task_id = None
    if live is not None:
        if await seeds_sql.get_seed_task_by_task_id(live.quant_task_id, config.psql_db) is None:
            task = await task_sql.get_task(live.quant_task_id, config.psql_db)
            logger.warning(f"Round {round_id}: registering the existing task {live.quant_task_id} of slot {spec.slot.value}")
        else:
            replacing_task_id = live.quant_task_id
    if task is None:
        text_seed = await quant_sources.slot_text_seed(binding, spec.slot, config)
        if text_seed is None:
            raise MissingBossRoundSeeds(tournament_id, [spec.input_role.value])
        upstream = await task_sql.get_task(text_seed.task_id, config.psql_db)
        if upstream is None:
            raise ValueError(
                f"Text seed task {text_seed.task_id} ({spec.input_role.value}) of tournament {text_seed.tournament_id} not found"
            )
        quant_format = spec.quant_format
        if quant_format is None and spec.task_type == TaskType.QUANTFIXEDTASK:
            quant_format = draw_quant_format(random.Random(f"{round_id}:quant_format:0"))
        created = await create_chained_quant_task(config, spec.task_type, upstream, quant_format=quant_format, boss=False)
        source = await quant_sources.record_new_quant_task(
            quant_sources.new_task_source(UUID(str(created.task_id)), binding, spec.slot, text_seed), config, replacing_task_id
        )
        is_ours = source.quant_task_id == UUID(str(created.task_id))
        task = created if is_ours else await task_sql.get_task(source.quant_task_id, config.psql_db)
        if task is None:
            raise RuntimeError(f"Round {round_id}: the task {source.quant_task_id} of slot {spec.slot.value} is gone")
    if await tournament_sql.get_tournament_task_by_task_id(str(task.task_id), config.psql_db) is None:
        await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
    logger.warning(f"Round {round_id}: seed of slot {spec.slot.value} failed; plays its own task {task.task_id} (no boss entry)")
    return task


async def _get_round_one_tasks(tournament_id: str, psql_db: PSQLDB) -> list[TournamentTask]:
    rounds = await get_tournament_rounds(tournament_id, psql_db)
    if not rounds:
        return []
    r1 = min(rounds, key=lambda r: r.round_number)
    return await get_tournament_tasks(r1.round_id, psql_db)


async def own_r1_model_repo(tournament_id: str, hotkey: str, psql_db: PSQLDB) -> str | None:
    """The repo OUR trainer uploaded for hotkey on its best-ranked R1 task, if it is a full model.

    The repo is in the tournament namespace, so an entrant cannot swap in a different model. None (with the
    reason logged) when hotkey has no ranked R1 result or expected repo, or the repo is an adapter or missing.
    """
    r1_tasks = await _get_round_one_tasks(tournament_id, psql_db)
    ranks_by_task = {task.task_id: await _task_ranks(task.task_id, psql_db) for task in r1_tasks}
    task_id = _best_ranked_task(hotkey, ranks_by_task)
    if task_id is None:
        logger.warning(f"{hotkey} has no ranked R1 result in {tournament_id}; no own R1 model")
        return None
    repo_name = await task_sql.get_expected_repo_name(task_id, hotkey, psql_db)
    if not repo_name:
        logger.warning(f"{hotkey} has no expected R1 repo on task {task_id}; no own R1 model")
        return None
    repo = f"{RAYONLABS_HF_USERNAME}/{repo_name}"
    if not await _is_full_model_repo(repo):
        logger.warning(f"{hotkey}'s R1 repo {repo} is not a full model; no own R1 model")
        return None
    return repo


async def _task_ranks(task_id: str, psql_db: PSQLDB) -> dict[str, int]:
    """hotkey -> 1-based rank on a finished task, for ranked results only."""
    results = calculate_miner_ranking_and_scores(await get_task_results_for_ranking(task_id, psql_db))
    return {result.hotkey: result.rank for result in results if result.rank is not None}


def _best_ranked_task(hotkey: str, ranks_by_task: dict[str, dict[str, int]]) -> str | None:
    ranked = [(ranks[hotkey], task_id) for task_id, ranks in ranks_by_task.items() if hotkey in ranks]
    return min(ranked, key=lambda item: item[0])[1] if ranked else None


async def _is_full_model_repo(repo_id: str) -> bool:
    """Whether a Hugging Face repo exists and holds a loadable full model (quant_input_repo.full_model_rejection).

    The hub check is retried once, so a transient hub error does not pass over a usable repo.
    """
    for attempt in range(2):
        try:
            rejection = await full_model_rejection(repo_id)
            if rejection is not None:
                logger.warning(f"{repo_id} is not a full model: {rejection}")
            return rejection is None
        except Exception as e:
            logger.warning(f"Could not list files of {repo_id} (attempt {attempt + 1}/2): {e}")
            if attempt == 0:
                await asyncio.sleep(t_cst.QUANT_INPUT_LISTING_RETRY_DELAY_SECONDS)
    return False


def _quant_models(config: Config) -> AsyncGenerator[str, None]:
    """Shape-compatible dense models in QUANT_MODEL_SIZE_B: the pool every drawn quant task takes its model from."""
    return filter_quant_compatible(_get_text_models(config.keypair, *QUANT_MODEL_SIZE_B))


async def _attach_boss_round_seeds(
    seeds: dict[SeedRole, TournamentSeedTask], tournament_id: str, round_id: str, pair_id: str, config: Config
) -> list[RawTask]:
    """Register the boss round's compression seed tasks under the round's pair; nothing is drawn for a live seed.

    Every role that has a live seed is attached. A role whose latest row is `failed` (out of replacements) does not halt the
    tournament or block the round: the round plays a task of its own for that slot with no boss entry
    (_create_boss_round_task_without_boss), which a challenger with a valid entry takes against the boss, so the crown is
    still decided over seven tasks. A role with no live seed and no failed row (a replacement is pending) raises, so the
    round stays PENDING and is retried instead of running short of tasks. A tournament whose seeds are the two compressions
    chained on a BOSS_INSTRUCT seed attaches those two. A tournament that also holds rows of BOSS_R2_FIXED and
    BOSS_R3_FREEFORM attaches them too; its boss round then holds more tasks than the crown rule is defined on, and the boss
    retains (round_results._get_quant_boss_round_outcome). Idempotent: a task already registered in the round is not
    inserted again.
    """
    latest = {row.role: row for row in await seeds_sql.get_latest_seed_tasks(tournament_id, config.psql_db)}
    if latest.keys() & RETIRED_BOSS_ROUND_SEED_ROLES:
        role_set = RETIRED_BOSS_ROUND_SEED_ROLES
    else:
        role_set = BOSS_ROUND_SEED_ROLES | (latest.keys() & UNCREATED_BOSS_ROUND_SEED_ROLES)
    roles = [role for role in SeedRole if role in role_set]
    failed = {role for role, row in latest.items() if row.status == SeedStatus.FAILED}
    missing = [role.value for role in roles if role not in seeds and role not in failed]
    if missing:
        raise MissingBossRoundSeeds(tournament_id, missing)
    registered = {str(task.task_id) for task in await _get_existing_tasks_by_identifier(round_id, config, pair_id=pair_id)}
    tasks: list[RawTask] = []
    for role in roles:
        if role not in seeds:
            if role in QUANT_SLOT_BY_SEED_ROLE:
                tasks.append(await _create_boss_round_task_without_boss(role, tournament_id, round_id, pair_id, config))
            else:
                logger.error(
                    f"Boss round {round_id}: seed {role.value} failed and has no slot to play a task of its own; the round "
                    f"goes on without it and the crown cannot be decided over seven tasks"
                )
            continue
        task = await task_sql.get_task(seeds[role].task_id, config.psql_db)
        if task is None:
            raise RuntimeError(f"Tournament {tournament_id}: boss-round seed {role.value} task {seeds[role].task_id} is gone")
        if str(task.task_id) not in registered:
            await _create_and_register_tournament_task(task, tournament_id, round_id, config, pair_id=pair_id)
            logger.info(f"Boss round {round_id}: attached seed {role.value} task {task.task_id}")
        # Durable "ever attached" marker (idempotent): the seed is never replaced from here on, whatever happens to its row.
        await seeds_sql.mark_seed_attached(task.task_id, round_id, config.psql_db)
        tasks.append(task)
    if not tasks:
        raise MissingBossRoundSeeds(tournament_id, [role.value for role in roles])
    return tasks


async def _create_boss_round_task_without_boss(
    role: SeedRole, tournament_id: str, round_id: str, pair_id: str, config: Config
) -> RawTask:
    """The boss-round task of a slot whose seed role is `failed`: a new held quant task drawn like the role's seed (same text
    seed, format and budget), registered under the round's pair with no seed row, so the boss has no entry on it and is not
    assigned to it (assign_nodes_to_tournament_tasks). It supersedes the failed seed's source row for the slot.

    Idempotent: a live source row of the slot whose task is not a seed (a stop between creation and registration) is
    registered instead of drawing another task. Raises MissingBossRoundSeeds while the quant tournament is not bound or the
    slot's text seed has no final row (a text replacement is pending), so the round stays PENDING, alerted once, and is
    retried.
    """
    # seed_tasks imports this module, so the draw is imported here.
    from validator.tournament.seed_tasks import draw_compression_task

    slot = QUANT_SLOT_BY_SEED_ROLE[role]
    text_role = QUANT_SLOT_SPECS[slot].text_role
    binding = await quant_sources.bind_quant_source_tournament(tournament_id, config)
    if binding is None:
        raise MissingBossRoundSeeds(tournament_id, [text_role.value])
    task = None
    replacing_task_id = None
    live = await quant_sources_sql.get_live_quant_task_source(tournament_id, slot, config.psql_db)
    if live is not None:
        if await seeds_sql.get_seed_task_by_task_id(live.quant_task_id, config.psql_db) is None:
            task = await task_sql.get_task(live.quant_task_id, config.psql_db)
        else:
            replacing_task_id = live.quant_task_id
    if task is None:
        text_seed = await quant_sources.slot_text_seed(binding, slot, config)
        if text_seed is None:
            raise MissingBossRoundSeeds(tournament_id, [text_role.value])
        created = await draw_compression_task(role, tournament_id, text_seed, config)
        source = await quant_sources.record_new_quant_task(
            quant_sources.new_task_source(UUID(str(created.task_id)), binding, slot, text_seed), config, replacing_task_id
        )
        is_ours = source.quant_task_id == UUID(str(created.task_id))
        task = created if is_ours else await task_sql.get_task(source.quant_task_id, config.psql_db)
        if task is None:
            raise RuntimeError(f"Boss round {round_id}: the task {source.quant_task_id} of slot {slot.value} is gone")
    if await tournament_sql.get_tournament_task_by_task_id(str(task.task_id), config.psql_db) is None:
        await _create_and_register_tournament_task(task, tournament_id, round_id, config, pair_id=pair_id)
    logger.warning(f"Boss round {round_id}: seed {role.value} failed; plays its own task {task.task_id} (no boss entry)")
    return task


async def _create_quant_boss_round_tasks(tournament_id: str, round_id: str, config: Config) -> list[RawTask]:
    """The boss round's tasks: the tournament's compression seed tasks attached; nothing is drawn here.

    A quant tournament always has seeds (created at first-round creation); without them there is nothing to attach and drawing
    boss-side work now would train the boss after R3, so this raises.
    """
    pair_id = f"{round_id}_pair_001"
    seeds = {seed.role: seed for seed in await seeds_sql.get_active_seed_tasks(tournament_id, config.psql_db)}
    if not seeds:
        raise MissingBossRoundSeeds(tournament_id, [role.value for role in SeedRole if role in BOSS_ROUND_SEED_ROLES])
    return await _attach_boss_round_seeds(seeds, tournament_id, round_id, pair_id, config)


async def _copy_own_model_state(original_task_id: str, new_task_id: str, node: Node, config: Config) -> None:
    """Carry a node's starting model and persisted quant failure onto a replacement task, so a replaced
    own-model task keeps every miner's own R1 model and the no-usable-model fails."""
    starting_model = await task_sql.get_starting_model_repo(original_task_id, node.hotkey, config.psql_db)
    if starting_model:
        await task_sql.set_starting_model_repo(new_task_id, node.hotkey, starting_model, config.psql_db)
    decode_rows = await submissions_and_scoring_sql.get_quant_decode_rows(original_task_id, config.psql_db)
    row = decode_rows.get(node.hotkey)
    if row and row.quant_failure:
        await submissions_and_scoring_sql.set_task_node_quant_failure(new_task_id, node.hotkey, row.quant_failure, config.psql_db)


async def _repoint_chained_dependents(original_task: RawTask, new_task: RawTask, round_id: str, config: Config) -> None:
    """Move the round's chained tasks off a replaced upstream onto its replacement, so they are released
    on (and take their base, data and inputs from) the task that actually runs."""
    for tournament_task in await get_tournament_tasks(round_id, config.psql_db):
        task = await task_sql.get_task(tournament_task.task_id, config.psql_db)
        if not task or not is_chained(task) or str(task.upstream_task_id) != str(original_task.task_id):
            continue
        repoint_chained_quant_task(task, new_task, pinned_data=task.ds == original_task.ds)
        await task_sql.update_task(task, config.psql_db)
        logger.info(f"Re-pointed chained task {task.task_id} from upstream {original_task.task_id} to {new_task.task_id}")
