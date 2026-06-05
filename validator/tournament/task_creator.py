import random
from datetime import datetime
from datetime import timedelta

from core.constants import EnvironmentName
from core.constants import TrainingStartPoint
from core.models.tournament_models import CompositeSubtask
from core.models.tournament_models import GroupRound
from core.models.tournament_models import TournamentType
from core.models.tournament_models import KnockoutRound
from core.models.tournament_models import Round
from core.models.tournament_models import TournamentTask
from core.models.utility_models import ImageModelType
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.constants import NULL_ACCOUNT_ID
from validator.core.models import CompositeRawTask
from validator.core.models import RawTask
from validator.db.sql import tasks as task_sql
from validator.db.sql.tournaments import add_composite_task_subtasks
from validator.db.sql.tournaments import add_tournament_tasks
from validator.db.sql.tournaments import get_composite_task_subtasks
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.tasks.diffusion_synth import create_synthetic_image_task
from validator.tasks.synthetic_scheduler import _get_dpo_datasets
from validator.tasks.synthetic_scheduler import _get_image_models
from validator.tasks.synthetic_scheduler import _get_instruct_text_datasets
from validator.tasks.synthetic_scheduler import _get_text_models
from validator.tasks.synthetic_scheduler import create_synthetic_dpo_task
from validator.tasks.synthetic_scheduler import create_synthetic_env_task
from validator.tasks.synthetic_scheduler import create_synthetic_grpo_task
from validator.tasks.synthetic_scheduler import create_synthetic_instruct_text_task
from validator.tournament import constants as t_cst
from validator.tournament.gpu import get_tournament_gpu_requirement
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def create_image_tournament_tasks(
    round_data: Round, tournament_id: str, config: Config, is_final_round: bool = False,
) -> list[str]:
    round_id = round_data.round_id
    image_models = _get_image_models(config.keypair)
    tasks = []

    if isinstance(round_data, GroupRound):
        tasks = await _create_group_image_tasks(round_data, tournament_id, config, image_models)
    elif is_final_round:
        tasks = await _create_new_image_boss_round_tasks(tournament_id, round_id, config)
    else:
        tasks = await _create_knockout_image_tasks(round_data, tournament_id, config, image_models)

    return [str(task.task_id) for task in tasks]


async def create_environment_tournament_tasks(
    round_data: Round, tournament_id: str, config: Config, is_final_round: bool = False,
) -> list[str]:
    """Create environment tournament tasks."""
    if not isinstance(round_data, GroupRound):
        raise ValueError("Environment tournaments only support group rounds")

    if is_final_round:
        tasks = await _create_environment_boss_round_tasks(round_data, tournament_id, config)
    else:
        tasks = await _create_environment_group_tasks(round_data, tournament_id, config)
    return [str(task.task_id) for task in tasks]


async def _get_tournament_base_model(tournament_id: str, config: Config) -> str | None:
    """Look up the base model used in R1 of this tournament so all rounds use the same model."""
    rounds = await get_tournament_rounds(tournament_id, config.psql_db)
    if not rounds:
        return None
    r1 = min(rounds, key=lambda r: r.round_number)
    r1_tasks = await get_tournament_tasks(r1.round_id, config.psql_db)
    if not r1_tasks:
        return None
    task_obj = await task_sql.get_task(r1_tasks[0].task_id, config.psql_db)
    return task_obj.model_id if task_obj else None


async def _get_prev_tourn_winner_model(tournament_id: str, config: Config) -> str:
    """Get the previous tournament winner's model for the PREVIOUS_WINNER boss task.

    Returns the winner's HF repo if available and base-compatible, else ENV_TARGET_TOURN_MODEL.
    """
    prev_tournament = await get_latest_completed_tournament(
        config.psql_db, TournamentType.ENVIRONMENT, exclude_tournament_id=tournament_id,
    )

    if prev_tournament and prev_tournament.winner_model_repo:
        if prev_tournament.winner_model_base == t_cst.ENV_TARGET_TOURN_MODEL:
            logger.info(f"Final task 3: winner continuation from {prev_tournament.winner_model_repo}")
            return prev_tournament.winner_model_repo
        logger.info(f"Final task 3: base changed, from-scratch on {t_cst.ENV_TARGET_TOURN_MODEL}")
    else:
        logger.info(f"Final task 3: no previous winner, from-scratch on {t_cst.ENV_TARGET_TOURN_MODEL}")

    return t_cst.ENV_TARGET_TOURN_MODEL


async def _create_environment_boss_round_tasks(
    round_data: GroupRound, tournament_id: str, config: Config,
) -> list[RawTask]:
    """Create 3 final round tasks with different starting points.

    Task 1: Continuous (random base, continuation via starting_model_repo)
    Task 2: From scratch (random base)
    Task 3: Winner continuation or TARGET_TOURN_MODEL
    """
    round_id = round_data.round_id
    group_id = f"{round_id}_group_001"
    num_envs = min(round_data.round_number * t_cst.ENV_ENVS_PER_ROUND_MULTIPLIER, len(EnvironmentName))

    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config)
    if len(existing_tasks) >= t_cst.ENV_FINAL_ROUND_TASK_COUNT:
        return await _get_existing_tasks(existing_tasks, config)

    models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    tasks: list[RawTask] = await _get_existing_tasks(existing_tasks, config) if existing_tasks else []

    tournament_base_model = await _get_tournament_base_model(tournament_id, config)
    prev_tourn_winner_model = await _get_prev_tourn_winner_model(tournament_id, config)

    logger.info(f"Boss round setup: tournament_base_model={tournament_base_model}, prev_winner_model={prev_tourn_winner_model}")

    boss_task_configs = [
        (tournament_base_model, TrainingStartPoint.CONTINUATION, None),
        (None, TrainingStartPoint.FROM_SCRATCH, t_cst.ENV_TRAINING_HOURS_BOSS_ROUND_FROM_SCRATCH),
        (prev_tourn_winner_model, TrainingStartPoint.PREVIOUS_WINNER, None),
    ]

    for i in range(len(tasks), t_cst.ENV_FINAL_ROUND_TASK_COUNT):
        model_override, start_point, hours = boss_task_configs[i]
        logger.info(f"Boss round task {i+1}/{t_cst.ENV_FINAL_ROUND_TASK_COUNT}: start_point={start_point.value}, model={model_override}, hours={hours}")
        task = await create_synthetic_env_task(
            config, models, instruct_datasets,
            num_environments=num_envs, round_number=round_data.round_number,
            model_id_override=model_override,
            training_start_point=start_point,
            exclude_models=[tournament_base_model] if tournament_base_model else None,
            hours_override=hours,
        )
        await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
        tasks.append(task)

    logger.info(f"Created {len(tasks)} boss round tasks: {[str(t.task_id) for t in tasks]}")
    return tasks


async def _create_environment_group_tasks(
    round_data: GroupRound, tournament_id: str, config: Config,
) -> list[RawTask]:
    """Create one environment task per group. Each task has the same parameters
    (num_envs, round_number, training_start_point) but an independent group_id."""
    round_id = round_data.round_id
    num_envs = round_data.round_number * t_cst.ENV_ENVS_PER_ROUND_MULTIPLIER
    num_envs = min(num_envs, len(EnvironmentName))
    start_point = TrainingStartPoint.CONTINUATION if round_data.round_number > 1 else TrainingStartPoint.DEFAULT

    logger.info(
        f"Creating environment tournament R{round_data.round_number} with {len(round_data.groups)} groups - "
        f"1 task per group, {num_envs} envs per task"
    )

    # R2+ must use the same base model as R1
    tournament_base_model = await _get_tournament_base_model(tournament_id, config) if round_data.round_number > 1 else None

    models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    tasks: list[RawTask] = []
    reference_task: RawTask | None = None

    for i, _group in enumerate(round_data.groups):
        group_id = f"{round_id}_group_{i + 1:03d}"

        existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
        if existing_tasks:
            existing = await _get_existing_tasks(existing_tasks, config)
            tasks.extend(existing)
            if not reference_task and existing:
                reference_task = existing[0]
            continue

        if reference_task:
            task = await create_synthetic_env_task(
                config, models, instruct_datasets,
                num_environments=num_envs, round_number=round_data.round_number,
                training_start_point=start_point,
                model_id_override=reference_task.model_id,
                environment_names_override=reference_task.environment_names,
                eval_seed_override=reference_task.eval_seed,
            )
        else:
            task = await create_synthetic_env_task(
                config, models, instruct_datasets,
                num_environments=num_envs, round_number=round_data.round_number,
                model_id_override=tournament_base_model,
                training_start_point=start_point,
            )
            reference_task = task

        await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
        tasks.append(task)

    logger.info(f"Created {len(tasks)} environment tasks for {len(round_data.groups)} groups: {[str(t.task_id) for t in tasks]}")
    return tasks


async def _create_group_image_tasks(
    round_data: GroupRound, tournament_id: str, config: Config, image_models: list
) -> list[RawTask]:
    num_groups = len(round_data.groups)
    logger.info(f"Creating image tournament for {num_groups} groups ({t_cst.IMAGE_TASKS_PER_GROUP} per group)")
    tasks = []

    for i, group in enumerate(round_data.groups):
        group_tasks = await _create_single_group_image_tasks(group, i, tournament_id, round_data.round_id, config, image_models)
        tasks.extend(group_tasks)

    return tasks


async def _create_single_group_image_tasks(
    group, group_index: int, tournament_id: str, round_id: str, config: Config, image_models: list
) -> list[RawTask]:
    group_id = f"{round_id}_group_{group_index + 1:03d}"
    logger.info(f"  Group {group_index + 1} ({len(group.member_ids)} members):")

    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
    existing_count = len(existing_tasks)

    assert t_cst.IMAGE_TASKS_PER_GROUP == 1, "Only 1 image task per group is supported"
    if existing_count >= t_cst.IMAGE_TASKS_PER_GROUP:
        logger.info(f"    Group {group_index + 1} already has {existing_count} task(s), skipping task creation")
        return await _get_existing_tasks(existing_tasks, config)

    logger.info(f"    Group {group_index + 1} has {existing_count}/{t_cst.IMAGE_TASKS_PER_GROUP} task, creating 1 more")

    task = await _create_single_image_task_with_retry(config, image_models, 0, group_index)
    await _create_and_register_tournament_task(
        task, tournament_id, round_id, config, group_id=group_id
    )

    return [task]


async def _create_knockout_image_tasks(
    round_data: KnockoutRound, tournament_id: str, config: Config, image_models: list
) -> list[RawTask]:
    num_pairs = len(round_data.pairs)
    logger.info(f"Creating image tournament for {num_pairs} knockout pairs ({t_cst.KNOCKOUT_PAIR_TASKS} per pair)")
    tasks = []

    for i, pair in enumerate(round_data.pairs):
        pair_tasks = await _create_single_knockout_image_task(pair, i, tournament_id, round_data.round_id, config, image_models)
        tasks.extend(pair_tasks)

    return tasks


async def _create_single_knockout_image_task(
    pair, pair_index: int, tournament_id: str, round_id: str, config: Config, image_models: list
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
    task = await _create_single_image_task_with_retry(config, image_models, 0, pair_index)
    await _create_and_register_tournament_task(
        task, tournament_id, round_id, config, pair_id=pair_id
    )
    return [task]


async def _create_single_image_task_with_retry(
    config: Config, image_models: list, task_num: int, group_index: int = None, is_final: bool = False
) -> RawTask:
    while True:
        try:
            task = await create_synthetic_image_task(config, image_models)
            break
        except Exception as e:
            context = f"final image task {task_num + 1}" if is_final else f"image task {task_num + 1} for group {group_index + 1}"
            logger.warning(f"Failed to create {context}: {e}. Retrying...")
    return task


async def _create_task_by_type(
    task_type: TaskType, config: Config, models: list, instruct_datasets: list, dpo_datasets: list,
    model_id_override: str | None = None, status_override: TaskStatus | None = None,
) -> RawTask:
    """Create a synthetic task of the specified type."""
    if task_type == TaskType.IMAGETASK:
        return await create_synthetic_image_task(config, models)
    elif task_type == TaskType.INSTRUCTTEXTTASK:
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets, model_id_override, status_override)
    elif task_type == TaskType.DPOTASK:
        return await create_synthetic_dpo_task(config, models, dpo_datasets, model_id_override, status_override)
    elif task_type == TaskType.GRPOTASK:
        return await create_synthetic_grpo_task(config, models, instruct_datasets, model_id_override, status_override)
    elif task_type == TaskType.ENVIRONMENTTASK:
        return await create_synthetic_env_task(config, models, instruct_datasets)
    else:
        # Default to instruct text task
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets, model_id_override, status_override)


def _round_subtask_types(round_number: int) -> list[TaskType]:
    """Subtask types for a round; R2's alignment method is chosen probabilistically."""
    if round_number <= 1:
        return [TaskType.INSTRUCTTEXTTASK]
    if round_number == 2:
        return [TaskType.INSTRUCTTEXTTASK, random.choice([TaskType.DPOTASK, TaskType.GRPOTASK])]
    return [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK]


async def get_tournament_track_models(tournament_id: str, config: Config) -> tuple[str, str]:
    """Derive (model_a, model_b) from R1's two composites, ordered by size (small = A, large = B)."""
    rounds = await get_tournament_rounds(tournament_id, config.psql_db)
    r1 = min(rounds, key=lambda r: r.round_number)
    composites = []
    for tournament_task in await get_tournament_tasks(r1.round_id, config.psql_db):
        task_obj = await task_sql.get_task(tournament_task.task_id, config.psql_db)
        if task_obj and task_obj.task_type == TaskType.COMPOSITETASK:
            composites.append(task_obj)
    composites.sort(key=lambda task: task.model_params_count)
    return composites[0].model_id, composites[1].model_id


async def _prior_track_subtasks(
    tournament_id: str, round_number: int, track_model: str, config: Config
) -> list[CompositeSubtask]:
    """The previous round's composite-for-this-track subtasks (with source rounds), carried forward."""
    if round_number <= 1:
        return []
    rounds = await get_tournament_rounds(tournament_id, config.psql_db)
    prev = next((r for r in rounds if r.round_number == round_number - 1), None)
    if not prev:
        return []
    for tournament_task in await get_tournament_tasks(prev.round_id, config.psql_db):
        task_obj = await task_sql.get_task(tournament_task.task_id, config.psql_db)
        if task_obj and task_obj.task_type == TaskType.COMPOSITETASK and task_obj.model_id == track_model:
            return await get_composite_task_subtasks(str(tournament_task.task_id), config.psql_db)
    return []


async def _create_text_composite(
    track_model: str, subtask_types: list[TaskType], hours: float, round_number: int,
    tournament_id: str, round_id: str, group_id: str, config: Config,
    start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT,
) -> RawTask:
    """Build one track's CompositeTask: its new (inert) subtasks + carried-over prior ones."""
    models = _get_text_models(config.keypair)  # ignored (model_id_override set) but required positionally
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    new_subtasks = []
    for subtask_type in subtask_types:
        subtask = await _create_task_by_type(
            subtask_type, config, models, instruct_datasets, dpo_datasets,
            model_id_override=track_model, status_override=TaskStatus.COMPOSITE_SUBTASK,
        )
        new_subtasks.append(subtask)

    prior_subtasks = await _prior_track_subtasks(tournament_id, round_number, track_model, config)

    now = datetime.utcnow()
    composite = await task_sql.add_task(
        CompositeRawTask(
            model_id=track_model,
            ds="composite",
            status=TaskStatus.PENDING,
            is_organic=False,
            created_at=now,
            termination_at=now + timedelta(hours=hours),
            hours_to_complete=hours,
            account_id=NULL_ACCOUNT_ID,
            training_start_point=start_point,
        ),
        config.psql_db,
    )
    await add_composite_task_subtasks(
        str(composite.task_id),
        [
            CompositeSubtask(subtask_task_id=str(subtask.task_id), source_round=round_number)
            for subtask in new_subtasks
        ]
        + prior_subtasks,
        config.psql_db,
    )
    await _create_and_register_tournament_task(composite, tournament_id, round_id, config, group_id=group_id)
    return composite


async def create_text_round_tasks(round_data: Round, tournament_id: str, config: Config) -> list[str]:
    """Create a round's per-track CompositeTask (A small, B large). Idempotent across restarts."""
    round_id = round_data.round_id
    round_number = round_data.round_number
    group_id = f"{round_id}_group_001"

    existing = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
    if existing:
        return [str(task.task_id) for task in existing]

    if round_number <= 1:
        model_a = await anext(_get_text_models(config.keypair, largest_size_b=t_cst.TEXT_TRACK_A_MAX_SIZE_B))
        model_b = await anext(
            _get_text_models(
                config.keypair, smallest_size_b=t_cst.TEXT_TRACK_A_MAX_SIZE_B, largest_size_b=t_cst.TEXT_TRACK_B_MAX_SIZE_B
            )
        )
    else:
        model_a, model_b = await get_tournament_track_models(tournament_id, config)

    subtask_types = _round_subtask_types(round_number)
    hours = t_cst.TEXT_ROUND_HOURS[round_number]

    composite_ids = []
    for track_model in (model_a, model_b):
        composite = await _create_text_composite(
            track_model, subtask_types, hours, round_number, tournament_id, round_id, group_id, config
        )
        composite_ids.append(str(composite.task_id))
    return composite_ids


async def create_text_boss_round_tasks(round_data: Round, tournament_id: str, config: Config) -> list[str]:
    """B-only boss round: the finalist faces the boss across 3 structurally-distinct composite scenarios.
    Idempotent across restarts.

      - continuation:    continue from the boss's model, evaluated over the whole accumulated history
      - from-scratch:    a fresh 3-stage {instruct, dpo, grpo} composite on any random model
      - previous-winner: a fixed instruct-only benchmark on a placeholder model
    """
    round_id = round_data.round_id
    round_number = round_data.round_number
    group_id = f"{round_id}_group_001"

    existing = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
    if existing:
        return [str(task.task_id) for task in existing]

    _, model_b = await get_tournament_track_models(tournament_id, config)
    random_model = await anext(_get_text_models(config.keypair))
    hours = t_cst.TEXT_ROUND_HOURS[max(t_cst.TEXT_ROUND_HOURS)]

    # (model_id, new subtask types, start point) — continuation creates no new subtasks; _create_text_composite
    # carries the prior round's same-model surface, which for continuation is the full accumulated lineage.
    scenarios = [
        (model_b, [], TrainingStartPoint.CONTINUATION),
        (random_model, [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK], TrainingStartPoint.FROM_SCRATCH),
        (t_cst.TEXT_PREV_WINNER_PLACEHOLDER_MODEL, [TaskType.INSTRUCTTEXTTASK], TrainingStartPoint.PREVIOUS_WINNER),
    ]
    composite_ids = []
    for model_id, subtask_types, start_point in scenarios:
        composite = await _create_text_composite(
            model_id, subtask_types, hours, round_number, tournament_id, round_id, group_id, config,
            start_point=start_point,
        )
        composite_ids.append(str(composite.task_id))
    return composite_ids


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
    gpu_req = get_tournament_gpu_requirement(task.task_type, task.model_params_count, task.model_id)

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


async def create_new_task_of_same_type(task: RawTask, config: Config) -> RawTask:
    if task.task_type == TaskType.IMAGETASK:
        models = _get_image_models(config.keypair)
        return await _create_task_by_type(task.task_type, config, models, [], [])

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

    return await _create_task_by_type(task.task_type, config, models, instruct_datasets, dpo_datasets)


def _is_round_one_group_text_task(task: RawTask, round_id: str, group_id: str | None, pair_id: str | None) -> bool:
    """Return True when task should follow round-1 group text constraints."""
    return (
        task.task_type == TaskType.INSTRUCTTEXTTASK
        and group_id is not None
        and pair_id is None
        and round_id.endswith("_round_001")
    )


async def _create_round_one_group_text_replacement_task(config: Config) -> RawTask:
    """
    Create a replacement task that matches round-1 group text constraints:
    - small text model pool (0.1B-4.0B)
    """
    models = _get_text_models(config.keypair, smallest_size_b=0.1, largest_size_b=4.0)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    return await create_synthetic_instruct_text_task(config, models, instruct_datasets)


async def _create_new_image_boss_round_tasks(tournament_id: str, round_id: str, config: Config) -> list[RawTask]:
    """Create boss round image tasks using new synthetic tasks."""
    pair_id = f"{round_id}_pair_001"

    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, pair_id=pair_id)
    existing_count = len(existing_tasks)

    if existing_count >= t_cst.FINAL_ROUND_IMAGE_TASKS:
        logger.info(f"Final round already has {existing_count} tasks, skipping task creation")
        return await _get_existing_tasks(existing_tasks, config)

    logger.info("Creating boss round image tasks using new synthetic tasks")

    existing_task_objects = await _get_existing_tasks(existing_tasks, config)
    existing_qwen_zimage = sum(
        1 for task in existing_task_objects
        if hasattr(task, 'model_type') and task.model_type in [ImageModelType.QWEN_IMAGE, ImageModelType.Z_IMAGE]
    )

    tasks = existing_task_objects
    num_needed = t_cst.FINAL_ROUND_IMAGE_TASKS - existing_count
    num_qwen_zimage = min(t_cst.FINAL_ROUND_IMAGE_QWEN_ZIMAGE_TASKS - existing_qwen_zimage, num_needed)
    num_regular = num_needed - num_qwen_zimage

    async def filtered_models(include_qwen_zimage: bool):
        async for model in _get_image_models(config.keypair):
            is_qwen_zimage = model.model_type in [ImageModelType.QWEN_IMAGE, ImageModelType.Z_IMAGE]
            if include_qwen_zimage == is_qwen_zimage:
                yield model

    qwen_zimage_gen = filtered_models(include_qwen_zimage=True)
    for i in range(num_qwen_zimage):
        try:
            task = await _create_single_image_task_with_retry(config, qwen_zimage_gen, i, is_final=True)
            await _create_and_register_tournament_task(task, tournament_id, round_id, config, pair_id=pair_id)
            tasks.append(task)
        except Exception as e:
            logger.error(f"Failed to create qwen/z-image task {i + 1}/{num_qwen_zimage}: {e}", exc_info=True)

    regular_gen = filtered_models(include_qwen_zimage=False)
    for i in range(num_regular):
        try:
            task = await _create_single_image_task_with_retry(config, regular_gen, i, is_final=True)
            await _create_and_register_tournament_task(task, tournament_id, round_id, config, pair_id=pair_id)
            tasks.append(task)
        except Exception as e:
            logger.error(f"Failed to create regular task {i + 1}/{num_regular}: {e}", exc_info=True)

    return tasks


async def replace_tournament_task(
    original_task_id: str, tournament_id: str, round_id: str, group_id: str | None, pair_id: str | None, config: Config
) -> str:
    logger.info(f"Starting task replacement for task {original_task_id}")
    logger.info(f"Tournament: {tournament_id}, Round: {round_id}, Group: {group_id}, Pair: {pair_id}")

    original_task_obj = await task_sql.get_task(original_task_id, config.psql_db)
    if not original_task_obj:
        logger.error(f"Could not find original task {original_task_id}")
        raise ValueError(f"Original task {original_task_id} not found")

    logger.info(f"Found original task - Type: {original_task_obj.task_type}, Status: {original_task_obj.status}")
    logger.info(f"Original task model params: {original_task_obj.model_params_count}")

    try:
        if _is_round_one_group_text_task(original_task_obj, round_id, group_id, pair_id):
            logger.info("Detected round-1 group text task replacement; enforcing small-model and 2h constraints")
            new_task = await _create_round_one_group_text_replacement_task(config)
        else:
            new_task = await create_new_task_of_same_type(original_task_obj, config)
        logger.info(f"Successfully created new task {new_task.task_id} of type {new_task.task_type}")
    except Exception as e:
        logger.error(f"Failed to create new task of type {original_task_obj.task_type}: {str(e)}", exc_info=True)
        raise

    try:
        await _create_and_register_tournament_task(
            new_task, tournament_id, round_id, config, group_id=group_id, pair_id=pair_id
        )
        logger.info(f"Created replacement task {new_task.task_id} for round {round_id}")
    except Exception as e:
        logger.error(f"Failed to add tournament task to database: {str(e)}", exc_info=True)
        raise

    original_assigned_nodes = await task_sql.get_nodes_assigned_to_task(original_task_id, config.psql_db)
    for node in original_assigned_nodes:
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

    await task_sql.delete_task(original_task_id, config.psql_db)
    logger.info(f"Deleted original task {original_task_id} from db.")

    return new_task.task_id
