import random

from core.constants.environments import EnvironmentName
from core.constants.environments import TrainingStartPoint
from core.logging import get_logger
from core.models.image_models import ImageModelType
from core.models.task_models import TaskType
from validator.app.config import Config
from validator.db.sql import tasks as task_sql
from validator.db.sql.continuous_sft import warn_orphaned_continuous_sft_state
from validator.db.sql.tournaments import add_tournament_tasks
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.tasks.models import EnvRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import RawTask
from validator.tasks.synthetics.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
from validator.tasks.synthetics.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO
from validator.tasks.synthetics.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
from validator.tasks.synthetics.diffusion import create_synthetic_image_task
from validator.tasks.synthetics.scheduler import _get_dpo_datasets
from validator.tasks.synthetics.scheduler import _get_image_models
from validator.tasks.synthetics.scheduler import _get_instruct_text_datasets
from validator.tasks.synthetics.scheduler import _get_text_models
from validator.tasks.synthetics.scheduler import create_continuous_sft_task
from validator.tasks.synthetics.scheduler import create_synthetic_dpo_task
from validator.tasks.synthetics.scheduler import create_synthetic_env_task
from validator.tasks.synthetics.scheduler import create_synthetic_grpo_task
from validator.tasks.synthetics.scheduler import create_synthetic_instruct_text_task
from validator.tournament import constants as t_cst
from validator.tournament.gpu_requirements import get_tournament_gpu_requirement
from validator.tournament.models import GroupRound
from validator.tournament.models import KnockoutRound
from validator.tournament.models import Round
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType


logger = get_logger(__name__)


def _env_names_excluding_forced_boss() -> list[EnvironmentName]:
    forced = t_cst.FORCED_BOSS_ENVIRONMENT
    return [env for env in EnvironmentName if env != forced]


def _task_includes_environment(task: RawTask, environment: EnvironmentName | None) -> bool:
    if environment is None or not isinstance(task, EnvRawTask):
        return False
    return environment in task.environment_names


def is_small_tournament_group(round_data: GroupRound) -> bool:
    """Whether a group round is the small text/image tournament round-1 format.

    Identified by round 1 (the only round the small format is ever created in — see
    organise_tournament_round) plus a single group whose membership is in the
    small-tournament band (3..9). The round-1 guard is load-bearing: a normal large
    tournament can narrow to a single group of 9 in a *later* round (a reduced group can
    be 9..19 members), which would otherwise match the structural check.
    """
    if round_data.round_number != 1 or len(round_data.groups) != 1:
        return False
    size = len(round_data.groups[0].member_ids)
    return t_cst.SMALL_TOURNAMENT_MIN_PARTICIPANTS <= size <= t_cst.SMALL_TOURNAMENT_MAX_PARTICIPANTS


async def create_text_tournament_tasks(
    round_data: Round,
    tournament_id: str,
    config: Config,
    is_final_round: bool = False,
) -> list[str]:
    round_id = round_data.round_id
    if isinstance(round_data, GroupRound):
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


async def create_image_tournament_tasks(
    round_data: Round, tournament_id: str, config: Config, is_final_round: bool = False,
) -> list[str]:
    round_id = round_data.round_id
    image_models = _get_image_models(config.keypair)
    tasks = []

    if isinstance(round_data, GroupRound):
        if round_data.round_number == 1:
            all_image_models = image_models
            round_one_model_type = random.choice(t_cst.ROUND_ONE_IMAGE_MODEL_TYPES)
            logger.info(f"Round 1 image tournament model type: {round_one_model_type.value}")

            async def round_one_image_models():
                async for model in all_image_models:
                    if model.model_type == round_one_model_type:
                        yield model

            image_models = round_one_image_models()
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


async def _get_prev_tournament_env_names(tournament_id: str, config: Config) -> set[EnvironmentName]:
    prev = await get_latest_completed_tournament(
        config.psql_db, TournamentType.ENVIRONMENT, exclude_tournament_id=tournament_id,
    )
    if not prev:
        return set()

    seen: set[EnvironmentName] = set()
    for prev_round in await get_tournament_rounds(prev.tournament_id, config.psql_db):
        for tourn_task in await get_tournament_tasks(prev_round.round_id, config.psql_db):
            task_obj = await task_sql.get_task(tourn_task.task_id, config.psql_db)
            if isinstance(task_obj, EnvRawTask):
                seen.update(task_obj.environment_names)
    return seen


def _select_r1_env_names(
    num_envs: int,
    seen_last_tournament: set[EnvironmentName],
) -> list[EnvironmentName]:
    all_envs = _env_names_excluding_forced_boss()
    unseen = [env for env in all_envs if env not in seen_last_tournament]
    seen = [env for env in all_envs if env in seen_last_tournament]
    random.shuffle(unseen)
    random.shuffle(seen)
    return (unseen + seen)[:num_envs]


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
    non_boss_envs = _env_names_excluding_forced_boss()
    num_envs = min(round_data.round_number * t_cst.ENV_ENVS_PER_ROUND_MULTIPLIER, len(non_boss_envs))
    forced_boss_env = t_cst.FORCED_BOSS_ENVIRONMENT

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
    has_forced_boss_env_task = any(_task_includes_environment(task, forced_boss_env) for task in tasks)

    for i in range(len(tasks), t_cst.ENV_FINAL_ROUND_TASK_COUNT):
        model_override, start_point, hours = boss_task_configs[i]
        force_boss_env_for_task = (
            forced_boss_env is not None
            and start_point == TrainingStartPoint.PREVIOUS_WINNER
            and not has_forced_boss_env_task
        )
        environment_names_override = [forced_boss_env] if force_boss_env_for_task else None
        exclude_environments = [forced_boss_env] if forced_boss_env is not None and not force_boss_env_for_task else None
        task_num_envs = 1 if force_boss_env_for_task else num_envs
        logger.info(
            f"Boss round task {i+1}/{t_cst.ENV_FINAL_ROUND_TASK_COUNT}: "
            f"start_point={start_point.value}, model={model_override}, hours={hours}, "
            f"forced_env={[env.value for env in environment_names_override or []]}"
        )
        task = await create_synthetic_env_task(
            config, models, instruct_datasets,
            num_environments=task_num_envs, round_number=round_data.round_number,
            exclude_environments=exclude_environments,
            model_id_override=model_override,
            training_start_point=start_point,
            environment_names_override=environment_names_override,
            exclude_models=[tournament_base_model] if tournament_base_model else None,
            hours_override=hours,
        )
        await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
        tasks.append(task)
        if force_boss_env_for_task:
            has_forced_boss_env_task = True

    logger.info(f"Created {len(tasks)} boss round tasks: {[str(t.task_id) for t in tasks]}")
    return tasks


async def _create_environment_group_tasks(
    round_data: GroupRound, tournament_id: str, config: Config,
) -> list[RawTask]:
    """Create one environment task per group. Each task has the same parameters
    (num_envs, round_number, training_start_point) but an independent group_id."""
    round_id = round_data.round_id
    num_envs = round_data.round_number * t_cst.ENV_ENVS_PER_ROUND_MULTIPLIER
    num_envs = min(num_envs, len(_env_names_excluding_forced_boss()))
    start_point = TrainingStartPoint.CONTINUATION if round_data.round_number > 1 else TrainingStartPoint.DEFAULT

    logger.info(
        f"Creating environment tournament R{round_data.round_number} with {len(round_data.groups)} groups - "
        f"1 task per group, {num_envs} envs per task"
    )

    # R2+ must use the same base model as R1
    tournament_base_model = await _get_tournament_base_model(tournament_id, config) if round_data.round_number > 1 else None

    r1_env_override: list[EnvironmentName] | None = None
    if round_data.round_number == 1:
        seen_last_tournament = await _get_prev_tournament_env_names(tournament_id, config)
        r1_env_override = _select_r1_env_names(num_envs, seen_last_tournament)
        logger.info(
            f"R1 env selection: seen_last_tournament={sorted(env.value for env in seen_last_tournament)} "
            f"-> selected={[env.value for env in r1_env_override]}"
        )

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
                exclude_environments=[t_cst.FORCED_BOSS_ENVIRONMENT] if t_cst.FORCED_BOSS_ENVIRONMENT else None,
                environment_names_override=r1_env_override,
            )
            reference_task = task

        await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
        tasks.append(task)

    logger.info(f"Created {len(tasks)} environment tasks for {len(round_data.groups)} groups: {[str(t.task_id) for t in tasks]}")
    return tasks


async def _create_group_image_tasks(
    round_data: GroupRound, tournament_id: str, config: Config, image_models: list
) -> list[RawTask]:
    # Small image tournament round 1: a single group plays SMALL_TOURNAMENT_GROUP_TASKS matches.
    is_small = is_small_tournament_group(round_data)
    tasks_per_group = t_cst.SMALL_TOURNAMENT_GROUP_TASKS if is_small else t_cst.IMAGE_TASKS_PER_GROUP

    num_groups = len(round_data.groups)
    logger.info(f"Creating image tournament for {num_groups} groups ({tasks_per_group} per group)")
    tasks = []

    for i, group in enumerate(round_data.groups):
        group_tasks = await _create_single_group_image_tasks(
            group, i, tournament_id, round_data.round_id, config, image_models, tasks_per_group
        )
        tasks.extend(group_tasks)

    return tasks


async def _create_single_group_image_tasks(
    group, group_index: int, tournament_id: str, round_id: str, config: Config, image_models: list, tasks_per_group: int
) -> list[RawTask]:
    group_id = f"{round_id}_group_{group_index + 1:03d}"
    logger.info(f"  Group {group_index + 1} ({len(group.member_ids)} members):")

    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
    existing_count = len(existing_tasks)

    if existing_count >= tasks_per_group:
        logger.info(f"    Group {group_index + 1} already has {existing_count} task(s), skipping task creation")
        return await _get_existing_tasks(existing_tasks, config)

    created: list[RawTask] = await _get_existing_tasks(existing_tasks, config)
    for _ in range(tasks_per_group - existing_count):
        logger.info(f"    Group {group_index + 1} has {len(created)}/{tasks_per_group} task(s), creating 1 more")
        task = await _create_single_image_task_with_retry(config, image_models, 0, group_index)
        await _create_and_register_tournament_task(task, tournament_id, round_id, config, group_id=group_id)
        created.append(task)

    return created


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
    task_type: TaskType, config: Config, models: list, instruct_datasets: list, dpo_datasets: list
) -> RawTask:
    """Create a synthetic task of the specified type."""
    if task_type == TaskType.IMAGETASK:
        return await create_synthetic_image_task(config, models)
    elif task_type == TaskType.INSTRUCTTEXTTASK:
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets, enable_kl=True)
    elif task_type == TaskType.DPOTASK:
        return await create_synthetic_dpo_task(config, models, dpo_datasets)
    elif task_type == TaskType.GRPOTASK:
        return await create_synthetic_grpo_task(config, models, instruct_datasets)
    elif task_type == TaskType.ENVIRONMENTTASK:
        return await create_synthetic_env_task(config, models, instruct_datasets)
    else:
        # Default to instruct text task
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets, enable_kl=True)


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
    gpu_req = get_tournament_gpu_requirement(
        task.task_type, task.model_params_count, task.model_id,
        use_kl=task.use_kl if isinstance(task, InstructTextRawTask) else False,
        training_start_point=task.training_start_point,
    )

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
    # Small text tournament round 1: a single group plays SMALL_TOURNAMENT_GROUP_TASKS instruct
    # matches (rather than one). It deliberately skips the round-1 restrictions (small models +
    # small datasets) so the few competitors are tested across the full model/dataset range.
    is_small = is_small_tournament_group(round_data)
    tasks_per_group = t_cst.SMALL_TOURNAMENT_GROUP_TASKS if is_small else t_cst.TEXT_TASKS_PER_GROUP

    if is_small:
        models = _get_text_models(config.keypair)
        instruct_datasets = _get_instruct_text_datasets(config.keypair, small_only=False)
    else:
        models = _get_text_models(config.keypair, smallest_size_b=0.1, largest_size_b=3.0)
        instruct_datasets = _get_instruct_text_datasets(config.keypair, small_only=round_data.round_number == 1)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    tasks = []
    for i, group in enumerate(round_data.groups):
        logger.info(f"  Group {i + 1} ({len(group.member_ids)} members): creating {tasks_per_group} instruct task(s)")
        group_tasks = await _create_single_group_text_tasks(
            group, i, tournament_id, round_data.round_id, config, models, instruct_datasets, dpo_datasets, tasks_per_group
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
) -> list[RawTask]:
    group_id = f"{round_id}_group_{group_index + 1:03d}"

    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
    existing_count = len(existing_tasks)

    if existing_count >= tasks_per_group:
        logger.info(f"    Group {group_index + 1} already has {existing_count} task(s), skipping task creation")
        return await _get_existing_tasks(existing_tasks, config)

    created: list[RawTask] = await _get_existing_tasks(existing_tasks, config)
    for _ in range(tasks_per_group - existing_count):
        logger.info(f"    Group {group_index + 1} has {len(created)}/{tasks_per_group} task(s), creating 1 more")
        task = await create_synthetic_instruct_text_task(config, models, instruct_datasets, enable_kl=True)
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

    # The pre-boss round is the knockout with exactly 2 competitors left: its winner becomes the
    # boss challenger, and its task always plays on quasar (see _create_pre_boss_quasar_task).
    # Keyed on competitor count, not task count — other rounds can also create a single task.
    competitors = {hotkey for pair in round_data.pairs for hotkey in pair}
    is_pre_boss_round = len(competitors) == 2

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
        if is_pre_boss_round:
            task = await _create_pre_boss_quasar_task(config, instruct_datasets)
        else:
            task = await _create_single_probability_task(
                config, models, instruct_datasets, dpo_datasets, instruct_prob, dpo_prob
            )

        await _create_and_register_tournament_task(
            task, tournament_id, round_data.round_id, config, pair_id=pair_id
        )
        tasks.append(task)
    return tasks


async def _create_pre_boss_quasar_task(config: Config, instruct_datasets) -> RawTask:
    """Create the pre-boss round's single task: a standard instruct task (normal dataset pull,
    computed hours, param-based GPU sizing) with only the model forced to the quasar seed mirror.
    Augmentation, KL and YaRN are disabled: the custom-arch seed can't be perturbed, reconfigured
    or re-uploaded, and remote-code pinning keys off the exact seed repo.
    """
    return await create_synthetic_instruct_text_task(
        config,
        None,  # no model pool: the model is forced
        instruct_datasets,
        enable_kl=False,
        model_id_override=t_cst.PRE_BOSS_QUASAR_MODEL,
        allow_augmentation=False,
        allow_yarn=False,
    )


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
    - small text model pool (0.1B-3.0B)
    """
    models = _get_text_models(config.keypair, smallest_size_b=0.1, largest_size_b=3.0)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    return await create_synthetic_instruct_text_task(config, models, instruct_datasets, enable_kl=True)


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
    big_models = _get_text_models(config.keypair, smallest_size_b=12.0, largest_size_b=71.0)
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

    # Fixed mix: 2 instruct + 1 dpo + 1 grpo (FINAL_ROUND_TEXT_TASK_DISTRIBUTION) + 2 continuous-SFT.
    for task_type, target_count in t_cst.FINAL_ROUND_TEXT_TASK_DISTRIBUTION.items():
        already = existing_task_type_counts.get(task_type.value, 0)
        for _ in range(target_count - already):
            models = big_models if random.random() < t_cst.PROBABILITY_OF_A_BIG_TEXT_MODEL else standard_models
            task = await _create_single_new_text_task(
                task_type,
                tournament_id,
                round_id,
                pair_id,
                config,
                models,
                instruct_datasets,
                dpo_datasets,
            )
            if task:
                tasks.append(task)

    # Surface any state row whose lineage was renamed/removed (its accumulated chain is now orphaned).
    await warn_orphaned_continuous_sft_state(set(t_cst.CONTINUOUS_SFT_LINEAGES), config.psql_db)

    # One continuous-SFT task per lineage (quasar + qwen); skip lineages already created.
    for lineage, seed_model in t_cst.CONTINUOUS_SFT_LINEAGES.items():
        if lineage in existing_continuous_sft_lineages:
            continue
        task = await _create_continuous_sft_boss_task(tournament_id, round_id, pair_id, config, lineage, seed_model)
        tasks.append(task)

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
) -> RawTask | None:
    """Create a single new synthetic text task of a specific type."""
    try:
        if task_type not in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK]:
            logger.error(f"Unknown task type {task_type} for boss round text task")
            return None

        task = await _create_task_by_type(task_type, config, models, instruct_datasets, dpo_datasets)
        await _create_and_register_tournament_task(
            task, tournament_id, round_id, config, pair_id=pair_id
        )
        return task
    except Exception as e:
        logger.error(f"Failed to create boss round {task_type.value} task: {e}", exc_info=True)
        return None


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
    tasks = existing_task_objects

    existing_counts_by_model_type = {
        model_type: sum(1 for task in existing_task_objects if getattr(task, "model_type", None) == model_type)
        for model_type in t_cst.FINAL_ROUND_IMAGE_TASK_DISTRIBUTION
    }

    async def filtered_models(model_type: ImageModelType):
        async for model in _get_image_models(config.keypair):
            if model.model_type == model_type:
                yield model

    for model_type, target_count in t_cst.FINAL_ROUND_IMAGE_TASK_DISTRIBUTION.items():
        remaining_slots = t_cst.FINAL_ROUND_IMAGE_TASKS - len(tasks)
        if remaining_slots <= 0:
            break
        already_created = existing_counts_by_model_type.get(model_type, 0)
        num_to_create = min(max(target_count - already_created, 0), remaining_slots)
        model_gen = filtered_models(model_type)
        for i in range(num_to_create):
            try:
                task = await _create_single_image_task_with_retry(config, model_gen, i, is_final=True)
                await _create_and_register_tournament_task(task, tournament_id, round_id, config, pair_id=pair_id)
                tasks.append(task)
            except Exception as e:
                logger.error(
                    f"Failed to create {model_type.value} image task {i + 1}/{num_to_create}: {e}",
                    exc_info=True,
                )

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
        elif t_cst.is_pre_boss_quasar_task(original_task_obj):
            logger.info("Detected pre-boss quasar task replacement; re-forcing the quasar seed model")
            new_task = await _create_pre_boss_quasar_task(config, _get_instruct_text_datasets(config.keypair))
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
