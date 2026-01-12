import random

from core.models.tournament_models import GroupRound
from core.models.tournament_models import KnockoutRound
from core.models.tournament_models import Round
from core.models.tournament_models import TournamentTask
from core.models.utility_models import ImageModelType
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO
from validator.core.constants import PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
from validator.core.models import RawTask
from validator.db.sql import tasks as task_sql
from validator.db.sql.tournaments import add_tournament_tasks
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
from validator.tournament.utils import get_tournament_gpu_requirement
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def create_text_tournament_tasks(
    round_data: Round,
    tournament_id: str,
    round_id: str,
    config: Config,
    is_final_round: bool = False,
) -> list[str]:
    if isinstance(round_data, GroupRound):
        num_groups = len(round_data.groups)
        logger.info(f"Creating text tournament for {num_groups} groups (1 task per group)")
        tasks = await _create_group_text_tasks(round_data, tournament_id, round_id, config, is_final_round)
    elif is_final_round:
        task_types = [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK]
        tasks_per_type = t_cst.FINAL_ROUND_TEXT_TASKS // len(task_types)
        logger.info(f"Creating final text tournament with new synthetic tasks ({tasks_per_type} of each: instruct, DPO, GRPO)")
        tasks = await _create_new_text_boss_round_tasks(tournament_id, round_id, config)
    else:
        num_pairs = len(round_data.pairs)
        logger.info(f"Creating text tournament for {num_pairs} knockout pairs (probability-based)")
        tasks = await _create_probability_based_text_tasks(round_data, tournament_id, round_id, config)

    return [str(task.task_id) for task in tasks]


async def create_image_tournament_tasks(
    round_data: Round, tournament_id: str, round_id: str, config: Config, is_final_round: bool = False
) -> list[str]:
    image_models = _get_image_models(config.keypair)
    tasks = []

    if isinstance(round_data, GroupRound):
        tasks = await _create_group_image_tasks(round_data, tournament_id, round_id, config, image_models)
    elif is_final_round:
        tasks = await _create_new_image_boss_round_tasks(tournament_id, round_id, config)
    else:
        tasks = await _create_knockout_image_tasks(round_data, tournament_id, round_id, config, image_models)

    return [str(task.task_id) for task in tasks]


async def create_environment_tournament_tasks(
    round_data: Round, tournament_id: str, round_id: str, config: Config, is_final_round: bool = False
) -> list[str]:
    """
    Create environment tournament tasks.
    """
    if not isinstance(round_data, GroupRound):
        raise ValueError("Environment tournaments only support group rounds")
    
    tasks = await _create_environment_group_tasks(round_data, tournament_id, round_id, config)
    return [str(task.task_id) for task in tasks]


async def _create_environment_group_tasks(
    round_data: GroupRound, tournament_id: str, round_id: str, config: Config
) -> list[RawTask]:
    """
    Create a single environment task that all groups (and all participants + boss) compete on.
    """
    logger.info(f"Creating environment tournament with {len(round_data.groups)} groups - single task for all participants")
    
    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config)
    
    if existing_tasks:
        logger.info(f"Environment tournament round {round_id} already has {len(existing_tasks)} task(s), skipping task creation")
        return await _get_existing_tasks(existing_tasks, config)
    
    models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    
    logger.info("Creating single environment task for all participants")
    task = await create_synthetic_env_task(config, models, instruct_datasets)

    group_id = f"{round_id}_group_001"
    await _create_and_register_tournament_task(
        task, tournament_id, round_id, config, group_id=group_id
    )
    
    logger.info(f"Created environment tournament task {task.task_id} for all participants")
    return [task]


async def _create_group_image_tasks(
    round_data: GroupRound, tournament_id: str, round_id: str, config: Config, image_models: list
) -> list[RawTask]:
    num_groups = len(round_data.groups)
    logger.info(f"Creating image tournament for {num_groups} groups ({t_cst.IMAGE_TASKS_PER_GROUP} per group)")
    tasks = []

    for i, group in enumerate(round_data.groups):
        group_tasks = await _create_single_group_image_tasks(group, i, tournament_id, round_id, config, image_models)
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
    round_data: KnockoutRound, tournament_id: str, round_id: str, config: Config, image_models: list
) -> list[RawTask]:
    num_pairs = len(round_data.pairs)
    logger.info(f"Creating image tournament for {num_pairs} knockout pairs ({t_cst.KNOCKOUT_PAIR_TASKS} per pair)")
    tasks = []

    for i, pair in enumerate(round_data.pairs):
        pair_tasks = await _create_single_knockout_image_task(pair, i, tournament_id, round_id, config, image_models)
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
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets)
    elif task_type == TaskType.DPOTASK:
        return await create_synthetic_dpo_task(config, models, dpo_datasets)
    elif task_type == TaskType.GRPOTASK:
        return await create_synthetic_grpo_task(config, models, instruct_datasets)
    elif task_type == TaskType.ENVIRONMENTTASK:
        return await create_synthetic_env_task(config, models, instruct_datasets)
    else:
        # Default to instruct text task
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets)


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
        duration_info = f" - Duration: {task.hours_to_complete} hours" if hasattr(task, 'hours_to_complete') and task.hours_to_complete else ""
        task_type_info = f"{task.task_type.value}: " if hasattr(task.task_type, 'value') else ""
        logger.info(f"{task_type_info}{task.task_id} - Model: {task.model_id}{dataset_info} - GPU: {gpu_req}{duration_info}")


async def _create_group_text_tasks(
    round_data: GroupRound, tournament_id: str, round_id: str, config: Config, is_final_round: bool
) -> list[RawTask]:
    models = _get_text_models(config.keypair, smallest_size_b=0.1, largest_size_b=4.0)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    tasks = []
    for i, group in enumerate(round_data.groups):
        logger.info(f"  Group {i + 1} ({len(group.member_ids)} members): creating 1 instruct task")
        group_tasks = await _create_single_group_text_tasks(
            group, i, tournament_id, round_id, config, models, instruct_datasets, dpo_datasets
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
) -> list[RawTask]:
    group_id = f"{round_id}_group_{group_index + 1:03d}"

    existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, group_id=group_id)
    existing_count = len(existing_tasks)

    if existing_count >= t_cst.TEXT_TASKS_PER_GROUP:
        logger.info(f"    Group {group_index + 1} already has {existing_count} task(s), skipping task creation")
        return await _get_existing_tasks(existing_tasks, config)

    logger.info(f"    Group {group_index + 1} has {existing_count}/{t_cst.TEXT_TASKS_PER_GROUP} task, creating 1 more")
    assert t_cst.TEXT_TASKS_PER_GROUP == 1, "Only 1 text task per group is supported"
    task = await create_synthetic_instruct_text_task(config, models, instruct_datasets)

    task.hours_to_complete = 2
    await task_sql.update_task(task, config.psql_db)

    await _create_and_register_tournament_task(
        task, tournament_id, round_id, config, group_id=group_id
    )

    return [task]


async def _create_probability_based_text_tasks(
    round_data: KnockoutRound, tournament_id: str, round_id: str, config: Config
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
        pair_id = f"{round_id}_pair_{i + 1:03d}"

        existing_tasks = await _get_existing_tasks_by_identifier(round_id, config, pair_id=pair_id)
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
        task = await _create_single_probability_task(config, models, instruct_datasets, dpo_datasets, instruct_prob, dpo_prob)

        await _create_and_register_tournament_task(
            task, tournament_id, round_id, config, pair_id=pair_id
        )
        tasks.append(task)
    return tasks


async def _create_single_probability_task(
    config: Config, models: list, instruct_datasets: list, dpo_datasets: list, instruct_prob: float, dpo_prob: float
) -> RawTask:
    rand_val = random.random()
    if rand_val < instruct_prob:
        return await create_synthetic_instruct_text_task(config, models, instruct_datasets)
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
            f"Original task has very small model params ({task.model_params_count}), using default range {t_cst.DEFAULT_MODEL_MIN_SIZE_B}-{t_cst.DEFAULT_MODEL_MAX_SIZE_B}B"
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


async def _create_new_text_boss_round_tasks(tournament_id: str, round_id: str, config: Config) -> list[RawTask]:
    """Create boss round text tasks using new synthetic tasks."""
    pair_id = f"{round_id}_pair_001"

    existing_pair_tasks = await _get_existing_tasks_by_identifier(round_id, config, pair_id=pair_id)
    existing_count = len(existing_pair_tasks)

    if existing_count >= t_cst.FINAL_ROUND_TEXT_TASKS:
        logger.info(f"Final round already has {existing_count} tasks, skipping task creation")
        return await _get_existing_tasks(existing_pair_tasks, config)

    logger.info("Creating boss round text tasks using new synthetic tasks")

    task_types = [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK]
    tasks_per_type = t_cst.FINAL_ROUND_TEXT_TASKS // len(task_types)

    standard_models = _get_text_models(config.keypair)
    big_models = _get_text_models(config.keypair, smallest_size_b=12.0, largest_size_b=71.0)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    existing_task_type_counts = {}
    tasks = []

    for task in existing_pair_tasks:
        task_obj = await task_sql.get_task(task.task_id, config.psql_db)
        if task_obj:
            task_type_value = task_obj.task_type.value if hasattr(task_obj.task_type, "value") else task_obj.task_type
            existing_task_type_counts[task_type_value] = existing_task_type_counts.get(task_type_value, 0) + 1
            tasks.append(task_obj)

    for task_type in task_types:
        existing_count = existing_task_type_counts.get(task_type.value, 0)
        for i in range(tasks_per_type - existing_count):
            rand_val = random.random()
            if rand_val < t_cst.PROBABILITY_OF_A_BIG_TEXT_MODEL:
                models = big_models
            else:
                models = standard_models
            task = await _create_single_new_text_task(task_type, tournament_id, round_id, pair_id, config, models, instruct_datasets, dpo_datasets)
            if task:
                tasks.append(task)

    return tasks


async def _create_single_new_text_task(
    task_type: TaskType, tournament_id: str, round_id: str, pair_id: str, config: Config, models: list, instruct_datasets: list, dpo_datasets: list
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
                f"Copied node {node.hotkey} with expected_repo_name {original_expected_repo_name} to replacement task {new_task.task_id}"
            )
        else:
            logger.warning(f"No expected repo name found for node {node.hotkey} in original task {original_task_id}")

    await task_sql.delete_task(original_task_id, config.psql_db)
    logger.info(f"Deleted original task {original_task_id} from db.")

    return new_task.task_id
