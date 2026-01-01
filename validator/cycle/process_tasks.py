import asyncio
import datetime

import validator.core.constants as cst
import validator.db.sql.nodes as nodes_sql
import validator.db.sql.tasks as tasks_sql
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.core.models import AnyTypeRawTask
from validator.core.models import RawTask
from validator.core.task_config_models import get_task_config
from validator.cycle.util_functions import get_model_num_params
from validator.db.database import PSQLDB
from validator.evaluation.scoring import evaluate_and_score
from validator.utils.cache_clear import clean_all_hf_datasets_cache
from validator.utils.cache_clear import manage_models_cache
from validator.utils.logging import LogContext
from validator.utils.logging import add_context_tag
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def _select_miner_pool_and_add_to_task(task: AnyTypeRawTask, config: Config) -> AnyTypeRawTask:
    """
    Assign a single miner using EMISSION_BURN_HOTKEY for legacy training tasks.
    """
    logger.info(f"Assigning single miner using EMISSION_BURN_HOTKEY for task {task.task_id}")

    emission_burn_node = await nodes_sql.get_node_by_hotkey(EMISSION_BURN_HOTKEY, config.psql_db)
    miners_already_assigned = await tasks_sql.get_miners_for_task(task.task_id, config.psql_db)
    already_assigned_hotkeys = [miner.hotkey for miner in miners_already_assigned]
    expected_repo_name = f"organic_{task.task_id}"

    if EMISSION_BURN_HOTKEY in already_assigned_hotkeys:
        logger.info(f"EMISSION_BURN_HOTKEY already assigned to task {task.task_id}")
        # Ensure expected_repo_name is set even if already assigned
        await tasks_sql.set_expected_repo_name(str(task.task_id), emission_burn_node, config.psql_db, expected_repo_name)

        task.status = TaskStatus.READY
        add_context_tag("status", task.status.value)
        return task

    await tasks_sql.assign_node_to_task(str(task.task_id), emission_burn_node, config.psql_db)
    logger.info(f"EMISSION_BURN_HOTKEY has been assigned to task {task.task_id}")

    await tasks_sql.set_expected_repo_name(str(task.task_id), emission_burn_node, config.psql_db, expected_repo_name)

    task.status = TaskStatus.READY
    add_context_tag("status", task.status.value)

    logger.info(f"Task {task.task_id} is ready with EMISSION_BURN_HOTKEY assigned")
    return task


async def _find_and_select_miners_for_task(task: AnyTypeRawTask, config: Config):
    with LogContext(task_id=str(task.task_id)):
        try:
            task = await _select_miner_pool_and_add_to_task(task, config)
            logger.info(f"After assigning miners here is the current task info {task}")
            await tasks_sql.update_task(task, config.psql_db)

        except Exception as e:
            logger.error(f"Error assigning miners to task {task.task_id}: {e}", exc_info=True)
            task = _attempt_delay_task(task)
            await tasks_sql.update_task(task, config.psql_db)


def _attempt_delay_task(task: AnyTypeRawTask):
    assert task.created_at is not None and task.next_delay_at is not None and task.times_delayed is not None, (
        "We wanted to check delay vs created timestamps but they are missing"
    )

    if task.times_delayed >= cst.MAX_DELAY_TIMES or not task.is_organic:
        if task.is_organic:
            logger.info(f"We have already delayed {task.times_delayed}")
        else:
            logger.info("This is a synth task - no need to add a delay when the network is busy")

        task.status = TaskStatus.FAILURE_FINDING_NODES
        add_context_tag("status", task.status.value)
    else:
        logger.info(f"Adding in a delay of {cst.TASK_TIME_DELAY} minutes for now since no miners accepted the task")
        task.next_delay_at = task.next_delay_at + datetime.timedelta(minutes=cst.TASK_TIME_DELAY)
        task.status = TaskStatus.DELAYED
        add_context_tag("status", task.status.value)
        task.times_delayed += 1
    return task


async def _find_miners_for_task(config: Config):
    pending_tasks = await tasks_sql.get_tasks_with_status(
        status=TaskStatus.LOOKING_FOR_NODES, psql_db=config.psql_db, tournament_filter="exclude"
    )
    await asyncio.gather(
        *[_find_and_select_miners_for_task(task, config) for task in pending_tasks[: cst.MAX_CONCURRENT_MINER_ASSIGNMENTS]]
    )


async def _prep_task(task: AnyTypeRawTask, config: Config):
    with LogContext(task_id=str(task.task_id)):
        try:
            task.status = TaskStatus.PREPARING_DATA
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)
            task = await get_task_config(task).task_prep_function(task, config.keypair, config.psql_db)
            logger.info(f"THE TASK HAS BEEN PREPPED {task}")
            await tasks_sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error during task prep: {e}", exc_info=True)
            task.status = TaskStatus.PREP_TASK_FAILURE
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)


async def _processing_pending_tasks(config: Config):
    logger.debug("Processing pending tasks")

    pending_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.PENDING, psql_db=config.psql_db)
    logger.info(f"Found {len(pending_tasks)} pending tasks! Will prep them all now...")
    await asyncio.gather(*[_prep_task(task, config) for task in pending_tasks[: cst.MAX_CONCURRENT_TASK_PREPS]])
    clean_all_hf_datasets_cache()


async def _evaluate_task(task: AnyTypeRawTask, gpu_ids: list[int], config: Config):
    gpu_ids_str = "," + ",".join(str(gpu_id) for gpu_id in gpu_ids) + ","
    with LogContext(task_id=str(task.task_id), gpu_ids=gpu_ids_str):
        try:
            task.status = TaskStatus.EVALUATING
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)
            task = await evaluate_and_score(task, gpu_ids, config)
            await tasks_sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)


async def _move_back_to_looking_for_nodes(task: AnyTypeRawTask, config: Config):
    logger.info("Moving back from delay to looking for nodes")
    task.status = TaskStatus.LOOKING_FOR_NODES
    add_context_tag("status", task.status.value)
    await tasks_sql.update_task(task, config.psql_db)


async def _handle_delayed_tasks(config: Config):
    finished_delay_tasks = await tasks_sql.get_tasks_with_status(
        TaskStatus.DELAYED, psql_db=config.psql_db, tournament_filter="exclude"
    )
    logger.info(f"We have {len(finished_delay_tasks)} that we're ready to offer to miners again")
    await asyncio.gather(*[_move_back_to_looking_for_nodes(task, config) for task in finished_delay_tasks])


async def _move_to_preevaluation_status(task, config):
    task.status = TaskStatus.PREEVALUATION
    add_context_tag("status", task.status.value)
    logger.info(f"Changing status to {task.status}")
    await tasks_sql.update_task(task, config.psql_db)


async def _move_any_evaluating_tasks_to_pending_evaluation(config: Config):
    stopped_mid_evaluation = await tasks_sql.get_tasks_with_status(
        TaskStatus.EVALUATING, psql_db=config.psql_db, benchmark_filter="include"
    )
    logger.info(f"WE ARE MOVING {len(stopped_mid_evaluation)} TASKS TO PREEVALUATION")
    await asyncio.gather(*[_move_to_preevaluation_status(task, config) for task in stopped_mid_evaluation])


async def _move_back_to_pending_status(task, config):
    task.status = TaskStatus.PENDING
    add_context_tag("status", task.status.value)
    await tasks_sql.update_task(task, config.psql_db)


async def _move_any_prep_data_to_pending(config):
    stopped_in_prep = await tasks_sql.get_tasks_with_status(TaskStatus.PREPARING_DATA, psql_db=config.psql_db)
    await asyncio.gather(*[_move_back_to_pending_status(task, config) for task in stopped_in_prep])


async def process_pending_tasks(config: Config) -> None:
    await _move_any_prep_data_to_pending(config)
    while True:
        try:
            await _processing_pending_tasks(config)
            await _find_miners_for_task(config)
            await _handle_delayed_tasks(config)
            await asyncio.sleep(30)
        except Exception as e:
            logger.info(f"There was a problem in processing: {e}")
            await asyncio.sleep(30)


async def cleanup_model_cache_loop(psql_db: PSQLDB):
    """Clean up model cache when it exceeds size limit."""
    while True:
        try:
            logger.info("Cleaning up model cache")
            training_tasks = await tasks_sql.get_tasks_with_status(
                TaskStatus.TRAINING, psql_db=psql_db, benchmark_filter="include"
            )
            evaluating_tasks = await tasks_sql.get_tasks_with_status(
                TaskStatus.EVALUATING, psql_db=psql_db, benchmark_filter="include"
            )
            preevaluation_tasks = await tasks_sql.get_tasks_with_status(
                TaskStatus.PREEVALUATION, psql_db=psql_db, benchmark_filter="include"
            )
            protected_models = set()
            for task in evaluating_tasks + preevaluation_tasks + training_tasks:
                if task.model_id:
                    protected_models.add(str(task.model_id))

            cache_stats = await tasks_sql.get_model_cache_stats(
                psql_db, tau_days=cst.CACHE_TAU_DAYS, max_lookup_days=cst.CACHE_MAX_LOOKUP_DAYS
            )

            # Set cache score to infinity for protected models to prevent deletion
            logger.info(f"Protected models: {protected_models}")
            for model_id in protected_models:
                if model_id not in cache_stats:
                    cache_stats[model_id] = {"cache_score": float("inf")}
                else:
                    cache_stats[model_id]["cache_score"] = float("inf")

            manage_models_cache(cache_stats, cst.MAX_CACHE_SIZE_BYTES)
        except Exception as e:
            logger.error(f"Error in cache cleanup: {e}", exc_info=True)
        finally:
            await asyncio.sleep(cst.CACHE_CLEANUP_INTERVAL)


async def evaluate_tasks_loop(config: Config):
    task_queue = asyncio.Queue()
    gpu_queue = asyncio.Queue()
    processing_task_ids = set()
    # Lock to prevent race conditions (thus potential deadlocks) during GPU acquisition
    gpu_acquisition_lock = asyncio.Lock()

    for gpu_id in cst.GPU_IDS:
        await gpu_queue.put(gpu_id)

    async def evaluation_worker():
        while True:
            try:
                task = await asyncio.wait_for(task_queue.get(), timeout=1)
                required_gpus = compute_required_gpus(task)
                gpu_ids = []

                # Acquire lock to prevent other tasks from taking GPUs until we get all we need
                async with gpu_acquisition_lock:
                    for _ in range(required_gpus):
                        gpu_ids.append(await gpu_queue.get())

                try:
                    await _evaluate_task(task, gpu_ids, config)
                finally:
                    for gpu_id in gpu_ids:
                        await gpu_queue.put(gpu_id)
                    processing_task_ids.remove(task.task_id)
                    task_queue.task_done()
            except asyncio.TimeoutError:
                await asyncio.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Error in evaluation worker: {str(e)}", exc_info=True)
                continue

    for _ in cst.GPU_IDS:
        asyncio.create_task(evaluation_worker())

    while True:
        if len(processing_task_ids) < 2 * len(cst.GPU_IDS):
            tasks_to_evaluate = await tasks_sql.get_tasks_with_status(
                TaskStatus.PREEVALUATION, psql_db=config.psql_db, tournament_filter="all", benchmark_filter="include"
            )
            if tasks_to_evaluate:
                logger.info(f"Found {len(tasks_to_evaluate)} new tasks awaiting evaluation, adding to queue")
                for task in tasks_to_evaluate:
                    # Only add to queue if not already added, some tasks in the queue might still have TaskStatus.PREEVALUATION
                    if task.task_id not in processing_task_ids:
                        processing_task_ids.add(task.task_id)
                        await task_queue.put(task)
            else:
                logger.info("No new tasks awaiting evaluation - waiting 30 seconds")
        else:
            logger.info("Evaluation queue is full - waiting for 30 seconds")
        await asyncio.sleep(30)


def compute_required_gpus(task: RawTask) -> int:
    model = task.model_id
    num_params = task.model_params_count
    if not num_params:
        num_params = get_model_num_params(model)
    if not num_params:
        return 1
    if task.task_type == TaskType.DPOTASK:
        num_params = num_params * 2
    elif task.task_type == TaskType.GRPOTASK:
        num_params = num_params * 3
    elif task.task_type == TaskType.ENVIRONMENTTASK:
        num_params = num_params * 3

    if num_params < cst.MODEL_SIZE_REQUIRING_2_GPUS:
        return 1
    elif num_params < cst.MODEL_SIZE_REQUIRING_3_GPUS:
        return 2
    elif num_params < cst.MODEL_SIZE_REQUIRING_4_GPUS:
        return 3
    else:
        return 4


async def process_completed_tasks(config: Config) -> None:
    await _move_any_evaluating_tasks_to_pending_evaluation(config)

    await asyncio.gather(evaluate_tasks_loop(config), cleanup_model_cache_loop(config.psql_db))
