import asyncio
import datetime

import basilica
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
from validator.evaluation.scoring import evaluate_and_score_hotkeys
from validator.tournament.utils import send_to_discord
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


async def _seed_task_evaluations_for_preevaluation(config: Config) -> None:
    tasks_to_seed = await tasks_sql.get_tasks_with_status(
        TaskStatus.PREEVALUATION, psql_db=config.psql_db, tournament_filter="all", benchmark_filter="include"
    )
    for task in tasks_to_seed:
        try:
            assert task.task_id is not None
            await tasks_sql.add_task_evaluation_pairs(task.task_id, config.psql_db)
            task.status = TaskStatus.EVALUATING
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)
            logger.info(f"Task {task.task_id} moved to EVALUATING and evaluation rows seeded")
        except Exception as e:
            logger.error(f"Failed to seed evaluations for task {task.task_id}: {e}", exc_info=True)


async def _finalize_task_status_from_evaluations(task: AnyTypeRawTask, config: Config) -> bool:
    assert task.task_id is not None
    rows = await tasks_sql.get_task_evaluation_rows(task.task_id, config.psql_db)
    if not rows:
        logger.warning(f"No evaluation rows found for task {task.task_id}")
        return False

    statuses = [row["evaluation_status"] for row in rows]
    if any(status in ("pending", "evaluating") for status in statuses):
        return False

    if any(status == "failure" for status in statuses):
        task.status = TaskStatus.FAILURE
        add_context_tag("status", task.status.value)
        failed_hotkeys = [row["hotkey"] for row in rows if row["evaluation_status"] == "failure"]
        if config.discord_url:
            try:
                await send_to_discord(
                    config.discord_url,
                    f"Evaluation failed for task {task.task_id}. Failed hotkeys: {failed_hotkeys}",
                )
            except Exception as e:
                logger.warning(f"Failed to send evaluation webhook: {e}")
    else:
        task.status = TaskStatus.SUCCESS
        add_context_tag("status", task.status.value)

    task.n_eval_attempts = (task.n_eval_attempts or 0) + 1
    await tasks_sql.update_task(task, config.psql_db)
    logger.info(f"Task {task.task_id} finalized as {task.status.value}")
    return True


async def _evaluate_pending_pairs_for_task(task: AnyTypeRawTask, num_gpus: int, config: Config):
    assert task.task_id is not None

    pending_rows = await tasks_sql.get_task_evaluations_by_status(task.task_id, "pending", config.psql_db)
    if not pending_rows:
        await _finalize_task_status_from_evaluations(task, config)
        return

    pending_hotkeys = [row["hotkey"] for row in pending_rows]
    await tasks_sql.update_task_evaluations_status(task.task_id, pending_hotkeys, "evaluating", config.psql_db)

    try:
        evaluated_hotkeys, failed_hotkeys = await evaluate_and_score_hotkeys(task, pending_hotkeys, num_gpus, config)
        not_evaluated_hotkeys = [h for h in pending_hotkeys if h not in set(evaluated_hotkeys)]
        failed_set = set(failed_hotkeys)
        failed_set.update(not_evaluated_hotkeys)
        success_hotkeys = [hotkey for hotkey in evaluated_hotkeys if hotkey not in failed_set]

        await tasks_sql.update_task_evaluations_status(task.task_id, success_hotkeys, "success", config.psql_db)
        await tasks_sql.update_task_evaluations_status(
            task.task_id,
            list(failed_set),
            "failure",
            config.psql_db,
        )
    except Exception as e:
        logger.error(f"Error evaluating pending pairs for task {task.task_id}: {e}", exc_info=True)
        await tasks_sql.update_task_evaluations_status(task.task_id, pending_hotkeys, "failure", config.psql_db)

    await _finalize_task_status_from_evaluations(task, config)


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
    await tasks_sql.reset_task_evaluations_to_pending(task.task_id, config.psql_db)
    task.status = TaskStatus.PREEVALUATION
    add_context_tag("status", task.status.value)
    logger.info(f"Changing status to {task.status}")
    await tasks_sql.update_task(task, config.psql_db)


async def _cleanup_all_running_basilica_deployments() -> None:
    """Cleanup of Basilica deployments on startup."""
    try:
        client = basilica.BasilicaClient()
        deployments = await asyncio.to_thread(client.list)
    except Exception as e:
        logger.warning(f"Failed to list Basilica deployments for cleanup: {e}")
        return

    deleted_count = 0
    for deployment in deployments:
        try:
            await asyncio.to_thread(deployment.delete)
            deleted_count += 1
        except Exception as e:
            logger.warning(f"Failed to delete Basilica deployment during startup cleanup: {e}")

    if deleted_count:
        logger.info(f"Deleted {deleted_count} Basilica deployments during startup cleanup")


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
    processing_task_ids: set[str] = set()

    while True:
        await _seed_task_evaluations_for_preevaluation(config)

        evaluating_tasks = await tasks_sql.get_tasks_with_status(
            TaskStatus.EVALUATING, psql_db=config.psql_db, tournament_filter="all", benchmark_filter="include"
        )
        if evaluating_tasks:
            logger.info(f"Found {len(evaluating_tasks)} tasks in EVALUATING")
            for task in evaluating_tasks:
                if task.task_id not in processing_task_ids:
                    processing_task_ids.add(task.task_id)
                    asyncio.create_task(_run_and_cleanup(task, processing_task_ids, config))
        else:
            logger.info("No tasks in EVALUATING - waiting 30 seconds")
        await asyncio.sleep(30)


async def _run_and_cleanup(task: RawTask, processing_task_ids: set[str], config: Config):
    try:
        num_gpus = compute_required_gpus(task)
        await _evaluate_pending_pairs_for_task(task, num_gpus, config)
    except Exception as e:
        logger.error(f"Error evaluating task {task.task_id}: {e}", exc_info=True)
    finally:
        processing_task_ids.discard(task.task_id)


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
    await _cleanup_all_running_basilica_deployments()
    await _move_any_evaluating_tasks_to_pending_evaluation(config)

    await asyncio.gather(evaluate_tasks_loop(config), cleanup_model_cache_loop(config.psql_db))
