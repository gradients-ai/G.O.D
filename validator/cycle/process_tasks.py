import asyncio
import datetime

import core.constants as core_cst
import validator.core.constants as cst
import validator.db.sql.nodes as nodes_sql
import validator.db.sql.tasks as tasks_sql
import validator.db.sql.tournaments as tournament_sql
from core.models.pvp_models import PvPIncompleteError
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
from validator.evaluation.scoring import finalize_task_scores_from_raw_losses
from validator.evaluation.scoring import should_use_tournament_eval
from validator.evaluation.basilica import EvaluationRetryableError
from validator.evaluation.utils import cleanup_all_basilica_deployments
from validator.utils.cache_clear import clean_all_hf_datasets_cache
from validator.utils.cache_clear import manage_models_cache
from validator.utils.logging import LogContext
from validator.utils.logging import add_context_tag
from validator.utils.logging import get_logger


logger = get_logger(__name__)
_TOURNAMENT_GROUP_EVAL_TYPES = frozenset({core_cst.EvalType.PVP, core_cst.EvalType.INDIVIDUAL})


def _tournament_environment_names() -> list[str]:
    return [
        name.value
        for name, env_config in core_cst.ENVIRONMENT_CONFIGS.items()
        if env_config.eval_type in _TOURNAMENT_GROUP_EVAL_TYPES
    ]


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

            type_enabled = cst.MODEL_PREP_ENABLED_BY_TASK_TYPE.get(task.task_type, False)
            needs_model_prep = type_enabled and not (
                task.is_organic and not cst.BASELINE_STATS_ENABLED_ORGANIC
            )

            if needs_model_prep:
                task.status = TaskStatus.AWAITING_MODEL_PREP
                logger.info(f"Task {task.task_id} prepped, moving to {task.status.value}")
            else:
                task.status = TaskStatus.LOOKING_FOR_NODES

            add_context_tag("status", task.status.value)
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


async def _seed_task_evaluations_for_evaluation(config: Config) -> None:
    preevaluation_tasks = await tasks_sql.get_tasks_with_status(
        TaskStatus.PREEVALUATION, psql_db=config.psql_db, tournament_filter="all", benchmark_filter="include"
    )
    organic_training_tasks = await tasks_sql.get_tasks_with_status(
        TaskStatus.TRAINING,
        psql_db=config.psql_db,
        tournament_filter="exclude",
        benchmark_filter="exclude",
    )

    for task in preevaluation_tasks:
        try:
            assert task.task_id is not None
            pvp = should_use_tournament_eval(task)
            await tasks_sql.add_task_evaluation_pairs(task.task_id, config.psql_db, include_failed_training=pvp)
            task.status = TaskStatus.EVALUATING
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)
            logger.info(f"Task {task.task_id} moved to EVALUATING and evaluation rows seeded")
        except Exception as e:
            logger.error(f"Failed to seed evaluations for task {task.task_id}: {e}", exc_info=True)

    for task in organic_training_tasks:
        try:
            assert task.task_id is not None
            training_statuses = await tournament_sql.get_training_status_for_task(str(task.task_id), config.psql_db)
            if training_statuses and any(status not in ("success", "failure") for status in training_statuses.values()):
                continue

            await tasks_sql.add_task_evaluation_pairs(task.task_id, config.psql_db)
            task.status = TaskStatus.EVALUATING
            add_context_tag("status", task.status.value)
            await tasks_sql.update_task(task, config.psql_db)
            logger.info(f"Organic task {task.task_id} moved from TRAINING to EVALUATING and evaluation rows seeded")
        except Exception as e:
            logger.error(f"Failed to seed evaluations for organic task {task.task_id}: {e}", exc_info=True)


async def _finalize_task_status_from_evaluations(task: AnyTypeRawTask, config: Config) -> bool:
    assert task.task_id is not None
    training_statuses = await tournament_sql.get_training_status_for_task(str(task.task_id), config.psql_db)
    is_tournament_task = bool(training_statuses)

    if is_tournament_task and any(status not in ("success", "failure") for status in training_statuses.values()):
        logger.info(f"Task {task.task_id} still has non-terminal training rows; deferring final scoring")
        return False

    rows = await tasks_sql.get_task_evaluation_rows(task.task_id, config.psql_db)
    if is_tournament_task:
        successful_training_hotkeys = {
            hotkey for hotkey, status in training_statuses.items() if status == "success"
        }
        expected_rows = [row for row in rows if row["hotkey"] in successful_training_hotkeys]
        evaluation_hotkeys = {row["hotkey"] for row in expected_rows}

        if evaluation_hotkeys != successful_training_hotkeys:
            missing_hotkeys = sorted(successful_training_hotkeys - evaluation_hotkeys)
            logger.info(
                "Task %s evaluation coverage mismatch; deferring final scoring. missing=%s",
                task.task_id,
                missing_hotkeys,
            )
            return False
        extra_hotkeys = sorted({row["hotkey"] for row in rows} - successful_training_hotkeys)
        if extra_hotkeys:
            logger.info(
                "Task %s ignoring stale evaluation rows for non-successful training hotkeys: %s",
                task.task_id,
                extra_hotkeys,
            )
        rows = expected_rows

    if not rows:
        if is_tournament_task:
            task.status = TaskStatus.FAILURE
            add_context_tag("status", task.status.value)
            task.n_eval_attempts = (task.n_eval_attempts or 0) + 1
            await tasks_sql.update_task(task, config.psql_db)
            logger.info(f"Task {task.task_id} finalized as failure because no tournament evaluations were produced")
            return True

        logger.warning(f"No evaluation rows found for task {task.task_id}")
        return False

    statuses = [row["evaluation_status"] for row in rows]
    if any(status in ("pending", "evaluating") for status in statuses):
        return False

    successful_hotkeys = [row["hotkey"] for row in rows if row["evaluation_status"] == "success"]
    if not successful_hotkeys:
        task.status = TaskStatus.FAILURE
        add_context_tag("status", task.status.value)
        task.n_eval_attempts = (task.n_eval_attempts or 0) + 1
        await tasks_sql.update_task(task, config.psql_db)
        logger.info(f"Task {task.task_id} finalized as failure because no evaluation succeeded")
        return True

    await finalize_task_scores_from_raw_losses(task, [row["hotkey"] for row in rows], config)

    if not successful_hotkeys:
        task.status = TaskStatus.FAILURE
        add_context_tag("status", task.status.value)
    else:
        task.status = TaskStatus.SUCCESS
        add_context_tag("status", task.status.value)
        failed_hotkeys = [row["hotkey"] for row in rows if row["evaluation_status"] == "failure"]
        if failed_hotkeys:
            logger.info(f"Task {task.task_id} succeeded with {len(failed_hotkeys)} failed hotkeys: {failed_hotkeys}")

    task.n_eval_attempts = (task.n_eval_attempts or 0) + 1
    await tasks_sql.update_task(task, config.psql_db)
    logger.info(f"Task {task.task_id} finalized as {task.status.value}")
    return True


async def _evaluate_and_update_hotkeys(task: AnyTypeRawTask, hotkeys: list[str], num_gpus: int, config: Config) -> None:
    assert task.task_id is not None

    try:
        eval_result = await evaluate_and_score_hotkeys(task, hotkeys, num_gpus, config)
        not_evaluated_hotkeys = [h for h in hotkeys if h not in set(eval_result.evaluated)]
        failed_set = set(eval_result.failed)
        failed_set.update(not_evaluated_hotkeys)
        success_hotkeys = [h for h in eval_result.evaluated if h not in failed_set]

        await tasks_sql.update_task_evaluations_status(task.task_id, success_hotkeys, "success", config.psql_db)
        await tasks_sql.update_task_evaluations_status(
            task.task_id,
            list(failed_set),
            "failure",
            config.psql_db,
        )
    except PvPIncompleteError as e:
        logger.info(f"PvP eval incomplete for task {task.task_id}: {e} — resetting to pending for retry")
        await tasks_sql.update_task_evaluations_status(task.task_id, hotkeys, "pending", config.psql_db)
        await tasks_sql.clear_task_evaluation_deployments(task.task_id, hotkeys, config.psql_db)
    except EvaluationRetryableError as e:
        logger.info(f"Evaluation retryable for task {task.task_id}: {e} — resetting to pending for retry")
        await tasks_sql.update_task_evaluations_status(task.task_id, hotkeys, "pending", config.psql_db)
        await tasks_sql.clear_task_evaluation_deployments(task.task_id, hotkeys, config.psql_db)
    except Exception as e:
        logger.error(f"Error evaluating pending pairs for task {task.task_id}: {e}", exc_info=True)
        await tasks_sql.update_task_evaluations_status(task.task_id, hotkeys, "failure", config.psql_db)
        await tasks_sql.clear_task_evaluation_deployments(task.task_id, hotkeys, config.psql_db)


async def _evaluate_pending_pairs_for_task(task: AnyTypeRawTask, num_gpus: int, config: Config):
    assert task.task_id is not None

    batch_together = task.task_type == TaskType.GRPOTASK or should_use_tournament_eval(task)
    if batch_together:
        training_statuses = await tournament_sql.get_training_status_for_task(str(task.task_id), config.psql_db)
        if training_statuses and any(status not in ("success", "failure") for status in training_statuses.values()):
            logger.info(f"Task {task.task_id} still has non-terminal training rows; deferring batch evaluation")
            return

    pending_rows = await tasks_sql.get_task_evaluations_by_status(task.task_id, "pending", config.psql_db)
    evaluating_rows = await tasks_sql.get_task_evaluations_by_status(task.task_id, "evaluating", config.psql_db)
    if not pending_rows and not evaluating_rows:
        finalized = await _finalize_task_status_from_evaluations(task, config)
        if finalized:
            await _cleanup_basilica_deployments_if_no_active_evaluations(config)
        return
    
    pending_hotkeys = [row["hotkey"] for row in pending_rows]
    evaluating_hotkeys = [row["hotkey"] for row in evaluating_rows]
    all_hotkeys = list(dict.fromkeys(pending_hotkeys + evaluating_hotkeys))

    hotkey_batches = [all_hotkeys] if batch_together else [[hotkey] for hotkey in all_hotkeys]
    pending_evaluations = []
    for hotkeys in hotkey_batches:
        pending_batch = [hotkey for hotkey in hotkeys if hotkey in pending_hotkeys]
        if pending_batch:
            await tasks_sql.update_task_evaluations_status(task.task_id, pending_batch, "evaluating", config.psql_db)

        pending_evaluations.append(_evaluate_and_update_hotkeys(task, hotkeys, num_gpus, config))

    if pending_evaluations:
        await asyncio.gather(*pending_evaluations)

    finalized = await _finalize_task_status_from_evaluations(task, config)
    if finalized:
        await _cleanup_basilica_deployments_if_no_active_evaluations(config)


async def _cleanup_basilica_deployments_if_no_active_evaluations(config: Config) -> None:
    active_evaluations = await tasks_sql.count_task_evaluations_by_status("evaluating", config.psql_db)
    if active_evaluations:
        logger.info("Skipping drained Basilica cleanup; %s evaluation rows still active", active_evaluations)
        return

    logger.info("No active evaluation rows remain; deleting all Basilica deployments")
    await cleanup_all_basilica_deployments()


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


async def _recover_evaluating_tasks(config: Config):
    stopped_mid_evaluation = await tasks_sql.get_tasks_with_status(
        TaskStatus.EVALUATING, psql_db=config.psql_db, tournament_filter="exclude", benchmark_filter="include"
    )
    logger.info(f"Recovering {len(stopped_mid_evaluation)} evaluating tasks by resetting in-progress rows to pending")
    for task in stopped_mid_evaluation:
        if task.task_id is None:
            continue
        if task.task_type == TaskType.GRPOTASK:
            logger.info(f"Resetting all GRPO evaluation rows to pending for task {task.task_id}")
            await tasks_sql.reset_all_task_evaluations_to_pending(task.task_id, config.psql_db)
        else:
            await tasks_sql.reset_task_evaluations_to_pending(task.task_id, config.psql_db)


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


async def _get_tasks_ready_for_evaluation(config: Config) -> list[RawTask]:
    tasks_by_id: dict[str, RawTask] = {}

    evaluating_tasks = await tasks_sql.get_tasks_with_status(
        TaskStatus.EVALUATING, psql_db=config.psql_db, tournament_filter="all", benchmark_filter="include"
    )
    for task in evaluating_tasks:
        tasks_by_id[str(task.task_id)] = task

    tournament_task_ids = await tasks_sql.get_task_ids_with_evaluation_statuses(
        ["pending", "evaluating"],
        config.psql_db,
        task_statuses=[TaskStatus.TRAINING.value, TaskStatus.EVALUATING.value],
        tournament_only=True,
    )
    for task_id in tournament_task_ids:
        task = await tasks_sql.get_task(task_id, config.psql_db)
        if task:
            tasks_by_id[str(task.task_id)] = task

    completed_training_task_ids = await tournament_sql.get_tasks_with_all_training_completed(config.psql_db)
    for task_id in completed_training_task_ids:
        task = await tasks_sql.get_task(task_id, config.psql_db)
        if task:
            tasks_by_id[str(task.task_id)] = task

    return list(tasks_by_id.values())


async def evaluate_tasks_loop(config: Config):
    processing_task_ids: set[str] = set()

    while True:
        try:
            await _seed_task_evaluations_for_evaluation(config)

            evaluating_tasks = await _get_tasks_ready_for_evaluation(config)
            if evaluating_tasks:
                logger.info(f"Found {len(evaluating_tasks)} tasks ready for evaluation work")
                for task in evaluating_tasks:
                    if task.task_id not in processing_task_ids:
                        processing_task_ids.add(task.task_id)
                        asyncio.create_task(_run_and_cleanup(task, processing_task_ids, config))
            else:
                logger.info("No tasks ready for evaluation - waiting 30 seconds")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"evaluate_tasks_loop iteration failed: {e!r}", exc_info=True)
        await asyncio.sleep(30)


async def _run_and_cleanup(
    task: RawTask,
    processing_task_ids: set[str],
    config: Config,
):
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
    await _recover_evaluating_tasks(config)

    await asyncio.gather(evaluate_tasks_loop(config), cleanup_model_cache_loop(config.psql_db))
