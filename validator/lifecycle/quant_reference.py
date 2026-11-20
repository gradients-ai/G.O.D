"""Finalize's check that every scored quant row has the bf16 reference for its own SKU.

The reference (the miner's own model on an own-model task, else the common input:
tournament.quant_reference.quant_reference_model) is not measured here. Each evaluation
deployment scores it on its own GPU when that SKU is not cached yet, and scoring.tasks caches it in
quant_sku_references. A scored row whose SKU still has no reference (the reference errored or did not score
in its pod) is an evaluation error: the row is re-evaluated, which measures
the reference, never ranked without it.
"""

import math

import validator.db.constants as db_cst
import validator.db.sql.tasks as tasks_sql
from core.logging import get_logger
from validator.app.config import Config
from validator.db.sql.quant_references import get_quant_sku_references
from validator.db.sql.submissions_and_scoring import get_task_node_losses
from validator.db.sql.submissions_and_scoring import set_task_node_quant_failure
from validator.evaluation.notifications import notify_evaluation_exception
from validator.scoring.quant_ranking import QuantReferenceUnavailable
from validator.scoring.quant_ranking import eval_gpu_from_column
from validator.scoring.quant_ranking import reference_on
from validator.tasks.models import AnyQuantRawTask
from validator.tournament.constants import QUANT_REFERENCE_EVAL_RETRIES
from validator.tournament.quant_reference import NO_REFERENCE_MODEL_FAILURE
from validator.tournament.quant_reference import is_own_model_start_point
from validator.tournament.quant_reference import quant_reference_model
from validator.tournament.quant_reference import reference_hotkey_for


logger = get_logger(__name__)

# Task ids already alerted for a missing reference. Process-local like orchestrator._QUANT_DECODE_ALERTED:
# finalize retries every cycle and would otherwise spam the webhook.
_QUANT_REFERENCE_ALERTED: set[str] = set()


async def _alert_once(task: AnyQuantRawTask, message: str, config: Config) -> None:
    task_id = str(task.task_id)
    if task_id in _QUANT_REFERENCE_ALERTED:
        return
    _QUANT_REFERENCE_ALERTED.add(task_id)
    await notify_evaluation_exception(
        config, task_id=task_id, task_type=task.task_type, context="Quant reference missing", error=message
    )


async def rows_missing_a_reference(task: AnyQuantRawTask, successful_hotkeys: list[str], config: Config) -> list[str]:
    """Successfully evaluated competing hotkeys with a finite loss whose SKU has no cached reference."""
    references = await get_quant_sku_references(task.task_id, config.psql_db)
    competing = set(successful_hotkeys)
    missing = []
    for row in await get_task_node_losses(task.task_id, config.psql_db):
        test_loss = row[db_cst.TEST_LOSS]
        if row[db_cst.HOTKEY] not in competing or test_loss is None or not math.isfinite(test_loss):
            continue
        reference_hotkey = reference_hotkey_for(task, row[db_cst.HOTKEY])
        if reference_on(reference_hotkey, eval_gpu_from_column(row.get(db_cst.EVAL_GPU)), references) is None:
            missing.append(row[db_cst.HOTKEY])
    return sorted(missing)


async def _fail_miners_without_a_reference_model(
    task: AnyQuantRawTask, successful_hotkeys: list[str], config: Config
) -> list[str]:
    """Own-model task: persist a scored fail for each competing miner with no reference model; returns the rest.

    Only that miner is unscoreable (it has no starting model), so the task itself stands.
    """
    scoreable = []
    for hotkey in sorted(set(successful_hotkeys)):
        if await quant_reference_model(task, hotkey, config.psql_db) is None:
            logger.error(f"Task {task.task_id}: no bf16 reference model for {hotkey}; scored fail")
            await set_task_node_quant_failure(task.task_id, hotkey, NO_REFERENCE_MODEL_FAILURE, config.psql_db)
        else:
            scoreable.append(hotkey)
    return scoreable


async def ensure_quant_references(task: AnyQuantRawTask, successful_hotkeys: list[str], config: Config) -> bool:
    """True once every scored row has its SKU's reference; False after resetting the rows that lack one.

    On an own-model task a competing miner with no reference model (no starting model) gets a persisted scored
    fail (NO_REFERENCE_MODEL_FAILURE) and is left out of the check; the others proceed. Raises
    QuantReferenceUnavailable when any other task has no reference model or the re-evaluations are spent:
    lifecycle swaps the task, as for any unusable reference. The retry budget shares n_eval_attempts with the
    rest of the task's evaluation.
    """
    if is_own_model_start_point(task.training_start_point):
        successful_hotkeys = await _fail_miners_without_a_reference_model(task, successful_hotkeys, config)
    elif await quant_reference_model(task, "", config.psql_db) is None:
        raise QuantReferenceUnavailable(f"Task {task.task_id} has no bf16 reference model to normalise by")
    missing = await rows_missing_a_reference(task, successful_hotkeys, config)
    if not missing:
        return True

    message = f"Task {task.task_id}: no same-SKU bf16 reference for {missing}"
    if (task.n_eval_attempts or 0) >= QUANT_REFERENCE_EVAL_RETRIES:
        raise QuantReferenceUnavailable(f"{message} after {task.n_eval_attempts} re-evaluation(s)")
    logger.warning(f"{message}; re-evaluating them so their deployments measure it")
    await _alert_once(task, message, config)
    await tasks_sql.update_task_evaluations_status(task.task_id, missing, "pending", config.psql_db)
    await tasks_sql.clear_task_evaluation_deployments(task.task_id, missing, config.psql_db)
    task.n_eval_attempts = (task.n_eval_attempts or 0) + 1
    await tasks_sql.update_task(task, config.psql_db)
    return False
