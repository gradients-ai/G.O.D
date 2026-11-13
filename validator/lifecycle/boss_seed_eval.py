"""Seed-time evaluation of the boss's boss-round entries (validator/tournament/README.md,
"Design notes").

A boss-round seed (the Free and the Fixed compression task) is trained by the boss when the first round is created. Its
boss row is evaluated as soon as it trains, through the ordinary per-hotkey path (an evaluation row is seeded the moment
the training row is SUCCESS), and the raw result is stored in task_nodes. Only the task's finalisation is held until the
challenger has been scored (seeds_sql.boss_round_seed_ready_to_finalise), so the boss round evaluates the challenger
alone and ranks the task from both stored rows.

This module holds the seed-time rules that keep that stored boss result complete and never wrongly scored:

  * an evaluation error of the boss's entry before the boss round attaches the seed (a scorer-side failure: an
    exception, a missing result, an unreadable repo) is never a scored fail: the row goes back to pending and is
    retried, up to SEED_TIME_BOSS_EVAL_RETRIES in all, each no sooner than SEED_TIME_BOSS_EVAL_BACKOFF_SECONDS after the
    last failure; after that it is parked (alerted once) and evaluated at the boss
    round instead;
  * a quant entry whose stored result has no bf16 reference for its own eval SKU is re-evaluated at seed time (same
    budget), so finalisation in the boss round never resets the boss's row to measure one;
  * at the boss round, a boss entry that trained but still has no settled evaluation is alerted once and evaluated then
    (the fallback); a stored boss entry and the challenger's scored on different eval sets is alerted once.

The retry count and the alert sets are process-local, like the reference pre-pass counters: a restart grants a fresh
budget at worst and never scores anything twice.
"""

import time

import validator.db.constants as db_cst
import validator.db.sql.tasks as tasks_sql
import validator.db.sql.tournament_seeds as seeds_sql
import validator.lifecycle.constants as lifecycle_cst
from core.logging import get_logger
from core.models.task_models import QUANT_TASK_TYPES
from validator.app.config import Config
from validator.db.sql.submissions_and_scoring import get_per_example_losses
from validator.db.sql.submissions_and_scoring import get_task_node_losses
from validator.evaluation.notifications import notify_evaluation_exception
from validator.lifecycle.eval_infra_errors import reference_requeue_sql
from validator.lifecycle.quant_reference import rows_missing_a_reference
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tasks.models import AnyTypeRawTask
from validator.tournament.models import ANY_QUANT_SEED_ROLES


logger = get_logger(__name__)

_TERMINAL_EVALUATION_STATUSES = frozenset({"success", "failure"})

_SEED_TIME_RETRIES: dict[str, int] = {}
_LAST_RETRY_AT: dict[str, float] = {}
_PARKED_ALERTED: set[str] = set()
_ROUND_OPEN_ALERTED: set[str] = set()
_EVAL_SET_MISMATCH_ALERTED: set[str] = set()
_REFERENCE_HELD_ALERTED: set[str] = set()


def seed_time_retries_spent(task_id) -> bool:
    return _SEED_TIME_RETRIES.get(str(task_id), 0) >= lifecycle_cst.SEED_TIME_BOSS_EVAL_RETRIES


async def evaluation_parked(task: AnyTypeRawTask, config: Config) -> bool:
    """An unattached boss-round seed whose seed-time evaluation used up its retries: left for the boss round."""
    if not seed_time_retries_spent(task.task_id):
        return False
    return await seeds_sql.is_unattached_boss_round_seed(task.task_id, config.psql_db)


async def evaluation_backing_off(task: AnyTypeRawTask, config: Config) -> bool:
    """An unattached boss-round seed whose seed-time evaluation failed less than SEED_TIME_BOSS_EVAL_BACKOFF_SECONDS ago."""
    failed_at = _LAST_RETRY_AT.get(str(task.task_id))
    if failed_at is None or time.monotonic() - failed_at >= lifecycle_cst.SEED_TIME_BOSS_EVAL_BACKOFF_SECONDS:
        return False
    return await seeds_sql.is_unattached_boss_round_seed(task.task_id, config.psql_db)


async def _retry_at_seed_time(task: AnyTypeRawTask, hotkeys: list[str], reason: str, config: Config) -> None:
    """Put the rows back to pending (never scored) and count the retry; alert once when the budget is spent."""
    task_id = str(task.task_id)
    await tasks_sql.update_task_evaluations_status(task.task_id, hotkeys, "pending", config.psql_db)
    await tasks_sql.clear_task_evaluation_deployments(task.task_id, hotkeys, config.psql_db)
    _SEED_TIME_RETRIES[task_id] = _SEED_TIME_RETRIES.get(task_id, 0) + 1
    _LAST_RETRY_AT[task_id] = time.monotonic()
    attempts = _SEED_TIME_RETRIES[task_id]
    if not seed_time_retries_spent(task_id):
        logger.warning(
            f"Seed-time evaluation of the boss's entry on {task_id} ({hotkeys}): {reason}; not scored, retrying "
            f"({attempts}/{lifecycle_cst.SEED_TIME_BOSS_EVAL_RETRIES})"
        )
        return
    message = (
        f"Seed-time evaluation of the boss's entry on {task_id} ({hotkeys}) did not settle after {attempts} attempts "
        f"(last: {reason}); parked, it will be evaluated when the boss round attaches the task"
    )
    logger.error(message)
    if task_id in _PARKED_ALERTED:
        return
    _PARKED_ALERTED.add(task_id)
    await notify_evaluation_exception(
        config,
        task_id=task_id,
        task_type=task.task_type,
        context="Seed-time boss evaluation parked",
        error=message,
        hotkeys=hotkeys,
    )


async def _scorer_side_failures(task: AnyTypeRawTask, hotkeys: list[str], config: Config) -> list[str]:
    """Failed rows whose failure the scorer recorded (an exception, an unreadable repo), not the evaluator's judgement.

    Before ranking the only score_reason on a raw row is the scorer's own error text; a model the evaluator scored badly
    (NaN loss, not a fine-tune) has none, and a quant scored fail carries quant_failure and is not a failure at all.
    """
    rows = {row[db_cst.HOTKEY]: row for row in await get_task_node_losses(task.task_id, config.psql_db)}
    return [
        hotkey
        for hotkey in hotkeys
        if hotkey in rows and rows[hotkey].get(db_cst.SCORE_REASON) and not rows[hotkey].get(db_cst.QUANT_FAILURE)
    ]


async def settle_seed_time_evaluation(
    task: AnyTypeRawTask, failed: list[str], not_evaluated: list[str], config: Config
) -> list[str]:
    """After an evaluation pass wrote its statuses, on an unattached boss-round seed (a no-op on any other task):

    the scorer-side failures and the rows that came back with no result go back to pending, and a quant row scored
    successfully without a stored bf16 reference on its own eval SKU is re-evaluated (check_seed_time_references).
    Returns the hotkeys put back to pending. Never raises: a lookup error leaves the pass's own statuses as written.
    """
    try:
        if not await seeds_sql.is_unattached_boss_round_seed(task.task_id, config.psql_db):
            return []
        retry = sorted(set(not_evaluated) | set(await _scorer_side_failures(task, failed, config)))
        if retry:
            await _retry_at_seed_time(task, retry, "evaluation error", config)
            return retry
        return await check_seed_time_references(task, config)
    except Exception as e:
        logger.error(f"Task {task.task_id}: settling the seed-time evaluation failed: {e}", exc_info=True)
        return []


async def retry_seed_time_exception(task: AnyTypeRawTask, hotkeys: list[str], error: Exception, config: Config) -> bool:
    """The evaluation pass raised. On an unattached boss-round seed the rows go back to pending and True is returned.

    Never raises: on a lookup error the caller's ordinary exception handling applies."""
    try:
        if not await seeds_sql.is_unattached_boss_round_seed(task.task_id, config.psql_db):
            return False
        await _retry_at_seed_time(task, hotkeys, f"{type(error).__name__}: {str(error)[:300]}", config)
        return True
    except Exception as e:
        logger.error(f"Task {task.task_id}: seed-time retry after an evaluation exception failed: {e}", exc_info=True)
        return False


async def check_seed_time_references(task: AnyTypeRawTask, config: Config) -> list[str]:
    """Quant, unattached boss-round seed: a successfully evaluated row with no bf16 reference on its own eval SKU is
    re-evaluated now (that deployment measures it), so the boss round never re-evaluates the boss's entry for it.

    Own-model compressions pair each side on its own reference (thresholds.sku_aligned_losses), so the boss's entry
    needs exactly the one on the SKU it was scored on. Stored references are never deleted. The caller
    (settle_seed_time_evaluation) has checked the task is an unattached boss-round seed. Returns the rows re-evaluated.
    """
    if task.task_type not in QUANT_TASK_TYPES or seed_time_retries_spent(task.task_id):
        return []
    rows = await tasks_sql.get_task_evaluation_rows(task.task_id, config.psql_db)
    successful = [row["hotkey"] for row in rows if row["evaluation_status"] == "success"]
    if not successful:
        return []
    missing = await rows_missing_a_reference(task, successful, config)
    if missing:
        await _retry_at_seed_time(task, missing, "no bf16 reference stored for its eval SKU", config)
    return missing


async def alert_if_boss_evaluation_missing(task: AnyTypeRawTask, config: Config) -> bool:
    """An attached boss-round seed whose boss row trained SUCCESS but has no settled evaluation: True, alerted once.

    Nothing waits on this: the row is evaluated in the boss round like any pending row.
    Alert only: never raises.
    """
    try:
        return await _alert_if_boss_evaluation_missing(task, config)
    except Exception as e:
        logger.error(f"Task {task.task_id}: check of the boss's seed-time evaluation failed: {e}", exc_info=True)
        return False


async def _alert_if_boss_evaluation_missing(task: AnyTypeRawTask, config: Config) -> bool:
    rows = await tasks_sql.get_task_evaluation_rows(task.task_id, config.psql_db)
    boss_row = next((row for row in rows if row["hotkey"] == EMISSION_BURN_HOTKEY), None)
    if boss_row is not None and boss_row["evaluation_status"] in _TERMINAL_EVALUATION_STATUSES:
        return False
    task_id = str(task.task_id)
    status = boss_row["evaluation_status"] if boss_row is not None else "no evaluation row"
    message = (
        f"Boss round opened with the boss's entry on {task_id} trained but not evaluated ({status}); "
        f"it is evaluated now, alongside the challenger"
    )
    logger.warning(message)
    if task_id not in _ROUND_OPEN_ALERTED:
        _ROUND_OPEN_ALERTED.add(task_id)
        await notify_evaluation_exception(
            config,
            task_id=task_id,
            task_type=task.task_type,
            context="Seed-time boss evaluation missing at the boss round",
            error=message,
            hotkeys=[EMISSION_BURN_HOTKEY],
        )
    return True


async def hold_for_missing_reference(task: AnyTypeRawTask, error: Exception, config: Config) -> bool:
    """A boss-round seed with no usable quant reference: True, held and alerted once; False on any other task.

    A missing reference is an infrastructure failure, never a verdict on either player, so it must not fail the task
    (a failed attached seed counts against the boss and hands the challenger the slot). The task keeps its status and
    finalisation is retried every cycle until the reference exists; the alert carries the statement that sets the rows
    back to 'pending' so their deployment measures it. A failed alert never raises; a database error does, and the
    caller must then keep holding rather than treat the task as an ordinary one.
    """
    seed = await seeds_sql.get_seed_task_by_task_id(task.task_id, config.psql_db)
    if seed is None or seed.role not in ANY_QUANT_SEED_ROLES:
        return False
    task_id = str(task.task_id)
    logger.error(f"Boss-round task {task_id} has no usable quant reference; held, not failed: {error}")
    if task_id not in _REFERENCE_HELD_ALERTED:
        _REFERENCE_HELD_ALERTED.add(task_id)
        missing = await _rows_missing_a_reference_or_none(task, config)
        await notify_evaluation_exception(
            config,
            task_id=task_id,
            task_type=task.task_type,
            context=(
                "Boss-round quant reference missing: task HELD, not failed. Once the reference can be measured, "
                f"set the rows back to pending: {reference_requeue_sql(task_id, missing)}"
            ),
            error=str(error),
        )
    return True


async def _rows_missing_a_reference_or_none(task: AnyTypeRawTask, config: Config) -> list[str]:
    """The task's scored rows that lack a same-SKU reference; empty when that cannot be read (the alert then names none)."""
    try:
        rows = await tasks_sql.get_task_evaluation_rows(task.task_id, config.psql_db)
        successful = [row[db_cst.HOTKEY] for row in rows if row["evaluation_status"] == "success"]
        return await rows_missing_a_reference(task, successful, config)
    except Exception as e:
        logger.warning(f"Task {task.task_id}: could not list the rows missing a reference: {e}")
        return []


async def alert_on_eval_set_mismatch(task: AnyTypeRawTask, hotkeys: list[str], config: Config) -> bool:
    """At finalisation of a boss-round seed: True (alerted once) when the stored rows were scored on different eval sets.

    Both rows are scored from the one task's test_data, which prep writes once, so a mismatch is not a stale boss result
    but a model-dependent eval set (the evaluator filters rows by the candidate's context length); it would arise the same
    with both scored together, and the round's own pairing check decides it (unpairable vectors stay with the boss).
    Alert only: never raises and never blocks finalisation. False on any other task.
    """
    try:
        seed = await seeds_sql.get_seed_task_by_task_id(task.task_id, config.psql_db)
        if seed is None or seed.role not in ANY_QUANT_SEED_ROLES:
            return False
        return await _alert_on_eval_set_mismatch(task, hotkeys, config)
    except Exception as e:
        logger.error(f"Task {task.task_id}: eval-set check of the stored boss-round entries failed: {e}", exc_info=True)
        return False


async def _alert_on_eval_set_mismatch(task: AnyTypeRawTask, hotkeys: list[str], config: Config) -> bool:
    fingerprints = {}
    for hotkey in hotkeys:
        _, fingerprint = await get_per_example_losses(task.task_id, hotkey, config.psql_db)
        if fingerprint is not None:
            fingerprints[hotkey] = fingerprint
    if len(set(fingerprints.values())) <= 1:
        return False
    task_id = str(task.task_id)
    message = f"Boss-round task {task_id}: the stored entries were scored on different eval sets {fingerprints}"
    logger.error(message)
    if task_id not in _EVAL_SET_MISMATCH_ALERTED:
        _EVAL_SET_MISMATCH_ALERTED.add(task_id)
        await notify_evaluation_exception(
            config,
            task_id=task_id,
            task_type=task.task_type,
            context="Boss-round eval set mismatch",
            error=message,
            hotkeys=hotkeys,
        )
    return True
