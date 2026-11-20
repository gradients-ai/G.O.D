"""Evaluation rows whose evaluation failed for a reason that is not the miner's: re-queue, count, alert, hold.

An EvaluationInfraError (or any exception that escaped evaluation: miner faults come back as results, never as
exceptions) never marks a row 'failure'. Each one adds to the row's infra_error_attempts and puts it back to
'pending'; the first one and the one that reaches MAX_EVAL_INFRA_ATTEMPTS alert. At the cap the row is held in
EVALUATION_STATUS_INFRA_HELD: never dispatched, never scored, and the task does not finalise while it is held, so an
infrastructure outage can neither zero a miner nor fail a task. A human re-queues the row (status 'pending',
infra_error_attempts 0) once the cause is fixed, or fails it deliberately.

Three rules keep that from misfiring:

  * backoff: a re-queued row is not dispatched again until EVAL_INFRA_REQUEUE_BACKOFF_SECONDS after its error
    (requeue_backing_off), so the attempts outlast a short hub or deployment-service outage instead of being spent on
    consecutive 30 s cycles. The row stays 'pending' meanwhile, so its task does not finalise;
  * guard: only rows still 'pending' or 'evaluating' are counted or re-queued. The error can surface after the same
    pass already wrote 'success' or 'failure' for other rows, and those are never reopened or reported as re-queued;
  * reminder: a task with held rows is re-alerted every EVAL_INFRA_HELD_REMINDER_SECONDS (remind_held_rows), the first
    one an interval after the hold, so a held row cannot block its task, round and tournament with a single alert.

The backoff and the reminder timers are process-local (monotonic clock): a restart retries at once and re-arms the
reminder, and never scores anything.
"""

import time

import validator.db.sql.tasks as tasks_sql
import validator.lifecycle.constants as lifecycle_cst
from core.logging import get_logger
from validator.app.config import Config
from validator.evaluation.errors import describe_error
from validator.evaluation.errors import is_infra_error
from validator.evaluation.notifications import notify_evaluation_exception
from validator.evaluation.notifications import task_deployment_ids_for_hotkeys
from validator.tasks.models import AnyTypeRawTask


logger = get_logger(__name__)


MAX_SQL_HOTKEYS = 5

# (task id, hotkey) -> when the row was last re-queued after an infra error; task id -> when its held rows were last
# alerted (or first seen held).
_REQUEUED_AT: dict[tuple[str, str], float] = {}
_HELD_ALERTED_AT: dict[str, float] = {}


def requeue_backing_off(task_id, hotkey: str) -> bool:
    """A row re-queued after an infra error less than EVAL_INFRA_REQUEUE_BACKOFF_SECONDS ago: not dispatched yet."""
    key = (str(task_id), hotkey)
    requeued_at = _REQUEUED_AT.get(key)
    if requeued_at is None:
        return False
    if time.monotonic() - requeued_at >= lifecycle_cst.EVAL_INFRA_REQUEUE_BACKOFF_SECONDS:
        del _REQUEUED_AT[key]
        return False
    return True


def requeue_sql(task_id: str) -> str:
    """The statement a human runs to retry a task's held rows once the cause is fixed.

    Selects by the held status, not by hotkey, so an alert for any number of held rows stays one short line.
    """
    return (
        f"UPDATE evaluations SET evaluation_status='pending', infra_error_attempts=0 "
        f"WHERE task_id='{task_id}' AND evaluation_status='{lifecycle_cst.EVALUATION_STATUS_INFRA_HELD}';"
    )


def fail_sql(task_id: str) -> str:
    """The statement a human runs to fail a task's held rows deliberately (finalisation then scores them 'Evaluation failed')."""
    return (
        f"UPDATE evaluations SET evaluation_status='failure' "
        f"WHERE task_id='{task_id}' AND evaluation_status='{lifecycle_cst.EVALUATION_STATUS_INFRA_HELD}';"
    )


def reference_requeue_sql(task_id: str, hotkeys: list[str]) -> str:
    """The statement a human runs to have rows measured again once their quant reference can be: back to 'pending'.

    Names the rows when there are few; otherwise (or when unknown) every scored row of the task, so it stays short.
    """
    if hotkeys and len(hotkeys) <= MAX_SQL_HOTKEYS:
        hotkey_list = ", ".join(f"'{hotkey}'" for hotkey in hotkeys)
        scope = f"hotkey IN ({hotkey_list})"
    else:
        scope = "evaluation_status='success'"
    return f"UPDATE evaluations SET evaluation_status='pending' WHERE task_id='{task_id}' AND {scope};"


async def record_infra_error(task: AnyTypeRawTask, hotkeys: list[str], error: BaseException, config: Config) -> None:
    """Re-queue the rows after an evaluation error that is not the miner's fault; hold the ones that reached the cap."""
    assert task.task_id is not None
    kind = "infrastructure error" if is_infra_error(error) else "evaluation error (likely ours)"
    described = describe_error(error)
    deployment_ids = await task_deployment_ids_for_hotkeys(task.task_id, config, hotkeys)
    attempts = await tasks_sql.record_task_evaluation_infra_errors(
        task.task_id,
        hotkeys,
        lifecycle_cst.MAX_EVAL_INFRA_ATTEMPTS,
        lifecycle_cst.EVALUATION_STATUS_INFRA_HELD,
        config.psql_db,
    )
    held = sorted(hotkey for hotkey, count in attempts.items() if count >= lifecycle_cst.MAX_EVAL_INFRA_ATTEMPTS)
    requeued = sorted(hotkey for hotkey in attempts if hotkey not in held)
    logger.error(
        f"Task {task.task_id}: {kind}, not the miner's fault, attempts={attempts} "
        f"(cap {lifecycle_cst.MAX_EVAL_INFRA_ATTEMPTS}); re-queued={requeued} held={held}: {described}"
    )

    now = time.monotonic()
    for hotkey in requeued:
        _REQUEUED_AT[(str(task.task_id), hotkey)] = now
    if held:
        _HELD_ALERTED_AT[str(task.task_id)] = now

    first_time = sorted(hotkey for hotkey in requeued if attempts[hotkey] == 1)
    if first_time:
        await notify_evaluation_exception(
            config,
            task_id=str(task.task_id),
            task_type=task.task_type,
            context=(
                f"Evaluation {kind}, NOT the miner's fault: re-queued "
                f"(attempt 1/{lifecycle_cst.MAX_EVAL_INFRA_ATTEMPTS})"
            ),
            error=error,
            hotkeys=first_time,
            deployment_ids=deployment_ids,
        )
    if held:
        await notify_evaluation_exception(
            config,
            task_id=str(task.task_id),
            task_type=task.task_type,
            context=(
                f"Evaluation {kind} {lifecycle_cst.MAX_EVAL_INFRA_ATTEMPTS} times, NOT the miner's fault: rows HELD "
                f"('{lifecycle_cst.EVALUATION_STATUS_INFRA_HELD}', not scored; the task will not finalise). "
                f"Fix the cause, then re-queue: {requeue_sql(str(task.task_id))} "
                f"Or fail the rows deliberately: {fail_sql(str(task.task_id))}"
            ),
            error=error,
            hotkeys=held,
            deployment_ids=deployment_ids,
        )


async def remind_held_rows(task: AnyTypeRawTask, held_hotkeys: list[str], config: Config) -> None:
    """Re-send the hold alert for a task that still has held rows, at most once per EVAL_INFRA_HELD_REMINDER_SECONDS.

    The hold alerts when it happens; the first reminder follows one interval after that (or after this process first
    sees the held rows), then one per interval.
    """
    task_id = str(task.task_id)
    now = time.monotonic()
    last_alerted_at = _HELD_ALERTED_AT.setdefault(task_id, now)
    if now - last_alerted_at < lifecycle_cst.EVAL_INFRA_HELD_REMINDER_SECONDS:
        return
    _HELD_ALERTED_AT[task_id] = now
    await notify_evaluation_exception(
        config,
        task_id=task_id,
        task_type=task.task_type,
        context=(
            f"REMINDER: evaluation rows still HELD ('{lifecycle_cst.EVALUATION_STATUS_INFRA_HELD}') after repeated errors that "
            f"are not the miner's fault; the task, its round and the tournament cannot finalise until they are handled. "
            f"Re-queue: {requeue_sql(task_id)} Or fail the rows deliberately: {fail_sql(task_id)}"
        ),
        error="held rows awaiting a human",
        hotkeys=sorted(held_hotkeys),
        deployment_ids=await task_deployment_ids_for_hotkeys(task.task_id, config, held_hotkeys),
    )
