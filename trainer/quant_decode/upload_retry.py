"""Background retry of a decoded upload that failed for a transient reason.

When a decode passes its checks but the decoded files cannot be uploaded because of the network, the job ends with an
`upload_pending` decode result and its decoded files stay on disk (job.py). This module then retries in rounds until
the retry window closes. A round uploads only the files that are not in storage yet, one attempt each; after a failed
round it waits (exponential backoff with jitter) and tries again. No upload slot is held while it waits. The result
on the job becomes `ok` as soon as a round completes, or `infra_error` once the window is spent or an error turns up
that no retry can fix.

The state lives in the work dir, so a restarted trainer resumes the retry (`resume_after_restart`). A retry whose work
dir was taken over (a new decode of the same job, the re-run CLI, a restart) stops without storing anything.
"""

import asyncio
from datetime import datetime
from datetime import timezone
from functools import partial
from pathlib import Path

import trainer.constants as cst
from core.models.payload_models import TrainerTaskLog
from core.models.quant_artifact_models import QuantDecodeResult
from trainer import slots
from trainer.job_state import log_task
from trainer.job_state import read_persisted_jobs
from trainer.job_state import update_quant_decode_result
from trainer.quant_decode import job as decode_job
from trainer.quant_decode.job import PENDING_UPLOAD_NAME
from trainer.quant_decode.job import PendingUpload
from trainer.quant_decode.job import ProgressReporter
from trainer.quant_decode.job import UploadFault
from trainer.quant_decode.job import UploadOptions
from trainer.quant_decode.job import _WorkDirs
from trainer.quant_decode.sandbox import sanitize_failure
from trainer.quant_decode.upload import UploadRetryPolicy
from trainer.telemetry import logger


SECONDS_PER_MINUTE = 60
LOST_IN_RESTART_FAILURE = "the retry of the decoded upload was lost when the trainer restarted"

# One background retry per job in this process.
_retries: dict[slots.JobKey, asyncio.Task] = {}


async def _no_report(message: str) -> None:
    return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _pause(seconds: float) -> None:
    await asyncio.sleep(seconds)


def round_policy() -> UploadRetryPolicy:
    """One attempt per file in a round; the delays are the waits between rounds."""
    return UploadRetryPolicy(
        attempts=1,
        base_delay_seconds=cst.QUANT_DECODE_UPLOAD_RETRY_ROUND_BASE_SECONDS,
        max_delay_seconds=cst.QUANT_DECODE_UPLOAD_RETRY_ROUND_MAX_SECONDS,
    )


def _infra_error(pending: PendingUpload, failure: str) -> QuantDecodeResult:
    return QuantDecodeResult(infra_error=sanitize_failure(failure), **pending.progress.model_dump())


async def _settle(work: _WorkDirs, result: QuantDecodeResult) -> QuantDecodeResult:
    await asyncio.to_thread(decode_job.remove_work_tree, work.root)
    return result


async def retry_pending_upload(
    work_root: Path, task_id: str, hotkey: str, report: ProgressReporter = _no_report
) -> QuantDecodeResult | None:
    """Retry the job's pending upload until it lands or its window closes; return the final decode result.

    None when there is nothing to retry or another decode or retry took the work dir over: the caller stores nothing.
    The work dir is removed once the result is final.
    """
    work = decode_job.work_dirs(work_root, task_id, hotkey)
    pending = await asyncio.to_thread(decode_job.load_pending, work)
    if pending is None:
        return None
    policy = round_policy()
    options = UploadOptions(policy=policy, confirm_stored=True)
    failed_rounds = 0
    while True:
        remaining = max(0.0, (pending.retry_until - _now()).total_seconds())
        delay = min(policy.delay_seconds(failed_rounds + 1), remaining)
        await report(
            f"decode: upload not complete ({pending.last_error}); next attempt in {delay:.0f}s, "
            f"giving up in {remaining / SECONDS_PER_MINUTE:.0f} min"
        )
        await _pause(delay)
        if not await asyncio.to_thread(decode_job.owns_pending, work, pending.token):
            return None
        try:
            await decode_job.upload_decoded(
                pending.request,
                work,
                pending.progress,
                task_id=task_id,
                hotkey=hotkey,
                report=report,
                options=options,
                file_done=partial(decode_job.save_pending, work, pending),
            )
        except Exception as e:  # an UploadFault, or a local error such as the decoded files being gone
            if not await asyncio.to_thread(decode_job.owns_pending, work, pending.token):
                return None
            if not isinstance(e, UploadFault):
                logger.exception(f"Upload retry crashed for task {task_id} hotkey {hotkey}")
                return await _settle(work, _infra_error(pending, f"upload retry failed ({type(e).__name__})"))
            if not e.transient:
                return await _settle(work, _infra_error(pending, str(e)))
            failed_rounds += 1
            pending.last_error = sanitize_failure(str(e))
            await asyncio.to_thread(decode_job.save_pending, work, pending)
            if _now() >= pending.retry_until:
                window_minutes = cst.QUANT_DECODE_UPLOAD_RETRY_WINDOW_SECONDS / SECONDS_PER_MINUTE
                failure = f"{pending.last_error}; gave up after retrying for {window_minutes:.0f} min"
                return await _settle(work, _infra_error(pending, failure))
            continue
        return await _settle(work, QuantDecodeResult(ok=True, **pending.progress.model_dump()))


def result_message(result: QuantDecodeResult) -> str:
    """The job-log line for a decode result."""
    if result.ok:
        return f"freeform decode ok: {result.n_tensors} tensors, {len(result.uploaded_files)} files uploaded"
    if result.quant_failure is not None:
        return f"freeform decode scored fail: {result.quant_failure}"
    if result.upload_pending:
        window_minutes = cst.QUANT_DECODE_UPLOAD_RETRY_WINDOW_SECONDS / SECONDS_PER_MINUTE
        return (
            f"freeform decode done, upload not complete: {len(result.uploaded_files)} files uploaded, "
            f"retrying the rest in the background for up to {window_minutes:.0f} min"
        )
    return f"[ERROR] freeform decode infra error: {result.infra_error}"


async def _retry_and_store(work_root: Path, task_id: str, hotkey: str) -> None:
    result = await retry_pending_upload(work_root, task_id, hotkey, partial(log_task, task_id, hotkey))
    if result is None:
        logger.info(f"Upload retry for task {task_id} hotkey {hotkey} was taken over; storing nothing")
        return
    await update_quant_decode_result(task_id, hotkey, result)
    message = result_message(result)
    logger.info(f"{message} | task={task_id} hotkey={hotkey}")
    await log_task(task_id, hotkey, message)


def start(work_root: Path, task_id: str, hotkey: str) -> None:
    """Retry the job's pending upload in the background and store the final result on the job."""
    key = (task_id, hotkey)
    cancel(key)
    task = asyncio.create_task(_retry_and_store(work_root, task_id, hotkey))
    _retries[key] = task
    task.add_done_callback(partial(_retry_done, key))


def _retry_done(key: slots.JobKey, task: asyncio.Task) -> None:
    if _retries.get(key) is task:
        del _retries[key]
    if not task.cancelled() and task.exception() is not None:
        logger.error(f"Upload retry for {key} failed: {task.exception()!r}")


def cancel(key: slots.JobKey) -> None:
    """Stop the job's background retry, if any. Called before a new decode of the same job takes the work dir."""
    task = _retries.pop(key, None)
    if task is not None:
        task.cancel()


async def resume_after_restart(work_root: Path) -> None:
    """Pick up every pending upload left in the work root, and settle the ones that cannot be picked up.

    Each state file found is claimed with a new token (so any other process still retrying it stops) and retried in
    the background until its original deadline. A job whose stored result says `upload_pending` but has no state file
    left gets an infra_error: its decode has to be run again.
    """
    resumed: set[slots.JobKey] = set()
    try:
        state_files = sorted(work_root.glob(f"*/{PENDING_UPLOAD_NAME}"))
    except OSError as e:
        logger.error(f"Could not look for pending uploads in {work_root}: {e}")
        state_files = []
    for state_file in state_files:
        work = _WorkDirs(root=state_file.parent)
        pending = decode_job.load_pending(work)
        if pending is None or decode_job.work_dirs(work_root, pending.task_id, pending.hotkey).root != work.root:
            continue
        try:
            decode_job.save_pending(work, pending.model_copy(update={"token": decode_job.new_token()}))
        except OSError as e:
            logger.error(f"Could not resume the decoded upload in {work.root}: {e}")
            continue
        logger.info(f"Resuming the decoded upload of task {pending.task_id} hotkey {pending.hotkey}")
        await log_task(pending.task_id, pending.hotkey, "decode: trainer restarted; resuming the decoded upload")
        start(work_root, pending.task_id, pending.hotkey)
        resumed.add((pending.task_id, pending.hotkey))

    for stored in read_persisted_jobs():
        if not isinstance(stored, TrainerTaskLog) or stored.quant_decode is None or not stored.quant_decode.upload_pending:
            continue
        key = (stored.training_data.task_id, stored.hotkey)
        if key in resumed:
            continue
        lost = stored.quant_decode.model_copy(
            update={"upload_pending": False, "upload_retry_until": None, "infra_error": LOST_IN_RESTART_FAILURE}
        )
        await update_quant_decode_result(*key, lost)
        await log_task(*key, result_message(lost))
