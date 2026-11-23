"""Background retry of a decoded upload that failed for a transient reason.

When a decode passes its checks but the decoded files cannot be uploaded because of the network, the job ends with an
`upload_pending` decode result and its decoded files stay on disk (job.py). This module then retries in rounds until
the retry window closes. A round uploads only the files that are not in storage yet, one attempt each; after a failed
round it waits (exponential backoff with jitter) and tries again. No upload slot is held while it waits. The result
on the job becomes `ok` as soon as a round completes, or `infra_error` once the window is spent or an error turns up
that no retry can fix.

Ownership. A round holds the job's lock (job_lock.py) from its first look at the work dir until its result is stored
and the dir removed, and so does every decode. Under the lock the round checks that the state file still carries its
token, again after each uploaded file and before it settles. A retry that finds the lock taken, or another token in
the work dir, was replaced (by a new decode of the job, the re-run CLI, or a restarted trainer): it stops and stores
nothing.

The state lives in the work dir, so a restarted trainer resumes the retry (`resume_after_restart`), and removes every
work dir that has nothing to resume.
"""

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
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
from trainer.quant_decode.job import WORK_ROOT_MARKER
from trainer.quant_decode.job import PendingUpload
from trainer.quant_decode.job import ProgressReporter
from trainer.quant_decode.job import UploadFault
from trainer.quant_decode.job import UploadOptions
from trainer.quant_decode.job import _WorkDirs
from trainer.quant_decode.job_lock import LOCK_DIR_NAME
from trainer.quant_decode.job_lock import LOCK_SUFFIX
from trainer.quant_decode.job_lock import JobLock
from trainer.quant_decode.sandbox import sanitize_failure
from trainer.quant_decode.upload import UploadRetryPolicy
from trainer.telemetry import logger


SECONDS_PER_MINUTE = 60
LOST_IN_RESTART_FAILURE = "the retry of the decoded upload was lost when the trainer restarted"

ResultStore = Callable[[QuantDecodeResult], Awaitable[None]]

# One background retry per job in this process.
_retries: dict[slots.JobKey, asyncio.Task] = {}


class _TakenOver(Exception):
    """The work dir no longer belongs to this retry."""


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


def _require_owner(work: _WorkDirs, pending: PendingUpload) -> None:
    if not decode_job.owns_pending(work, pending.token):
        raise _TakenOver()


def _save_progress(work: _WorkDirs, pending: PendingUpload) -> None:
    """Save the state after a file landed, if the work dir is still ours. A failed write is only logged: the upload
    goes on, and a restart finds the files this write would have recorded by asking storage."""
    _require_owner(work, pending)
    try:
        decode_job.save_pending(work, pending)
    except OSError as e:
        logger.error(f"Could not save the upload progress in {work.root}: {e}")


async def _report_safely(report: ProgressReporter, message: str) -> None:
    """A progress line that cannot be written must not end the retry."""
    try:
        await report(message)
    except Exception as e:
        logger.error(f"Could not report upload retry progress ({type(e).__name__}): {message}")


async def _round(
    work: _WorkDirs, pending: PendingUpload, options: UploadOptions, report: ProgressReporter
) -> QuantDecodeResult | None:
    """One pass over the files still missing, with the job's lock held. The final result, or None when the round
    failed for a transient reason and the window is still open. Raises _TakenOver."""
    await asyncio.to_thread(_require_owner, work, pending)
    try:
        await decode_job.upload_decoded(
            pending.request,
            work,
            pending.progress,
            task_id=pending.task_id,
            hotkey=pending.hotkey,
            report=report,
            options=options,
            file_done=partial(_save_progress, work, pending),
        )
    except _TakenOver:
        raise
    except Exception as e:  # an UploadFault, or a local error such as the decoded files being gone
        await asyncio.to_thread(_require_owner, work, pending)
        if not isinstance(e, UploadFault):
            logger.exception(f"Upload retry crashed for task {pending.task_id} hotkey {pending.hotkey}")
            return _infra_error(pending, f"upload retry failed ({type(e).__name__})")
        if not e.transient:
            return _infra_error(pending, str(e))
        pending.last_error = sanitize_failure(str(e))
        await asyncio.to_thread(_save_progress, work, pending)
        if _now() >= pending.retry_until:
            window_minutes = cst.QUANT_DECODE_UPLOAD_RETRY_WINDOW_SECONDS / SECONDS_PER_MINUTE
            return _infra_error(pending, f"{pending.last_error}; gave up after retrying for {window_minutes:.0f} min")
        return None
    await asyncio.to_thread(_require_owner, work, pending)
    return QuantDecodeResult(ok=True, **pending.progress.model_dump())


async def retry_pending_upload(
    work_root: Path,
    task_id: str,
    hotkey: str,
    report: ProgressReporter = _no_report,
    store: ResultStore | None = None,
) -> QuantDecodeResult | None:
    """Retry the job's pending upload until it lands or its window closes; return the final decode result.

    The final result is handed to `store` and only then is the work dir removed, both with the job's lock held, so a
    result is never lost between the two and never stored by a retry that was replaced.

    None when there is nothing to retry or another decode or retry took the work dir over: nothing is stored.
    """
    work = decode_job.work_dirs(work_root, task_id, hotkey)
    pending = await asyncio.to_thread(decode_job.load_pending, work)
    if pending is None:
        return None
    report = partial(_report_safely, report)
    lock = JobLock(work_root, work.root.name)
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
        try:
            if not lock.try_acquire():
                return None  # a decode of this job is running: it replaces this upload
        except OSError:
            return None  # the work root is gone
        try:
            result = await _round(work, pending, options, report)
            if result is not None:
                if store is not None:
                    await store(result)
                await asyncio.to_thread(decode_job.remove_work_tree, work.root)
                return result
        except _TakenOver:
            return None
        finally:
            lock.release()
        failed_rounds += 1


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
    try:
        result = await retry_pending_upload(
            work_root,
            task_id,
            hotkey,
            partial(log_task, task_id, hotkey),
            partial(update_quant_decode_result, task_id, hotkey),
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        # The state file is still on disk: the next trainer start resumes it.
        logger.exception(f"Upload retry for task {task_id} hotkey {hotkey} stopped on an unexpected error")
        return
    if result is None:
        logger.info(f"Upload retry for task {task_id} hotkey {hotkey} was taken over; storing nothing")
        return
    message = result_message(result)
    logger.info(f"{message} | task={task_id} hotkey={hotkey}")
    await _report_safely(partial(log_task, task_id, hotkey), message)


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


def cancel(key: slots.JobKey) -> None:
    """Stop the job's background retry, if any. Called before a new decode of the same job takes the work dir."""
    task = _retries.pop(key, None)
    if task is not None:
        task.cancel()


def _claim(work_root: Path, entry: Path) -> PendingUpload | None:
    """With the entry's lock held: give its pending upload a new token and return it, or remove the entry when it has
    nothing to resume (a decode cut short by the restart, a leftover moved aside)."""
    work = _WorkDirs(root=entry)
    pending = decode_job.load_pending(work) if entry.is_dir() and not entry.is_symlink() else None
    if pending is None or decode_job.work_dirs(work_root, pending.task_id, pending.hotkey).root != entry:
        decode_job.remove_work_tree(entry)
        logger.info(f"Removed decode work dir {entry}: nothing to resume")
        return None
    claimed = pending.model_copy(update={"token": decode_job.new_token()})
    decode_job.save_pending(work, claimed)
    return claimed


def _claim_work_dirs(work_root: Path) -> list[PendingUpload]:
    """Claim every resumable pending upload in the work root and remove the other work dirs. A dir whose lock is held
    (the re-run CLI is working in it) is left alone."""
    claimed: list[PendingUpload] = []
    for entry in sorted(work_root.iterdir()):
        if entry.name == LOCK_DIR_NAME or not (entry.is_dir() or entry.is_symlink()):
            continue
        lock = JobLock(work_root, entry.name)
        try:
            if not lock.try_acquire():
                continue
            try:
                pending = _claim(work_root, entry)
            finally:
                lock.release()
        except OSError as e:
            logger.error(f"Could not resume or remove decode work dir {entry}: {e}")
            continue
        if pending is not None:
            claimed.append(pending)
    return claimed


def _remove_unused_locks(work_root: Path) -> None:
    for lock_file in (work_root / LOCK_DIR_NAME).glob(f"*{LOCK_SUFFIX}"):
        name = lock_file.name.removesuffix(LOCK_SUFFIX)
        if (work_root / name).exists():
            continue
        lock = JobLock(work_root, name)
        try:
            if lock.try_acquire():
                lock.remove_and_release()
        except OSError as e:
            logger.warning(f"Could not remove unused lock {lock_file}: {e}")


async def _settle_lost(work_root: Path, resumed: set[slots.JobKey]) -> None:
    """A job stored as upload_pending that nothing is retrying gets an infra_error, unless its lock is held (the
    re-run CLI is decoding it and will store its own result)."""
    for stored in read_persisted_jobs():
        if not isinstance(stored, TrainerTaskLog) or stored.quant_decode is None or not stored.quant_decode.upload_pending:
            continue
        key = (stored.training_data.task_id, stored.hotkey)
        if key in resumed:
            continue
        lock = JobLock(work_root, decode_job.work_dirs(work_root, *key).root.name)
        try:
            if not lock.try_acquire():
                continue
        except OSError:
            pass  # no work root at all: nothing can be retrying it
        try:
            lost = stored.quant_decode.model_copy(
                update={
                    "upload_pending": False,
                    "upload_retry_until": None,
                    "upload_settle_deadline": None,
                    "infra_error": LOST_IN_RESTART_FAILURE,
                }
            )
            await update_quant_decode_result(*key, lost)
            await log_task(*key, result_message(lost))
        finally:
            lock.release()


async def resume_after_restart(work_root: Path) -> None:
    """Pick up every pending upload left in the work root, settle the ones that cannot be picked up, and free the
    disk of work dirs that have nothing to resume.

    Each pending upload found is claimed with a new token (so a retry from before the restart, in any process, no
    longer owns it) and retried in the background until its original deadline. A job whose stored result says
    `upload_pending` but has no state left gets an infra_error: its decode has to be run again. The work root is only
    swept when it carries the decode job's marker.
    """
    resumed: set[slots.JobKey] = set()
    if (work_root / WORK_ROOT_MARKER).exists():
        try:
            claimed = await asyncio.to_thread(_claim_work_dirs, work_root)
        except OSError as e:
            logger.error(f"Could not look for pending uploads in {work_root}: {e}")
            claimed = []
        for pending in claimed:
            logger.info(f"Resuming the decoded upload of task {pending.task_id} hotkey {pending.hotkey}")
            await log_task(pending.task_id, pending.hotkey, "decode: trainer restarted; resuming the decoded upload")
            start(work_root, pending.task_id, pending.hotkey)
            resumed.add((pending.task_id, pending.hotkey))
    await _settle_lost(work_root, resumed)
    if (work_root / LOCK_DIR_NAME).is_dir():
        await asyncio.to_thread(_remove_unused_locks, work_root)
