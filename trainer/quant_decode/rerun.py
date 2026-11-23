"""Ops re-run of one freeform decode: `python -m trainer.quant_decode.rerun <task_id> <hotkey>`.

The validator's orchestrator holds a freeform task until every successful hotkey has a decode outcome and re-polls
the trainer, so re-running the decode here after an `infra_error` (image missing, MinIO/HF down) is the whole fix:
the new outcome replaces the old one on the job in task_history.json and the next poll picks it up.

Runs the same decode as the training job (`trainer.runtime.freeform_decode_result`). The
presigned decoder snapshot URL in the stored request expires 7 days after the task was dispatched; past that the
re-run itself ends in an infra_error.

The decode holds the job's lock (trainer/quant_decode/job_lock.py), which the trainer's decodes and upload retries
take as well. A re-run started while the trainer is uploading for the same job waits for that attempt to end, then
replaces the pending upload: the trainer's background retry for it stops and stores nothing. When the re-run's own
upload fails for a transient reason, the command keeps retrying it in the foreground for the retry window
(trainer/quant_decode/upload_retry.py) before it exits. If the trainer restarts meanwhile, the trainer takes the
retry over and the command exits 3.

The command stores nothing when it never got the job's lock (exit 4): the trainer is still working on the job and
its result stands. When the command is interrupted (Ctrl-C, SIGTERM) after its decode replaced a pending upload, it
removes its work dir and stores an infra_error, so the job is not left pending with nobody retrying it.

Exit codes: 0 when the decode produced ok or quant_failure, 1 on infra_error, 2 when the job is unknown or has no
decode request, 3 when a restarted trainer took over the upload retry, 4 when the job stayed locked by the trainer,
130 when interrupted.
"""

import argparse
import asyncio
import signal
import sys
from functools import partial

import trainer.constants as cst
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestQuant
from core.models.quant_artifact_models import QuantDecodeResult
from trainer.job_state import get_task
from trainer.job_state import load_task_history
from trainer.job_state import update_quant_decode_result
from trainer.quant_decode import job as decode_job
from trainer.quant_decode import upload_retry
from trainer.quant_decode.job import JOB_BUSY_FAILURE
from trainer.quant_decode.job_lock import JobLock
from trainer.runtime import freeform_decode_result


EXIT_DECODED = 0
EXIT_INFRA_ERROR = 1
EXIT_NOT_RERUNNABLE = 2
EXIT_UPLOAD_TAKEN_OVER = 3
EXIT_JOB_BUSY = 4
EXIT_INTERRUPTED = 130
INTERRUPTED_FAILURE = "the decode re-run was interrupted"


async def _print_progress(message: str) -> None:
    print(message, file=sys.stderr)


def _proxy_request(job: TrainerTaskLog) -> TrainerProxyRequest:
    return TrainerProxyRequest(
        training_data=job.training_data,
        github_repo=job.github_repo,
        gpu_ids=job.gpu_ids,
        hotkey=job.hotkey,
        github_commit_hash=job.github_commit_hash,
        github_token=job.github_token,
        requested_datasets=job.requested_datasets,
    )


async def rerun_decode(task_id: str, hotkey: str) -> int:
    load_task_history()
    job = get_task(task_id, hotkey)
    if job is None:
        print(f"no training job for task {task_id} hotkey {hotkey} in task history", file=sys.stderr)
        return EXIT_NOT_RERUNNABLE
    if not isinstance(job.training_data, TrainRequestQuant) or job.training_data.decode is None:
        print(f"task {task_id} hotkey {hotkey} has no freeform decode request", file=sys.stderr)
        return EXIT_NOT_RERUNNABLE

    work = decode_job.work_dirs(cst.QUANT_DECODE_WORK_ROOT, task_id, hotkey)
    token_before = _pending_token(work)
    try:
        return await _rerun(job, task_id, hotkey)
    except BaseException:  # Ctrl-C and SIGTERM arrive as a cancellation; a crash is handled the same way
        try:
            await _settle_interrupted(work, task_id, hotkey, token_before)
        except Exception as e:
            print(f"could not settle the interrupted re-run: {e}", file=sys.stderr)
        raise


def _pending_token(work: decode_job._WorkDirs) -> str | None:
    pending = decode_job.load_pending(work)
    return None if pending is None else pending.token


async def _settle_interrupted(work: decode_job._WorkDirs, task_id: str, hotkey: str, token_before: str | None) -> None:
    """After an interruption: if this command's decode replaced the job's pending upload, nobody retries the job any
    more, so remove what the decode left and turn a stored `upload_pending` into an infra_error.

    Nothing is done while the job's lock is held by the trainer, or while the pending upload in the work dir is still
    the one from before the command (its decode never started and the trainer's retry is alive).
    """
    lock = JobLock(cst.QUANT_DECODE_WORK_ROOT, work.root.name)
    try:
        if not lock.try_acquire():
            return
    except OSError:
        return
    try:
        if _pending_token(work) == token_before:
            return
        decode_job.remove_work_tree(work.root)
        load_task_history()
        stored = get_task(task_id, hotkey)
        if stored is not None and stored.quant_decode is not None and stored.quant_decode.upload_pending:
            await update_quant_decode_result(task_id, hotkey, QuantDecodeResult(infra_error=INTERRUPTED_FAILURE))
            print(f"stored an infra_error on the job: {INTERRUPTED_FAILURE}", file=sys.stderr)
    finally:
        lock.release()


async def _rerun(job: TrainerTaskLog, task_id: str, hotkey: str) -> int:
    result = await freeform_decode_result(_proxy_request(job))
    if result.infra_error == JOB_BUSY_FAILURE:
        print(f"{JOB_BUSY_FAILURE}; nothing was stored, the trainer's result stands. Try again later.", file=sys.stderr)
        return EXIT_JOB_BUSY
    await update_quant_decode_result(task_id, hotkey, result)
    if result.upload_pending:
        print(upload_retry.result_message(result), file=sys.stderr)
        store = partial(update_quant_decode_result, task_id, hotkey)
        result = await upload_retry.retry_pending_upload(cst.QUANT_DECODE_WORK_ROOT, task_id, hotkey, _print_progress, store)
        if result is None:
            print("the trainer took over the upload retry; its result will be stored on the job", file=sys.stderr)
            return EXIT_UPLOAD_TAKEN_OVER
    print(result.model_dump_json(indent=2))
    return EXIT_INFRA_ERROR if result.infra_error is not None else EXIT_DECODED


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run the freeform decode of one trainer job and store its outcome.")
    parser.add_argument("task_id")
    parser.add_argument("hotkey")
    args = parser.parse_args()
    try:
        sys.exit(asyncio.run(_run_until_signalled(args.task_id, args.hotkey)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        sys.exit(EXIT_INTERRUPTED)


async def _run_until_signalled(task_id: str, hotkey: str) -> int:
    """SIGTERM cancels the re-run like Ctrl-C does, so both go through its interruption handling."""
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    return await rerun_decode(task_id, hotkey)


if __name__ == "__main__":
    main()
