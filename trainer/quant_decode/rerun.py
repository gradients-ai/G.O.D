"""Ops re-run of one freeform decode: `python -m trainer.quant_decode.rerun <task_id> <hotkey>`.

The validator's orchestrator holds a freeform task until every successful hotkey has a decode outcome and re-polls
the trainer, so re-running the decode here after an `infra_error` (image missing, MinIO/HF down) is the whole fix:
the new outcome replaces the old one on the job in task_history.json and the next poll picks it up.

Runs the same decode as the training job (`trainer.runtime.freeform_decode_result`). The
presigned decoder snapshot URL in the stored request expires 7 days after the task was dispatched; past that the
re-run itself ends in an infra_error.

The re-run is its own process, so it does not take one of the running trainer's decode or upload slots
(trainer/slots.py): it starts at once and shares the network link with whatever the trainer is transferring.

The decode holds the job's lock (trainer/quant_decode/job_lock.py), which the trainer's decodes and upload retries
take as well. A re-run started while the trainer is uploading for the same job waits for that attempt to end, then
replaces the pending upload: the trainer's background retry for it stops and stores nothing. When the re-run's own
upload fails for a transient reason, the command keeps retrying it in the foreground for the retry window
(trainer/quant_decode/upload_retry.py) before it exits. If the trainer restarts meanwhile, the trainer takes the
retry over and the command exits 3.

Exit codes: 0 when the decode produced ok or quant_failure, 1 on infra_error, 2 when the job is unknown or has no
decode request, 3 when a restarted trainer took over the upload retry.
"""

import argparse
import asyncio
import sys
from functools import partial

import trainer.constants as cst
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestQuant
from trainer.job_state import get_task
from trainer.job_state import load_task_history
from trainer.job_state import update_quant_decode_result
from trainer.quant_decode import upload_retry
from trainer.runtime import freeform_decode_result


EXIT_DECODED = 0
EXIT_INFRA_ERROR = 1
EXIT_NOT_RERUNNABLE = 2
EXIT_UPLOAD_TAKEN_OVER = 3


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

    result = await freeform_decode_result(_proxy_request(job))
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
    sys.exit(asyncio.run(rerun_decode(args.task_id, args.hotkey)))


if __name__ == "__main__":
    main()
