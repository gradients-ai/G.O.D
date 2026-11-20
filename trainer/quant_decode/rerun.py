"""Ops re-run of one freeform decode: `python -m trainer.quant_decode.rerun <task_id> <hotkey>`.

The validator's orchestrator holds a freeform task until every successful hotkey has a decode outcome and re-polls
the trainer, so re-running the decode here after an `infra_error` (image missing, MinIO/HF down) is the whole fix:
the new outcome replaces the old one on the job in task_history.json and the next poll picks it up.

Runs the same decode as the training job (`trainer.runtime.freeform_decode_result`). The
presigned decoder snapshot URL in the stored request expires 7 days after the task was dispatched; past that the
re-run itself ends in an infra_error.

Exit codes: 0 when the decode produced ok or quant_failure, 1 on infra_error, 2 when the job is unknown or has no
decode request.
"""

import argparse
import asyncio
import sys

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainerTaskLog
from core.models.payload_models import TrainRequestQuant
from trainer.job_state import get_task
from trainer.job_state import load_task_history
from trainer.job_state import update_quant_decode_result
from trainer.runtime import freeform_decode_result


EXIT_DECODED = 0
EXIT_INFRA_ERROR = 1
EXIT_NOT_RERUNNABLE = 2


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
