#!/usr/bin/env python3
"""
Run a manual Basilica environment evaluation for a fixed list of repos.

Usage:
    python -m scripts.test_basilica_env_eval
"""

import asyncio
import contextlib
import time
from datetime import datetime

from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import FileFormat
from validator.evaluation.docker_evaluation import run_evaluation_basilica_text


REPOS = [
    "gradients-io-tournaments/tournament-tourn_0abde12f6c97e789_20260316-42eed3e7-e4b1-41e5-9424-6eb29a0e63b2-5FYr4ssC",
    "gradients-io-tournaments/tournament-tourn_0abde12f6c97e789_20260316-42eed3e7-e4b1-41e5-9424-6eb29a0e63b2-5FkHMWCC"
]

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ENV_NAME = "gin_rummy"
NUM_GPUS = 1
EVAL_SEED = 42


def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


async def _heartbeat(started_at: float, interval_seconds: int = 20) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        elapsed = time.perf_counter() - started_at
        print(f"[{_ts()}] still running... elapsed={elapsed:.1f}s", flush=True)


async def main() -> None:
    dataset_type = EnvironmentDatasetType(environment_name=ENV_NAME)
    started_at = time.perf_counter()

    print(f"[{_ts()}] Starting Basilica environment evaluation", flush=True)
    print(f"Base model: {BASE_MODEL}", flush=True)
    print(f"Environment: {ENV_NAME}", flush=True)
    print(f"Repos ({len(REPOS)}):", flush=True)
    for repo in REPOS:
        print(f"  - {repo}", flush=True)

    # `dataset` is unused for EnvironmentDatasetType but required by signature.
    heartbeat_task = asyncio.create_task(_heartbeat(started_at))
    try:
        print(f"[{_ts()}] invoking run_evaluation_basilica_text(...)", flush=True)
        results = await run_evaluation_basilica_text(
            dataset="unused_for_environment_eval",
            models=REPOS,
            original_model=BASE_MODEL,
            dataset_type=dataset_type,
            file_format=FileFormat.S3,
            num_gpus=NUM_GPUS,
            eval_seed=EVAL_SEED,
        )
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

    print(f"\n[{_ts()}] Evaluation complete.", flush=True)
    print(f"Base model parameter count: {results.base_model_params_count}", flush=True)
    print("\nPer-repo results:", flush=True)
    for repo in REPOS:
        result = results.results.get(repo)
        if isinstance(result, Exception):
            print(f"  - {repo}: FAILURE -> {result}", flush=True)
        else:
            payload = result.model_dump()
            eval_loss = payload.get("eval_loss")
            score = -eval_loss if isinstance(eval_loss, (int, float)) else "n/a"
            print(f"  - {repo}: SUCCESS -> eval_loss={eval_loss}, approx_score={score}", flush=True)


if __name__ == "__main__":
    start = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - start
    print(f"\nTotal runtime: {elapsed:.2f}s")
