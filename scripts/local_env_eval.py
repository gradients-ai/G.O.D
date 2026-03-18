#!/usr/bin/env python3
"""
Manual environment evaluation script that reuses validator local env evaluation flow.

Edit the config constants below, then run:
    python -m scripts.manual_environment_eval
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from core.models.utility_models import EnvironmentDatasetType
from validator.evaluation.docker_evaluation import run_evaluation_local_environment


# --- Model Configuration ---
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_MODEL_NAME = "gradients-io-tournaments/tournament-tourn_0abde12f6c97e789_20260316-4d4dfde9-b82b-44b0-873a-cf4545344439-5GU4Xkd3"

# --- Evaluation Configuration ---
GAME_TO_EVAL = "liars_dice"
RANDOM_SEED = 1288591124
GPU_ID = 0
STREAM_SGLANG_LOGS = True
ENABLE_SGLANG_REQUEST_LOGS = True


def _setup_run_log_file() -> Path:
    log_dir = Path("/tmp/local_env_eval_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sglang-conversations-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(file_handler)
    return log_path

async def run_evaluation() -> None:
    dataset_type = EnvironmentDatasetType(environment_name=GAME_TO_EVAL)
    model_to_eval = LORA_MODEL_NAME or BASE_MODEL_NAME

    print(f"🚀 Running local environment evaluation for: {model_to_eval}")
    print(f"🎮 Environment: {GAME_TO_EVAL}")
    print(f"🎯 GPU ID: {GPU_ID}")
    print(f"🌱 Eval seed: {RANDOM_SEED}")
    log_path = _setup_run_log_file()
    print(f"📝 SGLang logs: {log_path}")
    os.environ["LOCAL_ENV_SGLANG_RAW_LOG_FILE"] = str(log_path)
    if STREAM_SGLANG_LOGS:
        os.environ["LOCAL_ENV_STREAM_SGLANG_LOGS"] = "1"
    if ENABLE_SGLANG_REQUEST_LOGS:
        os.environ["LOCAL_ENV_SGLANG_LOG_REQUESTS"] = "1"
        os.environ["LOCAL_ENV_SGLANG_LOG_REQUESTS_LEVEL"] = "3"
        os.environ["LOCAL_ENV_SGLANG_LOG_REQUESTS_FORMAT"] = "json"
        os.environ["LOCAL_ENV_SGLANG_LOG_REQUESTS_TARGET"] = "stdout"

    results = await run_evaluation_local_environment(
        models=[model_to_eval],
        original_model=BASE_MODEL_NAME,
        dataset_type=dataset_type,
        gpu_id=GPU_ID,
        eval_seed=RANDOM_SEED,
    )

    result_obj = results.results.get(model_to_eval)
    if isinstance(result_obj, Exception):
        raise RuntimeError(f"Evaluation failed: {result_obj}")

    print("\n✅ Evaluation complete.")
    print(f"Result for {model_to_eval}: {result_obj.model_dump()}")


if __name__ == "__main__":
    start = time.perf_counter()
    asyncio.run(run_evaluation())
    elapsed = time.perf_counter() - start
    print(f"Evaluation took: {elapsed:.2f} seconds")