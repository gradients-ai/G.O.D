#!/usr/bin/env python3
"""
Manual environment evaluation script that reuses validator local env evaluation flow.

Edit the config constants below, then run:
    python -m scripts.manual_environment_eval
"""

import asyncio
import subprocess
import time

from core.models.utility_models import EnvironmentDatasetType
from validator.evaluation.docker_evaluation import run_evaluation_local_environment


# --- Model Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_MODEL_NAME = None  # e.g. "your-org/your-lora-repo"

# --- Evaluation Configuration ---
GAME_TO_EVAL = "gin_rummy"
RANDOM_SEED = 42
GPU_ID = 0

async def run_evaluation() -> None:
    dataset_type = EnvironmentDatasetType(environment_name=GAME_TO_EVAL)
    model_to_eval = LORA_MODEL_NAME or BASE_MODEL_NAME

    print(f"🚀 Running local environment evaluation for: {model_to_eval}")
    print(f"🎮 Environment: {GAME_TO_EVAL}")
    print(f"🎯 GPU ID: {GPU_ID}")
    print(f"🌱 Eval seed: {RANDOM_SEED}")

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