#!/usr/bin/env python3
"""
Testing script for image evaluation with Basilica.

Runs diffusion (img2img) evaluation on Basilica cloud using the provided test data URL
and base model. Evaluates one or more LoRA models trained on the base model.

Prerequisites:
    - BASILICA_API_TOKEN set (create via: basilica tokens create)

Usage:
    cd /root/G.O.D && python -m scripts.test_basilica_image_eval

Configuration: edit the constants below.
"""

import asyncio
import sys

from core.models.utility_models import ImageModelType
from validator.evaluation.docker_evaluation import run_evaluation_basilica_image
from validator.utils.logging import get_logger


logger = get_logger(__name__)

# --- Configuration ---
# Signed URL for test data zip (refresh if expired)
TEST_DATA_URL = (
    "https://s3.eu-central-003.backblazeb2.com/gradients-validator/19770471784dac34_test_data.zip"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00362e8d6b742200000000002%2F20260214"
    "%2Feu-central-003%2Fs3%2Faws4_request&X-Amz-Date=20260214T005422Z&X-Amz-Expires=604800"
    "&X-Amz-SignedHeaders=host&X-Amz-Signature=347b74152348fe50ce96545d5fec20130da2775a4b0bff10b835434197c71f88"
)
BASE_MODEL = "Lykon/dreamshaper-xl-1-0"
MODEL_TYPE = ImageModelType.SDXL

# LoRA model repos to evaluate (must be HuggingFace repos containing SDXL LoRA weights).
# Replace with your own LoRA repos or use a known public SDXL LoRA for testing.
MODELS_TO_EVALUATE = [
    "gradients-io-tournaments/tournament-tourn_6794b4cf614185a9_20260212-22b8c9a1-1ae7-4484-83ba-e8b1b8ebaccc-5Gy6X7q2",  # Example public SDXL LoRA for pipeline testing
]

GPU_IDS = [0]  # Used for gpu_count; Basilica selects GPUs from its pool


async def main():
    logger.info("Starting Basilica image evaluation")
    logger.info(f"  Test data URL: {TEST_DATA_URL[:80]}...")
    logger.info(f"  Base model: {BASE_MODEL}")
    logger.info(f"  Model type: {MODEL_TYPE.value}")
    logger.info(f"  Models to evaluate: {MODELS_TO_EVALUATE}")

    results = await run_evaluation_basilica_image(
        test_split_url=TEST_DATA_URL,
        original_model_repo=BASE_MODEL,
        models=MODELS_TO_EVALUATE,
        model_type=MODEL_TYPE,
        gpu_ids=GPU_IDS,
    )

    logger.info("Evaluation complete")
    has_failures = False
    for repo, result in results.results.items():
        if isinstance(result, Exception):
            logger.error(f"  {repo}: FAILED - {result}")
            has_failures = True
        else:
            eval_loss = result.eval_loss
            logger.info(f"  {repo}: {eval_loss}")

    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
