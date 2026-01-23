#!/usr/bin/env python3
"""
Test script for run_evaluation_docker_environment using Basilica deployments.

This script tests the production evaluation function with Basilica deployments.
"""
import asyncio
import os

from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import FileFormat
from validator.evaluation.docker_evaluation import run_evaluation_docker_environment


BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" 

MODELS = [
    "gradients-io-tournaments/affine_gin_rummy_random"
]

GAME_NAME = "gin_rummy"
GAMES_TO_TASK_ID_RANGE = {
    "gin_rummy": (300000000, 399999999),
}
TASK_ID_RANGE = GAMES_TO_TASK_ID_RANGE[GAME_NAME]
AGENTGYM_IMAGE = "diagonalge/openspiel:latest" 

# Test configuration
DATASET = "test_dataset"  
NUM_EVAL_SAMPLES = 500 


async def main():
    """Run test evaluation."""
    # Check for API token
    if not os.getenv("BASILICA_API_TOKEN"):
        print("Error: BASILICA_API_TOKEN not set")
        print("  export BASILICA_API_TOKEN=your-token")
        return
    
    print("=" * 60)
    print("Testing run_evaluation_docker_environment with Basilica (Gin Rummy)")
    print("=" * 60)
    print(f"Base Model: {BASE_MODEL_NAME}")
    print(f"Models to Evaluate: {MODELS}")
    print(f"Game: {GAME_NAME}")
    print(f"Task ID Range: {TASK_ID_RANGE}")
    print(f"AgentGym Image: {AGENTGYM_IMAGE}")
    print(f"Dataset: {DATASET}")
    print(f"Num Eval Samples: {NUM_EVAL_SAMPLES}")
    print()
    
    dataset_type = EnvironmentDatasetType(environment_name=GAME_NAME)
    
    try:
        results = await run_evaluation_docker_environment(
            dataset=DATASET,
            models=MODELS,
            original_model=BASE_MODEL_NAME,
            dataset_type=dataset_type,
            file_format=FileFormat.JSON,
            gpu_ids=[], 
            num_eval_samples=NUM_EVAL_SAMPLES,
            env_image=AGENTGYM_IMAGE,
            task_id_range=TASK_ID_RANGE,
        )
        
        print("\n" + "=" * 60)
        print("Evaluation Results:")
        print("=" * 60)
        print(results)
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
