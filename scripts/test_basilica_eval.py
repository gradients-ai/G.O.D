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
    "gradients-io-tournaments/goofspiel_8x_h100_mcts_opponent_thinking_4hr"
]
DATASET = "test_dataset"


async def main():
    """Run test evaluation."""
    # Check for API token
    if not os.getenv("BASILICA_API_TOKEN"):
        print("Error: BASILICA_API_TOKEN not set")
        print("  export BASILICA_API_TOKEN=your-token")
        return
    
    print("=" * 60)
    print("Testing run_evaluation_docker_environment with Basilica (GoofSpiel)")
    print("=" * 60)
    print(f"Base Model: {BASE_MODEL_NAME}")
    print(f"Models to Evaluate: {MODELS}")
    print(f"Environment: goofspiel")
    print(f"Dataset: {DATASET}")
    print()
    
    dataset_type = EnvironmentDatasetType(environment_name="goofspiel")
    
    try:
        results = await run_evaluation_docker_environment(
            dataset=DATASET,
            models=MODELS,
            original_model=BASE_MODEL_NAME,
            dataset_type=dataset_type,
            file_format=FileFormat.JSON,
            gpu_ids=[], 
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
