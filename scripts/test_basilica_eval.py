#!/usr/bin/env python3
"""
Test script for run_evaluation_docker_environment using Basilica deployments.

This script tests the production evaluation function with Basilica deployments.
Logs for each repo are written to separate files in the logs/ directory.
Results from each run are saved to JSON files.
"""
import asyncio
import json
import logging
import os
from datetime import datetime

from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import FileFormat
from validator.evaluation.docker_evaluation import run_evaluation_docker_environment


BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" 

MODELS = [
    "gradients-io-tournaments/tournament-tourn_5b58cbbb12b8c212_20260130-2c0c4a91-4bed-4e5d-ab09-f04d17659b03-5GU4Xkd3",
    "gradients-io-tournaments/tournament-tourn_5b58cbbb12b8c212_20260130-2c0c4a91-4bed-4e5d-ab09-f04d17659b03-5C7vE26G",
    "gradients-io-tournaments/tournament-tourn_5b58cbbb12b8c212_20260130-2c0c4a91-4bed-4e5d-ab09-f04d17659b03-5Dt9U4c1",
    "gradients-io-tournaments/tournament-tourn_5b58cbbb12b8c212_20260130-2c0c4a91-4bed-4e5d-ab09-f04d17659b03-5Ca32LwM",
    "gradients-io-tournaments/tournament-tourn_5b58cbbb12b8c212_20260130-2c0c4a91-4bed-4e5d-ab09-f04d17659b03-5D2Qee4V",
    "gradients-io-tournaments/tournament-tourn_5b58cbbb12b8c212_20260130-2c0c4a91-4bed-4e5d-ab09-f04d17659b03-5DiBMCma"
]
DATASET = "test_dataset"

# Number of evaluation runs
NUM_RUNS = 3

# Directory for log files
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")


def setup_file_logging_for_repos(models: list[str], run_num: int, timestamp: str) -> dict[str, str]:
    """Set up file handlers for each repo's environment logger.
    
    Returns:
        Dict mapping repo name to log file path
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    log_files = {}
    
    for idx, repo in enumerate(models):
        repo_name = repo.split("/")[-1]
        # Create unique log file for each repo (with run number and index)
        log_filename = f"{timestamp}_run{run_num}_repo{idx}_{repo_name[:50]}.log"
        log_path = os.path.join(LOGS_DIR, log_filename)
        
        # Get or create the logger that will be used by the evaluation
        logger_name = f"environment.{repo_name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Add file handler
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
        
        log_files[repo] = log_path
        print(f"    Repo {idx}: {repo_name[-20:]} → {log_filename}")
    
    return log_files


def clear_repo_loggers(models: list[str]):
    """Clear file handlers from repo loggers between runs."""
    for repo in models:
        repo_name = repo.split("/")[-1]
        logger_name = f"environment.{repo_name}"
        logger = logging.getLogger(logger_name)
        # Remove all file handlers
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]


def save_results_to_json(results, run_num: int, timestamp: str) -> str:
    """Save evaluation results to a JSON file.
    
    Returns:
        Path to the saved JSON file
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Convert results to serializable format
    results_dict = {
        "run_number": run_num,
        "timestamp": datetime.now().isoformat(),
        "base_model": BASE_MODEL_NAME,
        "models": MODELS,
        "base_model_params_count": results.base_model_params_count,
        "results": {}
    }
    
    for repo, result in results.results.items():
        if isinstance(result, Exception):
            results_dict["results"][repo] = {"error": str(result)}
        else:
            # Convert pydantic model to dict
            results_dict["results"][repo] = result.model_dump() if hasattr(result, 'model_dump') else str(result)
    
    json_filename = f"{timestamp}_run{run_num}_results.json"
    json_path = os.path.join(LOGS_DIR, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return json_path


async def run_single_evaluation(run_num: int, timestamp: str) -> tuple:
    """Run a single evaluation and return results."""
    print(f"\n{'='*60}")
    print(f"RUN {run_num}/{NUM_RUNS}")
    print(f"{'='*60}")
    
    # Set up file logging for this run
    print(f"  Setting up log files for run {run_num}:")
    log_files = setup_file_logging_for_repos(MODELS, run_num, timestamp)
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
            eval_seed=423516563,
        )
        
        # Save results to JSON
        json_path = save_results_to_json(results, run_num, timestamp)
        print(f"\n  Results saved to: {json_path}")
        
        # Print summary
        print(f"\n  Run {run_num} Results Summary:")
        for repo, result in results.results.items():
            repo_name = repo.split("/")[-1][-20:]
            if isinstance(result, Exception):
                print(f"    {repo_name}: ERROR - {result}")
            else:
                eval_loss = getattr(result, 'eval_loss', 'N/A')
                print(f"    {repo_name}: eval_loss={eval_loss}")
        
        # Clear loggers for next run
        clear_repo_loggers(MODELS)
        
        return (run_num, results, json_path, None)
        
    except Exception as e:
        clear_repo_loggers(MODELS)
        return (run_num, None, None, e)


async def main():
    """Run test evaluation multiple times."""
    # Check for API token
    if not os.getenv("BASILICA_API_TOKEN"):
        print("Error: BASILICA_API_TOKEN not set")
        print("  export BASILICA_API_TOKEN=your-token")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("Testing run_evaluation_docker_environment with Basilica (GoofSpiel)")
    print("=" * 60)
    print(f"Base Model: {BASE_MODEL_NAME}")
    print(f"Models to Evaluate: {len(MODELS)} repos")
    print(f"Environment: goofspiel")
    print(f"Number of Runs: {NUM_RUNS}")
    print(f"Logs Directory: {LOGS_DIR}")
    
    all_results = []
    json_files = []
    
    for run_num in range(1, NUM_RUNS + 1):
        run_num, results, json_path, error = await run_single_evaluation(run_num, timestamp)
        
        if error:
            print(f"\n❌ Run {run_num} failed: {error}")
            import traceback
            traceback.print_exc()
        else:
            all_results.append((run_num, results))
            json_files.append(json_path)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Completed {len(all_results)}/{NUM_RUNS} runs successfully")
    print(f"\nJSON Results Files:")
    for json_path in json_files:
        print(f"  → {json_path}")
    
    # Print aggregate scores per repo
    if all_results:
        print(f"\nScores by Repo (across all runs):")
        repo_scores = {repo: [] for repo in MODELS}
        for run_num, results in all_results:
            for repo, result in results.results.items():
                if not isinstance(result, Exception):
                    eval_loss = getattr(result, 'eval_loss', None)
                    if eval_loss is not None:
                        repo_scores[repo].append(eval_loss)
        
        for repo, scores in repo_scores.items():
            repo_name = repo.split("/")[-1][-25:]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"  {repo_name}: {scores} (avg: {avg:.4f})")
            else:
                print(f"  {repo_name}: No scores")
    
    print("\n✅ All runs completed!")


if __name__ == "__main__":
    asyncio.run(main())
