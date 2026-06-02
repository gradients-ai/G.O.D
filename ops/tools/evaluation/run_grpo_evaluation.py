#!/usr/bin/env python3
"""
GRPO Evaluation Test Script

This script provides a simplified way to run GRPO evaluations for testing purposes.
It downloads a dataset from S3, prepares the environment, and runs the evaluation
in a Docker container similar to how it's done in production.

Usage:
    python -m ops.tools.evaluation.run_grpo_evaluation <s3_dataset_url> <original_model> <model1,model2,...>

Example:
    python -m ops.tools.evaluation.run_grpo_evaluation https://bucket.s3.amazonaws.com/dataset.jsonl \
        mistralai/Mistral-7B-v0.1 repo1/adapter,repo2/adapter
"""

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

import aiohttp


async def download_file(url, local_path):
    """Download a file from a URL to a local path."""
    print(f"Downloading {url} to {local_path}...")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with open(local_path, "wb") as f:
                    f.write(await response.read())
                print(f"Download completed: {local_path}")
                return local_path
            else:
                raise Exception(f"Failed to download file: {response.status}")


def run_docker_evaluation(dataset_path, original_model, models, dataset_type='{"field_prompt":"prompt"}', file_format="jsonl"):
    """Run the GRPO evaluation in a Docker container."""
    dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
    dataset_filename = os.path.basename(dataset_path)

    print(f"Running GRPO evaluation for models: {models}")

    # Map required environment variables
    env = {
        "DATASET": f"/workspace/input_data/{dataset_filename}",
        "MODELS": models,
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type,
        "FILE_FORMAT": file_format,
    }

    # Build the docker command
    cmd = [
        "docker",
        "run",
        "--rm",
        "--runtime",
        "nvidia",
        "--gpus",
        "all",
    ]

    # Add environment variables
    for key, value in env.items():
        cmd.extend(["-e", f"{key}={value}"])

    # Add volume mounts
    cmd.extend(
        [
            "-v",
            f"{dataset_dir}:/workspace/input_data:ro",
            "-v",
            f"{os.path.expanduser('~/.cache/huggingface')}:/root/.cache/huggingface:rw",
            "validator",
            "python",
            "-m",
            "validator.evaluation.evaluators.grpo",
        ]
    )

    # Run the docker command
    print("Executing docker command:")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print("Docker evaluation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Docker evaluation failed with exit code {e.returncode}")
        raise


async def main():
    parser = argparse.ArgumentParser(description="Run GRPO evaluation for testing")
    parser.add_argument("dataset_url", help="S3 URL to the dataset")
    parser.add_argument("original_model", help="Original base model to use")
    parser.add_argument("models", help="Comma-separated list of model repos to evaluate")

    args = parser.parse_args()

    # Create temporary directory for dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Download dataset from S3
            dataset_filename = os.path.basename(urlparse(args.dataset_url).path)
            local_dataset_path = os.path.join(temp_dir, dataset_filename)

            await download_file(args.dataset_url, local_dataset_path)

            # Run docker evaluation
            run_docker_evaluation(dataset_path=local_dataset_path, original_model=args.original_model, models=args.models)

            print("GRPO evaluation completed")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
