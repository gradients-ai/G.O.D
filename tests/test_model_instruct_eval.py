#!/usr/bin/env python3
"""
Test script for instruct evaluation of GPT-OSS-20B model.
This runs the evaluation in the proper Docker container.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def create_test_dataset(output_path: Path) -> str:
    """Create a small test dataset for instruct evaluation."""
    test_data = [
        {
            "instruction": "What is the capital of France?",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Explain photosynthesis in simple terms.",
            "output": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to make their own food (glucose) and release oxygen as a byproduct."
        },
        {
            "instruction": "Write a Python function to calculate factorial.",
            "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"
        },
        {
            "instruction": "What are the three states of matter?",
            "output": "The three states of matter are solid, liquid, and gas."
        },
        {
            "instruction": "Translate 'Hello, how are you?' to Spanish.",
            "output": "Hola, ¿cómo estás?"
        },
        {
            "instruction": "List three benefits of regular exercise.",
            "output": "Three benefits of regular exercise are: 1) Improved cardiovascular health, 2) Better mood and mental health, 3) Increased strength and endurance."
        },
        {
            "instruction": "What is machine learning?",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "instruction": "Calculate 15% of 200.",
            "output": "15% of 200 is 30."
        },
        {
            "instruction": "Name the planets in our solar system.",
            "output": "The planets in our solar system are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."
        },
        {
            "instruction": "What is the difference between a compiler and an interpreter?",
            "output": "A compiler translates the entire source code into machine code before execution, while an interpreter translates and executes code line by line during runtime."
        }
    ]
    
    # Save as JSON file
    dataset_file = output_path / "test_instruct_dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created test dataset at: {dataset_file}")
    return str(dataset_file)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test GPT-OSS-20B instruct evaluation in Docker container')
    parser.add_argument('--model', type=str, default='openai/gpt-oss-20b', help='Model ID to evaluate')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset JSON file')
    parser.add_argument('--output-dir', type=str, default='./test_eval_output', help='Output directory')
    parser.add_argument('--field-instruction', type=str, default='instruction', help='Field name for instructions')
    parser.add_argument('--field-output', type=str, default='output', help='Field name for outputs')
    parser.add_argument('--container-image', type=str, default='weightswandering/tuning_vali:latest', help='Docker image to use')
    parser.add_argument('--dry-run', action='store_true', help='Just print the docker command without running')
    
    args = parser.parse_args()
    
    print("="*60)
    print("GPT-OSS-20B Instruct Evaluation Test (Docker Container)")
    print("="*60)
    
    # Configuration
    MODEL_ID = args.model
    OUTPUT_DIR = Path(args.output_dir).absolute()
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create or use provided dataset
    if args.dataset:
        dataset_path = Path(args.dataset).absolute()
        if not dataset_path.exists():
            print(f"Error: Dataset file {dataset_path} does not exist")
            sys.exit(1)
        print(f"Using provided dataset: {dataset_path}")
    else:
        dataset_path = Path(create_test_dataset(OUTPUT_DIR)).absolute()
    
    # Define dataset type for instruct evaluation
    dataset_type = {
        "field_instruction": args.field_instruction,
        "field_output": args.field_output
    }
    
    print(f"\nConfiguration:")
    print(f"  Model to evaluate: {MODEL_ID}")
    print(f"  Dataset (host): {dataset_path}")
    print(f"  Dataset (container): /data/{dataset_path.name}")
    print(f"  Dataset Type: {dataset_type}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    print(f"  Container Image: {args.container_image}")
    
    # Build Docker command
    docker_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        # Mount the dataset directory to the expected location
        "-v", f"{dataset_path.parent}:/workspace/input_data:ro",
        # Mount output directory
        "-v", f"{OUTPUT_DIR}:/output:rw",
        # Environment variables
        "-e", f"DATASET=/workspace/input_data/{dataset_path.name}",
        "-e", f"ORIGINAL_MODEL={MODEL_ID}",
        "-e", f"DATASET_TYPE={json.dumps(dataset_type)}",
        "-e", f"FILE_FORMAT=json",
        "-e", f"MODELS={MODEL_ID}",  # Testing the base model itself
        # Container and command
        args.container_image,
        "python", "-m", "validator.evaluation.eval_instruct_text"
    ]
    
    print("\n" + "="*60)
    print("Docker Command:")
    print("="*60)
    print(" \\\n  ".join(docker_cmd))
    
    if not args.dry_run:
        print("\n" + "="*60)
        print("Running Evaluation in Container")
        print("="*60)
        
        try:
            # Check if docker is available
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            
            # Run the container
            result = subprocess.run(docker_cmd, capture_output=False, text=True, check=False)
            
            if result.returncode == 0:
                print("\n✓ Container evaluation completed successfully")
            else:
                print(f"\n⚠ Container exited with code {result.returncode}")
            
            # Check for results
            results_file = OUTPUT_DIR / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    print("\n" + "="*60)
                    print("Evaluation Results:")
                    print("="*60)
                    print(json.dumps(results, indent=2))
                    
                    if MODEL_ID in results:
                        print(f"\nSummary:")
                        print(f"  Model: {MODEL_ID}")
                        print(f"  Eval Loss: {results[MODEL_ID].get('eval_loss', 'N/A')}")
                        print(f"  Is Finetune: {results[MODEL_ID].get('is_finetune', 'N/A')}")
            else:
                print(f"\nNote: Results file not found at {results_file}")
                print("Check container output above for details.")
                
        except subprocess.CalledProcessError:
            print("\nError: Docker is not installed or not available")
            sys.exit(1)
        except Exception as e:
            print(f"\nError running container: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[DRY RUN] - Docker command printed above. Use without --dry-run to execute.")
    
    print("\n" + "="*60)
    print("Script Complete")
    print("="*60)


if __name__ == "__main__":
    main()