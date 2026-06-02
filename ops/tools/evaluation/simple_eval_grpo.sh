#!/bin/bash
# Simple GRPO evaluation script
# Usage: ./ops/tools/evaluation/simple_eval_grpo.sh

TEMP_DIR=$(mktemp -d)
DATASET_FILE="$TEMP_DIR/dataset.json"

# Download the dataset
echo "Downloading dataset..."
curl -L -o "$DATASET_FILE" "https://gradients.s3.eu-north-1.amazonaws.com/13abfd7d95984f0d_test_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250510%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250510T034142Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=9bcc25185db0e3d411f46b5063134bb66154734c21401cef6ec5427380ef4841"

echo "Dataset downloaded to $DATASET_FILE"
echo "Dataset size: $(wc -c < "$DATASET_FILE") bytes"

# Display dataset info
echo "Dataset size: $(wc -c < "$DATASET_FILE") bytes"

# Print first few lines of dataset
echo "Dataset preview:"
head -n 5 "$DATASET_FILE"

# Create a directory for results
RESULTS_DIR="$PWD/grpo_results"
mkdir -p "$RESULTS_DIR"

# Run Docker container with added mount for /aplp directory
echo "Starting GRPO evaluation..."
docker run --rm \
  -e DATASET="/workspace/input_data/dataset.json" \
  -e MODELS="robiual-awal/c621c6f1-40be-4a54-add1-38585b4e002f,Alphatao/3e05bf5e-0a8a-4c96-bf01-7a2d82bd333c" \
  -e ORIGINAL_MODEL="EleutherAI/pythia-70m" \
  -e DATASET_TYPE='{"field_prompt":"prompt","reward_functions":[{"reward_func":"def reward_func(completions, **kwargs):\n    # Count frequency of letter \"e\" in response\n    return [text.count(\"e\") / (len(text) + 1) for text in completions]","reward_weight":0.7,"name":"e_counter"},{"reward_func":"def reward_func(completions, **kwargs):\n    # Reward responses that are long but not too long\n    return [min(len(text)/100, 1.0) for text in completions]","reward_weight":0.3,"name":"length_scorer"}]}' \
  -e FILE_FORMAT="json" \
  -v "$TEMP_DIR:/workspace/input_data:rw" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:rw" \
  -v "$RESULTS_DIR:/aplp:rw" \
  --runtime nvidia \
  -e CUDA_VISIBLE_DEVICES=0 \
  --gpus '"device=0"' \
  weightswandering/tuning_vali:latest \
  python -m validator.evaluation.evaluators.grpo

# Check for results in both temp directory and our mounted /aplp directory
echo "Checking for result files..."
RESULTS_FILE=$(find "$TEMP_DIR" -type f -name "*evaluation_results.json" | head -n 1)

if [ -z "$RESULTS_FILE" ]; then
  # If not found in temp dir, check in the results dir
  RESULTS_FILE="$RESULTS_DIR/evaluation_results.json"
  if [ -f "$RESULTS_FILE" ]; then
    echo "Evaluation results found in the mounted /aplp directory: $RESULTS_FILE"
  else
    echo "No evaluation results found in either location."
    echo "Listing all files in both directories to debug:"
    echo "TEMP_DIR contents:"
    find "$TEMP_DIR" -type f | sort
    echo "RESULTS_DIR contents:"
    find "$RESULTS_DIR" -type f | sort
    exit 1
  fi
else
  echo "Evaluation results found in temp directory: $RESULTS_FILE"
fi

echo "Results content:"
cat "$RESULTS_FILE"

# Copy results to current directory with timestamp
cp "$RESULTS_FILE" "./grpo_eval_results_$(date +%s).json"
echo "Results copied to current directory"

echo "Evaluation complete"

# Run the scoring simulation
LATEST_RESULTS=$(ls -t ./grpo_eval_results_*.json 2>/dev/null | head -1)
if [ -n "$LATEST_RESULTS" ]; then
  RESULTS_FILE="$LATEST_RESULTS"
  echo ""
  echo "=== VALIDATOR SCORING SIMULATION ==="
  echo "Running scoring analysis on: $RESULTS_FILE"

  # Simple Python script to analyze results
  python3 - "$RESULTS_FILE" << 'ENDPYTHON'
import json
import sys

# Constants from validator/scoring/constants.py
FIRST_PLACE_SCORE = 3.0  # Top model gets this score
SCORE_PENALTY = -1.0     # Bottom 25% get this (when > 8 models)

# Load results
with open(sys.argv[1], 'r') as f:
    results = json.load(f)

# Get models and their scores
models_scores = []
for model_name in results:
    if isinstance(results[model_name], dict) and "eval_loss" in results[model_name]:
        model_data = results[model_name]
        if model_data.get("is_finetune", False):
            # For GRPO, we use the eval_loss field which contains the aggregated score
            grpo_score = model_data["eval_loss"]
            models_scores.append((model_name, grpo_score))

# Sort by GRPO score (higher is better)
models_scores.sort(key=lambda x: -x[1])

print(f"\nFound {len(models_scores)} valid models")

# Assign scores based on ranking
scored_results = {}
for rank, (model, score) in enumerate(models_scores, 1):
    if rank == 1:
        validator_score = FIRST_PLACE_SCORE
        reason = "Ranked 1st (GRPO score)"
    else:
        validator_score = 0.0
        reason = "Ranked below top"

    # Store scores
    scored_results[model] = {
        "validator_score": validator_score,
        "reason": reason,
        "grpo_score": score,
        "rank": rank
    }

    # Print ranking info
    print(f"\nRank {rank}: {model}")
    print(f"  GRPO Score: {score:.4f}")
    print(f"  Validator Score: {validator_score}")

# Print reward breakdown (just for the top model)
if models_scores:
    top_model = models_scores[0][0]
    model_info = results[top_model]
    print(f"\nScoring Details (Top Model: {top_model})")

    # Raw rewards
    if "raw_rewards" in model_info:
        print("  Raw Rewards (not directly used in final scoring):")
        for func_name, value in model_info["raw_rewards"].items():
            weight = model_info.get("reward_weights", {}).get(func_name, "?")
            print(f"    {func_name}: {value:.4f} (weight: {weight})")

    # Aggregate reward
    if "individual_rewards" in model_info and "wrapper" in model_info["individual_rewards"]:
        reward = model_info["individual_rewards"]["wrapper"]
        print(f"  Combined Reward: {reward:.4f}")

    print(f"  GRPO Score (used for ranking): {model_info['eval_loss']:.4f}")
    print(f"  Validator Score: {FIRST_PLACE_SCORE} (top model only)")
ENDPYTHON

else
  echo "No results file found to analyze."
fi

# Cleanup temporary directories but keep our results files
rm -rf "$TEMP_DIR"
# We don't remove $RESULTS_DIR since it contains useful output files
echo "Temporary files cleaned up. Results directory at: $RESULTS_DIR"
