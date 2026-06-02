#!/bin/bash

TASK_ID="8832a4bc-5e99-4d70-a5c9-2605c3b4e1f2"
MODEL="Qwen/Qwen3-0.6B"
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/ff48d393207a45b9_test_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250730%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250730T195033Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=5318790ec1960c516985bca92847bd5a2871f680034153a3d831fc7d0950a8d8"
DATASET_TYPE='{
  "field_prompt":"prompt",
  "reward_functions":[
    {
      "reward_func":"def reward_func(completions, **kwargs):\n    # Count frequency of letter \"e\" in response\n    return [text.count(\"e\") / (len(text) + 1) for text in completions]",
      "reward_weight":0.7,
      "name":"e_counter"
    },
    {
      "reward_func":"def reward_func(completions, **kwargs):\n    # Reward responses that are long but not too long\n    return [min(len(text)/100, 1.0) for text in completions]",
      "reward_weight":0.3,
      "name":"length_scorer"
    }
  ]
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=12


# For uploading the outputs
HUGGINGFACE_TOKEN="Your Huggingface Token"
WANDB_TOKEN="Your WandB Token"
HUGGINGFACE_USERNAME="Your Huggingface Username"
EXPECTED_REPO_NAME="grpotest"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"


CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 700 "$OUTPUTS_DIR"

# Build the downloader image
docker build --no-cache -t trainer-downloader -f ops/docker/trainer-downloader.dockerfile .

# Build the trainer image
docker build --no-cache -t standalone-text-trainer -f ops/docker/standalone-text-trainer.dockerfile .

# Build the hf uploader image
docker build --no-cache -t hf-uploader -f ops/docker/hf-uploader.dockerfile .

# Download model and dataset
echo "Downloading model and dataset..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "GrpoTask"


docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --network none \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name grpo-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "GrpoTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \


  # Upload the trained model to HuggingFace
echo "Uploading model to HuggingFace..."
docker run --rm --gpus all \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  --env HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME" \
  --env WANDB_TOKEN="$WANDB_TOKEN" \
  --env TASK_ID="$TASK_ID" \
  --env EXPECTED_REPO_NAME="$EXPECTED_REPO_NAME" \
  --env LOCAL_FOLDER="$LOCAL_FOLDER" \
  --name hf-uploader \
  hf-uploader
