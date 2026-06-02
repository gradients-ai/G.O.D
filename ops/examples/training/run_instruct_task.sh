#!/bin/bash

TASK_ID="0ace46bc-8f88-4e70-95b9-9502b5a4d1dc"
MODEL="Qwen/Qwen3-0.6B"
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/68742916c5fba7e2_test_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250718%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250718T123411Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=7cf19bb88267a6922afae0c845964412c45b63abe6b79983dd1ed10494453190"
DATASET_TYPE='{
  "field_instruction":"instruct",
  "field_output":"output"
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=6


# For uploading the outputs
HUGGINGFACE_TOKEN="Your Huggingface Token"
WANDB_TOKEN="Your WandB Token"
HUGGINGFACE_USERNAME="Your Huggingface Username"
EXPECTED_REPO_NAME="instructtest"
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
  --task-type "InstructTextTask"


docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --network none \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name instruct-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "InstructTextTask" \
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

