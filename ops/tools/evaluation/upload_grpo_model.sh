#!/bin/bash

# Upload GRPO model to HuggingFace Hub
# This script uploads a trained GRPO model from the specified directory to HuggingFace Hub
# Required environment variables:
# - HUGGINGFACE_TOKEN: HuggingFace API token
# - HUB_MODEL_ID: Target repository ID (username/repo-name)
# - GRPO_MINER_OUTPUT_DIR: Path to the model directory to upload

huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential && \
python -c "
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ['HUGGINGFACE_TOKEN'])
api.create_repo(os.environ['HUB_MODEL_ID'], exist_ok=True, private=False)
api.upload_folder(folder_path=os.environ['GRPO_MINER_OUTPUT_DIR'], repo_id=os.environ['HUB_MODEL_ID'])
print('Upload completed successfully')
"