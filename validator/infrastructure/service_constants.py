import os

from core.constants.network import NETUID


RAYONLABS_HF_USERNAME = "gradients-io-tournaments"  # "besimray"  # "rayonlabs"

SUCCESS = "success"
ACCOUNT_ID = "account_id"
STAKE = "stake"
COLDKEY = "coldkey"


BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
DELETE_S3_AFTER_COMPLETE = True

VALI_CONFIG_PATH = "validator/assets/test_axolotl.yml"

# db stuff
NULL_ACCOUNT_ID = "00000000-0000-0000-0000-000000000000"


# api stuff should move this out to be shared by both miner and vali code?
START_TRAINING_ENDPOINT = "/start_training/"
START_TRAINING_IMAGE_ENDPOINT = "/start_training_image/"
START_TRAINING_GRPO_ENDPOINT = "/start_training_grpo/"
TRAINING_REPO_ENDPOINT = "/training_repo"

DEV_CONTENT_BASE_URL = "https://dev.content.gradients.io"
PROD_CONTENT_BASE_URL = "https://content.gradients.io"


# 241 is testnet
CONTENT_BASE_URL = DEV_CONTENT_BASE_URL if NETUID == 241 else PROD_CONTENT_BASE_URL

GET_RANDOM_DATASETS_ENDPOINT = f"{CONTENT_BASE_URL}/datasets/random"
GET_RANDOM_MODELS_ENDPOINT = f"{CONTENT_BASE_URL}/models/random"
GET_COLUMNS_FOR_DATASET_ENDPOINT = f"{CONTENT_BASE_URL}/dataset/{{dataset}}/columns/suggest"
GET_IMAGE_MODELS_ENDPOINT = f"{CONTENT_BASE_URL}/images/models"

GET_ALL_MODELS_ID = "model_id"

PROXY_TRAINING_IMAGE_ENDPOINT = "/v1/trainer/start_training"
MODEL_PREP_ENDPOINT = "/v1/trainer/model_prep"
MODEL_PREP_STATUS_ENDPOINT = "/v1/trainer/model_prep/{task_id}"
GET_GPU_AVAILABILITY_ENDPOINT = "/v1/trainer/get_gpu_availability"
TASK_DETAILS_ENDPOINT = "/v1/trainer/{task_id}"
GET_RECENT_TASKS_ENDPOINT = "/v1/trainer/get_recent_tasks"

# Dstack API endpoints
DSTACK_RUNS_APPLY_ENDPOINT = "/api/project/{project}/runs/apply"
DSTACK_RUNS_GET_ENDPOINT = "/api/project/{project}/runs/get"
