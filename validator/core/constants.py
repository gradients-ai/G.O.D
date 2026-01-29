import os
from datetime import date

from core.constants import GRPO_DEFAULT_FIELD_PROMPT
from core.constants import NETUID


RAYONLABS_HF_USERNAME = "gradients-io-tournaments"  # "besimray"  # "rayonlabs"

SUCCESS = "success"
ACCOUNT_ID = "account_id"
STAKE = "stake"
COLDKEY = "coldkey"


BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
DELETE_S3_AFTER_COMPLETE = True

VALI_CONFIG_PATH = "validator/test_axolotl.yml"

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


# data stuff
TRAIN_TEST_SPLIT_PERCENTAGE = 0.1
MAX_TEST_DATA_POINTS = 400

IMAGE_TRAIN_SPLIT_ZIP_NAME = "train_data.zip"
IMAGE_TEST_SPLIT_ZIP_NAME = "test_data.zip"
TEMP_PATH_FOR_IMAGES = "/tmp/validator/temp_images"
SUPPORTED_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_FILE_SIZE_BYTES = 2_147_483_646  # pyarrow max json load size
MINIMUM_DATASET_ROWS = 2_000  # Minimum number of rows required in a dataset
EXAMPLE_PROMPTS_PATH = "validator/tasks/example_prompts.json"

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

_gpu_ids = os.getenv("GPU_IDS", "").strip()
GPU_IDS = [int(id) for id in _gpu_ids.split(",")] if _gpu_ids else [0]

# we sample datasets with these num_rows ranges equally
DATASET_BINS_TO_SAMPLE = [
    (20_000, 50_000),
    (50_000, 100_000),
    (100_000, 500_000),
]

# dataset row bins to training hours range
INSTRUCT_TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE = {
    (1_000, 10_000): (1, 3),
    (10_000, 25_000): (2, 4),
    (25_000, 50_000): (3, 5),
    (50_000, 100_000): (3, 6),
    (100_000, 500_000): (4, 6),
}

# text augmentation synth
TEXT_SYNTH_MODEL = "Qwen/Qwen3-32B"
TEXT_SYNTH_MODEL_TEMPERATURE = 0.6
TEXT_SYNTH_MODEL_MAX_TOKENS = 5024
END_OF_REASONING_TAG = "</think>"

# image prompt generation synth
IMAGE_PROMPT_GEN_MODEL = "Qwen/Qwen3-32B"
IMAGE_PROMPT_GEN_MODEL_TEMPERATURE = 0.4
IMAGE_PROMPT_GEN_MODEL_MAX_TOKENS = 5024
IMAGE_STYLE_PICKING_NUM_TRIES = 10
PERSON_GEN_RETRIES = 3

# endpoints
PROMPT_GEN_ENDPOINT = "https://llm.chutes.ai/v1/chat/completions"
IMAGE_GEN_ENDPOINT = "https://image.chutes.ai/generate"
PROMPT_PATH = "validator/prompts.yml"
NINETEEN_API_KEY = os.getenv("NINETEEN_API_KEY")
EMISSION_BURN_HOTKEY = "5GU4Xkd3dCGTU3s8VLcHGc5wsD5M8XyxDca5yDQhYm1mVXFu"

# Boss Round Historical Task Selection
BOSS_ROUND_HISTORICAL_START_DATE = date(2025, 6, 1)
BOSS_ROUND_HISTORICAL_END_DATE = date(2025, 8, 1)

MIN_SUCCESSFUL_SCORES_FOR_HISTORICAL_TASK = 2

# Tournament Start Requirements
MIN_MINERS_FOR_ENV_TOURN = 4
MIN_MINERS_FOR_TOURN = 8


TOURNAMENT_PARTICIPATION_WEIGHT = 0.0001  # Weight given to active participants

# Tournament weight distribution
TOURNAMENT_SIMPLE_DECAY_BASE = 0.3  # Base for simple exponential decay (1st=1.0, 2nd=0.2, 3rd=0.04, etc.)


# General miner pool sizes
MIN_IDEAL_NUM_MINERS_IN_POOL = 8

MIN_IMAGE_COMPETITION_HOURS = 1
MAX_IMAGE_COMPETITION_HOURS = 2
TASK_TIME_DELAY = 15  # number of minutes we wait to retry an organic request
# how many times in total do we attempt to delay an organic request looking for miners
MAX_DELAY_TIMES = 6
# Maximum number of evaluation attempts when all scores are zero (including the first one)
MAX_EVAL_ATTEMPTS = 4
MODEL_SIZE_REQUIRING_2_GPUS = 30 * 10**9  # 30B params

# Tournament GPU requirement thresholds (in billions of parameters)
TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100 = 4.0
TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100 = 12.0
TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100 = 40.0

# Tournament task type GPU multipliers
TOURNAMENT_DPO_GPU_MULTIPLIER = 3
TOURNAMENT_GRPO_GPU_MULTIPLIER = 2
MODEL_SIZE_REQUIRING_3_GPUS = 70 * 10**9
MODEL_SIZE_REQUIRING_4_GPUS = 100 * 10**9

# scoring stuff  - NOTE: Will want to slowly make more exponential now we have auditing
SCORE_PENALTY = -1
FIRST_PLACE_SCORE = 3

# processing stuff
MAX_CONCURRENT_MINER_ASSIGNMENTS = 5
MAX_CONCURRENT_TASK_PREPS = 3

PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT = 0.4
PERCENTAGE_OF_INSTRUCT_TASKS_THAT_SHOULD_BE_CHAT = 0.5
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_IMAGE = 0.2
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO = 0.15
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO = (
    1
    - PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
    - PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_IMAGE
    - PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
)
PERCENTAGE_OF_IMAGE_SYNTHS_SHOULD_BE_STYLE = (
    0.5  # person synth chance is 1 minus this (only for sdxl models, flux is always person)
)
PROBABILITY_STYLE_COMBINATION = 0.5
PERSON_SYNTH_DS_PREFIX = "person"
IMAGE_SYNTH_DOCKER_IMAGE = "diagonalge/image_synth:latest"
SYNTH_CONTAINER_SAVE_PATH = "/app/images/"

# grpo synth
MIN_NUM_REWARD_FUNCTIONS = 1
MAX_NUM_REWARD_FUNCTIONS = 5
PERCENTAGE_REWARD_FUNCTIONS_GENERIC_FROM_LLM = 0.0
PERCENTAGE_REWARD_FUNCTIONS_GENERIC_FROM_DB = 1 - PERCENTAGE_REWARD_FUNCTIONS_GENERIC_FROM_LLM

# affine grpo synth
GET_AFFINE_GRPO_DATA_ENDPOINT = f"{PROD_CONTENT_BASE_URL}/affine-grpo-data/latest"  # Force prod for affine data
AFFINE_REWARD_FN_IDS = [
    "2226678e-df0d-42d0-8adb-551aec0ed88e",  # sat_reward_function
    "dadf301b-14cc-4bb2-9bb8-7d658d29661c",  # abd_reward_function
    "b5008828-8628-4ef5-b3f2-f77580028b67",  # ded_reward_function
]

# diffusion eval stuff
LORA_SDXL_WORKFLOW_PATH = "validator/evaluation/comfy_workflows/lora_sdxl.json"
LORA_SDXL_WORKFLOW_PATH_DIFFUSERS = "validator/evaluation/comfy_workflows/lora_sdxl_diffusers.json"
LORA_FLUX_WORKFLOW_PATH = "validator/evaluation/comfy_workflows/lora_flux.json"
LORA_ZIMAGE_WORKFLOW_PATH = "validator/evaluation/comfy_workflows/lora_z-image.json"
LORA_QWEN_IMAGE_WORKFLOW_PATH = "validator/evaluation/comfy_workflows/lora_qwen-image.json"
CHECKPOINTS_SAVE_PATH = "validator/evaluation/ComfyUI/models/checkpoints"
UNET_SAVE_PATH = "validator/evaluation/ComfyUI/models/unet"
DIFFUSERS_PATH = "validator/evaluation/ComfyUI/models/diffusers"
DIFFUSION_MODELS_PATH = "validator/evaluation/ComfyUI/models/diffusion_models"
LORAS_SAVE_PATH = "validator/evaluation/ComfyUI/models/loras"
DIFFUSION_HF_DEFAULT_FOLDER = "checkpoint"
DIFFUSION_HF_DEFAULT_CKPT_NAME = "last.safetensors"
DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT = 0.25
EVAL_DEFAULTS = {
    "sdxl": {"steps": 20, "cfg": 8, "denoise": 0.9},
    "flux": {"steps": 35, "cfg": 100, "denoise": 0.75},
    "z-image": {"steps": 10, "cfg": 1, "denoise": 0.90},
    "qwen-image": {"steps": 20, "cfg": 8, "denoise": 0.93},
}

# Max jobs
MAX_CONCURRENT_JOBS = 60

# Image generation parameters
IMAGE_GEN_MODEL = "FLUX.1-schnell"
IMAGE_GEN_STEPS = 8
IMAGE_GEN_CFG_SCALE = 3

MIN_IMAGE_SYNTH_PAIRS = 10
MAX_IMAGE_SYNTH_PAIRS = 50

MIN_IMAGE_WIDTH = 1024
MAX_IMAGE_WIDTH = 1024
MIN_IMAGE_HEIGHT = 1024
MAX_IMAGE_HEIGHT = 1024
IMAGE_RESOLUTION_STEP = 64  # Ensures we get resolutions divisible by 64

# scoring stuff
MAX_TEXT_TOURNAMENT_WEIGHT = 0.55
MAX_IMAGE_TOURNAMENT_WEIGHT = 0.35
MAX_ENVIRONMENT_TOURNAMENT_WEIGHT = 0.10
TOURNAMENT_TEXT_WEIGHT = 0.18
TOURNAMENT_IMAGE_WEIGHT = 0.12
TOURNAMENT_ENVIRONMENT_WEIGHT = 0.05
TOURNAMENT_INTERVAL_HOURS = 72
TOURNAMENT_TEXT_WEIGHT = 0.20
TOURNAMENT_IMAGE_WEIGHT = 0.15
MAX_TEXT_TOURNAMENT_WEIGHT = 0.6
MAX_IMAGE_TOURNAMENT_WEIGHT = 0.4

# Tournament scheduling settings
# Tournaments start every week on these days and hours (UTC)
# Environment tournaments: Monday at 14:00 UTC
TOURNAMENT_SCHEDULE_ENVIRONMENT_DAY_OF_WEEK = 0  # 0=Monday
TOURNAMENT_SCHEDULE_ENVIRONMENT_HOUR = 14  # 0-23 (UTC time)

# Text and Image tournaments: Thursday at 14:00 UTC
TOURNAMENT_SCHEDULE_TEXT_DAY_OF_WEEK = 3  # 3=Thursday
TOURNAMENT_SCHEDULE_TEXT_HOUR = 14  # 0-23 (UTC time)
TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK = 3  # 3=Thursday
TOURNAMENT_SCHEDULE_IMAGE_HOUR = 14  # 0-23 (UTC time)

TOURNAMENT_INTERVAL_HOURS = (
    120  # Display value for frontend (5 days), not used for actual scheduling. TODO: remove once frontend is updated
)

BURN_REDUCTION_RATE = 5.0
MAX_BURN_REDUCTION = 0.8
EMISSION_MULTIPLIER_THRESHOLD = 0.05
EMISSION_MULTIPLIER_RATE = 2.0
EMISSION_BOOST_DECAY_PER_WIN = 0.01  # Deprecated - kept for backwards compatibility
# Time-based decay settings (replaces consecutive wins decay)
EMISSION_DAILY_TIME_DECAY_RATE = 0.0033  # 0.33%/day
EMISSION_TIME_DECAY_START_DATE = date(2025, 11, 26)
SECONDS_PER_DAY = 86400.0

ALPHA_PER_SECOND = 1.0 / 12.0
MINER_ALPHA_SHARE = 0.41
DAILY_ALPHA_TO_MINERS = ALPHA_PER_SECOND * SECONDS_PER_DAY * MINER_ALPHA_SHARE

# HF models cache management
CACHE_TAU_DAYS = 10  # Time constant (τ) for exponential decay in days
CACHE_MAX_LOOKUP_DAYS = 30  # Maximum number of days to look back for usage data
MAX_CACHE_SIZE_BYTES = 500 * 1024**3 if NETUID == 241 else 1000 * 1024**3  # in bytes
CACHE_CLEANUP_INTERVAL = 8 * 60 * 60  # in seconds

# Docker evaluation
DOCKER_EVAL_HF_CACHE_DIR = "/root/.cache/huggingface"

# DPO evaluation
TRL_DPO_FIELD_PROMPT = "prompt"
TRL_DPO_FIELD_CHOSEN = "chosen"
TRL_DPO_FIELD_REJECTED = "rejected"

# Tournament analytics cache constants
LATEST_TOURNAMENTS_CACHE_TTL = 3600
LATEST_TOURNAMENTS_CACHE_KEY = "latest_tournaments_details"

# GRPO evaluation
TRL_GRPO_FIELD_PROMPT = GRPO_DEFAULT_FIELD_PROMPT

# Default, fixed Hyperparameters
BETA_DPO = 0.1
BETA_GRPO = 0.5

# GRPO evaluation
GRPO_INITIAL_BATCH_SIZE = 16
GRPO_KL_BATCH_SIZE = 1
GRPO_DEFAULT_NUM_GENERATIONS = 2
GRPO_KL_SEQUENCE_LENGTH = 512

STANDARD_INSTRUCT_COLUMN = "instruct"
STANDARD_INPUT_COLUMN = "input"
STANDARD_OUTPUT_COLUMN = "output"
STANDARD_SYSTEM_COLUMN = "system"
STANDARD_GRPO_PROMPT_COLUMN = "prompt"
STANDARD_GRPO_EXTRA_COLUMN = "extra_data"
STANDARD_DPO_PROMPT_COLUMN = "prompt"
STANDARD_DPO_CHOSEN_COLUMN = "chosen"
STANDARD_DPO_REJECTED_COLUMN = "rejected"
STANDARD_CHAT_MESSAGES_COLUMN = "conversations"

# Trainer endpoints

PROXY_TRAINING_IMAGE_ENDPOINT = "/v1/trainer/start_training"
GET_GPU_AVAILABILITY_ENDPOINT = "/v1/trainer/get_gpu_availability"
TASK_DETAILS_ENDPOINT = "/v1/trainer/{task_id}"
GET_RECENT_TASKS_ENDPOINT = "/v1/trainer/get_recent_tasks"

# Dstack API endpoints
DSTACK_RUNS_APPLY_ENDPOINT = "/api/project/{project}/runs/apply"
DSTACK_RUNS_GET_ENDPOINT = "/api/project/{project}/runs/get"

# Tournament constants
DEFAULT_PARTICIPANT_REPO = "https://github.com/rayonlabs/G.O.D"
DEFAULT_PARTICIPANT_COMMIT = "8631451156e2915070f77e5547ca0d5ed3d0eb8a"

# YaRN extension constants
YARN_EXTENSION_PROBABILITY = 0.0  # Probability of applying YaRN extension to tournament tasks
YARN_TOURNAMENT_FACTORS = [2, 4]
MODEL_COPY_ENDPOINT = "https://huggingface.co/api/models/{source_repo}/duplicate"

# Basilica deployment constants
BASILICA_SGLANG_IMAGE = "lmsysorg/sglang:latest"
BASILICA_ENV_IMAGE = "affinefoundation/agentgym:alfworld"
BASILICA_SGLANG_GPU_COUNT = 1
BASILICA_SGLANG_GPU_MODELS = ["A100"]
BASILICA_SGLANG_MIN_GPU_MEMORY_GB = 80
BASILICA_SGLANG_MEMORY = "16Gi"
BASILICA_SGLANG_TTL_SECONDS = 10800
BASILICA_SGLANG_TIMEOUT = 1800
BASILICA_ENV_CPU = "4"
BASILICA_ENV_MEMORY = "16Gi"
BASILICA_ENV_TTL_SECONDS = 10800
BASILICA_ENV_TIMEOUT = 600

ENVIRONMENTS = {
    "alfworld": {
        "task_id_range": (1, 2500),
        "env_image": "affinefoundation/agentgym:alfworld"
    },
    "goofspiel": {
        "task_id_range": (0, 99999999),
        "env_image": "diagonalge/openspiel:latest"
    }
}

DEFAULT_ENV = "goofspiel"
NUM_EVAL_SAMPLES = 250
