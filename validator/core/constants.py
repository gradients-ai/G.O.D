import os
from datetime import date

from core.constants import GRPO_DEFAULT_FIELD_PROMPT
from core.constants import NETUID
from core.constants import EnvironmentName
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.utility_models import TaskType


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
MAX_TEST_DATA_POINTS = 1000

IMAGE_TRAIN_SPLIT_ZIP_NAME = "train_data.zip"
IMAGE_TEST_SPLIT_ZIP_NAME = "test_data.zip"
TEMP_PATH_FOR_IMAGES = "/tmp/validator/temp_images"
SUPPORTED_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_FILE_SIZE_BYTES = 2_147_483_646  # pyarrow max json load size
MINIMUM_DATASET_ROWS = 4_000  # Minimum number of rows required in a dataset
EXAMPLE_PROMPTS_PATH = "validator/tasks/example_prompts.json"

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

# we sample datasets with these num_rows ranges equally
DATASET_BINS_TO_SAMPLE = [
    (20_000, 50_000),
    (50_000, 100_000),
    (100_000, 500_000),
]

# Training hours
TRAINING_HOURS_SCALE_START_ROWS = 75_000
TRAINING_HOURS_MAX_ROWS = 500_000
TRAINING_HOURS_MIN = 1.0
TRAINING_HOURS_MAX_BASE = 6.0
FULL_HOURS_MODEL_PARAMS = 14e9
MIN_HOURS_SCALE = 0.5
TASK_TYPE_HOURS_MULTIPLIER: dict[TaskType, float] = {
    TaskType.INSTRUCTTEXTTASK: 1.0,
    TaskType.CHATTASK: 1.0,
    TaskType.DPOTASK: 1.4,
    TaskType.GRPOTASK: 1.3,
}
CTX_REF_SEQ_LEN = 512
CTX_SCALE_SPAN = 1024
CTX_SCALE_MIN = 0.25
CTX_SCALE_MAX = 3.0
MAX_TRAINING_HOURS = 6.0

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
IMAGE_SYNTH_FACE_IMAGE_URL = "https://thispersondoesnotexist.com/"
FAL_KEY = os.getenv("FAL_KEY")
FAL_TIMEOUT_SECONDS = 300
FAL_IMAGE_GENERATION_CONCURRENCY = 20
FAL_PERSON_PROMPT_MODEL = "openrouter/router/vision"
FAL_PERSON_PROMPT_VLM = "google/gemini-2.5-flash"
FAL_TEXT_PROMPT_MODEL = "openrouter/router"
FAL_TEXT_PROMPT_LLM = "google/gemini-2.5-flash"
FAL_AVATAR_MODEL = "fal-ai/nano-banana-2/edit"
FAL_STYLE_MODEL_NANO_BANANA_2 = "fal-ai/nano-banana-2"
FAL_STYLE_MODEL_GPT_IMAGE_2 = "openai/gpt-image-2"
FAL_IMAGE_MODELS = (FAL_STYLE_MODEL_NANO_BANANA_2, FAL_STYLE_MODEL_GPT_IMAGE_2)
FAL_GPT_IMAGE_2_QUALITY = "medium"
FAL_NANO_BANANA_RESOLUTION = "1K"
FAL_IMAGE_OUTPUT_FORMAT = "png"

# endpoints
PROMPT_GEN_ENDPOINT = "https://llm.chutes.ai/v1/chat/completions"
IMAGE_GEN_ENDPOINT = "https://image.chutes.ai/generate"
NINETEEN_API_KEY = os.getenv("NINETEEN_API_KEY")
EMISSION_BURN_HOTKEY = "5GU4Xkd3dCGTU3s8VLcHGc5wsD5M8XyxDca5yDQhYm1mVXFu"

# Boss Round Historical Task Selection
BOSS_ROUND_HISTORICAL_START_DATE = date(2025, 6, 1)
BOSS_ROUND_HISTORICAL_END_DATE = date(2025, 8, 1)

MIN_SUCCESSFUL_SCORES_FOR_HISTORICAL_TASK = 2

# Tournament Start Requirements
MIN_MINERS_FOR_ENV_TOURN = 5
MIN_MINERS_FOR_TOURN = 4  # within the small-tournament band (3..9): round 1 is a single group, top 2 advance to a knockout


TOURNAMENT_PARTICIPATION_WEIGHT = 0.0001  # Weight given to active participants

# Tournament weight distribution
TOURNAMENT_SIMPLE_DECAY_BASE = 0.3  # Base for simple exponential decay (1st=1.0, 2nd=0.2, 3rd=0.04, etc.)


# General miner pool sizes
MIN_IDEAL_NUM_MINERS_IN_POOL = 8

MIN_IMAGE_COMPETITION_HOURS = 0.5
MAX_IMAGE_COMPETITION_HOURS = 1.0
QWEN_IMAGE_EXTRA_COMPETITION_HOURS = 0.5
TASK_TIME_DELAY = 15  # number of minutes we wait to retry an organic request
# how many times in total do we attempt to delay an organic request looking for miners
MAX_DELAY_TIMES = 6
# Maximum number of evaluation attempts when all scores are zero (including the first one)
MAX_EVAL_ATTEMPTS = 4
# Maximum dispatch attempts for PvP pair evals and individual env containers
MAX_TOURNAMENT_EVAL_ATTEMPTS = 3
MODEL_SIZE_REQUIRING_2_GPUS = 30 * 10**9  # 30B params
# Tournament GPU requirement thresholds (in billions of parameters)
TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100 = 4.0
TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100 = 12.0
TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100 = 40.0

# Tournament task type GPU multipliers
TOURNAMENT_DPO_GPU_MULTIPLIER = 3
TOURNAMENT_GRPO_GPU_MULTIPLIER = 2
# Instruct KL tasks keep a frozen reference (base) model resident alongside the
# trainable model, so they need extra VRAM headroom — same class of overhead as
# the GRPO reference policy.
TOURNAMENT_KL_GPU_MULTIPLIER = 2
MODEL_SIZE_REQUIRING_3_GPUS = 70 * 10**9
MODEL_SIZE_REQUIRING_4_GPUS = 100 * 10**9

# scoring stuff  - NOTE: Will want to slowly make more exponential now we have auditing
SCORE_PENALTY = -1
FIRST_PLACE_SCORE = 3

# processing stuff
MAX_CONCURRENT_MINER_ASSIGNMENTS = 5
MAX_CONCURRENT_TASK_PREPS = 3
EVAL_MAX_GPUS = 10

PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT = 0.75
PERCENTAGE_OF_INSTRUCT_TASKS_THAT_SHOULD_BE_CHAT = 0.5
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_IMAGE = 0.15
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO = 0.20
# GRPO is the remainder of the text split (image is selected independently)
PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_GRPO = (
    1
    - PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_INSTRUCT_TEXT
    - PERCENTAGE_OF_TASKS_THAT_SHOULD_BE_DPO
)
PERCENTAGE_OF_IMAGE_SYNTHS_SHOULD_BE_STYLE = (
    0.5  # person synth chance is 1 minus this for every image model type
)
PROBABILITY_STYLE_COMBINATION = 0.5
IMAGE_SYNTH_CATEGORY_STYLE = "style"
IMAGE_SYNTH_CATEGORY_PERSON = "person"
IMAGE_SYNTH_CATEGORY_LOGO = "logo"
IMAGE_SYNTH_CATEGORY_SOCIAL = "social"
IMAGE_SYNTH_CATEGORY_DESIGN = "design"
IMAGE_SYNTH_CATEGORY_PRODUCT = "product"
IMAGE_SYNTH_CATEGORY_WEIGHTS = {
    IMAGE_SYNTH_CATEGORY_STYLE: 0.25,
    IMAGE_SYNTH_CATEGORY_PERSON: 0.25,
    IMAGE_SYNTH_CATEGORY_LOGO: 0.15,
    IMAGE_SYNTH_CATEGORY_SOCIAL: 0.15,
    IMAGE_SYNTH_CATEGORY_DESIGN: 0.10,
    IMAGE_SYNTH_CATEGORY_PRODUCT: 0.10,
}
PERSON_SYNTH_DS_PREFIX = "person"
LOGO_SYNTH_DS_PREFIX = "logo"
SOCIAL_SYNTH_DS_PREFIX = "social"
DESIGN_SYNTH_DS_PREFIX = "design"
PRODUCT_SYNTH_DS_PREFIX = "product"

# grpo synth

MIN_NUM_REWARD_FUNCTIONS = 2
MAX_NUM_REWARD_FUNCTIONS = 4

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
MAX_TEXT_TOURNAMENT_WEIGHT = 0.48
MAX_IMAGE_TOURNAMENT_WEIGHT = 0.32
MAX_ENVIRONMENT_TOURNAMENT_WEIGHT = 0.16
TOURNAMENT_TEXT_WEIGHT = 0.15
TOURNAMENT_IMAGE_WEIGHT = 0.125
TOURNAMENT_ENVIRONMENT_WEIGHT = 0.15
TOURNAMENT_INTERVAL_HOURS = 72

# Tournament scheduling settings
# Tournaments start every week on these days and hours (UTC)
# Environment tournaments: Monday at 14:00 UTC
TOURNAMENT_SCHEDULE_ENVIRONMENT_DAY_OF_WEEK = 0  # 0=Monday
TOURNAMENT_SCHEDULE_ENVIRONMENT_HOUR = 14  # 0-23 (UTC time)

# Text tournaments: Thursday at 14:00 UTC
TOURNAMENT_SCHEDULE_TEXT_DAY_OF_WEEK = 3  # 3=Thursday
TOURNAMENT_SCHEDULE_TEXT_HOUR = 14  # 0-23 (UTC time)
# Image tournaments: Thursday at 15:00 UTC
TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK = 3  # 3=Thursday
TOURNAMENT_SCHEDULE_IMAGE_HOUR = 15  # 0-23 (UTC time)

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

# Instruct KL regularisation
# Probability that a tournament instruct task asks miners to train with a KL term.
INSTRUCT_KL_TASK_PROBABILITY = 0.2
# Coefficient (beta) applied to KL(finetuned || base) when weighting the eval loss.
# Sampled uniformly per task in [MIN, MAX], stored per-task (kl_coef) and sent to
# miners so they can match the eval weighting. Range spans a light nudge up to an
# aggressive pull (~half of tasks land in the aggressive upper half).
INSTRUCT_KL_COEFFICIENT_MIN = 0.1
INSTRUCT_KL_COEFFICIENT_MAX = 1.5

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
MODEL_PREP_ENDPOINT = "/v1/trainer/model_prep"
MODEL_PREP_STATUS_ENDPOINT = "/v1/trainer/model_prep/{task_id}"
GET_GPU_AVAILABILITY_ENDPOINT = "/v1/trainer/get_gpu_availability"
TASK_DETAILS_ENDPOINT = "/v1/trainer/{task_id}"
GET_RECENT_TASKS_ENDPOINT = "/v1/trainer/get_recent_tasks"

# Dstack API endpoints
DSTACK_RUNS_APPLY_ENDPOINT = "/api/project/{project}/runs/apply"
DSTACK_RUNS_GET_ENDPOINT = "/api/project/{project}/runs/get"

# Tournament constants
DEFAULT_PARTICIPANT_REPO = "https://github.com/rayonlabs/G.O.D"
DEFAULT_PARTICIPANT_COMMIT = "8631451156e2915070f77e5547ca0d5ed3d0eb8a"

# Claude Agent SDK repo diff reports
CLAUDE_REPO_DIFF_MODEL = "claude-sonnet-4-5"
CLAUDE_REPO_DIFF_MAX_TURNS = 30
CLAUDE_REPO_DIFF_MAX_BUDGET_USD = 2
CLAUDE_REPO_DIFF_MAX_FOCUS_FILES = 180

# Tournament submission de-duplication (anti-spam)
# R1 (pre-training): deterministic exact-commit (T0) + normalized-content (T1) hashing auto-eliminates copies.
# R2: Claude pairwise functional-equivalence judgement (T2), gated behind Discord ping + manual DB approval.
TOURN_DEDUP_ENABLED = True
TOURN_DEDUP_CLAUDE_MODEL = "claude-opus-4-8"  # best current model for the judgement
# T2 runs the agent read-only (Read/Glob/Grep) over both cloned repos so it can inspect full
# contents itself and see through reordering/renaming. These bound a single pairwise judgement.
TOURN_DEDUP_CLAUDE_MAX_TURNS = 60
TOURN_DEDUP_CLAUDE_MAX_BUDGET_USD = 15  # per-pair ceiling; typical run ~$2-3

# YaRN extension constants
YARN_EXTENSION_PROBABILITY = 0.0  # Probability of applying YaRN extension to tournament tasks
YARN_TOURNAMENT_FACTORS = [2, 4]
MODEL_COPY_ENDPOINT = "https://huggingface.co/api/models/{source_repo}/duplicate"

# Model prep constants
BASELINE_STATS_ENABLED_ORGANIC = False  # Run model prep (stats) for organic requests
MODEL_PREP_ENABLED_TEXT = True  # Route text tasks through model prep (augmentation + baseline stats)
MODEL_PREP_ENABLED_IMAGE = False  # Route image tasks through model prep
MODEL_PREP_ENABLED_ENV = True  # Route environment tasks through model prep
MODEL_PREP_ENABLED_BY_TASK_TYPE: dict[TaskType, bool] = {
    TaskType.INSTRUCTTEXTTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.DPOTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.GRPOTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.CHATTASK: MODEL_PREP_ENABLED_TEXT,
    TaskType.IMAGETASK: MODEL_PREP_ENABLED_IMAGE,
    TaskType.ENVIRONMENTTASK: MODEL_PREP_ENABLED_ENV,
}


# Model augmentation constants
AUGMENTATION_ENABLED_TEXT = True  # Enable augmentations for text tasks
AUGMENTATION_ENABLED_IMAGE = False  # Enable augmentations for image tasks
AUGMENTATION_ENABLED_ENV = False  # Enable augmentations for environment tasks
AUGMENTATION_PROBABILITY = 0.5  # Probability that a task gets any augmentation at all

# Weighted distribution over augmentation types (normalised at runtime)
# When an augmentation is applied, one type is chosen according to these weights
AUGMENTATION_TYPE_WEIGHTS: dict[AugmentationType, float] = {
    AugmentationType.GAUSSIAN_NOISE: 0.20,
    AugmentationType.WEIGHT_SCALING: 0.40,
    AugmentationType.MAGNITUDE_PRUNING: 0.25,
    AugmentationType.LAYER_REINIT: 0.15,
}

# Weighted distribution over layer scope (normalised at runtime)
# Determines how many layers the augmentation targets
AUGMENTATION_SCOPE_WEIGHTS: dict[AugmentationScope, float] = {
    AugmentationScope.SINGLE_LAYER: 0.10,
    AugmentationScope.LAYER_TYPE_GROUP: 0.15,
    AugmentationScope.MULTI_LAYER: 0.35,
    AugmentationScope.ALL_LAYERS: 0.40,
}

# Intensity ranges per augmentation type (min, max) — sampled uniformly
AUGMENTATION_INTENSITY_RANGES: dict[AugmentationType, tuple[float, float]] = {
    AugmentationType.GAUSSIAN_NOISE: (0.01, 0.3),
    AugmentationType.WEIGHT_SCALING: (0.5, 1.5),
    AugmentationType.MAGNITUDE_PRUNING: (0.25, 0.50),
    AugmentationType.LAYER_REINIT: (0.05, 0.15),
}

# Environment evaluation constants
ENV_SERVER_CMD_DEFAULT = "python -m uvicorn _affinetes.server:app --host 0.0.0.0 --port 8001 --workers 1 --loop asyncio"
BASILICA_GPU_MODELS = ["A100"]
BASILICA_SGLANG_MIN_GPU_MEMORY_GB = 80

DEFAULT_ENV = EnvironmentName.GIN_RUMMY
ENV_EVAL_DEFAULT_SEED = 42
ENV_EVAL_NUM_SEEDS = 2000
ENV_EVAL_TEMPERATURE = 0.0
ENV_EVAL_MAX_CONCURRENT_REQUESTS = 4
ENV_EVAL_MAX_RETRIES = 3
ENV_EVAL_DEPLOYMENT_RETRY_DELAY = 1200
ENV_EVAL_TASK_RETRY_DELAY = 10
ENV_EVAL_TASK_MAX_RETRIES = 2
ENV_EVAL_TASK_TIMEOUT = 150
ENV_EVAL_SESSION_TIMEOUT = 4 * 60 * 60 # 4 hours

SGLANG_ENV_EVAL_EXTRA_CLI = (
    "--attention-backend triton --prefill-attention-backend triton "
    "--decode-attention-backend triton --sampling-backend pytorch"
)
SGLANG_FLASHINFER_WORKSPACE_MIN_BYTES = 4 * 1024 * 1024 * 1024

EVAL_BASILICA_CPU = "4"
EVAL_BASILICA_MEMORY = "64Gi"
EVAL_BASILICA_TTL_SECONDS = 16000
EVAL_BASILICA_TIMEOUT = 14400
EVAL_BASILICA_MAX_RETRIES = 3
EVAL_BASILICA_RETRY_DELAY_SECONDS = 900
EVAL_BASILICA_POLL_INTERVAL_SECONDS = 300
EVAL_BASILICA_MAX_POLL_SECONDS = 16000
EVAL_DEPLOYMENT_READY_TIMEOUT_SECONDS = 600
EVAL_DB_MAX_CONCURRENT_WRITES = 2
EVAL_DB_RETRY_ATTEMPTS = 4
EVAL_DB_RETRY_BASE_DELAY_SECONDS = 1.0

LOCAL_ENV_DOCKER_NETWORK = "agent_eval_net"
LOCAL_ENV_SGLANG_PORT = 30000
LOCAL_ENV_SERVER_PORT = 8001
LOCAL_ENV_SGLANG_HEALTH_TIMEOUT = 600
LOCAL_ENV_SERVER_HEALTH_TIMEOUT = 300
LOCAL_ENV_HF_CACHE_PATH = "/mnt/hf_cache"

# PvP evaluation constants
PVP_SGLANG_HOST = "127.0.0.1"
PVP_SGLANG_PORT_A = 30000
PVP_SGLANG_PORT_B = 30001
PVP_SGLANG_HEALTH_TIMEOUT = 1800
PVP_SGLANG_HEALTH_PATH = "/v1/models"
PVP_SGLANG_API_PATH = "/v1"
PVP_RESULTS_PATH = "/app/pvp_results.json"
PVP_CONFIG_PATH = "/config/pvp_eval.json"
PVP_CONFIG_ENV_VAR = "PVP_EVAL_CONFIG"
PVP_SEED_RANGE_MAX = 1_000_000
PVP_CONFIG_ID_DIVISOR = 100_000_000
PVP_LOG_INTERVAL_GAMES = 100
PVP_BOT_MAX_PARSING_RETRIES = 0
PVP_BOT_INVALID_ACTION_FORFEIT_THRESHOLD = 3
PVP_TURN_TIMEOUT_SECONDS = 5
PVP_RETRY_BACKOFF_CAP_SECONDS = 32
PVP_EPISODE_FORFEIT_THRESHOLD = 10
INDIVIDUAL_WIN_MARGIN = 0.015

# PvP tournament scoring
PVP_ENV_WIN_POINTS = 3
PVP_ENV_DRAW_POINTS = 1
PVP_ENV_LOSS_POINTS = 0
PVP_NUM_GAMES_PER_ENV = 100
PVP_CONSECUTIVE_LOSS_FORFEIT = 10
PVP_WIN_PCT_THRESHOLD = 0.60
PVP_PERF_DIFF_SLOPE = 0.125  # Linear map: 60% win rate → emission threshold, 100% → max boost

# PvP Basilica deployment
PVP_BASILICA_TTL_SECONDS = 28800
PVP_BASILICA_GPU_COUNT = 2
INDIVIDUAL_BASILICA_GPU_COUNT = 1
PVP_BASILICA_PORT = 8000

# HuggingFace container env vars (shared across all eval containers)
_HF_CONTAINER_ENV_BASE = {
    "HF_HOME": "/root/.cache/huggingface",
    "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
    "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
    "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
}
HF_CONTAINER_ENV = {**_HF_CONTAINER_ENV_BASE, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
HF_CONTAINER_ENV_IMAGE = {**_HF_CONTAINER_ENV_BASE, "HF_HUB_ENABLE_HF_TRANSFER": "0"}
