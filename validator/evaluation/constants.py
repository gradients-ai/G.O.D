from core.constants.datasets import GRPO_DEFAULT_FIELD_PROMPT
from core.constants.environments import EnvironmentName
from core.pvp import constants as pvp_cst


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


DOCKER_EVAL_HF_CACHE_DIR = "/root/.cache/huggingface"

# DPO evaluation
TRL_DPO_FIELD_PROMPT = "prompt"
TRL_DPO_FIELD_CHOSEN = "chosen"
TRL_DPO_FIELD_REJECTED = "rejected"

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
ENV_EVAL_SESSION_TIMEOUT = 4 * 60 * 60  # 4 hours

SGLANG_ENV_EVAL_EXTRA_CLI = (
    "--attention-backend triton --prefill-attention-backend triton --decode-attention-backend triton --sampling-backend pytorch"
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
EVAL_BASILICA_MAX_CONSECUTIVE_POLL_FAILURES = 5
EVAL_BASILICA_FAILED_POLL_RECHECK_SECONDS = 30
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
# Mid-eval recovery: if a model's SGLang server becomes unreachable DURING a
# matchup (infra blip / crash), wait for it to come back and replay the game
# rather than penalizing the miner. Shorter than the 30-min setup timeout —
# a server that doesn't recover in this window is treated as a hard failure
# and the eval is re-run by the orchestrator. Retries bound a flapping server.
PVP_SERVER_RECOVERY_HEALTH_TIMEOUT = 300
PVP_SERVER_RECOVERY_MAX_RETRIES = 2
PVP_SGLANG_API_PATH = "/v1"
PVP_RESULTS_PATH = "/app/pvp_results.json"
PVP_CONFIG_PATH = "/config/pvp_eval.json"
PVP_CONFIG_ENV_VAR = "PVP_EVAL_CONFIG"
PVP_SEED_RANGE_MAX = pvp_cst.PVP_SEED_RANGE_MAX
PVP_CONFIG_ID_DIVISOR = pvp_cst.PVP_CONFIG_ID_DIVISOR
PVP_LOG_INTERVAL_GAMES = 100
PVP_TURN_TIMEOUT_SECONDS = pvp_cst.PVP_TURN_TIMEOUT_SECONDS
PVP_REFLECTION_TIMEOUT_SECONDS = pvp_cst.PVP_REFLECTION_TIMEOUT_SECONDS
PVP_RETRY_BACKOFF_CAP_SECONDS = pvp_cst.PVP_RETRY_BACKOFF_CAP_SECONDS
PVP_HTTP_READ_TIMEOUT_SECONDS = pvp_cst.PVP_HTTP_READ_TIMEOUT_SECONDS
PVP_HTTP_MAX_RETRIES = pvp_cst.PVP_HTTP_MAX_RETRIES
PVP_TURN_MAX_TOKENS = pvp_cst.PVP_TURN_MAX_TOKENS
PVP_REFLECTION_MAX_TOKENS = pvp_cst.PVP_REFLECTION_MAX_TOKENS
PVP_WORKING_MEM_SLOTS = pvp_cst.PVP_WORKING_MEM_SLOTS
PVP_WORKING_SLOT_TOKENS = pvp_cst.PVP_WORKING_SLOT_TOKENS
PVP_LONGTERM_MEM_SLOTS = pvp_cst.PVP_LONGTERM_MEM_SLOTS
PVP_LONGTERM_SLOT_TOKENS = pvp_cst.PVP_LONGTERM_SLOT_TOKENS
PVP_EPISODE_FORFEIT_THRESHOLD = 10
PVP_MATCHUP_TIME_BUDGET_SECONDS = pvp_cst.PVP_MATCHUP_TIME_BUDGET_SECONDS
PVP_CONSECUTIVE_LOSS_FORFEIT = 10

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
