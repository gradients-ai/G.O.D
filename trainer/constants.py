# Training repository Dockerfiles
DEFAULT_IMAGE_DOCKERFILE_PATH = "ops/docker/standalone-image-trainer.dockerfile"
DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH = "ops/docker/standalone-image-toolkit-trainer.dockerfile"
DEFAULT_TEXT_DOCKERFILE_PATH = "ops/docker/standalone-text-trainer.dockerfile"

LEGACY_IMAGE_DOCKERFILE_PATH = "dockerfiles/standalone-image-trainer.dockerfile"
LEGACY_IMAGE_TOOLKIT_DOCKERFILE_PATH = "dockerfiles/standalone-image-toolkit-trainer.dockerfile"
LEGACY_TEXT_DOCKERFILE_PATH = "dockerfiles/standalone-text-trainer.dockerfile"

IMAGE_DOCKERFILE_PATHS = (DEFAULT_IMAGE_DOCKERFILE_PATH, LEGACY_IMAGE_DOCKERFILE_PATH)
IMAGE_TOOLKIT_DOCKERFILE_PATHS = (DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH, LEGACY_IMAGE_TOOLKIT_DOCKERFILE_PATH)
TEXT_DOCKERFILE_PATHS = (DEFAULT_TEXT_DOCKERFILE_PATH, LEGACY_TEXT_DOCKERFILE_PATH)

# Runtime state
TEMP_REPO_PATH = "/tmp/trainer/repos/"
TASKS_FILE_PATH = "trainer/task_history.json"

# Docker volumes and images
CHECKPOINTS_VOLUME_NAME = "checkpoints"
CACHE_VOLUME_NAME = "cache"
VOLUME_NAMES = [CHECKPOINTS_VOLUME_NAME, CACHE_VOLUME_NAME]

HF_UPLOAD_DOCKER_IMAGE = "gradientsio/trainer-uploader:latest"
TRAINER_DOWNLOADER_DOCKER_IMAGE = "gradientsio/trainer-downloader:latest"
CACHE_CLEANER_DOCKER_IMAGE = "gradientsio/trainer-cacher-cleaner:latest"
MODEL_PREP_DOCKER_IMAGE = "gradientsio/model-prep:latest"
INTERNAL_BRIDGE_NAME = "internal_bridge"

# Resource allocation
MEMORY_PER_GPU_GB = 110  # ~61% of 1440GB / 8 GPUs
CPUS_PER_GPU = 24  # Conservative allocation leaving headroom

# Timing and retries
CACHE_CLEANUP_CUTOFF_HOURS = 72
STALE_TASK_GRACE_MINUTES = 10
MODEL_PREP_TIMEOUT_MINUTES = 60
CONTAINER_START_MAX_RETRIES = 3
CONTAINER_START_RETRY_DELAY_SECONDS = 3

# Cache and output paths
CACHE_ROOT_PATH = "/cache"
HUGGINGFACE_CACHE_PATH = "/cache/hf_cache"
OUTPUT_CHECKPOINTS_PATH = "/app/checkpoints/"
CACHE_MODELS_DIR = "/cache/models"
CACHE_DATASETS_DIR = "/cache/datasets"
MINER_DATASETS_CACHE_DIR = "/cache/miner_datasets"
WANDB_LOGS_DIR = "/app/checkpoints/wandb_logs"

# Container training paths
IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH = "/workspace/core/training_templates"
IMAGE_CONTAINER_CONFIG_SAVE_PATH = "/dataset/configs"
IMAGE_CONTAINER_IMAGES_PATH = "/dataset/images"
IMAGE_TASKS_HF_SUBFOLDER_PATH = "checkpoints"

# Environment variables
MINER_DATASETS_DIR_ENV = "MINER_DATASETS_DIR"
MINER_DATASETS_ENV = "MINER_DATASETS"

# Observability
VECTOR_URL = "http://localhost:8688"  # Vector http_server for logging

# Diffusion defaults
DIFFUSION_SDXL_REPEATS = 10
DIFFUSION_FLUX_REPEATS = 1
DIFFUSION_DEFAULT_INSTANCE_PROMPT = "lora"
DIFFUSION_DEFAULT_CLASS_PROMPT = "style"

# Axolotl directories
AXOLOTL_DIRECTORIES = {
    "data": "/workspace/axolotl/data",
    "prepared": "/workspace/axolotl/data_prepared",
    "configs": "/workspace/axolotl/configs",
    "outputs": "/workspace/axolotl/outputs",
    "input": "/workspace/input_data",
    "root": "/workspace/axolotl",
    "src": "/workspace/axolotl/src/",
}

# W&B directories
WANDB_DIRECTORIES = [
    "WANDB_DIR",
    "WANDB_CACHE_DIR",
    "WANDB_ARTIFACT_DIR",
    "WANDB_DATA_DIR",
    "WANDB_CONFIG_DIR",
]
