DEFAULT_IMAGE_DOCKERFILE_PATH = "dockerfiles/standalone-image-trainer.dockerfile"
DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH = "dockerfiles/standalone-image-toolkit-trainer.dockerfile"
DEFAULT_TEXT_DOCKERFILE_PATH = "dockerfiles/standalone-text-trainer.dockerfile"
TEMP_REPO_PATH = "/tmp/trainer/repos/"
TASKS_FILE_PATH = "trainer/task_history.json"
VOLUME_NAMES = ["checkpoints", "cache"]
HF_UPLOAD_DOCKER_IMAGE = "diagonalge/hf-uploader:latest"
TRAINER_DOWNLOADER_DOCKER_IMAGE = "diagonalge/trainer-downloader:latest"
CACHE_CLEANER_DOCKER_IMAGE = "diagonalge/cache-cleaner:latest"
IMAGE_TASKS_HF_SUBFOLDER_PATH = "checkpoints"
VECTOR_URL = "http://localhost:8688"  # Vector http_server for logging
INTERNAL_BRIDGE_NAME = "internal_bridge"

# Dynamic resource allocation based on GPU count
# For 8xH100 with 1440GB RAM and 252 CPUs
MEMORY_PER_GPU_GB = 110  # ~61% of 1440GB / 8 GPUs
CPUS_PER_GPU = 24  # Conservative allocation leaving headroom

CACHE_CLEANUP_CUTOFF_HOURS = 72
STALE_TASK_GRACE_MINUTES = 10
CONTAINER_START_MAX_RETRIES = 3
CONTAINER_START_RETRY_DELAY_SECONDS = 3

# TRAINING PATHS
CACHE_ROOT_PATH = "/cache"
HUGGINGFACE_CACHE_PATH = "/cache/hf_cache"
OUTPUT_CHECKPOINTS_PATH = "/app/checkpoints/"
CACHE_MODELS_DIR = "/cache/models"
CACHE_DATASETS_DIR = "/cache/datasets"
WANDB_LOGS_DIR = "/app/checkpoints/wandb_logs"
IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH = "/workspace/core/config"
IMAGE_CONTAINER_CONFIG_SAVE_PATH = "/dataset/configs"
IMAGE_CONTAINER_IMAGES_PATH = "/dataset/images"

# Directories

AXOLOTL_DIRECTORIES = {
    "data": "/workspace/axolotl/data",
    "prepared": "/workspace/axolotl/data_prepared",
    "configs": "/workspace/axolotl/configs",
    "outputs": "/workspace/axolotl/outputs",
    "input": "/workspace/input_data",
    "root": "/workspace/axolotl",
    "src": "/workspace/axolotl/src/",
}

WANDB_DIRECTORIES = [
    "WANDB_DIR",
    "WANDB_CACHE_DIR",
    "WANDB_ARTIFACT_DIR",
    "WANDB_DATA_DIR",
    "WANDB_CONFIG_DIR",
]