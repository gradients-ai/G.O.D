import os


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
# Text tasks (instruct/dpo/grpo/chat incl. continuous-SFT custom-arch): transformers-v5 image
# without sglang (model-prep-text.dockerfile).
MODEL_PREP_TEXT_DOCKER_IMAGE = "gradientsio/model-prep-text:latest"
INTERNAL_BRIDGE_NAME = "internal_bridge"

# Prepared tournament images (trainer/prepared_images). Builds push to a localhost registry, then blobs are
# copied to the R2 image registry with parallel multipart uploads; pulls do parallel ranged GETs into the
# same local registry and `docker pull` from localhost.
IMAGE_BUILDER_ENABLED_ENV = "IMAGE_BUILDER_ENABLED"  # default true; builds still need IMAGE_REGISTRY_*
LOCAL_REGISTRY_CONTAINER_NAME = "god-image-registry"
LOCAL_REGISTRY_IMAGE = "registry:3.0.0"
LOCAL_REGISTRY_PORT = int(os.getenv("IMAGE_LOCAL_REGISTRY_PORT", "5050"))
LOCAL_REGISTRY_STORAGE_PATH = os.getenv("IMAGE_LOCAL_REGISTRY_STORAGE", "/tmp/trainer/image_registry")
IMAGE_BUILD_WORK_PATH = "/tmp/trainer/image_builds/"
PREPARED_IMAGES_STATE_PATH = "trainer/prepared_images_state.json"
# Measured on a 104 vCPU / 885 GB / 11 TB H100 host: builds are network/disk bound (CPU p95 <= 20% with
# 9 concurrent builds, <= 14 GB RAM) and disk grows ~75-80 GB per concurrent build (layers + image +
# local registry copy). External fetch failures (GitHub/DNS) appeared at 8-9 concurrent builds.
IMAGE_BUILD_DISK_GB_PER_SLOT = 80
IMAGE_BUILD_DISK_HEADROOM = 0.8
IMAGE_BUILD_VCPUS_PER_SLOT = 4
IMAGE_BUILD_RAM_GB_PER_SLOT = 8
IMAGE_BUILD_MAX_SLOTS = 16
IMAGE_BUILD_TIMEOUT_SECONDS = 60 * 60
IMAGE_BUILD_STALL_SECONDS = 10 * 60  # no build output for this long => kill (a hidden-output step hung 30+ min)
IMAGE_CLONE_TIMEOUT_SECONDS = 15 * 60
IMAGE_BUILD_LOG_TAIL_CHARS = 4000
IMAGE_BUILD_JOB_RETENTION_HOURS = 24
IMAGE_BUILD_CACHE_MAX_AGE = "24h"  # build cache older than this is pruned: cache is reused within a tournament only
R2_PART_SIZE_BYTES = 64 * 1024 * 1024
R2_PARALLEL_PARTS = 16
R2_PARALLEL_BLOBS = 4
R2_TRANSFER_RETRIES = 4
IMAGE_PULL_ATTEMPTS = 3
IMAGE_PULL_TIMEOUT_SECONDS = 60 * 60
PREPARED_IMAGE_TTL_HOURS = 72
PREPARED_IMAGE_MIN_FREE_DISK_GB = 500

# Resource allocation
MEMORY_PER_GPU_GB = 110  # ~61% of 1440GB / 8 GPUs
CPUS_PER_GPU = 24  # Conservative allocation leaving headroom

# Timing and retries
CACHE_CLEANUP_CUTOFF_HOURS = 72
STALE_TASK_GRACE_MINUTES = 10
MODEL_PREP_TIMEOUT_MINUTES = 60
DOCKER_BUILD_TIMEOUT_MINUTES = 30
CONTAINER_START_MAX_RETRIES = 3
CONTAINER_START_RETRY_DELAY_SECONDS = 3

# Cache and output paths
CACHE_ROOT_PATH = "/cache"
HUGGINGFACE_CACHE_PATH = "/cache/hf_cache"
AI_TOOLKIT_MODELS_PATH = "/cache/ai_toolkit_models"
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
