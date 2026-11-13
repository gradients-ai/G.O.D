import os
from pathlib import Path

from core.constants.docker import DEFAULT_TEXT_DOCKERFILE_PATH as DEFAULT_TEXT_DOCKERFILE_PATH
from core.constants.docker import LEGACY_TEXT_DOCKERFILE_PATH as LEGACY_TEXT_DOCKERFILE_PATH
from core.constants.docker import TEXT_DOCKERFILE_PATHS as TEXT_DOCKERFILE_PATHS


# Training repository Dockerfiles (the text ones are defined once in core.constants.docker, shared with the validator lint)
DEFAULT_IMAGE_DOCKERFILE_PATH = "ops/docker/standalone-image-trainer.dockerfile"
DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH = "ops/docker/standalone-image-toolkit-trainer.dockerfile"

LEGACY_IMAGE_DOCKERFILE_PATH = "dockerfiles/standalone-image-trainer.dockerfile"
LEGACY_IMAGE_TOOLKIT_DOCKERFILE_PATH = "dockerfiles/standalone-image-toolkit-trainer.dockerfile"

IMAGE_DOCKERFILE_PATHS = (DEFAULT_IMAGE_DOCKERFILE_PATH, LEGACY_IMAGE_DOCKERFILE_PATH)
IMAGE_TOOLKIT_DOCKERFILE_PATHS = (DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH, LEGACY_IMAGE_TOOLKIT_DOCKERFILE_PATH)


# Runtime state
TEMP_REPO_PATH = "/tmp/trainer/repos/"
TASKS_FILE_PATH = "trainer/task_history.json"

# Docker volumes and images
CHECKPOINTS_VOLUME_NAME = "checkpoints"
CACHE_VOLUME_NAME = "cache"
VOLUME_NAMES = [CHECKPOINTS_VOLUME_NAME, CACHE_VOLUME_NAME]
# Name prefixes of every container the trainer starts (runtime.py, containers/dataset_cache.py, quant_decode/sandbox.py).
# cleanup.py removes stale containers under these prefixes only; anything else on the host is left alone.
TRAINER_CONTAINER_NAME_PREFIXES = (
    "text-trainer-",
    "image-trainer-",
    "model-prep-",
    "downloader-",
    "environment-server-",
    "hf-upload-",
    "miner-ds-",
    "quant-decode-",
    "output-reclaim-",
)

HF_UPLOAD_DOCKER_IMAGE = "gradientsio/trainer-uploader:latest"
TRAINER_DOWNLOADER_DOCKER_IMAGE = "gradientsio/trainer-downloader:latest"
CACHE_CLEANER_DOCKER_IMAGE = "gradientsio/trainer-cacher-cleaner:latest"
# Text tasks (instruct/dpo/grpo/chat incl. continuous-SFT custom-arch): transformers-v5 image
# without sglang (model-prep-text.dockerfile).
MODEL_PREP_TEXT_DOCKER_IMAGE = "gradientsio/model-prep-text:latest"
INTERNAL_BRIDGE_NAME = "internal_bridge"
# Quant decoder sandbox (ops/docker/quant-decoder-*.dockerfile): build image = run image + compiler.
QUANT_DECODER_BUILD_IMAGE = "gradientsio/quant-decoder-build:pinned"
QUANT_DECODER_RUN_IMAGE = "gradientsio/quant-decoder-run:pinned"

# Resource allocation
MEMORY_PER_GPU_GB = 110  # ~61% of 1440GB / 8 GPUs
CPUS_PER_GPU = 24  # Conservative allocation leaving headroom
QUANT_DECODE_MIN_MEMORY_GB = 8  # memory floor for both decoder sandbox phases

# Timing and retries
CACHE_CLEANUP_CUTOFF_HOURS = 72
STALE_TASK_GRACE_MINUTES = 10
STALE_CONTAINER_MAX_AGE_HOURS = 1  # a created/exited trainer container older than this is removed by cleanup.py
MODEL_PREP_TIMEOUT_MINUTES = 60
DOCKER_BUILD_TIMEOUT_MINUTES = 30

# Miner images are always built with no cache (runtime.build_docker_image), so a build never reuses layers from another
# entrant's build. After each job the trainer also runs a host-wide `docker builder prune` (BuildKit cache), which
# reclaims nothing this trainer built (docker-py's build API defaults to the legacy builder) but wipes every other BuildKit
# cache on the host. On by default; TRAINER_PRUNE_BUILD_CACHE=false skips it on a shared box.
PRUNE_HOST_BUILD_CACHE = os.getenv("TRAINER_PRUNE_BUILD_CACHE", "true").lower() == "true"
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

# Quant decoder sandbox container paths
DECODE_DECODER_PATH = "/decoder"
DECODE_BUILD_OUT_PATH = "/build"
DECODE_ARTIFACT_PATH = "/artifact"
DECODE_OUTPUT_PATH = "/output"
DECODE_BUILD_SCRIPT = "build.sh"

# Quant freeform decode job (trainer/quant_decode/job.py)
# Private (0o700) decode work root, deliberately NOT on the cache volume: every training container mounts that volume
# at /cache, so a decoder placed there would be readable by co-tenants. Needs ~3x the largest bf16 model free.
QUANT_DECODE_WORK_ROOT = Path(
    os.getenv("QUANT_DECODE_WORK_ROOT") or Path.home() / ".local" / "share" / "god-trainer" / "quant_decode"
)
QUANT_DECODER_FETCH_SLACK_BYTES = 1024 * 1024  # snapshot download cap = DECODER_MAX_XZ_BYTES + this slack
QUANT_DECODER_FETCH_TIMEOUT_SECONDS = 60
# Freeform calibration set (trainer/quant_calibration.py): fetched on the host and handed to the miner as a file.
QUANT_CALIBRATION_FETCH_TIMEOUT_SECONDS = 60
QUANT_CALIBRATION_FETCH_ATTEMPTS = 3  # fetch, check and write attempts before the job fails without a calibration file
QUANT_CALIBRATION_RETRY_SLEEP_SECONDS = 5
QUANT_CALIBRATION_ATTEMPT_SLACK_SECONDS = 30  # per attempt, on top of the fetch timeout: parse, fingerprint, write
QUANT_CALIBRATION_MAX_BYTES = 256 * 1024 * 1024  # the file is ~1,000 tokenised rows; anything larger is not ours
# Decoded bf16 shards go to S3-compatible storage as multipart uploads; memory in flight is part size x parallel =
# 512 MiB.
QUANT_DECODE_UPLOAD_PART_BYTES = 32 * 1024 * 1024
QUANT_DECODE_UPLOAD_PARALLEL = 16
QUANT_DECODE_UPLOAD_MIN_TIMEOUT_SECONDS = 300  # per-file deadline floor
QUANT_DECODE_UPLOAD_MIN_BYTES_PER_SECOND = 2 * 1000 * 1000  # per-file deadline: floor + size at this rate
QUANT_ARTIFACT_FORBIDDEN_SUFFIXES = (
    ".py",
    ".pyc",
    ".pyo",
    ".pyw",
    ".so",
    ".sh",
    ".bash",
    ".exe",
    ".dll",
    ".dylib",
    ".wasm",
    ".js",
    ".jar",
    ".class",
    ".dockerfile",
)
QUANT_DECODE_LOG_TAIL_LINES = 50  # stderr lines read from a failed decoder container

# Output reclaim (trainer/output_reclaim.py): a short root container that chowns a training output dir to the trainer.
# The uploader image: it has coreutils and every trainer already pulls it, since every job uploads with it.
OUTPUT_RECLAIM_DOCKER_IMAGE = HF_UPLOAD_DOCKER_IMAGE
# chown any file, read/search any dir (miners can chmod 0 their own dirs), chmod files it no longer owns. Nothing else:
# no DAC_OVERRIDE, so it can write nothing it has not just been handed.
OUTPUT_RECLAIM_CAPS = ["CHOWN", "DAC_READ_SEARCH", "FOWNER"]
OUTPUT_RECLAIM_TIMEOUT_SECONDS = 300
OUTPUT_RECLAIM_MEM_LIMIT = "512m"
OUTPUT_RECLAIM_PIDS_LIMIT = 64
OUTPUT_RECLAIM_LOG_TAIL_LINES = 20

# Quant pre-upload check (trainer/quant_upload_check.py)
QUANT_OUTPUT_REJECTED_PREFIX = "quant output rejected: "
QUANT_MANIFEST_ERROR_MAX_CHARS = 500  # an invalid manifest's pydantic error is truncated to this in the job log

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
