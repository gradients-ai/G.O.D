import logging
import os
from pathlib import Path

from core.constants.docker import DEFAULT_TEXT_DOCKERFILE_PATH as DEFAULT_TEXT_DOCKERFILE_PATH
from core.constants.docker import LEGACY_TEXT_DOCKERFILE_PATH as LEGACY_TEXT_DOCKERFILE_PATH
from core.constants.docker import TEXT_DOCKERFILE_PATHS as TEXT_DOCKERFILE_PATHS


_logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int) -> int:
    """An integer setting from the environment. A value that does not parse falls back to the default and one below
    `minimum` is raised to it, each with a warning: a typo in the env file must not stop the trainer or break uploads."""
    return int(_env_number(name, default, minimum, int))


def _env_float(name: str, default: float, minimum: float) -> float:
    return float(_env_number(name, default, minimum, float))


def _env_number(name: str, default: float, minimum: float, parse: type) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = parse(raw.strip())
    except ValueError:
        _logger.warning(f"{name}={raw!r} is not a number; using the default {default}")
        return default
    if value != value or value < minimum:  # NaN compares unequal to itself
        _logger.warning(f"{name}={raw!r} is below the minimum {minimum}; using {minimum}")
        return minimum
    return value


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
# Each GPU owns an equal share of the host's cores and RAM (host_resources.py). These reserves stay with the host,
# the trainer and its helper containers.
HOST_CPU_RESERVE_FRACTION = 0.03  # of physical cores, rounded up, at least one core
HOST_MEMORY_RESERVE_FRACTION = 0.10
HOST_MEMORY_RESERVE_MIN_GB = 16
QUANT_DECODE_MIN_MEMORY_GB = 8  # memory floor for both decoder sandbox phases

# Timing and retries
CACHE_CLEANUP_CUTOFF_HOURS = 72
STALE_CONTAINER_MAX_AGE_HOURS = 1  # a created/exited trainer container older than this is removed by cleanup.py
MODEL_PREP_TIMEOUT_MINUTES = 60
DOCKER_BUILD_TIMEOUT_MINUTES = int(os.getenv("TRAINER_DOCKER_BUILD_TIMEOUT_MINUTES", "45"))
# Added to a job's hours for its stale deadline (cleanup.py). The clock starts when the job is accepted and
# restarts when its training container starts, so for a job of well under an hour the grace is what covers the image
# build. It applies after the restart as well, so a job that hangs after training is noticed that much later.
STALE_TASK_GRACE_MINUTES = 10 + DOCKER_BUILD_TIMEOUT_MINUTES
# Stopping a build that reused a too-old cached layer (runtime.BuildStepContainers): a step container the daemon has
# created but not started yet cannot be killed, so the kill is retried until it lands.
BUILD_STOP_KILL_RETRY_SECONDS = 0.5
BUILD_STOP_KILL_TIMEOUT_SECONDS = 10

# Miner images are built with the Docker layer cache, so a retry or a build that starts with the same steps does not
# download them again. TRAINER_IMAGE_BUILD_NO_CACHE=true builds every image from scratch and empties the cache after
# every job.
IMAGE_BUILD_NO_CACHE = os.getenv("TRAINER_IMAGE_BUILD_NO_CACHE", "false").lower() == "true"
# Ask the registry for a newer base image on every build. Off: a base tag stays at the copy first pulled on this host.
IMAGE_BUILD_PULL_BASE = os.getenv("TRAINER_IMAGE_BUILD_PULL", "false").lower() == "true"
# A build never keeps a cached layer made longer ago than this, and the layer cache is emptied (trainer.image_builds)
# once it has gone this long without being emptied, and whenever the disk under Docker's data directory is at least
# this full. That disk may hold more than images: if it stays over the limit, the cache is emptied on every cleanup.
IMAGE_BUILD_CACHE_MAX_AGE_HOURS = float(os.getenv("TRAINER_IMAGE_BUILD_CACHE_MAX_AGE_HOURS", "24"))
IMAGE_BUILD_CACHE_MAX_DISK_USED_PERCENT = float(os.getenv("TRAINER_IMAGE_BUILD_CACHE_MAX_DISK_USED_PERCENT", "80"))
IMAGE_BUILD_CACHE_STATE_PATH = "trainer/image_build_cache_state.json"
# When the layer cache is emptied, the trainer also runs a host-wide `docker builder prune`, which wipes every other
# build cache on the host. On by default; TRAINER_PRUNE_BUILD_CACHE=false skips it on a shared box.
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
# Cache downloads build in a hidden sibling ".<name>.download-<random>" and are renamed into place only once complete, so a
# path under a cache root is either a finished entry or absent. Directory entries carry this marker file; a directory
# without it predates atomic downloads and is verified before reuse. A partial left by a killed download is garbage that
# cache_cleanup.py removes once nothing in it has been written for CACHE_PARTIAL_DOWNLOAD_STALE_HOURS.
CACHE_DOWNLOAD_TAG = ".download-"
CACHE_COMPLETE_MARKER = ".cache_complete"
CACHE_PARTIAL_DOWNLOAD_STALE_HOURS = 12
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
# 64 MiB. A few parts in flight keep a link busy; more only multiply the chance that one of them hits a dropped
# connection, which fails the whole file. A part that fails is sent again in full, so smaller parts lose less. S3
# allows 10,000 parts per object (16 MiB parts carry files up to ~156 GiB) and no part under 5 MiB.
S3_MIN_PART_BYTES = 5 * 1024 * 1024
QUANT_DECODE_UPLOAD_PART_BYTES = _env_int("QUANT_DECODE_UPLOAD_PART_BYTES", 16 * 1024 * 1024, S3_MIN_PART_BYTES)
QUANT_DECODE_UPLOAD_PARALLEL = _env_int("QUANT_DECODE_UPLOAD_PARALLEL", 4, 1)
# A part's body goes to the socket in slices of this size, so the socket timeout bounds one slice, not the whole part.
# 0 sends each part in one write, which must then finish within the connect timeout.
QUANT_DECODE_UPLOAD_SEND_SLICE_BYTES = _env_int("QUANT_DECODE_UPLOAD_SEND_SLICE_BYTES", 1024 * 1024, 0)
# Each request (one part) is retried by the HTTP client on connection errors and 429/5xx, with backoff and jitter.
QUANT_DECODE_UPLOAD_HTTP_RETRIES = _env_int("QUANT_DECODE_UPLOAD_HTTP_RETRIES", 5, 0)
QUANT_DECODE_UPLOAD_HTTP_BACKOFF_FACTOR = 1.0  # urllib3: sleeps 0, 2, 4, 8, 16 s between retries
QUANT_DECODE_UPLOAD_HTTP_BACKOFF_MAX_SECONDS = 30
QUANT_DECODE_UPLOAD_HTTP_BACKOFF_JITTER_SECONDS = 1.0
# Connecting, and each socket write while the request body is sent (the HTTP client keeps this timeout on the socket
# until the body is out).
QUANT_DECODE_UPLOAD_HTTP_CONNECT_TIMEOUT_SECONDS = _env_float("QUANT_DECODE_UPLOAD_HTTP_CONNECT_TIMEOUT_SECONDS", 30, 1)
# Waiting for a response once the body is sent: seconds for a part; the complete call is the slow one. Kept short so
# a stalled request leaves room in the file deadline for a whole-file retry.
QUANT_DECODE_UPLOAD_HTTP_READ_TIMEOUT_SECONDS = _env_float("QUANT_DECODE_UPLOAD_HTTP_READ_TIMEOUT_SECONDS", 90, 1)
# When a file still fails on a transient error, the whole file is uploaded again (same key, so it overwrites) after
# an exponential backoff with jitter, up to this many attempts and always within the file's deadline below.
QUANT_DECODE_UPLOAD_ATTEMPTS = 4
QUANT_DECODE_UPLOAD_RETRY_BASE_SECONDS = 10
QUANT_DECODE_UPLOAD_RETRY_MAX_SECONDS = 60
QUANT_DECODE_UPLOAD_MIN_TIMEOUT_SECONDS = 300  # per-file deadline floor
# Per-file deadline: floor + size at this rate.
QUANT_DECODE_UPLOAD_MIN_BYTES_PER_SECOND = _env_int("QUANT_DECODE_UPLOAD_MIN_BYTES_PER_SECOND", 2 * 1000 * 1000, 1)
# When the decoded files still cannot be uploaded for a transient reason, the job finishes and the upload is retried
# in the background (trainer/quant_decode/upload_retry.py): one attempt per missing file per round, rounds spaced by
# an exponential backoff with jitter between the base and the cap, until this long after the first failure. Only then
# is the decode an infra_error. The decoded files stay on disk meanwhile. 0 turns the background retry off.
QUANT_DECODE_UPLOAD_RETRY_WINDOW_SECONDS = _env_float("QUANT_DECODE_UPLOAD_RETRY_WINDOW_SECONDS", 2 * 60 * 60, 0)
QUANT_DECODE_UPLOAD_RETRY_ROUND_BASE_SECONDS = _env_float("QUANT_DECODE_UPLOAD_RETRY_ROUND_BASE_SECONDS", 60, 1)
QUANT_DECODE_UPLOAD_RETRY_ROUND_MAX_SECONDS = _env_float(
    "QUANT_DECODE_UPLOAD_RETRY_ROUND_MAX_SECONDS", 900, QUANT_DECODE_UPLOAD_RETRY_ROUND_BASE_SECONDS
)
# Decoded files are kept for a retry only while the work root's disk has at least this much free after the decoder and
# the artifact are removed; below it the failed upload is an infra_error at once, so kept files never starve the next
# decodes of space.
QUANT_DECODE_UPLOAD_RETRY_MIN_FREE_DISK_BYTES = _env_int("QUANT_DECODE_UPLOAD_RETRY_MIN_FREE_DISK_GB", 50, 0) * 1000**3
# A decode, or a round of the upload retry, holds its job's lock (trainer/quant_decode/job_lock.py). A decode that
# finds the lock taken waits this long for it, then ends as an infra_error.
QUANT_DECODE_JOB_LOCK_WAIT_SECONDS = 60 * 60
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
# Emptying a job's directory before a new attempt (output_reclaim.clear_job_checkpoints): delete entries of any owner
# and mode, in sticky directories too. Only that job's directory is mounted.
JOB_DIR_CLEAR_CAPS = ["DAC_OVERRIDE", "FOWNER"]
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
