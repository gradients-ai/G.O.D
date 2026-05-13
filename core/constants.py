import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv


load_dotenv()

VERSION_KEY = 61_000
# Default NETUID if not set in environment
DEFAULT_NETUID = 56

try:
    NETUID = int(os.getenv("NETUID", DEFAULT_NETUID))
except (TypeError, ValueError):
    NETUID = DEFAULT_NETUID

IS_PROD_ENV = NETUID == DEFAULT_NETUID

VALIDATOR_DOCKER_IMAGE = "gradientsio/text-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_DIFFUSION = "gradientsio/image-evaluator:basilica"
VALIDATOR_DOCKER_IMAGE_ENV = "gradientsio/env-evaluator:basilica"
MCTS_API_DOCKER_IMAGE = "diagonalge/mcts-api:latest"


class EnvironmentName(str, Enum):
    GIN_RUMMY = "gin_rummy"
    LIARS_DICE = "liars_dice"
    LEDUC_POKER = "leduc_poker"


@dataclass(frozen=True)
class EnvironmentConfig:
    task_id_min: int
    task_id_max: int
    num_seeds: int
    num_baseline_episodes: int
    env_image: str
    eval_payload_extra: dict


ENVIRONMENT_CONFIGS: dict[EnvironmentName, EnvironmentConfig] = {
    EnvironmentName.LEDUC_POKER: EnvironmentConfig(
        task_id_min=200_000_000,
        task_id_max=299_999_999,
        num_seeds=2000,
        num_baseline_episodes=50,
        env_image=MCTS_API_DOCKER_IMAGE,
        eval_payload_extra={
            "opponent": "mcts",
            "mcts_max_simulations": 50,
            "mcts_num_rollouts": 1,
            "api_key": "dummy-key",
        },
    ),
    EnvironmentName.LIARS_DICE: EnvironmentConfig(
        task_id_min=100_000_000,
        task_id_max=199_999_999,
        num_seeds=10_000,
        num_baseline_episodes=50,
        env_image=MCTS_API_DOCKER_IMAGE,
        eval_payload_extra={
            "opponent": "mcts",
            "mcts_max_simulations": 225,
            "mcts_num_rollouts": 1,
            "api_key": "dummy-key",
        },
    ),
    EnvironmentName.GIN_RUMMY: EnvironmentConfig(
        task_id_min=300_000_000,
        task_id_max=399_999_999,
        num_seeds=1000,
        num_baseline_episodes=25,
        env_image=MCTS_API_DOCKER_IMAGE,
        eval_payload_extra={
            "opponent": "mcts",
            "mcts_max_simulations": 50,
            "mcts_num_rollouts": 1,
            "api_key": "dummy-key",
        },
    ),
}

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

CONFIG_DIR = "core/config/"
OUTPUT_DIR = "core/outputs/"
CACHE_DIR = "~/.cache/huggingface"
CACHE_DIR_HUB = os.path.expanduser("~/.cache/huggingface/hub")
GRPO_MINER_OUTPUT_DIR = "/root/.cache/huggingface/hub/trained_repo"
DIFFUSION_DATASET_DIR = "core/dataset/images"

DIFFUSION_SDXL_REPEATS = 10
DIFFUSION_FLUX_REPEATS = 1
DIFFUSION_DEFAULT_INSTANCE_PROMPT = "lora"
DIFFUSION_DEFAULT_CLASS_PROMPT = "style"

MIN_IMAGE_TEXT_PAIRS = 10
MAX_IMAGE_TEXT_PAIRS = 50

CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL = CONFIG_DIR + "base_diffusion_sdxl.toml"
CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX = CONFIG_DIR + "base_diffusion_flux.toml"


CONFIG_TEMPLATE_PATH = CONFIG_DIR + "base.yml"
CONFIG_TEMPLATE_PATH_GRPO = CONFIG_DIR + "base_grpo.yml"

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")

HUGGINGFACE_USERNAME = os.getenv("HUGGINGFACE_USERNAME")
RAYONLABS_HF_USERNAME = "gradients-io-tournaments"  # "besimray"  # "rayonlabs"

# DPO default dataset type
DPO_DEFAULT_DATASET_TYPE = "chatml.default"
# Field names must match exactly what Axolotl's formatter expects
DPO_DEFAULT_FIELD_PROMPT = "question"  # chatml.intel expects 'question'
DPO_DEFAULT_FIELD_SYSTEM = "system"
DPO_DEFAULT_FIELD_CHOSEN = "chosen"
DPO_DEFAULT_FIELD_REJECTED = "rejected"

GRPO_DEFAULT_FIELD_PROMPT = "prompt"

# YaRN extension HuggingFace credentials (separate from main HF credentials)
YARN_HUGGINGFACE_USERNAME = os.getenv("YARN_HUGGINGFACE_USERNAME", "gradients-io")
YARN_HUGGINGFACE_TOKEN = os.getenv("YARN_HUGGINGFACE_TOKEN")

YARN_VALID_FACTORS = [2, 4, 8, 16, 32]
