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
VALIDATOR_DOCKER_IMAGE_INTERCODE = "gradientsio/env-eval-intercode:basilica"
VALIDATOR_DOCKER_IMAGE_PVP = "gradientsio/pvp-evaluator:basilica"
MCTS_API_DOCKER_IMAGE = "gradientsio/mcts-api:latest"


class EvalType(str, Enum):
    INDIVIDUAL = "individual"
    PVP = "pvp"


class TrainingStartPoint(str, Enum):
    """What model a task trains from."""

    DEFAULT = "default"
    CONTINUATION = "continuation"
    FROM_SCRATCH = "from_scratch"
    PREVIOUS_WINNER = "previous_winner"



class EnvironmentName(str, Enum):
    GIN_RUMMY = "gin_rummy"
    LIARS_DICE = "liars_dice"
    LEDUC_POKER = "leduc_poker"
    OTHELLO = "othello"
    CLOBBER = "clobber"
    GOOFSPIEL = "goofspiel"
    INTERCODE = "intercode"


@dataclass(frozen=True)
class EnvironmentConfig:
    task_id_min: int
    task_id_max: int
    num_seeds: int
    num_baseline_episodes: int
    eval_type: EvalType
    env_image: str = ""
    env_server_command: list[str] | None = None
    tournament_eval_image: str = VALIDATOR_DOCKER_IMAGE_PVP
    tournament_eval_command: list[str] | None = None
    gpu_multiplier: int = 4
    eval_payload_extra: dict | None = None

    def __post_init__(self):
        if self.eval_type == EvalType.INDIVIDUAL and not self.tournament_eval_command:
            raise ValueError(
                "EnvironmentConfig with eval_type=INDIVIDUAL must define tournament_eval_command"
            )


ENVIRONMENT_CONFIGS: dict[EnvironmentName, EnvironmentConfig] = {
    EnvironmentName.LEDUC_POKER: EnvironmentConfig(
        task_id_min=200_000_000,
        task_id_max=299_999_999,
        num_seeds=2000,
        num_baseline_episodes=50,
        eval_type=EvalType.PVP,
        env_image=MCTS_API_DOCKER_IMAGE,
        tournament_eval_image=VALIDATOR_DOCKER_IMAGE_PVP,
        gpu_multiplier=4,
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
        eval_type=EvalType.PVP,
        env_image=MCTS_API_DOCKER_IMAGE,
        tournament_eval_image=VALIDATOR_DOCKER_IMAGE_PVP,
        gpu_multiplier=4,
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
        eval_type=EvalType.PVP,
        env_image=MCTS_API_DOCKER_IMAGE,
        tournament_eval_image=VALIDATOR_DOCKER_IMAGE_PVP,
        gpu_multiplier=4,
        eval_payload_extra={
            "opponent": "mcts",
            "mcts_max_simulations": 50,
            "mcts_num_rollouts": 1,
            "api_key": "dummy-key",
        },
    ),
    EnvironmentName.OTHELLO: EnvironmentConfig(
        task_id_min=400_000_000,
        task_id_max=499_999_999,
        num_seeds=10_000,
        num_baseline_episodes=25,
        eval_type=EvalType.PVP,
        env_image=MCTS_API_DOCKER_IMAGE,
        tournament_eval_image=VALIDATOR_DOCKER_IMAGE_PVP,
        gpu_multiplier=4,
        eval_payload_extra={
            "opponent": "mcts",
            "mcts_max_simulations": 50,
            "mcts_num_rollouts": 1,
            "api_key": "dummy-key",
        },
    ),
    EnvironmentName.CLOBBER: EnvironmentConfig(
        task_id_min=700_000_000,
        task_id_max=799_999_999,
        num_seeds=10_000,
        num_baseline_episodes=25,
        eval_type=EvalType.PVP,
        env_image=MCTS_API_DOCKER_IMAGE,
        tournament_eval_image=VALIDATOR_DOCKER_IMAGE_PVP,
        gpu_multiplier=4,
        eval_payload_extra={
            "opponent": "mcts",
            "mcts_max_simulations": 50,
            "mcts_num_rollouts": 1,
            "api_key": "dummy-key",
        },
    ),
    EnvironmentName.GOOFSPIEL: EnvironmentConfig(
        task_id_min=0,
        task_id_max=99_999_999,
        num_seeds=10_000,
        num_baseline_episodes=25,
        eval_type=EvalType.PVP,
        env_image=MCTS_API_DOCKER_IMAGE,
        tournament_eval_image=VALIDATOR_DOCKER_IMAGE_PVP,
        gpu_multiplier=4,
        eval_payload_extra={
            "opponent": "mcts",
            "mcts_max_simulations": 50,
            "mcts_num_rollouts": 1,
            "api_key": "dummy-key",
        },
    ),
    EnvironmentName.INTERCODE: EnvironmentConfig(
        task_id_min=1,
        task_id_max=200,
        num_seeds=20,
        num_baseline_episodes=7,
        eval_type=EvalType.INDIVIDUAL,
        env_image=VALIDATOR_DOCKER_IMAGE_INTERCODE,
        env_server_command=[
            "python",
            "-m",
            "uvicorn",
            "validator.evaluation.intercode_server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        tournament_eval_image=VALIDATOR_DOCKER_IMAGE_INTERCODE,
        tournament_eval_command=["python", "-m", "validator.evaluation.eval_intercode"],
        gpu_multiplier=4,
    ),
}

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"
CONTAINER_EVAL_SCORE_KEY = "eval_loss"
LORA_ADAPTER_CONFIG_FILE = "adapter_config.json"

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

# Env vars used to signal KL-regularised instruct training to miner containers and the evaluator.
USE_KL_ENV = "USE_KL"
KL_COEF_ENV = "KL_COEF"

# YaRN extension HuggingFace credentials (separate from main HF credentials)
YARN_HUGGINGFACE_USERNAME = os.getenv("YARN_HUGGINGFACE_USERNAME", "gradients-io")
YARN_HUGGINGFACE_TOKEN = os.getenv("YARN_HUGGINGFACE_TOKEN")

YARN_VALID_FACTORS = [2, 4, 8, 16, 32]
