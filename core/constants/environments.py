from dataclasses import dataclass
from enum import Enum

from core.constants.docker import MCTS_API_DOCKER_IMAGE
from core.constants.docker import VALIDATOR_DOCKER_IMAGE_INTERCODE
from core.constants.docker import VALIDATOR_DOCKER_IMAGE_PVP


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
    # Retained for payload compatibility. Model-prep environment baselines run
    # until the per-environment time budget expires.
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
            raise ValueError("EnvironmentConfig with eval_type=INDIVIDUAL must define tournament_eval_command")


ENVIRONMENT_CONFIGS: dict[EnvironmentName, EnvironmentConfig] = {
    EnvironmentName.LEDUC_POKER: EnvironmentConfig(
        task_id_min=200_000_000,
        task_id_max=299_999_999,
        num_seeds=2000,
        num_baseline_episodes=0,
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
        num_baseline_episodes=0,
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
        num_baseline_episodes=0,
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
        num_baseline_episodes=0,
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
        num_baseline_episodes=0,
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
        num_baseline_episodes=0,
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
        num_baseline_episodes=0,
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
        tournament_eval_command=["python", "-m", "validator.evaluation.evaluators.intercode"],
        gpu_multiplier=4,
    ),
}
