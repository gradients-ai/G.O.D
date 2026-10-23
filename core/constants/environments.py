from dataclasses import dataclass
from enum import Enum


class EvalType(str, Enum):
    INDIVIDUAL = "individual"
    PVP = "pvp"


class TrainingStartPoint(str, Enum):
    """What model a task trains from."""

    DEFAULT = "default"
    CONTINUATION = "continuation"
    FROM_SCRATCH = "from_scratch"
    PREVIOUS_WINNER = "previous_winner"
    # Lineage tag for the continuous-SFT boss task; the trainer ignores it, the validator uses it
    # to locate the task and advance continuous_sft_state.
    CONTINUOUS_SFT = "continuous_sft"


class EnvironmentName(str, Enum):
    GIN_RUMMY = "gin_rummy"
    LIARS_DICE = "liars_dice"
    LEDUC_POKER = "leduc_poker"
    OTHELLO = "othello"
    CLOBBER = "clobber"
    GOOFSPIEL = "goofspiel"
    INTERCODE = "intercode"
    SWE_INFINITE = "swe_infinite"


@dataclass(frozen=True)
class EnvironmentConfig:
    task_id_min: int
    task_id_max: int
    num_seeds: int
    # Retained for payload compatibility. Model-prep environment baselines run
    # until the per-environment time budget expires.
    num_baseline_episodes: int
    eval_type: EvalType
    # Environment evaluation/training containers were sunset. These image/command
    # fields are retained (empty) only so stored env rows and legacy payloads
    # continue to deserialize; no environment container is built or launched.
    env_image: str = ""
    env_server_command: list[str] | None = None
    tournament_eval_image: str = ""
    tournament_eval_command: list[str] | None = None
    gpu_multiplier: int = 4
    eval_payload_extra: dict | None = None


ENVIRONMENT_CONFIGS: dict[EnvironmentName, EnvironmentConfig] = {
    EnvironmentName.LEDUC_POKER: EnvironmentConfig(
        task_id_min=200_000_000,
        task_id_max=299_999_999,
        num_seeds=2000,
        num_baseline_episodes=0,
        eval_type=EvalType.PVP,
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
        gpu_multiplier=4,
    ),
    EnvironmentName.SWE_INFINITE: EnvironmentConfig(
        task_id_min=1,
        task_id_max=7000,
        num_seeds=6,
        num_baseline_episodes=0,
        eval_type=EvalType.INDIVIDUAL,
        gpu_multiplier=4,
    ),
}
