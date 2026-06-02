import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    task_id_range: tuple[int, int]
    opponent: str
    mcts_sim_range: tuple[int, int]
    mcts_num_rollouts: int
    num_episodes: int


GAME_CONFIGS: dict[str, GameConfig] = {
    "liars_dice": GameConfig(
        task_id_range=(100_000_000, 199_999_999),
        opponent="mcts",
        mcts_sim_range=(150, 300),
        mcts_num_rollouts=1,
        num_episodes=400,
    ),
    "leduc_poker": GameConfig(
        task_id_range=(200_000_000, 299_999_999),
        opponent="mcts",
        mcts_sim_range=(25, 150),
        mcts_num_rollouts=1,
        num_episodes=400,
    ),
    "gin_rummy": GameConfig(
        task_id_range=(300_000_000, 399_999_999),
        opponent="mcts",
        mcts_sim_range=(25, 150),
        mcts_num_rollouts=1,
        num_episodes=300,
    ),
}

MAX_TURNS = 30
CONTEXT_WINDOW = 6  # Send only last N messages to Claude (saves input tokens on long games)
CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CONCURRENCY = 4
DEFAULT_EPISODES_PER_GAME = 1000

FORMAT_INSTRUCTIONS = (
    'Your output must strictly follow this format: '
    '"Thought:\nyour thoughts ONLY in text.\n\nAction:\n'
    'ONLY your action ID (a single number)."'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SFT dataset by playing G.O.D games with Claude"
    )
    parser.add_argument(
        "--games",
        nargs="+",
        default=list(GAME_CONFIGS.keys()),
        choices=list(GAME_CONFIGS.keys()),
        help="Which games to generate episodes for",
    )
    parser.add_argument(
        "--episodes-per-game",
        type=int,
        default=DEFAULT_EPISODES_PER_GAME,
        help="Number of episodes to generate per game",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Max concurrent episodes",
    )
    parser.add_argument(
        "--env-url",
        default="http://localhost:8000",
        help="Environment server base URL",
    )
    parser.add_argument(
        "--output-dir",
        default="output/sft_dataset",
        help="Output directory for checkpoint and dataset",
    )
    parser.add_argument(
        "--model",
        default=CLAUDE_MODEL,
        help="Claude model to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Claude sampling temperature",
    )
    parser.add_argument(
        "--haiku-ratio",
        type=float,
        default=0.8,
        help="Fraction of episodes to play with Haiku (0.0=all Sonnet, 1.0=all Haiku)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 episode per game for testing",
    )
    return parser.parse_args()
