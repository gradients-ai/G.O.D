"""Single-player OpenSpiel evaluation helpers.

These reuse the same LLMBot tool-calling harness as PvP, but run one model
through independent single-player episodes and average OpenSpiel returns.
"""

import logging
import random
import time

from pydantic import BaseModel
from pydantic import Field

from core.constants import ENVIRONMENT_CONFIGS
from core.constants import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatFn
from core.models.pvp_models import MemoryArea
from core.pvp import constants as cst
from core.pvp.bot import LLMBot
from core.pvp.game_eval import _evaluate_game_with_timeout
from core.pvp.game_eval import agent_class_for
from core.pvp.game_eval import config_id_for_seed
from core.pvp.memory import SlotMemory
from core.pvp.tokenizer_counter import load_token_counter


logger = logging.getLogger(__name__)


class IndividualOpenSpielResult(BaseModel):
    scores: list[float] = Field(default_factory=list)

    @property
    def num_games(self) -> int:
        return len(self.scores)

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)


def run_individual_open_spiel_eval(
    env_name: EnvironmentName,
    chat_fn: ChatFn,
    config: ChatCompletionConfig,
    num_games: int,
    base_seed: int = 0,
    time_budget_seconds: float | None = None,
) -> IndividualOpenSpielResult:
    """Run num_games single-player OpenSpiel episodes and return raw scores."""
    agent = agent_class_for(env_name)()
    env_config = ENVIRONMENT_CONFIGS[env_name]
    counter = load_token_counter(config.tokenizer_repo or config.inference_model)
    long_term = SlotMemory(cst.PVP_LONGTERM_MEM_SLOTS, cst.PVP_LONGTERM_SLOT_TOKENS, counter)

    seed_rng = random.Random(base_seed)
    result = IndividualOpenSpielResult()
    started = time.monotonic()

    for i in range(num_games):
        if time_budget_seconds is not None and time.monotonic() - started >= time_budget_seconds:
            logger.warning(
                "%s individual time budget (%.0fs) exhausted after %d/%d games; returning partial scores",
                env_name.value, time_budget_seconds, result.num_games, num_games,
            )
            break

        seed = seed_rng.randint(1, cst.PVP_SEED_RANGE_MAX)
        config_id = config_id_for_seed(seed, env_config)
        game = agent.load_game(agent.generate_params(config_id))
        if game.num_players() != 1:
            raise ValueError(f"{env_name.value} is not a single-player OpenSpiel game")

        working = SlotMemory(cst.PVP_WORKING_MEM_SLOTS, cst.PVP_WORKING_SLOT_TOKENS, counter)
        bot = LLMBot(
            game=game,
            player_id=0,
            chat_fn=chat_fn,
            config=config,
            agent=agent,
            memories={MemoryArea.WORKING: working, MemoryArea.LONG_TERM: long_term},
        )

        state = game.new_initial_state()
        agent.setup_initial_state(state, seed)
        evaluation = _evaluate_game_with_timeout(state, [bot], seed)
        result.scores.append(float(evaluation.returns[0]))

        if (i + 1) % 10 == 0:
            logger.info(
                "%s individual: %d/%d games mean=%.3f",
                env_name.value, i + 1, num_games, result.mean_score,
            )

    logger.info(
        "%s individual complete: %d games mean=%.3f",
        env_name.value, result.num_games, result.mean_score,
    )
    return result
