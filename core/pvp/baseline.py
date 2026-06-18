"""In-harness MCTS baseline.

Play the model (the tool-calling LLMBot — identical format to eval) against a
pyspiel MCTSBot and score the result. This gives a baseline game-playing number
measured in the SAME interaction format as the PvP eval, with no external
MCTS-API server: the opponent is built in-process from pyspiel.

Used by model prep to gauge how well a base model plays before training.
"""

import logging
import random
import time

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from pydantic import BaseModel

from core.constants.environments import ENVIRONMENT_CONFIGS
from core.constants.environments import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatFn
from core.models.pvp_models import GameOutcome
from core.models.pvp_models import GameScoringContext
from core.models.pvp_models import MemoryArea
from core.pvp import constants as cst
from core.pvp.bot import LLMBot
from core.pvp.game_eval import _AGENT_REGISTRY
from core.pvp.game_eval import _evaluate_game_with_timeout
from core.pvp.game_eval import config_id_for_seed
from core.pvp.memory import SlotMemory
from core.pvp.scoring import determine_outcome
from core.pvp.tokenizer_counter import load_token_counter


logger = logging.getLogger(__name__)

_MCTS_UCT_C = 2.0
_DEFAULT_MCTS_SIMULATIONS = 50


class MctsBaselineResult(BaseModel):
    """Outcome counts for a model's baseline games against MCTS."""

    wins: int = 0
    draws: int = 0
    losses: int = 0
    num_games: int = 0

    @property
    def mean_score(self) -> float:
        """Win=1, draw=0.5, loss=0, averaged — a [0, 1] baseline score."""
        if self.num_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.num_games


def supports_in_harness_baseline(env_name: EnvironmentName) -> bool:
    """True when env_name has a pyspiel agent and can be baselined in-process."""
    return env_name in _AGENT_REGISTRY


def _mcts_simulations_for(env_name: EnvironmentName) -> int:
    extra = ENVIRONMENT_CONFIGS[env_name].eval_payload_extra or {}
    return int(extra.get("mcts_max_simulations", _DEFAULT_MCTS_SIMULATIONS))


def _make_mcts_bot(game: pyspiel.Game, simulations: int, seed: int) -> mcts.MCTSBot:
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=np.random.RandomState(seed))
    return mcts.MCTSBot(
        game,
        uct_c=_MCTS_UCT_C,
        max_simulations=simulations,
        evaluator=evaluator,
        random_state=np.random.RandomState(seed),
    )


def run_mcts_baseline(
    env_name: EnvironmentName,
    chat_fn: ChatFn,
    config: ChatCompletionConfig,
    num_games: int,
    mcts_simulations: int | None = None,
    base_seed: int = 0,
    time_budget_seconds: float | None = None,
) -> MctsBaselineResult:
    """Play num_games of env_name as the model (LLMBot) vs MCTS; return outcome counts.

    The model alternates seats for fairness and carries one long-term memory across
    the games (it builds a read on the MCTS opponent, exactly as in a real matchup).
    A model-side forfeit (timeout, repeated illegal moves, context overflow) scores
    as a loss, mirroring eval.

    time_budget_seconds bounds wall-clock: no new game starts past the deadline
    and the partial tally is returned — a baseline over fewer games beats blowing
    the caller's dispatch timeout. The turn alarm bounds a single turn, not a
    game, so slow models need this outer guard.
    """
    agent = _AGENT_REGISTRY[env_name]()
    env_config = ENVIRONMENT_CONFIGS[env_name]
    simulations = mcts_simulations if mcts_simulations is not None else _mcts_simulations_for(env_name)
    counter = load_token_counter(config.tokenizer_repo or config.inference_model)
    long_term = SlotMemory(cst.PVP_LONGTERM_MEM_SLOTS, cst.PVP_LONGTERM_SLOT_TOKENS, counter)

    seed_rng = random.Random(base_seed)
    result = MctsBaselineResult()
    started = time.monotonic()

    for i in range(num_games):
        if time_budget_seconds is not None and time.monotonic() - started >= time_budget_seconds:
            logger.warning(
                "%s baseline time budget (%.0fs) exhausted after %d/%d games; returning partial tally",
                env_name.value, time_budget_seconds, result.num_games, num_games,
            )
            break
        seed = seed_rng.randint(1, cst.PVP_SEED_RANGE_MAX)
        config_id = config_id_for_seed(seed, env_config)
        game = agent.load_game(agent.generate_params(config_id))
        game_type = game.get_type()

        model_seat = i % 2
        mcts_seat = 1 - model_seat
        working = SlotMemory(cst.PVP_WORKING_MEM_SLOTS, cst.PVP_WORKING_SLOT_TOKENS, counter)
        model_bot = LLMBot(
            game=game,
            player_id=model_seat,
            chat_fn=chat_fn,
            config=config,
            agent=agent,
            memories={MemoryArea.WORKING: working, MemoryArea.LONG_TERM: long_term},
        )
        bots: list = [None, None]
        bots[model_seat] = model_bot
        bots[mcts_seat] = _make_mcts_bot(game, simulations, seed + mcts_seat)

        state = game.new_initial_state()
        agent.setup_initial_state(state, seed)
        evaluation = _evaluate_game_with_timeout(state, bots, seed)
        outcome = determine_outcome(
            GameScoringContext(
                returns=evaluation.returns,
                player_id=model_seat,
                is_zero_sum=game_type.utility == pyspiel.GameType.Utility.ZERO_SUM,
                min_utility=game.min_utility(),
                max_utility=game.max_utility(),
            )
        )
        if outcome == GameOutcome.WIN:
            result.wins += 1
        elif outcome == GameOutcome.LOSS:
            result.losses += 1
        else:
            result.draws += 1
        result.num_games += 1

        # Mirror game_runner: a forfeiting bot is broken/slow, so reflection
        # would just hit the same wall — skip it.
        if evaluation.forfeiting_player_id != model_seat:
            model_bot.reflect(state, outcome)

    logger.info(
        "%s MCTS baseline: %d games, %d-%d-%d (W-D-L), score=%.3f",
        env_name.value, result.num_games, result.wins, result.draws, result.losses, result.mean_score,
    )
    return result
