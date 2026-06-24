"""Shared game-play primitives for the PvP harness.

The agent registry and the timeout/forfeit-aware evaluate_bots wrapper, used by
both the eval matchup runner (validator) and the MCTS baseline (core). Lives in
core so the model-prep image (core/ only) can run it.
"""

import logging
import random
from typing import NamedTuple

import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots

from core.constants import ENVIRONMENT_CONFIGS
from core.constants import EnvironmentConfig
from core.constants import EnvironmentName
from core.constants import EvalType
from core.pvp import constants as cst
from core.pvp.agents import BaseGameAgent
from core.pvp.agents import ClobberAgent
from core.pvp.agents import GinRummyAgent
from core.pvp.agents import GoofspielAgent
from core.pvp.agents import LeducPokerAgent
from core.pvp.agents import LiarsDiceAgent
from core.pvp.agents import OthelloAgent
from core.pvp.bot import ContextOverflowError
from core.pvp.bot import EmptyLegalActionsError
from core.pvp.bot import InvalidActionForfeitError
from core.pvp.bot import LLMBot
from core.pvp.bot import TurnTimeoutError


logger = logging.getLogger(__name__)


_AGENT_REGISTRY: dict[EnvironmentName, type[BaseGameAgent]] = {
    EnvironmentName.LIARS_DICE: LiarsDiceAgent,
    EnvironmentName.LEDUC_POKER: LeducPokerAgent,
    EnvironmentName.GIN_RUMMY: GinRummyAgent,
    EnvironmentName.OTHELLO: OthelloAgent,
    EnvironmentName.CLOBBER: ClobberAgent,
    EnvironmentName.GOOFSPIEL: GoofspielAgent,
}

# Every PVP env must have an agent: image_manager skips env sidecars for PVP
# envs on the assumption that this registry covers them, and a gap would ship
# silently-empty baseline stats. Fail at import (in the model-prep and eval
# containers) instead.
_pvp_envs = {name for name, cfg in ENVIRONMENT_CONFIGS.items() if cfg.eval_type == EvalType.PVP}
if set(_AGENT_REGISTRY) != _pvp_envs:
    raise RuntimeError(
        f"PvP env/agent registry drift: PVP envs without an agent: "
        f"{sorted(e.value for e in _pvp_envs - set(_AGENT_REGISTRY))}; "
        f"agents without a PVP env config: "
        f"{sorted(e.value for e in set(_AGENT_REGISTRY) - _pvp_envs)}"
    )


def config_id_for_seed(seed: int, env_config: EnvironmentConfig) -> int:
    """Deterministic seed -> game-config id (selects the game-variant params).

    Shared by eval and the MCTS baseline so both sample game variants from the
    same distribution (seeds themselves don't need to match across the two).
    """
    task_rng = random.Random(seed)
    task_id = task_rng.randint(env_config.task_id_min, env_config.task_id_max)
    return task_id % cst.PVP_CONFIG_ID_DIVISOR


class GameEvaluation(NamedTuple):
    """Raw game returns plus the player ID that forfeited, when any."""

    returns: list[float]
    forfeiting_player_id: int | None = None


def _forfeit_returns(state: pyspiel.State, forfeiting_player: int) -> list[float]:
    """Build returns where the forfeiting player gets min utility, opponent gets max."""
    game = state.get_game()
    min_util = game.min_utility()
    max_util = game.max_utility()
    returns = [max_util] * state.num_players()
    returns[forfeiting_player] = min_util
    return returns


def _evaluate_game_with_timeout(
    state: pyspiel.State,
    bots: list[LLMBot | None],
    seed: int,
) -> GameEvaluation:
    """Run evaluate_bots, catching bot-level forfeits.

    Per-turn timeouts are enforced inside LLMBot.step() via SIGALRM. Timeout,
    context overflow, and invalid-action strikeouts propagate up through
    evaluate_bots and are caught here as forfeits.
    """
    try:
        returns = evaluate_bots.evaluate_bots(state, bots, np.random.RandomState(seed))
        return GameEvaluation(returns=list(returns))
    except TurnTimeoutError as exc:
        logger.warning("Player %d timed out on turn — opponent wins by forfeit", exc.player_id)
        return GameEvaluation(returns=_forfeit_returns(state, exc.player_id), forfeiting_player_id=exc.player_id)
    except ContextOverflowError as exc:
        logger.warning("Player %d exceeded context length — opponent wins by forfeit", exc.player_id)
        return GameEvaluation(returns=_forfeit_returns(state, exc.player_id), forfeiting_player_id=exc.player_id)
    except InvalidActionForfeitError as exc:
        logger.warning(
            "Player %d did not commit a legal move this turn — opponent wins by forfeit",
            exc.player_id,
        )
        return GameEvaluation(returns=_forfeit_returns(state, exc.player_id), forfeiting_player_id=exc.player_id)
    except EmptyLegalActionsError:
        logger.warning("Game stuck with no legal actions — scoring as draw")
        return GameEvaluation(returns=[0.0] * state.num_players())


def _evaluate_with_timeout(state: pyspiel.State, bots: list[LLMBot | None], seed: int) -> list[float]:
    """Run evaluate_bots and return only game returns (forfeit attribution dropped)."""
    return _evaluate_game_with_timeout(state, bots, seed).returns
