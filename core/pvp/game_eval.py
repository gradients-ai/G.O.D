"""Shared game-play primitives for the PvP harness.

The agent registry and the timeout/forfeit-aware evaluate_bots wrapper, used by
both the eval matchup runner (validator) and the MCTS baseline (core). Lives in
core so the model-prep image (core/ only) can run it.
"""

import logging
from typing import NamedTuple

import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots

from core.constants import EnvironmentName
from core.pvp.agents import BaseGameAgent
from core.pvp.agents import GinRummyAgent
from core.pvp.agents import LeducPokerAgent
from core.pvp.agents import LiarsDiceAgent
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
}


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
