"""PvP game runner: plays head-to-head games and tallies results.

Drives OpenSpiel's evaluate_bots with two LLMBots, one per model.
Each seed is played twice with swapped positions for fairness.
Per-turn timeouts in LLMBot.step() ensure a slow/broken model
forfeits rather than dragging the opponent into a draw.
"""

import functools
import logging
import random
from typing import NamedTuple

import numpy as np
import openai
import pyspiel
from open_spiel.python.algorithms import evaluate_bots

from core.constants import ENVIRONMENT_CONFIGS
from core.constants import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatFn
from core.models.pvp_models import GameInstance
from core.models.pvp_models import GameOutcome
from core.models.pvp_models import GameScoringContext
from core.models.pvp_models import PvPEnvironmentResult
from core.models.pvp_models import PvPMatchupConfig
from validator.core import constants as vcst
from validator.evaluation.pvp.agents import BaseGameAgent
from validator.evaluation.pvp.agents import GinRummyAgent
from validator.evaluation.pvp.agents import LeducPokerAgent
from validator.evaluation.pvp.agents import LiarsDiceAgent
from validator.evaluation.pvp.bot import ContextOverflowError
from validator.evaluation.pvp.bot import EmptyLegalActionsError
from validator.evaluation.pvp.bot import InvalidActionForfeitError
from validator.evaluation.pvp.bot import LLMBot
from validator.evaluation.pvp.bot import TurnTimeoutError
from validator.evaluation.pvp.chat import chat_completion
from validator.evaluation.pvp.chat import create_client
from validator.evaluation.pvp.scoring import determine_outcome


logger = logging.getLogger(__name__)


def _forfeit_returns(state: pyspiel.State, forfeiting_player: int) -> list[float]:
    """Build returns where the forfeiting player gets min utility, opponent gets max."""
    game = state.get_game()
    min_util = game.min_utility()
    max_util = game.max_utility()
    returns = [max_util] * state.num_players()
    returns[forfeiting_player] = min_util
    return returns


class Player(NamedTuple):
    """A configured player: reusable client, config, and bound chat function."""

    client: openai.OpenAI
    config: ChatCompletionConfig
    chat_fn: ChatFn


def create_player(config: ChatCompletionConfig) -> Player:
    """Create a Player with a client bound to the config. Enforces client/config invariant."""
    client = create_client(config)
    bound_chat: ChatFn = functools.partial(chat_completion, client)
    return Player(client=client, config=config, chat_fn=bound_chat)


_AGENT_REGISTRY: dict[EnvironmentName, type[BaseGameAgent]] = {
    EnvironmentName.LIARS_DICE: LiarsDiceAgent,
    EnvironmentName.LEDUC_POKER: LeducPokerAgent,
    EnvironmentName.GIN_RUMMY: GinRummyAgent,
}


def run_matchup(
    env_name: EnvironmentName,
    matchup_config: PvPMatchupConfig,
    player_a: Player,
    player_b: Player,
    base_seed: int,
) -> PvPEnvironmentResult:
    """Run a full PvP matchup for one environment.

    Plays matchup_config.num_games seeds, each twice (swapped positions).
    """
    agent = _AGENT_REGISTRY[env_name]()
    instances = _build_instances(env_name, agent, matchup_config.num_games, base_seed)
    return _execute_matchup(env_name, instances, player_a, player_b, agent)


def _build_instances(
    env_name: EnvironmentName,
    agent: BaseGameAgent,
    num_games: int,
    base_seed: int,
) -> list[GameInstance]:
    """Generate paired GameInstances (original + position-swapped) for each seed."""
    env_config = ENVIRONMENT_CONFIGS[env_name]
    seed_rng = random.Random(base_seed)
    instances: list[GameInstance] = []

    for _ in range(num_games):
        seed = seed_rng.randint(1, vcst.PVP_SEED_RANGE_MAX)
        task_rng = random.Random(seed)
        task_id = task_rng.randint(env_config.task_id_min + 1, env_config.task_id_max)
        config_id = task_id % vcst.PVP_CONFIG_ID_DIVISOR
        game_params = agent.generate_params(config_id)

        game = pyspiel.load_game(agent.game_name, game_params)
        game_type = game.get_type()

        base = GameInstance(
            game_name=agent.game_name,
            game_params=game_params,
            model_a_player_id=0,
            seed=seed,
            is_zero_sum=game_type.utility == pyspiel.GameType.Utility.ZERO_SUM,
            min_utility=game.min_utility(),
            max_utility=game.max_utility(),
        )
        swapped = base.model_copy(update={"model_a_player_id": 1})

        instances.append(base)
        instances.append(swapped)

    return instances


def _check_early_forfeit(
    result: PvPEnvironmentResult,
    consec_a_losses: int,
    consec_b_losses: int,
    remaining: int,
    env_name: str,
    games_played: int,
) -> bool:
    """Award remaining games to the dominant player if the other lost too many in a row.

    Uses a tighter threshold for the opening games and a looser one after that.
    If a model loses the first N games straight, it's clearly outmatched — forfeit early.
    After the opening window, require 2N consecutive losses before forfeiting, giving
    models more chance to recover from a bad streak mid-game.

    Note: the threshold switches at games_played > early_limit, so a model that loses
    games 1-N forfeits at game N, but a streak starting after game N needs 2N in a row.
    """
    early_limit = vcst.PVP_CONSECUTIVE_LOSS_FORFEIT
    late_limit = early_limit * 2
    limit = early_limit if games_played <= early_limit else late_limit

    if consec_a_losses >= limit:
        loser, winner_attr = "a", "model_b_wins"
    elif consec_b_losses >= limit:
        loser, winner_attr = "b", "model_a_wins"
    else:
        return False

    logger.info(
        "%s: model_%s lost %d in a row (after %d games) — forfeiting %d remaining",
        env_name,
        loser,
        limit,
        games_played,
        remaining,
    )
    setattr(result, winner_attr, getattr(result, winner_attr) + remaining)
    result.total_games += remaining
    return True


def _execute_matchup(
    env_name: EnvironmentName,
    instances: list[GameInstance],
    player_a: Player,
    player_b: Player,
    agent: BaseGameAgent,
) -> PvPEnvironmentResult:
    """Play all game instances and tally results."""
    play = functools.partial(_play_game, player_a=player_a, player_b=player_b, agent=agent)

    result = PvPEnvironmentResult()
    consec_a_losses = 0
    consec_b_losses = 0

    for i, instance in enumerate(instances):
        outcome = play(instance)
        _tally(result, outcome)

        if outcome == GameOutcome.LOSS:
            consec_a_losses += 1
            consec_b_losses = 0
        elif outcome == GameOutcome.WIN:
            consec_b_losses += 1
            consec_a_losses = 0
        else:
            consec_a_losses = 0
            consec_b_losses = 0

        remaining = len(instances) - i - 1
        if _check_early_forfeit(result, consec_a_losses, consec_b_losses, remaining, env_name.value, i + 1):
            break

        if (i + 1) % vcst.PVP_LOG_INTERVAL_GAMES == 0:
            logger.info(
                "%s: %d/%d games, a=%d b=%d draws=%d",
                env_name.value, i + 1, len(instances),
                result.model_a_wins, result.model_b_wins, result.draws,
            )

    logger.info(
        "%s complete: %d games, a=%d b=%d draws=%d",
        env_name.value, result.total_games,
        result.model_a_wins, result.model_b_wins, result.draws,
    )
    return result


def _play_game(
    instance: GameInstance,
    player_a: Player,
    player_b: Player,
    agent: BaseGameAgent,
) -> GameOutcome:
    """Play a single game with timeout and return outcome from model_a's perspective."""
    game = pyspiel.load_game(instance.game_name, instance.game_params)
    model_b_player_id = 1 - instance.model_a_player_id

    bot_a = LLMBot(
        game=game,
        player_id=instance.model_a_player_id,
        chat_fn=player_a.chat_fn,
        config=player_a.config,
        agent=agent,
        rng_seed=instance.seed + instance.model_a_player_id,
    )
    bot_b = LLMBot(
        game=game,
        player_id=model_b_player_id,
        chat_fn=player_b.chat_fn,
        config=player_b.config,
        agent=agent,
        rng_seed=instance.seed + model_b_player_id,
    )

    bots = [None, None]
    bots[instance.model_a_player_id] = bot_a
    bots[model_b_player_id] = bot_b

    state = game.new_initial_state()
    returns = _evaluate_with_timeout(state, bots, instance.seed)

    scoring = GameScoringContext(
        returns=list(returns),
        player_id=instance.model_a_player_id,
        is_zero_sum=instance.is_zero_sum,
        min_utility=instance.min_utility,
        max_utility=instance.max_utility,
    )
    return determine_outcome(scoring)


def _evaluate_with_timeout(
    state: pyspiel.State,
    bots: list[LLMBot | None],
    seed: int,
) -> list[float]:
    """Run evaluate_bots, catching bot-level forfeits.

    Per-turn timeouts are enforced inside LLMBot.step() via SIGALRM.
    Timeout, context overflow, and invalid-action strikeouts propagate up
    through evaluate_bots and are caught here as forfeits.
    """
    try:
        returns = evaluate_bots.evaluate_bots(state, bots, np.random.RandomState(seed))
        return list(returns)
    except TurnTimeoutError as exc:
        logger.warning(
            "Player %d timed out on turn — opponent wins by forfeit",
            exc.player_id,
        )
        return _forfeit_returns(state, exc.player_id)
    except ContextOverflowError as exc:
        logger.warning(
            "Player %d exceeded context length — opponent wins by forfeit",
            exc.player_id,
        )
        return _forfeit_returns(state, exc.player_id)
    except InvalidActionForfeitError as exc:
        logger.warning(
            "Player %d failed to produce valid actions %d times — opponent wins by forfeit",
            exc.player_id,
            exc.invalid_action_failures,
        )
        return _forfeit_returns(state, exc.player_id)
    except EmptyLegalActionsError:
        logger.warning("Game stuck with no legal actions — scoring as draw")
        return [0.0] * state.num_players()


def _tally(result: PvPEnvironmentResult, outcome: GameOutcome) -> None:
    result.total_games += 1
    if outcome == GameOutcome.WIN:
        result.model_a_wins += 1
    elif outcome == GameOutcome.LOSS:
        result.model_b_wins += 1
    else:
        result.draws += 1
