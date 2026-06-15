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

import openai
import pyspiel

from core.constants import ENVIRONMENT_CONFIGS
from core.constants import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatFn
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatRole
from core.models.pvp_models import GameInstance
from core.models.pvp_models import GameOutcome
from core.models.pvp_models import GameScoringContext
from core.models.pvp_models import MemoryArea
from core.models.pvp_models import PvPEnvironmentResult
from core.models.pvp_models import PvPMatchupConfig
from core.pvp.agents import BaseGameAgent
from core.pvp.bot import LLMBot
from core.pvp.chat import chat_completion
from core.pvp.chat import create_client
from core.pvp.game_eval import _AGENT_REGISTRY
from core.pvp.game_eval import _evaluate_game_with_timeout
from core.pvp.game_eval import config_id_for_seed
from core.pvp.memory import SlotMemory
from core.pvp.memory import TokenCounter
from core.pvp.scoring import determine_outcome
from core.pvp.tokenizer_counter import load_token_counter
from validator.core import constants as vcst


logger = logging.getLogger(__name__)


class Player(NamedTuple):
    """A configured player: reusable client, config, and bound chat function."""

    client: openai.OpenAI
    config: ChatCompletionConfig
    chat_fn: ChatFn


class PlayedGame(NamedTuple):
    """Scored game outcome plus the model label that forfeited, when any."""

    outcome: GameOutcome
    forfeiting_model: str | None = None


def create_player(config: ChatCompletionConfig) -> Player:
    """Create a Player with a client bound to the config. Enforces client/config invariant."""
    client = create_client(config)
    bound_chat: ChatFn = functools.partial(chat_completion, client)
    return Player(client=client, config=config, chat_fn=bound_chat)


def warmup_player(player: Player) -> None:
    """One throwaway completion so the first scored turn doesn't absorb SGLang's
    cold-start (CUDA-graph capture), which can otherwise blow the turn timeout."""
    try:
        chat_completion(player.client, player.config, [ChatMessage(role=ChatRole.USER, content="warmup")])
    except Exception as exc:
        logger.warning("Warmup failed for %s (ignored): %s", player.config.inference_model, exc)


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
        game_params = agent.generate_params(config_id_for_seed(seed, env_config))

        game = agent.load_game(game_params)
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


def _check_episode_forfeit_limit(
    result: PvPEnvironmentResult,
    model_a_forfeits: int,
    model_b_forfeits: int,
    remaining: int,
    env_name: str,
    games_played: int,
) -> bool:
    """Award remaining games when one model forfeits too many episodes in a matchup."""
    if model_a_forfeits >= vcst.PVP_EPISODE_FORFEIT_THRESHOLD:
        loser, forfeits, winner_attr = "a", model_a_forfeits, "model_b_wins"
    elif model_b_forfeits >= vcst.PVP_EPISODE_FORFEIT_THRESHOLD:
        loser, forfeits, winner_attr = "b", model_b_forfeits, "model_a_wins"
    else:
        return False

    logger.info(
        "%s: model_%s forfeited %d episodes (after %d games) — forfeiting %d remaining",
        env_name,
        loser,
        forfeits,
        games_played,
        remaining,
    )
    setattr(result, winner_attr, getattr(result, winner_attr) + remaining)
    result.total_games += remaining
    return True


def _new_long_term_memory(counter: TokenCounter) -> SlotMemory:
    return SlotMemory(vcst.PVP_LONGTERM_MEM_SLOTS, vcst.PVP_LONGTERM_SLOT_TOKENS, counter)


def _game_memories(long_term: SlotMemory, counter: TokenCounter) -> dict[MemoryArea, SlotMemory]:
    """Fresh working memory for this game; long_term carried in from the matchup."""
    return {
        MemoryArea.WORKING: SlotMemory(vcst.PVP_WORKING_MEM_SLOTS, vcst.PVP_WORKING_SLOT_TOKENS, counter),
        MemoryArea.LONG_TERM: long_term,
    }


def _execute_matchup(
    env_name: EnvironmentName,
    instances: list[GameInstance],
    player_a: Player,
    player_b: Player,
    agent: BaseGameAgent,
) -> PvPEnvironmentResult:
    """Play all game instances and tally results."""
    # Per-player tokenizer so slot budgets are real model tokens (whitespace fallback).
    # Prefer tokenizer_repo over inference_model: a LoRA's inference_model is
    # 'base:lora' — not a loadable repo, which would degrade budgets to word counts.
    counter_a = load_token_counter(player_a.config.tokenizer_repo or player_a.config.inference_model)
    counter_b = load_token_counter(player_b.config.tokenizer_repo or player_b.config.inference_model)
    # One long-term memory per player, carried across every game of the matchup
    # so each side builds an opponent model over the series.
    long_term_a = _new_long_term_memory(counter_a)
    long_term_b = _new_long_term_memory(counter_b)
    play = functools.partial(
        _play_game,
        player_a=player_a,
        player_b=player_b,
        agent=agent,
        long_term_a=long_term_a,
        long_term_b=long_term_b,
        counter_a=counter_a,
        counter_b=counter_b,
    )

    result = PvPEnvironmentResult()
    consec_a_losses = 0
    consec_b_losses = 0
    model_a_forfeits = 0
    model_b_forfeits = 0

    for i, instance in enumerate(instances):
        played = play(instance)
        _tally(result, played.outcome)

        if played.outcome == GameOutcome.LOSS:
            consec_a_losses += 1
            consec_b_losses = 0
        elif played.outcome == GameOutcome.WIN:
            consec_b_losses += 1
            consec_a_losses = 0
        else:
            consec_a_losses = 0
            consec_b_losses = 0

        if played.forfeiting_model == "a":
            model_a_forfeits += 1
        elif played.forfeiting_model == "b":
            model_b_forfeits += 1

        remaining = len(instances) - i - 1
        if _check_episode_forfeit_limit(
            result,
            model_a_forfeits,
            model_b_forfeits,
            remaining,
            env_name.value,
            i + 1,
        ):
            break

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
    long_term_a: SlotMemory,
    long_term_b: SlotMemory,
    counter_a: TokenCounter,
    counter_b: TokenCounter,
) -> PlayedGame:
    """Play a single game with timeout and return outcome from model_a's perspective.

    Each bot gets fresh working memory plus the player's persistent long-term
    memory; after the game both surviving bots reflect to consolidate it.
    """
    game = agent.load_game(instance.game_params)
    model_b_player_id = 1 - instance.model_a_player_id

    bot_a = LLMBot(
        game=game,
        player_id=instance.model_a_player_id,
        chat_fn=player_a.chat_fn,
        config=player_a.config,
        agent=agent,
        memories=_game_memories(long_term_a, counter_a),
    )
    bot_b = LLMBot(
        game=game,
        player_id=model_b_player_id,
        chat_fn=player_b.chat_fn,
        config=player_b.config,
        agent=agent,
        memories=_game_memories(long_term_b, counter_b),
    )

    bots = [None, None]
    bots[instance.model_a_player_id] = bot_a
    bots[model_b_player_id] = bot_b

    state = game.new_initial_state()
    agent.setup_initial_state(state, instance.seed)
    evaluation = _evaluate_game_with_timeout(state, bots, instance.seed)

    def _outcome_for(player_id: int):
        return determine_outcome(
            GameScoringContext(
                returns=evaluation.returns,
                player_id=player_id,
                is_zero_sum=instance.is_zero_sum,
                min_utility=instance.min_utility,
                max_utility=instance.max_utility,
            )
        )

    outcome_a = _outcome_for(instance.model_a_player_id)
    outcome_b = _outcome_for(model_b_player_id)

    # Reflect to consolidate long-term memory — but skip a bot that forfeited
    # (its model is broken/slow, so reflection would just hit the same wall).
    forfeiting_pid = evaluation.forfeiting_player_id
    if forfeiting_pid != instance.model_a_player_id:
        bot_a.reflect(state, outcome_a)
    if forfeiting_pid != model_b_player_id:
        bot_b.reflect(state, outcome_b)

    forfeiting_model = None
    if forfeiting_pid is not None:
        forfeiting_model = "a" if forfeiting_pid == instance.model_a_player_id else "b"

    return PlayedGame(outcome=outcome_a, forfeiting_model=forfeiting_model)


def _tally(result: PvPEnvironmentResult, outcome: GameOutcome) -> None:
    result.total_games += 1
    if outcome == GameOutcome.WIN:
        result.model_a_wins += 1
    elif outcome == GameOutcome.LOSS:
        result.model_b_wins += 1
    else:
        result.draws += 1
