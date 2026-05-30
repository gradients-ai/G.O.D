"""LLM Bot implementation for OpenSpiel PvP evaluation.

Wraps an LLM inference endpoint as a pyspiel.Bot, maintaining
conversation history and parsing actions from model responses.
Accepts a ChatFn protocol for testability.
"""

import logging
import re
import signal

import numpy as np
import openai
import pyspiel

from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatFn
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatRole
from validator.core import constants as vcst
from validator.evaluation.pvp.agents import BaseGameAgent


logger = logging.getLogger(__name__)


class TurnTimeoutError(Exception):
    """Raised when a bot's step() exceeds the per-turn time limit."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        super().__init__(f"Player {player_id} exceeded {vcst.PVP_TURN_TIMEOUT_SECONDS}s turn timeout")


class ContextOverflowError(Exception):
    """Raised when a bot's input exceeds the model's context length."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        super().__init__(f"Player {player_id} exceeded model context length")


class EmptyLegalActionsError(Exception):
    """Raised when the game state has no legal actions for the current player."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        super().__init__(f"No legal actions for player {player_id}")


class InvalidActionForfeitError(Exception):
    """Raised when a bot repeatedly fails to produce legal actions in one game."""

    def __init__(self, player_id: int, invalid_action_failures: int):
        self.player_id = player_id
        self.invalid_action_failures = invalid_action_failures
        super().__init__(
            f"Player {player_id} failed to produce valid actions {invalid_action_failures} times and forfeits"
        )


class LLMBot(pyspiel.Bot):
    """OpenSpiel Bot backed by an LLM via injectable chat function.

    Maintains full conversation history per game. On each step(),
    generates a user prompt from the game state, calls the chat function,
    and parses an action ID from the response.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        player_id: int,
        chat_fn: ChatFn,
        config: ChatCompletionConfig,
        agent: BaseGameAgent,
        rng_seed: int,
    ):
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._chat_fn = chat_fn
        self._config = config
        self._agent = agent
        self._rng = np.random.RandomState(rng_seed)
        self._conversation: list[ChatMessage] = []
        self._system_prompt_set = False
        self._invalid_action_failures = 0

    def restart_at(self, state: pyspiel.State) -> None:
        self._conversation.clear()
        self._system_prompt_set = False
        self._invalid_action_failures = 0

    def inform_action(self, state: pyspiel.State, player_id: int, action: int) -> None:
        pass

    def step(self, state: pyspiel.State) -> int:
        """Choose an action by querying the LLM.

        Called by evaluate_bots during game play. Enforces a per-turn
        timeout — if the turn (including all retries) exceeds the limit,
        TurnTimeoutError propagates up to forfeit the game.
        """

        def _turn_timeout_handler(signum: int, frame: object) -> None:
            raise TurnTimeoutError(self._player_id)

        prev_handler = signal.signal(signal.SIGALRM, _turn_timeout_handler)
        signal.alarm(vcst.PVP_TURN_TIMEOUT_SECONDS)

        try:
            return self._step_inner(state)
        except TurnTimeoutError:
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)

    def _step_inner(self, state: pyspiel.State) -> int:
        """Core step logic: prompt the LLM, parse, retry, fallback."""
        if not self._system_prompt_set:
            system_prompt = self._agent.generate_system_prompt()
            self._conversation.append(ChatMessage(role=ChatRole.SYSTEM, content=system_prompt))
            self._system_prompt_set = True

        current = state.current_player()
        legal_actions = state.legal_actions(current)

        if current != self._player_id:
            logger.error(
                "Player ID mismatch: bot._player_id=%d current_player=%d "
                "legal_actions(current)=%s legal_actions(self)=%s "
                "is_terminal=%s is_chance=%s",
                self._player_id, current,
                state.legal_actions(current), state.legal_actions(self._player_id),
                state.is_terminal(), state.is_chance_node(),
            )

        if not legal_actions:
            logger.error(
                "Empty legal_actions: player_id=%d current_player=%d "
                "is_terminal=%s is_chance=%s game=%s",
                self._player_id, current,
                state.is_terminal(), state.is_chance_node(),
                self._game.get_type().short_name,
            )
            raise EmptyLegalActionsError(self._player_id)

        user_prompt = self._agent.generate_user_prompt(state, self._player_id, legal_actions)
        self._conversation.append(ChatMessage(role=ChatRole.USER, content=user_prompt))

        for attempt in range(vcst.PVP_BOT_MAX_PARSING_RETRIES + 1):
            try:
                result = self._chat_fn(self._config, self._conversation)
            except openai.BadRequestError as exc:
                if "context length" in str(exc).lower():
                    raise ContextOverflowError(self._player_id) from exc
                raise

            response_text = result.content or ""
            self._conversation.append(ChatMessage(role=ChatRole.ASSISTANT, content=response_text))

            if response_text:
                parsed_action = _parse_action(response_text, legal_actions)
                if parsed_action is not None:
                    return parsed_action

            retry_msg = (
                f"Invalid response. Respond with ONLY the action ID number. "
                f"Attempt {attempt + 1}/{vcst.PVP_BOT_MAX_PARSING_RETRIES + 1}."
            )
            self._conversation.append(ChatMessage(role=ChatRole.USER, content=retry_msg))

        self._invalid_action_failures += 1
        attempts = vcst.PVP_BOT_MAX_PARSING_RETRIES + 1
        if self._invalid_action_failures >= vcst.PVP_BOT_INVALID_ACTION_FORFEIT_THRESHOLD:
            logger.warning(
                "LLM failed to produce valid action after %d attempts; player %d reached %d/%d failures and forfeits",
                attempts,
                self._player_id,
                self._invalid_action_failures,
                vcst.PVP_BOT_INVALID_ACTION_FORFEIT_THRESHOLD,
            )
            raise InvalidActionForfeitError(self._player_id, self._invalid_action_failures)

        fallback = int(self._rng.choice(legal_actions))
        logger.warning(
            "LLM failed to produce valid action after %d attempts (%d/%d failures this game), falling back to %d",
            attempts,
            self._invalid_action_failures,
            vcst.PVP_BOT_INVALID_ACTION_FORFEIT_THRESHOLD,
            fallback,
        )
        self._conversation.append(ChatMessage(role=ChatRole.ASSISTANT, content=str(fallback)))
        return fallback


def _parse_action(response: str, legal_actions: list[int]) -> int | None:
    """Parse an action ID from LLM response text.

    Strategies (in priority order):
    1. Response is purely a number
    2. Last number in text that is a legal action
    """
    cleaned = response.strip()
    legal_set = set(legal_actions)

    match = re.match(r"^\s*(\d+)\s*$", cleaned)
    if match:
        action = int(match.group(1))
        if action in legal_set:
            return action

    numbers = [int(m) for m in re.findall(r"\b(\d+)\b", cleaned)]
    for num in reversed(numbers):
        if num in legal_set:
            return num

    return None
