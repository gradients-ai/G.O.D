"""Tool-calling LLM bot for OpenSpiel PvP evaluation.

A turn is a SINGLE model call: the model is given the game state, its memory
slots (rendered in the prompt — there is no read tool, so one call sees
everything), and a set of tools. In that one response it may edit memory and
must call game_action to commit a legal move. No multi-step loop, no nudge,
no retry — if the response carries no legal game_action, the player forfeits
the turn. The conversation is rebuilt fresh every turn; the only state carried
across turns is the memory in SlotMemory.

Two memory areas live behind the tools: working memory (reset each game) and
long-term memory (persists across games against the same opponent). Robustness
is layered: a per-turn SIGALRM wall-clock timeout, action_id validated against
the legal set (the tool schema's enum is advisory — servers don't grammar-enforce
tool arguments under tool_choice="auto"), and bad memory ops that no-op rather
than crash.
"""

import logging
import signal
from contextlib import contextmanager

import openai
import pyspiel

from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatFn
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatRole
from core.models.pvp_models import GameOutcome
from core.models.pvp_models import MemoryArea
from core.models.pvp_models import MemoryConfig
from core.models.pvp_models import ToolCall
from core.models.pvp_models import ToolSchema
from core.pvp import constants as cst
from core.pvp import tools as tool_lib
from core.pvp.chat import ChatUnavailableError
from core.pvp.agents import BaseGameAgent
from core.pvp.memory import SlotMemory
from core.pvp.memory import WhitespaceTokenCounter


logger = logging.getLogger(__name__)

_TOOL_GUIDANCE = (
    "You get ONE response this turn. In it, optionally edit your memory notes, and "
    "then call game_action with a legal action id to commit your move. If you do not "
    "call game_action, you forfeit the turn — so always include it."
)
_REFLECTION_GUIDANCE = (
    "The game is over. Use the memory tools to update your long-term notes on this "
    "opponent for future games — keep durable, generalisable reads (their tendencies, "
    "your counter-strategy) and drop move-by-move detail. There is no move to make."
)


class TurnTimeoutError(Exception):
    """Raised when a bot's step() exceeds the per-turn time limit."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        super().__init__(f"Player {player_id} exceeded {cst.PVP_TURN_TIMEOUT_SECONDS}s turn timeout")


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
    """Raised when a bot's single turn response does not commit a legal move."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        super().__init__(f"Player {player_id} did not commit a legal action this turn and forfeits")


class ChatTimeoutForfeitError(Exception):
    """Raised when a player's model was too slow to respond (timed out after retries).

    Treated as a forfeit: an unresponsive model is the miner's fault, mirroring
    TurnTimeoutError. The opponent wins the game.
    """

    def __init__(self, player_id: int):
        self.player_id = player_id
        super().__init__(f"Player {player_id} inference timed out after retries and forfeits")


class ModelUnreachableError(Exception):
    """Raised when a player's inference server was unreachable (connection refused).

    This is an infra failure, NOT the miner's fault, so it must not forfeit the
    player. It propagates past the game-eval forfeit handlers up to the matchup
    runner, which waits for the server to recover and replays the game.
    """

    def __init__(self, player_id: int):
        self.player_id = player_id
        super().__init__(f"Player {player_id} inference server unreachable (infra failure)")


def default_memories() -> dict[MemoryArea, SlotMemory]:
    """Build the standard working + long-term memory areas from constants.

    Uses a whitespace token counter as a dependency-free default; production
    wiring injects a tokenizer-backed counter so budgets match real tokens.
    """
    counter = WhitespaceTokenCounter()
    return {
        MemoryArea.WORKING: SlotMemory(cst.PVP_WORKING_MEM_SLOTS, cst.PVP_WORKING_SLOT_TOKENS, counter),
        MemoryArea.LONG_TERM: SlotMemory(cst.PVP_LONGTERM_MEM_SLOTS, cst.PVP_LONGTERM_SLOT_TOKENS, counter),
    }


class LLMBot(pyspiel.Bot):
    """OpenSpiel Bot backed by an LLM that manages memory slots via tools."""

    def __init__(
        self,
        game: pyspiel.Game,
        player_id: int,
        chat_fn: ChatFn,
        config: ChatCompletionConfig,
        agent: BaseGameAgent,
        memories: dict[MemoryArea, SlotMemory] | None = None,
    ):
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._chat_fn = chat_fn
        # A turn generates memory edits + the move in one response; reflection
        # only writes notes. Both override the inbound config's legacy max_tokens.
        self._config = config.model_copy(update={"max_tokens": cst.PVP_TURN_MAX_TOKENS})
        self._reflection_config = config.model_copy(update={"max_tokens": cst.PVP_REFLECTION_MAX_TOKENS})
        self._agent = agent
        self._memories = memories if memories is not None else default_memories()
        self._memory_tools = tool_lib.build_memory_tools(
            {
                area: MemoryConfig(n_slots=mem.n_slots, slot_token_budget=mem.slot_token_budget)
                for area, mem in self._memories.items()
            }
        )

    def restart_at(self, state: pyspiel.State) -> None:
        """Reset per-game memory at the start of a new game; keep persistent areas."""
        for area, mem in self._memories.items():
            if not area.persists_across_games:
                mem.reset()

    def inform_action(self, state: pyspiel.State, player_id: int, action: int) -> None:
        pass

    @contextmanager
    def _wall_clock(self, seconds: int):
        """Bound a block to `seconds` of wall-clock; on overshoot raise TurnTimeoutError.

        SIGALRM-based, so it interrupts a blocking model call. Main-thread only.
        """

        def _handler(signum: int, frame: object) -> None:
            raise TurnTimeoutError(self._player_id)

        prev_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)

    def step(self, state: pyspiel.State) -> int:
        """Run one turn (a single model call) under a wall-clock timeout."""
        with self._wall_clock(cst.PVP_TURN_TIMEOUT_SECONDS):
            return self._run_turn(state)

    def reflect(self, state: pyspiel.State, outcome: GameOutcome) -> None:
        """Single-shot, best-effort memory consolidation after a game ends.

        The model is shown the result and may call memory tools (no game_action)
        to update its notes. Bounded by its own wall-clock timeout; all failures
        (including timeout) are swallowed — the game is already decided, so a
        flaky or slow reflection must never affect the match.
        """
        try:
            with self._wall_clock(cst.PVP_REFLECTION_TIMEOUT_SECONDS):
                messages = [
                    ChatMessage(role=ChatRole.SYSTEM, content=self._reflection_system_prompt()),
                    ChatMessage(role=ChatRole.USER, content=self._reflection_user_prompt(state, outcome)),
                ]
                result = self._chat(messages, self._memory_tools, config=self._reflection_config)
                for call in result.tool_calls or []:
                    if call.name != tool_lib.GAME_ACTION_TOOL_NAME:
                        tool_lib.execute_memory_tool(self._memories, call.name, call.arguments)
        except Exception as exc:
            logger.warning("Reflection failed for player %d (ignored): %s", self._player_id, exc)

    def _run_turn(self, state: pyspiel.State) -> int:
        """One model call: apply any memory edits, then commit a legal move or forfeit."""
        legal_actions = state.legal_actions(self._player_id)
        if not legal_actions:
            raise EmptyLegalActionsError(self._player_id)
        legal_set = set(legal_actions)

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=self._system_prompt()),
            ChatMessage(role=ChatRole.USER, content=self._user_prompt(state, legal_actions)),
        ]
        tools = self._memory_tools + [tool_lib.build_game_action_tool(self._legal_hint(legal_actions), legal_actions)]

        result = self._chat(messages, tools)

        if result.content:
            logger.debug("Player %d response: %s", self._player_id, result.content)

        # Apply every memory edit; capture the first legal move (order-independent).
        action: int | None = None
        for call in result.tool_calls or []:
            if call.name == tool_lib.GAME_ACTION_TOOL_NAME:
                if action is None:
                    action = self._validate_action(call, legal_set)
                    if action is not None:
                        logger.debug("Player %d action: %d", self._player_id, action)
            else:
                tool_lib.execute_memory_tool(self._memories, call.name, call.arguments)

        if action is not None:
            return action

        logger.warning("Player %d committed no legal move in its turn response — forfeit", self._player_id)
        raise InvalidActionForfeitError(self._player_id)

    def _chat(self, messages: list[ChatMessage], tools: list[ToolSchema], config: ChatCompletionConfig | None = None):
        try:
            return self._chat_fn(config or self._config, messages, tools)
        except openai.BadRequestError as exc:
            if "context length" in str(exc).lower():
                raise ContextOverflowError(self._player_id) from exc
            raise
        except ChatUnavailableError as exc:
            # Check timeout first: openai.APITimeoutError subclasses APIConnectionError.
            # Slow model = the miner's fault (forfeit); unreachable server = infra
            # (wait-and-replay upstream).
            if isinstance(exc.cause, (openai.APITimeoutError, TimeoutError)):
                logger.warning(
                    "Player %d chat timed out after retries (%s) — forfeiting turn",
                    self._player_id, type(exc.cause).__name__,
                )
                raise ChatTimeoutForfeitError(self._player_id) from exc
            logger.error(
                "Player %d inference server unreachable after retries (%s) — infra failure, "
                "will wait for recovery and replay",
                self._player_id, type(exc.cause).__name__,
            )
            raise ModelUnreachableError(self._player_id) from exc

    @staticmethod
    def _validate_action(call: ToolCall, legal_set: set[int]) -> int | None:
        raw = call.arguments.get("action_id")
        if isinstance(raw, bool) or not isinstance(raw, (int, str)):
            return None
        try:
            action = int(raw)
        except ValueError:
            return None
        return action if action in legal_set else None

    def _memory_block(self) -> str:
        return "\n\n".join(
            mem.render(title=f"{area.value.upper()} (your notes):") for area, mem in self._memories.items()
        )

    def _system_prompt(self) -> str:
        return "\n\n".join([self._agent.generate_system_prompt(), self._memory_block(), _TOOL_GUIDANCE])

    def _reflection_system_prompt(self) -> str:
        return "\n\n".join([self._agent.generate_system_prompt(), self._memory_block(), _REFLECTION_GUIDANCE])

    def _reflection_user_prompt(self, state: pyspiel.State, outcome: GameOutcome) -> str:
        state_desc = self._agent.format_state(state, self._player_id)
        return (
            f"The game is over. Result for you: {outcome.value.upper()}.\n\n"
            f"Final state:\n{state_desc}\n\n"
            "Update your long-term notes on this opponent for future games."
        )

    def _user_prompt(self, state: pyspiel.State, legal_actions: list[int]) -> str:
        state_desc = self._agent.format_state(state, self._player_id)
        action_lines = "\n".join(self._action_line(state, action) for action in legal_actions)
        return (
            f"Current state:\n{state_desc}\n\n"
            f"You are Player {self._player_id}.\n"
            f"Legal actions:\n{action_lines}"
        )

    def _action_line(self, state: pyspiel.State, action: int) -> str:
        try:
            return f"{action} -> {state.action_to_string(self._player_id, action)}"
        except (RuntimeError, AttributeError):
            return str(action)

    @staticmethod
    def _legal_hint(legal_actions: list[int]) -> str:
        return "Legal action ids: " + ", ".join(str(action) for action in legal_actions) + "."
