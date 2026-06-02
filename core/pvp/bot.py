"""Tool-calling LLM bot for OpenSpiel PvP evaluation.

Each turn runs a short agentic loop: the model is given the game state, its
memory slots, and a set of tools. It may call memory tools (whose results are
fed back) and must call game_action to commit a legal move, which ends the
turn. The conversation is rebuilt fresh every turn — the only state carried
across turns is the memory in SlotMemory, not a growing transcript.

Two memory areas live behind the tools: working memory (reset each game) and
long-term memory (persists across games against the same opponent). Robustness
is layered: a per-turn SIGALRM timeout, a bounded number of inner tool
round-trips, illegal moves rejected with an error result (retry), and bad
memory ops that no-op rather than crash.
"""

import logging
import signal

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
from core.pvp import tools as tool_lib
from core.pvp.memory import SlotMemory
from core.pvp.memory import WhitespaceTokenCounter
from core.pvp import constants as cst
from core.pvp.agents import BaseGameAgent


logger = logging.getLogger(__name__)

_NUDGE = "You did not call a tool. Call game_action with a legal action id to make your move."
_TOOL_GUIDANCE = (
    "Use the memory tools to manage your notes between moves. When ready, call "
    "game_action with a legal action id to commit your move and end your turn."
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
    """Raised when a bot fails to commit a legal move within the turn's budget."""

    def __init__(self, player_id: int, invalid_action_failures: int):
        self.player_id = player_id
        self.invalid_action_failures = invalid_action_failures
        super().__init__(
            f"Player {player_id} failed to commit a legal action in {invalid_action_failures} tool steps and forfeits"
        )


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
        rng_seed: int,
        memories: dict[MemoryArea, SlotMemory] | None = None,
        max_inner_steps: int | None = None,
    ):
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._chat_fn = chat_fn
        # The tool loop sets its own per-step generation budget; the inbound
        # config's max_tokens (legacy action-only default) is not what we want.
        self._config = config.model_copy(update={"max_tokens": cst.PVP_PER_STEP_MAX_TOKENS})
        self._agent = agent
        self._rng_seed = rng_seed
        self._memories = memories if memories is not None else default_memories()
        self._max_inner_steps = max_inner_steps or cst.PVP_MAX_INNER_STEPS
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

    def step(self, state: pyspiel.State) -> int:
        """Run one turn under a per-turn wall-clock timeout."""

        def _timeout_handler(signum: int, frame: object) -> None:
            raise TurnTimeoutError(self._player_id)

        prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(cst.PVP_TURN_TIMEOUT_SECONDS)
        try:
            return self._run_turn(state)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)

    def reflect(self, state: pyspiel.State, outcome: GameOutcome) -> None:
        """Single-shot, best-effort memory consolidation after a game ends.

        The model is shown the result and may call memory tools (no game_action)
        to update its notes. Failures are swallowed — the game is already decided,
        so a flaky reflection must never affect the match.
        """
        try:
            messages = [
                ChatMessage(role=ChatRole.SYSTEM, content=self._reflection_system_prompt()),
                ChatMessage(role=ChatRole.USER, content=self._reflection_user_prompt(state, outcome)),
            ]
            result = self._chat(messages, self._memory_tools)
            for call in result.tool_calls or []:
                if call.name != tool_lib.GAME_ACTION_TOOL_NAME:
                    tool_lib.execute_memory_tool(self._memories, call.name, call.arguments)
        except Exception as exc:
            logger.warning("Reflection failed for player %d (ignored): %s", self._player_id, exc)

    def _run_turn(self, state: pyspiel.State) -> int:
        legal_actions = state.legal_actions(self._player_id)
        if not legal_actions:
            raise EmptyLegalActionsError(self._player_id)
        legal_set = set(legal_actions)

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=self._system_prompt()),
            ChatMessage(role=ChatRole.USER, content=self._user_prompt(state, legal_actions)),
        ]
        tools = self._memory_tools + [tool_lib.build_game_action_tool(self._legal_hint(legal_actions))]

        for _ in range(self._max_inner_steps):
            result = self._chat(messages, tools)

            if not result.tool_calls:
                messages.append(ChatMessage(role=ChatRole.ASSISTANT, content=result.content or ""))
                messages.append(ChatMessage(role=ChatRole.USER, content=_NUDGE))
                continue

            messages.append(
                ChatMessage(role=ChatRole.ASSISTANT, content=result.content, tool_calls=result.tool_calls)
            )
            for call in result.tool_calls:
                if call.name == tool_lib.GAME_ACTION_TOOL_NAME:
                    action = self._validate_action(call, legal_set)
                    if action is not None:
                        return action
                    messages.append(
                        self._tool_result(call.id, f"error: not a legal action; legal ids = {legal_actions}")
                    )
                else:
                    output = tool_lib.execute_memory_tool(self._memories, call.name, call.arguments)
                    messages.append(self._tool_result(call.id, output))

        logger.warning("Player %d did not commit a legal move in %d steps — forfeit", self._player_id, self._max_inner_steps)
        raise InvalidActionForfeitError(self._player_id, self._max_inner_steps)

    def _chat(self, messages: list[ChatMessage], tools: list[ToolSchema]):
        try:
            return self._chat_fn(self._config, messages, tools)
        except openai.BadRequestError as exc:
            if "context length" in str(exc).lower():
                raise ContextOverflowError(self._player_id) from exc
            raise

    @staticmethod
    def _validate_action(call: ToolCall, legal_set: set[int]) -> int | None:
        raw = call.arguments.get("action_id")
        if not isinstance(raw, (int, str)):
            return None
        try:
            action = int(raw)
        except ValueError:
            return None
        return action if action in legal_set else None

    @staticmethod
    def _tool_result(tool_call_id: str, content: str) -> ChatMessage:
        return ChatMessage(role=ChatRole.TOOL, tool_call_id=tool_call_id, content=content)

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
