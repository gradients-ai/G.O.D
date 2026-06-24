"""Single-player OpenSpiel evaluation helpers.

These reuse the same OpenSpiel timeout/forfeit wrapper as PvP, but use a
single-purpose action bot with no memory tools or reflection turns.
"""

import logging
import random
import signal
import time
from contextlib import contextmanager

import openai
import pyspiel
from pydantic import BaseModel
from pydantic import Field

from core.constants import ENVIRONMENT_CONFIGS
from core.constants import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatFn
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatRole
from core.models.pvp_models import ToolCall
from core.pvp import constants as cst
from core.pvp import tools as tool_lib
from core.pvp.agents import BaseGameAgent
from core.pvp.bot import ContextOverflowError
from core.pvp.bot import EmptyLegalActionsError
from core.pvp.bot import InvalidActionForfeitError
from core.pvp.bot import TurnTimeoutError
from core.pvp.game_eval import _evaluate_game_with_timeout
from core.pvp.game_eval import agent_class_for
from core.pvp.game_eval import config_id_for_seed


logger = logging.getLogger(__name__)

_ACTION_ONLY_GUIDANCE = (
    "You get ONE response this turn. Call exactly one tool: game_action. "
    "Choose one legal action id. Do not write prose."
)


class IndividualActionBot(pyspiel.Bot):
    """Single-player OpenSpiel bot with only the game_action tool.

    This intentionally avoids the PvP memory tools. Small models often fail by
    emitting malformed multi-tool output; 2048 only needs one legal slide
    direction each turn.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        player_id: int,
        chat_fn: ChatFn,
        config: ChatCompletionConfig,
        agent: BaseGameAgent,
    ):
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._chat_fn = chat_fn
        self._config = config.model_copy(update={"max_tokens": cst.PVP_TURN_MAX_TOKENS})
        self._agent = agent

    def restart_at(self, state: pyspiel.State) -> None:
        return None

    def inform_action(self, state: pyspiel.State, player_id: int, action: int) -> None:
        return None

    @contextmanager
    def _wall_clock(self, seconds: int):
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
        with self._wall_clock(cst.PVP_TURN_TIMEOUT_SECONDS):
            return self._run_turn(state)

    def _run_turn(self, state: pyspiel.State) -> int:
        legal_actions = state.legal_actions(self._player_id)
        if not legal_actions:
            raise EmptyLegalActionsError(self._player_id)
        legal_set = set(legal_actions)

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=self._system_prompt()),
            ChatMessage(role=ChatRole.USER, content=self._user_prompt(state, legal_actions)),
        ]
        tools = [tool_lib.build_game_action_tool(self._legal_hint(legal_actions), legal_actions)]

        try:
            result = self._chat_fn(self._config, messages, tools)
        except openai.BadRequestError as exc:
            if "context length" in str(exc).lower():
                raise ContextOverflowError(self._player_id) from exc
            raise

        for call in result.tool_calls or []:
            if call.name == tool_lib.GAME_ACTION_TOOL_NAME:
                action = self._validate_action(call, legal_set)
                if action is not None:
                    return action

        logger.warning("Player %d committed no legal move in its turn response — forfeit", self._player_id)
        raise InvalidActionForfeitError(self._player_id)

    def _system_prompt(self) -> str:
        return "\n\n".join([self._agent.generate_system_prompt(), _ACTION_ONLY_GUIDANCE])

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

        bot = IndividualActionBot(
            game=game,
            player_id=0,
            chat_fn=chat_fn,
            config=config,
            agent=agent,
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
