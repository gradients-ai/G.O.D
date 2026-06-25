"""Tests for the PvP bot: think-tag stripping plus per-turn timeout / forfeit.

The detailed turn-loop behaviour (tool dispatch, retries, memory, forfeits) is
covered in test_pvp_bot_loop.py with a scripted ChatFn. Here we keep the pure
strip_think_tags tests and the timeout/forfeit paths that need real pyspiel
states and evaluate_bots.
"""

import re
import time
from unittest.mock import patch

import pytest

from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatResult
from core.models.pvp_models import ToolCall
from core.models.pvp_models import ToolSchema
from validator.evaluation.pvp.chat import strip_think_tags


def _make_config() -> ChatCompletionConfig:
    return ChatCompletionConfig(inference_model="test-model", base_url="http://localhost:30000/v1")


class TestStripThinkTags:
    def test_complete_block(self) -> None:
        assert strip_think_tags("<think>reasoning</think>5") == "5"

    def test_thinking_variant(self) -> None:
        assert strip_think_tags("<thinking>stuff</thinking>7") == "7"

    def test_unclosed_tag(self) -> None:
        assert strip_think_tags("<think>still thinking... 5") == ""

    def test_no_tags(self) -> None:
        assert strip_think_tags("5") == "5"

    def test_only_closing_tag(self) -> None:
        assert strip_think_tags("garbage</think>5") == "5"

    def test_empty_after_strip(self) -> None:
        assert strip_think_tags("<think>only thinking</think>") == ""


class TestDecodeArguments:
    def test_think_tags_scrubbed_from_string_args(self) -> None:
        from core.pvp.chat import _decode_arguments

        args = _decode_arguments('{"slot": 1, "content": "<think>reasoning</think>opp folds early"}')
        assert args == {"slot": 1, "content": "opp folds early"}

    def test_non_string_args_untouched(self) -> None:
        from core.pvp.chat import _decode_arguments

        assert _decode_arguments('{"action_id": 37}') == {"action_id": 37}

    def test_nested_args_are_serialized(self) -> None:
        from core.pvp.chat import _decode_arguments

        args = _decode_arguments('{"payload": {"b": 2, "a": [1]}}')
        assert args == {"payload": '{"a":[1],"b":2}'}

    def test_malformed_json_returns_empty(self) -> None:
        from core.pvp.chat import _decode_arguments

        assert _decode_arguments('{"slot": ') == {}


try:
    import pyspiel

    from validator.evaluation.pvp.agents import LeducPokerAgent
    from validator.evaluation.pvp.bot import InvalidActionForfeitError
    from validator.evaluation.pvp.bot import LLMBot
    from validator.evaluation.pvp.bot import TurnTimeoutError

    HAS_PYSPIEL = True
except ImportError:
    HAS_PYSPIEL = False

needs_pyspiel = pytest.mark.skipif(not HAS_PYSPIEL, reason="pyspiel not installed")


def _first_legal_from_tools(tools: list[ToolSchema] | None) -> int:
    """Pull the first legal action id out of the game_action tool description."""
    for tool in tools or []:
        if tool.function.name == "game_action":
            ids = re.findall(r"\d+", tool.function.description)
            return int(ids[0]) if ids else 0
    return 0


def _committing_chat(sleep_seconds: float | None = None, sleep_on_call: int | None = None):
    """A ChatFn that commits the first legal action each turn, optionally sleeping."""
    calls = {"n": 0}

    def chat_fn(config, messages, tools=None) -> ChatResult:
        calls["n"] += 1
        if sleep_seconds is not None and (sleep_on_call is None or calls["n"] == sleep_on_call):
            time.sleep(sleep_seconds)
        action_id = _first_legal_from_tools(tools)
        return ChatResult(tool_calls=[ToolCall(id="c1", name="game_action", arguments={"action_id": action_id})])

    return chat_fn


def _decision_state():
    game = pyspiel.load_game("leduc_poker", {"players": 2})
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    return game, state


def _bot(game, player_id, chat_fn, agent=None):
    return LLMBot(
        game=game,
        player_id=player_id,
        chat_fn=chat_fn,
        config=_make_config(),
        agent=agent or LeducPokerAgent(),
    )


@needs_pyspiel
class TestTurnTimeout:
    def test_step_under_timeout_returns_action(self):
        game, state = _decision_state()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        bot = _bot(game, pid, _committing_chat(sleep_seconds=1))
        with patch("core.pvp.constants.PVP_TURN_TIMEOUT_SECONDS", 3):
            assert bot.step(state) == legal[0]

    def test_step_over_timeout_raises(self):
        game, state = _decision_state()
        pid = state.current_player()
        bot = _bot(game, pid, _committing_chat(sleep_seconds=3))
        with patch("core.pvp.constants.PVP_TURN_TIMEOUT_SECONDS", 1):
            with pytest.raises(TurnTimeoutError):
                bot.step(state)


@needs_pyspiel
class TestForfeitPropagation:
    def test_timeout_forfeits_to_opponent_via_evaluate_bots(self):
        from core.pvp.game_eval import _evaluate_with_timeout

        game = pyspiel.load_game("leduc_poker", {"players": 2})
        agent = LeducPokerAgent()
        bot_0 = _bot(game, 0, _committing_chat(), agent)
        bot_1 = _bot(game, 1, _committing_chat(sleep_seconds=2, sleep_on_call=2), agent)
        state = game.new_initial_state()

        with patch("core.pvp.constants.PVP_TURN_TIMEOUT_SECONDS", 1):
            returns = _evaluate_with_timeout(state, [bot_0, bot_1], seed=42)

        assert returns[0] == game.max_utility()
        assert returns[1] == game.min_utility()

    def test_invalid_action_forfeit_propagates(self):
        from core.pvp.game_eval import _evaluate_with_timeout

        game = pyspiel.load_game("leduc_poker", {"players": 2})

        class ForfeitingBot(pyspiel.Bot):
            def __init__(self, player_id: int):
                pyspiel.Bot.__init__(self)
                self.player_id = player_id

            def restart_at(self, state) -> None:
                pass

            def inform_action(self, state, player_id, action) -> None:
                pass

            def step(self, state) -> int:
                raise InvalidActionForfeitError(self.player_id)

        class ValidBot(pyspiel.Bot):
            def restart_at(self, state) -> None:
                pass

            def inform_action(self, state, player_id, action) -> None:
                pass

            def step(self, state) -> int:
                return state.legal_actions(state.current_player())[0]

        state = game.new_initial_state()
        returns = _evaluate_with_timeout(state, [ForfeitingBot(0), ValidBot()], seed=42)

        assert returns[0] == game.min_utility()
        assert returns[1] == game.max_utility()
