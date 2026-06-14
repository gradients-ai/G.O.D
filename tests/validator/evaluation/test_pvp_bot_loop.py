"""Tests for the tool-calling LLMBot turn: single model call per turn."""

import random

import pytest

from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatResult
from core.models.pvp_models import GameOutcome
from core.models.pvp_models import MemoryArea
from core.models.pvp_models import ToolCall
from core.pvp.memory import SlotMemory
from core.pvp.memory import WhitespaceTokenCounter


try:
    import pyspiel  # noqa: F401

    from validator.evaluation.pvp.agents import LeducPokerAgent
    from validator.evaluation.pvp.bot import EmptyLegalActionsError
    from validator.evaluation.pvp.bot import InvalidActionForfeitError
    from validator.evaluation.pvp.bot import LLMBot

    HAS_PYSPIEL = True
except ImportError:
    HAS_PYSPIEL = False

needs_pyspiel = pytest.mark.skipif(not HAS_PYSPIEL, reason="pyspiel not installed")


def _config() -> ChatCompletionConfig:
    return ChatCompletionConfig(inference_model="m", base_url="http://localhost:30000/v1", max_tokens=256)


def _memories() -> dict:
    counter = WhitespaceTokenCounter()
    return {
        MemoryArea.WORKING: SlotMemory(4, 128, counter),
        MemoryArea.LONG_TERM: SlotMemory(8, 128, counter),
    }


def _call(name: str, cid: str = "c1", **arguments) -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=arguments)


def _resp(*tool_calls: ToolCall, content: str | None = None) -> ChatResult:
    return ChatResult(content=content, tool_calls=list(tool_calls) or None)


class ScriptedChat:
    """A ChatFn that returns queued responses and records what it was sent."""

    def __init__(self, *responses: ChatResult):
        self.queue = list(responses)
        self.calls: list[dict] = []

    def __call__(self, config, messages, tools=None) -> ChatResult:
        self.calls.append({"messages": messages, "tools": tools})
        assert self.queue, "bot made more chat calls than the script provided"
        return self.queue.pop(0)


def _leduc():
    game = pyspiel.load_game("leduc_poker", {"players": 2})
    state = game.new_initial_state()
    rng = random.Random(0)
    while state.is_chance_node():
        state.apply_action(rng.choice([o for o, _ in state.chance_outcomes()]))
    return game, state


def _make_bot(game, chat, player_id, memories=None):
    return LLMBot(
        game=game,
        player_id=player_id,
        chat_fn=chat,
        config=_config(),
        agent=LeducPokerAgent(),
        memories=memories or _memories(),
    )


@needs_pyspiel
class TestCommitAction:
    def test_legal_game_action_is_returned(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        chat = ScriptedChat(_resp(_call("game_action", action_id=legal[0])))
        assert _make_bot(game, chat, pid).step(state) == legal[0]
        assert len(chat.calls) == 1

    def test_empty_legal_actions_raises(self):
        game, state = _leduc()
        pid = state.current_player()
        rng = random.Random(1)
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(rng.choice([o for o, _ in state.chance_outcomes()]))
            else:
                state.apply_action(state.legal_actions()[0])
        chat = ScriptedChat(_resp(_call("game_action", action_id=0)))
        with pytest.raises(EmptyLegalActionsError):
            _make_bot(game, chat, pid).step(state)


@needs_pyspiel
class TestMemoryWithMove:
    def test_memory_write_then_action_in_one_response(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        mems = _memories()
        chat = ScriptedChat(
            _resp(
                _call("working_memory_rewrite", cid="a", slot=1, content="bet aggressively"),
                _call("game_action", cid="b", action_id=legal[0]),
            )
        )
        action = _make_bot(game, chat, pid, memories=mems).step(state)
        assert action == legal[0]
        assert mems[MemoryArea.WORKING].slots[1] == "bet aggressively"

    def test_long_term_write_persists_in_injected_object(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        mems = _memories()
        chat = ScriptedChat(
            _resp(
                _call("long_term_memory_append", cid="a", slot=2, content="opp folds to raises"),
                _call("game_action", cid="b", action_id=legal[0]),
            )
        )
        _make_bot(game, chat, pid, memories=mems).step(state)
        assert "opp folds to raises" in mems[MemoryArea.LONG_TERM].slots[2]

    def test_bad_memory_op_does_not_forfeit(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        chat = ScriptedChat(
            _resp(
                _call("working_memory_rewrite", cid="a", slot=99, content="x"),
                _call("game_action", cid="b", action_id=legal[0]),
            )
        )
        assert _make_bot(game, chat, pid).step(state) == legal[0]

    def test_memory_applied_even_when_action_precedes_it(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        mems = _memories()
        chat = ScriptedChat(
            _resp(
                _call("game_action", cid="a", action_id=legal[0]),
                _call("working_memory_append", cid="b", slot=1, content="noted after move"),
            )
        )
        assert _make_bot(game, chat, pid, memories=mems).step(state) == legal[0]
        assert "noted after move" in mems[MemoryArea.WORKING].slots[1]

    def test_updated_memory_is_rendered_on_next_turn(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        mems = _memories()
        chat = ScriptedChat(
            _resp(
                _call("working_memory_rewrite", cid="a", slot=1, content="my-secret-plan"),
                _call("game_action", cid="b", action_id=legal[0]),
            ),
            _resp(_call("game_action", cid="c", action_id=legal[0])),
        )
        bot = _make_bot(game, chat, pid, memories=mems)
        bot.step(state)
        bot.step(state)
        system = chat.calls[1]["messages"][0].content
        assert "my-secret-plan" in system


@needs_pyspiel
class TestForfeit:
    def test_illegal_action_forfeits(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        illegal = max(legal) + 999
        chat = ScriptedChat(_resp(_call("game_action", action_id=illegal)))
        with pytest.raises(InvalidActionForfeitError) as exc:
            _make_bot(game, chat, pid).step(state)
        assert exc.value.player_id == pid
        assert len(chat.calls) == 1

    def test_text_only_response_forfeits(self):
        game, state = _leduc()
        pid = state.current_player()
        chat = ScriptedChat(_resp(content="hmm let me think about this hand"))
        with pytest.raises(InvalidActionForfeitError):
            _make_bot(game, chat, pid).step(state)
        assert len(chat.calls) == 1

    def test_boolean_action_id_forfeits(self):
        game, state = _leduc()
        pid = state.current_player()
        chat = ScriptedChat(_resp(_call("game_action", action_id=True)))
        with pytest.raises(InvalidActionForfeitError):
            _make_bot(game, chat, pid).step(state)

    def test_memory_only_response_forfeits_but_applies_writes(self):
        game, state = _leduc()
        pid = state.current_player()
        mems = _memories()
        chat = ScriptedChat(_resp(_call("working_memory_append", slot=1, content="note")))
        with pytest.raises(InvalidActionForfeitError):
            _make_bot(game, chat, pid, memories=mems).step(state)
        assert "note" in mems[MemoryArea.WORKING].slots[1]


@needs_pyspiel
class TestPromptAndTools:
    def test_tools_include_memory_and_game_action(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        chat = ScriptedChat(_resp(_call("game_action", action_id=legal[0])))
        _make_bot(game, chat, pid).step(state)
        names = {t.function.name for t in chat.calls[0]["tools"]}
        assert "game_action" in names
        assert {"working_memory_rewrite", "long_term_memory_append"} <= names

    def test_game_action_tool_constrains_action_id_to_legal_set(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        chat = ScriptedChat(_resp(_call("game_action", action_id=legal[0])))
        _make_bot(game, chat, pid).step(state)
        ga = next(t for t in chat.calls[0]["tools"] if t.function.name == "game_action")
        assert str(legal[0]) in ga.function.description
        assert ga.function.parameters["properties"]["action_id"]["enum"] == legal

    def test_system_prompt_has_rules_and_both_memories(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        chat = ScriptedChat(_resp(_call("game_action", action_id=legal[0])))
        _make_bot(game, chat, pid).step(state)
        system = chat.calls[0]["messages"][0].content
        assert "LEDUC" in system.upper()
        assert MemoryArea.WORKING.value.upper() in system.upper()
        assert MemoryArea.LONG_TERM.value.upper() in system.upper()

    def test_system_prompt_does_not_demand_a_bare_action_id(self):
        game, state = _leduc()
        pid = state.current_player()
        legal = state.legal_actions(pid)
        chat = ScriptedChat(_resp(_call("game_action", action_id=legal[0])))
        _make_bot(game, chat, pid).step(state)
        system = chat.calls[0]["messages"][0].content.lower()
        assert "only the action id" not in system
        assert "game_action" in system


@needs_pyspiel
class TestMemoryLifetime:
    def test_restart_clears_working_keeps_long_term(self):
        game, state = _leduc()
        pid = state.current_player()
        mems = _memories()
        mems[MemoryArea.WORKING].rewrite(1, "this game plan")
        mems[MemoryArea.LONG_TERM].rewrite(1, "durable opponent read")
        bot = _make_bot(game, chat=ScriptedChat(), player_id=pid, memories=mems)
        bot.restart_at(state)
        assert mems[MemoryArea.WORKING].slots[1] == ""
        assert mems[MemoryArea.LONG_TERM].slots[1] == "durable opponent read"


@needs_pyspiel
class TestReflection:
    def test_reflect_applies_memory_write(self):
        game, state = _leduc()
        pid = state.current_player()
        mems = _memories()
        chat = ScriptedChat(_resp(_call("long_term_memory_append", cid="r", slot=1, content="opp over-folds to raises")))
        bot = _make_bot(game, chat, pid, memories=mems)
        bot.reflect(state, GameOutcome.WIN)
        assert "opp over-folds to raises" in mems[MemoryArea.LONG_TERM].slots[1]

    def test_reflect_offers_memory_tools_but_not_game_action(self):
        game, state = _leduc()
        pid = state.current_player()
        chat = ScriptedChat(_resp())
        bot = _make_bot(game, chat, pid)
        bot.reflect(state, GameOutcome.LOSS)
        names = {t.function.name for t in chat.calls[0]["tools"]}
        assert "game_action" not in names
        assert {"long_term_memory_rewrite", "long_term_memory_append"} <= names

    def test_reflect_is_single_shot(self):
        game, state = _leduc()
        pid = state.current_player()
        chat = ScriptedChat(_resp(_call("long_term_memory_append", cid="r", slot=1, content="note")))
        bot = _make_bot(game, chat, pid)
        bot.reflect(state, GameOutcome.DRAW)
        assert len(chat.calls) == 1
