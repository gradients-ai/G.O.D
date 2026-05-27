"""Tests for LLMBot: action parsing, retry logic, conversation management.

Uses a scripted ChatFn to test the full step() path without any HTTP calls.
Pure function tests (parse_action, strip_think_tags) run without pyspiel.
Bot integration tests require pyspiel and are skipped if unavailable.
"""

import pytest

from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatResult
from core.models.pvp_models import ChatRole
from validator.evaluation.pvp.chat import strip_think_tags


# --- Helpers ---


def _make_config() -> ChatCompletionConfig:
    return ChatCompletionConfig(
        inference_model="test-model",
        base_url="http://localhost:30000/v1",
    )


# --- strip_think_tags tests ---


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


# --- _parse_action tests (requires pyspiel for import, but function itself is pure) ---


try:
    import pyspiel

    from validator.evaluation.pvp.bot import InvalidActionForfeitError
    from validator.evaluation.pvp.bot import LLMBot
    from validator.evaluation.pvp.bot import _parse_action
    HAS_PYSPIEL = True
except ImportError:
    HAS_PYSPIEL = False

needs_pyspiel = pytest.mark.skipif(not HAS_PYSPIEL, reason="pyspiel not installed")


@needs_pyspiel
class TestParseAction:

    def test_pure_number(self) -> None:
        assert _parse_action("5", [3, 5, 7]) == 5

    def test_pure_number_not_legal(self) -> None:
        assert _parse_action("99", [3, 5, 7]) is None

    def test_last_legal_wins(self) -> None:
        assert _parse_action("considering 3, I pick 7", [3, 7, 13]) == 7

    def test_no_legal_match(self) -> None:
        assert _parse_action("I fold", [3, 5, 7]) is None

    def test_empty_string(self) -> None:
        assert _parse_action("", [3, 5, 7]) is None

    def test_whitespace_around_number(self) -> None:
        assert _parse_action("  5  ", [3, 5, 7]) == 5

    def test_word_boundary_prevents_substring(self) -> None:
        assert _parse_action("13", [3, 13]) == 13

    def test_single_legal_action(self) -> None:
        assert _parse_action("42", [42]) == 42


# --- LLMBot.step() integration tests ---


def _make_scripted_chat_fn(responses: list[str]):
    """Return a ChatFn that yields responses in order, cycling if exhausted."""
    call_count = 0

    def chat_fn(config: ChatCompletionConfig, messages: list[ChatMessage]) -> ChatResult:
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        return ChatResult(content=responses[idx])

    return chat_fn


def _make_bot(chat_fn, player_id: int = 0) -> LLMBot:
    game = pyspiel.load_game("leduc_poker", {"players": 2})
    from validator.evaluation.pvp.agents import LeducPokerAgent
    return LLMBot(
        game=game,
        player_id=player_id,
        chat_fn=chat_fn,
        config=_make_config(),
        agent=LeducPokerAgent(),
        rng_seed=42,
    )


def _get_state_with_legal_actions():
    """Advance a leduc poker game past chance nodes so a player can act."""
    game = pyspiel.load_game("leduc_poker", {"players": 2})
    state = game.new_initial_state()
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, _ = zip(*outcomes)
        state.apply_action(action_list[0])
    return state


@needs_pyspiel
class TestBotStep:

    def test_valid_action_first_try(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn([str(legal[0])])
        bot = _make_bot(chat_fn, player_id=0)
        action = bot.step(state)
        assert action == legal[0]

    def test_invalid_then_valid(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn(["nonsense", str(legal[0])])
        bot = _make_bot(chat_fn, player_id=0)
        action = bot.step(state)
        assert action == legal[0]

    def test_all_retries_exhausted_uses_random_fallback(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn(["bad"] * 10)
        bot = _make_bot(chat_fn, player_id=0)
        action = bot.step(state)
        assert action in legal
        assert bot._invalid_action_failures == 1

    def test_three_invalid_action_failures_forfeits(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn(["bad"] * 10)
        bot = _make_bot(chat_fn, player_id=0)

        assert bot.step(state) in legal
        assert bot.step(state) in legal
        with pytest.raises(InvalidActionForfeitError) as exc_info:
            bot.step(state)

        assert exc_info.value.player_id == 0
        assert exc_info.value.invalid_action_failures == 3
        assert bot._invalid_action_failures == 3

    def test_none_response_retries(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        call_count = 0

        def chat_fn(config, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResult(content=None)
            return ChatResult(content=str(legal[0]))

        bot = _make_bot(chat_fn, player_id=0)
        action = bot.step(state)
        assert action == legal[0]

    def test_restart_at_clears_conversation(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn([str(legal[0])])
        bot = _make_bot(chat_fn, player_id=0)
        bot.step(state)
        assert len(bot._conversation) > 0
        bot._invalid_action_failures = 1
        bot.restart_at(state)
        assert len(bot._conversation) == 0
        assert bot._system_prompt_set is False
        assert bot._invalid_action_failures == 0

    def test_conversation_alternates_roles(self) -> None:
        state = _get_state_with_legal_actions()
        legal = state.legal_actions(0)
        chat_fn = _make_scripted_chat_fn([str(legal[0])])
        bot = _make_bot(chat_fn, player_id=0)
        bot.step(state)

        roles = [msg.role for msg in bot._conversation]
        assert roles[0] == ChatRole.SYSTEM
        for i in range(1, len(roles) - 1, 2):
            assert roles[i] == ChatRole.USER
            assert roles[i + 1] == ChatRole.ASSISTANT


# --- Turn timeout integration test ---


@needs_pyspiel
class TestTurnTimeoutForfeit:
    """Integration tests: play real games via evaluate_bots with per-turn timeouts.

    Uses a 7s turn timeout. A 5s sleep completes within the limit (no forfeit),
    while a 10s sleep exceeds it (forfeit).
    """

    @staticmethod
    def _make_bots(game, agent, slow_sleep: float):
        """Build bot pair: player 0 is fast, player 1 sleeps on 2nd call."""
        import time

        def fast_chat_fn(config, messages):
            return ChatResult(content="0")

        call_count = 0

        def slow_on_second_chat_fn(config, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                time.sleep(slow_sleep)
            return ChatResult(content="0")

        bot_0 = LLMBot(
            game=game, player_id=0, chat_fn=fast_chat_fn,
            config=_make_config(), agent=agent, rng_seed=42,
        )
        bot_1 = LLMBot(
            game=game, player_id=1, chat_fn=slow_on_second_chat_fn,
            config=_make_config(), agent=agent, rng_seed=43,
        )
        return bot_0, bot_1

    def test_5s_sleep_completes_within_10s_timeout(self) -> None:
        """Player 1 sleeps 5s on 2nd call. With a 10s timeout the game
        completes normally — no forfeit."""
        from unittest.mock import patch

        from validator.evaluation.pvp.agents import LeducPokerAgent
        from validator.evaluation.pvp.game_runner import _evaluate_with_timeout

        agent = LeducPokerAgent()
        game = pyspiel.load_game("leduc_poker", {"players": 2})
        bot_0, bot_1 = self._make_bots(game, agent, slow_sleep=5)
        state = game.new_initial_state()

        with patch("validator.core.constants.PVP_TURN_TIMEOUT_SECONDS", 10):
            returns = _evaluate_with_timeout(state, [bot_0, bot_1], seed=42)

        # Game completed normally — returns should NOT be forfeit values.
        # Both can't be at the extremes simultaneously in a real game.
        assert not (returns[0] == game.max_utility() and returns[1] == game.min_utility()), (
            f"Expected normal game completion, got forfeit returns: {returns}"
        )

    def test_10s_sleep_exceeds_10s_timeout_and_forfeits(self) -> None:
        """Player 1 sleeps 10s on 2nd call. With a 10s timeout the alarm
        fires first and the game is forfeited to player 0."""
        from unittest.mock import patch

        from validator.evaluation.pvp.agents import LeducPokerAgent
        from validator.evaluation.pvp.game_runner import _evaluate_with_timeout

        agent = LeducPokerAgent()
        game = pyspiel.load_game("leduc_poker", {"players": 2})
        bot_0, bot_1 = self._make_bots(game, agent, slow_sleep=15)
        state = game.new_initial_state()

        with patch("validator.core.constants.PVP_TURN_TIMEOUT_SECONDS", 10):
            returns = _evaluate_with_timeout(state, [bot_0, bot_1], seed=42)

        assert returns[0] == game.max_utility()
        assert returns[1] == game.min_utility()


@needs_pyspiel
class TestInvalidActionForfeit:
    def test_repeated_invalid_actions_forfeit_to_opponent(self) -> None:
        from validator.evaluation.pvp.game_runner import _evaluate_with_timeout

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
                raise InvalidActionForfeitError(self.player_id, 3)

        class ValidBot(pyspiel.Bot):
            def restart_at(self, state) -> None:
                pass

            def inform_action(self, state, player_id, action) -> None:
                pass

            def step(self, state) -> int:
                return state.legal_actions(state.current_player())[0]

        bot_0 = ForfeitingBot(player_id=0)
        bot_1 = ValidBot()
        state = game.new_initial_state()

        returns = _evaluate_with_timeout(state, [bot_0, bot_1], seed=42)

        assert returns[0] == game.min_utility()
        assert returns[1] == game.max_utility()
