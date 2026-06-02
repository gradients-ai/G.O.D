"""Integration tests for cross-game memory persistence (Phase 3).

Drives a real matchup through game_runner with a scripted ChatFn that commits
moves during play and writes a marker to long-term memory at each reflection.
Long-term memory must survive across the games of a matchup (opponent model),
while working memory resets each game.
"""

import re
from unittest.mock import MagicMock

import pytest

from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatResult
from core.models.pvp_models import EnvironmentName
from core.models.pvp_models import ToolCall
from core.models.pvp_models import PvPMatchupConfig


try:
    import pyspiel  # noqa: F401

    from validator.evaluation.pvp.game_runner import run_matchup

    HAS_PYSPIEL = True
except ImportError:
    HAS_PYSPIEL = False

needs_pyspiel = pytest.mark.skipif(not HAS_PYSPIEL, reason="pyspiel not installed")


def _first_legal_from_tools(tools) -> int:
    for tool in tools or []:
        if tool.function.name == "game_action":
            ids = re.findall(r"\d+", tool.function.description)
            return int(ids[0]) if ids else 0
    return 0


class MatchupChat:
    """Commits the first legal action on a turn; on reflection, appends a unique
    marker to long-term memory. Records the long-term render seen on each turn."""

    def __init__(self):
        self.reflections = 0
        self.turn_systems: list[str] = []

    def __call__(self, config, messages, tools=None) -> ChatResult:
        names = {t.function.name for t in tools or []}
        if "game_action" in names:
            self.turn_systems.append(messages[0].content)  # system prompt (renders long-term)
            action_id = _first_legal_from_tools(tools)
            return ChatResult(tool_calls=[ToolCall(id="c", name="game_action", arguments={"action_id": action_id})])
        # reflection turn: no game_action offered -> write a durable marker
        self.reflections += 1
        return ChatResult(
            tool_calls=[
                ToolCall(id="r", name="long_term_memory_append", arguments={"slot": 1, "content": f"REFLECT-MARK-{self.reflections}"})
            ]
        )


def _player(port: int, chat) -> MagicMock:
    return MagicMock(
        config=ChatCompletionConfig(inference_model="test", base_url=f"http://localhost:{port}/v1"),
        chat_fn=chat,
        client=MagicMock(),
    )


@needs_pyspiel
class TestMatchupPersistence:
    def test_long_term_marker_persists_into_later_game(self):
        chat_a = MatchupChat()
        chat_b = MatchupChat()
        player_a = _player(30000, chat_a)
        player_b = _player(30001, chat_b)

        run_matchup(
            env_name=EnvironmentName.LEDUC_POKER,
            matchup_config=PvPMatchupConfig(num_games=1),  # 1 seed x 2 positions = 2 games
            player_a=player_a,
            player_b=player_b,
            base_seed=42,
        )

        # Reflection ran after each game (2 games -> 2 reflections per player).
        assert chat_a.reflections == 2

        # The marker written during game 1's reflection is visible in a later
        # game's turn prompt — proof long-term memory carried across the game boundary.
        assert any("REFLECT-MARK-1" in system for system in chat_a.turn_systems)

    def test_working_memory_does_not_leak_marker_across_games(self):
        # Long-term carries the marker; working memory never should (it's not written
        # here, and it resets each game). The marker only ever appears under the
        # long-term heading, never the working one.
        chat_a = MatchupChat()
        player_a = _player(30000, chat_a)
        player_b = _player(30001, MatchupChat())

        run_matchup(
            env_name=EnvironmentName.LEDUC_POKER,
            matchup_config=PvPMatchupConfig(num_games=1),
            player_a=player_a,
            player_b=player_b,
            base_seed=7,
        )

        for system in chat_a.turn_systems:
            working_section = system.split("LONG_TERM")[0]
            assert "REFLECT-MARK" not in working_section
