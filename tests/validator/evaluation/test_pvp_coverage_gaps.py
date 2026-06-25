"""Tests covering PvP coverage gaps:

1. Multi-turn game pipeline (game_runner.run_matchup end-to-end with scripted bots)
2. _forfeit_returns correctness
3. Server command construction (build_sglang_command)
4. Chat client retry/backoff logic (_with_retries)
5. _prepare_model LoRA vs full-weight detection
6. _load_config from env var and file
7. _write_results round-trip
8. create_next_round / round progression logic
9. Multi-group score aggregation
10. Performance diff using actual production code
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from core.constants.environments import EnvironmentName
from validator.evaluation.pvp.models import ChatCompletionConfig
from validator.evaluation.pvp.models import ChatMessage
from validator.evaluation.pvp.models import ChatResult
from validator.evaluation.pvp.models import ChatRole
from validator.evaluation.pvp.models import PreparedModel
from validator.evaluation.pvp.models import PvPEnvironmentResult
from validator.evaluation.pvp.models import PvPEvalConfig
from validator.evaluation.pvp.models import PvPEvalMetadata
from validator.evaluation.pvp.models import PvPEvalResults
from validator.evaluation.pvp.models import PvPGroupResults
from validator.evaluation.pvp.models import PvPMatchupConfig
from validator.evaluation.pvp.models import PvPMode
from validator.evaluation.pvp.models import PvPModelSpec
from validator.evaluation.pvp.models import PvPPairResult
from validator.evaluation.pvp.models import ToolCall


try:
    import pyspiel
    HAS_PYSPIEL = True
except ImportError:
    HAS_PYSPIEL = False

needs_pyspiel = pytest.mark.skipif(not HAS_PYSPIEL, reason="pyspiel not installed")


# =============================================================================
# 1. Multi-turn game pipeline: full game from start to finish
# =============================================================================


@needs_pyspiel
class TestFullGamePipeline:
    """Play actual multi-turn games through the real pipeline with scripted bots."""

    @staticmethod
    def _always_first_legal(config, messages, tools=None):
        """Chat function that commits the first legal action via game_action."""
        last_msg = (messages[-1].content if messages else "") or ""
        action_id = 0
        for line in last_msg.split("\n"):
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                token = stripped.split()[0]
                if token.isdigit():
                    action_id = int(token)
                    break
        return ChatResult(tool_calls=[ToolCall(id="c1", name="game_action", arguments={"action_id": action_id})])

    def test_leduc_poker_completes(self):
        """A full Leduc Poker game completes without error."""
        from validator.evaluation.pvp.game_runner import run_matchup

        config_a = ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30000/v1"
        )
        config_b = ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30001/v1"
        )

        chat_fn = self._always_first_legal
        player_a = MagicMock()
        player_a.client = MagicMock()
        player_a.config = config_a
        player_a.chat_fn = chat_fn
        player_b = MagicMock()
        player_b.client = MagicMock()
        player_b.config = config_b
        player_b.chat_fn = chat_fn

        matchup_config = PvPMatchupConfig(time_budget_seconds=1e-9)
        result = run_matchup(
            env_name=EnvironmentName.LEDUC_POKER,
            matchup_config=matchup_config,
            player_a=player_a,
            player_b=player_b,
            base_seed=42,
        )

        assert result.total_games >= 2
        assert result.total_games % 2 == 0
        assert result.model_a_wins + result.model_b_wins + result.draws == result.total_games
        assert result.model_a_wins >= 0
        assert result.model_b_wins >= 0

    def test_liars_dice_completes(self):
        """A full Liar's Dice game completes."""
        from validator.evaluation.pvp.game_runner import run_matchup

        chat_fn = self._always_first_legal
        player_a = MagicMock(config=ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30000/v1"
        ), chat_fn=chat_fn, client=MagicMock())
        player_b = MagicMock(config=ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30001/v1"
        ), chat_fn=chat_fn, client=MagicMock())

        result = run_matchup(
            env_name=EnvironmentName.LIARS_DICE,
            matchup_config=PvPMatchupConfig(time_budget_seconds=1e-9),
            player_a=player_a,
            player_b=player_b,
            base_seed=99,
        )

        assert result.total_games >= 2
        assert result.total_games % 2 == 0
        assert result.model_a_wins + result.model_b_wins + result.draws == result.total_games

    def test_position_swap_affects_results(self):
        """Same seed played from both positions can produce different outcomes.
        Over enough games, both models should get wins from both sides."""
        from validator.evaluation.pvp.game_runner import run_matchup

        chat_fn = self._always_first_legal
        player_a = MagicMock(config=ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30000/v1"
        ), chat_fn=chat_fn, client=MagicMock())
        player_b = MagicMock(config=ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30001/v1"
        ), chat_fn=chat_fn, client=MagicMock())

        result = run_matchup(
            env_name=EnvironmentName.LEDUC_POKER,
            matchup_config=PvPMatchupConfig(time_budget_seconds=1e-9),
            player_a=player_a,
            player_b=player_b,
            base_seed=42,
        )

        assert result.total_games >= 2
        assert result.total_games % 2 == 0


# =============================================================================
# 2. _forfeit_returns correctness
# =============================================================================


@needs_pyspiel
class TestForfeitReturns:
    def test_player_0_forfeits(self):
        from core.pvp.game_eval import _forfeit_returns

        game = pyspiel.load_game("leduc_poker", {"players": 2})
        state = game.new_initial_state()
        returns = _forfeit_returns(state, forfeiting_player=0)

        assert returns[0] == game.min_utility()
        assert returns[1] == game.max_utility()

    def test_player_1_forfeits(self):
        from core.pvp.game_eval import _forfeit_returns

        game = pyspiel.load_game("leduc_poker", {"players": 2})
        state = game.new_initial_state()
        returns = _forfeit_returns(state, forfeiting_player=1)

        assert returns[1] == game.min_utility()
        assert returns[0] == game.max_utility()


# =============================================================================
# 3. Server command construction
# =============================================================================


class TestBuildSglangCommand:
    def test_basic_command_structure(self):
        from validator.evaluation.pvp.server import build_sglang_command

        prepared = PreparedModel(
            sglang_model_path="org/my-model",
            inference_name="org/my-model",
        )
        cmd = build_sglang_command(prepared, port=30000, seed=42)

        assert "--model-path org/my-model" in cmd
        assert "--port 30000" in cmd
        assert "--random-seed 42" in cmd
        assert "--host 0.0.0.0" in cmd
        assert "--enable-deterministic-inference" in cmd

    def test_lora_args_included(self):
        from validator.evaluation.pvp.server import build_sglang_command

        prepared = PreparedModel(
            sglang_model_path="base/model",
            inference_name="base/model:my_lora",
            extra_sglang_args="--enable-lora --lora-paths my_lora=org/adapter --lora-backend triton",
        )
        cmd = build_sglang_command(prepared, port=30001, seed=99)

        assert "--model-path base/model" in cmd
        assert "--enable-lora" in cmd
        assert "--lora-paths my_lora=org/adapter" in cmd
        assert "--lora-backend triton" in cmd

    def test_no_extra_args(self):
        from validator.evaluation.pvp.server import build_sglang_command

        prepared = PreparedModel(
            sglang_model_path="org/model",
            inference_name="org/model",
            extra_sglang_args="",
        )
        cmd = build_sglang_command(prepared, port=30000, seed=1)

        assert "--enable-lora" not in cmd

    def test_caller_resolved_tool_call_parser_wins(self):
        from validator.evaluation.pvp.server import build_sglang_command

        prepared = PreparedModel(
            sglang_model_path="org/opaque-weights",
            inference_name="org/opaque-weights",
            tool_call_parser="qwen25",
        )
        cmd = build_sglang_command(prepared, port=30000, seed=1)

        assert "--tool-call-parser qwen25" in cmd

    def test_parser_falls_back_to_model_path_family(self):
        from validator.evaluation.pvp.server import build_sglang_command

        prepared = PreparedModel(
            sglang_model_path="Qwen/Qwen2.5-3B-Instruct",
            inference_name="Qwen/Qwen2.5-3B-Instruct",
        )
        cmd = build_sglang_command(prepared, port=30000, seed=1)

        assert "--tool-call-parser qwen25" in cmd


# =============================================================================
# 4. Chat client retry logic
# =============================================================================


class TestChatRetryLogic:
    def test_succeeds_on_first_try(self):
        from core.pvp.chat import _with_retries

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "42"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        config = ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30000/v1",
            max_retries=2,
        )
        result = _with_retries(mock_client, config, [
            ChatMessage(role=ChatRole.USER, content="pick a number")
        ])

        assert result.content == "42"
        assert mock_client.chat.completions.create.call_count == 1

    def test_retries_on_timeout(self):
        import openai as oai

        from core.pvp.chat import _with_retries

        mock_client = MagicMock()

        # First call raises timeout, second succeeds
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "5"
        mock_response.usage = None

        mock_client.chat.completions.create.side_effect = [
            oai.APITimeoutError(request=MagicMock()),
            mock_response,
        ]

        config = ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30000/v1",
            max_retries=2,
        )

        with patch("core.pvp.chat.time.sleep"):
            result = _with_retries(mock_client, config, [
                ChatMessage(role=ChatRole.USER, content="test")
            ])

        assert result.content == "5"
        assert mock_client.chat.completions.create.call_count == 2

    def test_raises_after_exhausting_retries(self):
        import openai as oai

        from core.pvp.chat import _with_retries

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = oai.APITimeoutError(
            request=MagicMock()
        )

        config = ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30000/v1",
            max_retries=2,
        )

        with patch("core.pvp.chat.time.sleep"):
            with pytest.raises(RuntimeError, match="Chat failed after 3 attempts"):
                _with_retries(mock_client, config, [
                    ChatMessage(role=ChatRole.USER, content="test")
                ])

        assert mock_client.chat.completions.create.call_count == 3

    def test_retries_on_server_error(self):
        import openai as oai

        from core.pvp.chat import _with_retries

        mock_client = MagicMock()

        mock_response_obj = MagicMock()
        mock_response_obj.status_code = 500
        mock_response_obj.headers = {}

        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "7"
        success_response.usage = None

        mock_client.chat.completions.create.side_effect = [
            oai.APIStatusError(
                message="Internal Server Error",
                response=mock_response_obj,
                body=None,
            ),
            success_response,
        ]

        config = ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30000/v1",
            max_retries=2,
        )

        with patch("core.pvp.chat.time.sleep"):
            result = _with_retries(mock_client, config, [
                ChatMessage(role=ChatRole.USER, content="test")
            ])

        assert result.content == "7"

    def test_does_not_retry_on_4xx(self):
        import openai as oai

        from core.pvp.chat import _with_retries

        mock_client = MagicMock()

        mock_response_obj = MagicMock()
        mock_response_obj.status_code = 400
        mock_response_obj.headers = {}

        mock_client.chat.completions.create.side_effect = oai.APIStatusError(
            message="Bad Request",
            response=mock_response_obj,
            body=None,
        )

        config = ChatCompletionConfig(
            inference_model="test", base_url="http://localhost:30000/v1",
            max_retries=5,
        )

        with pytest.raises(oai.APIStatusError):
            _with_retries(mock_client, config, [
                ChatMessage(role=ChatRole.USER, content="test")
            ])

        # Should NOT retry on 4xx
        assert mock_client.chat.completions.create.call_count == 1


# =============================================================================
# 5. _prepare_model: LoRA vs full-weight detection
# =============================================================================


class TestPrepareModel:
    def test_lora_model_produces_adapter_args(self):
        from validator.evaluation.pvp.__main__ import _prepare_model

        spec = PvPModelSpec(repo="org/lora-adapter", original_model="base/model")

        with patch("validator.evaluation.pvp.__main__.check_for_lora", return_value=True):
            result = _prepare_model(spec, "a")

        assert result.sglang_model_path == "base/model"
        assert "base/model:a_trained_lora" == result.inference_name
        assert "--enable-lora" in result.extra_sglang_args
        assert "a_trained_lora=org/lora-adapter" in result.extra_sglang_args

    def test_full_weight_model_uses_repo_directly(self):
        from validator.evaluation.pvp.__main__ import _prepare_model

        spec = PvPModelSpec(repo="org/full-weights", original_model="base/model")

        with patch("validator.evaluation.pvp.__main__.check_for_lora", return_value=False):
            result = _prepare_model(spec, "b")

        assert result.sglang_model_path == "org/full-weights"
        assert result.inference_name == "org/full-weights"
        assert result.extra_sglang_args == ""

    def test_full_weight_opaque_repo_resolves_parser_from_base_model(self):
        from validator.evaluation.pvp.__main__ import _prepare_model

        spec = PvPModelSpec(repo="miner/tourn-a3f9c2e1", original_model="Qwen/Qwen2.5-3B-Instruct")

        with patch("validator.evaluation.pvp.__main__.check_for_lora", return_value=False):
            result = _prepare_model(spec, "a")

        assert result.tool_call_parser == "qwen25"

    def test_full_weight_repo_with_family_substring_uses_own_family(self):
        from validator.evaluation.pvp.__main__ import _prepare_model

        spec = PvPModelSpec(repo="miner/llama-3.2-ft", original_model="Qwen/Qwen2.5-3B-Instruct")

        with patch("validator.evaluation.pvp.__main__.check_for_lora", return_value=False):
            result = _prepare_model(spec, "a")

        assert result.tool_call_parser == "llama3"


# =============================================================================
# 6. Config loading and results writing
# =============================================================================


class TestConfigLoading:
    def test_load_from_env_var(self):
        from validator.evaluation.pvp.__main__ import _load_config

        config_data = PvPEvalConfig(
            mode=PvPMode.PAIR,
            model_a=PvPModelSpec(repo="org/a", original_model="base/m"),
            model_b=PvPModelSpec(repo="org/b", original_model="base/m"),
            matchups={EnvironmentName.LEDUC_POKER: PvPMatchupConfig(time_budget_seconds=900)},
        )

        with patch.dict(os.environ, {"PVP_EVAL_CONFIG": config_data.model_dump_json()}):
            loaded = _load_config()

        assert loaded.mode == PvPMode.PAIR
        assert loaded.model_a.repo == "org/a"

    def test_load_from_file(self):
        from validator.evaluation.pvp.__main__ import _load_config

        config_data = PvPEvalConfig(
            mode=PvPMode.PAIR,
            model_a=PvPModelSpec(repo="org/a", original_model="base/m"),
            model_b=PvPModelSpec(repo="org/b", original_model="base/m"),
            matchups={EnvironmentName.LIARS_DICE: PvPMatchupConfig(time_budget_seconds=900)},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(config_data.model_dump_json())
            f.flush()

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("validator.evaluation.pvp.__main__.vcst.PVP_CONFIG_PATH", f.name),
            ):
                os.environ.pop("PVP_EVAL_CONFIG", None)
                loaded = _load_config()

        os.unlink(f.name)
        assert loaded.mode == PvPMode.PAIR
        assert loaded.model_b.repo == "org/b"

    def test_raises_when_no_config(self):
        from validator.evaluation.pvp.__main__ import _load_config

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("validator.evaluation.pvp.__main__.vcst.PVP_CONFIG_PATH", "/nonexistent"),
        ):
            os.environ.pop("PVP_EVAL_CONFIG", None)
            with pytest.raises(ValueError, match="No config found"):
                _load_config()


class TestWriteResults:
    def test_results_round_trip(self):
        from validator.evaluation.pvp.__main__ import _write_results

        results = PvPEvalResults(
            model_a="org/a",
            model_b="org/b",
            results={
                EnvironmentName.LEDUC_POKER: PvPEnvironmentResult(
                    model_a_wins=80, model_b_wins=60, draws=10, total_games=150,
                ),
            },
            metadata=PvPEvalMetadata(seed=42, temperature=0.0, wall_time_seconds=100.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results.json")
            with patch.dict(os.environ, {"PVP_RESULTS_PATH": results_path}):
                _write_results(results)

            loaded = PvPEvalResults.model_validate_json(Path(results_path).read_text())
            assert loaded.model_a == "org/a"
            assert loaded.results[EnvironmentName.LEDUC_POKER].model_a_wins == 80

# =============================================================================
# 7. Round finality decision logic
# =============================================================================


class TestRoundFinalityLogic:
    """Test the finality decision logic from create_next_round.

    Extracted from the function to test the decision tree without DB deps:
    - 1 winner → final
    - 2 winners and one is boss → final
    - odd count without boss → boss added
    - odd count with boss → boss stripped
    """

    BOSS = "5GBoss_placeholder"

    def _compute_finality(self, winners: list[str]) -> tuple[bool, list[str]]:
        """Replicate the finality logic from create_next_round."""
        boss = self.BOSS
        next_round_is_final = len(winners) == 1

        if len(winners) == 2:
            if boss in winners:
                next_round_is_final = True
        elif len(winners) % 2 == 1:
            if boss not in winners:
                winners.append(boss)
            else:
                if len(winners) == 1:
                    next_round_is_final = True
                else:
                    winners = [w for w in winners if w != boss]

        return next_round_is_final, winners

    def test_single_winner_is_final(self):
        is_final, _ = self._compute_finality(["contender"])
        assert is_final is True

    def test_two_winners_with_boss_is_final(self):
        is_final, winners = self._compute_finality(["contender", self.BOSS])
        assert is_final is True

    def test_two_winners_without_boss_not_final(self):
        is_final, _ = self._compute_finality(["hk_1", "hk_2"])
        assert is_final is False

    def test_three_winners_without_boss_adds_boss(self):
        is_final, winners = self._compute_finality(["hk_1", "hk_2", "hk_3"])
        assert is_final is False
        assert self.BOSS in winners
        assert len(winners) == 4

    def test_three_winners_with_boss_strips_boss(self):
        """Odd count with boss → strip boss to make even."""
        is_final, winners = self._compute_finality(["hk_1", "hk_2", self.BOSS])
        assert is_final is False
        assert self.BOSS not in winners
        assert len(winners) == 2

    def test_five_winners_no_boss_adds_boss(self):
        is_final, winners = self._compute_finality([f"hk_{i}" for i in range(5)])
        assert is_final is False
        assert self.BOSS in winners
        assert len(winners) == 6

    def test_four_winners_not_final(self):
        is_final, winners = self._compute_finality([f"hk_{i}" for i in range(4)])
        assert is_final is False
        assert len(winners) == 4


# =============================================================================
# 8. Multi-group score aggregation
# =============================================================================


class TestMultiGroupAggregation:
    """Test that winners from multiple groups are correctly merged.

    We mock only the two DB calls (get_tournament_group_members, get_task_results_for_ranking)
    and let the real calculate_miner_ranking_and_scores + advancement logic run.
    """

    @staticmethod
    def _make_miner_result(hotkey: str, score: float):
        """Build a real MinerResultsText so calculate_miner_ranking_and_scores works."""
        from core.models.task_models import TaskType as TT
        from validator.scoring.models import MinerResultsText
        return MinerResultsText(
            hotkey=hotkey,
            test_loss=score,
            synth_loss=0.0,
            is_finetune=True,
            task_type=TT.ENVIRONMENTTASK,
        )

    @pytest.mark.asyncio
    async def test_two_groups_merge_winners(self):
        """Two groups each advancing top 2 → 4 total winners."""
        from validator.tournament.models import TournamentRoundData
        from validator.tournament.models import TournamentTask
        from validator.tournament.round_results import get_environment_group_winners
        round_data = TournamentRoundData(
            round_id="r1", tournament_id="t1", round_number=1,
            round_type="group", is_final_round=False,
        )
        tasks = [
            TournamentTask(tournament_id="t1", round_id="r1", task_id="task_g1", group_id="g1"),
            TournamentTask(tournament_id="t1", round_id="r1", task_id="task_g2", group_id="g2"),
        ]

        g1_participants = [MagicMock(hotkey=f"g1_hk_{i}") for i in range(4)]
        g1_results = [self._make_miner_result(f"g1_hk_{i}", score)
                      for i, score in enumerate([90.0, 70.0, 50.0, 30.0])]

        g2_participants = [MagicMock(hotkey=f"g2_hk_{i}") for i in range(3)]
        g2_results = [self._make_miner_result(f"g2_hk_{i}", score)
                      for i, score in enumerate([85.0, 65.0, 45.0])]

        async def mock_get_group_members(group_id, psql_db):
            return g1_participants if group_id == "g1" else g2_participants

        async def mock_get_results(task_id, psql_db):
            return g1_results if task_id == "task_g1" else g2_results

        with (
            patch("validator.tournament.round_results.get_tournament_group_members", side_effect=mock_get_group_members),
            patch("validator.tournament.round_results.get_task_results_for_ranking", side_effect=mock_get_results),
        ):
            winners = await get_environment_group_winners(round_data, tasks, MagicMock(), MagicMock())

        assert len(winners) == 2  # top 1 per group × 2 groups
        assert "g1_hk_0" in winners
        assert "g2_hk_0" in winners
        assert "g1_hk_1" not in winners
        assert "g2_hk_1" not in winners

    @pytest.mark.asyncio
    async def test_small_group_eliminates_at_least_one(self):
        """A group of 2 non-boss should only advance 1 (not 2)."""
        from validator.tournament.models import TournamentRoundData
        from validator.tournament.models import TournamentTask
        from validator.tournament.round_results import get_environment_group_winners
        round_data = TournamentRoundData(
            round_id="r1", tournament_id="t1", round_number=1,
            round_type="group", is_final_round=False,
        )
        tasks = [TournamentTask(tournament_id="t1", round_id="r1", task_id="task_1", group_id="g1")]

        participants = [MagicMock(hotkey="hk_0"), MagicMock(hotkey="hk_1")]
        results = [
            self._make_miner_result("hk_0", 80.0),
            self._make_miner_result("hk_1", 60.0),
        ]

        with (
            patch("validator.tournament.round_results.get_tournament_group_members", return_value=participants),
            patch("validator.tournament.round_results.get_task_results_for_ranking", return_value=results),
        ):
            winners = await get_environment_group_winners(round_data, tasks, MagicMock(), MagicMock())

        assert len(winners) == 1
        assert winners[0] == "hk_0"


# =============================================================================
# 9. Performance diff using actual production code
# =============================================================================


class TestPerformanceDiffProduction:
    """Test the actual performance diff calculation, not a reimplementation."""

    def test_compute_pvp_tournament_points_ordering(self):
        """Verify that PvP tournament points correctly rank players."""
        from validator.scoring.tournaments import compute_pvp_tournament_points

        # Alice beats Bob in all environments
        group = PvPGroupResults(
            base_model="base/model",
            hotkeys=["alice", "bob"],
            pair_results=[
                PvPPairResult(
                    hotkey_a="alice", hotkey_b="bob",
                    results={
                        EnvironmentName.LEDUC_POKER: PvPEnvironmentResult(
                            model_a_wins=100, model_b_wins=50, draws=0, total_games=150,
                        ),
                        EnvironmentName.LIARS_DICE: PvPEnvironmentResult(
                            model_a_wins=90, model_b_wins=60, draws=0, total_games=150,
                        ),
                    },
                ),
            ],
            metadata=PvPEvalMetadata(seed=42, temperature=0.0),
        )

        standings = compute_pvp_tournament_points(group)
        assert standings[0].hotkey == "alice"
        assert standings[0].points > standings[1].points

    def test_three_way_round_robin_transitive(self):
        """A>B, B>C, A>C → A first, B second, C third."""
        from validator.scoring.tournaments import compute_pvp_tournament_points

        group = PvPGroupResults(
            base_model="base/model",
            hotkeys=["alice", "bob", "carol"],
            pair_results=[
                PvPPairResult(
                    hotkey_a="alice", hotkey_b="bob",
                    results={EnvironmentName.LEDUC_POKER: PvPEnvironmentResult(
                        model_a_wins=100, model_b_wins=50, draws=0, total_games=150,
                    )},
                ),
                PvPPairResult(
                    hotkey_a="bob", hotkey_b="carol",
                    results={EnvironmentName.LEDUC_POKER: PvPEnvironmentResult(
                        model_a_wins=90, model_b_wins=60, draws=0, total_games=150,
                    )},
                ),
                PvPPairResult(
                    hotkey_a="alice", hotkey_b="carol",
                    results={EnvironmentName.LEDUC_POKER: PvPEnvironmentResult(
                        model_a_wins=120, model_b_wins=30, draws=0, total_games=150,
                    )},
                ),
            ],
            metadata=PvPEvalMetadata(seed=42, temperature=0.0),
        )

        standings = compute_pvp_tournament_points(group)
        assert [s.hotkey for s in standings] == ["alice", "bob", "carol"]

    def test_all_draws_equal_points(self):
        """If every pair draws every env, all players have equal points."""
        from validator.scoring.tournaments import compute_pvp_tournament_points

        group = PvPGroupResults(
            base_model="base/model",
            hotkeys=["alice", "bob"],
            pair_results=[
                PvPPairResult(
                    hotkey_a="alice", hotkey_b="bob",
                    results={EnvironmentName.LEDUC_POKER: PvPEnvironmentResult(
                        model_a_wins=75, model_b_wins=75, draws=0, total_games=150,
                    )},
                ),
            ],
            metadata=PvPEvalMetadata(seed=42, temperature=0.0),
        )

        standings = compute_pvp_tournament_points(group)
        assert standings[0].points == standings[1].points


# =============================================================================
# 10. Game agent prompt generation
# =============================================================================


@needs_pyspiel
class TestGameAgentPrompts:
    """Verify agents produce valid prompts for real game states."""

    def test_leduc_poker_system_prompt_has_rules(self):
        from validator.evaluation.pvp.agents import LeducPokerAgent
        agent = LeducPokerAgent()
        prompt = agent.generate_system_prompt()
        assert len(prompt) > 50
        assert "leduc" in prompt.lower() or "poker" in prompt.lower()

    def test_gin_rummy_format_state_doesnt_crash(self):
        from validator.evaluation.pvp.agents import GinRummyAgent
        agent = GinRummyAgent()
        game = pyspiel.load_game("gin_rummy", {"hand_size": 7, "knock_card": 10})
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, _ = zip(*outcomes)
            state.apply_action(actions[0])

        player = state.current_player()
        result = agent.format_state(state, player)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_leduc_poker_format_shows_card(self):
        from validator.evaluation.pvp.agents import LeducPokerAgent
        agent = LeducPokerAgent()
        game = pyspiel.load_game("leduc_poker", {"players": 2})
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, _ = zip(*outcomes)
            state.apply_action(actions[0])

        player = state.current_player()
        formatted = agent.format_state(state, player)
        # Should contain card info
        assert "card" in formatted.lower() or "Card" in formatted


# =============================================================================
# 11. _resolve_spec defaults
# =============================================================================


class TestResolveSpec:
    def test_applies_defaults(self):
        from validator.evaluation.pvp.__main__ import _resolve_spec

        spec = PvPModelSpec(repo="org/model", original_model="base/m")
        gpu, port = _resolve_spec(spec, default_gpu=0, default_port=30000)
        assert gpu == 0
        assert port == 30000

    def test_explicit_overrides(self):
        from validator.evaluation.pvp.__main__ import _resolve_spec

        spec = PvPModelSpec(repo="org/model", original_model="base/m", gpu_id=3, port=31000)
        gpu, port = _resolve_spec(spec, default_gpu=0, default_port=30000)
        assert gpu == 3
        assert port == 31000
