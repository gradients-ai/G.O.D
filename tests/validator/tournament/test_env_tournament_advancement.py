"""Tests for environment tournament advancement: thresholds, boss round structure,
env scaling via real task creator calls, and model continuation logic.
"""

from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import validator.tournament.constants as t_cst
from core.constants.environments import EnvironmentName
from core.constants.environments import TrainingStartPoint
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.thresholds import get_progressive_threshold


BOSS = "5GBoss"
CONTENDER = "5GContender"


# --- Progressive threshold ---


class TestProgressiveThreshold:
    def test_first_win_uses_base_threshold(self):
        t = get_progressive_threshold(1, TournamentType.TEXT)
        assert t == t_cst.EXPONENTIAL_BASE_THRESHOLD

    def test_env_uses_same_base_threshold(self):
        # Thresholds are disabled; env and text share the same zero base threshold.
        t_env = get_progressive_threshold(1, TournamentType.ENVIRONMENT)
        t_text = get_progressive_threshold(1, TournamentType.TEXT)
        assert t_env == t_cst.EXPONENTIAL_BASE_THRESHOLD_ENVIRONMENT
        assert t_env == t_text

    def test_thresholds_disabled_no_decay(self):
        t1 = get_progressive_threshold(1, TournamentType.TEXT)
        t2 = get_progressive_threshold(2, TournamentType.TEXT)
        t3 = get_progressive_threshold(3, TournamentType.TEXT)
        assert t1 == t2 == t3 == 0.0

    def test_floor_at_min_threshold(self):
        t = get_progressive_threshold(100, TournamentType.TEXT)
        assert t == t_cst.EXPONENTIAL_MIN_THRESHOLD

    def test_decay_rate_applied_correctly(self):
        t2 = get_progressive_threshold(2, TournamentType.TEXT)
        expected = t_cst.EXPONENTIAL_BASE_THRESHOLD * t_cst.EXPONENTIAL_DECAY_RATE
        assert abs(t2 - expected) < 1e-9

    def test_none_tournament_type_uses_default_base(self):
        t = get_progressive_threshold(1, None)
        assert t == t_cst.EXPONENTIAL_BASE_THRESHOLD


# --- Boss round 3-task configuration ---


class TestBossRoundTaskConfig:
    """Verify _create_environment_boss_round_tasks produces 3 tasks with correct start points."""

    @pytest.mark.asyncio
    async def test_three_tasks_with_correct_start_points(self):
        round_data = GroupRound(
            round_id="tourn_abc_round_004",
            round_number=4,
            groups=[Group(member_ids=[CONTENDER, BOSS])],
        )

        created_tasks = []

        async def mock_create_env_task(config, models, datasets, **kwargs):
            task = MagicMock()
            task.task_id = f"task_{len(created_tasks)}"
            task.model_id = kwargs.get("model_id_override", "random_model")
            task.training_start_point = kwargs.get("training_start_point", TrainingStartPoint.DEFAULT)
            created_tasks.append(kwargs)
            return task

        with (
            patch("validator.tournament.task_creator._get_existing_tasks_by_identifier", return_value=[]),
            patch("validator.tournament.task_creator._get_text_models", return_value=["model1"]),
            patch("validator.tournament.task_creator._get_instruct_text_datasets", return_value=["ds1"]),
            patch("validator.tournament.task_creator._get_tournament_base_model", return_value="Qwen/Qwen2.5-7B-Instruct"),
            patch("validator.tournament.task_creator._get_prev_tourn_winner_model", return_value="prev-winner/model"),
            patch("validator.tournament.task_creator.create_synthetic_env_task", side_effect=mock_create_env_task),
            patch("validator.tournament.task_creator._create_and_register_tournament_task", new_callable=AsyncMock),
        ):
            from validator.tournament.task_creator import _create_environment_boss_round_tasks
            config = MagicMock()
            await _create_environment_boss_round_tasks(round_data, "tourn_abc", config)

        assert len(created_tasks) == 3

        # Task 0: CONTINUATION with tournament base model
        assert created_tasks[0]["training_start_point"] == TrainingStartPoint.CONTINUATION
        assert created_tasks[0]["model_id_override"] == "Qwen/Qwen2.5-7B-Instruct"

        # Task 1: FROM_SCRATCH with no model override (random)
        assert created_tasks[1]["training_start_point"] == TrainingStartPoint.FROM_SCRATCH
        assert created_tasks[1]["model_id_override"] is None

        # Task 2: PREVIOUS_WINNER with previous tournament winner model
        assert created_tasks[2]["training_start_point"] == TrainingStartPoint.PREVIOUS_WINNER
        assert created_tasks[2]["model_id_override"] == "prev-winner/model"

    @pytest.mark.asyncio
    async def test_prev_winner_fallback_to_target_model(self):
        """When no previous winner exists, falls back to ENV_TARGET_TOURN_MODEL."""
        from validator.tournament.task_creator import _get_prev_tourn_winner_model

        with patch(
            "validator.tournament.task_creator.get_latest_completed_tournament",
            return_value=None,
        ):
            config = MagicMock()
            result = await _get_prev_tourn_winner_model("tourn_xyz", config)

        assert result == t_cst.ENV_TARGET_TOURN_MODEL

    @pytest.mark.asyncio
    async def test_prev_winner_incompatible_base_falls_back(self):
        """Winner exists but was trained from a different base → fallback."""
        from validator.tournament.task_creator import _get_prev_tourn_winner_model

        prev_tourn = MagicMock()
        prev_tourn.winner_model_repo = "prev-winner/repo"
        prev_tourn.winner_model_base = "different/base-model"

        with patch(
            "validator.tournament.task_creator.get_latest_completed_tournament",
            return_value=prev_tourn,
        ):
            config = MagicMock()
            result = await _get_prev_tourn_winner_model("tourn_xyz", config)

        assert result == t_cst.ENV_TARGET_TOURN_MODEL

    @pytest.mark.asyncio
    async def test_prev_winner_compatible_base_returns_repo(self):
        """Winner trained from ENV_TARGET_TOURN_MODEL → use their model."""
        from validator.tournament.task_creator import _get_prev_tourn_winner_model

        prev_tourn = MagicMock()
        prev_tourn.winner_model_repo = "prev-winner/repo"
        prev_tourn.winner_model_base = t_cst.ENV_TARGET_TOURN_MODEL

        with patch(
            "validator.tournament.task_creator.get_latest_completed_tournament",
            return_value=prev_tourn,
        ):
            config = MagicMock()
            result = await _get_prev_tourn_winner_model("tourn_xyz", config)

        assert result == "prev-winner/repo"


class TestWinnerModelRepoSaving:
    @pytest.mark.asyncio
    async def test_resolve_winner_base_model_reads_adapter_config(self, tmp_path):
        from validator.tournament.tournament_manager import _resolve_winner_base_model

        cfg_path = tmp_path / "adapter_config.json"
        cfg_path.write_text('{"base_model_name_or_path": "foundation/base"}')

        with patch("validator.tournament.tournament_manager.asyncio.to_thread", new_callable=AsyncMock) as to_thread:
            to_thread.return_value = str(cfg_path)

            result = await _resolve_winner_base_model("gradients-io-tournaments/winner-repo", "fallback/base")

        assert result == "foundation/base"

    @pytest.mark.asyncio
    async def test_save_previous_winner_records_resolved_foundation_base(self):
        from validator.tournament.tournament_manager import _save_winner_model_repo

        round_tasks = [
            TournamentTask(
                tournament_id="tourn_1",
                round_id="round_004",
                task_id="task_prev",
            )
        ]
        task = MagicMock(training_start_point=TrainingStartPoint.PREVIOUS_WINNER, model_id="fallback/base")
        update_tournament_winner_model = AsyncMock()

        with (
            patch("validator.tournament.tournament_manager.task_sql.get_expected_repo_name", return_value="winner-repo"),
            patch("validator.tournament.tournament_manager.task_sql.get_task", return_value=task),
            patch("validator.tournament.tournament_manager._resolve_winner_base_model", return_value="foundation/base"),
            patch(
                "validator.tournament.tournament_manager.update_tournament_winner_model",
                update_tournament_winner_model,
            ),
        ):
            await _save_winner_model_repo("tourn_1", "winner_hotkey", round_tasks, MagicMock())

        update_tournament_winner_model.assert_awaited_once_with(
            "tourn_1",
            "gradients-io-tournaments/winner-repo",
            "foundation/base",
            ANY,
        )


# --- Environment group tasks: env scaling and model continuation ---


class TestEnvironmentGroupTasks:
    """Call real _create_environment_group_tasks, verify num_envs, start_point,
    and model_id_override are passed correctly through to create_synthetic_env_task."""

    def _make_round(self, round_number: int, num_groups: int) -> GroupRound:
        groups = [Group(member_ids=[f"hk_{i}"]) for i in range(num_groups)]
        return GroupRound(round_id=f"tourn_x_round_{round_number:03d}", round_number=round_number, groups=groups)

    async def _run_group_task_creation(self, round_number: int, num_groups: int = 2):
        """Run _create_environment_group_tasks and capture the kwargs passed to create_synthetic_env_task."""
        round_data = self._make_round(round_number, num_groups)
        captured_calls = []

        async def mock_create_env_task(config, models, datasets, **kwargs):
            task = MagicMock()
            task.task_id = f"task_{len(captured_calls)}"
            task.model_id = kwargs.get("model_id_override", "base-model")
            task.environment_names = [EnvironmentName.LIARS_DICE]
            task.eval_seed = 42
            captured_calls.append(kwargs)
            return task

        with (
            patch("validator.tournament.task_creator._get_existing_tasks_by_identifier", return_value=[]),
            patch("validator.tournament.task_creator._get_text_models", return_value=["model1"]),
            patch("validator.tournament.task_creator._get_instruct_text_datasets", return_value=["ds1"]),
            patch("validator.tournament.task_creator._get_tournament_base_model", return_value="Qwen/Qwen2.5-7B-Instruct"),
            patch("validator.tournament.task_creator._get_prev_tournament_env_names", return_value=set()),
            patch("validator.tournament.task_creator.create_synthetic_env_task", side_effect=mock_create_env_task),
            patch("validator.tournament.task_creator._create_and_register_tournament_task", new_callable=AsyncMock),
        ):
            from validator.tournament.task_creator import _create_environment_group_tasks
            config = MagicMock()
            await _create_environment_group_tasks(round_data, "tourn_x", config)

        return captured_calls

    @pytest.mark.asyncio
    async def test_round_1_gets_2_envs_and_default_start(self):
        calls = await self._run_group_task_creation(round_number=1)
        assert len(calls) >= 1
        assert calls[0]["num_environments"] == 2
        assert calls[0]["training_start_point"] == TrainingStartPoint.DEFAULT

    @pytest.mark.asyncio
    async def test_round_2_gets_capped_envs_and_continuation(self):
        calls = await self._run_group_task_creation(round_number=2)
        expected_envs = min(2 * t_cst.ENV_ENVS_PER_ROUND_MULTIPLIER, len(EnvironmentName))
        assert calls[0]["num_environments"] == expected_envs
        assert calls[0]["training_start_point"] == TrainingStartPoint.CONTINUATION

    @pytest.mark.asyncio
    async def test_round_3_envs_capped_at_total(self):
        calls = await self._run_group_task_creation(round_number=3)
        assert calls[0]["num_environments"] == len(EnvironmentName)
        assert calls[0]["training_start_point"] == TrainingStartPoint.CONTINUATION

    @pytest.mark.asyncio
    async def test_round_2_uses_tournament_base_model(self):
        """R2+ should pass the R1 base model as model_id_override."""
        calls = await self._run_group_task_creation(round_number=2)
        assert calls[0]["model_id_override"] == "Qwen/Qwen2.5-7B-Instruct"

    @pytest.mark.asyncio
    async def test_round_1_no_model_override(self):
        """R1 should not force a model (lets the task creator pick randomly)."""
        calls = await self._run_group_task_creation(round_number=1)
        assert calls[0].get("model_id_override") is None

    @pytest.mark.asyncio
    async def test_round_1_passes_selected_env_override(self):
        calls = await self._run_group_task_creation(round_number=1)
        assert calls[0].get("environment_names_override") is not None
        assert len(calls[0]["environment_names_override"]) == t_cst.ENV_ENVS_PER_ROUND_MULTIPLIER

    @pytest.mark.asyncio
    async def test_one_task_per_group(self):
        """Each group gets exactly one task."""
        calls = await self._run_group_task_creation(round_number=1, num_groups=3)
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_subsequent_groups_reuse_first_task_config(self):
        """Groups 2+ should use same environment_names and eval_seed as group 1,
        ensuring all groups play the same games with the same seed."""
        calls = await self._run_group_task_creation(round_number=1, num_groups=3)
        # First group creates the reference; groups 2+ should get env/seed overrides from it
        for call in calls[1:]:
            assert call.get("environment_names_override") is not None, "Subsequent groups should reuse reference envs"
            assert call.get("eval_seed_override") is not None, "Subsequent groups should reuse reference seed"

    @pytest.mark.asyncio
    async def test_r2_subsequent_groups_reuse_base_model(self):
        """R2+: all groups should use the same base model from R1."""
        calls = await self._run_group_task_creation(round_number=2, num_groups=3)
        for call in calls:
            assert call.get("model_id_override") == "Qwen/Qwen2.5-7B-Instruct"


class TestEnvironmentTaskAssignment:
    @staticmethod
    def _node(hotkey: str):
        node = MagicMock()
        node.hotkey = hotkey
        return node

    @pytest.mark.asyncio
    async def test_existing_boss_group_prevents_double_assignment(self):
        from validator.tournament.tournament_manager import assign_nodes_to_tournament_tasks

        round_data = GroupRound(
            round_id="round_001",
            round_number=1,
            groups=[
                Group(member_ids=["hk_a", "hk_b"]),
                Group(member_ids=[EMISSION_BURN_HOTKEY, "hk_c"]),
            ],
        )
        tournament = TournamentData(tournament_id="tourn_1", tournament_type=TournamentType.ENVIRONMENT)
        tasks = [
            TournamentTask(tournament_id="tourn_1", round_id="round_001", task_id="task_g1", group_id="round_001_group_001"),
            TournamentTask(tournament_id="tourn_1", round_id="round_001", task_id="task_g2", group_id="round_001_group_002"),
        ]
        assigned: list[tuple[str, str]] = []

        async def assign_node_to_task(task_id, node, psql_db):
            assigned.append((task_id, node.hotkey))

        with (
            patch("validator.tournament.tournament_manager.get_tournament", return_value=tournament),
            patch("validator.tournament.tournament_manager.get_tournament_tasks", return_value=tasks),
            patch("validator.tournament.tournament_manager.get_node_by_hotkey", side_effect=lambda hotkey, _: self._node(hotkey)),
            patch("validator.tournament.tournament_manager.task_sql.get_nodes_assigned_to_task", return_value=[]),
            patch("validator.tournament.tournament_manager.task_sql.assign_node_to_task", side_effect=assign_node_to_task),
            patch("validator.tournament.tournament_manager.task_sql.set_expected_repo_name", new_callable=AsyncMock),
            patch(
                "validator.tournament.tournament_manager.task_sql.get_task",
                return_value=MagicMock(training_start_point=TrainingStartPoint.DEFAULT),
            ),
        ):
            await assign_nodes_to_tournament_tasks("tourn_1", round_data, MagicMock())

        assert ("task_g1", EMISSION_BURN_HOTKEY) not in assigned
        assert ("task_g2", EMISSION_BURN_HOTKEY) in assigned

    @pytest.mark.asyncio
    async def test_missing_continuation_repo_falls_back_to_base_model(self):
        from validator.tournament.tournament_manager import assign_nodes_to_tournament_tasks

        hotkey = "hk_contender"
        round_data = GroupRound(
            round_id="round_002",
            round_number=2,
            groups=[Group(member_ids=[hotkey])],
        )
        tournament = TournamentData(tournament_id="tourn_2", tournament_type=TournamentType.ENVIRONMENT)
        tasks = [
            TournamentTask(tournament_id="tourn_2", round_id="round_002", task_id="task_g1", group_id="round_002_group_001"),
        ]
        task = MagicMock(training_start_point=TrainingStartPoint.CONTINUATION, model_id="base/model")
        set_starting_model_repo = AsyncMock()

        with (
            patch("validator.tournament.tournament_manager.get_tournament", return_value=tournament),
            patch("validator.tournament.tournament_manager.get_tournament_tasks", return_value=tasks),
            patch("validator.tournament.tournament_manager.get_node_by_hotkey", side_effect=lambda hotkey, _: self._node(hotkey)),
            patch("validator.tournament.tournament_manager.task_sql.get_nodes_assigned_to_task", return_value=[]),
            patch("validator.tournament.tournament_manager.task_sql.assign_node_to_task", new_callable=AsyncMock),
            patch("validator.tournament.tournament_manager.task_sql.set_expected_repo_name", new_callable=AsyncMock),
            patch("validator.tournament.tournament_manager.task_sql.get_task", return_value=task),
            patch("validator.tournament.tournament_manager._get_previous_round_repo", return_value="gradients/missing-model"),
            patch("validator.tournament.tournament_manager._hf_repo_exists", return_value=False),
            patch(
                "validator.tournament.tournament_manager.task_sql.set_starting_model_repo",
                set_starting_model_repo,
            ),
        ):
            await assign_nodes_to_tournament_tasks("tourn_2", round_data, MagicMock())

        set_starting_model_repo.assert_any_await("task_g1", hotkey, "base/model", ANY)
