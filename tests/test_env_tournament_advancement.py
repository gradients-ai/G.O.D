"""Tests for environment tournament advancement: thresholds, boss round structure,
env scaling via real task creator calls, and model continuation logic.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.constants import EnvironmentName
from core.constants import TrainingStartPoint
from core.models.tournament_models import TournamentType
from core.models.tournament_models import Group
from core.models.tournament_models import GroupRound
from validator.tournament.utils import get_progressive_threshold
import validator.tournament.constants as t_cst


BOSS = "5GBoss"
CONTENDER = "5GContender"


# --- Progressive threshold ---


class TestProgressiveThreshold:
    def test_first_win_uses_base_threshold(self):
        t = get_progressive_threshold(1, TournamentType.TEXT)
        assert t == t_cst.EXPONENTIAL_BASE_THRESHOLD

    def test_env_uses_same_base_threshold(self):
        # Thresholds are disabled; env and text share the same (zero) base threshold.
        t_env = get_progressive_threshold(1, TournamentType.ENVIRONMENT)
        t_text = get_progressive_threshold(1, TournamentType.TEXT)
        assert t_env == t_cst.EXPONENTIAL_BASE_THRESHOLD_ENVIRONMENT
        assert t_env == t_text

    def test_thresholds_disabled_no_decay(self):
        # With progressive thresholds disabled every consecutive-win count returns 0.
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
