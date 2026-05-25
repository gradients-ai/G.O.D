"""Tests for the environment-eval scoring integration layer:
tournament eval gate, environment ranking direction, batching.
"""

from unittest.mock import MagicMock

from core.constants import ENVIRONMENT_CONFIGS
from core.constants import EnvironmentName
from core.constants import EvalType
from core.models.utility_models import TaskType
from validator.core.models import MinerResultsText
from validator.evaluation.scoring import calculate_miner_ranking_and_scores
from validator.evaluation.scoring import should_use_tournament_eval


# --- 5a: should_use_tournament_eval gate ---


class TestShouldUseTournamentEval:
    def test_env_task_with_pvp_env_returns_true(self):
        task = MagicMock()
        task.task_type = TaskType.ENVIRONMENTTASK
        task.environment_names = [EnvironmentName.LIARS_DICE]
        assert should_use_tournament_eval(task) is True

    def test_env_task_with_individual_env_returns_true(self):
        task = MagicMock()
        task.task_type = TaskType.ENVIRONMENTTASK
        task.environment_names = [EnvironmentName.INTERCODE]
        assert should_use_tournament_eval(task) is True

    def test_non_env_task_returns_false(self):
        task = MagicMock()
        task.task_type = TaskType.INSTRUCTTEXTTASK
        task.environment_names = [EnvironmentName.LIARS_DICE]
        assert should_use_tournament_eval(task) is False

    def test_env_task_no_env_names_returns_false(self):
        task = MagicMock()
        task.task_type = TaskType.ENVIRONMENTTASK
        task.environment_names = []
        assert should_use_tournament_eval(task) is False

    def test_all_configured_tournament_eval_envs_trigger(self):
        """Every tournament-evaluated environment should use the tournament eval path."""
        for env_name, config in ENVIRONMENT_CONFIGS.items():
            if config.eval_type in {EvalType.PVP, EvalType.INDIVIDUAL}:
                task = MagicMock()
                task.task_type = TaskType.ENVIRONMENTTASK
                task.environment_names = [env_name]
                assert should_use_tournament_eval(task) is True, f"{env_name} should trigger tournament eval"


# --- 5b: Environment ranking direction (higher = better) ---


class TestEnvRankingDirection:
    def _make_result(self, hotkey: str, test_loss: float, task_type: TaskType = TaskType.ENVIRONMENTTASK) -> MinerResultsText:
        return MinerResultsText(
            hotkey=hotkey,
            test_loss=test_loss,
            synth_loss=0.0,
            is_finetune=True,
            task_type=task_type,
        )

    def test_env_higher_score_ranked_first(self):
        """For environment tasks, higher test_loss = better → ranked first."""
        results = [
            self._make_result("low", 10.0),
            self._make_result("high", 90.0),
            self._make_result("mid", 50.0),
        ]
        ranked = calculate_miner_ranking_and_scores(results)

        # Find the one with FIRST_PLACE_SCORE
        first_place = next(r for r in ranked if r.score > 0 and "1st" in (r.score_reason or ""))
        assert first_place.hotkey == "high"

    def test_text_lower_loss_ranked_first(self):
        """For text tasks, lower test_loss = better → ranked first."""
        results = [
            self._make_result("high_loss", 5.0, TaskType.INSTRUCTTEXTTASK),
            self._make_result("low_loss", 0.5, TaskType.INSTRUCTTEXTTASK),
        ]
        ranked = calculate_miner_ranking_and_scores(results)

        first_place = next(r for r in ranked if r.score > 0 and "1st" in (r.score_reason or ""))
        assert first_place.hotkey == "low_loss"

    def test_grpo_higher_is_better(self):
        """GRPO tasks: higher loss = better, same as environment."""
        results = [
            self._make_result("low", 1.0, TaskType.GRPOTASK),
            self._make_result("high", 9.0, TaskType.GRPOTASK),
        ]
        ranked = calculate_miner_ranking_and_scores(results)

        first_place = next(r for r in ranked if r.score > 0 and "1st" in (r.score_reason or ""))
        assert first_place.hotkey == "high"
