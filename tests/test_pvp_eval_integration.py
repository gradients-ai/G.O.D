"""Tests for the environment-eval scoring integration layer:
tournament eval gate, environment ranking direction, batching.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

import validator.core.constants as validator_cst
from core.models.pvp_models import PvPEvalMetadata
from core.models.pvp_models import PvPGroupResults


def _preload_tournament_gpu_module() -> None:
    module_name = "validator.tournament.gpu"
    if module_name in sys.modules:
        return

    repo_root = Path(__file__).resolve().parents[1]
    package_name = "validator.tournament"
    package = types.ModuleType(package_name)
    package.__path__ = [str(repo_root / "validator" / "tournament")]
    sys.modules.setdefault(package_name, package)

    module_path = repo_root / "validator" / "tournament" / "gpu.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


_preload_tournament_gpu_module()

from core.constants import ENVIRONMENT_CONFIGS
from core.constants import EnvironmentName
from core.constants import EvalType
from core.models.scoring_models import IndividualEvalResult
from core.models.scoring_models import IndividualScoresByEnv
from core.models.scoring_models import MinerRepos
from core.models.utility_models import TaskType
from validator.core.models import MinerResultsText
from validator.evaluation import scoring
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


@pytest.mark.asyncio
async def test_pvp_env_eval_requests_two_h100(monkeypatch):
    captured_kwargs = {}

    async def fake_get_or_run_pvp_pairs(**kwargs):
        captured_kwargs.update(kwargs)
        return PvPGroupResults(
            base_model=kwargs["base_model"],
            hotkeys=kwargs["miners"].hotkeys,
            pair_results=[],
            metadata=PvPEvalMetadata(seed=kwargs["seed"], temperature=0.0),
        )

    monkeypatch.setattr(scoring, "_get_or_run_pvp_pairs", fake_get_or_run_pvp_pairs)

    await scoring._eval_pvp_envs(
        task_id=str(uuid4()),
        pvp_envs=[EnvironmentName.LIARS_DICE],
        miners=MinerRepos(by_hotkey={"hk_a": "org/repo-a", "hk_b": "org/repo-b"}),
        base_model="Qwen/Qwen2.5-72B-Instruct",
        seed=42,
        config=SimpleNamespace(psql_db=object()),
    )

    assert captured_kwargs["gpu_count"] == validator_cst.PVP_BASILICA_GPU_COUNT


@pytest.mark.asyncio
async def test_individual_env_eval_requests_one_h100(monkeypatch):
    captured_kwargs = {}

    async def fake_run_evaluation_individual(**kwargs):
        captured_kwargs.update(kwargs)
        return IndividualEvalResult(
            environment_name=kwargs["environment_name"],
            scores_by_hotkey={"hk_a": 0.75, "hk_b": 0.25},
        )

    async def fake_save_individual_score(*args, **kwargs):
        return None

    monkeypatch.setattr(scoring, "run_evaluation_individual", fake_run_evaluation_individual)
    monkeypatch.setattr(scoring.tournament_sql, "save_individual_score", fake_save_individual_score)

    await scoring._dispatch_missing_individual(
        env=EnvironmentName.INTERCODE,
        task_id=uuid4(),
        task_id_str="task-id",
        miners=MinerRepos(by_hotkey={"hk_a": "org/repo-a", "hk_b": "org/repo-b"}),
        base_model="Qwen/Qwen2.5-72B-Instruct",
        model_params=72_000_000_000,
        seed=42,
        config=SimpleNamespace(psql_db=object()),
        scores=IndividualScoresByEnv(),
        db_scores=[],
    )

    assert captured_kwargs["gpu_count"] == validator_cst.INDIVIDUAL_BASILICA_GPU_COUNT


def test_tournament_group_slot_envs_include_individual_envs():
    from validator.cycle import process_tasks

    names = process_tasks._tournament_environment_names()

    assert EnvironmentName.INTERCODE.value in names
    assert EnvironmentName.LIARS_DICE.value in names
