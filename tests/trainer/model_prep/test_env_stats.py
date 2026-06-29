import asyncio

import pytest


pytest.importorskip("pyspiel")
pytest.importorskip("open_spiel")

from core.constants.environments import EnvironmentName
from core.models.model_prep_models import EnvBaselineConfig
from core.models.model_prep_models import EnvStats
from core.models.model_prep_models import WeightStats
from core.pvp.baseline import MctsBaselineResult
from trainer.model_prep import env_stats


def test_start_process_discards_output_by_default(monkeypatch):
    captured = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(env_stats.subprocess, "Popen", fake_popen)

    proc = env_stats.start_process("python -m sglang.launch_server", "sglang")

    assert proc is not None
    assert captured["stdout"] is env_stats.subprocess.DEVNULL
    assert captured["stderr"] is env_stats.subprocess.DEVNULL


def test_start_process_can_capture_output_for_debug_logging(monkeypatch):
    captured = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(env_stats.subprocess, "Popen", fake_popen)

    proc = env_stats.start_process("python -m sglang.launch_server", "sglang", capture_stdout=True)

    assert proc is not None
    assert captured["stdout"] is env_stats.subprocess.PIPE
    assert captured["stderr"] is env_stats.subprocess.STDOUT


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return {"result": {"score": 0.0, "error": "TimeoutError", "task_id": 1}}


class _FakeSession:
    def post(self, *args, **kwargs):
        return _FakeResponse()


@pytest.mark.asyncio
async def test_play_episodes_reports_sidecar_result_errors(monkeypatch, capsys):
    ticks = iter([0.0, 0.0, 2.0])

    monkeypatch.setattr(env_stats.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(env_stats, "ENV_BASELINE_TIME_BUDGET_SECONDS", 1.0)

    stats = await env_stats._play_episodes(
        session=_FakeSession(),
        env_name=EnvironmentName.INTERCODE,
        env_server_url="http://env-server",
        sglang_base_url="http://sglang/v1",
        model_name="model",
        task_id_min=1,
        task_id_max=1,
        eval_payload_extra=None,
    )

    captured = capsys.readouterr()

    assert stats.num_episodes == 1
    assert stats.mean_score == 0.0
    assert "intercode episode 1: error task_id=1 seed=" in captured.out
    assert "TimeoutError" in captured.out


def test_mcts_baseline_stats_uses_time_budget_instead_of_episode_count(monkeypatch):
    captured = {}

    def fake_run_mcts_baseline(**kwargs):
        captured.update(kwargs)
        return MctsBaselineResult(wins=2, draws=1, losses=1, num_games=4)

    monkeypatch.setattr(env_stats, "create_client", lambda config: object())
    monkeypatch.setattr(env_stats, "run_mcts_baseline", fake_run_mcts_baseline)
    monkeypatch.setattr(env_stats, "ENV_BASELINE_TIME_BUDGET_SECONDS", 420.0)

    stats = env_stats._mcts_baseline_stats(
        env_name=EnvironmentName.LIARS_DICE,
        sglang_base_url="http://sglang/v1",
        model_name="model",
        model_path="/cache/models/model",
        eval_payload_extra={"mcts_max_simulations": 225},
    )

    assert captured["num_games"] is None
    assert captured["time_budget_seconds"] == 420.0
    assert captured["mcts_simulations"] == 225
    assert stats.num_episodes == 4
    assert stats.mean_score == 0.625


def test_compute_env_stats_routes_games_in_harness_and_others_to_sidecar(monkeypatch):
    """Pyspiel games use in-harness MCTS; intercode uses its env sidecar."""
    calls: dict[str, list[EnvironmentName]] = {"mcts": [], "http": []}

    def fake_mcts_baseline_stats(*, env_name, **kwargs):
        assert "num_episodes" not in kwargs
        calls["mcts"].append(env_name)
        return EnvStats(num_episodes=11, mean_score=1.0)

    async def fake_play_episodes(*, env_name, env_server_url, **kwargs):
        assert env_server_url == "http://10.0.0.42:8000"
        assert "num_episodes" not in kwargs
        calls["http"].append(env_name)
        return EnvStats(num_episodes=13, mean_score=0.5)

    async def fake_wait_for_health(*args, **kwargs):
        return None

    monkeypatch.setattr(env_stats, "_mcts_baseline_stats", fake_mcts_baseline_stats)
    monkeypatch.setattr(env_stats, "_play_episodes", fake_play_episodes)
    monkeypatch.setattr(env_stats, "wait_for_health", fake_wait_for_health)
    monkeypatch.setattr(env_stats, "start_process", lambda *a, **k: None)
    monkeypatch.setattr(env_stats, "compute_weight_stats", lambda model: WeightStats(by_group={}))

    env_configs = {
        EnvironmentName.OTHELLO: EnvBaselineConfig(
            url="http://10.0.0.1:8000", task_id_min=1, task_id_max=10, num_episodes=3,
        ),
        EnvironmentName.INTERCODE: EnvBaselineConfig(
            url="http://10.0.0.42:8000", task_id_min=1, task_id_max=200, num_episodes=7,
        ),
    }

    result = asyncio.run(env_stats.compute_env_stats("/cache/models/m", model=object(), env_configs=env_configs))

    assert calls["mcts"] == [EnvironmentName.OTHELLO]
    assert calls["http"] == [EnvironmentName.INTERCODE]
    assert result.env_stats[EnvironmentName.OTHELLO].num_episodes == 11
    assert result.env_stats[EnvironmentName.INTERCODE].num_episodes == 13


def test_compute_env_stats_one_env_failing_degrades_to_empty_stats(monkeypatch):
    """An exception in one env's baseline must not take down the others."""

    def exploding_mcts_baseline_stats(*, env_name, **kwargs):
        raise RuntimeError("sglang returned 400")

    async def fake_play_episodes(*, env_name, **kwargs):
        assert "num_episodes" not in kwargs
        return EnvStats(num_episodes=13, mean_score=0.5)

    async def fake_wait_for_health(*args, **kwargs):
        return None

    monkeypatch.setattr(env_stats, "_mcts_baseline_stats", exploding_mcts_baseline_stats)
    monkeypatch.setattr(env_stats, "_play_episodes", fake_play_episodes)
    monkeypatch.setattr(env_stats, "wait_for_health", fake_wait_for_health)
    monkeypatch.setattr(env_stats, "start_process", lambda *a, **k: None)
    monkeypatch.setattr(env_stats, "compute_weight_stats", lambda model: WeightStats(by_group={}))

    env_configs = {
        EnvironmentName.OTHELLO: EnvBaselineConfig(
            url="http://10.0.0.1:8000", task_id_min=1, task_id_max=10, num_episodes=3,
        ),
        EnvironmentName.INTERCODE: EnvBaselineConfig(
            url="http://10.0.0.42:8000", task_id_min=1, task_id_max=200, num_episodes=7,
        ),
    }

    result = asyncio.run(env_stats.compute_env_stats("/cache/models/m", model=object(), env_configs=env_configs))

    assert result.env_stats[EnvironmentName.OTHELLO].num_episodes == 0
    assert result.env_stats[EnvironmentName.INTERCODE].num_episodes == 13
