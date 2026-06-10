import asyncio

from core.constants import EnvironmentName
from core.models.model_prep_models import EnvBaselineConfig
from core.models.model_prep_models import EnvStats
from core.models.model_prep_models import WeightStats
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


def test_compute_env_stats_routes_games_in_harness_and_others_to_sidecar(monkeypatch):
    """Pyspiel games go through the in-harness MCTS baseline; intercode goes
    through its env server sidecar (the main-era HTTP path)."""
    calls: dict[str, list[EnvironmentName]] = {"mcts": [], "http": []}

    def fake_mcts_baseline_stats(*, env_name, **kwargs):
        calls["mcts"].append(env_name)
        return EnvStats(num_episodes=kwargs["num_episodes"], mean_score=1.0)

    async def fake_play_episodes(*, env_name, env_server_url, **kwargs):
        assert env_server_url == "http://10.0.0.42:8000"
        calls["http"].append(env_name)
        return EnvStats(num_episodes=kwargs["num_episodes"], mean_score=0.5)

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
    assert result.env_stats[EnvironmentName.OTHELLO].num_episodes == 3
    assert result.env_stats[EnvironmentName.INTERCODE].num_episodes == 7


def test_compute_env_stats_one_env_failing_degrades_to_empty_stats(monkeypatch):
    """An exception in one env's baseline must not take down the others."""

    def exploding_mcts_baseline_stats(*, env_name, **kwargs):
        raise RuntimeError("sglang returned 400")

    async def fake_play_episodes(*, env_name, **kwargs):
        return EnvStats(num_episodes=kwargs["num_episodes"], mean_score=0.5)

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
    assert result.env_stats[EnvironmentName.INTERCODE].num_episodes == 7
