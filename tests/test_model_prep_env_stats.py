import pytest

from core.constants import EnvironmentName
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
async def test_play_episodes_reports_sidecar_result_errors(capsys):
    stats = await env_stats._play_episodes(
        session=_FakeSession(),
        env_name=EnvironmentName.INTERCODE,
        env_server_url="http://env-server",
        sglang_base_url="http://sglang/v1",
        model_name="model",
        num_episodes=1,
        task_id_min=1,
        task_id_max=1,
        eval_payload_extra=None,
    )

    captured = capsys.readouterr()

    assert stats.num_episodes == 1
    assert stats.mean_score == 0.0
    assert "intercode episode 1: error task_id=1 seed=" in captured.out
    assert "TimeoutError" in captured.out
