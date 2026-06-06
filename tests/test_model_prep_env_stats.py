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
