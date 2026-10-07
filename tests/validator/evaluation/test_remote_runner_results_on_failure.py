"""Remote runner should still serve results when the eval command exits non-zero."""

import importlib
import json
import sys
from pathlib import Path


def _load_remote_runner(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("EVAL_RUNNER_COMMAND", json.dumps(["true"]))
    monkeypatch.setenv("EVAL_RUNNER_RESULT_PATH", str(tmp_path / "evaluation_results.json"))
    monkeypatch.setenv("EVAL_RUNNER_MODE", "standard")
    sys.modules.pop("validator.evaluation.remote_runner", None)
    return importlib.import_module("validator.evaluation.remote_runner")


def test_run_eval_serves_results_when_exit_nonzero_but_file_exists(tmp_path: Path, monkeypatch):
    remote_runner = _load_remote_runner(monkeypatch, tmp_path)
    results_path = tmp_path / "evaluation_results.json"
    results_path.write_text(json.dumps({"org/model": {"eval_loss": 0.5, "is_finetune": True}}))

    monkeypatch.setattr(remote_runner, "RESULT_PATH", str(results_path))
    monkeypatch.setattr(remote_runner, "COMMAND", ["false"])
    remote_runner._state.update({"status": "running", "result": None, "error": None})

    class _Proc:
        returncode = -11

    monkeypatch.setattr(remote_runner.subprocess, "run", lambda *a, **k: _Proc())

    remote_runner._run_eval()

    assert remote_runner._state["status"] == "completed"
    assert remote_runner._state["error"] is None
    assert remote_runner._state["result"]["org/model"]["eval_loss"] == 0.5


def test_run_eval_fails_when_exit_nonzero_and_no_results(tmp_path: Path, monkeypatch):
    remote_runner = _load_remote_runner(monkeypatch, tmp_path)
    results_path = tmp_path / "missing.json"
    monkeypatch.setattr(remote_runner, "RESULT_PATH", str(results_path))
    monkeypatch.setattr(remote_runner, "COMMAND", ["false"])
    remote_runner._state.update({"status": "running", "result": None, "error": None})

    class _Proc:
        returncode = 1

    monkeypatch.setattr(remote_runner.subprocess, "run", lambda *a, **k: _Proc())

    remote_runner._run_eval()

    assert remote_runner._state["status"] == "failed"
    assert "exit code 1" in (remote_runner._state["error"] or "")
