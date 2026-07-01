import os

import pytest

from core.constants.environments import EnvironmentName
from ops.tools.evaluation import basilica_swe_infinite_eval
from validator.scoring.models import IndividualEvalResult


def test_build_swe_env_overrides_includes_only_configured_values():
    args = basilica_swe_infinite_eval.parse_args(
        [
            "--num-seeds",
            "2",
            "--task-id-min",
            "10",
            "--task-id-max",
            "12",
            "--task-timeout-seconds",
            "60",
            "--session-timeout-seconds",
            "120",
            "--task-id",
            "7",
            "83",
            "45",
            "--agent",
            "miniswe",
            "--collect-logprobs",
            "--model-api-key",
            "secret",
        ]
    )

    overrides = basilica_swe_infinite_eval.build_swe_env_overrides(args, "https://swe.example")

    assert overrides == {
        "SWE_INFINITE_SERVER_BASE_URL": "https://swe.example",
        "SWE_INFINITE_TASK_IDS": "7,83,45",
        "SWE_INFINITE_TASK_ID_MIN": "10",
        "SWE_INFINITE_TASK_ID_MAX": "12",
        "SWE_INFINITE_NUM_SEEDS": "2",
        "SWE_INFINITE_TASK_TIMEOUT_SECONDS": "60",
        "SWE_INFINITE_SESSION_TIMEOUT": "120",
        "SWE_INFINITE_AGENT": "miniswe",
        "SWE_INFINITE_MODEL_API_KEY": "secret",
        "SWE_INFINITE_COLLECT_LOGPROBS": "true",
    }


@pytest.mark.asyncio
async def test_run_uses_individual_swe_eval_path(monkeypatch, tmp_path):
    captured = {}

    async def fake_run_evaluation_individual(**kwargs):
        captured.update(kwargs)
        return IndividualEvalResult(
            environment_name=EnvironmentName.SWE_INFINITE,
            scores_by_hotkey={"hk_test": 0.75},
        )

    monkeypatch.setenv("BASILICA_API_KEY", "basilica-test-key")
    monkeypatch.delenv("SWE_INFINITE_SERVER_BASE_URL", raising=False)
    monkeypatch.setattr(basilica_swe_infinite_eval, "run_evaluation_individual", fake_run_evaluation_individual)

    args = basilica_swe_infinite_eval.parse_args(
        [
            "--env-file",
            str(tmp_path / "missing.env"),
            "--swe-server-url",
            "https://swe.example",
            "--model",
            "org/model",
            "--base-model",
            "org/base",
            "--hotkey",
            "hk_test",
            "--num-seeds",
            "1",
            "--task-id",
            "7",
            "83",
            "--base-chain-json",
            '["org/previous-base"]',
        ]
    )

    await basilica_swe_infinite_eval.run(args)

    assert captured["miners"].by_hotkey == {"hk_test": "org/model"}
    assert captured["base_model"] == "org/base"
    assert captured["environment_name"] == EnvironmentName.SWE_INFINITE
    assert captured["task_id"] is None
    assert captured["psql_db"] is None
    assert captured["base_chains"] == {"hk_test": ["org/previous-base"]}
    assert os.environ["SWE_INFINITE_SERVER_BASE_URL"] == "https://swe.example"
    assert os.environ["SWE_INFINITE_NUM_SEEDS"] == "1"
    assert os.environ["SWE_INFINITE_TASK_IDS"] == "7,83"
