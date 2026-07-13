import os

import pytest

from core.constants.environments import EnvironmentName
from ops.tools.evaluation import basilica_swe_infinite_eval
from validator.scoring.models import IndividualEvalResult


def test_build_swe_config_and_task_selection_include_configured_values():
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
            "--collect-logprobs",
            "--model-api-key",
            "secret",
        ]
    )

    config = basilica_swe_infinite_eval.build_swe_eval_config(args)
    task_selection = basilica_swe_infinite_eval.build_task_selection_override(args)

    assert task_selection.task_ids == (7, 83, 45)
    assert task_selection.task_id_min == 10
    assert task_selection.task_id_max == 12
    assert task_selection.num_seeds == 2
    assert config.task_timeout_seconds == 60
    assert config.session_timeout_seconds == 120
    assert config.model_api_key == "secret"
    assert config.collect_logprobs is True


@pytest.mark.asyncio
async def test_run_uses_individual_swe_eval_path(monkeypatch, tmp_path):
    captured = {}

    async def fake_run_evaluation_individual(**kwargs):
        captured.update(kwargs)
        return IndividualEvalResult(
            environment_name=EnvironmentName.SWE_INFINITE,
            scores_by_hotkey={"hk_test": 0.75},
        )

    monkeypatch.setenv("BASILICA_API_TOKEN", "basilica-test-token")
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
    assert captured["swe_task_selection_override"].num_seeds == 1
    assert captured["swe_task_selection_override"].task_ids == (7, 83)
    assert os.environ["SWE_INFINITE_SERVER_BASE_URL"] == "https://swe.example"
