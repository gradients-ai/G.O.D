import asyncio
from dataclasses import replace

import pytest

import core.constants.environments as env_cst
from validator.evaluation.evaluators import swe
from validator.evaluation.swe_infinite_config import DEFAULT_SWE_INFINITE_EVAL_CONFIG
from validator.evaluation.swe_infinite_config import SweInfiniteTaskSelectionOverride


def test_with_v1_preserves_existing_v1_suffix():
    assert swe._with_v1("https://deployment.example/v1") == "https://deployment.example/v1"
    assert swe._with_v1("https://deployment.example") == "https://deployment.example/v1"


def test_build_eval_list_is_deterministic_and_in_range():
    first = swe._build_eval_list(base_seed=42, num_seeds=5, task_id_min=10, task_id_max=20)
    second = swe._build_eval_list(base_seed=42, num_seeds=5, task_id_min=10, task_id_max=20)

    assert first == second
    assert len(first) == 5
    assert all(10 <= task_id <= 20 for _seed, task_id in first)
    assert len({task_id for _seed, task_id in first}) == 5


def test_build_eval_list_uses_half_vetted_and_half_range_tasks():
    eval_list = swe._build_eval_list(base_seed=42, num_seeds=10, task_id_min=1, task_id_max=2_000)
    task_ids = [task_id for _seed, task_id in eval_list]
    vetted_task_ids = set(swe.SWE_VETTED_TASK_IDS)

    assert len(task_ids) == 10
    assert len(set(task_ids)) == 10
    assert sum(task_id in vetted_task_ids for task_id in task_ids) == 5
    assert all(1 <= task_id <= 2_000 for task_id in task_ids)


def test_build_eval_list_rounds_odd_split_toward_vetted_tasks():
    eval_list = swe._build_eval_list(base_seed=42, num_seeds=5, task_id_min=1, task_id_max=2_000)
    task_ids = [task_id for _seed, task_id in eval_list]
    vetted_task_ids = set(swe.SWE_VETTED_TASK_IDS)

    assert sum(task_id in vetted_task_ids for task_id in task_ids) == 3


def test_vetted_task_ids_are_deduped():
    assert len(swe.SWE_VETTED_TASK_IDS) == len(set(swe.SWE_VETTED_TASK_IDS))


def test_build_eval_list_for_task_ids_preserves_requested_task_ids():
    first = swe._build_eval_list_for_task_ids(base_seed=42, task_ids=[7, 83, 45])
    second = swe._build_eval_list_for_task_ids(base_seed=42, task_ids=[7, 83, 45])

    assert first == second
    assert [task_id for _seed, task_id in first] == [7, 83, 45]


def test_parse_task_ids_accepts_commas_and_spaces():
    assert swe._parse_task_ids("7,83 45") == [7, 83, 45]


def test_parse_task_ids_rejects_invalid_values():
    with pytest.raises(ValueError, match="positive integer"):
        swe._parse_task_ids("7 0")

    with pytest.raises(ValueError, match="expected integers"):
        swe._parse_task_ids("7 nope")


def test_unwrap_affinetes_response_accepts_function_based_envelope():
    payload = {"status": "success", "result": {"score": 1.0, "time_taken": 2.0}}

    assert swe._unwrap_affinetes_response(payload) == {"score": 1.0, "time_taken": 2.0}


def test_unwrap_affinetes_response_rejects_failed_envelope():
    with pytest.raises(RuntimeError, match="Affinetes call failed"):
        swe._unwrap_affinetes_response({"status": "failed", "result": {"error": "boom"}})


def test_build_swe_payload_always_uses_miniswe(monkeypatch):
    monkeypatch.setenv("SWE_INFINITE_AGENT", "codex")

    payload = swe._build_swe_payload(
        model="org/model",
        model_base_url="https://model.example/v1",
        task_id=7,
        seed=101,
        temperature=0.0,
        task_timeout=60,
    )

    assert payload["agent"] == "miniswe"


@pytest.mark.asyncio
async def test_resolve_task_range_prefers_environment_max(monkeypatch):
    async def exploding_fetch(_url):
        raise AssertionError("metadata should not be fetched when EnvironmentConfig.task_id_max is set")

    monkeypatch.setattr(swe, "_fetch_swe_completed_up_to", exploding_fetch)

    env_config = env_cst.ENVIRONMENT_CONFIGS[env_cst.EnvironmentName.SWE_INFINITE]
    assert await swe._resolve_task_range(env_config) == (1, env_config.task_id_max)


@pytest.mark.asyncio
async def test_resolve_task_range_uses_public_metadata_when_environment_max_is_zero(monkeypatch):
    async def fake_fetch(_url):
        return 123

    monkeypatch.setattr(swe, "_fetch_swe_completed_up_to", fake_fetch)

    env_config = replace(env_cst.ENVIRONMENT_CONFIGS[env_cst.EnvironmentName.SWE_INFINITE], task_id_max=0)
    assert await swe._resolve_task_range(env_config) == (1, 123)


@pytest.mark.asyncio
async def test_resolve_task_range_allows_explicit_override(monkeypatch):
    async def exploding_fetch(_url):
        raise AssertionError("metadata should not be fetched when max is explicit")

    monkeypatch.setattr(swe, "_fetch_swe_completed_up_to", exploding_fetch)

    env_config = env_cst.ENVIRONMENT_CONFIGS[env_cst.EnvironmentName.SWE_INFINITE]
    task_selection_override = SweInfiniteTaskSelectionOverride(task_id_min=7, task_id_max=9)
    assert await swe._resolve_task_range(env_config, task_selection_override=task_selection_override) == (7, 9)


@pytest.mark.asyncio
async def test_run_swe_evaluation_counts_session_timeouts_as_zero(monkeypatch):
    async def fake_post(_session, _swe_server_url, payload, _task_timeout, _eval_config):
        if payload["task_id"] == 1:
            return {"score": 1.0, "time_taken": 0.01}
        await asyncio.sleep(10)
        return {"score": 1.0, "time_taken": 10.0}

    monkeypatch.setattr(swe, "_post_affinetes_evaluate", fake_post)
    eval_config = DEFAULT_SWE_INFINITE_EVAL_CONFIG.with_overrides(session_timeout_seconds=1, max_concurrent_requests=2)

    avg = await swe._run_swe_evaluation(
        swe_server_url="https://swe.example",
        model_base_url="https://model.example/v1",
        inference_model_name="org/model",
        eval_list=[(101, 1), (102, 2)],
        temperature=0.0,
        task_timeout=60,
        eval_config=eval_config,
    )

    assert avg == 0.5
