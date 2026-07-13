from types import SimpleNamespace
from uuid import uuid4

import pytest

from core.constants.environments import EnvironmentName
from core.models.dataset_models import EnvironmentDatasetType
from core.models.dataset_models import FileFormat
from validator.evaluation import basilica
from validator.evaluation import docker_evaluation
from validator.evaluation.basilica_deployments import create_basilica_public_sglang_eval_runner_source
from validator.scoring.models import MinerRepos


@pytest.mark.asyncio
async def test_resolve_verified_deployment_name_rejects_returned_url():
    expected_name = "expected-deployment"
    listed = SimpleNamespace(name=expected_name, url="https://deployments.example/eval")
    returned = SimpleNamespace(name="https://deployments.example/eval")
    client = SimpleNamespace(list=lambda: [listed])

    resolved = await basilica._resolve_verified_deployment_name(client, returned, expected_name)

    assert resolved == expected_name


@pytest.mark.asyncio
async def test_resolve_verified_deployment_name_uses_url_only_as_lookup_hint():
    listed = SimpleNamespace(name="listed-name", url="https://deployments.example/eval")
    returned = SimpleNamespace(url="https://deployments.example/eval")
    client = SimpleNamespace(list=lambda: [listed])

    resolved = await basilica._resolve_verified_deployment_name(client, returned, "requested-name")

    assert resolved == "listed-name"


@pytest.mark.asyncio
async def test_resolve_verified_deployment_name_uses_basilica_url_slug_as_hint():
    deployment_name = "0044a7fe-e8f5-4bcd-94ce-8ae0cf6db55c"
    deployment_url = f"https://{deployment_name}.deployments.basilica.ai"
    listed = SimpleNamespace(name=deployment_name, url=deployment_url)
    returned = SimpleNamespace(url=deployment_url)
    client = SimpleNamespace(list=lambda: [listed])

    resolved = await basilica._resolve_verified_deployment_name(client, returned, "requested-name")

    assert resolved == deployment_name


@pytest.mark.asyncio
async def test_delete_deployment_not_found_counts_as_deleted(monkeypatch):
    deployment_name = "already-gone"
    log_steps = []

    class MissingDeployment:
        def delete(self):
            raise RuntimeError("Not found: Deployment not found")

    monkeypatch.setattr(basilica, "log_basilica_logs_block", lambda *_args, **_kwargs: None)

    deleted = await basilica._delete_terminal_deployment(
        client=SimpleNamespace(list=lambda: []),
        deployment=MissingDeployment(),
        deployment_name=deployment_name,
        reason="test",
        repo="repo",
        eval_logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        deleted_deployment_names=set(),
        log_eval_step=lambda step, **fields: log_steps.append((step, fields)),
    )

    assert deleted is True
    assert any(step == "delete_already_gone" for step, _fields in log_steps)


@pytest.mark.asyncio
async def test_deploy_persists_verified_name_before_readiness(monkeypatch):
    deployment_name = "verified-before-ready"
    deployment = SimpleNamespace(name=deployment_name, url=f"https://{deployment_name}.deployments.basilica.ai")
    persisted = []

    class Client:
        def deploy(self, **_kwargs):
            return deployment

        def list(self):
            return [deployment]

    async def never_ready(*_args, **_kwargs):
        return None

    async def delete_ok(*_args, **_kwargs):
        return True

    async def persist(name: str):
        persisted.append(name)

    monkeypatch.setattr(basilica, "_wait_for_deployment_ready", never_ready)
    monkeypatch.setattr(basilica, "_delete_eval_deployment", delete_ok)

    with pytest.raises(basilica.DeploymentNotReadyError):
        await basilica._deploy_with_readiness_timeout(
            ctx=SimpleNamespace(log_eval_step=lambda *_args, **_kwargs: None),
            client=Client(),
            deployment_name=deployment_name,
            deploy_kwargs={"name": deployment_name},
            on_verified_deployment_name=persist,
        )

    assert persisted == [deployment_name]


@pytest.mark.asyncio
async def test_deploy_readiness_failed_delete_is_retryable(monkeypatch):
    deployment_name = "not-ready-still-live"
    deployment = SimpleNamespace(name=deployment_name, url=f"https://{deployment_name}.deployments.basilica.ai")

    class Client:
        def deploy(self, **_kwargs):
            return deployment

        def list(self):
            return [deployment]

    async def never_ready(*_args, **_kwargs):
        return None

    async def delete_failed(*_args, **_kwargs):
        return False

    async def noop_persist(_name: str):
        return None

    monkeypatch.setattr(basilica, "_wait_for_deployment_ready", never_ready)
    monkeypatch.setattr(basilica, "_delete_eval_deployment", delete_failed)

    with pytest.raises(basilica.EvaluationRetryableError):
        await basilica._deploy_with_readiness_timeout(
            ctx=SimpleNamespace(log_eval_step=lambda *_args, **_kwargs: None),
            client=Client(),
            deployment_name=deployment_name,
            deploy_kwargs={"name": deployment_name},
            on_verified_deployment_name=noop_persist,
        )


@pytest.mark.asyncio
async def test_environment_text_eval_does_not_persist_to_evaluations(monkeypatch):
    captured = {}

    async def fake_load_eval_pair_state_for_models(*_args, **_kwargs):
        return {"org/repo-a": "stale-eval-table-deployment"}, {"org/repo-a": "hk_a"}

    async def fake_run_basilica_eval_repos(**kwargs):
        captured.update(kwargs)
        return {"org/repo-a": {"org/repo-a": {"eval_loss": 1.0, "is_finetune": True}}}

    monkeypatch.setattr(docker_evaluation, "load_eval_pair_state_for_models", fake_load_eval_pair_state_for_models)
    monkeypatch.setattr(docker_evaluation, "run_basilica_eval_repos", fake_run_basilica_eval_repos)

    await docker_evaluation.run_evaluation_basilica_text(
        dataset="proxy",
        models=["org/repo-a"],
        original_model="org/base",
        dataset_type=EnvironmentDatasetType(environment_names=[EnvironmentName.INTERCODE]),
        file_format=FileFormat.JSON,
        num_gpus=1,
        task_id=uuid4(),
        psql_db=object(),
    )

    assert captured["deployment_ids_by_repo"] == {}
    assert captured["persist_deployment_ids"] is False
    assert captured["reserve_deployment_id"] is False


@pytest.mark.asyncio
async def test_individual_env_eval_uses_individual_score_deployment_owner(monkeypatch):
    captured = {}
    persisted = []

    async def fake_get_individual_deployment_ids(*_args, **_kwargs):
        return {"hk_a": "previous-individual-deployment"}

    async def fake_set_individual_score_deployment_id(*args):
        persisted.append(args)

    async def fake_run_basilica_eval_repos(**kwargs):
        captured.update(kwargs)
        await kwargs["deployment_id_persister"]("org/repo-a", "verified-new-deployment")
        return {"org/repo-a": {"org/repo-a": {"eval_loss": 0.25}}}

    monkeypatch.setattr(
        docker_evaluation.tournament_sql,
        "get_individual_deployment_ids",
        fake_get_individual_deployment_ids,
    )
    monkeypatch.setattr(
        docker_evaluation.tournament_sql,
        "set_individual_score_deployment_id",
        fake_set_individual_score_deployment_id,
    )
    monkeypatch.setattr(docker_evaluation, "run_basilica_eval_repos", fake_run_basilica_eval_repos)

    task_id = uuid4()
    await docker_evaluation.run_evaluation_individual(
        miners=MinerRepos(by_hotkey={"hk_a": "org/repo-a"}),
        base_model="org/base",
        environment_name=EnvironmentName.INTERCODE,
        seed=1,
        image="validator-image",
        gpu_count=1,
        task_id=task_id,
        psql_db=object(),
    )

    assert captured["deployment_ids_by_repo"] == {"org/repo-a": "previous-individual-deployment"}
    assert captured["persist_deployment_ids"] is False
    assert captured["reserve_deployment_id"] is False
    assert persisted == [(str(task_id), "hk_a", EnvironmentName.INTERCODE.value, "verified-new-deployment", captured["psql_db"])]


@pytest.mark.asyncio
async def test_individual_env_eval_without_db_skips_deployment_lookup(monkeypatch):
    captured = {}

    async def exploding_get_individual_deployment_ids(*_args, **_kwargs):
        raise AssertionError("DB deployment lookup should not run without psql_db")

    async def fake_run_basilica_eval_repos(**kwargs):
        captured.update(kwargs)
        return {"org/repo-a": {"org/repo-a": {"eval_loss": 0.25}}}

    monkeypatch.setattr(
        docker_evaluation.tournament_sql,
        "get_individual_deployment_ids",
        exploding_get_individual_deployment_ids,
    )
    monkeypatch.setattr(docker_evaluation, "run_basilica_eval_repos", fake_run_basilica_eval_repos)

    result = await docker_evaluation.run_evaluation_individual(
        miners=MinerRepos(by_hotkey={"hk_a": "org/repo-a"}),
        base_model="org/base",
        environment_name=EnvironmentName.INTERCODE,
        seed=1,
        image="validator-image",
        gpu_count=1,
        task_id=None,
        psql_db=None,
    )

    assert result.scores_by_hotkey == {"hk_a": 0.25}
    assert captured["deployment_ids_by_repo"] == {}
    assert captured["task_id"] is None
    assert captured["psql_db"] is None


def test_public_sglang_runner_source_compiles_and_exposes_proxy():
    source = create_basilica_public_sglang_eval_runner_source(
        ["python", "-m", "validator.evaluation.evaluators.swe"],
        "/aplp/evaluation_results.json",
    )

    compile(source, "<swe-runner>", "exec")
    assert "SWE_INFINITE_MODEL_BASE_URL" in source
    assert "SWE_INFINITE_MODEL_API_KEY" in source
    assert "startsWith" not in source
    assert 'self.path == "/v1"' in source


@pytest.mark.asyncio
async def test_individual_swe_eval_uses_public_sglang_runner(monkeypatch):
    captured = {}

    async def fake_get_individual_deployment_ids(*_args, **_kwargs):
        return {}

    async def fake_run_basilica_eval_repos(**kwargs):
        captured.update(kwargs)
        return {"org/repo-a": {"org/repo-a": {"eval_loss": 0.5}}}

    monkeypatch.setenv("SWE_INFINITE_SERVER_BASE_URL", "https://swe.example")
    monkeypatch.setattr(
        docker_evaluation.tournament_sql,
        "get_individual_deployment_ids",
        fake_get_individual_deployment_ids,
    )
    monkeypatch.setattr(docker_evaluation, "run_basilica_eval_repos", fake_run_basilica_eval_repos)

    result = await docker_evaluation.run_evaluation_individual(
        miners=MinerRepos(by_hotkey={"hk_a": "org/repo-a"}),
        base_model="org/base",
        environment_name=EnvironmentName.SWE_INFINITE,
        seed=1,
        image="validator-swe-image",
        gpu_count=1,
        task_id=uuid4(),
        psql_db=object(),
    )

    repo_env = captured["build_env_for_repo"]("org/repo-a")
    assert result.scores_by_hotkey == {"hk_a": 0.5}
    assert "SWE_INFINITE_MODEL_BASE_URL" in captured["source"]
    assert repo_env["SWE_INFINITE_SERVER_BASE_URL"] == "https://swe.example"
    assert repo_env["ENVIRONMENT_NAME"] == EnvironmentName.SWE_INFINITE.value
