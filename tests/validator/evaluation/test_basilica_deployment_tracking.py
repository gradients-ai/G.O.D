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
async def test_standard_deploy_starts_evaluation_cost_after_capacity_reservation(monkeypatch):
    task_id = uuid4()
    psql_db = object()
    deployment_name = "costed-deployment"
    deployment = SimpleNamespace(name=deployment_name, url=f"https://{deployment_name}.deployments.basilica.ai")
    events = []

    async def reserve(*_args, **_kwargs):
        events.append("reserve")
        return True

    async def start_cost(**kwargs):
        events.append(("start", kwargs))
        return True

    async def deploy(**_kwargs):
        events.append("deploy")
        return deployment, deployment_name

    monkeypatch.setattr(basilica.basilica, "BasilicaClient", lambda: object())
    monkeypatch.setattr(basilica.tasks_sql, "try_reserve_evaluation_gpus", reserve)
    monkeypatch.setattr(basilica.gpu_costs_sql, "start_cost_run", start_cost)
    monkeypatch.setattr(basilica, "_deploy_with_readiness_timeout", deploy)
    monkeypatch.setattr(basilica, "update_environment_logger_labels", lambda *_args, **_kwargs: None)

    ctx = basilica._BasilicaEvalContext(
        repo="org/repo",
        eval_logger=SimpleNamespace(
            info=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None,
        ),
        deleted_deployment_names=set(),
        log_eval_step=lambda *_args, **_kwargs: None,
    )
    await basilica._deploy_basilica_eval_repo(
        ctx=ctx,
        deployment_name=deployment_name,
        image="image",
        source="source",
        env={},
        gpu_count=3,
        gpu_models=["A100"],
        min_gpu_memory_gb=40,
        storage=False,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey={"org/repo": "hotkey"},
        deployment_id_persister=None,
        reserve_deployment_id=True,
    )

    assert events[0] == "reserve"
    assert events[1][0] == "start"
    assert events[1][1] == {
        "run_key": f"evaluation:{task_id}:{deployment_name}",
        "task_id": str(task_id),
        "category": "evaluation",
        "gpu_type": "A100",
        "gpu_count": 3,
        "psql_db": psql_db,
        "metadata": {"deployment_name": deployment_name, "repo": "org/repo"},
    }
    assert events[2] == "deploy"


@pytest.mark.asyncio
async def test_standard_deploy_does_not_start_cost_when_capacity_unavailable(monkeypatch):
    starts = []

    async def no_capacity(*_args, **_kwargs):
        return False

    async def start_cost(**kwargs):
        starts.append(kwargs)

    monkeypatch.setattr(basilica.basilica, "BasilicaClient", lambda: object())
    monkeypatch.setattr(basilica.tasks_sql, "try_reserve_evaluation_gpus", no_capacity)
    monkeypatch.setattr(basilica.gpu_costs_sql, "start_cost_run", start_cost)

    ctx = basilica._BasilicaEvalContext(
        repo="org/repo",
        eval_logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None, error=lambda *_args, **_kwargs: None),
        deleted_deployment_names=set(),
        log_eval_step=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(basilica.EvaluationCapacityUnavailable):
        await basilica._deploy_basilica_eval_repo(
            ctx=ctx,
            deployment_name="no-capacity",
            image="image",
            source="source",
            env={},
            gpu_count=2,
            gpu_models=["A100"],
            min_gpu_memory_gb=40,
            storage=False,
            task_id=uuid4(),
            psql_db=object(),
            repo_to_hotkey={"org/repo": "hotkey"},
            deployment_id_persister=None,
            reserve_deployment_id=True,
        )

    assert starts == []


@pytest.mark.asyncio
async def test_standard_attempt_finishes_cost_with_terminal_success(monkeypatch):
    task_id = uuid4()
    psql_db = object()
    deployment = SimpleNamespace(name="resolved-deployment", url="https://resolved-deployment.example")
    deployed_names = []
    finishes = []

    async def deploy(**kwargs):
        deployed_names.append(kwargs["deployment_name"])
        return object(), deployment, kwargs["deployment_name"]

    async def poll(**_kwargs):
        return {"score": 1}

    async def finish(**kwargs):
        finishes.append(kwargs)

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(basilica, "_deploy_basilica_eval_repo", deploy)
    monkeypatch.setattr(basilica, "_poll_eval_deployment", poll)
    monkeypatch.setattr(basilica, "_finish_evaluation_cost_run", finish)
    monkeypatch.setattr(basilica, "_fetch_attempt_logs", noop)
    monkeypatch.setattr(basilica, "update_environment_logger_labels", lambda *_args, **_kwargs: None)

    result = await basilica._run_single_basilica_eval_repo(
        repo="org/repo",
        model_name="org/base",
        task_type="individual",
        image="image",
        source="source",
        env={},
        gpu_count=2,
        gpu_models=["A100"],
        min_gpu_memory_gb=40,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey={"org/repo": "hotkey"},
        local_logging=True,
    )

    assert result == {"score": 1}
    assert len(deployed_names) == 1
    assert len(finishes) == 1
    assert finishes[0]["run_key"] == f"evaluation:{task_id}:{deployed_names[0]}"
    assert finishes[0]["success"] is True
    assert finishes[0]["psql_db"] is psql_db


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
