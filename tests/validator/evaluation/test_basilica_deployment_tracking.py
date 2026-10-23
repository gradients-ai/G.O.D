from types import SimpleNamespace

import pytest

from validator.evaluation import basilica


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
