import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import verify_get_request

from miner.endpoints.training_repo import factory_router


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(factory_router())
    app.dependency_overrides[blacklist_low_stake] = lambda: None
    app.dependency_overrides[verify_get_request] = lambda: None
    return TestClient(app)


def test_configured_types_return_repo_and_others_decline(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINER_TOURNAMENT_TYPES", "text, Quant")

    for entered in ("text", "quant"):
        response = client.get(f"/training_repo/{entered}")
        assert response.status_code == 200
        assert response.json()["github_repo"] and response.json()["commit_hash"]
    assert client.get("/training_repo/image").status_code == 404


def test_unset_declines_everything(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINER_TOURNAMENT_TYPES", raising=False)

    for task_type in ("text", "image", "quant"):
        assert client.get(f"/training_repo/{task_type}").status_code == 404
