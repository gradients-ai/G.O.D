from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from validator.app.dependencies import get_config
from validator.endpoints import tournament_orchestrator
from validator.endpoints import transfer_balances


KEY = "frontend-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FRONTEND_API_KEY", KEY)
    monkeypatch.setattr(tournament_orchestrator.tournament_sql, "remove_trainer", AsyncMock())
    monkeypatch.setattr(tournament_orchestrator.tournament_sql, "get_trainers", AsyncMock(return_value=[]))
    monkeypatch.setattr(transfer_balances, "refund_tournament_participants", AsyncMock(return_value=0))
    app = FastAPI()
    app.include_router(tournament_orchestrator.factory_router())
    app.include_router(transfer_balances.factory_router())
    app.dependency_overrides[get_config] = lambda: MagicMock()
    return TestClient(app)


def _bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


ROUTES = [
    ("delete", "/v1/tournament_orchestrator/remove_trainer/10.0.0.1"),
    ("post", "/tournament/t1/refund"),
    ("get", "/v1/tournament_orchestrator/trainers"),
]


@pytest.mark.parametrize("method,path", ROUTES)
def test_protected_routes_refuse_a_wrong_key(client, method, path):
    assert getattr(client, method)(path, headers=_bearer("wrong-key")).status_code == 403


@pytest.mark.parametrize("method,path", ROUTES)
def test_protected_routes_accept_the_api_key(client, method, path):
    assert getattr(client, method)(path, headers=_bearer(KEY)).status_code == 200


def test_errors_are_not_returned_to_the_caller(client, monkeypatch):
    monkeypatch.setattr(
        tournament_orchestrator.tournament_sql, "remove_trainer", AsyncMock(side_effect=RuntimeError("db password=x"))
    )
    response = client.delete(ROUTES[0][1], headers=_bearer(KEY))
    assert response.status_code == 500
    assert "password" not in response.text
