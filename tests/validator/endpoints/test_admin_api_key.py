from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from validator.app.dependencies import get_config
from validator.endpoints import tournament_orchestrator
from validator.endpoints import transfer_balances


ADMIN = "admin-key"
FRONTEND = "frontend-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN)
    monkeypatch.setenv("FRONTEND_API_KEY", FRONTEND)
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


ADMIN_ROUTES = [
    ("delete", "/v1/tournament_orchestrator/remove_trainer/10.0.0.1"),
    ("post", "/tournament/t1/refund"),
    ("post", "/v1/tournament_orchestrator/add_trainer?trainer_ip=10.0.0.1"),
]


@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
def test_admin_routes_refuse_the_frontend_key(client, method, path):
    assert getattr(client, method)(path, headers=_bearer(FRONTEND)).status_code == 403


@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
def test_admin_routes_fail_closed_without_admin_key(client, monkeypatch, method, path):
    monkeypatch.delenv("ADMIN_API_KEY")
    assert getattr(client, method)(path, headers=_bearer("")).status_code in (401, 403)
    assert getattr(client, method)(path, headers=_bearer(FRONTEND)).status_code == 403
    assert getattr(client, method)(path, headers=_bearer(ADMIN)).status_code == 403


def test_admin_key_is_accepted(client):
    assert client.delete(ADMIN_ROUTES[0][1], headers=_bearer(ADMIN)).status_code == 200


def test_read_routes_keep_the_frontend_key(client):
    assert client.get("/v1/tournament_orchestrator/trainers", headers=_bearer(FRONTEND)).status_code == 200
    assert client.get("/v1/tournament_orchestrator/trainers", headers=_bearer(ADMIN)).status_code == 403


def test_errors_are_not_returned_to_the_caller(client, monkeypatch):
    monkeypatch.setattr(
        tournament_orchestrator.tournament_sql, "remove_trainer", AsyncMock(side_effect=RuntimeError("db password=x"))
    )
    response = client.delete(ADMIN_ROUTES[0][1], headers=_bearer(ADMIN))
    assert response.status_code == 500
    assert "password" not in response.text
