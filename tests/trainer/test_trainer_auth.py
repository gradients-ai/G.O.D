import ast
from pathlib import Path

import pytest
from fastapi import Depends
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.auth import verify_orchestrator_ip
from validator.infrastructure.service_constants import TRAINER_AUTH_HEADER


KEY = "trainer-secret"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_IP", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_IPS", "10.0.0.5")
    monkeypatch.setenv("TRAINER_API_KEY", KEY)
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(verify_orchestrator_ip)])
    async def probe():
        return {"ok": True}

    return TestClient(app, client=("10.0.0.5", 50000))


def test_allowed_ip_with_key_is_accepted(client):
    assert client.get("/probe", headers={TRAINER_AUTH_HEADER: KEY}).status_code == 200


@pytest.mark.parametrize("headers", [{}, {TRAINER_AUTH_HEADER: "wrong"}, {TRAINER_AUTH_HEADER: ""}])
def test_missing_or_wrong_key_is_rejected(client, headers):
    assert client.get("/probe", headers=headers).status_code == 403


def test_unset_key_rejects_every_request(client, monkeypatch):
    monkeypatch.delenv("TRAINER_API_KEY")
    assert client.get("/probe", headers={TRAINER_AUTH_HEADER: ""}).status_code == 403
    assert client.get("/probe", headers={TRAINER_AUTH_HEADER: KEY}).status_code == 403


def test_ip_outside_the_allowlist_is_rejected_even_with_the_key(client, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_IPS", "10.0.0.6")
    assert client.get("/probe", headers={TRAINER_AUTH_HEADER: KEY}).status_code == 403


def test_there_is_no_default_orchestrator_ip(client, monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_IPS")
    assert client.get("/probe", headers={TRAINER_AUTH_HEADER: KEY}).status_code == 403


def test_localhost_still_needs_the_key(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_IPS", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_IP", raising=False)
    monkeypatch.setenv("TRAINER_API_KEY", KEY)
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(verify_orchestrator_ip)])
    async def probe():
        return {"ok": True}

    local = TestClient(app, client=("127.0.0.1", 50000))
    assert local.get("/probe").status_code == 403
    assert local.get("/probe", headers={TRAINER_AUTH_HEADER: KEY}).status_code == 200


def test_every_trainer_route_is_guarded():
    # trainer/endpoints.py pulls in the whole trainer runtime, so its routes are checked in the source.
    source = (Path(__file__).resolve().parents[2] / "trainer" / "endpoints.py").read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_api_route"
    ]
    assert calls
    for call in calls:
        deps = next(kw.value for kw in call.keywords if kw.arg == "dependencies")
        assert "Depends(verify_orchestrator_ip)" in ast.unparse(deps)


def test_validator_sends_the_key(monkeypatch):
    from validator.infrastructure.trainer_auth import trainer_auth_headers

    monkeypatch.setenv("TRAINER_API_KEY", KEY)
    assert trainer_auth_headers() == {TRAINER_AUTH_HEADER: KEY}
    monkeypatch.delenv("TRAINER_API_KEY")
    assert trainer_auth_headers() == {}

