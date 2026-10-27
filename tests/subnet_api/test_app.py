import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

from subnet_api.app import create_app
from subnet_api.settings import Settings


API_KEY = "validator-key-for-tests"
HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX"
TASK_ID = "7f9c2c4e-3c1b-4d55-9a57-1f0f3b6c9b10"

TASK = {
    "is_organic": True,
    "task_id": TASK_ID,
    "status": "success",
    "model_id": "unsloth/Llama-3.2-3B-Instruct",
    "ds": "org/dataset",
    "account_id": "4d0b7a49-0c76-4f7e-9e0b-6c3e7a9f3d21",
    "hours_to_complete": 2,
    "created_at": "2026-10-01T00:00:00Z",
    "task_type": "DpoTask",
    "training_start_point": "default",
    "backend": "some-new-backend",
    "field_prompt": "prompt",
    "field_chosen": "chosen",
    "field_rejected": "rejected",
    "file_format": "hf",
    "result_model_name": None,
    "validator_secret_token": "must-not-leak",
}

DETAILS = {
    "tournament_id": "tourn_abc",
    "tournament_type": "text",
    "status": "completed",
    "base_winner_hotkey": HOTKEY,
    "winner_hotkey": HOTKEY,
    "diff_report": "internal",
    "participants": [
        {
            "hotkey": HOTKEY,
            "final_position": 1,
            "training_repo": "https://github.com/miner/private",
            "training_commit_hash": "abc123",
            "backup_repo": "private/backup",
        }
    ],
    "rounds": [],
    "final_scores": [{"hotkey": HOTKEY, "score": 1.0}],
    "text_tournament_weight": 0.5,
    "image_tournament_weight": 0.2,
}


class FakeValidator:
    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.routes: dict[str, httpx.Response] = {}
        self.delay = 0.0

    def respond(self, path: str, status_code: int = 200, json=None) -> None:
        self.routes[path] = httpx.Response(status_code, json=json)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        response = self.routes.get(request.url.path)
        if response is None:
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(response.status_code, content=response.content, headers=response.headers)


@asynccontextmanager
async def client(validator: FakeValidator, cache_seconds: float = 30):
    settings = Settings(validator_url="http://validator.test", validator_api_key=API_KEY, cache_seconds=cache_seconds)
    app = create_app(settings, transport=httpx.MockTransport(validator.handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            yield api


@pytest.mark.asyncio
async def test_paths_are_rewritten_and_key_is_sent() -> None:
    validator = FakeValidator()
    validator.respond(
        f"/transfer/balance/{HOTKEY}",
        json={"coldkey": HOTKEY, "balance_rao": 5, "total_sent_rao": 5, "transfer_count": 1},
    )
    validator.respond("/v1/tournaments/latest/details", json={"text": DETAILS})
    async with client(validator) as api:
        balance = await api.get(f"/tournament/balance/{HOTKEY}")
        latest = await api.get("/tournament/latest/details")
    assert balance.status_code == 200 and balance.json()["balance_rao"] == 5
    assert latest.status_code == 200
    assert [c.url.path for c in validator.calls] == [f"/transfer/balance/{HOTKEY}", "/v1/tournaments/latest/details"]
    assert all(c.headers["authorization"] == f"Bearer {API_KEY}" for c in validator.calls)
    assert API_KEY not in balance.text + latest.text


@pytest.mark.asyncio
async def test_tournament_details_drop_private_participant_fields() -> None:
    validator = FakeValidator()
    validator.respond("/v1/tournaments/tourn_abc/details", json=DETAILS)
    async with client(validator) as api:
        response = await api.get("/tournament/tourn_abc/details")
    assert response.status_code == 200
    body = response.json()
    assert body["participants"] == [{"hotkey": HOTKEY, "eliminated_in_round_id": None, "final_position": 1}]
    assert "diff_report" not in body
    assert "private" not in response.text and "abc123" not in response.text


@pytest.mark.asyncio
async def test_audit_tasks_keep_type_fields_and_drop_unknown_ones() -> None:
    validator = FakeValidator()
    hotkey_task = TASK | {"hotkey_details": [{"hotkey": HOTKEY, "quality_score": 0.9, "repo": "org/model", "rank": 1}]}
    validator.respond("/auditing/tasks", json=[TASK, TASK | {"task_type": "SomeFutureTask"}])
    validator.respond(f"/auditing/tasks/{TASK_ID}", json=hotkey_task)
    async with client(validator) as api:
        listed = await api.get("/auditing/tasks", params={"limit": 5})
        single = await api.get(f"/auditing/tasks/{TASK_ID}")
    assert listed.status_code == 200
    dpo, future = listed.json()
    assert dpo["field_chosen"] == "chosen" and dpo["backend"] == "some-new-backend"
    assert future["task_type"] == "SomeFutureTask" and "field_chosen" not in future
    assert single.json()["hotkey_details"][0]["repo"] == "org/model"
    assert listed.headers.get("content-encoding") == "gzip"
    assert "must-not-leak" not in listed.text + single.text
    assert validator.calls[0].url.params["limit"] == "5"


@pytest.mark.asyncio
async def test_fees_and_weights_accept_responses_without_environment_fields() -> None:
    validator = FakeValidator()
    validator.respond("/tournament/fees", json={"text_tournament_fee_rao": 1, "image_tournament_fee_rao": 2})
    validator.respond(
        "/v1/performance/latest-tournament-weights",
        json={
            "burn_data": {
                "text_performance_diff": None,
                "image_performance_diff": 0.1,
                "text_burn_proportion": 0.2,
                "image_burn_proportion": 0.3,
                "text_tournament_weight": 0.4,
                "image_tournament_weight": 0.5,
                "burn_weight": 0.1,
            },
            "text_top_miners": [{"hotkey": HOTKEY, "rank": 1, "weight": 0.4}],
            "image_top_miners": [],
        },
    )
    async with client(validator) as api:
        fees = await api.get("/tournament/fees")
        weights = await api.get("/v1/performance/latest-tournament-weights")
    assert fees.status_code == 200 and fees.json()["environment_tournament_fee_rao"] is None
    assert weights.status_code == 200 and weights.json()["text_top_miners"][0]["rank"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["/tournament/balance/not-a-key", "/auditing/tasks/not-a-uuid", "/auditing/tasks?limit=1000", "/tournament/bad%20id/details"],
)
async def test_bad_input_is_rejected_without_calling_validator(path: str) -> None:
    validator = FakeValidator()
    async with client(validator) as api:
        response = await api.get(path)
    assert response.status_code in (404, 422)
    assert validator.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream", "expected"),
    [(404, 404), (400, 400), (401, 502), (403, 502), (500, 502), (503, 502)],
)
async def test_validator_errors_are_mapped(upstream: int, expected: int) -> None:
    validator = FakeValidator()
    validator.respond("/auditing/scores-url", upstream, json={"detail": "No scores url found"})
    async with client(validator) as api:
        response = await api.get("/auditing/scores-url")
    assert response.status_code == expected
    if expected >= 500:
        assert response.json() == {"detail": "Validator request failed"}


@pytest.mark.asyncio
async def test_unreachable_or_malformed_validator_returns_502() -> None:
    async def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    settings = Settings(validator_url="http://validator.test", validator_api_key=API_KEY)
    app = create_app(settings, transport=httpx.MockTransport(unreachable))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            down = await api.get("/tournament/fees")
    assert down.status_code == 502

    validator = FakeValidator()
    validator.respond("/tournament/fees", json={"unexpected": True})
    async with client(validator) as api:
        malformed = await api.get("/tournament/fees")
    assert malformed.status_code == 502


@pytest.mark.asyncio
async def test_responses_are_cached_and_concurrent_misses_share_one_request() -> None:
    validator = FakeValidator()
    validator.delay = 0.05
    validator.respond("/tournament/fees", json={"text_tournament_fee_rao": 1, "image_tournament_fee_rao": 2})
    async with client(validator) as api:
        results = await asyncio.gather(*(api.get("/tournament/fees") for _ in range(10)))
        again = await api.get("/tournament/fees")
    assert all(r.status_code == 200 for r in [*results, again])
    assert len(validator.calls) == 1


@pytest.mark.asyncio
async def test_failures_are_not_cached() -> None:
    validator = FakeValidator()
    validator.respond("/tournament/fees", 500, json={})
    async with client(validator) as api:
        first = await api.get("/tournament/fees")
        validator.respond("/tournament/fees", json={"text_tournament_fee_rao": 1, "image_tournament_fee_rao": 2})
        second = await api.get("/tournament/fees")
    assert (first.status_code, second.status_code) == (502, 200)


def test_settings_require_validator_url_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VALIDATOR_URL", raising=False)
    monkeypatch.setenv("VALIDATOR_API_KEY", API_KEY)
    with pytest.raises(RuntimeError):
        Settings.from_env()
    monkeypatch.setenv("VALIDATOR_URL", "http://validator.test/")
    settings = Settings.from_env()
    assert settings.validator_url == "http://validator.test"
    assert API_KEY not in repr(settings)
