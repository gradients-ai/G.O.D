"""Small async client for the dstack HTTP API used by training and evaluation."""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin

import httpx

from core.models.payload_models import DstackRunStatus
from validator.infrastructure.service_constants import DSTACK_LOGS_POLL_ENDPOINT
from validator.infrastructure.service_constants import DSTACK_RUNS_APPLY_ENDPOINT
from validator.infrastructure.service_constants import DSTACK_RUNS_DELETE_ENDPOINT
from validator.infrastructure.service_constants import DSTACK_RUNS_GET_ENDPOINT
from validator.infrastructure.service_constants import DSTACK_RUNS_LIST_ENDPOINT
from validator.infrastructure.service_constants import DSTACK_RUNS_STOP_ENDPOINT


@dataclass(frozen=True)
class DstackConfig:
    url: str
    token: str
    project: str


@dataclass(frozen=True)
class DstackLogPage:
    messages: list[str]
    next_token: str | None


def _decode_log_message(value: object) -> str:
    message = str(value or "")
    if not message or len(message) % 4:
        return message
    try:
        decoded = base64.b64decode(message, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return message
    if not decoded or any(ord(char) < 9 for char in decoded):
        return message
    return decoded.rstrip()


def load_dstack_config(*, required: bool = True) -> DstackConfig | None:
    values = {
        "url": (os.getenv("DSTACK_URL") or "").rstrip("/"),
        "token": os.getenv("DSTACK_TOKEN") or "",
        "project": os.getenv("DSTACK_PROJECT") or "",
    }
    missing = [f"DSTACK_{key.upper()}" for key, value in values.items() if not value]
    if missing:
        if required:
            raise RuntimeError(f"Missing dstack configuration: {', '.join(missing)}")
        return None
    return DstackConfig(**values)


class DstackClient:
    def __init__(self, config: DstackConfig | None = None, *, timeout: float = 30.0):
        self.config = config or load_dstack_config(required=True)
        assert self.config is not None
        self.timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.token}",
            "Content-Type": "application/json",
        }

    def _project_url(self, endpoint: str) -> str:
        return f"{self.config.url}{endpoint.format(project=self.config.project)}"

    async def _post(self, url: str, payload: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=self._headers, json=payload)
        response.raise_for_status()
        return response

    async def apply_run(self, plan: dict) -> dict:
        response = await self._post(self._project_url(DSTACK_RUNS_APPLY_ENDPOINT), plan)
        return response.json()

    async def get_run(self, run_name: str) -> dict:
        response = await self._post(
            self._project_url(DSTACK_RUNS_GET_ENDPOINT),
            {"run_name": run_name},
        )
        return response.json()

    async def get_run_status(self, run_name: str) -> DstackRunStatus:
        return DstackRunStatus.model_validate(await self.get_run(run_name))

    async def list_active_runs(self, *, limit: int = 100) -> list[dict]:
        response = await self._post(
            f"{self.config.url}{DSTACK_RUNS_LIST_ENDPOINT}",
            {
                "project_name": self.config.project,
                "only_active": True,
                "include_jobs": True,
                "job_submissions_limit": 1,
                "limit": limit,
            },
        )
        return response.json()

    async def stop_run(self, run_name: str, *, abort: bool = True) -> None:
        await self._post(
            self._project_url(DSTACK_RUNS_STOP_ENDPOINT),
            {"runs_names": [run_name], "abort": abort},
        )

    async def delete_run(self, run_name: str) -> None:
        await self._post(
            self._project_url(DSTACK_RUNS_DELETE_ENDPOINT),
            {"runs_names": [run_name]},
        )

    async def poll_logs(
        self,
        run_name: str,
        job_submission_id: str,
        *,
        next_token: str | None = None,
        limit: int = 1000,
    ) -> DstackLogPage:
        payload = {
            "run_name": run_name,
            "job_submission_id": job_submission_id,
            "limit": limit,
        }
        if next_token:
            payload["next_token"] = next_token
        response = await self._post(self._project_url(DSTACK_LOGS_POLL_ENDPOINT), payload)
        body = response.json()
        messages = [
            _decode_log_message(event.get("message", ""))
            for event in body.get("logs", [])
            if str(event.get("message", "")).strip()
        ]
        return DstackLogPage(messages=messages, next_token=body.get("next_token"))

    def resolve_service_url(self, run: dict) -> str | None:
        service = run.get("service") or {}
        value = service.get("url")
        if not value:
            return None
        return urljoin(f"{self.config.url}/", str(value).lstrip("/"))


def parse_dstack_submitted_at(run: dict) -> datetime | None:
    value = run.get("submitted_at")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
