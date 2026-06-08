"""HTTP compatibility sidecar for InterCode model-prep rollouts."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from validator.evaluation.eval_intercode import DEFAULT_MAX_TOKENS_PER_CALL
from validator.evaluation.eval_intercode import DEFAULT_MAX_TURNS
from validator.evaluation.eval_intercode import DEFAULT_PER_TASK_TIMEOUT_SECONDS
from validator.evaluation.eval_intercode import InterCodeAssets
from validator.evaluation.eval_intercode import load_intercode_assets
from validator.evaluation.eval_intercode import run_intercode_task


logger = logging.getLogger(__name__)

app = FastAPI()
_assets: InterCodeAssets | None = None
_eval_lock = asyncio.Lock()


class InterCodeEvaluateRequest(BaseModel):
    model: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    task_id: int
    temperature: float = 0.0
    seed: int | None = None


def _normalize_openai_base_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _get_assets() -> InterCodeAssets:
    global _assets
    if _assets is None:
        _assets = load_intercode_assets()
        logger.info("intercode_server loaded %s tasks across %s", _assets.total_tasks, _assets.ranges)
    return _assets


def _format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    return message or type(exc).__name__


@app.on_event("startup")
async def startup() -> None:
    _get_assets()


@app.get("/health")
async def health() -> dict[str, str | int]:
    assets = _get_assets()
    return {"status": "ok", "tasks": assets.total_tasks}


@app.post("/evaluate")
async def evaluate(payload: InterCodeEvaluateRequest) -> dict:
    assets = _get_assets()
    start = time.time()
    openai_base_url = _normalize_openai_base_url(payload.base_url)
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy"), base_url=openai_base_url)
    max_turns = int(os.getenv("INTERCODE_MAX_TURNS", str(DEFAULT_MAX_TURNS)))
    max_tokens_per_call = int(os.getenv("INTERCODE_MAX_TOKENS_PER_CALL", str(DEFAULT_MAX_TOKENS_PER_CALL)))
    per_task_timeout = int(os.getenv("INTERCODE_PER_TASK_TIMEOUT", str(DEFAULT_PER_TASK_TIMEOUT_SECONDS)))

    try:
        async with _eval_lock:
            score = await run_intercode_task(
                payload.task_id,
                assets,
                client,
                payload.model,
                payload.temperature,
                max_turns=max_turns,
                max_tokens_per_call=max_tokens_per_call,
                per_task_timeout=per_task_timeout,
            )
        error = None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning(
            "intercode_server task_id=%s seed=%s failed: %s",
            payload.task_id,
            payload.seed,
            exc,
            exc_info=True,
        )
        score = 0.0
        error = _format_exception(exc)

    result = {
        "score": float(score),
        "time_taken": time.time() - start,
        "task_id": payload.task_id,
    }
    if error:
        result["error"] = error
    return {"result": result}
