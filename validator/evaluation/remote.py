"""Backend selection for production remote evaluations."""

from __future__ import annotations

import os
from enum import Enum


class EvalBackend(str, Enum):
    BASILICA = "basilica"
    RUNPOD = "runpod"


def get_eval_backend() -> EvalBackend:
    raw = (os.getenv("EVAL_BACKEND") or EvalBackend.BASILICA.value).strip().lower()
    try:
        return EvalBackend(raw)
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in EvalBackend)
        raise ValueError(f"Unsupported EVAL_BACKEND={raw!r}; expected one of: {supported}") from exc


async def run_remote_eval_repos(**kwargs):
    if get_eval_backend() == EvalBackend.RUNPOD:
        from validator.evaluation.runpod import run_runpod_eval_repos

        return await run_runpod_eval_repos(**kwargs)

    from validator.evaluation.basilica import run_basilica_eval_repos

    return await run_basilica_eval_repos(**kwargs)
