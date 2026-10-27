"""Narrow runtime dependencies for reusable validator library functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from core.interfaces import KeypairLike


@dataclass(frozen=True)
class TaskRuntimeDeps:
    psql_db: Any
    redis_db: Any | None = None
    httpx_client: httpx.AsyncClient | None = None
    signer: KeypairLike | None = None
