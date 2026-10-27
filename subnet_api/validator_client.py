import asyncio
import time
from typing import Any

import httpx
from fastapi import HTTPException
from loguru import logger


CacheKey = tuple[str, tuple[tuple[str, str], ...]]


class ValidatorClient:
    """GETs from the validator with a short TTL cache.

    Concurrent misses for the same key share one upstream request, so a burst of public traffic
    costs the validator a single query per key per TTL.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        timeout: float,
        cache_seconds: float,
        max_entries: int = 1024,
    ) -> None:
        self.http = http
        self.timeout = timeout
        self.cache_seconds = cache_seconds
        self.max_entries = max_entries
        self._cache: dict[CacheKey, tuple[float, Any]] = {}
        self._inflight: dict[CacheKey, asyncio.Future] = {}

    async def get(self, path: str, params: dict[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        clean = tuple(sorted((k, str(v)) for k, v in (params or {}).items() if v is not None))
        key = (path, clean)
        cached = self._cache.get(key)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        if key in self._inflight:
            return await asyncio.shield(self._inflight[key])

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            body = await self._fetch(path, dict(clean), timeout or self.timeout)
            self._store(key, body)
            future.set_result(body)
            return body
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved so waiter-less failures are not logged as unhandled
            raise
        finally:
            del self._inflight[key]

    async def _fetch(self, path: str, params: dict[str, str], timeout: float) -> Any:
        try:
            response = await self.http.get(path, params=params, timeout=timeout)
        except httpx.HTTPError as exc:
            logger.warning(f"Validator request failed: {path} ({type(exc).__name__})")
            raise HTTPException(status_code=502, detail="Validator unavailable") from exc
        if response.status_code == 200:
            return response.json()
        if 400 <= response.status_code < 500 and response.status_code not in (401, 403):
            raise HTTPException(status_code=response.status_code, detail=_detail(response))
        logger.warning(f"Validator returned {response.status_code} for {path}")
        raise HTTPException(status_code=502, detail="Validator request failed")

    def _store(self, key: CacheKey, body: Any) -> None:
        if self.cache_seconds <= 0:
            return
        now = time.monotonic()
        if len(self._cache) >= self.max_entries:
            self._cache = {k: v for k, v in self._cache.items() if v[0] > now}
            if len(self._cache) >= self.max_entries:
                self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (now + self.cache_seconds, body)


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return detail if isinstance(detail, str) else "Request failed"
