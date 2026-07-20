"""Live TAO -> USD price lookup.

Kept intentionally tiny and best-effort: callers (e.g. the cost dashboard) use this only to
annotate TAO-denominated amounts with an approximate USD value, so a lookup failure must never
break the caller. We cache briefly so repeated refreshes don't hammer the source.
"""

import asyncio
import time

import requests

from core.logging import get_logger


logger = get_logger(__name__)

_COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
_CACHE_TTL_SECONDS = 60

_cached_price: float | None = None
_cached_at: float = 0.0


async def get_tao_price_usd() -> float | None:
    """Return the current TAO price in USD, or the last known value (possibly None) on failure."""
    global _cached_price, _cached_at

    now = time.monotonic()
    if _cached_price is not None and (now - _cached_at) < _CACHE_TTL_SECONDS:
        return _cached_price

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(
                _COINGECKO_URL,
                params={"ids": "bittensor", "vs_currencies": "usd"},
                timeout=5,
            ),
        )
        response.raise_for_status()
        price = float(response.json()["bittensor"]["usd"])
        _cached_price = price
        _cached_at = now
        return price
    except Exception as exc:
        logger.warning(f"Failed to fetch live TAO/USD price: {exc}")
        return _cached_price
