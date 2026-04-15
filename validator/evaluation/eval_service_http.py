"""Lightweight HTTP path helpers for eval service mode.

Kept free of validator DB / async dependencies so containers that run
``service_mode`` alone do not need asyncpg or the full utils import graph.
"""

import re
from urllib.parse import unquote
from urllib.parse import urlparse

EVAL_RESULT_STATUS_PATH = "/result"


def eval_service_normalize_http_path(raw_path: str) -> str:
    """Return path without query; strip trailing slash. dstack may send ``/proxy/services/.../result``."""
    p = urlparse(raw_path).path
    p = unquote(p)
    p = re.sub(r"/+", "/", p)
    if not p.startswith("/"):
        p = f"/{p}"
    return p.rstrip("/") or "/"


def _last_path_segment(norm_path: str) -> str:
    parts = [s for s in norm_path.lower().split("/") if s]
    return parts[-1] if parts else ""


def eval_service_path_is_health(norm_path: str) -> bool:
    lp = norm_path.lower()
    if lp == "/health" or lp.endswith("/health"):
        return True
    return _last_path_segment(lp) == "health"


def eval_service_path_is_result(norm_path: str) -> bool:
    lp = norm_path.lower()
    if lp == "/result" or lp.endswith("/result"):
        return True
    return _last_path_segment(lp) == "result"
