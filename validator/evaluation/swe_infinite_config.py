from __future__ import annotations

import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace

import validator.evaluation.constants as vcst
from core.logging import get_logger


logger = get_logger(__name__)


SWE_INFINITE_SERVER_BASE_URL_ENV = "SWE_INFINITE_SERVER_BASE_URL"
SWE_INFINITE_MODEL_BASE_URL_ENV = "SWE_INFINITE_MODEL_BASE_URL"
SWE_INFINITE_MODEL_API_KEY_ENV = "SWE_INFINITE_MODEL_API_KEY"
SWE_INFINITE_EVAL_CONFIG_ENV = "SWE_INFINITE_EVAL_CONFIG_JSON"
# Written by the Basilica public SGLang proxy, read by the evaluator subprocess so it
# can tell "the SWE server never called the model" from "the model call is slow".
SWE_INFINITE_PROXY_ACTIVITY_ENV = "SWE_INFINITE_PROXY_ACTIVITY_FILE"
SWE_INFINITE_PROXY_ACTIVITY_PATH = "/tmp/god_swe_proxy_activity.json"
# Per-turn generation ceiling applied by the proxy when the agent does not send one.
# A MiniSWE turn is a short reasoning block plus one shell command; adapters that never
# emit a stop token otherwise generate tens of thousands of tokens and no task can finish.
# Measured decode rate is ~20 tok/s, so 1024 tokens is ~50s per turn.
# Applied uniformly to every miner, so it does not advantage any submission.
SWE_INFINITE_MAX_COMPLETION_TOKENS = 1024
# Abandon a task whose model was never called instead of waiting out the full task
# timeout. Passed as its own env var rather than a config field: the eval image bakes
# its own copy of this module, and an older image rejects unknown config fields.
SWE_INFINITE_STALL_TIMEOUT_ENV = "SWE_INFINITE_STALL_TIMEOUT_SECONDS"
SWE_INFINITE_STALL_TIMEOUT_SECONDS = 300
SWE_INFINITE_TASK_SELECTION_OVERRIDE_ENV = "SWE_INFINITE_TASK_SELECTION_OVERRIDE_JSON"
DEFAULT_SWE_INFINITE_MODEL_API_KEY = "x"
SWE_INFINITE_AGENT_NAME = "miniswe"
SWE_INFINITE_VETTED_TASK_IDS = (
    73, 87, 101, 139, 152, 158, 202, 256, 261, 262,
    293, 318, 428, 431, 437, 443, 478, 495, 509, 677,
    692, 703, 713, 715, 721, 761, 765, 796, 800, 829,
    851, 852, 898, 966, 971, 975, 1020, 1023, 1077, 1088,
    1113, 1224, 1240, 1263, 1288, 1313, 1325, 1329, 1402,
    1406, 1434, 1436, 1444, 1467, 1481, 1559, 1565, 1582,
    1603, 1609, 1665, 1669, 1686, 1718, 1756, 1761, 1802,
    1819, 1874, 1900, 1905, 1921, 1967, 1991,
)


@dataclass(frozen=True)
class SweInfiniteEvalConfig:
    metadata_url: str = "https://pub-7882418a56434a479bf9a7febd660b36.r2.dev/bugs/metadata.json"
    affinetes_call_path: str = "/call"
    max_iterations: int = 25
    # 900s at the measured ~50s/turn is ~18 agent turns, the same number a 1800s
    # budget bought at a 2048-token cap, for half the wall-clock and half the
    # context growth. Raise this only alongside the per-turn token cap.
    task_timeout_seconds: int = 900
    session_timeout_seconds: int = vcst.ENV_EVAL_SESSION_TIMEOUT
    max_concurrent_requests: int = 1
    connect_max_attempts: int = 3
    connect_retry_backoff_seconds: float = 1.0
    collect_logprobs: bool = False
    model_base_url: str | None = None
    model_api_key: str | None = None

    def with_overrides(self, **overrides) -> "SweInfiniteEvalConfig":
        clean = {
            key: value
            for key, value in overrides.items()
            if value is not None
        }
        return replace(self, **clean)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "SweInfiniteEvalConfig":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("SWE Infinite eval config must be a JSON object")
        allowed_fields = {field.name for field in fields(cls)}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            # The eval image bakes its own copy of this module, so a validator that is
            # newer than the image will send fields this build has never heard of.
            # Dropping them keeps old images runnable instead of failing the eval.
            logger.warning("Ignoring unknown SWE Infinite eval config fields: %s", unknown_fields)
            payload = {key: value for key, value in payload.items() if key in allowed_fields}
        return cls(**payload)


@dataclass(frozen=True)
class SweInfiniteTaskSelectionOverride:
    """Optional smoke-test override for task selection.

    Production task ranges and default task count are intentionally owned by
    core/constants/environments.py, not by SweInfiniteEvalConfig.
    """

    task_id_min: int | None = None
    task_id_max: int | None = None
    num_seeds: int | None = None
    task_ids: tuple[int, ...] = ()

    def is_empty(self) -> bool:
        return (
            self.task_id_min is None
            and self.task_id_max is None
            and self.num_seeds is None
            and not self.task_ids
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "SweInfiniteTaskSelectionOverride":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("SWE Infinite task selection override must be a JSON object")
        allowed_fields = {field.name for field in fields(cls)}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            raise ValueError(f"Unknown SWE Infinite task selection override fields: {unknown_fields}")
        if "task_ids" in payload:
            payload["task_ids"] = tuple(int(task_id) for task_id in payload["task_ids"])
        return cls(**payload)


DEFAULT_SWE_INFINITE_EVAL_CONFIG = SweInfiniteEvalConfig()


def load_swe_infinite_eval_config() -> SweInfiniteEvalConfig:
    raw = os.getenv(SWE_INFINITE_EVAL_CONFIG_ENV, "").strip()
    if not raw:
        return DEFAULT_SWE_INFINITE_EVAL_CONFIG
    return SweInfiniteEvalConfig.from_json(raw)


def load_swe_infinite_task_selection_override() -> SweInfiniteTaskSelectionOverride:
    raw = os.getenv(SWE_INFINITE_TASK_SELECTION_OVERRIDE_ENV, "").strip()
    if not raw:
        return SweInfiniteTaskSelectionOverride()
    return SweInfiniteTaskSelectionOverride.from_json(raw)


def build_swe_infinite_container_env(
    eval_config: SweInfiniteEvalConfig | None = None,
    task_selection_override: SweInfiniteTaskSelectionOverride | None = None,
) -> dict[str, str]:
    server_url = os.getenv(SWE_INFINITE_SERVER_BASE_URL_ENV, "").strip()
    if not server_url:
        raise ValueError(f"{SWE_INFINITE_SERVER_BASE_URL_ENV} is required for SWE Infinite evaluation")

    env = {SWE_INFINITE_SERVER_BASE_URL_ENV: server_url}
    # Always inject config so validator-side defaults (e.g. task_timeout_seconds) reach
    # the container without requiring an image rebuild.
    effective_config = eval_config if eval_config is not None else load_swe_infinite_eval_config()
    env[SWE_INFINITE_EVAL_CONFIG_ENV] = effective_config.to_json()
    if task_selection_override is not None and not task_selection_override.is_empty():
        env[SWE_INFINITE_TASK_SELECTION_OVERRIDE_ENV] = task_selection_override.to_json()
    return env
