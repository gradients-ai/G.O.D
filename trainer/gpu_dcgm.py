"""DCGM health as an optional source. Missing dcgmi or a stopped hostengine means "not in use"."""

import json
import subprocess
from typing import Any

from pydantic import BaseModel


_FAIL = {"failure", "fail"}
_WARN = {"warning", "warn"}
_GPU_ID_KEYS = ("GPU ID", "gpuId", "gpu_id", "GPU")


class DcgmSubsystem(BaseModel):
    name: str
    errors: list[str] = []

    @property
    def summary(self) -> str:
        return "; ".join(self.errors) if self.errors else self.name


class DcgmGpuHealth(BaseModel):
    gpu_index: int
    failures: list[DcgmSubsystem] = []
    warnings: list[DcgmSubsystem] = []


def parse_dcgm_health(payload: str) -> list[DcgmGpuHealth]:
    """Turn `dcgmi health -c a -j` into per-GPU failures and warnings.

    NVLink and other subsystems reported as skipped or healthy are ignored, so a PCIe
    box with no NVLink and no fabric manager does not produce a fault.
    """
    data = json.loads(payload)
    body = data.get("body", data) if isinstance(data, dict) else data
    return [_parse_gpu(obj) for obj in _find_gpu_objects(body)]


def query_dcgm_health() -> list[DcgmGpuHealth] | None:
    """Run dcgmi. None when the binary or the hostengine is unavailable."""
    try:
        proc = subprocess.run(
            ["dcgmi", "health", "-c", "a", "-j"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return parse_dcgm_health(proc.stdout)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _find_gpu_objects(node: Any) -> list[dict]:
    found: list[dict] = []
    if isinstance(node, dict):
        if _gpu_index(node) is not None and _subsystem_items(node):
            found.append(node)
        else:
            for value in node.values():
                found.extend(_find_gpu_objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_gpu_objects(item))
    return found


def _parse_gpu(obj: dict) -> DcgmGpuHealth:
    index = _gpu_index(obj)
    assert index is not None
    failures: list[DcgmSubsystem] = []
    warnings: list[DcgmSubsystem] = []
    for name, status, errors in _subsystem_items(obj):
        if status in _FAIL:
            failures.append(DcgmSubsystem(name=name, errors=errors))
        elif status in _WARN:
            warnings.append(DcgmSubsystem(name=name, errors=errors))
    return DcgmGpuHealth(gpu_index=index, failures=failures, warnings=warnings)


def _gpu_index(obj: dict) -> int | None:
    for key in _GPU_ID_KEYS:
        value = obj.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _subsystem_items(obj: dict) -> list[tuple[str, str, list[str]]]:
    items = []
    for key, value in obj.items():
        if key in _GPU_ID_KEYS or key in {"GPU", "Overall Health", "status"}:
            continue
        status, errors = _status_of(value)
        if status is None:
            continue
        items.append((str(key), status.strip().lower(), errors))
    return items


def _status_of(value: Any) -> tuple[str | None, list[str]]:
    if isinstance(value, str):
        return value, []
    if isinstance(value, dict):
        status = value.get("status") or value.get("value") or value.get("Status")
        errors = value.get("errors") or value.get("error") or []
        if isinstance(errors, str):
            errors = [errors]
        if not isinstance(errors, list):
            errors = []
        return (str(status) if status is not None else None), [str(item) for item in errors]
    return None, []
