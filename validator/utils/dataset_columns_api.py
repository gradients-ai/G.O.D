from __future__ import annotations

import asyncio
import json
from collections import OrderedDict

import httpx
from datasets import DatasetDict
from datasets import load_dataset
from datasets import load_dataset_builder

import validator.core.constants as v_cst
from core.models.payload_models import NewTaskRequestChat
from core.models.payload_models import NewTaskRequestDPO
from core.models.payload_models import NewTaskRequestGrpo
from core.models.payload_models import NewTaskRequestInstructText
from core.models.utility_models import FileFormat
from validator.evaluation.utils import get_default_dataset_config
from validator.utils.logging import get_logger

logger = get_logger(__name__)

_JSON_PREFIX_MAX_BYTES = 1 * 1024 * 1024
_HF_COLUMN_CHECK_TIMEOUT_SEC = 45.0


def _normalize_column_names(columns: list[str | None]) -> list[str]:
    """Dedupe preserving order; drop empties."""
    seen: OrderedDict[str, None] = OrderedDict()
    for c in columns:
        if not c:
            continue
        s = str(c).strip()
        if not s:
            continue
        if s not in seen:
            seen[s] = None
    return list(seen.keys())


def collect_columns_instruct(req: NewTaskRequestInstructText) -> list[str]:
    cols: list[str | None] = [
        req.field_instruction,
        req.field_input,
        req.field_output,
        req.field_system,
    ]
    return _normalize_column_names(cols)


def collect_columns_chat(req: NewTaskRequestChat) -> list[str]:
    chat_col = req.chat_column.strip() if req.chat_column else v_cst.STANDARD_CHAT_MESSAGES_COLUMN
    return _normalize_column_names([chat_col])


def collect_columns_dpo(req: NewTaskRequestDPO) -> list[str]:
    cols = [req.field_prompt, req.field_system, req.field_chosen, req.field_rejected]
    return _normalize_column_names(cols)


def collect_columns_grpo(req: NewTaskRequestGrpo) -> list[str]:
    cols = [req.field_prompt, req.extra_column]
    return _normalize_column_names(cols)


def _hf_column_names_from_features_or_stream(dataset_id: str) -> set[str]:
    """
    Prefer dataset schema from the builder (no full dataset materialization).
    If features are missing (common for some repos), fall back to streaming one row.
    """
    config_name = get_default_dataset_config(dataset_id)

    builder = load_dataset_builder(dataset_id, name=config_name, trust_remote_code=True)
    if builder.info and builder.info.features:
        feat = builder.info.features
        names = list(feat.keys())
        if names:
            return set(names)

    dsd = load_dataset(dataset_id, config_name, streaming=True, trust_remote_code=True)

    if isinstance(dsd, DatasetDict):
        errors: list[str] = []
        for split_name in list(dsd.keys()):
            ds = dsd[split_name]
            try:
                row = next(iter(ds))
                return set(row.keys())
            except StopIteration:
                errors.append(f"split '{split_name}' has no rows")
                continue
            except Exception as ex:
                errors.append(f"split '{split_name}': {ex}")
                continue
        detail = "; ".join(errors) if errors else "no splits"
        raise ValueError(f"Dataset {dataset_id!r} has no readable rows ({detail}).")

    try:
        row = next(iter(dsd))
    except StopIteration as ex:
        raise ValueError(f"Dataset {dataset_id!r} appears to have no rows.") from ex
    return set(row.keys())


async def _hf_column_names_best_effort(dataset_id: str) -> set[str] | None:
    """
    Resolve HF column names when the Hub is reachable. Returns None on timeout or any error
    so task creation can proceed (validation is best-effort for HF).
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_hf_column_names_from_features_or_stream, dataset_id),
            timeout=_HF_COLUMN_CHECK_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "HF dataset column check timed out after %ss for %r; skipping column validation",
            _HF_COLUMN_CHECK_TIMEOUT_SEC,
            dataset_id,
        )
        return None
    except Exception as exc:
        logger.warning(
            "HF dataset column check failed for %r (%s); skipping column validation",
            dataset_id,
            exc,
        )
        return None


def _column_names_from_json_prefix(text: str) -> set[str]:
    """
    Infer column names from the first object in:
    - a JSON array of objects,
    - JSON Lines (first non-empty line that parses as an object),
    - or a single JSON object at the start of the file.

    Operates on a prefix of the file only; may raise if the first row is incomplete in the prefix.
    """
    text = text.lstrip()
    if not text:
        raise ValueError("Empty dataset response.")

    if text.startswith("{"):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, 0)
            if isinstance(obj, dict):
                return set(obj.keys())
        except json.JSONDecodeError:
            pass

    if text.startswith("["):
        idx = 1
        while idx < len(text) and text[idx] in " \t\n\r":
            idx += 1
        if idx < len(text):
            try:
                obj, _ = json.JSONDecoder().raw_decode(text, idx)
                if isinstance(obj, dict):
                    return set(obj.keys())
            except json.JSONDecodeError as ex:
                mib = _JSON_PREFIX_MAX_BYTES // (1024 * 1024)
                raise ValueError(
                    f"Could not parse the first object in the JSON array within the downloaded prefix ({mib} MiB). "
                    "Try JSON Lines, or ensure the first row is not larger than the preview window."
                ) from ex

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") or line.startswith("["):
            try:
                val = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(val, dict):
                return set(val.keys())

    raise ValueError(
        "Could not infer column names from the start of the JSON file. "
        "Supported shapes: JSON array of objects, JSON Lines (one object per line), or a leading JSON object."
    )


async def _http_json_sample_column_names(url: str) -> set[str]:
    """Stream only a prefix of the URL response; parse first row keys without full download."""
    buf = bytearray()
    timeout = httpx.Timeout(60.0, connect=15.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) >= _JSON_PREFIX_MAX_BYTES:
                        break
    except httpx.HTTPError as ex:
        raise ValueError(f"Unable to fetch dataset URL: {ex}") from ex

    text = bytes(buf).decode("utf-8", errors="replace")
    return _column_names_from_json_prefix(text)


async def validate_dataset_columns(
    dataset_ref: str,
    file_format: FileFormat,
    requested_columns: list[str],
) -> None:
    """
    Ensure requested column names exist (S3/JSON: from a streamed prefix; HF: from builder
    features or one streaming row). Hugging Face checks are best-effort: on timeout or any
    hub error, validation is skipped and the request is not rejected.
    """
    cols = _normalize_column_names(requested_columns)
    if not cols:
        return

    dataset_ref = dataset_ref.strip()
    if not dataset_ref:
        raise ValueError("Dataset reference is empty.")

    # Same convention as prepare_text_task / download_and_load_dataset (JSON files at HTTP/S3 URLs)
    if file_format == FileFormat.S3:
        present = await _http_json_sample_column_names(dataset_ref)
        source = dataset_ref[:120] + ("..." if len(dataset_ref) > 120 else "")
        kind = "dataset URL"
    else:
        present = await _hf_column_names_best_effort(dataset_ref)
        if present is None:
            return
        source = dataset_ref[:120] + ("..." if len(dataset_ref) > 120 else "")
        kind = "Hugging Face dataset"

    missing = [c for c in cols if c not in present]
    if missing:
        avail = sorted(present)
        preview = avail[:80]
        suffix = "" if len(avail) <= 80 else f" (showing first {len(preview)} of {len(avail)})"
        raise ValueError(
            f"Missing column(s) for {kind} {source!r}: {missing}. "
            f"Columns present on a sample row: {preview}{suffix}"
        )


async def validate_instruct_task_columns(req: NewTaskRequestInstructText, dataset_ref: str, file_format: FileFormat) -> None:
    await validate_dataset_columns(dataset_ref, file_format, collect_columns_instruct(req))


async def validate_chat_task_columns(req: NewTaskRequestChat, dataset_ref: str, file_format: FileFormat) -> None:
    await validate_dataset_columns(dataset_ref, file_format, collect_columns_chat(req))


async def validate_dpo_task_columns(req: NewTaskRequestDPO, dataset_ref: str, file_format: FileFormat) -> None:
    await validate_dataset_columns(dataset_ref, file_format, collect_columns_dpo(req))


async def validate_grpo_task_columns(req: NewTaskRequestGrpo, dataset_ref: str, file_format: FileFormat) -> None:
    await validate_dataset_columns(dataset_ref, file_format, collect_columns_grpo(req))
