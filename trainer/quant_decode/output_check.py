"""Check a decoder's output: exactly the base model's tensor names and shapes, bf16, every value finite.

safetensors layout: 8-byte little-endian u64 header length, a JSON header, then raw tensor data.
Read with numpy only so the check needs neither torch nor the safetensors package.
"""

import json
import math
import struct
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from pydantic import ValidationError

from core.constants.quant import DECODED_TENSOR_DTYPE


SAFETENSORS_SUFFIX = ".safetensors"
SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
SAFETENSORS_METADATA_KEY = "__metadata__"
HEADER_LENGTH_BYTES = 8
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024  # same cap as the safetensors library
BF16_BYTES = 2
BF16_EXPONENT_MASK = 0x7F80  # all exponent bits set = inf or NaN
FINITE_CHECK_CHUNK_BYTES = 64 * 1024 * 1024
MAX_REPORTED_NAMES = 5


class TensorSpec(BaseModel):
    dtype: str
    shape: list[int]


class ExpectedTensors(BaseModel):
    tensors: dict[str, TensorSpec]

    @classmethod
    def from_safetensors_dir(cls, path: Path) -> "ExpectedTensors":
        """Collect every tensor declared by the *.safetensors headers in a base-model snapshot."""
        tensors: dict[str, TensorSpec] = {}
        for file in sorted(path.glob(f"*{SAFETENSORS_SUFFIX}")):
            specs, _, _ = read_safetensors_header(file)
            tensors.update(specs)
        return cls(tensors=tensors)


class DecodeCheck(BaseModel):
    ok: bool
    failure: str | None = None
    n_tensors: int = 0


class _FileHeader(BaseModel):
    path: Path
    specs: dict[str, TensorSpec]
    offsets: dict[str, tuple[int, int]]
    data_start: int


def read_safetensors_header(path: Path) -> tuple[dict[str, TensorSpec], dict[str, tuple[int, int]], int]:
    """Return (specs, data_offsets per tensor relative to the data start, data start). Raises ValueError if corrupt."""
    file_size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(HEADER_LENGTH_BYTES)
        if len(prefix) != HEADER_LENGTH_BYTES:
            raise ValueError("file shorter than header length prefix")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"header length {header_len} exceeds cap")
        data_start = HEADER_LENGTH_BYTES + header_len
        if data_start > file_size:
            raise ValueError("header length beyond file size")
        try:
            header = json.loads(f.read(header_len))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"header is not JSON: {e}") from e
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")

    specs: dict[str, TensorSpec] = {}
    offsets: dict[str, tuple[int, int]] = {}
    for name, entry in header.items():
        if name == SAFETENSORS_METADATA_KEY:
            continue
        try:
            specs[name] = TensorSpec(dtype=entry["dtype"], shape=entry["shape"])
            begin, end = (int(x) for x in entry["data_offsets"])
        except (KeyError, TypeError, ValueError, ValidationError) as e:
            raise ValueError(f"bad entry for {name}: {e}") from e
        if not 0 <= begin <= end or data_start + end > file_size:
            raise ValueError(f"offsets of {name} beyond file size")
        offsets[name] = (begin, end)
    return specs, offsets, data_start


def _fail(failure: str) -> DecodeCheck:
    return DecodeCheck(ok=False, failure=failure)


def _unexpected_file(output_dir: Path) -> str | None:
    """First entry that is not a regular *.safetensors or index file. The index is accepted but ignored by the
    check; the decode job deletes it, so only the checked shards reach vLLM."""
    for entry in sorted(output_dir.iterdir()):
        if entry.is_symlink():  # would resolve on the host, e.g. to the base model's own weights
            return entry.name
        is_safetensors = entry.is_file() and entry.suffix == SAFETENSORS_SUFFIX
        is_index = entry.is_file() and entry.name == SAFETENSORS_INDEX_NAME
        if not (is_safetensors or is_index):
            return entry.name
    return None


def _name_mismatch(found: set[str], expected: set[str]) -> str | None:
    missing = sorted(expected - found)[:MAX_REPORTED_NAMES]
    extra = sorted(found - expected)[:MAX_REPORTED_NAMES]
    if not missing and not extra:
        return None
    return f"tensor names differ from base model; missing {missing}, extra {extra}"


def _spec_mismatch(headers: list[_FileHeader], expected: ExpectedTensors) -> str | None:
    for header in headers:
        for name, spec in header.specs.items():
            if spec.dtype != DECODED_TENSOR_DTYPE:
                return f"tensor {name} has dtype {spec.dtype}, expected {DECODED_TENSOR_DTYPE}"
            if spec.shape != expected.tensors[name].shape:
                return f"tensor {name} has shape {spec.shape}, expected {expected.tensors[name].shape}"
    return None


def _byte_length_mismatch(headers: list[_FileHeader]) -> str | None:
    for header in headers:
        for name, (begin, end) in header.offsets.items():
            expected_bytes = math.prod(header.specs[name].shape) * BF16_BYTES
            if end - begin != expected_bytes:
                return f"tensor {name} has {end - begin} data bytes, expected {expected_bytes}"
    return None


def _has_non_finite(data: np.memmap, start: int, stop: int) -> bool:
    for chunk_start in range(start, stop, FINITE_CHECK_CHUNK_BYTES):
        chunk = data[chunk_start : min(chunk_start + FINITE_CHECK_CHUNK_BYTES, stop)].view("<u2")
        if np.any((chunk & BF16_EXPONENT_MASK) == BF16_EXPONENT_MASK):
            return True
    return False


def _first_non_finite(headers: list[_FileHeader]) -> str | None:
    for header in headers:
        data = np.memmap(header.path, dtype=np.uint8, mode="r")
        for name, (begin, end) in header.offsets.items():
            if _has_non_finite(data, header.data_start + begin, header.data_start + end):
                return f"tensor {name} has non-finite values"
    return None


def _read_file_header(file: Path) -> _FileHeader:
    try:
        specs, offsets, data_start = read_safetensors_header(file)
    except ValueError as e:
        raise ValueError(f"in {file.name}: {e}") from e
    except OSError as e:
        raise ValueError(f"in {file.name}: unreadable file ({e.strerror})") from e
    return _FileHeader(path=file, specs=specs, offsets=offsets, data_start=data_start)


def _duplicate_name(headers: list[_FileHeader]) -> str | None:
    seen: set[str] = set()
    for header in headers:
        for name in header.specs:
            if name in seen:
                return name
            seen.add(name)
    return None


def check_decoded_output(output_dir: Path, expected: ExpectedTensors) -> DecodeCheck:
    """Validate decoded output against the base model's tensors. The first failure wins; never raises on bad output."""
    try:
        unexpected = _unexpected_file(output_dir)
    except OSError as e:
        return _fail(f"unreadable output ({e.strerror})")
    if unexpected is not None:
        return _fail(f"unexpected file {unexpected}")
    files = sorted(output_dir.glob(f"*{SAFETENSORS_SUFFIX}"))
    if not files:
        return _fail("no safetensors output")

    try:
        headers = [_read_file_header(file) for file in files]
    except ValueError as e:
        return _fail(f"corrupt header {e}")
    duplicate = _duplicate_name(headers)
    if duplicate is not None:
        return _fail(f"tensor {duplicate} appears in more than one file")

    found = {name for header in headers for name in header.specs}
    failure = (
        _name_mismatch(found, set(expected.tensors))
        or _spec_mismatch(headers, expected)
        or _byte_length_mismatch(headers)
        or _first_non_finite(headers)
    )
    if failure is not None:
        return _fail(failure)
    return DecodeCheck(ok=True, n_tensors=len(found))
