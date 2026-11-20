"""Decoded-output check on hand-built safetensors files (numpy bf16 bit patterns, hand-written JSON header)."""

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from trainer.quant_decode import output_check
from trainer.quant_decode.output_check import ExpectedTensors
from trainer.quant_decode.output_check import TensorSpec
from trainer.quant_decode.output_check import check_decoded_output


BF16_ONE = 0x3F80
BF16_NAN = 0x7FC0
BF16_INF = 0x7F80

BASE_TENSORS = {
    "embed.weight": [4, 3],
    "layer.0.weight": [3, 3],
    "norm.weight": [3],
}


def _bf16(shape: list[int], fill: int = BF16_ONE) -> np.ndarray:
    return np.full(shape, fill, dtype="<u2")


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], dtype: str = "BF16", truncate: int = 0) -> None:
    header: dict = {"__metadata__": {"format": "pt"}}
    blobs: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        blob = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    header_bytes = json.dumps(header).encode()
    payload = struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(blobs)
    path.write_bytes(payload[: len(payload) - truncate])


def _expected() -> ExpectedTensors:
    return ExpectedTensors(tensors={name: TensorSpec(dtype="BF16", shape=shape) for name, shape in BASE_TENSORS.items()})


def _good_tensors() -> dict[str, np.ndarray]:
    return {name: _bf16(shape) for name, shape in BASE_TENSORS.items()}


def _check(tmp_path: Path, tensors: dict[str, np.ndarray], **kwargs) -> str | None:
    write_safetensors(tmp_path / "model.safetensors", tensors, **kwargs)
    result = check_decoded_output(tmp_path, _expected())
    assert not result.ok
    return result.failure


def test_happy_path_ok(tmp_path: Path):
    write_safetensors(tmp_path / "model-00001.safetensors", {"embed.weight": _bf16([4, 3])})
    write_safetensors(tmp_path / "model-00002.safetensors", {"layer.0.weight": _bf16([3, 3]), "norm.weight": _bf16([3])})
    (tmp_path / "model.safetensors.index.json").write_text("{}")

    result = check_decoded_output(tmp_path, _expected())

    assert result.ok
    assert result.failure is None
    assert result.n_tensors == 3


def test_no_output_fails(tmp_path: Path):
    result = check_decoded_output(tmp_path, _expected())
    assert not result.ok
    assert result.failure == "no safetensors output"


def test_extra_tensor_fails(tmp_path: Path):
    tensors = {**_good_tensors(), "lm_head.weight": _bf16([3, 4])}
    failure = _check(tmp_path, tensors)
    assert "extra ['lm_head.weight']" in failure


def test_missing_tensor_fails(tmp_path: Path):
    tensors = _good_tensors()
    del tensors["norm.weight"]
    failure = _check(tmp_path, tensors)
    assert "missing ['norm.weight']" in failure


def test_f32_dtype_fails(tmp_path: Path):
    failure = _check(tmp_path, _good_tensors(), dtype="F32")
    assert "dtype F32" in failure


def test_wrong_shape_fails(tmp_path: Path):
    tensors = {**_good_tensors(), "norm.weight": _bf16([4])}
    failure = _check(tmp_path, tensors)
    assert failure == "tensor norm.weight has shape [4], expected [3]"


def test_nan_fails(tmp_path: Path):
    layer = _bf16([3, 3])
    layer[1, 2] = BF16_NAN
    failure = _check(tmp_path, {**_good_tensors(), "layer.0.weight": layer})
    assert failure == "tensor layer.0.weight has non-finite values"


def test_inf_fails(tmp_path: Path):
    norm = _bf16([3])
    norm[0] = BF16_INF
    failure = _check(tmp_path, {**_good_tensors(), "norm.weight": norm})
    assert failure == "tensor norm.weight has non-finite values"


def test_negative_inf_fails(tmp_path: Path):
    norm = _bf16([3])
    norm[2] = 0xFF80
    failure = _check(tmp_path, {**_good_tensors(), "norm.weight": norm})
    assert failure == "tensor norm.weight has non-finite values"


def test_stray_decoder_file_fails(tmp_path: Path):
    (tmp_path / "decode.py").write_text("print('hi')")
    failure = _check(tmp_path, _good_tensors())
    assert failure == "unexpected file decode.py"


def test_subdirectory_fails(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    failure = _check(tmp_path, _good_tensors())
    assert failure == "unexpected file nested"


def test_symlinked_output_fails(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    write_safetensors(base / "model.safetensors", _good_tensors())
    output = tmp_path / "output"
    output.mkdir()
    (output / "model.safetensors").symlink_to(base / "model.safetensors")

    result = check_decoded_output(output, _expected())

    assert not result.ok
    assert result.failure == "unexpected file model.safetensors"


def test_name_duplicated_across_files_fails(tmp_path: Path):
    write_safetensors(tmp_path / "a.safetensors", {"embed.weight": _bf16([4, 3]), "norm.weight": _bf16([3])})
    write_safetensors(tmp_path / "b.safetensors", {"layer.0.weight": _bf16([3, 3]), "norm.weight": _bf16([3])})

    result = check_decoded_output(tmp_path, _expected())

    assert not result.ok
    assert result.failure == "tensor norm.weight appears in more than one file"


def test_truncated_data_fails(tmp_path: Path):
    failure = _check(tmp_path, _good_tensors(), truncate=2)
    assert failure.startswith("corrupt header in model.safetensors")


def test_garbage_header_fails(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", 4) + b"\xff\xfe{x")
    result = check_decoded_output(tmp_path, _expected())
    assert not result.ok
    assert result.failure.startswith("corrupt header")


def test_oversized_header_length_fails(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", 1024**3))

    result = check_decoded_output(tmp_path, _expected())

    assert not result.ok
    assert result.failure.startswith("corrupt header")
    assert f"header length {1024**3} exceeds cap" in result.failure


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads files regardless of mode")
def test_unreadable_shard_fails_without_raising(tmp_path: Path):
    shard = tmp_path / "model.safetensors"
    write_safetensors(shard, _good_tensors())
    shard.chmod(0o000)
    try:
        result = check_decoded_output(tmp_path, _expected())
    finally:
        shard.chmod(0o644)

    assert not result.ok
    assert result.failure.startswith("corrupt header in model.safetensors: unreadable file")


def test_byte_length_mismatch_fails(tmp_path: Path):
    offsets = {"embed.weight": [0, 24], "layer.0.weight": [0, 18], "norm.weight": [0, 4]}
    header = {name: {"dtype": "BF16", "shape": BASE_TENSORS[name], "data_offsets": span} for name, span in offsets.items()}
    header_bytes = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + _bf16([12]).tobytes())

    result = check_decoded_output(tmp_path, _expected())

    assert not result.ok
    assert result.failure == "tensor norm.weight has 4 data bytes, expected 6"


def test_from_safetensors_dir_round_trips(tmp_path: Path):
    write_safetensors(tmp_path / "model-00001.safetensors", {"embed.weight": _bf16([4, 3])})
    write_safetensors(tmp_path / "model-00002.safetensors", {"layer.0.weight": _bf16([3, 3]), "norm.weight": _bf16([3])})
    (tmp_path / "config.json").write_text("{}")

    expected = ExpectedTensors.from_safetensors_dir(tmp_path)

    assert expected == _expected()


def test_non_finite_found_past_first_chunk(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(output_check, "FINITE_CHECK_CHUNK_BYTES", 4)
    embed = _bf16([4, 3])
    embed[3, 2] = BF16_NAN
    failure = _check(tmp_path, {**_good_tensors(), "embed.weight": embed})
    assert failure == "tensor embed.weight has non-finite values"
