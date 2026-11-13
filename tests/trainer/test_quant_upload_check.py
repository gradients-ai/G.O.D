"""Pre-upload check of a quant output dir against its quant_manifest.json (tmp dirs only, no Docker)."""

import json
import os
import struct
from pathlib import Path

import pytest

from core.constants.quant import QUANT_ARTIFACT_MAX_BYTES_MULTIPLIER
from core.constants.quant import QUANT_MANIFEST_NAME
from core.models.task_models import TaskType
from trainer.quant_upload_check import QuantUploadVerdict
from trainer.quant_upload_check import check_quant_output
from trainer.quant_upload_check import quant_output_max_bytes


TASK_ID = "task-1"
NO_CAP = None


def _write(output_dir: Path, name: str, size: int) -> None:
    path = output_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _write_manifest(output_dir: Path, files: dict[str, int], task_id: str = TASK_ID, track: str = "fixed") -> int:
    payload = {"version": 1, "task_id": task_id, "track": track, "files": [{"name": n, "bytes": b} for n, b in files.items()]}
    path = output_dir / QUANT_MANIFEST_NAME
    path.write_text(json.dumps(payload))
    return path.stat().st_size


def _artifact(output_dir: Path, files: dict[str, int], **manifest_kwargs) -> int:
    for name, size in files.items():
        _write(output_dir, name, size)
    return _write_manifest(output_dir, files, **manifest_kwargs)


def _check(output_dir: Path, task_type: TaskType = TaskType.QUANTFIXEDTASK, max_bytes: int | None = NO_CAP) -> QuantUploadVerdict:
    return check_quant_output(output_dir, task_id=TASK_ID, task_type=task_type, max_bytes=max_bytes)


def _assert_rejected(verdict: QuantUploadVerdict, reason: str) -> None:
    assert not verdict.ok
    assert verdict.failure.startswith("quant output rejected: ")
    assert reason in verdict.failure


def test_missing_manifest_means_the_pipeline_did_not_finish(tmp_path: Path):
    _write(tmp_path, "model.safetensors", 10)

    _assert_rejected(_check(tmp_path), "quant_manifest.json missing: the pipeline did not finish")


def test_invalid_manifest_reports_the_schema_error(tmp_path: Path):
    (tmp_path / QUANT_MANIFEST_NAME).write_text('{"version": 1}')

    _assert_rejected(_check(tmp_path), "quant_manifest.json invalid")


def test_wrong_task_id_fails(tmp_path: Path):
    _artifact(tmp_path, {"model.safetensors": 10}, task_id="other-task")

    _assert_rejected(_check(tmp_path), "task_id 'other-task'")


def test_wrong_track_fails(tmp_path: Path):
    _artifact(tmp_path, {"model.safetensors": 10}, track="freeform")

    _assert_rejected(_check(tmp_path, TaskType.QUANTFIXEDTASK), "track 'freeform'")


def test_listed_file_with_wrong_size_fails(tmp_path: Path):
    _write(tmp_path, "model.safetensors", 9)
    _write_manifest(tmp_path, {"model.safetensors": 10})

    _assert_rejected(_check(tmp_path), "model.safetensors is 9 bytes, manifest says 10")


def test_listed_missing_file_fails(tmp_path: Path):
    _write_manifest(tmp_path, {"model.safetensors": 10})

    _assert_rejected(_check(tmp_path), "model.safetensors is missing")


def test_listed_symlink_fails(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write(tmp_path, "elsewhere.safetensors", 10)
    (output_dir / "model.safetensors").symlink_to(tmp_path / "elsewhere.safetensors")
    _write_manifest(output_dir, {"model.safetensors": 10})

    _assert_rejected(_check(output_dir), "model.safetensors is a symlink")


def test_listed_file_under_symlinked_dir_fails(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write(tmp_path, "elsewhere/model.safetensors", 10)
    (output_dir / "payload").symlink_to(tmp_path / "elsewhere")
    _write_manifest(output_dir, {"payload/model.safetensors": 10})

    _assert_rejected(_check(output_dir), "payload/model.safetensors is a symlink or under one")


def test_unlisted_files_are_deleted_and_the_manifest_survives(tmp_path: Path):
    manifest_bytes = _artifact(tmp_path, {"model.safetensors": 10, "config.json": 3})
    _write(tmp_path, "optimizer.pt", 50)
    _write(tmp_path, "checkpoint-100/model.safetensors", 10)

    verdict = _check(tmp_path)

    assert verdict == QuantUploadVerdict(ok=True, total_bytes=13 + manifest_bytes, n_files=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json", "model.safetensors", QUANT_MANIFEST_NAME]


def test_symlinked_output_dir_is_rejected(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    _artifact(real, {"model.safetensors": 10})
    link = tmp_path / "link"
    link.symlink_to(real)

    _assert_rejected(_check(link), "output dir is a symlink")


@pytest.mark.skipif(os.geteuid() == 0, reason="root removes the file regardless of the dir's mode")
def test_unlisted_file_the_trainer_cannot_remove_is_a_clean_rejection(tmp_path: Path):
    """Without the reclaim, a dir the miner's root container wrote is read-only to a non-root trainer."""
    _artifact(tmp_path, {"model.safetensors": 10})
    _write(tmp_path, "logs/train.log", 5)
    (tmp_path / "logs").chmod(0o555)
    try:
        verdict = _check(tmp_path)
    finally:
        (tmp_path / "logs").chmod(0o755)

    _assert_rejected(verdict, "output not accessible to the trainer")
    assert "Permission denied" in verdict.failure


def test_total_above_max_bytes_fails(tmp_path: Path):
    manifest_bytes = _artifact(tmp_path, {"model.safetensors": 10})

    _assert_rejected(_check(tmp_path, max_bytes=10 + manifest_bytes - 1), f"output is {10 + manifest_bytes} bytes")
    assert _check(tmp_path, max_bytes=10 + manifest_bytes).ok


def test_listed_python_file_fails_the_artifact_file_check(tmp_path: Path):
    _artifact(tmp_path, {"model.safetensors": 10, "loader.py": 4})

    _assert_rejected(_check(tmp_path), "loader.py has forbidden suffix .py")


def test_listed_executable_fails_the_artifact_file_check(tmp_path: Path):
    _artifact(tmp_path, {"payload.bin": 4})
    os.chmod(tmp_path / "payload.bin", 0o755)

    _assert_rejected(_check(tmp_path), "payload.bin is executable")


def test_freeform_artifact_with_nested_directory_passes(tmp_path: Path):
    files = {"payload/layer0.bin": 7, "payload/deep/layer1.bin": 5, "aux.bin": 2}
    _artifact(tmp_path, files, track="freeform")
    _write(tmp_path, "payload/scratch.tmp", 99)
    _write(tmp_path, "logs/train.log", 99)

    verdict = _check(tmp_path, TaskType.QUANTFREEFORMTASK)

    assert verdict.ok
    assert verdict.n_files == 3
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()) == sorted(
        [*files, QUANT_MANIFEST_NAME]
    )
    assert (tmp_path / "payload" / "deep").is_dir()
    assert not (tmp_path / "logs").exists()


def _write_base_model(model_dir: Path, shapes: dict[str, list[int]]) -> None:
    header: dict = {}
    offset = 0
    for name, shape in shapes.items():
        size = 2 * shape[0] * shape[1]
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header_bytes = json.dumps(header).encode()
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0" * offset)


def test_max_bytes_is_the_multiplier_times_the_base_model_bf16_bytes(tmp_path: Path):
    _write_base_model(tmp_path / "base", {"embed.weight": [4, 3], "layer.0.weight": [3, 3]})

    assert quant_output_max_bytes(tmp_path / "base") == int((12 + 9) * 2 * QUANT_ARTIFACT_MAX_BYTES_MULTIPLIER)


@pytest.mark.parametrize("contents", [None, b"", b"not safetensors"])
def test_unreadable_base_model_skips_the_cap(tmp_path: Path, contents: bytes | None):
    base = tmp_path / "base"
    base.mkdir()
    if contents is not None:
        (base / "model.safetensors").write_bytes(contents)

    assert quant_output_max_bytes(base) is None
