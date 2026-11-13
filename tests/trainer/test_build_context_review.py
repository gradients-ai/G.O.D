"""Build-time 1 MB scan: files over the review size in a text job's build context are flagged, never rejected."""

from pathlib import Path

from core.constants.quant import QUANT_LARGE_FILE_REVIEW_BYTES
from trainer import runtime


def test_large_build_context_files_are_flagged_for_review(tmp_path: Path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "weights_delta.npy").write_bytes(b"\0" * (QUANT_LARGE_FILE_REVIEW_BYTES + 1))
    (tmp_path / "train.py").write_bytes(b"print(1)\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "pack").write_bytes(b"\0" * (QUANT_LARGE_FILE_REVIEW_BYTES + 1))
    warnings: list[str] = []
    monkeypatch.setattr(runtime.logger, "warning", lambda message, **_: warnings.append(message))

    message = runtime.flag_large_build_context_files(str(tmp_path))

    assert message is not None and message.startswith("[REVIEW]")
    assert f"scripts/weights_delta.npy ({QUANT_LARGE_FILE_REVIEW_BYTES + 1} B)" in message
    assert ".git" not in message
    assert warnings == [message]


def test_small_build_context_is_silent(tmp_path: Path):
    (tmp_path / "train.py").write_bytes(b"print(1)\n")

    assert runtime.flag_large_build_context_files(str(tmp_path)) is None
