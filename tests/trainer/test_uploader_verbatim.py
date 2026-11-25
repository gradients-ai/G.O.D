"""Uploader container main(): UPLOAD_VERBATIM=1 uploads the output root untouched; without it behaviour is unchanged."""

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


# The uploader image ships wandb; the test venv does not, and main() only touches it when WANDB_TOKEN is set.
sys.modules.setdefault("wandb", ModuleType("wandb"))
uploader = importlib.import_module("trainer.containers.uploader")


class FakeHfApi:
    created: list[dict] = []
    uploads: list[dict] = []

    def create_repo(self, **kwargs) -> None:
        FakeHfApi.created.append(kwargs)

    def upload_folder(self, **kwargs) -> None:
        FakeHfApi.uploads.append(kwargs)


class Calls:
    def __init__(self):
        self.patched: list[str] = []
        self.detected: list[str] = []


@pytest.fixture
def calls(monkeypatch, tmp_path: Path) -> Calls:
    rec = Calls()
    real_detect = uploader.detect_subfolder

    def detect(folder: str):
        rec.detected.append(folder)
        return real_detect(folder)

    FakeHfApi.created = []
    FakeHfApi.uploads = []
    monkeypatch.setattr(uploader, "HfApi", FakeHfApi)
    monkeypatch.setattr(uploader, "login", lambda token: None)
    monkeypatch.setattr(uploader, "patch_model_metadata", lambda folder, model: rec.patched.append(folder))
    monkeypatch.setattr(uploader, "detect_subfolder", detect)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-token")
    monkeypatch.setenv("HUGGINGFACE_USERNAME", "trainer-user")
    monkeypatch.setenv("TASK_ID", "task-1")
    monkeypatch.setenv("EXPECTED_REPO_NAME", "repo-1")
    monkeypatch.setenv("MODEL", "org/model")
    monkeypatch.setenv("LOCAL_FOLDER", str(tmp_path / "out"))
    monkeypatch.delenv("WANDB_TOKEN", raising=False)
    monkeypatch.delenv("HF_REPO_SUBFOLDER", raising=False)
    monkeypatch.delenv("UPLOAD_VERBATIM", raising=False)
    return rec


def _output(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    (out / "codes").mkdir(parents=True)
    (out / "codes" / "layer0.safetensors").write_bytes(b"x" * 2048)
    (out / "quant_manifest.json").write_text("{}")
    return out


def test_verbatim_uploads_the_root_without_patching(calls: Calls, monkeypatch, tmp_path: Path):
    out = _output(tmp_path)
    monkeypatch.setenv("UPLOAD_VERBATIM", "1")

    uploader.main()

    assert calls.patched == []
    assert calls.detected == []
    [upload] = FakeHfApi.uploads
    assert upload["folder_path"] == str(out)
    assert upload["repo_id"] == "trainer-user/repo-1"
    assert upload["delete_patterns"] == ["*"]
    assert upload["ignore_patterns"] is None


def test_without_flag_detects_subfolder_and_patches_metadata(calls: Calls, tmp_path: Path):
    out = _output(tmp_path)

    uploader.main()

    assert calls.detected == [str(out)]
    assert calls.patched == [str(out / "codes")]
    [upload] = FakeHfApi.uploads
    assert upload["folder_path"] == str(out / "codes")
    assert upload["ignore_patterns"] == uploader.TRAINING_STATE_PATTERNS


@pytest.mark.parametrize("verbatim", [False, True])
def test_symlinks_and_special_files_are_removed_before_upload(calls: Calls, monkeypatch, tmp_path: Path, verbatim):
    out = _output(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not for upload")
    (out / "codes" / "leak.safetensors").symlink_to(outside / "secret.txt")
    (out / "linked_dir").symlink_to(outside, target_is_directory=True)
    (out / "dangling").symlink_to(tmp_path / "missing")
    os.mkfifo(out / "pipe")
    uploaded: list[set[str]] = []

    def record(**kwargs):
        root = Path(kwargs["folder_path"])
        uploaded.append({str(p.relative_to(root)) for p in root.rglob("*")})

    monkeypatch.setattr(FakeHfApi, "upload_folder", lambda self, **kwargs: record(**kwargs))
    if verbatim:
        monkeypatch.setenv("UPLOAD_VERBATIM", "1")

    uploader.main()

    [files] = uploaded
    assert not any("leak" in f or "linked_dir" in f or "dangling" in f or "pipe" in f for f in files)
    assert (outside / "secret.txt").read_text() == "not for upload"


def test_a_symlinked_output_folder_is_refused(calls: Calls, monkeypatch, tmp_path: Path):
    real = _output(tmp_path)
    link = tmp_path / "linked-out"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("LOCAL_FOLDER", str(link))

    with pytest.raises(ValueError, match="symlink"):
        uploader.main()
    assert FakeHfApi.uploads == []
