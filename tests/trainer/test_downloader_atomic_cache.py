import json
import os
import time
from pathlib import Path

import pytest

from trainer import constants as cst
from trainer.containers import cache_cleanup
from trainer.containers import downloader
from trainer.model_artifacts import get_anonymous_model_dir


REPO_ID = "org/model-7b"
INDEX = {"weight_map": {"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"}}
MODEL_FILES = {
    "config.json": json.dumps({"_name_or_path": REPO_ID, "model_type": "llama"}).encode(),
    "tokenizer.json": b"{}",
    "model.safetensors.index.json": json.dumps(INDEX).encode(),
    "model-00001-of-00002.safetensors": b"shard one",
    "model-00002-of-00002.safetensors": b"shard two",
}


class DownloadKilled(BaseException):
    """Stands in for the container being stopped mid-download."""


def _write(root: Path, files: dict[str, bytes]) -> None:
    for relative_path, contents in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)


class FakeSnapshot:
    def __init__(self, files: dict[str, bytes], fail_after: int | None = None):
        self.files = files
        self.fail_after = fail_after
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        for written, (relative_path, contents) in enumerate(self.files.items()):
            if self.fail_after is not None and written == self.fail_after:
                raise DownloadKilled()
            _write(local_dir, {relative_path: contents})
        return str(local_dir)


def _install(monkeypatch, fake: FakeSnapshot) -> FakeSnapshot:
    monkeypatch.setattr(downloader, "snapshot_download", fake)
    return fake


def _partials(root: Path) -> list[Path]:
    return [entry for entry in root.iterdir() if cst.CACHE_DOWNLOAD_TAG in entry.name]


def _assert_complete_model(model_dir: Path) -> None:
    for name in MODEL_FILES:
        assert (model_dir / name).is_file(), name
    assert (model_dir / cst.CACHE_COMPLETE_MARKER).is_file()


@pytest.mark.asyncio
async def test_interrupted_download_leaves_no_entry_and_next_call_downloads(monkeypatch, tmp_path):
    killed = _install(monkeypatch, FakeSnapshot(MODEL_FILES, fail_after=4))
    with pytest.raises(DownloadKilled):
        await downloader.download_axolotl_base_model(REPO_ID, str(tmp_path), anonymize=True)

    model_dir = tmp_path / get_anonymous_model_dir(REPO_ID)
    assert not os.path.lexists(model_dir)
    assert _partials(tmp_path) == []
    assert len(killed.calls) == 1

    retry = _install(monkeypatch, FakeSnapshot(MODEL_FILES))
    path = await downloader.download_axolotl_base_model(REPO_ID, str(tmp_path), anonymize=True)

    assert Path(path) == model_dir
    assert len(retry.calls) == 1
    assert Path(retry.calls[0]["local_dir"]).parent == tmp_path
    assert Path(retry.calls[0]["local_dir"]) != model_dir
    _assert_complete_model(model_dir)
    assert "_name_or_path" not in json.loads((model_dir / "config.json").read_text())
    assert _partials(tmp_path) == []


@pytest.mark.asyncio
async def test_completed_download_is_reused(monkeypatch, tmp_path):
    fake = _install(monkeypatch, FakeSnapshot(MODEL_FILES))
    first = await downloader.download_axolotl_base_model(REPO_ID, str(tmp_path), anonymize=False)
    second = await downloader.download_axolotl_base_model(REPO_ID, str(tmp_path), anonymize=False)

    assert first == second == str(tmp_path / "org--model-7b")
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_leftover_partial_from_a_killed_download_is_never_used(monkeypatch, tmp_path):
    model_dir = tmp_path / "org--model-7b"
    leftover = tmp_path / f".{model_dir.name}{cst.CACHE_DOWNLOAD_TAG}killed"
    _write(leftover, {"config.json": b"{}", cst.CACHE_COMPLETE_MARKER: b""})

    fake = _install(monkeypatch, FakeSnapshot(MODEL_FILES))
    path = await downloader.download_axolotl_base_model(REPO_ID, str(tmp_path), anonymize=False)

    assert path == str(model_dir)
    assert len(fake.calls) == 1
    _assert_complete_model(model_dir)


@pytest.mark.asyncio
async def test_legacy_dir_missing_a_shard_is_downloaded_again(monkeypatch, tmp_path):
    model_dir = tmp_path / get_anonymous_model_dir(REPO_ID)
    partial_snapshot = {name: data for name, data in MODEL_FILES.items() if name != "model-00002-of-00002.safetensors"}
    _write(model_dir, partial_snapshot)
    _write(model_dir, {".cache/huggingface/download/model-00002-of-00002.safetensors.incomplete": b"half"})

    fake = _install(monkeypatch, FakeSnapshot(MODEL_FILES))
    path = await downloader.download_axolotl_base_model(REPO_ID, str(tmp_path), anonymize=True)

    assert path == str(model_dir)
    assert len(fake.calls) == 1
    _assert_complete_model(model_dir)
    assert not (model_dir / ".cache/huggingface/download/model-00002-of-00002.safetensors.incomplete").exists()
    assert [entry.name for entry in tmp_path.iterdir()] == [model_dir.name]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy_files",
    [
        {"config.json": b"{}", "tokenizer.json": b"{}"},  # weights never landed
        {"config.json": b"{}", "model.safetensors": b""},  # empty weight file
        {"adapter_config.json": b"{}", "config.json": b"{}", "adapter_model.safetensors": b"lora"},  # unmerged adapter
        {"model.safetensors": b"weights"},  # no config
    ],
)
async def test_other_incomplete_legacy_dirs_are_downloaded_again(monkeypatch, tmp_path, legacy_files):
    model_dir = tmp_path / "org--model-7b"
    _write(model_dir, legacy_files)

    fake = _install(monkeypatch, FakeSnapshot(MODEL_FILES))
    await downloader.download_axolotl_base_model(REPO_ID, str(tmp_path), anonymize=False)

    assert len(fake.calls) == 1
    _assert_complete_model(model_dir)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy_files",
    [MODEL_FILES, {"config.json": b"{}", "model.safetensors": b"weights"}],
    ids=["sharded", "unsharded"],
)
async def test_complete_legacy_dir_is_reused_and_marked(monkeypatch, tmp_path, legacy_files):
    model_dir = tmp_path / "org--model-7b"
    _write(model_dir, legacy_files)

    fake = _install(monkeypatch, FakeSnapshot(MODEL_FILES))
    path = await downloader.download_axolotl_base_model(REPO_ID, str(tmp_path), anonymize=False)

    assert path == str(model_dir)
    assert fake.calls == []
    assert (model_dir / cst.CACHE_COMPLETE_MARKER).is_file()


@pytest.mark.asyncio
async def test_lora_is_merged_inside_the_partial_dir_before_caching(monkeypatch, tmp_path):
    merged_in: list[Path] = []

    def fake_merge(model_dir: str) -> None:
        merged_in.append(Path(model_dir))

    monkeypatch.setattr(downloader, "_detect_and_merge_lora", fake_merge)
    _install(monkeypatch, FakeSnapshot(MODEL_FILES))
    path = await downloader.download_axolotl_base_model(REPO_ID, str(tmp_path), anonymize=False)

    assert len(merged_in) == 1
    assert merged_in[0] != Path(path)
    assert cst.CACHE_DOWNLOAD_TAG in merged_in[0].name


@pytest.mark.asyncio
async def test_huggingface_snapshot_legacy_incomplete_dir_is_downloaded_again(monkeypatch, tmp_path):
    save_path = tmp_path / "org--model-7b"
    _write(save_path, {"config.json": b"{}", "model.safetensors.index.json": json.dumps(INDEX).encode()})

    fake = _install(monkeypatch, FakeSnapshot(MODEL_FILES))
    path = await downloader.download_huggingface_snapshot(REPO_ID, str(tmp_path))
    again = await downloader.download_huggingface_snapshot(REPO_ID, str(tmp_path))

    assert path == again == str(save_path)
    assert len(fake.calls) == 1
    _assert_complete_model(save_path)


DATASET_FILES = {"data/train-00000.parquet": b"rows", "README.md": b"card"}


@pytest.mark.asyncio
async def test_hf_text_dataset_interrupted_then_downloaded_and_reused(monkeypatch, tmp_path):
    _install(monkeypatch, FakeSnapshot(DATASET_FILES, fail_after=1))
    with pytest.raises(DownloadKilled):
        await downloader.download_text_dataset("task", "org/data", "hf", str(tmp_path))
    assert not (tmp_path / "org--data").exists()

    fake = _install(monkeypatch, FakeSnapshot(DATASET_FILES))
    path, _ = await downloader.download_text_dataset("task", "org/data", "hf", str(tmp_path))
    await downloader.download_text_dataset("task", "org/data", "hf", str(tmp_path))

    assert path == str(tmp_path / "org--data")
    assert len(fake.calls) == 1
    assert fake.calls[0]["repo_type"] == "dataset"
    assert (tmp_path / "org--data/data/train-00000.parquet").read_bytes() == b"rows"


@pytest.mark.asyncio
async def test_hf_text_dataset_legacy_dir_without_marker_is_downloaded_again(monkeypatch, tmp_path):
    _write(tmp_path / "org--data", {"README.md": b"card"})

    fake = _install(monkeypatch, FakeSnapshot(DATASET_FILES))
    await downloader.download_text_dataset("task", "org/data", "hf", str(tmp_path))

    assert len(fake.calls) == 1
    assert (tmp_path / "org--data/data/train-00000.parquet").is_file()


def test_miner_dataset_legacy_dir_is_reused_and_new_download_is_atomic(monkeypatch, tmp_path):
    _write(tmp_path / "org--old", {"README.md": b"card"})
    fake = _install(monkeypatch, FakeSnapshot(DATASET_FILES))

    assert downloader.download_miner_dataset("org/old", str(tmp_path)) == str(tmp_path / "org--old")
    assert fake.calls == []

    killed = _install(monkeypatch, FakeSnapshot(DATASET_FILES, fail_after=1))
    with pytest.raises(DownloadKilled):
        downloader.download_miner_dataset("org/new", str(tmp_path))
    assert not (tmp_path / "org--new").exists()
    assert len(killed.calls) == 1

    fake = _install(monkeypatch, FakeSnapshot(DATASET_FILES))
    assert downloader.download_miner_dataset("org/new", str(tmp_path)) == str(tmp_path / "org--new")
    assert (tmp_path / "org--new" / cst.CACHE_COMPLETE_MARKER).is_file()
    assert _partials(tmp_path) == []


def test_file_copy_into_place_never_leaves_a_partial_destination(monkeypatch, tmp_path):
    source = tmp_path / "source.json"
    source.write_text("[1, 2, 3]")
    dest = tmp_path / "cache" / "task_train_data.json"

    def killed_copy(src, dst):
        Path(dst).write_text("[1,")
        raise DownloadKilled()

    monkeypatch.setattr(downloader.shutil, "copyfile", killed_copy)
    with pytest.raises(DownloadKilled):
        downloader._copy_file_into_place(source, dest)
    assert list(dest.parent.iterdir()) == []

    monkeypatch.undo()
    downloader._copy_file_into_place(source, dest)
    assert dest.read_text() == "[1, 2, 3]"
    assert list(dest.parent.iterdir()) == [dest]


@pytest.mark.asyncio
async def test_adapter_download_goes_through_a_partial_beside_the_cache(monkeypatch, tmp_path):
    seen_dirs: list[Path] = []

    def fake_hf_hub_download(repo_id, filename, local_dir, token):
        seen_dirs.append(Path(local_dir))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"adapter")
        return str(path)

    monkeypatch.setattr(downloader, "hf_hub_download", fake_hf_hub_download)
    path = await downloader.download_adapter("org/adapters", "sub/adapter.safetensors", str(tmp_path))

    assert Path(path).read_bytes() == b"adapter"
    assert seen_dirs[0].parent == tmp_path
    assert cst.CACHE_DOWNLOAD_TAG in seen_dirs[0].name
    assert _partials(tmp_path) == []


def _age(path: Path, hours: float) -> None:
    stamp = time.time() - hours * 3600
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            os.utime(os.path.join(root, name), (stamp, stamp))
    os.utime(path, (stamp, stamp))


def test_cleaner_removes_stale_partials_and_keeps_running_downloads(tmp_path):
    models = tmp_path / "models"
    stale = models / f".abc123{cst.CACHE_DOWNLOAD_TAG}old"
    running = models / f".def456{cst.CACHE_DOWNLOAD_TAG}new"
    stale_backup = models / ".abc123.backup-x"
    stale_legacy_tmp = models / "org--data.tmp.1"
    stale_legacy_merge = models / "abc123.merged_tmp"
    finished = models / "abc123"
    for path in (stale, running, stale_backup, stale_legacy_tmp, stale_legacy_merge, finished):
        _write(path, {"config.json": b"{}"})
    _write(running, {".cache/huggingface/download/model.safetensors.incomplete": b"bytes"})
    stale_file = models / f".task_train_data.json{cst.CACHE_DOWNLOAD_TAG}x"
    stale_file.write_text("[1,")

    hours = cst.CACHE_PARTIAL_DOWNLOAD_STALE_HOURS + 1
    for path in (stale, running, stale_backup, stale_legacy_tmp, stale_legacy_merge, finished, stale_file):
        _age(path, hours)
    # The running download's top-level dir is old but it is still writing its current file.
    os.utime(running / ".cache/huggingface/download/model.safetensors.incomplete")

    cache_cleanup.clean_partial_downloads(roots=(models, tmp_path / "missing"))

    assert sorted(entry.name for entry in models.iterdir()) == sorted([running.name, finished.name])
