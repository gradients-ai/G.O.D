import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.models.image_models import ImageModelType
from trainer.containers import downloader
from trainer.model_artifacts import get_anonymous_model_dir


def _legacy_image_base_model_resolver(base_path: Path) -> Path:
    files = [
        filename
        for filename in os.listdir(base_path)
        if os.path.isfile(os.path.join(base_path, filename))
    ]
    if len(files) == 1 and files[0].endswith(".safetensors"):
        return base_path / files[0]
    return base_path


def _write_repo_files(root: Path, files: dict[str, bytes]) -> None:
    for relative_path, contents in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)


def _mock_repo_tree(monkeypatch, files: dict[str, bytes]) -> None:
    metadata = [SimpleNamespace(path=path, size=len(contents)) for path, contents in files.items()]
    monkeypatch.setattr(downloader.hf_api, "list_repo_tree", lambda **kwargs: metadata)


def _mock_snapshot_download(monkeypatch, files: dict[str, bytes], calls: list[dict] | None = None) -> None:
    def fake_snapshot_download(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        _write_repo_files(local_dir, files)
        return str(local_dir)

    monkeypatch.setattr(downloader, "snapshot_download", fake_snapshot_download)


@pytest.mark.asyncio
async def test_new_standalone_flux_downloads_only_root_checkpoint(monkeypatch, tmp_path):
    repo_id = "dataautogpt3/FLUX-MonochromeManga"
    checkpoint_name = "FLUX-DEV_MonochromeManga.safetensors"
    repo_files = {
        ".gitattributes": b"lfs",
        "README.md": b"model card",
        checkpoint_name: b"checkpoint weights",
    }
    _mock_repo_tree(monkeypatch, repo_files)

    def fake_hf_hub_download(**kwargs):
        assert kwargs["filename"] == checkpoint_name
        checkpoint_path = Path(kwargs["local_dir"]) / checkpoint_name
        checkpoint_path.write_bytes(repo_files[checkpoint_name])
        return str(checkpoint_path)

    monkeypatch.setattr(downloader, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        downloader,
        "snapshot_download",
        lambda **kwargs: pytest.fail("standalone FLUX checkpoints must not download a full snapshot"),
    )

    result = Path(await downloader.download_base_model(repo_id, str(tmp_path), ImageModelType.FLUX, anonymize=False))

    assert result == tmp_path / "dataautogpt3--FLUX-MonochromeManga"
    assert [path.name for path in result.iterdir() if path.is_file()] == [checkpoint_name]
    assert _legacy_image_base_model_resolver(result) == result / checkpoint_name


@pytest.mark.asyncio
async def test_existing_standalone_flux_snapshot_is_normalized(monkeypatch, tmp_path):
    repo_id = "dataautogpt3/FLUX-MonochromeManga"
    checkpoint_name = "FLUX-DEV_MonochromeManga.safetensors"
    repo_files = {
        ".gitattributes": b"lfs",
        "README.md": b"model card",
        checkpoint_name: b"checkpoint weights",
    }
    cache_path = tmp_path / "dataautogpt3--FLUX-MonochromeManga"
    _write_repo_files(cache_path, repo_files)
    _mock_repo_tree(monkeypatch, repo_files)
    monkeypatch.setattr(
        downloader,
        "hf_hub_download",
        lambda **kwargs: pytest.fail("a complete checkpoint must not be downloaded again"),
    )

    result = Path(await downloader.download_base_model(repo_id, str(tmp_path), "flux", anonymize=False))

    assert result == cache_path
    assert [path.name for path in cache_path.iterdir() if path.is_file()] == [checkpoint_name]
    assert _legacy_image_base_model_resolver(cache_path) == cache_path / checkpoint_name


@pytest.mark.asyncio
async def test_genuine_diffusers_layout_is_not_normalized(monkeypatch, tmp_path):
    repo_id = "example/diffusers-flux"
    repo_files = {
        ".gitattributes": b"lfs",
        "README.md": b"model card",
        "model_index.json": b"{}",
        "extra.safetensors": b"root weights",
        "transformer/config.json": b"{}",
        "transformer/diffusion_pytorch_model.safetensors": b"transformer weights",
    }
    cache_path = tmp_path / "example--diffusers-flux"
    _write_repo_files(cache_path, repo_files)
    _mock_repo_tree(monkeypatch, repo_files)
    monkeypatch.setattr(
        downloader,
        "snapshot_download",
        lambda **kwargs: pytest.fail("a complete Diffusers snapshot must not be downloaded again"),
    )

    result = Path(await downloader.download_base_model(repo_id, str(tmp_path), ImageModelType.FLUX, anonymize=False))

    assert result == cache_path
    assert (cache_path / "README.md").is_file()
    assert (cache_path / "model_index.json").is_file()
    assert (cache_path / "transformer/diffusion_pytorch_model.safetensors").is_file()
    assert _legacy_image_base_model_resolver(cache_path) == cache_path


@pytest.mark.asyncio
async def test_sharded_flux_snapshot_is_not_treated_as_standalone(monkeypatch, tmp_path):
    repo_id = "example/sharded-flux"
    shard_name = "model-00001-of-00001.safetensors"
    index_contents = json.dumps({"weight_map": {"transformer.weight": shard_name}}).encode()
    repo_files = {
        "README.md": b"model card",
        "model.safetensors.index.json": index_contents,
        shard_name: b"sharded weights",
    }
    cache_path = tmp_path / "example--sharded-flux"
    _write_repo_files(cache_path, repo_files)
    _mock_repo_tree(monkeypatch, repo_files)
    monkeypatch.setattr(
        downloader,
        "snapshot_download",
        lambda **kwargs: pytest.fail("a complete sharded snapshot must not be downloaded again"),
    )
    monkeypatch.setattr(
        downloader,
        "hf_hub_download",
        lambda **kwargs: pytest.fail("a sharded checkpoint must not use the standalone download path"),
    )

    await downloader.download_base_model(repo_id, str(tmp_path), ImageModelType.FLUX, anonymize=False)

    assert (cache_path / "README.md").is_file()
    assert (cache_path / "model.safetensors.index.json").is_file()
    assert (cache_path / shard_name).is_file()


@pytest.mark.asyncio
async def test_incomplete_existing_snapshot_is_repaired(monkeypatch, tmp_path):
    repo_id = "example/incomplete-flux"
    shard_name = "transformer-00001-of-00001.safetensors"
    index_contents = json.dumps({"weight_map": {"transformer.weight": shard_name}}).encode()
    repo_files = {
        "README.md": b"model card",
        "model.safetensors.index.json": index_contents,
        shard_name: b"sharded weights",
    }
    cache_path = tmp_path / "example--incomplete-flux"
    _write_repo_files(
        cache_path,
        {
            "README.md": repo_files["README.md"],
            "model.safetensors.index.json": index_contents,
        },
    )
    _mock_repo_tree(monkeypatch, repo_files)
    snapshot_calls = []
    _mock_snapshot_download(monkeypatch, repo_files, snapshot_calls)

    result = Path(await downloader.download_base_model(repo_id, str(tmp_path), ImageModelType.FLUX, anonymize=False))

    assert result == cache_path
    assert (cache_path / shard_name).read_bytes() == repo_files[shard_name]
    assert len(snapshot_calls) == 1
    assert Path(snapshot_calls[0]["local_dir"]).parent == tmp_path
    assert Path(snapshot_calls[0]["local_dir"]) != cache_path


@pytest.mark.asyncio
async def test_failed_download_does_not_leave_final_cache(monkeypatch, tmp_path):
    repo_id = "example/failed-qwen-image"
    repo_files = {
        "model_index.json": b"{}",
        "transformer/config.json": b"{}",
    }
    _mock_repo_tree(monkeypatch, repo_files)

    def failing_snapshot_download(**kwargs):
        _write_repo_files(Path(kwargs["local_dir"]), {"model_index.json": b"{}"})
        raise RuntimeError("simulated download failure")

    monkeypatch.setattr(downloader, "snapshot_download", failing_snapshot_download)

    with pytest.raises(RuntimeError, match="simulated download failure"):
        await downloader.download_base_model(repo_id, str(tmp_path), ImageModelType.QWEN_IMAGE, anonymize=False)

    assert not (tmp_path / "example--failed-qwen-image").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_anonymous_model_directory_naming_is_unchanged(monkeypatch, tmp_path):
    repo_id = "example/private-image-model"
    repo_files = {
        "config.json": json.dumps({"_name_or_path": repo_id, "hidden_size": 1}).encode(),
        "model_index.json": b"{}",
        "transformer/config.json": b"{}",
    }
    monkeypatch.setenv("MODEL_HASH_SALT", "cache-layout-test-salt")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-huggingface-token")
    _mock_repo_tree(monkeypatch, repo_files)
    snapshot_calls = []
    _mock_snapshot_download(monkeypatch, repo_files, snapshot_calls)

    result = Path(await downloader.download_base_model(repo_id, str(tmp_path), ImageModelType.KREA2, anonymize=True))
    cached_result = Path(await downloader.download_base_model(repo_id, str(tmp_path), ImageModelType.KREA2, anonymize=True))

    assert result == tmp_path / get_anonymous_model_dir(repo_id)
    assert cached_result == result
    assert "_name_or_path" not in json.loads((result / "config.json").read_text())
    assert len(snapshot_calls) == 1
    assert snapshot_calls[0]["token"] == "test-huggingface-token"


@pytest.mark.asyncio
async def test_non_flux_image_model_keeps_full_snapshot(monkeypatch, tmp_path):
    repo_id = "example/z-image-checkpoint"
    repo_files = {
        "README.md": b"model card",
        "z-image.safetensors": b"checkpoint weights",
    }
    _mock_repo_tree(monkeypatch, repo_files)
    snapshot_calls = []
    _mock_snapshot_download(monkeypatch, repo_files, snapshot_calls)
    monkeypatch.setattr(
        downloader,
        "hf_hub_download",
        lambda **kwargs: pytest.fail("non-FLUX models must retain full snapshot behavior"),
    )

    result = Path(await downloader.download_base_model(repo_id, str(tmp_path), ImageModelType.Z_IMAGE, anonymize=False))

    assert len(snapshot_calls) == 1
    assert (result / "README.md").is_file()
    assert (result / "z-image.safetensors").is_file()
    assert _legacy_image_base_model_resolver(result) == result


@pytest.mark.asyncio
async def test_image_dataset_download_does_not_log_signed_url(monkeypatch, tmp_path, capsys):
    signed_url = "https://datasets.example/task.zip?X-Amz-Signature=secret"
    local_path = tmp_path / "task.zip"

    async def fake_download_s3_file(url, destination):
        assert url == signed_url
        return destination

    monkeypatch.setattr(downloader, "download_s3_file", fake_download_s3_file)
    monkeypatch.setattr(downloader.train_paths, "get_image_training_zip_save_path", lambda task_id: str(local_path))

    await downloader.download_image_dataset(signed_url, "task-id", str(tmp_path))

    assert signed_url not in capsys.readouterr().out
