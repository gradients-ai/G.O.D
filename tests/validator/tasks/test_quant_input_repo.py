"""full_model_rejection: a quant input repo needs a config.json, no adapter, and weights vLLM can load.

The hub is faked: list_repo_files returns the repo's file names and hf_hub_download writes the named index
file to a temp dir.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from validator.tasks import quant_input_repo


SHARDS = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
INDEX = {"metadata": {"total_size": 1}, "weight_map": {f"layer.{i}.weight": shard for i, shard in enumerate(SHARDS)}}


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Fake one repo: hub(files, {index_name: contents}) sets its file list and index contents."""

    def set_repo(files: list[str], indexes: dict[str, object] | None = None) -> None:
        indexes = indexes or {}

        def download(repo_id, filename, token=None):
            path = Path(tmp_path, filename.replace("/", "__"))
            contents = indexes[filename]
            path.write_text(contents if isinstance(contents, str) else json.dumps(contents))
            return str(path)

        monkeypatch.setattr(quant_input_repo, "list_repo_files", MagicMock(return_value=files))
        monkeypatch.setattr(quant_input_repo, "hf_hub_download", MagicMock(side_effect=download))

    return set_repo


async def test_stale_index_with_missing_shards_is_rejected(hub):
    """One complete model.safetensors next to an index of shards that do not exist."""
    hub(["config.json", "model.safetensors", "model.safetensors.index.json"], {"model.safetensors.index.json": INDEX})

    rejection = await quant_input_repo.full_model_rejection("org/repo")

    assert rejection is not None and "4 shard(s) missing" in rejection and SHARDS[0] in rejection


async def test_index_missing_one_shard_is_rejected(hub):
    hub(["config.json", "model.safetensors.index.json", *SHARDS[:3]], {"model.safetensors.index.json": INDEX})

    rejection = await quant_input_repo.full_model_rejection("org/repo")

    assert rejection is not None and SHARDS[3] in rejection and SHARDS[0] not in rejection


async def test_consistent_sharded_repo_is_accepted(hub):
    hub(["config.json", "model.safetensors.index.json", *SHARDS], {"model.safetensors.index.json": INDEX})

    assert await quant_input_repo.full_model_rejection("org/repo") is None


async def test_single_file_repo_is_accepted_without_reading_an_index(hub):
    hub(["config.json", "model.safetensors", "tokenizer.json"])

    assert await quant_input_repo.full_model_rejection("org/repo") is None
    quant_input_repo.hf_hub_download.assert_not_called()


@pytest.mark.parametrize(
    "files",
    [
        ["adapter_config.json", "adapter_model.safetensors"],
        ["config.json", "adapter_config.json", "adapter_model.safetensors"],
    ],
)
async def test_adapter_is_rejected(hub, files):
    hub(files)

    assert "adapter" in await quant_input_repo.full_model_rejection("org/repo")


async def test_repo_without_config_is_rejected(hub):
    hub(["model.safetensors"])

    assert "config.json" in await quant_input_repo.full_model_rejection("org/repo")


@pytest.mark.parametrize("files", [["config.json", "tokenizer.json"], ["config.json", "pytorch_model.bin"]])
async def test_repo_without_safetensors_weights_is_rejected(hub, files):
    hub(files)

    assert "no .safetensors weights" in await quant_input_repo.full_model_rejection("org/repo")


@pytest.mark.parametrize("contents", ["{not json", {"metadata": {}}, {"weight_map": {}}])
async def test_unparseable_or_empty_index_is_rejected(hub, contents):
    hub(["config.json", "model.safetensors", "model.safetensors.index.json"], {"model.safetensors.index.json": contents})

    assert "model.safetensors.index.json" in await quant_input_repo.full_model_rejection("org/repo")


async def test_hub_errors_propagate(monkeypatch):
    monkeypatch.setattr(quant_input_repo, "list_repo_files", MagicMock(side_effect=RepositoryNotFoundError("missing")))

    with pytest.raises(RepositoryNotFoundError):
        await quant_input_repo.full_model_rejection("org/repo")


@pytest.mark.parametrize(
    ("files", "is_adapter"),
    [
        (["adapter_config.json", "adapter_model.safetensors"], True),
        (["config.json", "adapter_config.json", "model.safetensors"], True),
        (["config.json", "model.safetensors"], False),
        (["model.safetensors"], False),
    ],
)
async def test_is_adapter_repo_looks_only_at_the_adapter_config(hub, files, is_adapter):
    hub(files)

    assert await quant_input_repo.is_adapter_repo("org/repo") is is_adapter


async def test_is_adapter_repo_lets_hub_errors_propagate(monkeypatch):
    monkeypatch.setattr(quant_input_repo, "list_repo_files", MagicMock(side_effect=RepositoryNotFoundError("missing")))

    with pytest.raises(RepositoryNotFoundError):
        await quant_input_repo.is_adapter_repo("org/repo")
