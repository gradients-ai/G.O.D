"""Decoded-model materialisation and HF byte counting for quant tasks (CPU, file:// urls, no network)."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.constants.docker as docker_cst
from validator.evaluation import quant_artifacts
from validator.evaluation.quant_artifacts import SHA256_KIND_GIT_BLOB
from validator.evaluation.quant_artifacts import SHA256_KIND_LFS
from validator.evaluation.quant_artifacts import DecodedModelCorrupt
from validator.evaluation.quant_artifacts import artifact_bytes_from_hub
from validator.evaluation.quant_artifacts import artifact_listing_from_hub
from validator.evaluation.quant_artifacts import decoded_model_prefix
from validator.evaluation.quant_artifacts import materialise_decoded_model
from validator.evaluation.quant_artifacts import parse_decoded_model_urls
from validator.scoring.models import HubArtifactFile


SHARD = b"decoded-bf16-shard"


@pytest.fixture
def base_snapshot(monkeypatch):
    """Fake snapshot_download: writes base metadata plus a stray weight index, records the call."""
    calls = []

    def fake_snapshot_download(repo_id, local_dir, ignore_patterns, token):
        calls.append({"repo_id": repo_id, "ignore_patterns": ignore_patterns, "token": token})
        Path(local_dir, "config.json").write_text("{}")
        Path(local_dir, "model.safetensors.index.json").write_text("{}")
        return str(local_dir)

    monkeypatch.setattr(quant_artifacts, "snapshot_download", fake_snapshot_download)
    return calls


@pytest.fixture
def store(tmp_path):
    """Writes objects to a fake bucket dir and returns file:// urls for them."""
    bucket = tmp_path / "bucket"
    bucket.mkdir()

    def put(name: str, content: bytes) -> str:
        path = bucket / name
        path.write_bytes(content)
        return path.as_uri()

    return put


def _index(files: list[dict]) -> bytes:
    return json.dumps({"files": files, "n_tensors": 3, "decoder_sha256": "ab", "artifact_bytes": 9, "hotkey": "hk"}).encode()


def _entry(name: str, content: bytes) -> dict:
    return {"name": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _urls(store, files: list[dict], **extra: bytes) -> dict[str, str]:
    urls = {docker_cst.DECODED_MODEL_INDEX_NAME: store("index.json", _index(files))}
    urls.update({name: store(name, content) for name, content in extra.items()})
    return urls


def _materialise(tmp_path, urls):
    dest = tmp_path / "decoded"
    dest.mkdir()
    return materialise_decoded_model("org/base", urls, dest, hf_token="tok")


class TestMaterialiseDecodedModel:
    def test_writes_listed_shards_over_base_metadata(self, tmp_path, store, base_snapshot) -> None:
        urls = _urls(store, [_entry("model.safetensors", SHARD)], **{"model.safetensors": SHARD})

        dest = _materialise(tmp_path, urls)

        assert sorted(p.name for p in dest.iterdir()) == ["config.json", "model.safetensors"]
        assert (dest / "model.safetensors").read_bytes() == SHARD
        assert base_snapshot[0]["repo_id"] == "org/base"
        assert base_snapshot[0]["token"] == "tok"
        assert "*.safetensors" in base_snapshot[0]["ignore_patterns"]

    def test_deletes_stray_weight_index(self, tmp_path, store, base_snapshot) -> None:
        urls = _urls(store, [_entry("model.safetensors", SHARD)], **{"model.safetensors": SHARD})

        dest = _materialise(tmp_path, urls)

        assert not (dest / "model.safetensors.index.json").exists()
        assert not (dest / docker_cst.DECODED_MODEL_INDEX_NAME).exists()

    @pytest.mark.parametrize(
        "entry",
        [
            {"name": "model.safetensors", "bytes": len(SHARD), "sha256": "0" * 64},
            {"name": "model.safetensors", "bytes": len(SHARD) + 1, "sha256": hashlib.sha256(SHARD).hexdigest()},
            {"name": "model.safetensors", "bytes": len(SHARD) - 1, "sha256": hashlib.sha256(SHARD).hexdigest()},
        ],
        ids=["sha256", "short", "long"],
    )
    def test_rejects_shard_not_matching_index(self, tmp_path, store, base_snapshot, entry) -> None:
        urls = _urls(store, [entry], **{"model.safetensors": SHARD})

        with pytest.raises(DecodedModelCorrupt):
            _materialise(tmp_path, urls)

    def test_ignores_url_not_in_index(self, tmp_path, store, base_snapshot) -> None:
        urls = _urls(store, [_entry("model.safetensors", SHARD)], **{"model.safetensors": SHARD, "payload.safetensors": b"x"})

        dest = _materialise(tmp_path, urls)

        assert not (dest / "payload.safetensors").exists()

    @pytest.mark.parametrize("name", ["sub/model.safetensors", "../model.safetensors", "model.bin", "run.py"])
    def test_refuses_unsafe_index_entry(self, tmp_path, store, base_snapshot, name) -> None:
        urls = _urls(store, [_entry(name, SHARD)], **{"model.safetensors": SHARD})
        urls[name] = urls["model.safetensors"]

        with pytest.raises(DecodedModelCorrupt, match="only flat"):
            _materialise(tmp_path, urls)

    def test_listed_shard_without_url_is_corrupt(self, tmp_path, store, base_snapshot) -> None:
        urls = _urls(store, [_entry("model.safetensors", SHARD)])

        with pytest.raises(DecodedModelCorrupt, match="no url"):
            _materialise(tmp_path, urls)

    def test_invalid_index_is_corrupt(self, tmp_path, store, base_snapshot) -> None:
        urls = {docker_cst.DECODED_MODEL_INDEX_NAME: store("index.json", b'{"files": "nope"}')}

        with pytest.raises(DecodedModelCorrupt, match="invalid"):
            _materialise(tmp_path, urls)


def _hf_api(siblings, calls: list | None = None):
    model_info = SimpleNamespace(siblings=siblings)

    def fake_model_info(repo, files_metadata, timeout):
        if calls is not None:
            calls.append({"repo": repo, "timeout": timeout})
        return model_info

    return lambda token=None: SimpleNamespace(model_info=fake_model_info)


class TestArtifactBytesFromHub:
    def test_sums_every_sibling_including_gitattributes(self, monkeypatch) -> None:
        siblings = [
            SimpleNamespace(rfilename=".gitattributes", size=1519),
            SimpleNamespace(rfilename="model.safetensors", size=4_000_000),
            SimpleNamespace(rfilename="README.md", size=12),
        ]
        monkeypatch.setattr(quant_artifacts, "HfApi", _hf_api(siblings))

        assert artifact_bytes_from_hub("org/miner") == 4_001_531

    def test_raises_on_missing_size(self, monkeypatch) -> None:
        siblings = [SimpleNamespace(rfilename="model.safetensors", size=None)]
        monkeypatch.setattr(quant_artifacts, "HfApi", _hf_api(siblings))

        with pytest.raises(ValueError, match="model.safetensors"):
            artifact_bytes_from_hub("org/miner")

    def test_forwards_timeout_defaulting_to_hub_timeout(self, monkeypatch) -> None:
        calls = []
        siblings = [SimpleNamespace(rfilename="model.safetensors", size=1)]
        monkeypatch.setattr(quant_artifacts, "HfApi", _hf_api(siblings, calls))

        artifact_bytes_from_hub("org/miner")
        artifact_bytes_from_hub("org/miner", timeout=5.0)

        assert [call["timeout"] for call in calls] == [quant_artifacts.HUB_TIMEOUT_SECONDS, 5.0]


def _sibling(name, size, blob_id, lfs_sha=None):
    lfs = SimpleNamespace(sha256=lfs_sha, size=size, pointer_size=130) if lfs_sha else None
    return SimpleNamespace(rfilename=name, size=size, blob_id=blob_id, lfs=lfs)


class TestArtifactListingFromHub:
    def _api(self, siblings, calls, sha="f" * 40):
        model_info = SimpleNamespace(siblings=siblings, sha=sha)

        def fake_model_info(repo, files_metadata, timeout):
            calls.append((repo, files_metadata))
            return model_info

        return lambda token=None: SimpleNamespace(model_info=fake_model_info)

    def test_one_call_gives_bytes_hashes_and_revision(self, monkeypatch) -> None:
        calls: list = []
        siblings = [
            _sibling("model.safetensors", 4_000_000, "blob-big", lfs_sha="a" * 64),
            _sibling(".gitattributes", 1519, "b" * 40),
            _sibling("quant_meta.json", 12, "c" * 40),
        ]
        monkeypatch.setattr(quant_artifacts, "HfApi", self._api(siblings, calls))

        listing = artifact_listing_from_hub("org/miner")

        assert calls == [("org/miner", True)]
        assert listing.bytes == 4_001_531  # every sibling, .gitattributes included: the rule is unchanged
        assert listing.revision == "f" * 40
        assert [f.model_dump() for f in listing.files] == [
            {"name": ".gitattributes", "bytes": 1519, "sha256": "b" * 40, "sha256_kind": SHA256_KIND_GIT_BLOB},
            {"name": "model.safetensors", "bytes": 4_000_000, "sha256": "a" * 64, "sha256_kind": SHA256_KIND_LFS},
            {"name": "quant_meta.json", "bytes": 12, "sha256": "c" * 40, "sha256_kind": SHA256_KIND_GIT_BLOB},
        ]
        assert all(isinstance(f, HubArtifactFile) for f in listing.files)

    def test_bytes_agree_with_artifact_bytes_from_hub(self, monkeypatch) -> None:
        siblings = [_sibling("a", 5, "x" * 40), _sibling("b", 7, "y" * 40, lfs_sha="z" * 64)]
        monkeypatch.setattr(quant_artifacts, "HfApi", self._api(siblings, []))

        assert artifact_listing_from_hub("org/miner").bytes == artifact_bytes_from_hub("org/miner") == 12

    def test_missing_size_or_hash_or_files_raise(self, monkeypatch) -> None:
        for siblings, match in (
            ([_sibling("a", None, "x" * 40)], "no size"),
            ([_sibling("a", 1, None)], "no hash"),
            ([], "lists no files"),
        ):
            monkeypatch.setattr(quant_artifacts, "HfApi", self._api(siblings, []))
            with pytest.raises(ValueError, match=match):
                artifact_listing_from_hub("org/miner")


@pytest.mark.parametrize("name", ["a/b.safetensors", "a\\b.safetensors", "x.bin"])
def test_check_listed_name_rejects_nested_or_non_safetensors(name) -> None:
    with pytest.raises(DecodedModelCorrupt, match="only flat"):
        quant_artifacts._check_listed_name(name)


def test_check_listed_name_accepts_flat_shard() -> None:
    quant_artifacts._check_listed_name("model-00001.safetensors")


def test_decoded_model_prefix_is_stable() -> None:
    assert decoded_model_prefix("task-1", "5Hot") == "quant-decoded/task-1/5Hot"


@pytest.mark.parametrize("value", [None, ""])
def test_parse_decoded_model_urls_empty(value) -> None:
    assert parse_decoded_model_urls(value) == {}


def test_parse_decoded_model_urls_round_trips() -> None:
    mapping = {"org/miner": {"index.json": "https://s3/i", "model.safetensors": "https://s3/m"}}
    assert parse_decoded_model_urls(json.dumps(mapping)) == mapping
