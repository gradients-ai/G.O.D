"""Caller-selected Hugging Face datasets must never run repository Python in the validator."""

from types import SimpleNamespace

import pytest

import validator.evaluation.dataset_configs as dataset_configs
import validator.tasks.datasets.columns as columns
import validator.tasks.datasets.hub_safety as hub_safety
import validator.tasks.datasets.preparation as preparation
from core.models.dataset_models import FileFormat
from validator.tasks.datasets.hub_safety import ScriptedDatasetError
from validator.tasks.datasets.hub_safety import resolve_code_free_hub_dataset


SHA = "0123456789abcdef0123456789abcdef01234567"


def _stub_hub(monkeypatch, files: list[str]) -> None:
    info = SimpleNamespace(sha=SHA, siblings=[SimpleNamespace(rfilename=f) for f in files])

    class FakeApi:
        def dataset_info(self, repo_id):
            return info

    monkeypatch.setattr(hub_safety, "HfApi", FakeApi)


def _forbid_loaders(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("datasets loader must not be reached for a scripted dataset")

    monkeypatch.setattr(columns, "load_dataset_builder", boom)
    monkeypatch.setattr(columns, "load_dataset", boom)
    monkeypatch.setattr(preparation, "load_dataset", boom)
    monkeypatch.setattr(dataset_configs, "get_dataset_config_names", boom)


def test_code_free_repo_resolves_to_pinned_commit(monkeypatch):
    _stub_hub(monkeypatch, ["README.md", "data/train-00000.parquet", "train.jsonl"])
    assert resolve_code_free_hub_dataset("org/data") == SHA


@pytest.mark.parametrize("files", [["org_data.py"], ["data.parquet", "helpers/loader.py"]])
def test_repo_with_python_files_is_rejected(monkeypatch, files):
    _stub_hub(monkeypatch, files)
    with pytest.raises(ScriptedDatasetError):
        resolve_code_free_hub_dataset("org/data")


@pytest.mark.parametrize("ref", ["/etc", "../x/y", "not a repo id"])
def test_local_paths_and_malformed_ids_are_rejected(monkeypatch, ref):
    _stub_hub(monkeypatch, [])
    with pytest.raises(ScriptedDatasetError):
        resolve_code_free_hub_dataset(ref)


async def test_column_validation_rejects_scripted_dataset_before_loading(monkeypatch):
    _stub_hub(monkeypatch, ["evil.py"])
    _forbid_loaders(monkeypatch)
    with pytest.raises(ScriptedDatasetError):
        await columns.validate_dataset_columns("attacker/evil", FileFormat.HF, ["instruction"])


async def test_column_validation_rejects_scripted_dataset_without_columns(monkeypatch):
    _stub_hub(monkeypatch, ["evil.py"])
    _forbid_loaders(monkeypatch)
    with pytest.raises(ScriptedDatasetError):
        await columns.validate_dataset_columns("attacker/evil", FileFormat.HF, [])


async def test_column_validation_stays_best_effort_on_hub_errors(monkeypatch):
    class DownApi:
        def dataset_info(self, repo_id):
            raise ConnectionError("hub down")

    monkeypatch.setattr(hub_safety, "HfApi", DownApi)
    _forbid_loaders(monkeypatch)
    await columns.validate_dataset_columns("org/data", FileFormat.HF, ["instruction"])


async def test_column_validation_loads_pinned_revision_without_remote_code(monkeypatch):
    _stub_hub(monkeypatch, ["data.parquet"])
    calls = []

    def fake_config_names(name, **kwargs):
        calls.append(("configs", kwargs))
        return ["default"]

    def fake_builder(path, **kwargs):
        calls.append(("builder", kwargs))
        return SimpleNamespace(info=SimpleNamespace(features={"instruction": None, "output": None}))

    monkeypatch.setattr(dataset_configs, "get_dataset_config_names", fake_config_names)
    monkeypatch.setattr(columns, "load_dataset_builder", fake_builder)

    await columns.validate_dataset_columns("org/data", FileFormat.HF, ["instruction", "output"])

    assert [c[0] for c in calls] == ["configs", "builder"]
    for _, kwargs in calls:
        assert kwargs["revision"] == SHA
        assert kwargs["trust_remote_code"] is False


async def test_preparation_rejects_scripted_dataset(monkeypatch):
    _stub_hub(monkeypatch, ["evil.py"])
    _forbid_loaders(monkeypatch)
    with pytest.raises(ScriptedDatasetError):
        await preparation.download_and_load_dataset("attacker/evil", FileFormat.HF)


async def test_preparation_loads_pinned_revision_without_remote_code(monkeypatch):
    from datasets import Dataset

    _stub_hub(monkeypatch, ["data.parquet"])
    monkeypatch.setattr(dataset_configs, "get_dataset_config_names", lambda name, **kwargs: [])
    seen = {}

    def fake_load_dataset(name, config_name, **kwargs):
        seen.update(kwargs)
        return Dataset.from_list([{"instruction": "a"}])

    monkeypatch.setattr(preparation, "load_dataset", fake_load_dataset)
    ds = await preparation.download_and_load_dataset("org/data", FileFormat.HF)
    assert len(ds) == 1
    assert seen == {"revision": SHA, "trust_remote_code": False}
