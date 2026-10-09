"""The uploader must record the model a submission was trained against, verbatim.

On a continuation round the trainer is handed the miner's own adapter from the previous round
and merges the whole declared lineage before training. The adapter it then uploads is a delta on
THAT merged base, so adapter_config.json has to name it. Flattening the stamp to the foundation
(what _resolve_base_model used to do) severs the chain: every consumer walks the lineage itself,
finds a single hop, and merges this adapter alone onto the foundation -- dropping every earlier
round. Uses importorskip because the uploader pulls wandb, absent on a validator-only box.
"""

import json

import pytest


FOUNDATION = "unsloth/Meta-Llama-3.1-8B-Instruct"


@pytest.fixture
def uploader():
    return pytest.importorskip("trainer.containers.uploader")


def _write_adapter(tmp_path, declared_base: str = "stale/value"):
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": declared_base, "r": 16}))
    return cfg


class TestPatchModelMetadata:
    def test_records_immediate_parent_not_foundation(self, uploader, tmp_path, monkeypatch):
        """A round-3 upload names round 2's adapter, not the foundation it descends from.

        The parent is a resolvable adapter here: a uploader that walks the lineage would
        reach FOUNDATION and stamp that, which is the regression this pins.
        """
        import huggingface_hub

        parent = "gradients-io-tournaments/tournament-round2-5ELGMNNW"
        lookups: list[str] = []

        def fake_hf_hub_download(repo_id, filename, *a, **k):
            lookups.append(repo_id)
            if repo_id != parent:
                raise OSError(f"{repo_id} has no {filename}")
            cfg = tmp_path / "parent_adapter_config.json"
            cfg.write_text(json.dumps({"base_model_name_or_path": FOUNDATION}))
            return str(cfg)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
        monkeypatch.setattr(uploader, "hf_hub_download", fake_hf_hub_download, raising=False)
        cfg = _write_adapter(tmp_path)

        uploader.patch_model_metadata(str(tmp_path), parent)

        assert json.loads(cfg.read_text())["base_model_name_or_path"] == parent
        assert lookups == [], f"the uploader must not resolve the lineage, but looked up {lookups}"

    def test_preserves_other_adapter_config_fields(self, uploader, tmp_path):
        cfg = _write_adapter(tmp_path)

        uploader.patch_model_metadata(str(tmp_path), "org/parent")

        assert json.loads(cfg.read_text())["r"] == 16

    def test_updates_readme_base_model(self, uploader, tmp_path):
        _write_adapter(tmp_path)
        readme = tmp_path / "README.md"
        readme.write_text("---\nbase_model: stale/value\n---\n\nnotes\n")

        uploader.patch_model_metadata(str(tmp_path), "org/parent")

        assert "base_model: org/parent\n" in readme.read_text()

    def test_full_finetune_upload_is_untouched(self, uploader, tmp_path):
        """No adapter_config means a full-weight submission: nothing to stamp, no crash."""
        uploader.patch_model_metadata(str(tmp_path), "org/parent")

        assert not (tmp_path / "adapter_config.json").exists()
