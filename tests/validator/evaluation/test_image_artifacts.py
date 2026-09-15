from types import SimpleNamespace

import pytest

from validator.evaluation import image_artifacts


class ModelApi:
    def __init__(self, *files):
        self.files = files

    def model_info(self, repo, files_metadata=True):
        assert files_metadata
        return SimpleNamespace(siblings=self.files)


def artifact(name, size=6 * 1024**3):
    return SimpleNamespace(rfilename=name, size=size)


def capture_materialize(monkeypatch):
    calls = []

    def materialize(api, repo, filename, directory):
        calls.append((repo, filename, directory))
        return "base.safetensors", {"repo": repo}

    monkeypatch.setattr(image_artifacts, "materialize_model", materialize)
    return calls


def test_krea_uses_raw_checkpoint_from_training_repository(tmp_path, monkeypatch):
    calls = capture_materialize(monkeypatch)
    api = ModelApi(
        artifact("raw.safetensors"),
        artifact("transformer/shard.safetensors"),
    )

    image_artifacts.prepare_base(
        api, {"model_id": "krea/Krea-2-Raw", "model_type": "krea2"}, tmp_path
    )

    assert calls == [
        (
            "krea/Krea-2-Raw",
            "raw.safetensors",
            tmp_path / "models" / "diffusion_models",
        )
    ]


def test_ideogram_uses_exact_training_conversion(tmp_path, monkeypatch):
    calls = capture_materialize(monkeypatch)

    image_artifacts.prepare_base(
        ModelApi(),
        {
            "model_id": "gradients-io-tournaments/ideogram-4-fp8",
            "model_type": "ideogram4",
        },
        tmp_path,
    )

    assert calls == [
        (
            "Comfy-Org/Ideogram-4",
            "diffusion_models/ideogram4_fp8_scaled.safetensors",
            tmp_path / "models" / "diffusion_models",
        )
    ]


def test_base_checkpoint_must_be_unambiguous(tmp_path):
    api = ModelApi(artifact("first.safetensors"), artifact("second.safetensors"))

    with pytest.raises(ValueError, match="unambiguous"):
        image_artifacts.prepare_base(
            api, {"model_id": "example/model", "model_type": "qwen-image"}, tmp_path
        )
