import json
import sys
import types
from pathlib import Path

import pytest


# wandb ships only in the uploader image, not the dev env; the upload path under test never touches it.
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from trainer.containers import uploader  # noqa: E402


V5_TOKENIZER_CONFIG = {
    "backend": "tokenizers",
    "tokenizer_class": "Qwen2Tokenizer",
    "extra_special_tokens": ["<|im_start|>", "<|im_end|>"],
}


class _FakeHfApi:
    def __init__(self, uploads: list[dict]):
        self._uploads = uploads

    def create_repo(self, **kwargs):
        return None

    def upload_folder(self, **kwargs):
        folder = Path(kwargs["folder_path"])
        self._uploads.append(json.loads((folder / "tokenizer_config.json").read_text()))


def test_upload_sanitizes_v5_tokenizer_before_push(tmp_path, monkeypatch):
    """A winner's v5 tokenizer must reach HF in v4-loadable form: it becomes the next week's base."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "model.safetensors").write_bytes(b"\0" * 2048)
    (output_dir / "tokenizer.json").write_text("{}")
    (output_dir / "tokenizer_config.json").write_text(json.dumps(V5_TOKENIZER_CONFIG))

    uploads: list[dict] = []
    monkeypatch.setattr(uploader, "login", lambda token: None)
    monkeypatch.setattr(uploader, "HfApi", lambda: _FakeHfApi(uploads))
    monkeypatch.setattr(uploader.shutil, "rmtree", lambda path: None)
    for key, value in {
        "HUGGINGFACE_TOKEN": "token",
        "HUGGINGFACE_USERNAME": "user",
        "TASK_ID": "task",
        "EXPECTED_REPO_NAME": "repo",
        "LOCAL_FOLDER": str(output_dir),
        "MODEL": "base/model",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("WANDB_TOKEN", raising=False)

    uploader.main()

    assert len(uploads) == 1
    pushed = uploads[0]
    assert pushed["extra_special_tokens"] == {"<|im_start|>": "<|im_start|>", "<|im_end|>": "<|im_end|>"}


@pytest.mark.parametrize("config", [{"extra_special_tokens": {}}, {"tokenizer_class": "Qwen2Tokenizer"}])
def test_sanitize_leaves_v4_tokenizer_untouched(tmp_path, config):
    path = tmp_path / "tokenizer_config.json"
    path.write_text(json.dumps(config))
    before = path.read_text()

    uploader.sanitize_tokenizer_config(str(tmp_path))

    assert path.read_text() == before
