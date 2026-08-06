import json
from pathlib import Path
from types import SimpleNamespace

import transformers

from trainer.containers import downloader


class _FakeTokenizer:
    def __init__(self):
        self.chat_template = None

    def save_pretrained(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": self.chat_template})
        )


class _FakeModel:
    def save_pretrained(self, output_dir, safe_serialization):
        assert safe_serialization is True
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "config.json").write_text(json.dumps({"model_type": "gemma4_text"}))
        (output_path / "model.safetensors").write_bytes(b"weights")


def test_normalize_exact_outer_snapshot_replaces_modalities_and_installs_template(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "vision-weight.safetensors").write_bytes(b"vision")

    config = SimpleNamespace(model_type="gemma4")
    tokenizer = _FakeTokenizer()
    calls = []
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        downloader,
        "load_causal_language_model",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _FakeModel(),
    )
    monkeypatch.setattr(
        downloader,
        "load_causal_tokenizer",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(downloader, "_checked_in_template", lambda _path: "fallback-template")

    downloader._normalize_text_only_base_snapshot("google/gemma-4-E2B", str(model_dir))

    assert calls[0][1]["config"] is config
    assert calls[0][1]["local_files_only"] is True
    assert not (model_dir / "vision-weight.safetensors").exists()
    assert json.loads((model_dir / "config.json").read_text())["model_type"] == "gemma4_text"
    assert json.loads((model_dir / "tokenizer_config.json").read_text())["chat_template"] == "fallback-template"


def test_normalized_exact_snapshot_only_repairs_missing_template(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "ministral3"}))
    (model_dir / "model.safetensors").write_bytes(b"weights")

    config = SimpleNamespace(model_type="ministral3")
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(downloader, "load_causal_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(downloader, "_checked_in_template", lambda _path: "fallback-template")
    monkeypatch.setattr(
        downloader,
        "load_causal_language_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model should not be rewritten")),
    )

    downloader._normalize_text_only_base_snapshot(
        "mistralai/Ministral-3-3B-Base-2512",
        str(model_dir),
    )

    assert (model_dir / "model.safetensors").read_bytes() == b"weights"
    assert json.loads((model_dir / "tokenizer_config.json").read_text())["chat_template"] == "fallback-template"


def test_unlisted_base_is_not_normalized(monkeypatch):
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("config should not be loaded")),
    )

    downloader._normalize_text_only_base_snapshot("Qwen/Qwen3.5-2B", "/unused")
