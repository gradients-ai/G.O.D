import sys
import types
import warnings
from types import SimpleNamespace

from trainer.generation_config import reset_invalid_generation_config


class _SafeGenerationConfig:
    def validate(self):
        return None


class _WarningGenerationConfig:
    eos_token_id = [1, 2]
    pad_token_id = 0

    def validate(self):
        warnings.warn("do_sample is false but temperature is set", UserWarning, stacklevel=2)


class _ValidGenerationConfig:
    def validate(self):
        return None


class _StrictInvalidGenerationConfig:
    decoder_start_token_id = 8

    def validate(self, strict=False):
        if strict:
            raise ValueError("GenerationConfig is invalid: top_p")
        return None


def test_reset_invalid_generation_config_replaces_warning_config(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.GenerationConfig = _SafeGenerationConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = SimpleNamespace(
        config=SimpleNamespace(bos_token_id=9, eos_token_id=3),
        generation_config=_WarningGenerationConfig(),
    )

    assert reset_invalid_generation_config(model, "test save")
    assert isinstance(model.generation_config, _SafeGenerationConfig)
    assert model.generation_config.bos_token_id == 9
    assert model.generation_config.eos_token_id == [1, 2]
    assert model.generation_config.pad_token_id == 0


def test_reset_invalid_generation_config_leaves_valid_config(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.GenerationConfig = _SafeGenerationConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    original_config = _ValidGenerationConfig()
    model = SimpleNamespace(config=SimpleNamespace(), generation_config=original_config)

    assert not reset_invalid_generation_config(model, "test save")
    assert model.generation_config is original_config


def test_reset_invalid_generation_config_uses_strict_validation(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.GenerationConfig = _SafeGenerationConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = SimpleNamespace(config=SimpleNamespace(), generation_config=_StrictInvalidGenerationConfig())

    assert reset_invalid_generation_config(model, "strict save")
    assert isinstance(model.generation_config, _SafeGenerationConfig)
    assert model.generation_config.decoder_start_token_id == 8
