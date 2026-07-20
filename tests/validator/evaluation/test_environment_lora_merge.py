import sys
from types import SimpleNamespace

import pytest

from validator.evaluation.evaluators import environment
from validator.evaluation.evaluators import intercode


@pytest.mark.parametrize("evaluator", [environment, intercode])
def test_merge_disables_remote_code_and_stale_peft_state(monkeypatch, tmp_path, evaluator):
    class FakeTokenizer:
        chat_template = None

        def __len__(self):
            return 10

        def save_pretrained(self, _output_dir):
            pass

    class FakeBaseModel:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(shape=(10, 4)))

    class FakeMergedModel:
        _hf_peft_config_loaded = True

        def save_pretrained(self, _output_dir, **_kwargs):
            assert self._hf_peft_config_loaded is False

    merged = FakeMergedModel()
    fake_torch = SimpleNamespace(
        float16=object(),
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )

    def load_tokenizer(*_args, **kwargs):
        assert kwargs["trust_remote_code"] is False
        return FakeTokenizer()

    def load_model(*_args, **kwargs):
        assert kwargs["trust_remote_code"] is False
        return FakeBaseModel()

    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
    )
    fake_peft = SimpleNamespace(
        PeftModel=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(
                merge_and_unload=lambda **_kwargs: merged,
            )
        )
    )

    monkeypatch.setattr(evaluator.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(evaluator, "ensure_chat_template", lambda *_args: None)
    monkeypatch.setattr(evaluator, "read_chat_template", lambda *_args: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    output_dir = tmp_path / "merged"

    assert evaluator._merge_base_and_lora("base", "adapter", str(output_dir)) == str(output_dir)


@pytest.mark.parametrize("evaluator", [environment, intercode])
def test_model_snapshot_downloads_exclude_python_modules(monkeypatch, evaluator):
    calls = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append((args, kwargs))
        return kwargs.get("local_dir", "/tmp/model-snapshot")

    monkeypatch.setattr(evaluator, "snapshot_download", fake_snapshot_download)

    assert evaluator._download_model_with_retry("org/model") == "/tmp/model-snapshot"
    assert evaluator._download_lora_with_retry("org/adapter", "/tmp/adapter") == "/tmp/adapter"
    assert all(kwargs["ignore_patterns"] == ["*.py"] for _args, kwargs in calls)


@pytest.mark.parametrize("evaluator", [environment, intercode])
def test_environment_sglang_commands_reject_remote_code(monkeypatch, evaluator):
    monkeypatch.setenv("SGLANG_ENV_EVAL_EXTRA_CLI", "--trust-remote-code")

    with pytest.raises(ValueError, match="trust-remote-code"):
        evaluator._build_sglang_command("/tmp/model", 42)
