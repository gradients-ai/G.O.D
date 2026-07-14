import sys
from types import SimpleNamespace

from validator.evaluation.evaluators import environment


def test_merge_disables_stale_transformers_peft_state_before_saving(monkeypatch, tmp_path):
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
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeBaseModel()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeTokenizer()),
    )
    fake_peft = SimpleNamespace(
        PeftModel=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(
                merge_and_unload=lambda **_kwargs: merged,
            )
        )
    )

    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(environment, "ensure_chat_template", lambda *_args: None)
    monkeypatch.setattr(environment, "read_chat_template", lambda *_args: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    output_dir = tmp_path / "merged"

    assert environment._merge_base_and_lora("base", "adapter", str(output_dir)) == str(output_dir)
