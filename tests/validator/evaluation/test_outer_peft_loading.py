import importlib.util
import sys
from types import ModuleType
from types import SimpleNamespace


if importlib.util.find_spec("axolotl") is None:
    axolotl_module = ModuleType("axolotl")
    axolotl_module.__path__ = []
    utils_module = ModuleType("axolotl.utils")
    utils_module.__path__ = []
    dict_module = ModuleType("axolotl.utils.dict")
    dict_module.DictDefault = dict
    sys.modules["axolotl"] = axolotl_module
    sys.modules["axolotl.utils"] = utils_module
    sys.modules["axolotl.utils.dict"] = dict_module


from validator.evaluation import common


class _FakeBaseModel:
    def __init__(self, embedding_count=32):
        self.embedding_count = embedding_count
        self.resized_to = None

    def get_input_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(shape=(self.embedding_count, 8)))

    def resize_token_embeddings(self, size):
        self.resized_to = size


class _FakeTokenizer:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_explicit_outer_peft_load_uses_expected_language_base_and_resizes(monkeypatch):
    base_config = SimpleNamespace(model_type="gemma4")
    base_model = _FakeBaseModel()
    peft_config = object()
    merged_model = object()
    model_calls = []
    tokenizer_calls = []
    peft_config_calls = []
    peft_model_calls = []

    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    monkeypatch.setattr(common.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: base_config)
    monkeypatch.setattr(
        common,
        "load_causal_language_model",
        lambda *args, **kwargs: model_calls.append((args, kwargs)) or base_model,
    )
    monkeypatch.setattr(
        common,
        "load_causal_tokenizer",
        lambda *args, **kwargs: tokenizer_calls.append((args, kwargs)) or _FakeTokenizer(35),
    )
    monkeypatch.setattr(common, "create_finetuned_cache_dir", lambda: "/cache/finetuned")
    monkeypatch.setattr(
        common.PeftConfig,
        "from_pretrained",
        lambda *args, **kwargs: peft_config_calls.append((args, kwargs)) or peft_config,
    )
    monkeypatch.setattr(
        common.PeftModel,
        "from_pretrained",
        lambda *args, **kwargs: peft_model_calls.append((args, kwargs)) or merged_model,
    )

    result = common._load_expected_outer_peft_model(
        "miner/adapter",
        "google/gemma-4-E2B",
        local_files_only=False,
        trust_remote_code=False,
    )

    assert result is merged_model
    assert model_calls[0][0] == ("google/gemma-4-E2B",)
    assert model_calls[0][1]["config"] is base_config
    assert tokenizer_calls[0][0] == ("miner/adapter",)
    assert base_model.resized_to == 35
    assert peft_config_calls == [
        (
            ("miner/adapter",),
            {"cache_dir": "/cache/finetuned", "local_files_only": False, "token": "test-token"},
        )
    ]
    assert peft_model_calls[0][0] == (base_model, "miner/adapter")
    assert peft_model_calls[0][1]["config"] is peft_config
    assert peft_model_calls[0][1]["is_trainable"] is False


def test_existing_peft_families_keep_legacy_auto_path(monkeypatch):
    base_config = SimpleNamespace(model_type="llama")
    monkeypatch.setattr(common.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: base_config)
    monkeypatch.setattr(
        common,
        "load_causal_language_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("base should not be loaded explicitly")),
    )

    result = common._load_expected_outer_peft_model(
        "miner/adapter",
        "meta-llama/model",
        local_files_only=False,
        trust_remote_code=False,
    )

    assert result is None
