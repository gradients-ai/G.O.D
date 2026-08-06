from types import SimpleNamespace

import pytest
import torch
import transformers

from core.model_loading import load_causal_language_model
from core.model_loading import load_causal_tokenizer


def _tiny_mistral3_outer_model():
    from transformers import Ministral3Config
    from transformers import Mistral3Config
    from transformers import Mistral3ForConditionalGeneration
    from transformers import PixtralVisionConfig

    text_config = Ministral3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    vision_config = PixtralVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=2,
    )
    config = Mistral3Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=10,
        spatial_merge_size=2,
        tie_word_embeddings=False,
    )
    return Mistral3ForConditionalGeneration(config)


def _tiny_gemma4_outer_model():
    from transformers import Gemma4AudioConfig
    from transformers import Gemma4Config
    from transformers import Gemma4ForConditionalGeneration
    from transformers import Gemma4TextConfig
    from transformers import Gemma4VisionConfig

    text_config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["full_attention"],
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=2,
        num_global_key_value_heads=1,
        global_head_dim=8,
        tie_word_embeddings=False,
    )
    vision_config = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        patch_size=2,
        position_embedding_size=16,
    )
    audio_config = Gemma4AudioConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        subsampling_conv_channels=(4, 4),
        conv_kernel_size=3,
        attention_chunk_size=2,
        attention_context_left=2,
        output_proj_dims=16,
    )
    config = Gemma4Config(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=audio_config,
        boi_token_id=20,
        eoi_token_id=21,
        image_token_id=22,
        video_token_id=23,
        boa_token_id=24,
        eoa_token_index=25,
        audio_token_id=26,
        tie_word_embeddings=False,
    )
    return Gemma4ForConditionalGeneration(config)


@pytest.mark.parametrize(
    ("family", "outer_factory", "expected_class"),
    [
        ("mistral3", _tiny_mistral3_outer_model, "Ministral3ForCausalLM"),
        ("gemma4", _tiny_gemma4_outer_model, "Gemma4ForCausalLM"),
    ],
)
def test_outer_checkpoint_loads_and_resaves_as_language_only(
    tmp_path,
    family,
    outer_factory,
    expected_class,
):
    source_dir = tmp_path / f"{family}-outer"
    normalized_dir = tmp_path / f"{family}-text"
    outer_model = outer_factory()
    outer_model.save_pretrained(source_dir, safe_serialization=True)

    model, loading_info = load_causal_language_model(
        str(source_dir),
        local_files_only=True,
        output_loading_info=True,
    )

    assert type(model).__name__ == expected_class
    assert not loading_info["missing_keys"]
    assert not loading_info["mismatched_keys"]
    assert model._weight_conversions == []
    assert model.config.architectures == [expected_class]
    assert all(
        excluded not in parameter_name
        for parameter_name, _ in model.named_parameters()
        for excluded in ("vision", "audio", "projector")
    )
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, outer_model.model.language_model.state_dict()[key])
    torch.testing.assert_close(model.lm_head.weight, outer_model.lm_head.weight)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    assert outputs.logits.shape == (1, 4, 32)
    assert torch.isfinite(outputs.loss)

    model.save_pretrained(normalized_dir, safe_serialization=True)
    reloaded, reload_info = transformers.AutoModelForCausalLM.from_pretrained(
        normalized_dir,
        local_files_only=True,
        output_loading_info=True,
    )
    assert type(reloaded).__name__ == expected_class
    assert not reload_info["missing_keys"]
    assert not reload_info["unexpected_keys"]
    assert not reload_info["mismatched_keys"]


def test_generic_model_uses_auto_model_unchanged(monkeypatch):
    config = SimpleNamespace(model_type="llama")
    sentinel = object()
    calls = []

    def fake_from_pretrained(source, *args, **kwargs):
        calls.append((source, args, kwargs))
        return sentinel

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)

    result = load_causal_language_model("model/repo", "model-arg", config=config, device_map="cpu")

    assert result is sentinel
    assert calls == [("model/repo", ("model-arg",), {"config": config, "device_map": "cpu"})]


def test_mistral_tokenizer_uses_trainable_backend(monkeypatch):
    config = SimpleNamespace(model_type="mistral3")
    sentinel = object()
    calls = []

    def fake_from_pretrained(source, **kwargs):
        calls.append((source, kwargs))
        return sentinel

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    result = load_causal_tokenizer("model/repo", config=config, local_files_only=True)

    assert result is sentinel
    assert calls == [
        (
            "model/repo",
            {"fix_mistral_regex": True, "local_files_only": True},
        )
    ]


def test_outer_load_rejects_missing_language_weights(monkeypatch):
    text_config = SimpleNamespace(model_type="ministral3")
    config = SimpleNamespace(model_type="mistral3", get_text_config=lambda: text_config)

    monkeypatch.setattr(
        transformers.Ministral3ForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: (object(), {"missing_keys": ["model.layers.0.weight"]}),
    )

    with pytest.raises(RuntimeError, match="Language weights were not loaded completely"):
        load_causal_language_model("model/repo", config=config)
