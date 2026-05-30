"""
Comprehensive stats test on distilgpt2.
Verifies all fields are populated, types correct, and values sensible.
Run with: python -m pytest tests/test_comprehensive_stats.py -v -o addopts= -s
"""

import json

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.models.model_prep_models import (
    AugmentationConfig,
    AugmentationScope,
    AugmentationType,
    BaselineStats,
)
from trainer.model_prep.augmentation import augment_model
from trainer.model_prep.stats import compute_text_stats, classify_layer


MODEL_ID = "distilgpt2"

EVAL_DATA = [
    {"text": "The capital of France is Paris and it is known for the Eiffel Tower."},
    {"text": "Machine learning is a subset of artificial intelligence that focuses on data."},
    {"text": "Python is a popular programming language used for web development."},
    {"text": "The quick brown fox jumps over the lazy dog in the garden."},
    {"text": "Climate change is one of the biggest challenges facing humanity today."},
    {"text": "The stock market experienced significant volatility during the pandemic."},
    {"text": "Quantum computing promises to revolutionize cryptography and drug discovery."},
    {"text": "The Great Wall of China is one of the most impressive structures ever built."},
    {"text": "Neural networks learn hierarchical representations of data through backpropagation."},
    {"text": "The Pythagorean theorem states that a squared plus b squared equals c squared."},
] * 10


@pytest.fixture(scope="module")
def baseline_stats():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return compute_text_stats(model, tokenizer, EVAL_DATA, max_samples=100)


@pytest.fixture(scope="module")
def augmented_stats():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    augment_model(model, AugmentationConfig(
        aug_type=AugmentationType.GAUSSIAN_NOISE,
        scope=AugmentationScope.ALL_LAYERS,
        seed=42,
        intensity=0.05,
    ))
    return compute_text_stats(model, tokenizer, EVAL_DATA, max_samples=100)


class TestDatasetStats:
    def test_total_tokens_positive(self, baseline_stats):
        assert baseline_stats.dataset.total_tokens > 0

    def test_seq_length_distribution(self, baseline_stats):
        d = baseline_stats.dataset.seq_length_distribution
        assert d.mean > 0
        assert d.p50 > 0
        assert d.p95 >= d.p50
        assert d.p99 >= d.p95
        assert d.max >= d.p99

    def test_near_duplicate_rate(self, baseline_stats):
        # We duplicated the data 10x, so dup rate should be high
        assert baseline_stats.dataset.near_duplicate_rate > 0.5

    def test_bits_per_byte_sensible(self, baseline_stats):
        # BPB should be positive and typically 0.5-3.0 for English text
        assert 0.1 < baseline_stats.dataset.bits_per_byte < 5.0

    def test_vocab_size(self, baseline_stats):
        assert baseline_stats.dataset.vocab_size == 50257


class TestWeightStats:
    def test_has_expected_groups(self, baseline_stats):
        groups = set(baseline_stats.weights.by_group.keys())
        assert "attention_qkv" in groups
        assert "ffn_up" in groups
        assert "embedding" in groups

    def test_no_empty_groups(self, baseline_stats):
        for group, stats in baseline_stats.weights.by_group.items():
            assert stats.weight_rms > 0, f"{group} has zero weight_rms"
            assert stats.weight_norm > 0, f"{group} has zero weight_norm"
            assert stats.max_abs > 0, f"{group} has zero max_abs"

    def test_rms_less_than_max(self, baseline_stats):
        for group, stats in baseline_stats.weights.by_group.items():
            assert stats.weight_rms <= stats.max_abs, f"{group}: rms > max_abs"


class TestTrainingDynamics:
    def test_init_loss_sensible(self, baseline_stats):
        # distilgpt2 on random text should be ~3-5
        assert 1.0 < baseline_stats.training.init_loss < 12.0

    def test_grad_norms_populated(self, baseline_stats):
        assert len(baseline_stats.training.grad_norms) > 0
        for name, norm in baseline_stats.training.grad_norms.items():
            assert norm >= 0, f"{name} has negative grad norm"

    def test_gradient_noise_scale_positive(self, baseline_stats):
        assert baseline_stats.training.gradient_noise_scale >= 0

    def test_activation_rms_populated(self, baseline_stats):
        assert len(baseline_stats.training.activation_rms) > 0
        for name, rms in baseline_stats.training.activation_rms.items():
            assert rms >= 0, f"{name} has negative activation RMS"

    def test_grad_stats_have_svd(self, baseline_stats):
        assert len(baseline_stats.training.grad_stats) > 0
        for name, gs in baseline_stats.training.grad_stats.items():
            assert gs.frobenius_norm >= 0
            assert gs.rms >= 0
            assert gs.max_abs >= 0
            assert len(gs.top_singular_values) > 0

    def test_output_entropy_positive(self, baseline_stats):
        assert baseline_stats.training.output_entropy > 0


class TestAugmentedVsBase:
    def test_augmentation_changes_init_loss(self, baseline_stats, augmented_stats):
        assert baseline_stats.training.init_loss != augmented_stats.training.init_loss

    def test_augmentation_changes_weight_stats(self, baseline_stats, augmented_stats):
        # At least some weight groups should differ
        changed = 0
        for group in baseline_stats.weights.by_group:
            if group in augmented_stats.weights.by_group:
                base_rms = baseline_stats.weights.by_group[group].weight_rms
                aug_rms = augmented_stats.weights.by_group[group].weight_rms
                if abs(base_rms - aug_rms) > 1e-6:
                    changed += 1
        assert changed > 0

    def test_dataset_stats_unchanged(self, baseline_stats, augmented_stats):
        # Dataset stats should be identical — augmentation doesn't change the data
        assert baseline_stats.dataset.total_tokens == augmented_stats.dataset.total_tokens
        assert baseline_stats.dataset.vocab_size == augmented_stats.dataset.vocab_size


class TestJsonRoundtrip:
    def test_serialise_deserialise(self, baseline_stats):
        json_str = baseline_stats.model_dump_json()
        restored = BaselineStats.model_validate_json(json_str)
        assert restored.dataset.total_tokens == baseline_stats.dataset.total_tokens
        assert restored.training.init_loss == baseline_stats.training.init_loss
        assert len(restored.training.grad_stats) == len(baseline_stats.training.grad_stats)


class TestLayerClassification:
    """Test classify_layer across different architectures."""

    def test_gpt2_layers(self):
        assert classify_layer("transformer.h.0.attn.c_attn.weight") == "attention_qkv"
        assert classify_layer("transformer.h.0.attn.c_proj.weight") == "attention_output"
        assert classify_layer("transformer.h.0.mlp.c_fc.weight") == "ffn_up"
        assert classify_layer("transformer.h.0.mlp.c_proj.weight") == "ffn_down"
        assert classify_layer("transformer.wte.weight") == "embedding"
        assert classify_layer("transformer.h.0.ln_1.weight") == "layer_norm"
        assert classify_layer("transformer.ln_f.weight") == "layer_norm"

    def test_llama_layers(self):
        assert classify_layer("model.layers.0.self_attn.q_proj.weight") == "attention_qkv"
        assert classify_layer("model.layers.0.self_attn.k_proj.weight") == "attention_qkv"
        assert classify_layer("model.layers.0.self_attn.v_proj.weight") == "attention_qkv"
        assert classify_layer("model.layers.0.self_attn.o_proj.weight") == "attention_output"
        assert classify_layer("model.layers.0.mlp.gate_proj.weight") == "ffn_up"
        assert classify_layer("model.layers.0.mlp.up_proj.weight") == "ffn_up"
        assert classify_layer("model.layers.0.mlp.down_proj.weight") == "ffn_down"
        assert classify_layer("model.embed_tokens.weight") == "embedding"
        assert classify_layer("lm_head.weight") == "unembedding"
        assert classify_layer("model.layers.0.input_layernorm.weight") == "layer_norm"

    def test_falcon_layers(self):
        assert classify_layer("transformer.h.0.self_attention.query_key_value.weight") == "attention_qkv"
        assert classify_layer("transformer.h.0.self_attention.dense.weight") == "attention_output"
        assert classify_layer("transformer.h.0.mlp.dense_h_to_4h.weight") == "ffn_up"
        assert classify_layer("transformer.h.0.mlp.dense_4h_to_h.weight") == "ffn_down"
        assert classify_layer("transformer.word_embeddings.weight") == "embedding"

    def test_unknown_falls_to_other(self):
        assert classify_layer("some.weird.custom_layer.weight") == "other"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
