"""Fixed-track structural check: computed tensor sets against recorded llmcompressor headers, and the mutants."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import NotASafetensorsRepoError

from core.models.quant_models import QuantFormat
from validator.evaluation import quant_structure
from validator.evaluation.quant_structure import StructuralCheckUnavailable
from validator.evaluation.quant_structure import TensorSet
from validator.evaluation.quant_structure import TensorSpec
from validator.evaluation.quant_structure import check_fixed_submission
from validator.evaluation.quant_structure import expected_fixed_tensor_set
from validator.evaluation.quant_structure import fetch_tensor_set
from validator.evaluation.quant_structure import fixed_submission_failure


FIXTURES = Path(__file__).parent / "fixtures" / "quant_headers"
DOWN_PROJ = "model.layers.0.mlp.down_proj"
W4 = QuantFormat.W4A16_SYM_G128
# Formats with a recorded llmcompressor header.
RECORDED_FORMATS = (W4, QuantFormat.W4A8_FP8_G128, QuantFormat.W3A16_ASYM_G128)


def _tensor_set(name: str) -> TensorSet:
    return TensorSet(tensors=json.loads((FIXTURES / f"{name}.json").read_text()))


def _base_config() -> dict:
    return json.loads((FIXTURES / "base_config.json").read_text())


def _expected(quant_format: QuantFormat = W4) -> TensorSet:
    return expected_fixed_tensor_set(_tensor_set("base_header"), _base_config(), quant_format.spec())


def _mutant(**changes: TensorSpec | None) -> TensorSet:
    """The W4A16 llmcompressor output with tensors replaced (a TensorSpec), added, or removed (None)."""
    tensors = dict(_tensor_set(W4.value).tensors)
    for name, spec in changes.items():
        if spec is None:
            del tensors[name]
        else:
            tensors[name] = spec
    return TensorSet(tensors=tensors)


@pytest.mark.parametrize("quant_format", RECORDED_FORMATS, ids=lambda f: f.value)
def test_expected_set_equals_llmcompressor_output(quant_format) -> None:
    actual = _tensor_set(quant_format.value)

    assert _expected(quant_format).tensors == actual.tensors
    assert check_fixed_submission(actual, _expected(quant_format)) is None


def test_w2a16_asym_g64_expected_shapes() -> None:
    """No recorded llmcompressor header for W2: 2-bit packs 16 codes per int32, g64 doubles the scale columns."""
    expected = _expected(QuantFormat.W2A16_ASYM_G64).tensors

    assert expected[f"{DOWN_PROJ}.weight_packed"] == TensorSpec(dtype="I32", shape=[896, 304])  # 4864 / 16
    assert expected[f"{DOWN_PROJ}.weight_scale"] == TensorSpec(dtype="BF16", shape=[896, 76])  # 4864 / 64
    assert expected[f"{DOWN_PROJ}.weight_zero_point"] == TensorSpec(dtype="I32", shape=[56, 76])  # 896 / 16
    assert expected[f"{DOWN_PROJ}.weight_shape"] == TensorSpec(dtype="I64", shape=[2])
    gate = "model.layers.0.mlp.gate_proj"
    assert expected[f"{gate}.weight_packed"] == TensorSpec(dtype="I32", shape=[4864, 56])  # 896 / 16
    assert expected[f"{gate}.weight_zero_point"] == TensorSpec(dtype="I32", shape=[304, 14])  # 4864 / 16, 896 / 64


def test_w2a16_partial_zero_point_word_is_rounded_up() -> None:
    layout = quant_structure.packed_layout(1000, 384, QuantFormat.W2A16_ASYM_G64.spec())

    assert layout["weight_zero_point"] == TensorSpec(dtype="I32", shape=[63, 6])  # ceil(1000 * 2 / 32)
    assert layout["weight_packed"] == TensorSpec(dtype="I32", shape=[1000, 24])


def test_w3_submission_fails_the_w2_structure_check() -> None:
    reason = check_fixed_submission(_tensor_set(QuantFormat.W3A16_ASYM_G128.value), _expected(QuantFormat.W2A16_ASYM_G64))

    assert reason.startswith("structural:")


class TestMutants:
    def test_extra_tensor(self) -> None:
        extra = {"model.layers.0.mlp.hidden_copy": TensorSpec(dtype="BF16", shape=[896, 4864])}
        reason = check_fixed_submission(_mutant(**extra), _expected())
        assert reason.startswith("structural: extra [model.layers.0.mlp.hidden_copy]")

    def test_group_64_scales_for_group_128(self) -> None:
        reason = check_fixed_submission(
            _mutant(**{f"{DOWN_PROJ}.weight_scale": TensorSpec(dtype="BF16", shape=[896, 76])}), _expected()
        )
        assert f"first mismatch {DOWN_PROJ}.weight_scale expected BF16[896, 38] got BF16[896, 76]" in reason

    def test_fp32_scales(self) -> None:
        reason = check_fixed_submission(
            _mutant(**{f"{DOWN_PROJ}.weight_scale": TensorSpec(dtype="F32", shape=[896, 38])}), _expected()
        )
        assert f"first mismatch {DOWN_PROJ}.weight_scale expected BF16[896, 38] got F32[896, 38]" in reason

    def test_mlp_projection_left_bf16(self) -> None:
        mutant = _mutant(
            **{
                f"{DOWN_PROJ}.weight_packed": None,
                f"{DOWN_PROJ}.weight_scale": None,
                f"{DOWN_PROJ}.weight_shape": None,
                f"{DOWN_PROJ}.weight": TensorSpec(dtype="BF16", shape=[896, 4864]),
            }
        )
        reason = check_fixed_submission(mutant, _expected())
        assert reason.startswith(f"structural: missing [{DOWN_PROJ}.weight_packed, {DOWN_PROJ}.weight_scale")
        assert f"extra [{DOWN_PROJ}.weight]" in reason

    def test_g_idx_present(self) -> None:
        reason = check_fixed_submission(
            _mutant(**{f"{DOWN_PROJ}.weight_g_idx": TensorSpec(dtype="I32", shape=[4864])}), _expected()
        )
        assert reason == f"structural: extra [{DOWN_PROJ}.weight_g_idx]"

    def test_input_scale_present(self) -> None:
        reason = check_fixed_submission(_mutant(**{f"{DOWN_PROJ}.input_scale": TensorSpec(dtype="BF16", shape=[1])}), _expected())
        assert reason == f"structural: extra [{DOWN_PROJ}.input_scale]"

    def test_changed_embedding_shape(self) -> None:
        resized = {"model.embed_tokens.weight": TensorSpec(dtype="BF16", shape=[32000, 896])}
        reason = check_fixed_submission(_mutant(**resized), _expected())
        assert reason == "structural: first mismatch model.embed_tokens.weight expected BF16[151936, 896] got BF16[32000, 896]"

    def test_long_reason_is_capped_and_counts_the_rest(self) -> None:
        reason = check_fixed_submission(TensorSet(tensors={}), _expected())
        assert len(reason) <= quant_structure.STRUCTURAL_FAILURE_MAX_CHARS
        assert "more" in reason

    async def test_bin_only_repo_is_a_scored_fail(self, monkeypatch) -> None:
        class BinOnlyHfApi:
            def __init__(self, token=None):
                pass

            def get_safetensors_metadata(self, repo):
                raise NotASafetensorsRepoError(f"{repo} is not a safetensors repo")

        monkeypatch.setattr(quant_structure, "HfApi", BinOnlyHfApi)

        assert await fixed_submission_failure("org/bin-only", _expected()) == "structural: no safetensors weights"


class TestEmbeddingTying:
    def test_tied_base_expects_no_lm_head(self) -> None:
        assert "lm_head.weight" not in _expected().tensors

    def test_untied_base_expects_a_bf16_lm_head(self) -> None:
        base = _tensor_set("base_header")
        base.tensors["lm_head.weight"] = TensorSpec(dtype="F32", shape=[151936, 896])
        config = {**_base_config(), "tie_word_embeddings": False}

        expected = expected_fixed_tensor_set(base, config, W4.spec())

        assert expected.tensors["lm_head.weight"] == TensorSpec(dtype="BF16", shape=[151936, 896])

    def test_missing_tie_key_follows_the_base_tensors(self) -> None:
        config = {key: value for key, value in _base_config().items() if key != "tie_word_embeddings"}
        tied_base = _tensor_set("base_header")
        untied_base = _tensor_set("base_header")
        untied_base.tensors["lm_head.weight"] = TensorSpec(dtype="F32", shape=[151936, 896])

        assert "lm_head.weight" not in expected_fixed_tensor_set(tied_base, config, W4.spec()).tensors
        assert "lm_head.weight" in expected_fixed_tensor_set(untied_base, config, W4.spec()).tensors

    def test_untied_config_without_a_stored_head_is_our_error(self) -> None:
        with pytest.raises(ValueError, match="tie_word_embeddings"):
            expected_fixed_tensor_set(_tensor_set("base_header"), {**_base_config(), "tie_word_embeddings": False}, W4.spec())


EMBED = TensorSpec(dtype="BF16", shape=[151936, 896])


def _tied_base_with_head(dtype: str = "F32") -> TensorSet:
    """Granite-like: config says tied, safetensors still store lm_head.weight (at an arbitrary dtype)."""
    base = _tensor_set("base_header")
    base.tensors["lm_head.weight"] = TensorSpec(dtype=dtype, shape=[151936, 896])
    return base


def _with_head(spec: TensorSpec | None) -> TensorSet:
    tensors = dict(_tensor_set(W4.value).tensors)
    if spec is not None:
        tensors["lm_head.weight"] = spec
    return TensorSet(tensors=tensors)


class TestTiedBaseStoringLmHead:
    def expected(self) -> TensorSet:
        return expected_fixed_tensor_set(_tied_base_with_head(), _base_config(), W4.spec())

    def test_base_check_does_not_raise_and_head_is_optional_bf16_at_embedding_shape(self) -> None:
        expected = self.expected()

        assert "lm_head.weight" not in expected.tensors
        assert expected.optional == {"lm_head.weight": EMBED}

    def test_submission_without_head_passes(self) -> None:
        assert check_fixed_submission(_with_head(None), self.expected()) is None

    def test_submission_with_bf16_head_at_embedding_shape_passes(self) -> None:
        assert check_fixed_submission(_with_head(EMBED), self.expected()) is None

    def test_f32_head_fails_with_a_readable_reason(self) -> None:
        reason = check_fixed_submission(_with_head(TensorSpec(dtype="F32", shape=[151936, 896])), self.expected())

        assert reason is not None
        assert "first mismatch lm_head.weight expected BF16[151936, 896] got F32[151936, 896]" in reason

    def test_wrong_shape_head_fails_with_a_readable_reason(self) -> None:
        reason = check_fixed_submission(_with_head(TensorSpec(dtype="BF16", shape=[151936, 512])), self.expected())

        assert reason is not None and "first mismatch lm_head.weight" in reason and "[151936, 512]" in reason

    def test_other_problems_are_still_reported_next_to_an_optional_head(self) -> None:
        actual = _with_head(EMBED)
        del actual.tensors[f"{DOWN_PROJ}.weight_scale"]
        actual.tensors["model.extra"] = TensorSpec(dtype="BF16", shape=[1])

        reason = check_fixed_submission(actual, self.expected())

        assert reason is not None and "missing" in reason and "extra [model.extra]" in reason

    def test_missing_tie_key_with_stored_head_stays_untied(self) -> None:
        config = {key: value for key, value in _base_config().items() if key != "tie_word_embeddings"}

        expected = expected_fixed_tensor_set(_tied_base_with_head("BF16"), config, W4.spec())

        assert expected.tensors["lm_head.weight"] == EMBED and not expected.optional


class TestTiedBaseWithoutStoredHead:
    def test_head_is_neither_required_nor_allowed(self) -> None:
        expected = _expected()

        assert expected.optional == {}
        assert check_fixed_submission(_with_head(None), expected) is None
        assert "extra [lm_head.weight]" in (check_fixed_submission(_with_head(EMBED), expected) or "")


class TestUntiedBase:
    def expected(self) -> TensorSet:
        config = {**_base_config(), "tie_word_embeddings": False}
        return expected_fixed_tensor_set(_tied_base_with_head(), config, W4.spec())

    def test_head_is_required_bf16_at_base_shape(self) -> None:
        expected = self.expected()

        assert expected.tensors["lm_head.weight"] == EMBED and expected.optional == {}

    def test_absent_head_is_a_mismatch(self) -> None:
        assert "missing [lm_head.weight]" in (check_fixed_submission(_with_head(None), self.expected()) or "")

    def test_bf16_head_passes(self) -> None:
        assert check_fixed_submission(_with_head(EMBED), self.expected()) is None

    def test_f32_head_fails(self) -> None:
        reason = check_fixed_submission(_with_head(TensorSpec(dtype="F32", shape=[151936, 896])), self.expected())

        assert reason is not None and "first mismatch lm_head.weight" in reason


def test_base_with_indivisible_in_features_is_our_error() -> None:
    base = _tensor_set("base_header")
    base.tensors[f"{DOWN_PROJ}.weight"] = TensorSpec(dtype="BF16", shape=[896, 4800])
    with pytest.raises(ValueError, match="group size"):
        expected_fixed_tensor_set(base, _base_config(), W4.spec())


class TestFetchTensorSet:
    def test_merges_every_shard(self, monkeypatch) -> None:
        def shard(**tensors):
            return SimpleNamespace(
                tensors={name: SimpleNamespace(dtype=dtype, shape=shape) for name, (dtype, shape) in tensors.items()}
            )

        metadata = SimpleNamespace(
            files_metadata={"a.safetensors": shard(x=("BF16", [2])), "b.safetensors": shard(y=("I32", [3, 4]))}
        )

        class FakeHfApi:
            def __init__(self, token=None):
                self.token = token

            def get_safetensors_metadata(self, repo):
                return metadata

        monkeypatch.setattr(quant_structure, "HfApi", FakeHfApi)

        assert fetch_tensor_set("org/repo", token="t").tensors == {
            "x": TensorSpec(dtype="BF16", shape=[2]),
            "y": TensorSpec(dtype="I32", shape=[3, 4]),
        }

    def test_hub_errors_are_unavailable_not_a_fail(self, monkeypatch) -> None:
        class DownHfApi:
            def __init__(self, token=None):
                pass

            def get_safetensors_metadata(self, repo):
                raise ConnectionError("hub down")

        monkeypatch.setattr(quant_structure, "HfApi", DownHfApi)

        with pytest.raises(StructuralCheckUnavailable, match="hub down"):
            fetch_tensor_set("org/repo")
