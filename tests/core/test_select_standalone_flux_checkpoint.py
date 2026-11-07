import pytest

from core.models.image_models import select_standalone_flux_checkpoint


def test_single_checkpoint_is_returned_unchanged():
    assert select_standalone_flux_checkpoint([("only.safetensors", 12 * 1024**3)]) == "only.safetensors"


def test_prefers_named_bf16_over_fp8_and_nf4():
    selected = select_standalone_flux_checkpoint(
        [
            ("pixelwave_flux1_dev_fp8_03.safetensors", 11_100_000_000),
            ("pixelwave_flux1_dev_bf16_03.safetensors", 22_200_000_000),
            ("pixelwave_flux1_dev_nf4_03.safetensors", 5_700_000_000),
        ]
    )
    assert selected == "pixelwave_flux1_dev_bf16_03.safetensors"


def test_prefers_fp16_when_no_bf16():
    selected = select_standalone_flux_checkpoint(
        [
            ("model_fp8.safetensors", 11 * 1024**3),
            ("model_fp16.safetensors", 22 * 1024**3),
        ]
    )
    assert selected == "model_fp16.safetensors"


def test_unmarked_beats_quantized_when_no_named_full_precision():
    selected = select_standalone_flux_checkpoint(
        [
            ("flux_dev.safetensors", 23 * 1024**3),
            ("flux_dev_fp8.safetensors", 11 * 1024**3),
        ]
    )
    assert selected == "flux_dev.safetensors"


def test_rejects_only_quantized_siblings():
    with pytest.raises(ValueError, match="bf16/fp16"):
        select_standalone_flux_checkpoint(
            [
                ("model_fp8.safetensors", 11 * 1024**3),
                ("model_nf4.safetensors", 6 * 1024**3),
            ]
        )
