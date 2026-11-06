import io
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from validator.evaluation import edit_masks
from validator.evaluation.evaluators import diffusion_objective
from validator.evaluation.evaluators.diffusion_objective import EVAL_HIDREAM_MAX_DENOISING_STEPS
from validator.evaluation.evaluators.diffusion_objective import EVAL_HIDREAM_MUSUBI_MIN_TIMESTEP
from validator.evaluation.evaluators.diffusion_objective import EVAL_MAX_DENOISING_STEPS
from validator.evaluation.evaluators.diffusion_objective import EVAL_MIN_DENOISING_STEPS
from validator.evaluation.evaluators.diffusion_objective import EVAL_MUSUBI_MAX_TIMESTEP
from validator.evaluation.evaluators.diffusion_objective import EVAL_MUSUBI_MIN_TIMESTEP
from validator.evaluation.evaluators.diffusion_objective import _evaluation_config
from validator.evaluation.evaluators.diffusion_objective import _mean_losses_by_sample
from validator.evaluation.evaluators.diffusion_objective import _write_dataset
from validator.evaluation.evaluators.diffusion_objective import lora_modules_from_file
from validator.evaluation.evaluators.diffusion_objective import lora_network_spec
from validator.evaluation.objective_seeding import CaseSeeder
from validator.evaluation.objective_seeding import case_seed


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _source_and_local_edit(size=(512, 384)) -> tuple[bytes, bytes]:
    source = Image.new("RGB", size, (40, 90, 160))
    target = source.copy()
    target.paste((230, 200, 30), (300, 100, 420, 220))
    return _png(source), _png(target)


def _lora(path: Path, ranks=(16, 16), alpha=None) -> Path:
    weights = {}
    for index, rank in enumerate(ranks):
        weights[f"diffusion_model.blocks.{index}.attn.lora_A.weight"] = torch.randn(rank, 64)
        weights[f"diffusion_model.blocks.{index}.attn.lora_B.weight"] = torch.randn(64, rank)
        if alpha is not None:
            weights[f"diffusion_model.blocks.{index}.attn.alpha"] = torch.tensor(float(alpha))
    save_file(weights, str(path))
    return path


def test_edit_dataset_preserves_target_source_and_blank_conditioning(tmp_path: Path):
    source, target = _source_and_local_edit()
    rows = [("sample", target, "replace the sky", source)]

    targets = _write_dataset(rows, tmp_path / "edit", "edit", blank=True)

    assert (targets / "000000.png").read_bytes() == target
    assert (targets / "000000.txt").read_text() == ""
    assert (tmp_path / "edit" / "controls" / "0" / "000000.png").read_bytes() == source
    mask = Image.open(tmp_path / "edit" / "masks" / "000000.png")
    assert mask.size == (512, 384)


def test_edit_mask_covers_changed_region_only():
    source, target = _source_and_local_edit()

    values = np.asarray(edit_masks.build_edit_mask(source, target), dtype=np.float32) / 255.0

    assert values[160, 360] == 1.0
    assert values[20, 20] == 0.0
    assert 0.05 < values.mean() < 0.2


def test_edit_mask_skips_unchanged_and_global_edits():
    source = Image.new("RGB", (256, 256), (40, 90, 160))
    assert edit_masks.build_edit_mask(_png(source), _png(source)) is None
    assert edit_masks.build_edit_mask(_png(source), _png(Image.new("RGB", (256, 256), (250, 10, 10)))) is None


def test_edit_mask_weights_keep_a_floor_on_unchanged_pixels():
    # ~80% coverage → floor clamps to the legacy max (0.2).
    values = np.zeros((5, 5), dtype=np.uint8)
    values.flat[:20] = 255  # 20/25 = 0.8
    mask = Image.fromarray(values, mode="L")

    weights = edit_masks.mask_weights(mask)

    assert float(weights[values == 0][0]) == pytest.approx(edit_masks.MASK_UNCHANGED_WEIGHT)
    assert float(weights[values == 255][0]) == pytest.approx(1.0)


def test_edit_mask_weights_boost_small_changed_regions():
    # ~5% coverage should drop the unchanged floor so edit mass stays dominant.
    values = np.zeros((20, 20), dtype=np.uint8)
    values[:2, :10] = 255  # 20 / 400 = 0.05
    mask = Image.fromarray(values, mode="L")

    floor = edit_masks.unchanged_weight_for_coverage(0.05)
    weights = edit_masks.mask_weights(mask)
    changed_mass = float(weights[values == 255].sum())
    total_mass = float(weights.sum())

    assert floor == pytest.approx(edit_masks.MASK_UNCHANGED_WEIGHT_MIN)
    assert floor < edit_masks.MASK_UNCHANGED_WEIGHT
    assert weights[values == 0][0] == pytest.approx(floor)
    assert weights[values == 255][0] == pytest.approx(1.0)
    assert changed_mass / total_mass == pytest.approx(edit_masks.MASK_TARGET_CHANGED_MASS, abs=0.03)


def test_region_mean_losses_split_changed_and_unchanged():
    per_pixel = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])

    changed, unchanged, coverage = edit_masks.region_mean_losses(per_pixel, mask)

    assert coverage == pytest.approx(0.5)
    assert changed == pytest.approx(1.5)
    assert unchanged == pytest.approx(3.5)


def test_musubi_mask_alignment_matches_bucket_cover_crop():
    mask = Image.new("L", (300, 200), 0)
    mask.paste(255, (0, 0, 150, 200))

    aligned = edit_masks.cover_and_center_crop(mask, 256, 256)

    assert aligned.size == (256, 256)
    values = np.asarray(aligned)
    assert values[128, 10] == 255
    assert values[128, 245] == 0


def test_case_seed_depends_on_sample_identity_not_visit_order():
    keys = ["000000", "000001", "000002"] * 3
    shuffled = keys.copy()
    random.Random(7).shuffle(shuffled)

    def seeds(order):
        seeder = CaseSeeder(42000)
        return {(key, seeder.next(key)[1]): case_seed(42000, key, seeder.counts[key] - 1) for key in order}

    assert seeds(keys) == seeds(shuffled)
    assert case_seed(42000, "000000", 0) != case_seed(42000, "000000", 1)
    assert case_seed(42000, "000000", 0) != case_seed(42001, "000000", 0)


def test_case_seeder_reproduces_noise_for_the_same_case():
    first = CaseSeeder(1)
    first.next("000000")
    noise_a = torch.randn(4)
    second = CaseSeeder(1)
    second.next("000001")
    second.next("000000")
    noise_b = torch.randn(4)

    assert torch.equal(noise_a, noise_b)


def test_mean_losses_are_independent_of_record_order():
    records = [
        {"path": f"/x/{stem}.png", "loss": loss}
        for stem, loss in [("000000", 0.1), ("000001", 1e16), ("000000", 0.2), ("000001", 1.0), ("000001", -1e16)]
    ]
    records = records[:4] + [{"path": "/x/000000.png", "loss": 0.3}] + records[4:]
    reversed_records = list(reversed(records))

    assert _mean_losses_by_sample(records, 2, 3, "test") == _mean_losses_by_sample(reversed_records, 2, 3, "test")
    with pytest.raises(ValueError, match="incomplete"):
        _mean_losses_by_sample(records[:-1], 2, 3, "test")


def test_lora_spec_reads_rank_and_alpha(tmp_path: Path):
    assert lora_network_spec(_lora(tmp_path / "a.safetensors", alpha=8)) == (16, 8.0)
    assert lora_network_spec(_lora(tmp_path / "b.safetensors")) == (16, 16.0)


def test_lora_spec_allows_mixed_ranks_and_keeps_per_module_alphas(tmp_path: Path):
    path = tmp_path / "mixed.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_attn.lora_down.weight": torch.randn(16, 64),
            "lora_unet_blocks_0_attn.lora_up.weight": torch.randn(64, 16),
            "lora_unet_blocks_0_attn.alpha": torch.tensor(8.0),
            "lora_unet_blocks_1_attn.lora_down.weight": torch.randn(32, 64),
            "lora_unet_blocks_1_attn.lora_up.weight": torch.randn(64, 32),
            "lora_unet_patch_conv.lora_down.weight": torch.randn(4, 3, 3, 3),
            "lora_unet_patch_conv.lora_up.weight": torch.randn(3, 4, 1, 1),
            "lora_unet_patch_conv.alpha": torch.tensor(1.0),
        },
        str(path),
    )

    modules_dim, modules_alpha, linear_names = lora_modules_from_file(path)
    assert modules_dim["lora_unet_blocks_0_attn"] == 16
    assert modules_dim["lora_unet_blocks_1_attn"] == 32
    assert modules_dim["lora_unet_patch_conv"] == 4
    assert modules_alpha["lora_unet_blocks_0_attn"] == 8.0
    assert modules_alpha["lora_unet_blocks_1_attn"] == 32.0
    assert modules_alpha["lora_unet_patch_conv"] == 1.0
    assert linear_names == {"lora_unet_blocks_0_attn", "lora_unet_blocks_1_attn"}
    assert lora_network_spec(path) == (32, 32.0)


def test_lora_spec_rejects_loras_without_linear_weights(tmp_path: Path):
    path = tmp_path / "conv_only.safetensors"
    save_file(
        {
            "lora_unet_patch_conv.lora_down.weight": torch.randn(4, 3, 3, 3),
            "lora_unet_patch_conv.lora_up.weight": torch.randn(3, 4, 1, 1),
        },
        str(path),
    )
    with pytest.raises(ValueError, match="no linear"):
        lora_network_spec(path)


def test_ai_toolkit_config_uses_frozen_training_loss_without_updates(tmp_path: Path):
    lora = _lora(tmp_path / "candidate.safetensors", ranks=(64, 64), alpha=32)
    config = _evaluation_config(
        "moss-image-base",
        "edit",
        tmp_path / "dataset" / "targets",
        lora,
        tmp_path / "records.json",
        steps=8,
        seed=123,
    )

    process = config["config"]["process"][0]
    assert process["type"] == "god_objective_evaluator"
    assert process["network"]["pretrained_lora_path"] == str(lora)
    assert process["network"]["linear"] == 64
    assert process["network"]["linear_alpha"] == 32.0
    assert process["evaluation_mask_dir"] == str(tmp_path / "dataset" / "masks")
    assert process["datasets"][0]["control_path"] == [
        str(tmp_path / "dataset" / "controls" / "0")
    ]
    assert process["datasets"][0]["random_crop"] is False
    assert process["train"]["steps"] == 8
    assert process["train"]["gradient_checkpointing"] is True
    assert process["datasets"][0]["resolution"] == 1024
    assert process["train"]["cache_text_embeddings"] is False
    assert process["train"]["min_denoising_steps"] == 0
    assert process["train"]["max_denoising_steps"] == 250
    assert process["train"]["timestep_type"] == "linear"
    assert EVAL_MIN_DENOISING_STEPS == 0
    assert EVAL_MAX_DENOISING_STEPS == 250
    assert EVAL_MUSUBI_MIN_TIMESTEP == 750
    assert EVAL_MUSUBI_MAX_TIMESTEP == 1000
    assert EVAL_HIDREAM_MAX_DENOISING_STEPS == 50
    assert EVAL_HIDREAM_MUSUBI_MIN_TIMESTEP == 950
    assert process["model"]["model_kwargs"]["edit"] is True


def test_hidream_eval_uses_harder_timestep_band(tmp_path: Path):
    config = _evaluation_config(
        "hidream-o1",
        "text-to-image",
        tmp_path / "dataset",
        _lora(tmp_path / "candidate.safetensors"),
        tmp_path / "records.json",
        steps=8,
        seed=123,
    )

    process = config["config"]["process"][0]
    assert "evaluation_mask_dir" not in process
    assert process["train"]["min_denoising_steps"] == 0
    assert process["train"]["max_denoising_steps"] == 50


def test_hidream_edit_scores_each_model_in_its_own_directory(tmp_path: Path, monkeypatch):
    source, target = _source_and_local_edit()
    loras = [_lora(tmp_path / f"m{index}.safetensors") for index in range(2)]
    prepared = []
    scored = []

    monkeypatch.setattr(
        diffusion_objective, "read_generation_dataset", lambda *_args: [("s", target, "edit", source)]
    )
    monkeypatch.setattr(diffusion_objective, "_hidream_checkpoint", lambda _work: tmp_path / "dit.safetensors")

    def prepare(dataset, checkpoint, work_dir, label):
        prepared.append(label)
        return work_dir / f"{label}.toml", work_dir / f"{label}-cache"

    def run(dataset, dataset_config, cache, lora, checkpoint, model_dir, label, *_rest):
        (model_dir / f"{label}-unused-checkpoints").mkdir()
        scored.append((lora, model_dir.name, label))
        return {"losses": [0.5]}

    monkeypatch.setattr(diffusion_objective, "_prepare_hidream_edit_cache", prepare)
    monkeypatch.setattr(diffusion_objective, "_run_hidream_edit_objective", run)
    config = {
        "dataset": "unused",
        "model_type": "hidream-o1",
        "training_mode": "edit",
        "models": [str(path) for path in loras],
        "sample_limit": 4,
        "repeats": 1,
        "seed": 1,
    }

    result = diffusion_objective.evaluate(config, tmp_path / "results.json")

    assert prepared == ["conditional", "no_text"]
    assert [entry[1] for entry in scored] == ["model-000", "model-000", "model-001", "model-001"]
    assert all(isinstance(result[str(path)], dict) for path in loras)
