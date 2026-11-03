from pathlib import Path

from validator.evaluation.evaluators.diffusion_objective import EVAL_HIDREAM_MAX_DENOISING_STEPS
from validator.evaluation.evaluators.diffusion_objective import EVAL_HIDREAM_MUSUBI_MIN_TIMESTEP
from validator.evaluation.evaluators.diffusion_objective import EVAL_MAX_DENOISING_STEPS
from validator.evaluation.evaluators.diffusion_objective import EVAL_MIN_DENOISING_STEPS
from validator.evaluation.evaluators.diffusion_objective import EVAL_MUSUBI_MAX_TIMESTEP
from validator.evaluation.evaluators.diffusion_objective import EVAL_MUSUBI_MIN_TIMESTEP
from validator.evaluation.evaluators.diffusion_objective import _evaluation_config
from validator.evaluation.evaluators.diffusion_objective import _write_dataset


def test_edit_dataset_preserves_target_source_and_blank_conditioning(tmp_path: Path):
    rows = [("sample", b"target", "replace the sky", b"source")]

    targets = _write_dataset(rows, tmp_path / "edit", "edit", blank=True)

    assert (targets / "000000.png").read_bytes() == b"target"
    assert (targets / "000000.txt").read_text() == ""
    assert (tmp_path / "edit" / "controls" / "0" / "000000.png").read_bytes() == b"source"


def test_ai_toolkit_config_uses_frozen_training_loss_without_updates(tmp_path: Path):
    config = _evaluation_config(
        "moss-image-base",
        "edit",
        tmp_path / "dataset" / "targets",
        tmp_path / "candidate.safetensors",
        tmp_path / "records.json",
        steps=8,
        seed=123,
    )

    process = config["config"]["process"][0]
    assert process["type"] == "god_objective_evaluator"
    assert process["network"]["pretrained_lora_path"] == str(tmp_path / "candidate.safetensors")
    assert process["datasets"][0]["control_path"] == [
        str(tmp_path / "dataset" / "controls" / "0")
    ]
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
        tmp_path / "candidate.safetensors",
        tmp_path / "records.json",
        steps=8,
        seed=123,
    )

    process = config["config"]["process"][0]
    assert process["train"]["min_denoising_steps"] == 0
    assert process["train"]["max_denoising_steps"] == 50
