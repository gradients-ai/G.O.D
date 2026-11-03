"""Held-out image evaluation using each model's native training objective."""

import gc
import json
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

from validator.evaluation.evaluation_logging import configure_eval_logging
from validator.evaluation.image_artifacts import atomic_json, materialize_model
from validator.evaluation.image_test_data import read_generation_dataset


logger = logging.getLogger(__name__)
AI_TOOLKIT_ROOT = Path(os.environ.get("AI_TOOLKIT_ROOT", "/app/ai-toolkit"))
MUSUBI_ROOT = Path(os.environ.get("MUSUBI_ROOT", "/app/musubi-tuner"))
MUSUBI_PYTHON = os.environ.get("MUSUBI_PYTHON", "python")
MUSUBI_ACCELERATE = os.environ.get("MUSUBI_ACCELERATE", "accelerate")

MODEL_PARAMS = {
    "flux2-klein-4b": 4_000_000_000,
    "qwen-image-edit-2511": 20_000_000_000,
    "hidream-o1": 8_000_000_000,
    "moss-image-base": 5_000_000_000,
}
TEMPLATES = {
    ("flux2-klein-4b", "text-to-image"): "base_diffusion_flux2_klein_4b.yaml",
    ("flux2-klein-4b", "edit"): "base_diffusion_flux2_klein_4b_edit.yaml",
    ("qwen-image-edit-2511", "edit"): "base_diffusion_qwen_image_edit_2511.yaml",
    ("hidream-o1", "text-to-image"): "base_diffusion_hidream_o1.yaml",
    ("moss-image-base", "text-to-image"): "base_diffusion_moss_image_base.yaml",
    ("moss-image-base", "edit"): "base_diffusion_moss_image_base_edit.yaml",
}
MODEL_REPOS = {
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-base-4B",
    "qwen-image-edit-2511": "Qwen/Qwen-Image-Edit-2511",
    "hidream-o1": "HiDream-ai/HiDream-O1-Image",
    "moss-image-base": "OpenMOSS-Team/MOSS-Image-Base",
}
TEMPLATE_ROOT = Path(__file__).resolve().parents[3] / "core" / "training_templates"

# AI Toolkit flow-match timesteps are linspace(1000, 1), so index 0 is pure noise.
# Keep scoring in the noisiest quartile (t ≈ 750–1000, σ ≈ 0.75–1.0).
EVAL_MIN_DENOISING_STEPS = 0
EVAL_MAX_DENOISING_STEPS = 250
EVAL_MUSUBI_MIN_TIMESTEP = 750
EVAL_MUSUBI_MAX_TIMESTEP = 1000
# HiDream O1 predicts pixel-space x0 with noise_scale=8; push into the noisiest 5%.
EVAL_HIDREAM_MAX_DENOISING_STEPS = 50
EVAL_HIDREAM_MUSUBI_MIN_TIMESTEP = 950


def _write_dataset(rows, root: Path, training_mode: str, blank: bool) -> Path:
    targets = root / "targets" if training_mode == "edit" else root
    controls = root / "controls" / "0"
    targets.mkdir(parents=True)
    if training_mode == "edit":
        controls.mkdir(parents=True)
    for index, row in enumerate(rows):
        _, target, caption, *remainder = row
        stem = f"{index:06d}"
        (targets / f"{stem}.png").write_bytes(target)
        (targets / f"{stem}.txt").write_text("" if blank else caption, encoding="utf-8")
        if training_mode == "edit":
            (controls / f"{stem}.png").write_bytes(remainder[0])
    return targets


def _evaluation_config(
    model_type: str,
    training_mode: str,
    dataset: Path,
    lora: Path,
    output: Path,
    steps: int,
    seed: int,
) -> dict:
    template = TEMPLATE_ROOT / TEMPLATES[(model_type, training_mode)]
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    process = config["config"]["process"][0]
    process["type"] = "god_objective_evaluator"
    process["training_folder"] = str(output.parent / "unused-checkpoints")
    process["evaluation_output"] = str(output)
    process["evaluation_seed"] = seed
    process["network"]["pretrained_lora_path"] = str(lora)
    process["datasets"] = [
        {
            "folder_path": str(dataset),
            "caption_ext": "txt",
            "caption_dropout_rate": 0.0,
            "cache_latents_to_disk": True,
            "num_workers": 0,
            "cache_latents_num_workers": 1,
            "batch_load_threads": 1,
            "resolution": 1024,
        }
    ]
    if training_mode == "edit":
        process["datasets"][0]["control_path"] = [str(dataset.parent / "controls" / "0")]
    process["train"].update(
        {
            "batch_size": 1,
            "steps": steps,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "cache_text_embeddings": False,
            "do_cfg": False,
            "do_random_cfg": False,
            "prompt_dropout_prob": 0.0,
            "noise_multiplier": 1.0,
            "min_denoising_steps": EVAL_MIN_DENOISING_STEPS,
            "max_denoising_steps": (
                EVAL_HIDREAM_MAX_DENOISING_STEPS
                if model_type == "hidream-o1"
                else EVAL_MAX_DENOISING_STEPS
            ),
            "timestep_type": "linear",
            "skip_first_sample": True,
            "disable_sampling": True,
        }
    )
    process["save"]["save_every"] = steps + 1
    process["model"]["name_or_path"] = MODEL_REPOS[model_type]
    return config


def _run_objective(
    model_type: str,
    training_mode: str,
    dataset: Path,
    lora: Path,
    work_dir: Path,
    label: str,
    sample_count: int,
    repeats: int,
    seed: int,
) -> list[float]:
    output = work_dir / f"{label}_records.json"
    config_path = work_dir / f"{label}.yaml"
    config = _evaluation_config(
        model_type,
        training_mode,
        dataset,
        lora,
        output,
        sample_count * repeats,
        seed,
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    subprocess.run(["python", "run.py", str(config_path)], cwd=AI_TOOLKIT_ROOT, check=True)
    records = json.loads(output.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for record in records:
        grouped[Path(record["path"]).stem].append(float(record["loss"]))
    if len(grouped) != sample_count or any(len(values) != repeats for values in grouped.values()):
        raise ValueError("AI Toolkit returned incomplete objective-loss cases")
    return [sum(grouped[key]) / len(grouped[key]) for key in sorted(grouped)]


def _hidream_checkpoint(work_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    configured = os.environ.get("HIDREAM_DIT")
    if configured:
        checkpoint = Path(configured)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Configured HiDream checkpoint does not exist: {checkpoint}")
        return checkpoint
    return Path(
        hf_hub_download(
            "Comfy-Org/HiDream-O1-Image",
            "checkpoints/hidream_o1_image_bf16.safetensors",
            local_dir=work_dir / "hidream-model",
        )
    )


def _run_hidream_edit_objective(
    dataset: Path,
    lora: Path,
    checkpoint: Path,
    work_dir: Path,
    label: str,
    sample_count: int,
    repeats: int,
    seed: int,
) -> list[float]:
    cache = work_dir / f"{label}-cache"
    dataset_config = work_dir / f"{label}.toml"
    dataset_config.write_text(
        "\n".join(
            [
                "[general]",
                "resolution = [1024, 1024]",
                'caption_extension = ".txt"',
                "batch_size = 1",
                "enable_bucket = true",
                "bucket_no_upscale = false",
                "",
                "[[datasets]]",
                f'image_directory = "{dataset}"',
                f'control_directory = "{dataset.parent / "controls" / "0"}"',
                f'cache_directory = "{cache}"',
                "num_repeats = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    common = ["--dataset_config", str(dataset_config), "--batch_size", "1"]
    subprocess.run(
        [MUSUBI_PYTHON, "src/musubi_tuner/hidream_o1_cache_pixel.py", *common],
        cwd=MUSUBI_ROOT,
        check=True,
    )
    subprocess.run(
        [
            MUSUBI_PYTHON,
            "src/musubi_tuner/hidream_o1_cache_text_encoder_outputs.py",
            *common,
            "--dit",
            str(checkpoint),
            "--model_type",
            "full",
        ],
        cwd=MUSUBI_ROOT,
        check=True,
    )
    logging_dir = work_dir / f"{label}-logs"
    output_dir = work_dir / f"{label}-unused-checkpoints"
    output_dir.mkdir()
    subprocess.run(
        [
            MUSUBI_ACCELERATE,
            "launch",
            "--num_cpu_threads_per_process",
            "1",
            "--mixed_precision",
            "bf16",
            "src/musubi_tuner/hidream_o1_train_network.py",
            "--dit",
            str(checkpoint),
            "--dataset_config",
            str(dataset_config),
            "--model_type",
            "full",
            "--task",
            "i2i",
            "--sdpa",
            "--gradient_checkpointing",
            "--mixed_precision",
            "bf16",
            "--timestep_sampling",
            "uniform",
            "--min_timestep",
            str(EVAL_HIDREAM_MUSUBI_MIN_TIMESTEP),
            "--max_timestep",
            str(EVAL_MUSUBI_MAX_TIMESTEP),
            "--weighting_scheme",
            "none",
            "--noise_scale_start",
            "8.0",
            "--noise_scale_end",
            "8.0",
            "--noise_clip_std",
            "0.0",
            "--optimizer_type",
            "adamw8bit",
            "--learning_rate",
            "1e-12",
            "--network_module",
            "networks.lora_hidream_o1",
            "--network_dim",
            "32",
            "--network_alpha",
            "32",
            "--network_args",
            "conv_dim=4",
            "conv_alpha=1",
            "--network_weights",
            str(lora),
            "--max_train_steps",
            str(sample_count * repeats),
            "--logging_dir",
            str(logging_dir),
            "--log_with",
            "tensorboard",
            "--save_every_n_epochs",
            "999999",
            "--seed",
            str(seed),
            "--output_dir",
            str(output_dir),
            "--output_name",
            "unused",
        ],
        cwd=MUSUBI_ROOT,
        check=True,
    )
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = list(logging_dir.rglob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise ValueError(f"Expected one Musubi TensorBoard event file; found {len(event_files)}")
    events = EventAccumulator(str(event_files[0]))
    events.Reload()
    records = [float(event.value) for event in events.Scalars("loss/current")]
    if len(records) != sample_count * repeats:
        raise ValueError("Musubi returned incomplete objective-loss cases")
    return [
        sum(records[index::sample_count]) / repeats
        for index in range(sample_count)
    ]


def evaluate(config: dict, output_path: Path) -> dict:
    from huggingface_hub import HfApi

    combination = (config["model_type"], config["training_mode"])
    hidream_edit = combination == ("hidream-o1", "edit")
    if combination not in TEMPLATES and not hidream_edit:
        raise ValueError(f"Unsupported objective evaluator combination: {combination}")

    result = {"model_params_count": MODEL_PARAMS[config["model_type"]]}
    with tempfile.TemporaryDirectory(prefix="image-objective-eval-") as temporary:
        work_dir = Path(temporary)
        rows = read_generation_dataset(config["dataset"], work_dir / "test.zip", config["training_mode"])
        rows = rows[: config["sample_limit"]]
        if not rows:
            raise ValueError("Image objective evaluation dataset is empty")
        conditional = _write_dataset(rows, work_dir / "conditional", config["training_mode"], blank=False)
        no_text = _write_dataset(rows, work_dir / "no_text", config["training_mode"], blank=True)
        api = HfApi()
        hidream_checkpoint = _hidream_checkpoint(work_dir) if hidream_edit else None
        for repo in config["models"]:
            try:
                local_candidate = Path(repo)
                if local_candidate.is_file():
                    lora = local_candidate.resolve()
                else:
                    lora_name, _ = materialize_model(api, repo, None, work_dir / "loras")
                    lora = work_dir / "loras" / lora_name
                if hidream_edit:
                    def run(dataset, label):
                        return _run_hidream_edit_objective(
                            dataset,
                            lora,
                            hidream_checkpoint,
                            work_dir,
                            label,
                            len(rows),
                            config["repeats"],
                            config["seed"],
                        )
                else:
                    def run(dataset, label):
                        return _run_objective(
                            *combination,
                            dataset,
                            lora,
                            work_dir,
                            label,
                            len(rows),
                            config["repeats"],
                            config["seed"],
                        )
                result[repo] = {
                    "eval_loss": {
                        "text_guided_losses": run(conditional, "conditional"),
                        "no_text_losses": run(no_text, "no_text"),
                    },
                    "is_finetune": True,
                }
            except Exception as error:
                logger.exception("Objective evaluation failed for %s", repo)
                result[repo] = f"Image objective evaluation failed ({type(error).__name__})"
            finally:
                atomic_json(output_path, result)
                gc.collect()
    return result


def main() -> None:
    configure_eval_logging()
    output = Path(os.environ.get("EVALUATION_RESULTS_PATH", "/aplp/evaluation_results.json"))
    config = {
        "dataset": os.environ.get("DATASET") or os.environ.get("TEST_SPLIT_URL"),
        "model_type": os.environ.get("MODEL_TYPE"),
        "training_mode": os.environ.get("IMAGE_TRAINING_MODE", "text-to-image"),
        "models": [value.strip() for value in os.environ.get("MODELS", "").split(",") if value.strip()],
        "sample_limit": int(os.environ.get("IMAGE_OBJECTIVE_EVAL_SAMPLES", "4")),
        "repeats": int(os.environ.get("IMAGE_OBJECTIVE_EVAL_REPEATS", "10")),
        "seed": int(os.environ.get("IMAGE_OBJECTIVE_EVAL_SEED", "42000")),
    }
    if not config["dataset"] or not config["models"]:
        raise SystemExit("Image objective evaluation requires TEST_SPLIT_URL and MODELS")
    evaluate(config, output)


if __name__ == "__main__":
    main()
