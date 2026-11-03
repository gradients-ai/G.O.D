#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import yaml


CACHE_ROOT = Path("/cache")
CONFIG_TEMPLATE_ROOT = Path("/workspace/core/training_templates")
CONFIG_OUTPUT_ROOT = Path("/dataset/configs")
DATASET_ZIP_ROOT = CACHE_ROOT / "datasets"
IMAGE_DATASET_ROOT = Path("/dataset/images")
EXTRACT_ROOT = Path("/dataset/extracted")
CHECKPOINTS_ROOT = Path("/app/checkpoints")
AI_TOOLKIT_ROOT = Path("/app/ai-toolkit")
MUSUBI_ROOT = Path("/app/musubi-tuner")
MUSUBI_PYTHON = os.environ.get("MUSUBI_PYTHON", "/opt/musubi/bin/python")
MUSUBI_ACCELERATE = os.environ.get("MUSUBI_ACCELERATE", "/opt/musubi/bin/accelerate")
HIDREAM_DIT = CACHE_ROOT / "hf_cache" / "checkpoints" / "hidream_o1_image_bf16.safetensors"

TEXT_TO_IMAGE = "text-to-image"
EDIT = "edit"
TEMPLATE_BY_MODEL_AND_MODE = {
    ("flux", TEXT_TO_IMAGE): "base_diffusion_flux.yaml",
    ("z-image", TEXT_TO_IMAGE): "base_diffusion_zimage.yaml",
    ("qwen-image", TEXT_TO_IMAGE): "base_diffusion_qwen_image.yaml",
    ("ideogram4", TEXT_TO_IMAGE): "base_diffusion_ideogram4.yaml",
    ("krea2", TEXT_TO_IMAGE): "base_diffusion_krea2.yaml",
    ("flux2-klein-4b", TEXT_TO_IMAGE): "base_diffusion_flux2_klein_4b.yaml",
    ("flux2-klein-4b", EDIT): "base_diffusion_flux2_klein_4b_edit.yaml",
    ("qwen-image-edit-2511", EDIT): "base_diffusion_qwen_image_edit_2511.yaml",
    ("hidream-o1", TEXT_TO_IMAGE): "base_diffusion_hidream_o1.yaml",
    ("moss-image-base", TEXT_TO_IMAGE): "base_diffusion_moss_image_base.yaml",
    ("moss-image-base", EDIT): "base_diffusion_moss_image_base_edit.yaml",
}
IDEOGRAM4_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-8B-Instruct"
KREA2_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-4B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted({model_type for model_type, _ in TEMPLATE_BY_MODEL_AND_MODE}),
    )
    parser.add_argument("--training-mode", choices=[TEXT_TO_IMAGE, EDIT], default=TEXT_TO_IMAGE)
    parser.add_argument("--expected-repo-name", required=True)
    parser.add_argument("--hours-to-complete", required=True)
    parser.add_argument("--trigger-word")
    return parser.parse_args()


def cached_model_path(model: str) -> Path:
    cache_name = model.replace("/", "--")
    return CACHE_ROOT / "models" / cache_name


def _manifest_path(root: Path, value: str, sample_id: str) -> Path:
    path = (root / value).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Invalid or missing manifest path for sample {sample_id}: {value}")
    return path


def _prepare_edit_dataset(extract_root: Path) -> tuple[Path, list[Path]]:
    from PIL import Image

    manifests = list(extract_root.rglob("manifest.jsonl"))
    if len(manifests) != 1:
        raise ValueError(f"Edit dataset must contain exactly one manifest.jsonl; found {len(manifests)}")

    manifest = manifests[0]
    dataset_root = manifest.parent
    targets_path = IMAGE_DATASET_ROOT / "targets"
    controls_root = IMAGE_DATASET_ROOT / "controls"
    targets_path.mkdir(parents=True)
    controls_root.mkdir(parents=True)

    expected_source_count: int | None = None
    sample_count = 0
    with manifest.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample_id = sample.get("sample_id")
            instruction = sample.get("instruction")
            sources = sample.get("sources")
            target = sample.get("target")
            if not isinstance(sample_id, str) or not sample_id or "/" in sample_id or "\\" in sample_id:
                raise ValueError(f"Invalid sample_id on manifest line {line_number}")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Missing instruction for sample {sample_id}")
            if not isinstance(target, str):
                raise ValueError(f"Missing target for sample {sample_id}")
            if not isinstance(sources, list) or not 1 <= len(sources) <= 3 or not all(
                isinstance(source, str) for source in sources
            ):
                raise ValueError(f"Sample {sample_id} must have one to three source images")
            if expected_source_count is None:
                expected_source_count = len(sources)
            elif len(sources) != expected_source_count:
                raise ValueError("Every sample in an edit dataset must have the same number of sources")

            target_source = _manifest_path(dataset_root, target, sample_id)
            target_destination = targets_path / f"{sample_id}.png"
            with Image.open(target_source) as image:
                image.convert("RGB").save(target_destination)
            target_destination.with_suffix(".txt").write_text(instruction.strip() + "\n", encoding="utf-8")

            for index, source_value in enumerate(sources):
                control_path = controls_root / str(index)
                control_path.mkdir(exist_ok=True)
                with Image.open(_manifest_path(dataset_root, source_value, sample_id)) as image:
                    image.convert("RGB").save(control_path / target_destination.name)
            sample_count += 1

    if sample_count == 0:
        raise ValueError("Edit dataset manifest contains no samples")
    return targets_path, [controls_root / str(index) for index in range(expected_source_count or 0)]


def prepare_dataset(task_id: str, training_mode: str) -> tuple[Path, list[Path]]:
    zip_path = DATASET_ZIP_ROOT / f"{task_id}_tourn.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected cached image dataset zip at {zip_path}")

    if IMAGE_DATASET_ROOT.exists():
        shutil.rmtree(IMAGE_DATASET_ROOT)
    IMAGE_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    extract_root = EXTRACT_ROOT / task_id
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        resolved_extract_root = extract_root.resolve()
        for member in archive.infolist():
            destination = (extract_root / member.filename).resolve()
            if destination != resolved_extract_root and resolved_extract_root not in destination.parents:
                raise ValueError(f"Unsafe path in image dataset zip: {member.filename}")
        archive.extractall(extract_root)

    if training_mode == EDIT:
        return _prepare_edit_dataset(extract_root)

    supported_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".txt"}
    copied = 0
    for source in extract_root.rglob("*"):
        if source.is_file() and source.suffix.lower() in supported_suffixes:
            destination = IMAGE_DATASET_ROOT / source.name
            if destination.exists():
                destination = IMAGE_DATASET_ROOT / f"{source.parent.name}_{source.name}"
            shutil.copy2(source, destination)
            copied += 1

    if copied == 0:
        raise RuntimeError(f"No supported image/caption files found in {zip_path}")
    return IMAGE_DATASET_ROOT, []


def prepare_config(args: argparse.Namespace, dataset_path: Path, control_paths: list[Path]) -> Path:
    try:
        template_name = TEMPLATE_BY_MODEL_AND_MODE[(args.model_type, args.training_mode)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported AI Toolkit training combination: {args.model_type}/{args.training_mode}"
        ) from exc
    template_path = CONFIG_TEMPLATE_ROOT / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Missing ai-toolkit template for {args.model_type}: {template_path}")

    with template_path.open() as file:
        config = yaml.safe_load(file)

    config_body = config.setdefault("config", {})
    config_body["name"] = args.expected_repo_name

    process = config_body.setdefault("process", [{}])[0]
    process["training_folder"] = str(CHECKPOINTS_ROOT / args.task_id)
    process["trigger_word"] = args.trigger_word
    if steps_override := os.environ.get("TRAIN_STEPS_OVERRIDE"):
        process.setdefault("train", {})["steps"] = int(steps_override)

    datasets = process.setdefault("datasets", [{}])
    datasets[0]["folder_path"] = str(dataset_path)
    if control_paths:
        datasets[0]["control_path"] = [str(path) for path in control_paths]
    else:
        datasets[0].pop("control_path", None)

    model_config = process.setdefault("model", {})
    model_path = cached_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected cached base model at {model_path}")
    # Trainer container has no internet; ai-toolkit must load every component from
    # the local cache dir. Its ideogram4 loader uses the local branch when
    # <name_or_path>/<subfolder> exists, so point it at the downloaded snapshot.
    model_config["name_or_path"] = str(model_path)

    if args.model_type == "ideogram4":
        text_encoder_path = IDEOGRAM4_TEXT_ENCODER_CACHE
        if not text_encoder_path.exists():
            raise FileNotFoundError(
                f"Expected cached Ideogram 4 text encoder at {text_encoder_path}"
            )
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["text_encoder_path"] = str(text_encoder_path)

    elif args.model_type == "krea2":
        text_encoder_path = KREA2_TEXT_ENCODER_CACHE
        if not text_encoder_path.exists():
            raise FileNotFoundError(
                f"Expected cached Krea 2 text encoder at {text_encoder_path}"
            )
        vae_path = model_path / "vae"
        if not vae_path.exists():
            raise FileNotFoundError(f"Expected cached Krea 2 VAE at {vae_path}")
        model_kwargs = model_config.setdefault("model_kwargs", {})
        model_kwargs["text_encoder_path"] = str(text_encoder_path)
        # Krea2Model appends the "vae" subfolder when loading the VAE.
        model_kwargs["vae_path"] = str(model_path)

    CONFIG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG_OUTPUT_ROOT / f"{args.task_id}.yaml"
    with output_path.open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return output_path


def prepare_hidream_edit_config(dataset_path: Path, control_paths: list[Path], task_id: str) -> Path:
    if len(control_paths) != 1:
        raise ValueError("HiDream O1 edit training requires exactly one source image per target")
    CONFIG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = Path("/dataset/cache") / task_id
    output_path = CONFIG_OUTPUT_ROOT / f"{task_id}_hidream_edit.toml"
    output_path.write_text(
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
                f'image_directory = "{dataset_path}"',
                f'control_directory = "{control_paths[0]}"',
                f'cache_directory = "{cache_path}"',
                "num_repeats = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def run_hidream_edit(args: argparse.Namespace, dataset_path: Path, control_paths: list[Path]) -> None:
    if not HIDREAM_DIT.is_file():
        raise FileNotFoundError(f"Expected cached HiDream O1 checkpoint at {HIDREAM_DIT}")
    dataset_config = prepare_hidream_edit_config(dataset_path, control_paths, args.task_id)
    output_dir = CHECKPOINTS_ROOT / args.task_id / args.expected_repo_name
    output_dir.mkdir(parents=True, exist_ok=True)
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
            str(HIDREAM_DIT),
            "--model_type",
            "full",
        ],
        cwd=MUSUBI_ROOT,
        check=True,
    )
    train_command = [
            MUSUBI_ACCELERATE,
            "launch",
            "--num_cpu_threads_per_process",
            "1",
            "--mixed_precision",
            "bf16",
            "src/musubi_tuner/hidream_o1_train_network.py",
            "--dit",
            str(HIDREAM_DIT),
            "--dataset_config",
            str(dataset_config),
            "--model_type",
            "full",
            "--task",
            "i2i",
            "--sdpa",
            "--mixed_precision",
            "bf16",
            "--timestep_sampling",
            "uniform",
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
            "4e-5",
            "--gradient_checkpointing",
            "--network_module",
            "networks.lora_hidream_o1",
            "--network_dim",
            "32",
            "--network_alpha",
            "32",
            "--network_args",
            "conv_dim=4",
            "conv_alpha=1",
            "--save_every_n_epochs",
            os.environ.get("SAVE_EVERY_N_EPOCHS", "1"),
            "--seed",
            os.environ.get("SEED", "42"),
            "--output_dir",
            str(output_dir),
            "--output_name",
            args.expected_repo_name,
        ]
    if max_train_steps := os.environ.get("MAX_TRAIN_STEPS"):
        train_command.extend(["--max_train_steps", max_train_steps])
    else:
        train_command.extend(["--max_train_epochs", os.environ.get("MAX_TRAIN_EPOCHS", "16")])
    subprocess.run(
        train_command,
        cwd=MUSUBI_ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    dataset_path, control_paths = prepare_dataset(args.task_id, args.training_mode)
    if (args.model_type, args.training_mode) == ("hidream-o1", EDIT):
        run_hidream_edit(args, dataset_path, control_paths)
        return
    config_path = prepare_config(args, dataset_path, control_paths)
    subprocess.run(["python3", "run.py", str(config_path)], cwd=AI_TOOLKIT_ROOT, check=True)


if __name__ == "__main__":
    main()
