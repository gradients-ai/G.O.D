#!/usr/bin/env python3
import argparse
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
CHECKPOINTS_ROOT = Path("/app/checkpoints")
AI_TOOLKIT_ROOT = Path("/app/ai-toolkit")

TEMPLATE_BY_MODEL_TYPE = {
    "flux": "base_diffusion_flux.yaml",
    "z-image": "base_diffusion_zimage.yaml",
    "qwen-image": "base_diffusion_qwen_image.yaml",
    "ideogram4": "base_diffusion_ideogram4.yaml",
    "krea2": "base_diffusion_krea2.yaml",
}
IDEOGRAM4_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-8B-Instruct"
KREA2_TEXT_ENCODER_CACHE = CACHE_ROOT / "hf_cache" / "Qwen--Qwen3-VL-4B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--model-type", required=True, choices=sorted(TEMPLATE_BY_MODEL_TYPE))
    parser.add_argument("--expected-repo-name", required=True)
    parser.add_argument("--hours-to-complete", required=True)
    parser.add_argument("--trigger-word")
    return parser.parse_args()


def cached_model_path(model: str) -> Path:
    cache_name = model.replace("/", "--")
    return CACHE_ROOT / "models" / cache_name


def prepare_dataset(task_id: str) -> Path:
    zip_path = DATASET_ZIP_ROOT / f"{task_id}_tourn.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected cached image dataset zip at {zip_path}")

    if IMAGE_DATASET_ROOT.exists():
        shutil.rmtree(IMAGE_DATASET_ROOT)
    IMAGE_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    extract_root = Path("/dataset/extracted") / task_id
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_root)

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
    return IMAGE_DATASET_ROOT


def prepare_config(args: argparse.Namespace, dataset_path: Path) -> Path:
    template_path = CONFIG_TEMPLATE_ROOT / TEMPLATE_BY_MODEL_TYPE[args.model_type]
    if not template_path.exists():
        raise FileNotFoundError(f"Missing ai-toolkit template for {args.model_type}: {template_path}")

    with template_path.open() as file:
        config = yaml.safe_load(file)

    config_body = config.setdefault("config", {})
    config_body["name"] = args.expected_repo_name

    process = config_body.setdefault("process", [{}])[0]
    process["training_folder"] = str(CHECKPOINTS_ROOT / args.task_id)
    process["trigger_word"] = args.trigger_word

    datasets = process.setdefault("datasets", [{}])
    datasets[0]["folder_path"] = str(dataset_path)

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


def main() -> None:
    args = parse_args()
    dataset_path = prepare_dataset(args.task_id)
    config_path = prepare_config(args, dataset_path)
    subprocess.run(["python3", "run.py", str(config_path)], cwd=AI_TOOLKIT_ROOT, check=True)


if __name__ == "__main__":
    main()
