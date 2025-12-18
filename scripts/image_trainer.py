#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL, Flux, Z-Image, Qwen-Image)
"""

import argparse
import asyncio
import os
import subprocess
import sys

import toml
import yaml


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def create_config(task_id, model, model_type, expected_repo_name, trigger_word: str | None = None):
    """Create the diffusion config file (TOML for SDXL/Flux, YAML for ai-toolkit models)"""
    config_template_path = train_paths.get_image_training_config_template_path(model_type)
    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    dataset_path = train_paths.get_image_training_images_dir(task_id)
                    for dataset in process['datasets']:
                        dataset['folder_path'] = dataset_path

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        # Update config
        config["pretrained_model_name_or_path"] = model
        config["train_data_dir"] = train_paths.get_image_training_images_dir(task_id)
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        # Save TOML config
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"Created config at {config_path}", flush=True)
    
    return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        training_command = [
            "python3",
            "/app/ai-toolkit/run.py",
            config_path
        ]
    else:
        training_command = [
            "accelerate", "launch",
            "--dynamo_backend", "no",
            "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-scripts/{model_type}_train_network.py",
            "--config_file", config_path
        ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "z-image", "qwen-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)
    
    model_path = train_paths.get_image_base_model_path(args.model)

    # Create config file
    config_path = create_config(
        args.task_id,
        model_path,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word
    )

    print("Preparing dataset...", flush=True)
    training_images_repeat = cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS
    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=training_images_repeat,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    # Run training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
