#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
import pathlib
import shutil
import subprocess
import sys

import yaml
from transformers import AutoTokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.dataset_utils import adapt_columns_for_environment_dataset
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
from core.config.config_handler import create_reward_funcs_file


def patch_wandb_symlinks(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def copy_dataset_to_axolotl_directories(dataset_path):
    dataset_filename = os.path.basename(dataset_path)
    data_path, root_path = train_paths.get_axolotl_dataset_paths(dataset_filename)
    shutil.copy(dataset_path, data_path)
    shutil.copy(dataset_path, root_path)

    return data_path


def create_config(task_id, model, dataset, dataset_type, file_format, output_dir, expected_repo_name=None, log_wandb=True):
    """Create the axolotl config file with appropriate settings."""

    print(f"Dataset type: {dataset_type}", flush=True)
    config_path = train_paths.get_axolotl_base_config_path(dataset_type)
    print(f"Config path: {config_path}", flush=True)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = [create_dataset_entry(dataset, dataset_type, FileFormat(file_format))]
    model_path = str(train_paths.get_text_base_model_path(model))
    config["base_model"] = model_path
    config["mlflow_experiment_name"] = dataset
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = str(output_dir)

    if log_wandb:
        config["wandb_runid"] = f"{task_id}_{expected_repo_name}"
        config["wandb_name"] = f"{task_id}_{expected_repo_name}"
        config["wandb_mode"] = "offline"
        os.makedirs(train_cst.WANDB_LOGS_DIR, exist_ok=True)
    else:
        for key in list(config.keys()):
            if key.startswith("wandb"):
                config.pop(key)

    config = update_flash_attention(config, model)

    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions],
            task_id,
            destination_dir=train_cst.AXOLOTL_DIRECTORIES["src"],
        )
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]
    elif isinstance(dataset_type, EnvironmentDatasetType):
        # Switch based on the environment
        if dataset_type.environment_name == "alfworld":
            config["trl"]["rollout_func"] = "alfworld.alfworld_rollout_first_prompt_and_completion"
            config["trl"]["reward_funcs"] = ["alfworld.alfworld_rollout_reward_func"]
            config["trl"]["reward_weights"] = [1.0]

    if file_format != FileFormat.HF.value:
        for ds in config["datasets"]:
            ds["ds_type"] = "json"

            if "path" in ds:
                ds["path"] = train_cst.AXOLOTL_DIRECTORIES["data"]

            ds["data_files"] = [os.path.basename(dataset)]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        config["special_tokens"] = {"pad_token": tokenizer.eos_token}
    if tokenizer.bos_token_id is None:
        if tokenizer.eos_token is not None:
            config["special_tokens"] = {"bos_token": tokenizer.eos_token}
        else:
            config["special_tokens"] = {"bos_token": ""}

    config_path = os.path.join(train_cst.AXOLOTL_DIRECTORIES["configs"], f"{task_id}.yml")
    save_config(config, config_path)
    return config_path


def run_training(config_path):
    print(f"Starting training with config: {config_path}", flush=True)
    """Run the training process using the specified config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    training_env = os.environ.copy()
    training_env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    training_env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    training_command = ["accelerate", "launch", "-m", "axolotl.cli.train", config_path]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(training_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

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
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--dataset-type", required=True, help="JSON string of dataset type config")
    parser.add_argument(
        "--task-type", required=True, choices=["InstructTextTask", "DpoTask", "GrpoTask", "ChatTask", "EnvTask"], help="Type of task"
    )
    parser.add_argument("--file-format", required=True, choices=["csv", "json", "hf", "s3"], help="File format")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)

        if args.task_type == TaskType.DPOTASK.value:
            dataset_type = DpoDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.INSTRUCTTEXTTASK.value:
            dataset_type = InstructTextDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.CHATTASK.value:
            dataset_type = ChatTemplateDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.GRPOTASK.value:
            dataset_type = GrpoDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.ENVIRONMENTTASK.value:
            dataset_type = EnvironmentDatasetType(**dataset_type_dict)
        else:
            sys.exit(f"Unsupported task type: {args.task_type}")
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    if args.task_type == TaskType.DPOTASK.value:
        adapt_columns_for_dpo_dataset(dataset_path, dataset_type, apply_formatting=True)
    elif args.task_type == TaskType.GRPOTASK.value:
        adapt_columns_for_grpo_dataset(dataset_path, dataset_type)
    elif args.task_type == TaskType.ENVIRONMENTTASK.value:
        #adapt_columns_for_environment_dataset(dataset_path, dataset_type) # NOTE: Remove to test RO cache
        pass
    
    dataset_path = copy_dataset_to_axolotl_directories(dataset_path)

    output_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    config_path = create_config(
        args.task_id,
        args.model,
        dataset_path,
        dataset_type,
        args.file_format,
        output_dir,
        args.expected_repo_name,
        log_wandb=True,
    )

    run_training(config_path)

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    asyncio.run(main())
