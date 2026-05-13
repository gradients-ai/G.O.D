"""
Model prep container entrypoint.
Augments model (if config provided), computes baseline stats, uploads to HF.
Outputs JSON result on the last line of stdout for the caller to parse.
"""

import argparse
import asyncio
import hashlib
import json
import os
import tempfile

import torch
from huggingface_hub import HfApi
from huggingface_hub import repo_exists
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from core.utils import download_s3_file

from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.constants import EnvironmentName
from core.models.utility_models import TaskType
from trainer.model_prep.augmentation import augment_model
from trainer.model_prep.env_stats import compute_env_stats
from trainer.model_prep.stats import compute_text_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--training-data", required=True, help="S3 URL or local path to training data")
    parser.add_argument("--task-type", default="instruct", help="Task type: instruct, dpo, grpo, chat")
    parser.add_argument("--aug-type", choices=[t.value for t in AugmentationType], default=None)
    parser.add_argument("--scope", choices=[s.value for s in AugmentationScope], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--intensity", type=float, default=None)
    parser.add_argument("--reward-functions", default=None, help="JSON list of reward function objects (for GRPO)")
    parser.add_argument("--env-configs", default=None, help="JSON dict of {env_name: {url, task_id_min, task_id_max, num_episodes, eval_payload_extra}}")
    return parser.parse_args()


def build_augmentation_config(args) -> AugmentationConfig | None:
    if args.aug_type is None:
        return None
    return AugmentationConfig(
        aug_type=AugmentationType(args.aug_type),
        scope=AugmentationScope(args.scope),
        seed=args.seed,
        intensity=args.intensity,
    )


def generate_anonymous_repo_name(model_id: str, seed: int) -> str:
    """Generate an opaque repo name that doesn't leak the original model identity."""
    hf_username = os.environ.get("HUGGINGFACE_USERNAME", "gradients-io")
    hash_input = f"{model_id}:{seed}"
    repo_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    return f"{hf_username}/augmented-{repo_hash}"


def load_training_data(path: str) -> list[dict]:
    """Load all training data from a JSON file."""
    if path.startswith("http"):
        local_path = asyncio.run(download_s3_file(path))
    else:
        local_path = path

    with open(local_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    return []


def upload_augmented_model(model, tokenizer, repo_id: str, hf_token: str) -> None:
    """Upload augmented model to HuggingFace, scrubbing identity."""
    print(f"Uploading augmented model to {repo_id}")

    model.config._name_or_path = repo_id
    model.push_to_hub(repo_id, token=hf_token, private=False)
    tokenizer.push_to_hub(repo_id, token=hf_token, private=False)

    # Scrub _name_or_path from config
    api = HfApi(token=hf_token)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = api.hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=tmp, token=hf_token)
        with open(config_path, "r") as f:
            config = json.load(f)
        if "_name_or_path" in config:
            del config["_name_or_path"]
            modified_path = os.path.join(tmp, "config_clean.json")
            with open(modified_path, "w") as f:
                json.dump(config, f, indent=2)
            api.upload_file(
                path_or_fileobj=modified_path,
                path_in_repo="config.json",
                repo_id=repo_id,
            )

    print(f"Upload complete: {repo_id}")


def main():
    args = parse_args()
    aug_config = build_augmentation_config(args)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")

    print(f"Loading model: {args.model}", flush=True)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Multi-GPU detected ({n_gpus}), using device_map=auto", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, token=hf_token, device_map="auto",
        )
    elif torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, token=hf_token,
        )
        model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)

    augmented_model_id = None

    if aug_config is not None:
        repo_id = generate_anonymous_repo_name(args.model, aug_config.seed)

        if repo_exists(repo_id, token=hf_token):
            print(f"Augmented model already exists at {repo_id}, skipping augmentation")
            augmented_model_id = repo_id
        else:
            print(f"Applying augmentation: {aug_config.aug_type.value}", flush=True)
            augment_model(model, aug_config)
            upload_augmented_model(model, tokenizer, repo_id, hf_token)
            augmented_model_id = repo_id

    # Compute baseline stats
    print("Computing baseline stats...", flush=True)

    if args.env_configs:
        raw_configs: dict[str, dict] = json.loads(args.env_configs)
        env_configs = {EnvironmentName(k): v for k, v in raw_configs.items()}
        stats = asyncio.run(compute_env_stats(
            model_path=args.model,
            model=model,
            env_configs=env_configs,
        ))
    else:
        data_records = load_training_data(args.training_data)
        reward_functions = json.loads(args.reward_functions) if args.reward_functions else None

        if data_records:
            task_type_enum = TaskType(args.task_type)
            stats = compute_text_stats(
                model, tokenizer, data_records,
                task_type=task_type_enum,
                reward_functions=reward_functions,
            )
        else:
            print("Warning: no training data available for stats", flush=True)
            stats = None

    if stats and hasattr(stats, "training"):
        print(f"Baseline stats: loss={stats.training.init_loss:.4f}, entropy={stats.training.output_entropy:.4f}", flush=True)
    elif stats and hasattr(stats, "env_stats"):
        for env_name, env_stat in stats.env_stats.items():
            print(f"  {env_name.value}: {env_stat.num_episodes} episodes, mean={env_stat.mean_score:.3f}", flush=True)

    # Output result as JSON on last line (parsed by caller)
    result = {
        "augmented_model_id": augmented_model_id,
        "baseline_stats": stats.model_dump() if stats else None,
    }
    def sanitize_floats(obj):
        """Replace NaN/Inf with None for JSON compliance."""
        if isinstance(obj, float):
            if obj != obj or obj == float("inf") or obj == float("-inf"):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: sanitize_floats(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_floats(v) for v in obj]
        return obj

    print(json.dumps(sanitize_floats(result)), flush=True)


if __name__ == "__main__":
    main()
