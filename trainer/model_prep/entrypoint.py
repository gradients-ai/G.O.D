"""
Model prep container entrypoint.
Augments model (if config provided), computes baseline stats, uploads to HF.
Outputs JSON result on the last line of stdout for the caller to parse.
"""

import argparse
import asyncio
import json
import os

from huggingface_hub import repo_exists

from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.constants import EnvironmentName
from core.models.utility_models import TaskType
from core.utils import download_s3_file
from trainer.model_prep.augmentation import augment_model
from trainer.model_prep.boss_base_model import prepare_boss_base_model
from trainer.model_prep.env_stats import compute_env_stats
from trainer.model_prep.model_io import generate_anonymous_repo_name
from trainer.model_prep.model_io import load_model_and_tokenizer
from trainer.model_prep.model_io import upload_augmented_model
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
    parser.add_argument("--source-model-repo", default=None, help="Prior tournament trained repo to prepare")
    parser.add_argument("--source-task-id", default=None, help="Prior tournament task ID that produced the source repo")
    parser.add_argument("--source-base-model-id", default=None, help="Original base model for the source repo")
    parser.add_argument("--reward-functions", default=None, help="JSON list of reward function objects (for GRPO)")
    parser.add_argument("--env-configs", default=None, help="JSON dict of {env_name: {url, task_id_min, task_id_max, num_episodes, eval_payload_extra}}")
    return parser.parse_args()


def build_augmentation_config(args) -> AugmentationConfig | None:
    if args.aug_type is None:
        return None
    return AugmentationConfig(
        aug_type=AugmentationType(args.aug_type),
        scope=AugmentationScope(args.scope) if args.scope else None,
        seed=args.seed,
        intensity=args.intensity,
        source_model_repo=args.source_model_repo,
        source_task_id=args.source_task_id,
        source_base_model_id=args.source_base_model_id,
    )


def load_training_data(path: str, max_records: int = 100) -> list[dict]:
    """Load training data from a JSON file."""
    if path.startswith("http"):
        local_path = asyncio.run(download_s3_file(path))
    else:
        local_path = path

    with open(local_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data[:max_records]
    return []


def main():
    args = parse_args()
    aug_config = build_augmentation_config(args)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")

    augmented_model_id = None
    repo_id = generate_anonymous_repo_name(args.model, aug_config.seed) if aug_config is not None else None

    if aug_config is None:
        print(f"Loading model: {args.model}", flush=True)
        model, tokenizer = load_model_and_tokenizer(args.model, hf_token)
    elif aug_config.aug_type == AugmentationType.BOSS_BASE_MODEL:
        model, tokenizer = prepare_boss_base_model(aug_config, repo_id, hf_token)
        augmented_model_id = repo_id
    else:
        if repo_exists(repo_id, token=hf_token):
            print(f"Augmented model already exists at {repo_id}, skipping augmentation")
            augmented_model_id = repo_id
            model, tokenizer = load_model_and_tokenizer(repo_id, hf_token)
        else:
            print(f"Loading model: {args.model}", flush=True)
            model, tokenizer = load_model_and_tokenizer(args.model, hf_token)
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
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
