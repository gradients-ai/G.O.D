#!/usr/bin/env python3
"""Smoke-test model prep for selected environment baselines.

Examples:
    python -m ops.tools.trainer.model_prep_env_smoke --model Qwen/Qwen2.5-0.5B-Instruct
    python -m ops.tools.trainer.model_prep_env_smoke --model Qwen/Qwen2.5-0.5B-Instruct --episodes 2 --gpu-ids 0,1
"""

import argparse
import json
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.constants.environments import EnvironmentName
from core.models.payload_models import EnvConfig
from core.models.payload_models import ModelPrepResponse
from core.models.task_models import TaskType
from validator.tasks.prep.model import _build_env_configs


DEFAULT_ENVS = ("liarsdice", "intercode")
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DUMMY_ENV_DATASET = "dummy_environment_dataset.json"

_ENV_ALIASES = {
    "clobber": EnvironmentName.CLOBBER.value,
    "liarsdice": EnvironmentName.LIARS_DICE.value,
    "liars-dice": EnvironmentName.LIARS_DICE.value,
    "liars_dice": EnvironmentName.LIARS_DICE.value,
    "intercode": EnvironmentName.INTERCODE.value,
}


def parse_env_name(raw_env: str) -> EnvironmentName:
    normalized = raw_env.strip().lower()
    value = _ENV_ALIASES.get(normalized, normalized)
    try:
        return EnvironmentName(value)
    except ValueError as exc:
        supported = ", ".join(env.value for env in EnvironmentName)
        raise argparse.ArgumentTypeError(f"unsupported environment {raw_env!r}; supported: {supported}") from exc


def parse_gpu_ids(raw_gpu_ids: str) -> list[int]:
    try:
        gpu_ids = [int(part.strip()) for part in raw_gpu_ids.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpu-ids must be a comma-separated list of integers") from exc
    if not gpu_ids:
        raise argparse.ArgumentTypeError("--gpu-ids must include at least one GPU id")
    return gpu_ids


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def build_selected_env_configs(
    env_names: list[EnvironmentName],
    episodes: int | None,
) -> dict[EnvironmentName, EnvConfig]:
    canonical_configs = _build_env_configs()
    selected_configs: dict[EnvironmentName, EnvConfig] = {}

    for env_name in env_names:
        cfg = canonical_configs[env_name]
        if episodes is not None:
            cfg = cfg.model_copy(update={"num_episodes": episodes})
        selected_configs[env_name] = cfg

    return selected_configs


def print_env_plan(env_configs: dict[EnvironmentName, EnvConfig]) -> None:
    print("Model prep environment smoke test")
    for env_name, cfg in env_configs.items():
        print(
            f"- {env_name.value}: image={cfg.env_image}, "
            f"task_ids={cfg.task_id_min}-{cfg.task_id_max}, "
            "baseline=time-budgeted (MODEL_PREP_ENV_TIME_BUDGET_SECONDS, default 420s)"
        )
        if cfg.env_server_command:
            print(f"  command={' '.join(cfg.env_server_command)}")


def print_result_summary(result: ModelPrepResponse) -> None:
    print("\nModel prep completed.")
    print(f"Augmented model: {result.augmented_model_id or '(none)'}")

    stats = result.baseline_stats
    if stats is not None and hasattr(stats, "env_stats"):
        print("Environment stats:")
        for env_name, env_stat in stats.env_stats.items():
            print(
                f"- {env_name.value}: episodes={env_stat.num_episodes}, "
                f"mean={env_stat.mean_score:.4f}, min={env_stat.min_score:.4f}, max={env_stat.max_score:.4f}"
            )

    print("\nRaw response:")
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model repo/path to prep. Default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--envs",
        nargs="+",
        type=parse_env_name,
        default=[parse_env_name(env) for env in DEFAULT_ENVS],
        help="Environments to test. Defaults to liarsdice intercode.",
    )
    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=1,
        help="Compatibility override for legacy baseline episode payloads. Time-budgeted baselines ignore this. Default: 1.",
    )
    parser.add_argument(
        "--use-default-episodes",
        action="store_true",
        help="Use canonical compatibility episode counts instead of --episodes.",
    )
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, default=parse_gpu_ids("0"), help="Comma-separated GPU IDs.")
    parser.add_argument("--task-id", default=None, help="Optional task id for cache paths/container names.")
    parser.add_argument(
        "--training-data-url",
        default=DUMMY_ENV_DATASET,
        help="Placeholder dataset URL/path. Env model prep generates a proxy dataset in the downloader.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the selected env config and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_id = args.task_id or f"model-prep-env-smoke-{uuid.uuid4().hex[:8]}"
    episodes = None if args.use_default_episodes else args.episodes
    env_configs = build_selected_env_configs(args.envs, episodes)

    print_env_plan(env_configs)
    print(f"Model: {args.model}")
    print(f"Task ID: {task_id}")
    print(f"GPU IDs: {args.gpu_ids}")

    if args.dry_run:
        return 0

    from trainer.runtime import run_model_prep_container

    try:
        result = run_model_prep_container(
            task_id=task_id,
            model_id=args.model,
            training_data_url=args.training_data_url,
            task_type=TaskType.ENVIRONMENTTASK,
            augmentation_config=None,
            gpu_ids=args.gpu_ids,
            env_configs=env_configs,
            log_labels={"task_id": task_id, "script": "model_prep_env_smoke"},
        )
    except Exception as exc:
        print(f"\nModel prep smoke test failed: {exc}", file=sys.stderr)
        return 1

    print_result_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
