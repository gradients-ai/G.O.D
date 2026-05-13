#!/usr/bin/env python3
"""
Full pipeline E2E test. Uses the real validator code path:
  create task → prep (dataset download/split/S3 upload) → model prep dispatch → verify

Requires:
  - .vali.env loaded (DB, S3, HF, content service creds)
  - Trainer proxy running on a GPU node
  - Trainer registered in DB with available GPUs

Usage:
    source .vali.env
    python tests/e2e/run_full_pipeline.py [--skip-env] [--skip-augmentation] [--rounds N]
"""

import argparse
import asyncio
import json
import sys
import time
import traceback

from dotenv import load_dotenv
load_dotenv(".vali.env", override=True)

from core.models.model_prep_models import (
    DpoBaselineStats,
    EnvBaselineStats,
    GrpoBaselineStats,
    InstructBaselineStats,
)
from core.models.utility_models import TaskType
from validator.core.config import load_config
from validator.core.task_config_models import get_task_config
from validator.db.sql import tasks as task_sql
from validator.tasks.synthetic_scheduler import (
    _get_dpo_datasets,
    _get_instruct_text_datasets,
    _get_text_models,
    create_synthetic_dpo_task,
    create_synthetic_env_task,
    create_synthetic_grpo_task,
    create_synthetic_instruct_text_task,
)
from validator.utils.logging import get_logger
from validator.utils.model_prep import dispatch_augmentation_and_stats


logger = get_logger(__name__)


EXPECTED_TYPES = {
    "instruct": InstructBaselineStats,
    "dpo": DpoBaselineStats,
    "grpo": GrpoBaselineStats,
    "env": EnvBaselineStats,
}


async def test_instruct(config, models, datasets) -> dict:
    """Create + prep + model prep for an instruct task."""
    task = await create_synthetic_instruct_text_task(config, models, datasets)
    return await _prep_and_verify(task, config, "instruct")


async def test_dpo(config, models, datasets) -> dict:
    """Create + prep + model prep for a DPO task."""
    task = await create_synthetic_dpo_task(config, models, datasets)
    return await _prep_and_verify(task, config, "dpo")


async def test_grpo(config, models, datasets) -> dict:
    """Create + prep + model prep for a GRPO task."""
    task = await create_synthetic_grpo_task(config, models, datasets)
    return await _prep_and_verify(task, config, "grpo")


async def test_env(config, models, datasets) -> dict:
    """Create + prep + model prep for an env task."""
    task = await create_synthetic_env_task(config, models, datasets)
    return await _prep_and_verify(task, config, "env")


async def _prep_and_verify(task, config, task_type_label: str) -> dict:
    """Run task prep + model prep dispatch, verify results."""
    name = f"{task_type_label}_{task.model_id.split('/')[-1]}"
    start = time.time()

    try:
        # Step 1: Task prep (dataset download, split, S3 upload)
        print(f"  Prepping task (dataset download + split)...")
        task_config = get_task_config(task)
        task = await task_config.task_prep_function(task, config.keypair, config.psql_db)
        print(f"  Prep done: training_data={task.training_data is not None}, test_data={task.test_data is not None}")

        # Step 2: Model prep dispatch
        print(f"  Dispatching model prep to trainer...")
        reward_fns = getattr(task, "reward_functions", None)
        is_env_task = task.task_type == TaskType.ENVIRONMENTTASK

        prep_result = await dispatch_augmentation_and_stats(
            task_id=str(task.task_id),
            model_id=task.model_id,
            training_data_url=task.training_data,
            augmentation_config=task.augmentation_config,
            model_params_count=task.model_params_count,
            task_type=task.task_type,
            config=config,
            reward_functions=reward_fns,
            is_env_task=is_env_task,
        )

        elapsed = time.time() - start

        if prep_result is None:
            return {"name": name, "status": "FAIL", "error": "dispatch returned None (no GPUs?)", "elapsed": elapsed}

        # Step 2b: Save results to DB
        if prep_result.augmented_model_id:
            task.augmented_model_id = prep_result.augmented_model_id
        if prep_result.baseline_stats:
            task.baseline_stats = prep_result.baseline_stats
        await task_sql.update_task(task, config.psql_db)

        # Step 3: Verify results
        errors = []
        stats = prep_result.baseline_stats

        if stats is None:
            errors.append("baseline_stats is None")
        else:
            actual_type = type(stats).__name__
            expected_type = EXPECTED_TYPES.get(task_type_label)
            if expected_type and not isinstance(stats, expected_type):
                errors.append(f"Wrong type: {actual_type}, expected {expected_type.__name__}")

            if hasattr(stats, "training"):
                if stats.training.init_loss <= 0:
                    errors.append(f"init_loss <= 0: {stats.training.init_loss}")
                if not stats.training.grad_norms:
                    errors.append("Empty grad_norms")

            if hasattr(stats, "dataset"):
                if stats.dataset.total_tokens <= 0:
                    errors.append(f"total_tokens <= 0")
                if stats.dataset.vocab_size <= 0:
                    errors.append(f"vocab_size <= 0")

            if hasattr(stats, "weights"):
                if not stats.weights.by_group:
                    errors.append("Empty weight stats")

            if hasattr(stats, "env_stats"):
                if not stats.env_stats.episode_scores:
                    errors.append("Empty episode_scores")

            # Type-specific checks
            if isinstance(stats, InstructBaselineStats):
                if stats.dataset.prompt_tokens <= 0:
                    errors.append("prompt_tokens <= 0")
                if stats.dataset.completion_tokens <= 0:
                    errors.append("completion_tokens <= 0")

            elif isinstance(stats, DpoBaselineStats):
                if stats.training.implicit_reward_gap is None:
                    errors.append("Missing implicit_reward_gap")
                if stats.dataset.chosen_tokens <= 0:
                    errors.append("chosen_tokens <= 0")

            elif isinstance(stats, GrpoBaselineStats):
                if stats.dataset.prompt_tokens <= 0:
                    errors.append("prompt_tokens <= 0")

        if task.augmentation_config and not prep_result.augmented_model_id:
            errors.append("Augmentation config set but no augmented_model_id returned")

        status = "PASS" if not errors else "FAIL"

        # Print results
        print(f"  Status: {status} ({elapsed:.1f}s)")
        print(f"  Model: {task.model_id}")
        if task.ds:
            print(f"  Dataset: {task.ds}")
        if prep_result.augmented_model_id:
            print(f"  Augmented: {prep_result.augmented_model_id}")
        if stats and hasattr(stats, "training"):
            print(f"  Loss: {stats.training.init_loss:.4f}")
        if stats and hasattr(stats, "env_stats"):
            scores = stats.env_stats.episode_scores
            print(f"  Episodes: {len(scores)}, mean={sum(scores)/max(len(scores),1):.3f}")
        for e in errors:
            print(f"  ERROR: {e}")

        return {"name": name, "status": status, "errors": errors, "elapsed": elapsed}

    except Exception as e:
        elapsed = time.time() - start
        print(f"  EXCEPTION: {e}")
        traceback.print_exc()
        return {"name": name, "status": "ERROR", "error": str(e), "elapsed": elapsed}


async def main():
    parser = argparse.ArgumentParser(description="Full pipeline E2E test")
    parser.add_argument("--skip-env", action="store_true", help="Skip environment tasks")
    parser.add_argument("--skip-augmentation", action="store_true", help="Skip augmented runs")
    parser.add_argument("--rounds", type=int, default=1, help="Repeat each task type N times")
    parser.add_argument("--only", choices=["instruct", "dpo", "grpo", "env"], help="Run only this task type")
    args = parser.parse_args()

    print("Loading config...")
    config = load_config()
    await config.psql_db.connect()

    # Create model/dataset generators
    models = _get_text_models(config.keypair)
    instruct_datasets = _get_instruct_text_datasets(config.keypair)
    dpo_datasets = _get_dpo_datasets(config.keypair)

    results = []

    for round_num in range(args.rounds):
        if args.rounds > 1:
            print(f"\n{'#'*60}")
            print(f"ROUND {round_num + 1}/{args.rounds}")
            print(f"{'#'*60}")

        if not args.only or args.only == "instruct":
            print(f"\n--- Instruct Task ---")
            results.append(await test_instruct(config, models, instruct_datasets))

        if not args.only or args.only == "dpo":
            print(f"\n--- DPO Task ---")
            results.append(await test_dpo(config, models, dpo_datasets))

        if not args.only or args.only == "grpo":
            print(f"\n--- GRPO Task ---")
            results.append(await test_grpo(config, models, instruct_datasets))

        if (not args.only or args.only == "env") and not args.skip_env:
            print(f"\n--- Environment Task ---")
            results.append(await test_env(config, models, instruct_datasets))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] in ("FAIL", "ERROR"))

    for r in results:
        icon = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"  [{icon}] {r['name']} ({r.get('elapsed', 0):.1f}s)")

    total = len(results)
    print(f"\n  {passed}/{total} passed, {failed} failed")
    print(f"  Total: {sum(r.get('elapsed', 0) for r in results):.0f}s")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
