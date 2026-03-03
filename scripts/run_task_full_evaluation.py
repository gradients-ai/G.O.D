#!/usr/bin/env python3
"""
Run full evaluation flow for a single task ID with logging.

Usage:
  python -m scripts.run_task_full_evaluation <task_id> [--num-gpus 4] [--no-fresh]
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv

from core.models.utility_models import TaskStatus
from validator.core.config import load_config
from validator.cycle.process_tasks import _evaluate_pending_pairs_for_task
from validator.cycle.process_tasks import compute_required_gpus
from validator.db.sql import tasks as tasks_sql


load_dotenv(".vali.env")


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamped = f"{datetime.utcnow().isoformat()}Z | {message}"
        print(stamped)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(stamped + "\n")


async def run(task_id: str, num_gpus: int | None, fresh: bool, log_path: Path) -> int:
    run_log = RunLogger(log_path)
    run_log.log(f"Starting task evaluation run for task_id={task_id}")

    config = load_config()
    await config.psql_db.connect()

    try:
        task = await tasks_sql.get_task(UUID(task_id), config.psql_db)
        if task is None:
            run_log.log("Task not found")
            return 1
        assert task.task_id is not None

        run_log.log(f"Task found: type={task.task_type} status={task.status} model={task.model_id}")

        if fresh:
            run_log.log("Fresh mode enabled: rebuilding evaluation pairs and forcing task to EVALUATING")
            await tasks_sql.add_task_evaluation_pairs(task.task_id, config.psql_db)
            task.status = TaskStatus.EVALUATING
            await tasks_sql.update_task(task, config.psql_db)
        else:
            run_log.log("Fresh mode disabled: using existing evaluation rows")

        if num_gpus is None:
            num_gpus = compute_required_gpus(task)
        run_log.log(f"Using num_gpus={num_gpus}")

        pending_rows = await tasks_sql.get_task_evaluations_by_status(task.task_id, "pending", config.psql_db)
        run_log.log(f"Pending pairs before run: {len(pending_rows)}")

        run_log.log("Running validator flow: _evaluate_pending_pairs_for_task(...)")
        await _evaluate_pending_pairs_for_task(task, num_gpus, config)
        run_log.log("Validator flow finished")

        rows = await tasks_sql.get_task_evaluation_rows(task.task_id, config.psql_db)
        status_counts: dict[str, int] = {}
        for row in rows:
            status = row["evaluation_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        run_log.log(f"Evaluation row status counts: {status_counts}")
        latest_task = await tasks_sql.get_task(UUID(task_id), config.psql_db)
        run_log.log(
            f"Task status after run: {latest_task.status}, n_eval_attempts={latest_task.n_eval_attempts}"
        )
        run_log.log("Run completed")

        final_status = str(latest_task.status)
        if final_status in (TaskStatus.SUCCESS.value, TaskStatus.FAILURE.value):
            return 0
        return 2
    finally:
        await config.psql_db.close()
        try:
            await config.httpx_client.aclose()
        except Exception:
            pass
        try:
            await config.redis_db.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full evaluation flow for one task")
    parser.add_argument("task_id", help="Task UUID")
    parser.add_argument("--num-gpus", type=int, default=None, help="Override computed GPU count")
    parser.add_argument("--no-fresh", action="store_true", help="Do not rebuild evaluation rows before run")
    parser.add_argument("--log-file", default=None, help="Optional explicit log file path")
    args = parser.parse_args()

    default_log = Path("logs") / f"task_eval_{args.task_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = Path(args.log_file) if args.log_file else default_log

    exit_code = asyncio.run(run(args.task_id, args.num_gpus, not args.no_fresh, log_path))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

