#!/usr/bin/env python3
"""Re-evaluate specific miner submissions for a task. Edit the params below and run: python -m scripts.reevaluate"""

import asyncio
import os
import sys
from uuid import UUID

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".vali.env"))

from validator.cycle.process_tasks import compute_required_gpus
from validator.db.database import PSQLDB
from validator.db.sql.tasks import get_nodes_assigned_to_task, get_task
from validator.evaluation.scoring import _get_dataset_type, _update_scores, calculate_miner_ranking_and_scores, process_miners_pool


TASK_ID = ""
HOTKEYS = []
NUM_GPUS = None  # auto-detected from task if None



async def main() -> None:
    assert TASK_ID, "Set TASK_ID above"
    assert HOTKEYS, "Set HOTKEYS above"

    psql_db = PSQLDB()
    await psql_db.connect()

    try:
        task = await get_task(UUID(TASK_ID), psql_db)
        if task is None:
            print(f"Task {TASK_ID} not found")
            return

        all_miners = await get_nodes_assigned_to_task(str(task.task_id), psql_db)
        hotkey_set = set(HOTKEYS)
        miners = [m for m in all_miners if m.hotkey in hotkey_set]

        missing = hotkey_set - {m.hotkey for m in miners}
        if missing:
            print(f"Hotkeys not assigned to this task: {missing}")
            return

        num_gpus = NUM_GPUS if NUM_GPUS is not None else compute_required_gpus(task)
        dataset_type = _get_dataset_type(task)

        print(f"Task:     {task.task_id}")
        print(f"Model:    {task.model_id}")
        print(f"Type:     {task.task_type}")
        print(f"GPUs:     {num_gpus}")
        print(f"Miners:   {[m.hotkey for m in miners]}")
        print()

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.psql_db = psql_db

        results = await process_miners_pool(miners, task, cfg, num_gpus, dataset_type)
        results = calculate_miner_ranking_and_scores(results)
        await _update_scores(task, results, psql_db)

        print("=" * 60)
        print("Results:")
        print("=" * 60)
        for r in results:
            print(f"  {r.hotkey}  loss={r.test_loss:.6f}  score={r.score}  finetune={r.is_finetune}")
            if r.score_reason:
                print(f"    reason: {r.score_reason}")

    finally:
        await psql_db.close()


if __name__ == "__main__":
    asyncio.run(main())
