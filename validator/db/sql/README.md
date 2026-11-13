# Validator SQL modules

Query functions for the validator database, one module per domain. Table and column names come from `validator/db/constants.py`.

## Contents

| Path | Purpose |
| --- | --- |
| `auditing.py` | Task history and audit detail queries, and the latest scores URL. |
| `benchmark_tasks.py` | Benchmark root tasks, their copies and timelines. |
| `calibration.py` | Dataset rows already used in a calibration set (`dataset_calibration_rows`). |
| `continuous_sft.py` | Continuous-SFT lineage state and the per-task position record. |
| `dedup.py` | Duplicate-submission reviews (`tournament_dedup_reviews`). |
| `gpu_costs.py` | GPU usage runs and trainer capacity intervals for cost reports. |
| `grpo.py` | Reward function storage and lookup. |
| `nodes.py` | Node and metagraph persistence. |
| `quant_calibration_references.py` | Reference loss on the published calibration rows, per task and hotkey. |
| `quant_references.py` | Reference loss per task, hotkey and evaluation GPU type (`quant_sku_references`). |
| `quant_sources.py` | Links from a compression tournament and its tasks to their text tournament, task and model. |
| `submissions_and_scoring.py` | Submissions, task node scores and losses, compression decode results. |
| `task_failure_reviews.py` | Reviews of tasks where most trainings failed (`tournament_task_failure_reviews`). |
| `tasks.py` | Task create, read, update, stats and status transitions. |
| `tournament_performance.py` | Tournament performance queries. |
| `tournament_seeds.py` | Seed tasks: the tasks a boss trains ahead of a round (`tournament_seed_tasks`). |
| `tournaments.py` | Tournaments, rounds, participants, training rows and results. |
| `transfers.py` | Transfers, balances and balance events. |

## See also

- Schema: [migrations/README.md](../migrations/README.md).
- Tables that matter: [validator/db/README.md](../README.md).
