# Validator SQL Modules

Database query helpers grouped by domain.

## Contents

- `auditing.py`: audit detail and hotkey task history queries.
- `benchmark_tasks.py`: benchmark root task and copied benchmark task queries.
- `continuous_sft.py`: continuous-SFT lineage state and the per-task chunk record (`continuous_sft_state`, `continuous_sft_task_chunks`), and the cursor advance across chunk layouts.
- `grpo.py`: reward function persistence and GRPO helper queries.
- `nodes.py`: node/metagraph persistence.
- `quant_sources.py`: the links from a quant tournament to its text tournament and from each quant task to the text task and model it compresses (`quant_tournament_sources`, `quant_task_sources`).
- `quant_references.py`: the per-(task, SKU) bf16 reference cache (`quant_sku_references`).
- `submissions_and_scoring.py`: submissions, task node scores, losses, and node stats.
- `tasks.py`: task CRUD, task stats, and task status transitions.
- `tournament_performance.py`: tournament performance history queries.
- `tournaments.py`: tournament, round, participant, training, and result persistence.
- `transfers.py`: balance event and transfer persistence.
