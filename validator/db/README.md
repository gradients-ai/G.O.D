# Validator database

Postgres access for the validator: the connection pool, the schema migrations, the table and column names, and the SQL modules.

## Contents

| Path | Purpose |
| --- | --- |
| `constants.py` | Table and column names used by every SQL module. |
| `database.py` | Postgres connection and pool wrapper. Reads `DATABASE_URL`, or the `POSTGRES_*` values. |
| `migrations/` | dbmate migration files. See [migrations/README.md](migrations/README.md). |
| `sql/` | Query modules grouped by domain. See [sql/README.md](sql/README.md). |

## Tables that matter

| Table | Holds |
| --- | --- |
| `tasks` | One row per task, with the fields shared by all task types. |
| `instruct_text_tasks`, `chat_tasks`, `dpo_tasks`, `grpo_tasks`, `image_tasks`, `quant_tasks` | The fields of one task type, keyed by task id. |
| `task_nodes` | One row per (task, miner): scores, losses and, for compression tasks, the artifact and decoder measurements. |
| `submissions` | The repository each miner submitted for a task. |
| `evaluations` | The evaluation jobs and their state. |
| `tournaments`, `tournament_rounds`, `tournament_participants`, `tournament_tasks` | Tournament state. |
| `tournament_task_hotkey_trainings` | One row per (task, miner) training run. |
| `tournament_seed_tasks` | The tasks a boss trains ahead of the round that uses them. |
| `quant_tournament_sources`, `quant_task_sources` | Which text tournament and text task each compression tournament and task takes its model from. |
| `quant_sku_references` | The uncompressed reference loss per task, hotkey and evaluation GPU type. |
| `continuous_sft_state`, `continuous_sft_task_chunks` | The position of each continuous-SFT lineage and the position each task was created with. |
| `tournament_boss_rank_comparisons` | The rank comparison between the boss and each paid contender, written when a text or compression tournament completes with the boss keeping the crown. |
| `trainers_gpus` | The GPUs each trainer reports. |
| `transfers`, `coldkey_balances`, `balance_events` | Entry-fee transfers and balances. |

<details>
<summary>Compression columns</summary>

- `tournament_participants.decoder_object_name`, `decoder_xz_bytes`, `decoder_sha256`: the snapshot of an entrant's `decoder/` at the pinned commit. NULL for entrants without a decoder.
- `task_nodes.artifact_bytes`, `decoder_bytes`, `decoder_sha256`, `quant_failure`, `decoder_build_warning`, `decode_artifact_files`, `decoded_files`: the decode outcome of a freeform entry, written by `set_quant_decode_result`.
- `task_nodes.eval_gpu`: the GPU a compression entry was scored on, as JSON (`QuantEvalGpu`). NULL for other task types.
- `quant_tasks.upstream_task_id`: the text task whose model the compression task compresses.

</details>

<details>
<summary>Seed task rows</summary>

- Primary key `(tournament_id, role, attempt)`. A replacement is a row with the next `attempt`.
- `status` is one of `training`, `trained`, `ready`, `replaced`, `failed`.
- `boss_repo` and `boss_commit` pin the code the seed was trained with.
- `attached_round_id` is written once, when a round attaches the seed.
- Model: `TournamentSeedTask` in `validator/tournament/models.py`. SQL: `sql/tournament_seeds.py`.

</details>

## See also

- Tournament rules and flows: [validator/tournament/README.md](../tournament/README.md).
- Scoring: [validator/scoring/README.md](../scoring/README.md).
