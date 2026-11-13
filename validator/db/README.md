# Validator Database

Database connection code, schema migrations, table/column constants, and SQL access modules.

## Contents

- `constants.py`: table names, column names, and SQL constants.
- `database.py`: Postgres connection/pool wrapper.
- `migrations/`: dbmate migration files.
- Deploy order: apply migrations before starting new code (see `migrations/README.md`).
- `sql/`: query modules grouped by domain.


## Quant decoder snapshot

- `tournament_participants.decoder_object_name`, `decoder_xz_bytes`, `decoder_sha256`: the snapshot of the entrant's `decoder/` at their registered commit (MinIO object, counted .tar.xz bytes, .tar.xz sha256 the trainer verifies). Written by `add_tournament_participants` and `enroll_tournament_participant_with_fee`; NULL for image/environment entrants and a boss without a decoder.
- `submissions_and_scoring.set_quant_decode_result` writes a freeform entry's trainer-side decode outcome (`artifact_bytes`, `decoder_bytes`, `decoder_sha256`, `quant_failure`, `decoder_build_warning`) to its `task_nodes` row; `get_quant_decode_rows` reads them back as `QuantDecodeRow`, settled once `decoder_sha256` or `quant_failure` is set (evaluation later overwrites the byte counts and `quant_failure`, never `decoder_sha256`).

## Quant references (`20261020100000_quant_references.sql`)

- `quant_tasks.reference_hotkey`: unused; Python neither reads nor writes it.
- `quant_tasks.upstream_task_id`: the text seed task whose boss output this chained task compresses (a seed of the bound text tournament: `text_r1`, `text_r2`, `text_r3` or `text_continuous`), for the miner rounds R1-R3 and the six boss-round slots (FK to `tasks`, `ON DELETE SET NULL`); written by chaining, NULL otherwise.

## Quant eval GPU (`20261027120000_add_quant_eval_gpu.sql`)

- `task_nodes.eval_gpu` (JSONB `QuantEvalGpu`: `name`, `sm_count`, `sku`, `override`): the GPU a quant row was scored on, written by `set_task_node_losses` with every raw quant result. Quant evals run on H100 SXM or PCIe (`QUANT_EVAL_ALLOWED_SKUS`), so this is the record of which; `sku` NULL with `override` true marks a test-box override and must never appear in prod. NULL for non-quant rows and rows scored before the column.
- `task_nodes.decoder_build_warning`: the trainer's note that the entry's `build.sh` exited 0 but left `/build` empty or logged a failure. Informational, never scored.

## Quant SKU references (`20261027180000_add_quant_sku_references.sql`)

- `quant_sku_references(task_id, hotkey, sku, model, loss, per_example_losses JSONB, eval_set_fingerprint, eval_gpu JSONB, measured_at)`, primary key `(task_id, hotkey, sku)` from `20261029100000_quant_sku_references_per_hotkey.sql`, `ON DELETE CASCADE` with the task: the bf16 reference every quant loss on the task is normalised by, one row per (reference hotkey, SKU). `hotkey` is the miner whose own model was scored on an own-model task (`TrainingStartPoint.OWN_R1_MODEL`), `''` the task-common reference (boss and organic tasks). `sku` is `QuantEvalGpu.reference_key` (`H100 SXM`, `H100 PCIe`, or `override:<name>/<sm_count>` on a test box); `model` is the repo scored (the miner's own model on an own-model task, else the common input). Written by `sql/quant_references.add_quant_sku_reference` (first writer wins per `(task_id, hotkey, sku)`, `ON CONFLICT DO NOTHING`, returns the stored row), read by `get_quant_sku_references` (keyed `(hotkey, sku)`). `per_example_losses` keeps the scorer's example order with a non-finite entry (an example with no supervised token) stored as null and read back as NaN, through the same `json_safe_losses` / `losses_from_column` pair as `task_nodes.per_example_losses`; `QuantSkuReference.matches` treats two such entries as equal.
- The migration backfills each `quant_tasks` reference that recorded its SKU, then drops the four `quant_tasks.reference_*` columns. The down migration restores each task's earliest-measured common-input reference.
- `task_nodes.eval_gpu` is read back: `get_task_node_losses` and `get_all_scores_and_losses_for_task` return it (raw JSON), and `get_task_node_eval_gpus` returns it per hotkey for the paired gates.

## Tournament seed tasks (`20261109120000_tournament_seed_tasks.sql`)

- `tournament_seed_tasks(tournament_id, role, attempt, task_id, status, boss_repo, boss_commit, attached_round_id, prep_rejected, merged_output_repo, created_at, updated_at)`, primary key `(tournament_id, role, attempt)`, `task_id` `ON DELETE CASCADE` with the task. One row per task the champion trains upfront (`validator/tournament/seed_tasks.py`). `role` is one of `text_r1`, `text_r2`, `text_r3`, `text_continuous` (the text tournament's seeds, trained by the text boss at tournament start; added by `20261113130000_text_seed_roles.sql`) or `boss_r2_freeform`, `boss_r2_fixed`, `boss_r3_freeform`, `boss_r3_fixed`, `boss_continuous_freeform`, `boss_continuous_fixed` (the quant boss round's six compression tasks, trained by the quant boss as each source model lands). The roles `input_small`, `input_large`, `boss_instruct`, `boss_chained_freeform`, `boss_chained_fixed` and `boss_continuous_instruct` stay legal in the CHECK and are not written. `merged_output_repo` (nullable, text seeds) is the full model the validator merged from the boss's LoRA adapter output, used as the seed's output when set. `status` is `training`, `trained` (attached seeds: the boss's run is done, held for its round), `ready`, `replaced` (superseded by attempt+1) or `failed` (out of `SEED_MAX_REPLACEMENTS`; the tournament halts). `boss_repo` and `boss_commit` are the pinned code the seed trained with. `attached_round_id` (nullable) is written once when a round attaches the seed and never cleared, so a seed that was ever attached stays recognised after its `tournament_tasks` row is detached; an attached seed is never replaced (the boss would train after R3), and a post-attach prep failure resolves as a boss-side failure in the round result. `prep_rejected` (`BOOLEAN NOT NULL DEFAULT FALSE`, meaningful on `replaced` rows) is set when the row was replaced without charging the role's `SEED_MAX_REPLACEMENTS`: its own prep-stage dataset rejection (dataset-drawn roles only) or, for a compression seed, retirement because its text seed was replaced; the cap counts only `replaced` rows with `prep_rejected = FALSE`, and the datasets of the prep-rejected rows are excluded from the tournament's later redraws. Access in `validator/db/sql/tournament_seeds.py` (`TournamentSeedTask` in `validator/tournament/models.py`).
- Seed tasks are not `tournament_tasks` rows until a round attaches them (the text seeds by rounds 1-3 and the boss round, the quant seeds by the quant boss round). They count as tournament tasks for the task-list filters and `get_tournament_id_by_task_id`, and `get_hidden_seed_task_ids` names the seeds, and the quant tasks chained on them, whose data stays hidden on public and auditing endpoints until the last round using them completes. A text seed also stays hidden while a live `quant_task_sources` row of a running quant tournament names it, and a quant task chained on a text seed stays hidden until its own quant tournament is completed or cancelled (the text round's completion does not release it).

## Continuous-SFT chunks (`20261113120000_continuous_sft_50k_chunks.sql`)

- `continuous_sft_state.chunk_scheme` (a `ChunkScheme` value): the chunk layout `train_index` counts. The migration doubles every lineage's `train_index` once and sets the current scheme. Apply it between two tournaments, in the same window in which the content service switches.
- `continuous_sft_task_chunks(task_id, lineage, train_index, chunk_scheme, created_at)`, primary key `task_id`, `ON DELETE CASCADE` with the task: the cursor value and scheme each continuous-SFT task was created with. A continuous-SFT task with no row was created before the migration, on the previous scheme. At completion the cursor becomes `max(current, next)`, where `next` is `index + 1` for a task of the current scheme and `2 * index + 2` for one of the previous scheme, so no chunk repeats. Access in `validator/db/sql/continuous_sft.py`.

## Quant task sources (`20261113140000_quant_task_sources.sql`)

- `quant_tournament_sources(quant_tournament_id, source_tournament_id, created_at)`: which text tournament feeds which quant tournament. `quant_tournament_id` is the primary key and `source_tournament_id` is `UNIQUE`, so one text tournament feeds at most one quant tournament.
- `quant_task_sources(quant_task_id, quant_tournament_id, slot, source_tournament_id, source_role, source_task_id, source_model_repo, source_is_fallback, superseded, created_at, updated_at)`: one row per quant task of such a tournament, `quant_task_id` `ON DELETE CASCADE` with the task. `slot` is `r1`, `r2`, `r3` or one of the six boss-round compressions; `source_role` and `source_task_id` name the text seed task whose boss model the quant task compresses; `source_model_repo` is the exact hub repo handed to the task when it was released (NULL while held) and `source_is_fallback` marks the text task's base model used because the text boss produced no usable model; `superseded` marks a task replaced for the same slot. A unique index allows one live (not superseded) task per `(quant_tournament_id, slot)`. Access in `validator/db/sql/quant_sources.py`.
