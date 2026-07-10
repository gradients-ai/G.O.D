-- migrate:up
-- Per-lineage state for the continuous-SFT boss-round task (one row per lineage slug, e.g.
-- 'quasar' / 'qwen'). Rows are created lazily by the app (upsert on completion); no seed rows
-- here so the lineage set stays defined in code (CONTINUOUS_SFT_LINEAGES), not the migration.
-- train_index = monotonic cursor passed to the content service each run; the (stateless) service
-- maps it to a stage-1 chunk (train_index % num_chunks) and returns fresh randomized URLs.
-- Advanced by one each time that lineage's continuous-SFT boss task completes.
-- last_winner_repo = HF repo of the lowest-eval-loss winner from the previous continuous-SFT
-- task for this lineage, carried forward as the next task's base model (NULL on first run).
CREATE TABLE IF NOT EXISTS continuous_sft_state (
    lineage TEXT PRIMARY KEY,
    train_index INT NOT NULL DEFAULT 0,
    last_winner_repo TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- migrate:down
DROP TABLE IF EXISTS continuous_sft_state;
