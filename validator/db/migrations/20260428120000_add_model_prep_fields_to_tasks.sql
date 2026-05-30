-- migrate:up
ALTER TABLE tasks
    ADD COLUMN augmented_model_id TEXT DEFAULT NULL,
    ADD COLUMN baseline_stats JSONB DEFAULT NULL;

-- migrate:down
ALTER TABLE tasks
    DROP COLUMN IF EXISTS augmented_model_id,
    DROP COLUMN IF EXISTS baseline_stats;
