-- migrate:up
ALTER TABLE task_nodes
    ADD COLUMN IF NOT EXISTS baseline_stats JSONB DEFAULT NULL;

-- migrate:down
ALTER TABLE task_nodes
    DROP COLUMN IF EXISTS baseline_stats;
