-- migrate:up
ALTER TABLE env_tasks ADD COLUMN IF NOT EXISTS eval_seed INTEGER;

-- migrate:down
ALTER TABLE env_tasks DROP COLUMN IF EXISTS eval_seed;
