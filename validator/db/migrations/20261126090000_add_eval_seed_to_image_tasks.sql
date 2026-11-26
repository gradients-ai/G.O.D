-- migrate:up
ALTER TABLE image_tasks
ADD COLUMN IF NOT EXISTS eval_seed BIGINT;

-- migrate:down
ALTER TABLE image_tasks
DROP COLUMN IF EXISTS eval_seed;
