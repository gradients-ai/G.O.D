-- migrate:up
ALTER TABLE image_tasks
ADD COLUMN IF NOT EXISTS trigger_word TEXT;

-- migrate:down
ALTER TABLE image_tasks
DROP COLUMN IF EXISTS trigger_word;
