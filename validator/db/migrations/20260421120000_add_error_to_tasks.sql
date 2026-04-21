-- migrate:up

ALTER TABLE tasks
ADD COLUMN IF NOT EXISTS error TEXT;

-- migrate:down

ALTER TABLE tasks DROP COLUMN IF EXISTS error;
