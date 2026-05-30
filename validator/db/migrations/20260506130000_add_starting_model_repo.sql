-- migrate:up

ALTER TABLE task_nodes ADD COLUMN IF NOT EXISTS starting_model_repo TEXT;

-- migrate:down

ALTER TABLE task_nodes DROP COLUMN IF EXISTS starting_model_repo;
