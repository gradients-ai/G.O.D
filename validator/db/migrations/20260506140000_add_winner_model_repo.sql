-- migrate:up

ALTER TABLE tournaments ADD COLUMN IF NOT EXISTS winner_model_repo TEXT;
ALTER TABLE tournaments ADD COLUMN IF NOT EXISTS winner_model_base TEXT;

-- migrate:down

ALTER TABLE tournaments DROP COLUMN IF EXISTS winner_model_repo;
ALTER TABLE tournaments DROP COLUMN IF EXISTS winner_model_base;
