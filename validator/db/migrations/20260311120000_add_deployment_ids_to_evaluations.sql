-- migrate:up

ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS deployment_id TEXT;
ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS deployment_env_id TEXT;

-- migrate:down

ALTER TABLE evaluations DROP COLUMN IF EXISTS deployment_id;
ALTER TABLE evaluations DROP COLUMN IF EXISTS deployment_env_id;
