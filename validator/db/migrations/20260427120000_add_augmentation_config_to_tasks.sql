-- migrate:up
ALTER TABLE tasks
    ADD COLUMN augmentation_config JSONB DEFAULT NULL;

-- migrate:down
ALTER TABLE tasks
    DROP COLUMN IF EXISTS augmentation_config;
