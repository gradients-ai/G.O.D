-- migrate:up

ALTER TABLE trainers_gpus
    ADD COLUMN IF NOT EXISTS product_name TEXT,
    ADD COLUMN IF NOT EXISTS interconnect VARCHAR(32) NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS nvlink BOOLEAN NOT NULL DEFAULT FALSE;

-- migrate:down

ALTER TABLE trainers_gpus
    DROP COLUMN IF EXISTS product_name,
    DROP COLUMN IF EXISTS interconnect,
    DROP COLUMN IF EXISTS nvlink;
