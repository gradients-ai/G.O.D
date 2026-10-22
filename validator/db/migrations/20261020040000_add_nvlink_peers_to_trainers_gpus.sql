-- migrate:up

-- NULL means the trainer could not report NVLink topology (older trainer build).
ALTER TABLE trainers_gpus
    ADD COLUMN IF NOT EXISTS nvlink_peers INT[];

-- migrate:down

ALTER TABLE trainers_gpus
    DROP COLUMN IF EXISTS nvlink_peers;
