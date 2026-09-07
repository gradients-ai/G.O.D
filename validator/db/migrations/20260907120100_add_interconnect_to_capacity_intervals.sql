-- migrate:up

ALTER TABLE trainer_gpu_capacity_intervals
    ADD COLUMN IF NOT EXISTS interconnect VARCHAR(32) NOT NULL DEFAULT 'unknown';

-- Backfill from currently registered trainers when available.
UPDATE trainer_gpu_capacity_intervals AS intervals
SET interconnect = COALESCE(gpus.interconnect, 'unknown')
FROM trainers_gpus AS gpus
WHERE intervals.trainer_ip = gpus.trainer_ip
  AND intervals.gpu_id = gpus.gpu_id
  AND intervals.ended_at IS NULL;

-- migrate:down

ALTER TABLE trainer_gpu_capacity_intervals
    DROP COLUMN IF EXISTS interconnect;
