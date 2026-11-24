-- migrate:up

-- The job holding the reservation (validator/tournament/gpu_reservations.py) and when it took it. NULL reserved_by on a
-- reserved GPU means the hold comes from the trainer's own report (busy or unreachable), not from a dispatched job.
ALTER TABLE trainers_gpus
    ADD COLUMN IF NOT EXISTS reserved_by TEXT,
    ADD COLUMN IF NOT EXISTS reserved_at TIMESTAMPTZ;

-- migrate:down

ALTER TABLE trainers_gpus
    DROP COLUMN IF EXISTS reserved_at,
    DROP COLUMN IF EXISTS reserved_by;
