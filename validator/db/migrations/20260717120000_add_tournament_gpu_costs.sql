-- migrate:up

CREATE TABLE IF NOT EXISTS gpu_usage_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_key TEXT NOT NULL,
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (category IN ('training', 'prep', 'evaluation')),
    gpu_type TEXT NOT NULL,
    gpu_count INTEGER NOT NULL CHECK (gpu_count > 0),
    hourly_rate_per_gpu_usd NUMERIC(20, 8) NOT NULL CHECK (hourly_rate_per_gpu_usd >= 0),
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMPTZ,
    outcome TEXT CHECK (outcome IN ('success', 'failure')),
    wall_seconds NUMERIC(20, 6),
    gpu_seconds NUMERIC(20, 6),
    cost_usd NUMERIC(20, 8),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (
        (ended_at IS NULL AND outcome IS NULL AND wall_seconds IS NULL AND gpu_seconds IS NULL AND cost_usd IS NULL)
        OR
        (ended_at IS NOT NULL AND outcome IS NOT NULL AND wall_seconds IS NOT NULL AND gpu_seconds IS NOT NULL AND cost_usd IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_gpu_usage_runs_active_source
    ON gpu_usage_runs(source_key)
    WHERE ended_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_gpu_usage_runs_task
    ON gpu_usage_runs(task_id);
CREATE INDEX IF NOT EXISTS idx_gpu_usage_runs_tournament
    ON gpu_usage_runs(tournament_id);
CREATE INDEX IF NOT EXISTS idx_gpu_usage_runs_window
    ON gpu_usage_runs(started_at, ended_at);

CREATE TABLE IF NOT EXISTS trainer_gpu_capacity_intervals (
    id BIGSERIAL PRIMARY KEY,
    trainer_ip TEXT NOT NULL,
    gpu_id INTEGER NOT NULL,
    gpu_type TEXT NOT NULL,
    vram_gb INTEGER NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMPTZ,
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_trainer_gpu_capacity_active
    ON trainer_gpu_capacity_intervals(trainer_ip, gpu_id)
    WHERE ended_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_trainer_gpu_capacity_window
    ON trainer_gpu_capacity_intervals(started_at, ended_at);

-- Seed currently registered GPUs. Accurate history starts when this migration is applied.
INSERT INTO trainer_gpu_capacity_intervals (trainer_ip, gpu_id, gpu_type, vram_gb, started_at)
SELECT trainer_ip, gpu_id, gpu_type, vram_gb, CURRENT_TIMESTAMP
FROM trainers_gpus
ON CONFLICT (trainer_ip, gpu_id) WHERE ended_at IS NULL DO NOTHING;

CREATE OR REPLACE FUNCTION close_deleted_trainer_gpu_capacity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE trainer_gpu_capacity_intervals
    SET ended_at = CURRENT_TIMESTAMP
    WHERE trainer_ip = OLD.trainer_ip
      AND gpu_id = OLD.gpu_id
      AND ended_at IS NULL;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER close_deleted_trainer_gpu_capacity_trigger
AFTER DELETE ON trainers_gpus
FOR EACH ROW EXECUTE FUNCTION close_deleted_trainer_gpu_capacity();

-- migrate:down

DROP TRIGGER IF EXISTS close_deleted_trainer_gpu_capacity_trigger ON trainers_gpus;
DROP FUNCTION IF EXISTS close_deleted_trainer_gpu_capacity();
DROP TABLE IF EXISTS gpu_usage_runs;
DROP TABLE IF EXISTS trainer_gpu_capacity_intervals;
