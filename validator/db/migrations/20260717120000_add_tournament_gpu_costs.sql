-- migrate:up

CREATE TABLE IF NOT EXISTS task_gpu_costs (
    task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id) ON DELETE CASCADE,
    training_wall_seconds NUMERIC(20, 6) NOT NULL DEFAULT 0,
    training_gpu_seconds NUMERIC(20, 6) NOT NULL DEFAULT 0,
    training_cost_usd NUMERIC(20, 8) NOT NULL DEFAULT 0,
    training_success_count INTEGER NOT NULL DEFAULT 0,
    training_failure_count INTEGER NOT NULL DEFAULT 0,
    prep_wall_seconds NUMERIC(20, 6) NOT NULL DEFAULT 0,
    prep_gpu_seconds NUMERIC(20, 6) NOT NULL DEFAULT 0,
    prep_cost_usd NUMERIC(20, 8) NOT NULL DEFAULT 0,
    prep_success_count INTEGER NOT NULL DEFAULT 0,
    prep_failure_count INTEGER NOT NULL DEFAULT 0,
    evaluation_wall_seconds NUMERIC(20, 6) NOT NULL DEFAULT 0,
    evaluation_gpu_seconds NUMERIC(20, 6) NOT NULL DEFAULT 0,
    evaluation_cost_usd NUMERIC(20, 8) NOT NULL DEFAULT 0,
    evaluation_success_count INTEGER NOT NULL DEFAULT 0,
    evaluation_failure_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_task_gpu_costs_tournament
    ON task_gpu_costs(tournament_id);

CREATE TABLE IF NOT EXISTS active_gpu_cost_runs (
    run_key TEXT PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (category IN ('training', 'prep', 'evaluation')),
    gpu_type TEXT NOT NULL,
    gpu_count INTEGER NOT NULL CHECK (gpu_count > 0),
    hourly_rate_per_gpu_usd NUMERIC(20, 8) NOT NULL CHECK (hourly_rate_per_gpu_usd >= 0),
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_active_gpu_cost_runs_task
    ON active_gpu_cost_runs(task_id);
CREATE INDEX IF NOT EXISTS idx_active_gpu_cost_runs_tournament
    ON active_gpu_cost_runs(tournament_id);

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

-- migrate:down

DROP TABLE IF EXISTS active_gpu_cost_runs;
DROP TABLE IF EXISTS task_gpu_costs;
DROP TABLE IF EXISTS trainer_gpu_capacity_intervals;
