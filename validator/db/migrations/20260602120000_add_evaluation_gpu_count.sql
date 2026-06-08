-- migrate:up

ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS gpu_count INTEGER;
ALTER TABLE evaluations DROP COLUMN IF EXISTS deployment_env_id;

CREATE INDEX IF NOT EXISTS idx_evaluations_active_deployment_gpu
    ON evaluations(netuid, evaluation_status, deployment_id)
    WHERE deployment_id IS NOT NULL;

-- migrate:down

DROP INDEX IF EXISTS idx_evaluations_active_deployment_gpu;

ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS deployment_env_id TEXT;
ALTER TABLE evaluations DROP COLUMN IF EXISTS gpu_count;
