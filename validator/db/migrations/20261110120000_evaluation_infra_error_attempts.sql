-- migrate:up

-- Evaluation attempts of this row that ended in an infrastructure/evaluation error (never the miner's fault).
-- Lifecycle re-queues the row until this reaches MAX_EVAL_INFRA_ATTEMPTS, then holds it in evaluation_status
-- 'infra_held' for a human instead of scoring it; a successful evaluation resets it.
ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS infra_error_attempts INTEGER NOT NULL DEFAULT 0;

-- migrate:down

ALTER TABLE evaluations DROP COLUMN IF EXISTS infra_error_attempts;
