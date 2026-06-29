-- migrate:up
ALTER TABLE pvp_individual_scores
ADD COLUMN IF NOT EXISTS deployment_id TEXT;

CREATE INDEX IF NOT EXISTS idx_pvp_individual_scores_deployment_id
ON pvp_individual_scores(deployment_id);

UPDATE evaluations e
SET deployment_id = NULL,
    updated_at = CURRENT_TIMESTAMP
FROM env_tasks et
WHERE e.task_id = et.task_id
  AND e.deployment_id IS NOT NULL;

-- migrate:down
DROP INDEX IF EXISTS idx_pvp_individual_scores_deployment_id;

ALTER TABLE pvp_individual_scores
DROP COLUMN IF EXISTS deployment_id;
