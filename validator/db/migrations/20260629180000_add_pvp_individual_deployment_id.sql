-- migrate:up
ALTER TABLE pvp_individual_scores
ADD COLUMN IF NOT EXISTS deployment_id TEXT;

CREATE INDEX IF NOT EXISTS idx_pvp_individual_scores_deployment_id
ON pvp_individual_scores(deployment_id);

UPDATE pvp_pair_results ppr
SET deployment_id = pair_deployments.deployment_id,
    updated_at = CURRENT_TIMESTAMP
FROM (
    SELECT
        ppr_inner.task_id,
        ppr_inner.hotkey_a,
        ppr_inner.hotkey_b,
        MAX(e.deployment_id) AS deployment_id
    FROM pvp_pair_results ppr_inner
    JOIN evaluations e
        ON e.task_id = ppr_inner.task_id
        AND e.hotkey IN (ppr_inner.hotkey_a, ppr_inner.hotkey_b)
    WHERE e.deployment_id IS NOT NULL
    GROUP BY ppr_inner.task_id, ppr_inner.hotkey_a, ppr_inner.hotkey_b
    HAVING COUNT(DISTINCT e.deployment_id) = 1
) pair_deployments
WHERE ppr.task_id = pair_deployments.task_id
  AND ppr.hotkey_a = pair_deployments.hotkey_a
  AND ppr.hotkey_b = pair_deployments.hotkey_b
  AND ppr.deployment_id IS NULL
  AND pair_deployments.deployment_id IS NOT NULL;

UPDATE pvp_individual_scores pis
SET deployment_id = e.deployment_id,
    updated_at = CURRENT_TIMESTAMP
FROM evaluations e
WHERE pis.task_id = e.task_id
  AND pis.hotkey = e.hotkey
  AND pis.deployment_id IS NULL
  AND e.deployment_id IS NOT NULL;

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
