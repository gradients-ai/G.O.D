-- migrate:up
ALTER TABLE pvp_pair_results
ADD COLUMN IF NOT EXISTS deployment_id TEXT;

CREATE INDEX IF NOT EXISTS idx_pvp_pair_results_deployment_id
ON pvp_pair_results(deployment_id);

-- migrate:down
DROP INDEX IF EXISTS idx_pvp_pair_results_deployment_id;

ALTER TABLE pvp_pair_results
DROP COLUMN IF EXISTS deployment_id;
