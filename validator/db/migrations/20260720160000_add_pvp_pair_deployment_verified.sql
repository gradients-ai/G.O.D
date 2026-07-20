-- migrate:up
ALTER TABLE pvp_pair_results
ADD COLUMN IF NOT EXISTS deployment_verified BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_pvp_pair_results_deployment_verified
ON pvp_pair_results(deployment_verified)
WHERE deployment_id IS NOT NULL;

-- migrate:down
DROP INDEX IF EXISTS idx_pvp_pair_results_deployment_verified;

ALTER TABLE pvp_pair_results
DROP COLUMN IF EXISTS deployment_verified;
