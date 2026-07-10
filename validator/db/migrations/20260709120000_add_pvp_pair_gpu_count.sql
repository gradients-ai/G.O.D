-- migrate:up
ALTER TABLE pvp_pair_results
ADD COLUMN IF NOT EXISTS gpu_count INT;

CREATE INDEX IF NOT EXISTS idx_pvp_pair_results_gpu_count
ON pvp_pair_results(gpu_count) WHERE gpu_count IS NOT NULL;

-- migrate:down
DROP INDEX IF EXISTS idx_pvp_pair_results_gpu_count;

ALTER TABLE pvp_pair_results
DROP COLUMN IF EXISTS gpu_count;
