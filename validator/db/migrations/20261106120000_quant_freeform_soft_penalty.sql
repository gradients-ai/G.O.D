-- migrate:up
-- Freeform quant entries are ranked on effective size, total bytes x exp(penalty_curvature x
-- max(0, r - (1 + penalty_knee))^2) with r the served loss over the same-SKU bf16 reference, and are out
-- above (1 + bar_cap). Both constants are pinned per task at creation like bar_cap; a NULL pair
-- ranks under the cliff at bar_cap. task_nodes.freeform_penalty is
-- ln P as scored, so the tournament can re-rank a task on effective size from its rows alone.
ALTER TABLE quant_tasks ADD COLUMN IF NOT EXISTS penalty_knee DOUBLE PRECISION;
ALTER TABLE quant_tasks ADD COLUMN IF NOT EXISTS penalty_curvature DOUBLE PRECISION;
ALTER TABLE task_nodes ADD COLUMN IF NOT EXISTS freeform_penalty DOUBLE PRECISION;

-- migrate:down
ALTER TABLE task_nodes DROP COLUMN IF EXISTS freeform_penalty;
ALTER TABLE quant_tasks DROP COLUMN IF EXISTS penalty_curvature;
ALTER TABLE quant_tasks DROP COLUMN IF EXISTS penalty_knee;
