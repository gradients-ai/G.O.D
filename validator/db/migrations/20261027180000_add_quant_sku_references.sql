-- migrate:up
-- The bf16 reference every quant loss on a task is normalised by, measured once per (task, SKU) in the same
-- deployment as a submission. sku is QuantEvalGpu.reference_key: the SKU label ('H100 SXM', 'H100 PCIe'), or
-- 'override:<name>/<sm_count>' for a test-box override. model is the repo that was scored (the common input,
-- or the R3 reference entrant's repo). The first writer wins; a later measurement on the same SKU is compared
-- with this row, never written over it.
CREATE TABLE IF NOT EXISTS quant_sku_references (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    sku TEXT NOT NULL,
    model TEXT NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    per_example_losses JSONB NOT NULL,
    eval_set_fingerprint TEXT NOT NULL,
    eval_gpu JSONB NOT NULL,
    measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (task_id, sku)
);

-- The single common-input reference it replaces, kept where it recorded its SKU.
INSERT INTO quant_sku_references (task_id, sku, model, loss, per_example_losses, eval_set_fingerprint, eval_gpu)
SELECT qt.task_id, qt.reference_eval_gpu->>'sku', COALESCE(t.augmented_model_id, t.model_id), qt.reference_loss,
       qt.reference_per_example_losses, qt.reference_fingerprint, qt.reference_eval_gpu
FROM quant_tasks qt
JOIN tasks t ON t.task_id = qt.task_id
WHERE qt.reference_loss IS NOT NULL
  AND qt.reference_per_example_losses IS NOT NULL
  AND qt.reference_fingerprint IS NOT NULL
  AND qt.reference_eval_gpu->>'sku' IS NOT NULL
ON CONFLICT (task_id, sku) DO NOTHING;

ALTER TABLE quant_tasks
    DROP COLUMN IF EXISTS reference_loss,
    DROP COLUMN IF EXISTS reference_per_example_losses,
    DROP COLUMN IF EXISTS reference_fingerprint,
    DROP COLUMN IF EXISTS reference_eval_gpu;

-- migrate:down
ALTER TABLE quant_tasks
    ADD COLUMN IF NOT EXISTS reference_loss DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS reference_per_example_losses JSONB,
    ADD COLUMN IF NOT EXISTS reference_fingerprint TEXT,
    ADD COLUMN IF NOT EXISTS reference_eval_gpu JSONB;

-- The earliest measurement of the common input per task (an R3 task kept none).
UPDATE quant_tasks qt
SET reference_loss = r.loss,
    reference_per_example_losses = r.per_example_losses,
    reference_fingerprint = r.eval_set_fingerprint,
    reference_eval_gpu = r.eval_gpu
FROM (
    SELECT DISTINCT ON (task_id) task_id, loss, per_example_losses, eval_set_fingerprint, eval_gpu
    FROM quant_sku_references
    ORDER BY task_id, measured_at
) r
WHERE qt.task_id = r.task_id AND qt.reference_hotkey IS NULL;

DROP TABLE IF EXISTS quant_sku_references;
