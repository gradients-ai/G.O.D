-- migrate:up
-- The GPU each quant eval ran on, as QuantEvalGpu JSON ({name, sm_count, sku, override}). Quant evals
-- may run on either allowed H100 SKU (SXM or PCIe), so every scored row and the common input's bf16
-- reference record theirs. NULL for non-quant rows and for rows with no recorded GPU.
-- decoder_build_warning: the trainer's note that a freeform entry's build.sh exited 0 but left /build empty
-- or logged a failure. Informational, never a scored fail.
ALTER TABLE task_nodes
    ADD COLUMN IF NOT EXISTS eval_gpu JSONB,
    ADD COLUMN IF NOT EXISTS decoder_build_warning TEXT;
ALTER TABLE quant_tasks
    ADD COLUMN IF NOT EXISTS reference_eval_gpu JSONB;

-- migrate:down
ALTER TABLE task_nodes
    DROP COLUMN IF EXISTS eval_gpu,
    DROP COLUMN IF EXISTS decoder_build_warning;
ALTER TABLE quant_tasks
    DROP COLUMN IF EXISTS reference_eval_gpu;
