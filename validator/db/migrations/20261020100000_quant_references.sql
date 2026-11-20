-- migrate:up
-- reference_per_example_losses and reference_fingerprint complete the common input's bf16 reference
-- (reference_loss), measured through the quant scorer on the task's tokenised test set before finalize.
-- upstream_task_id is the text seed task whose boss output this chained task compresses;
-- written by chaining, NULL otherwise.
ALTER TABLE quant_tasks
    ADD COLUMN IF NOT EXISTS reference_per_example_losses JSONB,
    ADD COLUMN IF NOT EXISTS reference_fingerprint TEXT,
    ADD COLUMN IF NOT EXISTS upstream_task_id UUID NULL REFERENCES tasks(task_id) ON DELETE SET NULL;

-- migrate:down
ALTER TABLE quant_tasks
    DROP COLUMN IF EXISTS reference_per_example_losses,
    DROP COLUMN IF EXISTS reference_fingerprint,
    DROP COLUMN IF EXISTS upstream_task_id;
