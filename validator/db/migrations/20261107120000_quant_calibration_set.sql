-- migrate:up
-- Calibration set for freeform quant tasks: rows carved from the train side of the R1 split (never trained on by R1,
-- never scored) that a freeform miner receives for its own use. calibration_data is the raw JSON URL on an
-- instruct task and the tokenised JSONL on a freeform quant task; fingerprint and count are set at quant prep.
-- carve_calibration marks a task whose prep carves the set. Every new column is nullable or defaulted, so existing
-- tasks have no calibration.
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS calibration_data TEXT;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS calibration_fingerprint TEXT;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS calibration_n_examples INTEGER;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS carve_calibration BOOLEAN NOT NULL DEFAULT FALSE;

-- The raw rows already carved into a calibration set, per dataset: a later split of the dataset keeps them out of
-- its test set.
CREATE TABLE IF NOT EXISTS dataset_calibration_rows (
    ds TEXT NOT NULL,
    row_hash TEXT NOT NULL,
    task_id UUID REFERENCES tasks(task_id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ds, row_hash)
);

-- The miner's own bf16 model on the calibration rows, published with them and never used to score. One row per
-- (task, hotkey); first writer wins.
CREATE TABLE IF NOT EXISTS quant_calibration_references (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    hotkey TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    per_example_losses JSONB NOT NULL,
    eval_set_fingerprint TEXT NOT NULL,
    eval_gpu JSONB NOT NULL,
    measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (task_id, hotkey)
);

-- migrate:down
DROP TABLE IF EXISTS quant_calibration_references;
DROP TABLE IF EXISTS dataset_calibration_rows;
ALTER TABLE tasks DROP COLUMN IF EXISTS carve_calibration;
ALTER TABLE tasks DROP COLUMN IF EXISTS calibration_n_examples;
ALTER TABLE tasks DROP COLUMN IF EXISTS calibration_fingerprint;
ALTER TABLE tasks DROP COLUMN IF EXISTS calibration_data;
