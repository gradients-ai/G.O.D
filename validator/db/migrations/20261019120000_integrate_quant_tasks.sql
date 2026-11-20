-- migrate:up
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type t
        JOIN pg_enum e ON t.oid = e.enumtypid
        WHERE t.typname = 'tasktype' AND e.enumlabel = 'QuantFixedTask'
    ) THEN
        ALTER TYPE tasktype ADD VALUE 'QuantFixedTask';
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type t
        JOIN pg_enum e ON t.oid = e.enumtypid
        WHERE t.typname = 'tasktype' AND e.enumlabel = 'QuantFreeformTask'
    ) THEN
        ALTER TYPE tasktype ADD VALUE 'QuantFreeformTask';
    END IF;
END$$;

-- One row per quant task, for both tracks. quant_spec is set only for QuantFixedTask; bar_cap
-- only for QuantFreeformTask, pinned at creation so re-tuning the constant never moves an old bar.
CREATE TABLE IF NOT EXISTS quant_tasks (
    task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    dataset_type JSONB NOT NULL,
    file_format TEXT NOT NULL DEFAULT 'hf',
    quant_spec JSONB,
    bar_cap DOUBLE PRECISION,
    reference_loss DOUBLE PRECISION,
    reference_hotkey TEXT,
    synthetic_data TEXT
);

-- Per-entrant quant results. NULL for every non-quant task.
--
-- artifact_bytes and decoder_bytes are the two counted parts of an entry's size: the artifact on
-- HF and the decoder snapshot (xz -9). Freeform ranks passing entries on their sum.
-- decoder_sha256 identifies the exact decoder snapshot that was counted and run.
--
-- passed_bar is the freeform bar verdict: loss at or under (1 + bar_cap) x the uncompressed
-- reference loss. NULL on the fixed track, which has no bar.
--
-- quant_failure is the reason an entry was scored as a fail: it did not load, hit a kernel mismatch,
-- produced bad decoder output or timed out decoding. A bar fail is passed_bar = false with
-- quant_failure NULL. It is kept separate from score_reason so a scored fail is not mistaken for an error.
ALTER TABLE task_nodes
    ADD COLUMN IF NOT EXISTS artifact_bytes BIGINT,
    ADD COLUMN IF NOT EXISTS decoder_bytes BIGINT,
    ADD COLUMN IF NOT EXISTS decoder_sha256 TEXT,
    ADD COLUMN IF NOT EXISTS passed_bar BOOLEAN,
    ADD COLUMN IF NOT EXISTS quant_failure TEXT;

-- migrate:down

ALTER TABLE task_nodes
    DROP COLUMN IF EXISTS artifact_bytes,
    DROP COLUMN IF EXISTS decoder_bytes,
    DROP COLUMN IF EXISTS decoder_sha256,
    DROP COLUMN IF EXISTS passed_bar,
    DROP COLUMN IF EXISTS quant_failure;

DROP TABLE IF EXISTS quant_tasks;

DELETE FROM tasks
  WHERE task_type IN ('QuantFixedTask', 'QuantFreeformTask');

ALTER TYPE tasktype RENAME TO tasktype_temp;
CREATE TYPE tasktype AS ENUM ('InstructTextTask', 'ImageTask', 'DpoTask', 'GrpoTask', 'ChatTask', 'EnvTask');

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE VARCHAR;

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE tasktype USING task_type::tasktype;

DROP TYPE tasktype_temp;
