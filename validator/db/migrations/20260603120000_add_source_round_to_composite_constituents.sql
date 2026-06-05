-- migrate:up
ALTER TABLE composite_task_constituents
    ADD COLUMN source_round INTEGER NOT NULL DEFAULT 1;

-- migrate:down
ALTER TABLE composite_task_constituents
    DROP COLUMN IF EXISTS source_round;
