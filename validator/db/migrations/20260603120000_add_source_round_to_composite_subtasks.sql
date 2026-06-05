-- migrate:up
ALTER TABLE composite_task_subtasks
    ADD COLUMN source_round INTEGER NOT NULL DEFAULT 1;

-- migrate:down
ALTER TABLE composite_task_subtasks
    DROP COLUMN IF EXISTS source_round;
