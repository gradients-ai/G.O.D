-- migrate:up
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type t
        JOIN pg_enum e ON t.oid = e.enumtypid
        WHERE t.typname = 'tasktype' AND e.enumlabel = 'EnvTask'
    ) THEN
        ALTER TYPE tasktype ADD VALUE 'EnvTask';
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS env_tasks (
    task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    environment_name TEXT
);

-- migrate:down

DROP TABLE IF EXISTS env_tasks;

DELETE FROM tasks
  WHERE task_type = 'EnvTask';

ALTER TYPE tasktype RENAME TO tasktype_temp;
CREATE TYPE tasktype AS ENUM ('InstructTextTask', 'ImageTask', 'DpoTask', 'GrpoTask', 'ChatTask');

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE VARCHAR;

ALTER TABLE tasks
  ALTER COLUMN task_type TYPE tasktype USING task_type::tasktype;

DROP TYPE tasktype_temp;

