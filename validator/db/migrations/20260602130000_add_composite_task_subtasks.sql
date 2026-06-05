-- migrate:up
CREATE TABLE IF NOT EXISTS composite_task_subtasks (
    composite_task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    subtask_task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    position INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (composite_task_id, subtask_task_id)
);

CREATE INDEX idx_composite_task_subtasks_composite ON composite_task_subtasks(composite_task_id);

-- migrate:down
DROP TABLE IF EXISTS composite_task_subtasks;
