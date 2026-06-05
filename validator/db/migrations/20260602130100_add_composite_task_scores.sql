-- migrate:up
CREATE TABLE IF NOT EXISTS composite_task_scores (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    hotkey TEXT NOT NULL,
    subtask_task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    score DOUBLE PRECISION,
    n_attempts INT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, hotkey, subtask_task_id)
);

CREATE INDEX idx_composite_task_scores_task_status ON composite_task_scores(task_id, status);

-- migrate:down
DROP TABLE IF EXISTS composite_task_scores;
