-- migrate:up

CREATE TABLE IF NOT EXISTS evaluations (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    hotkey TEXT NOT NULL,
    expected_repo_name TEXT,
    evaluation_status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, hotkey)
);

CREATE INDEX IF NOT EXISTS idx_evaluations_status ON evaluations(evaluation_status);
CREATE INDEX IF NOT EXISTS idx_evaluations_task_id ON evaluations(task_id);

-- migrate:down

DROP TABLE IF EXISTS evaluations;
