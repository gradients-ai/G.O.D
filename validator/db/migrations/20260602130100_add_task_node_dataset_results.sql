-- migrate:up
CREATE TABLE IF NOT EXISTS task_node_dataset_results (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    hotkey TEXT NOT NULL,
    dataset_task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    score DOUBLE PRECISION,
    n_attempts INT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, hotkey, dataset_task_id)
);

CREATE INDEX idx_task_node_dataset_results_task_status ON task_node_dataset_results(task_id, status);

-- migrate:down
DROP TABLE IF EXISTS task_node_dataset_results;
