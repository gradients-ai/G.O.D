-- migrate:up
CREATE TABLE IF NOT EXISTS pvp_individual_scores (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    hotkey TEXT NOT NULL,
    environment_name TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    n_attempts INT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, hotkey, environment_name)
);

CREATE INDEX idx_pvp_individual_scores_task_status ON pvp_individual_scores(task_id, status);

-- migrate:down
DROP TABLE IF EXISTS pvp_individual_scores;
