-- migrate:up
DROP TABLE IF EXISTS pvp_pair_results;
CREATE TABLE pvp_pair_results (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    hotkey_a TEXT NOT NULL,
    hotkey_b TEXT NOT NULL,
    environment_name TEXT NOT NULL,
    model_a_wins INT NOT NULL DEFAULT 0,
    model_b_wins INT NOT NULL DEFAULT 0,
    draws INT NOT NULL DEFAULT 0,
    total_games INT NOT NULL DEFAULT 0,
    n_attempts INT NOT NULL DEFAULT 0,
    deployment_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, hotkey_a, hotkey_b, environment_name)
);

CREATE INDEX idx_pvp_pair_results_task_status ON pvp_pair_results(task_id, status);
CREATE INDEX idx_pvp_pair_results_deployment_id ON pvp_pair_results(deployment_id);

-- migrate:down
DROP TABLE IF EXISTS pvp_pair_results;
