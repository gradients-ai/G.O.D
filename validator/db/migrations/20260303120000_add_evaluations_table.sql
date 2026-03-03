-- migrate:up

CREATE TABLE IF NOT EXISTS evaluations (
    task_id UUID NOT NULL,
    hotkey TEXT NOT NULL,

    evaluation_status TEXT NOT NULL DEFAULT 'pending',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (task_id, hotkey),

    CONSTRAINT fk_evaluations_task_node
        FOREIGN KEY (task_id, hotkey)
        REFERENCES task_nodes(task_id, hotkey)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evaluations_status
    ON evaluations(evaluation_status);

CREATE INDEX IF NOT EXISTS idx_evaluations_task_id
    ON evaluations(task_id);

-- migrate:down

DROP TABLE IF EXISTS evaluations;