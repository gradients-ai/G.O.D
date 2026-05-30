-- migrate:up

CREATE INDEX IF NOT EXISTS idx_evaluations_netuid_status_task_id
    ON evaluations(netuid, evaluation_status, task_id);

CREATE INDEX IF NOT EXISTS idx_evaluations_task_id_netuid_status
    ON evaluations(task_id, netuid, evaluation_status);

CREATE INDEX IF NOT EXISTS idx_tournament_task_hotkey_trainings_task_status
    ON tournament_task_hotkey_trainings(task_id, training_status);

-- migrate:down

DROP INDEX IF EXISTS idx_tournament_task_hotkey_trainings_task_status;
DROP INDEX IF EXISTS idx_evaluations_task_id_netuid_status;
DROP INDEX IF EXISTS idx_evaluations_netuid_status_task_id;
