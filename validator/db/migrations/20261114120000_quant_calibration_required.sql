-- migrate:up
-- A compression task (QuantFreeformTask, QuantFixedTask) that has left data prep and can still be handed to miners always has its published
-- calibration set: the tokenised file (calibration_data), its fingerprint and a positive example count. The five statuses are
-- QUANT_CALIBRATION_REQUIRED_STATUSES in validator/tasks/models.py (a test keeps the two lists equal). Tasks that are pending,
-- awaiting their upstream, in prep, failed, or past training (preevaluation, evaluating, success, failure) are not constrained.
--
-- Existing rows are neither altered nor checked: the constraint is added NOT VALID, so it applies only to rows inserted or
-- updated from now on. An UPDATE of an existing violating row is rejected until the row is given calibration or moved to a
-- finished status (success, failure). The block below lists those rows first; it changes nothing.
DO $$
DECLARE
    violating RECORD;
    n_violating INTEGER := 0;
BEGIN
    FOR violating IN
        SELECT task_id, task_type, status, created_at
        FROM tasks
        WHERE task_type::text IN ('QuantFreeformTask', 'QuantFixedTask')
          AND status IN ('awaiting_model_prep', 'looking_for_nodes', 'delayed', 'ready', 'training')
          AND NOT (calibration_data IS NOT NULL AND calibration_fingerprint IS NOT NULL AND COALESCE(calibration_n_examples, 0) > 0)
        ORDER BY created_at
    LOOP
        n_violating := n_violating + 1;
        RAISE NOTICE 'tasks_quant_calibration_required: existing row without calibration: task_id=% task_type=% status=% created_at=%',
            violating.task_id, violating.task_type, violating.status, violating.created_at;
    END LOOP;
    RAISE NOTICE 'tasks_quant_calibration_required: % existing row(s) violate the constraint and are left as they are', n_violating;
END
$$;

ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_quant_calibration_required;
ALTER TABLE tasks ADD CONSTRAINT tasks_quant_calibration_required CHECK (
    task_type::text NOT IN ('QuantFreeformTask', 'QuantFixedTask')
    OR status NOT IN ('awaiting_model_prep', 'looking_for_nodes', 'delayed', 'ready', 'training')
    OR (calibration_data IS NOT NULL AND calibration_fingerprint IS NOT NULL AND COALESCE(calibration_n_examples, 0) > 0)
) NOT VALID;

-- migrate:down
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_quant_calibration_required;
