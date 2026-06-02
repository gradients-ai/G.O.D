--------------------------------
-- GET 0-SCORE TASKS SECTION
--------------------------------

WITH tasks_last_12h AS (
    SELECT *
    FROM tasks
    WHERE termination_at >= NOW() - INTERVAL '12 HOURS'
),

zero_score_tasks AS (
    SELECT DISTINCT t.task_id
    FROM tasks_last_12h t
    JOIN task_nodes tn ON t.task_id = tn.task_id
    WHERE t.n_eval_attempts = 4 -- MAX_EVAL_ATTEMPTS in validator/scoring/constants.py
    GROUP BY t.task_id
    HAVING MAX(tn.quality_score) = 0 AND MIN(tn.quality_score) = 0
),

-- Select miners submissions
tasks_submissions_last_12h AS (
SELECT tn.task_id, tn.hotkey, tn.quality_score, s.repo
FROM task_nodes tn
LEFT JOIN submissions s ON tn.task_id = s.task_id AND tn.hotkey = s.hotkey
WHERE tn.task_id IN (SELECT task_id FROM zero_score_tasks)
)

-- Basic stats
-- SELECT COUNT(*) as total_tasks_12h
--     , (SELECT COUNT(*) FROM zero_score_tasks) as zero_score_tasks_count
--     , ROUND(
--         (SELECT COUNT(*) FROM zero_score_tasks)::numeric /
--         NULLIF(COUNT(*), 0) * 100::numeric
--     , 2) as zero_score_percentage
-- FROM tasks_last_12h;

-- Detailed results in regular table format
-- SELECT t.*
-- FROM tasks_last_12h t
-- JOIN zero_score_tasks zst ON t.task_id = zst.task_id
-- ORDER BY t.created_at DESC;

-- Selected fields in regular table format
-- SELECT
--     t.task_id,
--     t.model_id,
--     t.created_at,
--     t.ds,
--     t.status,
--     t.test_data,
--     t.training_data,
--     t.is_organic
-- FROM tasks_last_12h t
-- JOIN zero_score_tasks zst ON t.task_id = zst.task_id
-- ORDER BY t.created_at DESC;

-- Join with tasks
SELECT ts.*, t.model_id, t.ds, t.status, t.test_data, t.training_data, t.is_organic, t.trained_model_repository
FROM tasks_submissions_last_12h ts
JOIN tasks t ON ts.task_id = t.task_id
ORDER BY ts.task_id;

--------------------------------
-- GET GENERAL INSIGHTS SECTION
--------------------------------

-- Count tasks for all possible statuses
SELECT
    status,
    COUNT(*) AS count
FROM tasks
WHERE status IN (
    'pending',
    'preparing_data',
    'idle',
    'ready',
    'success',
    'looking_for_nodes',
    'delayed',
    'evaluating',
    'preevaluation',
    'training',
    'failure',
    'failure_finding_nodes',
    'prep_task_failure',
    'node_training_failure'
)
GROUP BY status
ORDER BY count DESC;

-- Get most recent tasks
SELECT * FROM tasks
-- WHERE status in ('training', 'success')
-- WHERE status = 'training'
ORDER BY created_at DESC;

-- Get most recent tasks, specific fields
SELECT task_id, created_at, is_organic, status, task_type, n_eval_attempts, termination_at, * FROM tasks
ORDER BY created_at DESC;

--------------------------------
-- GET GRANULAR INSIGHTS SECTION
--------------------------------

-- Get full task details
SELECT t.*, tt.*, it.*
FROM tasks t
LEFT JOIN instruct_text_tasks tt ON t.task_id = tt.task_id
LEFT JOIN image_tasks it ON t.task_id = it.task_id
WHERE t.task_id IN (
    'TASK_UUID'
);

-- Get task breakdown
SELECT DISTINCT s.repo, tn.*, t.*
FROM tasks t
JOIN task_nodes tn ON t.task_id = tn.task_id
LEFT JOIN submissions s ON tn.task_id = s.task_id AND tn.hotkey = s.hotkey
WHERE t.task_id = 'TASK_UUID'
--   AND s.repo LIKE '%SOME_REPO%'
ORDER BY tn.quality_score DESC;

-- Get task breakdown to reproduce evaluation locally
WITH merged_tasks AS (
    SELECT t.*, tt.field_system, tt.field_instruction, tt.field_input, tt.field_output, tt.synthetic_data, tt.format, tt.no_input_format, tt.system_format, tt.file_format
    FROM tasks t
    LEFT JOIN instruct_text_tasks tt ON t.task_id = tt.task_id
    LEFT JOIN image_tasks it ON t.task_id = it.task_id
)
SELECT DISTINCT mt.task_id, s.repo, tn.test_loss,tn.synth_loss, mt.model_id, mt.ds, mt.field_instruction, mt.field_input, mt.field_output, mt.test_data, mt.synthetic_data, mt.training_data, mt.status, mt.termination_at, mt.n_eval_attempts, mt.is_organic, tn.quality_score, tn.hotkey
FROM merged_tasks mt
JOIN task_nodes tn ON mt.task_id = tn.task_id
LEFT JOIN submissions s ON mt.task_id = s.task_id AND tn.hotkey = s.hotkey
WHERE mt.task_id = 'TASK_UUID'
--   AND s.repo LIKE '%SOME_REPO%'
ORDER BY tn.quality_score DESC;

--------------------------------
-- UPDATE QUERIES SECTION
--------------------------------

-- Move finished training tasks to evaluation
UPDATE tasks SET termination_at = NOW() WHERE status = 'training' AND task_id IN ('TASK_UUID');

-- Re-evaluate tasks once more
UPDATE tasks SET status = 'preevaluation' WHERE task_id IN ('TASK_UUID');

-- Move all tasks in training to failure status
UPDATE tasks
SET status = 'failure'
WHERE status IN ('training', 'preevaluation', 'looking_for_nodes');

-- Only eval the task in last line, others wait
UPDATE tasks
SET
    status = 'training',
    termination_at = NOW() + INTERVAL '1 hour'
WHERE
    status = 'preevaluation'
    AND task_id != 'TASK_UUID';
