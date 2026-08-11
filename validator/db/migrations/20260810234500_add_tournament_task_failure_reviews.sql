-- migrate:up
-- Stores the majority-training-failure review gate for a tournament task.
-- One row per guarded task. While status = 'pending_review' the task is not treated as
-- complete, so the round does not advance; a human approves by updating the row.
--
-- Before this gate existed, a task whose trainings were >50% failed could never complete:
-- the failures are terminal so the ratio never improved, the round wedged forever, and the
-- Discord warning re-fired every cycle. Now the block is explicit and a human can clear it.
CREATE TABLE IF NOT EXISTS tournament_task_failure_reviews (
    task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id) ON DELETE CASCADE,
    round_id TEXT NOT NULL,
    -- pending_review: gate active, task not treated as complete
    -- approved:       failures accepted as legitimate, task completes and the round advances
    status TEXT NOT NULL DEFAULT 'pending_review',
    failed_hotkeys JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_trainings INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_tournament_task_failure_reviews_tournament
    ON tournament_task_failure_reviews(tournament_id);
CREATE INDEX IF NOT EXISTS idx_tournament_task_failure_reviews_status
    ON tournament_task_failure_reviews(status);

-- migrate:down
DROP TABLE IF EXISTS tournament_task_failure_reviews;
