-- migrate:up
-- Stores the R2 functional-duplication review gate for a tournament.
-- One row per guarded round (the R2 round_id). While status = 'pending_review' the
-- tournament cycle halts advancement; a human approves/skips by updating the row.
CREATE TABLE IF NOT EXISTS tournament_dedup_reviews (
    round_id TEXT PRIMARY KEY,
    tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id) ON DELETE CASCADE,
    tournament_type TEXT NOT NULL,
    -- pending_review: gate active, advancement halted
    -- approved:       eliminate approved_eliminations, then advance
    -- skipped:        advance with no eliminations
    status TEXT NOT NULL DEFAULT 'pending_review',
    cohort JSONB NOT NULL DEFAULT '[]'::jsonb,                 -- hotkeys evaluated at R2
    clusters JSONB NOT NULL DEFAULT '[]'::jsonb,               -- detected duplicate clusters + reasons
    pair_verdicts JSONB NOT NULL DEFAULT '[]'::jsonb,          -- raw per-pair tier/verdict/reason
    flagged_hotkeys JSONB NOT NULL DEFAULT '[]'::jsonb,        -- Claude's recommended eliminations
    approved_eliminations JSONB NOT NULL DEFAULT '[]'::jsonb,  -- human-editable; defaults to flagged
    published_repos JSONB NOT NULL DEFAULT '[]'::jsonb,        -- public re-uploaded repos of confirmed dupes (for auditing)
    report_url TEXT,                                           -- full human-readable report in object storage
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ                                    -- set once eliminations + re-uploads applied
);

CREATE INDEX idx_tournament_dedup_reviews_tournament ON tournament_dedup_reviews(tournament_id);
CREATE INDEX idx_tournament_dedup_reviews_status ON tournament_dedup_reviews(status);

-- migrate:down
DROP TABLE IF EXISTS tournament_dedup_reviews;
