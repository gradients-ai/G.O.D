-- migrate:up
-- Prepared tournament training images (validator/tournament/image_builds). One row per
-- (tournament, hotkey, variant, repo, commit): the image is built once on an idle trainer or fallback builder,
-- stored in the R2 image registry, and every training of that miner in the tournament pulls it by digest.
-- A row that ends 'failed' makes training fall back to today's on-trainer build, never a miner failure.
-- New table names on purpose: the unmerged Depot branch used image_builds/image_build_attempts.
CREATE TABLE IF NOT EXISTS tournament_image_builds (
    build_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id) ON DELETE CASCADE,
    hotkey TEXT NOT NULL,
    variant TEXT NOT NULL CHECK (variant IN ('text', 'image-flux', 'image-toolkit')),
    repo TEXT NOT NULL,
    -- Commit sha requested, or 'HEAD' for backup repos that carry no commit.
    commit_ref TEXT NOT NULL,
    repository TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'building', 'ready', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    builder TEXT,
    lease_expires_at TIMESTAMPTZ,
    digest TEXT CHECK (digest IS NULL OR digest ~ '^sha256:[0-9a-f]{64}$'),
    resolved_commit TEXT,
    dockerfile_path TEXT,
    size_bytes BIGINT,
    failure_class TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ready_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,
    pruned_at TIMESTAMPTZ,
    UNIQUE (tournament_id, hotkey, variant, repo, commit_ref),
    CHECK (status <> 'ready' OR digest IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS tournament_image_builds_claim_idx
    ON tournament_image_builds (next_attempt_at, created_at)
    WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS tournament_image_builds_building_idx
    ON tournament_image_builds (lease_expires_at)
    WHERE status = 'building';
CREATE INDEX IF NOT EXISTS tournament_image_builds_tournament_idx
    ON tournament_image_builds (tournament_id, status);

CREATE TABLE IF NOT EXISTS tournament_image_build_attempts (
    build_id UUID NOT NULL REFERENCES tournament_image_builds(build_id) ON DELETE CASCADE,
    attempt INTEGER NOT NULL,
    builder TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    outcome TEXT CHECK (outcome IS NULL OR outcome IN ('ready', 'failed')),
    failure_class TEXT,
    error_message TEXT,
    PRIMARY KEY (build_id, attempt)
);

-- migrate:down
DROP TABLE IF EXISTS tournament_image_build_attempts;
DROP TABLE IF EXISTS tournament_image_builds;
