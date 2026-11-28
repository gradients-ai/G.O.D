-- migrate:up
-- Trainer images. Trainers never build: every training job runs an image built here first. One row per (scope, source,
-- variant): scope is the tournament id (or 'untracked' for a training row outside a tournament), source identifies the
-- code (sha256 of repo and commit, first 16 hex digits), variant the Dockerfile family (text, image, image-toolkit).
-- Participants' rows are added when the tournament becomes active; any other training row adds its own when first
-- scheduled. status: pending (to be snapshotted and handed to the build pipeline), building (Woodpecker pipeline
-- build_number running), ready (image_ref is registry/repo@sha256:digest) or failed (error says why; the training rows
-- needing it fail). github_token is cleared once the snapshot is taken.
CREATE TABLE IF NOT EXISTS tournament_images (
    scope TEXT NOT NULL,
    source TEXT NOT NULL,
    variant TEXT NOT NULL,
    repo TEXT NOT NULL,
    commit_hash TEXT,
    github_token TEXT,
    hotkey TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'building', 'ready', 'failed')),
    build_number INTEGER,
    build_attempts INTEGER NOT NULL DEFAULT 0,
    image_ref TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scope, source, variant)
);

CREATE INDEX IF NOT EXISTS idx_tournament_images_status ON tournament_images (status);

-- migrate:down
DROP TABLE IF EXISTS tournament_images;
