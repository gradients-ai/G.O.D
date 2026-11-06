-- migrate:up
CREATE TABLE image_build_batches (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tournament_id TEXT NOT NULL UNIQUE REFERENCES tournaments(tournament_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cleaned_at TIMESTAMPTZ
);

CREATE TABLE image_builds (
    build_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL REFERENCES image_build_batches(batch_id) ON DELETE CASCADE,
    tournament_id TEXT NOT NULL,
    hotkey TEXT NOT NULL,
    variant TEXT NOT NULL CHECK (variant IN ('text', 'image-flux', 'image-toolkit')),
    stable_build_id TEXT NOT NULL,
    repo TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    dockerfile_path TEXT NOT NULL,
    build_context TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'linux/amd64',
    depot_project_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'building', 'publishing', 'ready', 'retry_wait', 'failed')),
    current_attempt INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT,
    pull_ref TEXT,
    digest TEXT,
    failure_class TEXT,
    log_ref TEXT,
    error_message TEXT,
    dequeued_at TIMESTAMPTZ,
    cli_started_at TIMESTAMPTZ,
    first_progress_at TIMESTAMPTZ,
    build_finished_at TIMESTAMPTZ,
    publication_confirmed_at TIMESTAMPTZ,
    cancel_requested_at TIMESTAMPTZ,
    retry_after_at TIMESTAMPTZ,
    ready_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (tournament_id, hotkey)
        REFERENCES tournament_participants(tournament_id, hotkey) ON DELETE CASCADE,
    UNIQUE (tournament_id, stable_build_id),
    UNIQUE (tournament_id, hotkey, variant)
);

CREATE TABLE image_build_attempts (
    attempt_id BIGSERIAL PRIMARY KEY,
    build_id UUID NOT NULL REFERENCES image_builds(build_id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    depot_build_id TEXT,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    first_progress_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    duration_seconds DOUBLE PRECISION,
    log_ref TEXT,
    error_message TEXT,
    failure_class TEXT,
    UNIQUE (build_id, attempt_number)
);

CREATE INDEX image_builds_claim_idx
    ON image_builds(status, retry_after_at, created_at)
    WHERE status IN ('queued', 'retry_wait');
CREATE INDEX image_builds_tournament_status_idx ON image_builds(tournament_id, status);
CREATE INDEX image_build_attempts_build_idx ON image_build_attempts(build_id, attempt_number);

ALTER TABLE tournament_task_hotkey_trainings
    ADD COLUMN image_build_variant TEXT,
    ADD COLUMN image_build_id UUID REFERENCES image_builds(build_id),
    ADD COLUMN image_pull_ref TEXT,
    ADD COLUMN image_digest TEXT;

-- migrate:down
ALTER TABLE tournament_task_hotkey_trainings
    DROP COLUMN IF EXISTS image_digest,
    DROP COLUMN IF EXISTS image_pull_ref,
    DROP COLUMN IF EXISTS image_build_id,
    DROP COLUMN IF EXISTS image_build_variant;
DROP TABLE IF EXISTS image_build_attempts;
DROP TABLE IF EXISTS image_builds;
DROP TABLE IF EXISTS image_build_batches;
