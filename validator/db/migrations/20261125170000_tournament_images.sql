-- migrate:up
-- One trainer image per (tournament, participant, variant), built once when the tournament is active and pulled by every
-- training job of that participant. variant: text, image or image-toolkit (the Dockerfile family). status: pending (to be
-- snapshotted and handed to the build pipeline), building (Woodpecker pipeline build_number running), ready (image_ref is
-- registry/repo@sha256:digest) or failed (error says why; that participant's jobs needing the variant fail).
CREATE TABLE IF NOT EXISTS tournament_images (
    tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id) ON DELETE CASCADE,
    hotkey TEXT NOT NULL,
    variant TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'building', 'ready', 'failed')),
    build_number INTEGER,
    build_attempts INTEGER NOT NULL DEFAULT 0,
    image_ref TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tournament_id, hotkey, variant)
);

CREATE INDEX IF NOT EXISTS idx_tournament_images_status ON tournament_images (status);

-- migrate:down
DROP TABLE IF EXISTS tournament_images;
