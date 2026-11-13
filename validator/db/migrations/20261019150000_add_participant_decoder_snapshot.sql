-- migrate:up
-- The decoder counted for a participant is the snapshot of decoder/ at their registered commit,
-- taken once at registration and stored in MinIO under decoder_object_name. decoder_sha256 is the
-- sha256 of that .tar.xz (what the trainer verifies before running it) and decoder_xz_bytes is its
-- size, the decoder bytes counted toward a freeform entry. NULL for image/environment participants
-- and for a boss whose repo had no decoder/ at its pinned commit.
ALTER TABLE tournament_participants
    ADD COLUMN IF NOT EXISTS decoder_object_name TEXT,
    ADD COLUMN IF NOT EXISTS decoder_xz_bytes BIGINT,
    ADD COLUMN IF NOT EXISTS decoder_sha256 TEXT;

-- migrate:down
ALTER TABLE tournament_participants
    DROP COLUMN IF EXISTS decoder_object_name,
    DROP COLUMN IF EXISTS decoder_xz_bytes,
    DROP COLUMN IF EXISTS decoder_sha256;
