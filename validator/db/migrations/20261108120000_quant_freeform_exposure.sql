-- migrate:up
-- Freeform exposure: what the scorecard shows for a freeform submission. task_nodes keeps the hub file listing read at
-- scoring (artifact_files, artifact_revision), the files the trainer's decode downloaded and produced
-- (decode_artifact_files, decoded_files) and the public HF copy of the decoded model (decoded_hf_repo). The participant
-- keeps the decoder snapshot hashes (tar sha256, git tree id of decoder/, xz version) and, once the champion's decoder
-- is released at handover, its public URL and time. Every column is nullable: existing rows have no scorecard data.
ALTER TABLE task_nodes ADD COLUMN IF NOT EXISTS artifact_files JSONB;
ALTER TABLE task_nodes ADD COLUMN IF NOT EXISTS artifact_revision TEXT;
ALTER TABLE task_nodes ADD COLUMN IF NOT EXISTS decode_artifact_files JSONB;
ALTER TABLE task_nodes ADD COLUMN IF NOT EXISTS decoded_files JSONB;
ALTER TABLE task_nodes ADD COLUMN IF NOT EXISTS decoded_hf_repo TEXT;

ALTER TABLE tournament_participants ADD COLUMN IF NOT EXISTS decoder_tar_sha256 TEXT;
ALTER TABLE tournament_participants ADD COLUMN IF NOT EXISTS decoder_tree_sha TEXT;
ALTER TABLE tournament_participants ADD COLUMN IF NOT EXISTS decoder_xz_version TEXT;
ALTER TABLE tournament_participants ADD COLUMN IF NOT EXISTS decoder_released_url TEXT;
ALTER TABLE tournament_participants ADD COLUMN IF NOT EXISTS decoder_released_at TIMESTAMPTZ;

-- migrate:down
ALTER TABLE tournament_participants DROP COLUMN IF EXISTS decoder_released_at;
ALTER TABLE tournament_participants DROP COLUMN IF EXISTS decoder_released_url;
ALTER TABLE tournament_participants DROP COLUMN IF EXISTS decoder_xz_version;
ALTER TABLE tournament_participants DROP COLUMN IF EXISTS decoder_tree_sha;
ALTER TABLE tournament_participants DROP COLUMN IF EXISTS decoder_tar_sha256;
ALTER TABLE task_nodes DROP COLUMN IF EXISTS decoded_hf_repo;
ALTER TABLE task_nodes DROP COLUMN IF EXISTS decoded_files;
ALTER TABLE task_nodes DROP COLUMN IF EXISTS decode_artifact_files;
ALTER TABLE task_nodes DROP COLUMN IF EXISTS artifact_revision;
ALTER TABLE task_nodes DROP COLUMN IF EXISTS artifact_files;
