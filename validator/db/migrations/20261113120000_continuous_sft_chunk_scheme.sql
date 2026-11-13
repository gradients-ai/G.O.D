-- migrate:up
-- This migration IS the continuous-SFT chunk scheme switch on the G.O.D side. Apply it (i.e. deploy
-- this change) in the same window, between two tournaments, in which the content service switches
-- scheme. Applied earlier, the doubled indices would read chunks of the previous scheme; applied later,
-- one week would re-read part of an earlier chunk.
--
-- Chunk k of the previous scheme is chunks 2k and 2k+1 of the new one, so every lineage cursor doubles once. The WHERE makes
-- a second application a no-op.
ALTER TABLE continuous_sft_state
    ADD COLUMN IF NOT EXISTS chunk_scheme TEXT NOT NULL DEFAULT 'v1' CHECK (chunk_scheme IN ('v1', 'v2'));

UPDATE continuous_sft_state SET train_index = train_index * 2, chunk_scheme = 'v2' WHERE chunk_scheme = 'v1';

ALTER TABLE continuous_sft_state ALTER COLUMN chunk_scheme SET DEFAULT 'v2';

-- One row per continuous-SFT task: the cursor value and chunk scheme the task was created with.
-- A continuous-SFT task with no row was created before this migration, on the previous scheme.
CREATE TABLE IF NOT EXISTS continuous_sft_task_chunks (
    task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    lineage TEXT NOT NULL,
    train_index INT NOT NULL,
    chunk_scheme TEXT NOT NULL CHECK (chunk_scheme IN ('v1', 'v2')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- migrate:down
DROP TABLE IF EXISTS continuous_sft_task_chunks;
UPDATE continuous_sft_state SET train_index = train_index / 2 WHERE chunk_scheme = 'v2';
ALTER TABLE continuous_sft_state DROP COLUMN IF EXISTS chunk_scheme;
