-- migrate:up
-- Idempotency key for continuous-SFT carry-forward: the round that last advanced each lineage, so a
-- crash/restart that reprocesses the round can't advance train_index twice and skip a weekly chunk.
ALTER TABLE continuous_sft_state ADD COLUMN IF NOT EXISTS last_source_round_id TEXT;

-- migrate:down
ALTER TABLE continuous_sft_state DROP COLUMN IF EXISTS last_source_round_id;
