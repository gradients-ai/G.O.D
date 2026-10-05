-- migrate:up
-- Per-entrant round-2 cheat-check status. The tournament-level code_review column
-- was the old boss-round gate and is no longer written by the validator.
ALTER TABLE tournament_participants ADD COLUMN IF NOT EXISTS code_review VARCHAR(20);

-- migrate:down
ALTER TABLE tournament_participants DROP COLUMN IF EXISTS code_review;
