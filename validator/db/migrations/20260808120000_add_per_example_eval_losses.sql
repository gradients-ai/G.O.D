-- migrate:up

-- Per-example held-out losses, for the paired boss-round comparison (see compare_paired_losses in
-- validator/tournament/thresholds.py). A scalar mean carries no information about its own
-- uncertainty, so deciding a boss-round task on it cannot separate a real win from held-out
-- sampling noise; the vector lets boss and challenger be compared example by example.
--
-- Populated only for final-round tournament tasks and NULL everywhere else. Every other consumer
-- ranks on task_nodes.test_loss, which is unchanged, and writing a vector for every miner on every
-- task would bloat the table for no gain.
--
-- eval_set_fingerprint identifies the held-out set the vector was produced against. Pairing by
-- index is only valid if both models saw the same examples in the same order; a mismatch here
-- means the comparison must be refused rather than silently producing a wrong verdict.
ALTER TABLE task_nodes
    ADD COLUMN IF NOT EXISTS per_example_losses JSONB,
    ADD COLUMN IF NOT EXISTS eval_set_fingerprint TEXT;

-- migrate:down

ALTER TABLE task_nodes
    DROP COLUMN IF EXISTS per_example_losses,
    DROP COLUMN IF EXISTS eval_set_fingerprint;
