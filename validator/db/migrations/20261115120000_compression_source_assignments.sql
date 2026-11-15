-- migrate:up
-- Which text round boss model each compression slot compresses, recorded once per text tournament when its seeds are drawn
-- (validator/tournament/compression_assignment.py).
--   text_tournament_id   the text tournament (one row each).
--   fixed_text_role      the text round whose model compression round 2 (the fixed round) compresses: text_r1, text_r2 or
--                        text_r3. The other two feed compression rounds 1 and 3 (smaller model band first); the boss round's
--                        freeform task compresses the fixed round's model and its fixed task the round-3 model.
--   rng_seed             the seed fixed_text_role was drawn with; NULL for a text tournament seeded before this table, which
--                        keeps text_r2 (the earlier text R1/R2/R3 -> compression R1/R2/R3 mapping).
CREATE TABLE IF NOT EXISTS compression_source_assignments (
    text_tournament_id TEXT PRIMARY KEY REFERENCES tournaments(tournament_id) ON DELETE CASCADE,
    fixed_text_role TEXT NOT NULL CHECK (fixed_text_role IN ('text_r1', 'text_r2', 'text_r3')),
    rng_seed TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- migrate:down
DROP TABLE IF EXISTS compression_source_assignments;
