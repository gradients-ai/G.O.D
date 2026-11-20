-- migrate:up
-- A quant tournament takes its tasks from a text tournament: each quant task is a new task of its own that compresses the
-- model the text boss trained on one of the text tournament's seed tasks. These two tables are the persisted link.
--
-- quant_tournament_sources: which text tournament feeds which quant tournament.
--   quant_tournament_id    the quant tournament (one row each).
--   source_tournament_id   the text tournament it takes its tasks from; UNIQUE, so one text tournament feeds at most one
--                          quant tournament.
--
-- quant_task_sources: one row per quant task of such a tournament.
--   quant_task_id          the quant task (its own tasks row; deleting the task deletes the row).
--   quant_tournament_id    the quant tournament it belongs to.
--   slot                   the place it fills: r1/r2/r3 (the miner rounds) or one of the six quant boss-round compressions.
--   source_tournament_id   the text tournament the source task is in.
--   source_role            the text seed role of the source task (text_r1, text_r2, text_r3 or text_continuous).
--   source_task_id         the text seed task whose boss model the quant task compresses; it follows a replaced text seed.
--   source_model_repo      the exact hub repo handed to the quant task as its model when it was released; NULL while the
--                          task is held.
--   source_is_fallback     true when that repo is the text task's base model because the text boss produced no usable model.
--   superseded             true for a quant task that was replaced by another task for the same slot.
-- At most one live (not superseded) task per quant tournament and slot: the database itself refuses a duplicate.
CREATE TABLE IF NOT EXISTS quant_tournament_sources (
    quant_tournament_id TEXT PRIMARY KEY REFERENCES tournaments(tournament_id),
    source_tournament_id TEXT NOT NULL UNIQUE REFERENCES tournaments(tournament_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS quant_task_sources (
    quant_task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    quant_tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id),
    slot TEXT NOT NULL CHECK (slot IN (
        'r1', 'r2', 'r3',
        'boss_r2_freeform', 'boss_r2_fixed', 'boss_r3_freeform', 'boss_r3_fixed',
        'boss_continuous_freeform', 'boss_continuous_fixed'
    )),
    source_tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id),
    source_role TEXT NOT NULL,
    source_task_id UUID NOT NULL REFERENCES tasks(task_id),
    source_model_repo TEXT,
    source_is_fallback BOOLEAN NOT NULL DEFAULT FALSE,
    superseded BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_quant_task_sources_live_slot
    ON quant_task_sources (quant_tournament_id, slot) WHERE NOT superseded;
CREATE INDEX IF NOT EXISTS idx_quant_task_sources_source_task_id ON quant_task_sources (source_task_id);

-- migrate:down
DROP INDEX IF EXISTS idx_quant_task_sources_source_task_id;
DROP INDEX IF EXISTS idx_quant_task_sources_live_slot;
DROP TABLE IF EXISTS quant_task_sources;
DROP TABLE IF EXISTS quant_tournament_sources;
