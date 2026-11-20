-- migrate:up
-- Tournament seed tasks: every task the boss trains, created and trained upfront at tournament activation (see
-- validator/tournament/README.md). Seeds are not tournament_tasks rows (no round, no pair) until the boss
-- round attaches the boss-round ones. One row per (tournament, role, attempt): a replacement is a new attempt and the
-- old row becomes 'replaced'.
--   role:   input_small, input_large (the instruct inputs of R1-R3), boss_instruct, boss_chained_freeform,
--           boss_chained_fixed, boss_continuous_instruct, boss_continuous_freeform, boss_continuous_fixed (the boss
--           round's six tasks).
--   status: training  the seed task exists and the boss's run (or the input's evaluation) is not finished
--           trained   boss-round seeds only: the boss's run is done and the task is held for the boss round
--           ready     usable: an input seed scored SUCCESS with a full-model boss output; a boss-round seed SUCCESS
--           replaced  superseded by a later attempt of the same role
--           failed    out of replacements (SEED_MAX_REPLACEMENTS); the tournament halts until a human decides
--   boss_repo / boss_commit: the pinned boss code the seed was trained with.
--   prep_rejected: only meaningful on a 'replaced' row: replaced without charging the role's SEED_MAX_REPLACEMENTS, because
--           the dataset-drawn role (input_small, input_large, boss_instruct) ended PREP_TASK_FAILURE (near-duplicate or
--           any prep rejection before training), or a chained compression seed was retired because its upstream
--           instruct seed was replaced. The prep-rejected datasets of a tournament are excluded from its later
--           redraws; SEED_MAX_PREP_REJECTIONS bounds the redraws per role.
--   attached_round_id: the boss round that attached this seed (boss-round seeds only). Written once and never
--           cleared, so a seed that was ever attached is recognised even after its tournament_tasks row is detached;
--           an attached seed is never replaced (a replacement would train the boss after R3).
CREATE TABLE IF NOT EXISTS tournament_seed_tasks (
    tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id),
    role TEXT NOT NULL CHECK (role IN (
        'input_small', 'input_large', 'boss_instruct', 'boss_chained_freeform', 'boss_chained_fixed',
        'boss_continuous_instruct', 'boss_continuous_freeform', 'boss_continuous_fixed'
    )),
    attempt INTEGER NOT NULL DEFAULT 1,
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('training', 'trained', 'ready', 'replaced', 'failed')),
    boss_repo TEXT,
    boss_commit TEXT,
    attached_round_id TEXT,
    prep_rejected BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tournament_id, role, attempt)
);
CREATE INDEX IF NOT EXISTS idx_tournament_seed_tasks_task_id ON tournament_seed_tasks(task_id);

-- migrate:down
DROP INDEX IF EXISTS idx_tournament_seed_tasks_task_id;
DROP TABLE IF EXISTS tournament_seed_tasks;
