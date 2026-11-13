-- migrate:up
-- Seed roles of the text tournament (text_r1, text_r2, text_r3, text_continuous: the instruct task of each of rounds 1-3 and
-- the continuous-SFT task of the boss round, all trained by the boss at tournament start) and of the quant boss round
-- (boss_r2_freeform, boss_r2_fixed, boss_r3_freeform, boss_r3_fixed). The role CHECK is replaced with the existing list plus these.
ALTER TABLE tournament_seed_tasks DROP CONSTRAINT IF EXISTS tournament_seed_tasks_role_check;
ALTER TABLE tournament_seed_tasks ADD CONSTRAINT tournament_seed_tasks_role_check CHECK (role IN (
    'input_small', 'input_large', 'boss_instruct', 'boss_chained_freeform', 'boss_chained_fixed',
    'boss_continuous_instruct', 'boss_continuous_freeform', 'boss_continuous_fixed',
    'text_r1', 'text_r2', 'text_r3', 'text_continuous',
    'boss_r2_freeform', 'boss_r2_fixed', 'boss_r3_freeform', 'boss_r3_fixed'
));

-- migrate:down
-- The previous constraint is restored NOT VALID: rows with the new roles are kept and not checked (they are never deleted here);
-- the constraint only applies to rows written or updated from now on.
ALTER TABLE tournament_seed_tasks DROP CONSTRAINT IF EXISTS tournament_seed_tasks_role_check;
ALTER TABLE tournament_seed_tasks ADD CONSTRAINT tournament_seed_tasks_role_check CHECK (role IN (
    'input_small', 'input_large', 'boss_instruct', 'boss_chained_freeform', 'boss_chained_fixed',
    'boss_continuous_instruct', 'boss_continuous_freeform', 'boss_continuous_fixed'
)) NOT VALID;
