-- migrate:up
-- Seed roles of the quant tournament's rounds 1-3 (quant_r1, quant_r2, quant_r3): the one quant task of each round, trained by
-- the quant boss as soon as its source model lands and by the round's miners once the round attaches it. The role CHECK is
-- replaced with the existing list plus these. boss_r2_fixed and boss_r3_freeform stay listed: they are no longer created, but
-- rows of them remain valid.
ALTER TABLE tournament_seed_tasks DROP CONSTRAINT IF EXISTS tournament_seed_tasks_role_check;
ALTER TABLE tournament_seed_tasks ADD CONSTRAINT tournament_seed_tasks_role_check CHECK (role IN (
    'input_small', 'input_large', 'boss_instruct', 'boss_chained_freeform', 'boss_chained_fixed',
    'boss_continuous_instruct', 'boss_continuous_freeform', 'boss_continuous_fixed',
    'text_r1', 'text_r2', 'text_r3', 'text_continuous',
    'boss_r2_freeform', 'boss_r2_fixed', 'boss_r3_freeform', 'boss_r3_fixed',
    'quant_r1', 'quant_r2', 'quant_r3'
));

-- migrate:down
-- The previous constraint is restored NOT VALID: rows with the new roles are kept and not checked (they are never deleted here);
-- the constraint only applies to rows written or updated from now on.
ALTER TABLE tournament_seed_tasks DROP CONSTRAINT IF EXISTS tournament_seed_tasks_role_check;
ALTER TABLE tournament_seed_tasks ADD CONSTRAINT tournament_seed_tasks_role_check CHECK (role IN (
    'input_small', 'input_large', 'boss_instruct', 'boss_chained_freeform', 'boss_chained_fixed',
    'boss_continuous_instruct', 'boss_continuous_freeform', 'boss_continuous_fixed',
    'text_r1', 'text_r2', 'text_r3', 'text_continuous',
    'boss_r2_freeform', 'boss_r2_fixed', 'boss_r3_freeform', 'boss_r3_fixed'
)) NOT VALID;
