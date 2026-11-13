-- migrate:up
-- merged_output_repo: the full model a seed's boss output was merged into when the boss's own training uploaded a LoRA
-- adapter instead of full weights (see validator/tournament/orchestrator.py, process_seed_adapter_merges). NULL while the raw output is the
-- seed's output (a full model, or an adapter still waiting for its merge). Once set, every consumer of the boss's output
-- on the seed (the round tasks chained on it, the boss round's compression seeds) reads this repo instead.
ALTER TABLE tournament_seed_tasks ADD COLUMN IF NOT EXISTS merged_output_repo TEXT;

-- migrate:down
ALTER TABLE tournament_seed_tasks DROP COLUMN IF EXISTS merged_output_repo;
