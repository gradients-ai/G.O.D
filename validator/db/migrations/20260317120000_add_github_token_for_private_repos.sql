-- migrate:up

ALTER TABLE tournament_participants
ADD COLUMN IF NOT EXISTS github_token TEXT;

ALTER TABLE tournament_task_hotkey_trainings
ADD COLUMN IF NOT EXISTS github_token TEXT;

-- migrate:down

ALTER TABLE tournament_task_hotkey_trainings
DROP COLUMN IF EXISTS github_token;

ALTER TABLE tournament_participants
DROP COLUMN IF EXISTS github_token;
