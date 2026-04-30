-- migrate:up

ALTER TABLE tournament_participants
ADD COLUMN IF NOT EXISTS requested_datasets TEXT;

ALTER TABLE tournament_task_hotkey_trainings
ADD COLUMN IF NOT EXISTS requested_datasets TEXT;

-- migrate:down

ALTER TABLE tournament_task_hotkey_trainings
DROP COLUMN IF EXISTS requested_datasets;

ALTER TABLE tournament_participants
DROP COLUMN IF EXISTS requested_datasets;
