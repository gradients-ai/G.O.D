-- migrate:up

-- Add environment_names array + environment_weights JSONB, migrate existing data, drop old column.
ALTER TABLE env_tasks ADD COLUMN IF NOT EXISTS environment_names TEXT[] DEFAULT '{}';
ALTER TABLE env_tasks ADD COLUMN IF NOT EXISTS environment_weights JSONB DEFAULT '[]';
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS training_start_point TEXT DEFAULT 'default';

UPDATE env_tasks
SET environment_names = ARRAY[environment_name]
WHERE environment_name IS NOT NULL;

ALTER TABLE env_tasks DROP COLUMN IF EXISTS environment_name;

-- migrate:down

ALTER TABLE env_tasks ADD COLUMN IF NOT EXISTS environment_name TEXT;

UPDATE env_tasks
SET environment_name = environment_names[1]
WHERE environment_names IS NOT NULL AND array_length(environment_names, 1) > 0;

UPDATE env_tasks
SET environment_name = environment_names[1]
WHERE environment_name IS NULL;

ALTER TABLE env_tasks DROP COLUMN IF EXISTS environment_names;
ALTER TABLE env_tasks DROP COLUMN IF EXISTS environment_weights;
ALTER TABLE tasks DROP COLUMN IF EXISTS training_start_point;
