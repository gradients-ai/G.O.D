-- migrate:up
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'image_training_mode_enum') THEN
        CREATE TYPE image_training_mode_enum AS ENUM ('text-to-image', 'edit');
    END IF;
END $$;

ALTER TABLE image_tasks
ADD COLUMN IF NOT EXISTS training_mode image_training_mode_enum NOT NULL DEFAULT 'text-to-image';

ALTER TABLE image_text_pairs
ADD COLUMN IF NOT EXISTS source_image_url TEXT;

-- migrate:down
ALTER TABLE image_text_pairs DROP COLUMN IF EXISTS source_image_url;
ALTER TABLE image_tasks DROP COLUMN IF EXISTS training_mode;
DROP TYPE IF EXISTS image_training_mode_enum;
