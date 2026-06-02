-- migrate:up
ALTER TABLE instruct_text_tasks
    ADD COLUMN use_kl BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN kl_coef DOUBLE PRECISION DEFAULT NULL;

-- migrate:down
ALTER TABLE instruct_text_tasks
    DROP COLUMN IF EXISTS use_kl,
    DROP COLUMN IF EXISTS kl_coef;
