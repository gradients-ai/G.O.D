-- migrate:up
-- Timestamp ...120001 to not collide with 20260602120000_add_evaluation_gpu_count.sql.
ALTER TABLE instruct_text_tasks
    ADD COLUMN IF NOT EXISTS use_kl BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS kl_coef DOUBLE PRECISION DEFAULT NULL;

-- migrate:down
ALTER TABLE instruct_text_tasks
    DROP COLUMN IF EXISTS use_kl,
    DROP COLUMN IF EXISTS kl_coef;
