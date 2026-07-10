-- migrate:up
ALTER TABLE tournaments ADD COLUMN IF NOT EXISTS code_review VARCHAR(20);

-- migrate:down
ALTER TABLE tournaments DROP COLUMN IF EXISTS code_review;
