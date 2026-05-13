-- migrate:up

ALTER TABLE tournaments
ADD COLUMN diff_report TEXT;

-- migrate:down

ALTER TABLE tournaments
DROP COLUMN IF EXISTS diff_report;
