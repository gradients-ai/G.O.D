-- migrate:up
ALTER TABLE tasks
    ALTER COLUMN hours_to_complete TYPE DOUBLE PRECISION
    USING hours_to_complete::double precision;

-- migrate:down
ALTER TABLE tasks
    ALTER COLUMN hours_to_complete TYPE INTEGER
    USING ROUND(hours_to_complete)::integer;
