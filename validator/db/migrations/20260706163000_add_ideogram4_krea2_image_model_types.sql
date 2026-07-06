-- migrate:up
ALTER TYPE model_type_enum ADD VALUE IF NOT EXISTS 'ideogram4';
ALTER TYPE model_type_enum ADD VALUE IF NOT EXISTS 'krea2';

-- migrate:down
-- PostgreSQL enum values cannot be removed safely without recreating the enum.
