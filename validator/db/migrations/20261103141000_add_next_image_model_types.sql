-- migrate:up
ALTER TYPE model_type_enum ADD VALUE IF NOT EXISTS 'flux2-klein-4b';
ALTER TYPE model_type_enum ADD VALUE IF NOT EXISTS 'qwen-image-edit-2511';
ALTER TYPE model_type_enum ADD VALUE IF NOT EXISTS 'hidream-o1';
ALTER TYPE model_type_enum ADD VALUE IF NOT EXISTS 'moss-image-base';

-- migrate:down
-- PostgreSQL enum values cannot be removed safely without recreating the enum.
