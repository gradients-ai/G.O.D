-- migrate:up
-- A quant reference is keyed per (task, hotkey, SKU). hotkey is the miner whose own bf16 model was scored, for
-- the own-model quant rounds where every miner quantises its own round-1 output; '' is the task-common
-- reference (boss and organic tasks, where every miner quantises the same model). Existing rows are all
-- task-common.
ALTER TABLE quant_sku_references ADD COLUMN IF NOT EXISTS hotkey TEXT NOT NULL DEFAULT '';
ALTER TABLE quant_sku_references DROP CONSTRAINT IF EXISTS quant_sku_references_pkey;
ALTER TABLE quant_sku_references ADD PRIMARY KEY (task_id, hotkey, sku);

-- migrate:down
DELETE FROM quant_sku_references WHERE hotkey <> '';
ALTER TABLE quant_sku_references DROP CONSTRAINT IF EXISTS quant_sku_references_pkey;
ALTER TABLE quant_sku_references ADD PRIMARY KEY (task_id, sku);
ALTER TABLE quant_sku_references DROP COLUMN IF EXISTS hotkey;
