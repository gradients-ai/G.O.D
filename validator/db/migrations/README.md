# Database Migrations

Timestamped dbmate migrations for the validator database schema.

## Contents

- `*.sql`: ordered schema migrations. Filenames start with a timestamp and should remain immutable after they have been applied in production.

Add new schema changes as new migration files rather than editing old migrations.

## Deploy order

Apply migrations first, then start the new code: text and image code paths read and write objects added by the quant migrations
(`tasks.carve_calibration`, `evaluations.infra_error_attempts`, `tournament_participants.decoder_*`, `tournament_seed_tasks`,
`quant_task_sources`). Rolling the code back after migrating is safe, since every migration is additive, nullable or defaulted.

Three quant migrations (`20261019120000`, `20261019150000`, `20261020100000`) sort before the latest migration already applied in
production (`20261020040000`). `dbmate up`, as the Taskfile runs it, applies them; `dbmate --strict` would refuse.

`20261113120000_continuous_sft_50k_chunks.sql` switches the continuous-SFT chunk layout: it doubles every lineage cursor once. Apply it (deploy) between two tournaments, in the same window in which the content service switches layout; applied earlier or later, a task reads the wrong chunk. `20261113130000_text_seed_roles.sql` adds the text seed roles and the quant boss-round roles to `tournament_seed_tasks`, and `20261113140000_quant_task_sources.sql` adds `quant_tournament_sources` and `quant_task_sources`.
