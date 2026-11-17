# Tournament Image Builds

Builds each participant's training image once per tournament, stores it in Cloudflare R2, and has every
training of that miner pull it by digest instead of rebuilding on the trainer. Trainer side:
`trainer/prepared_images/`. Storage contract: `core/image_registry.py`. Enabled with
`TOURNAMENT_IMAGE_BUILDS_ENABLED=true`; with it off nothing changes.

## Flow

1. Activation (`tournament_manager.process_pending_tournaments`) queues one `tournament_image_builds` row per
   participant (`text`, or `image-toolkit`; `image-flux` only if a FLUX task appears). Rows are also created on
   demand for any (tournament, hotkey, variant, repo, commit) a training needs, e.g. after a backup-repo switch.
2. `worker.py` (in `validator/tournament/runner.py`) every 15s: polls in-flight attempts, dispatches due builds to
   builders with free slots, runs the Shadeform fallback, and prunes on Fridays.
   - Builders are trainers that report `enabled` (registry settings present, `IMAGE_BUILDER_ENABLED` not `false`) and run no GPU jobs (`IMAGE_BUILDS_ALLOW_BUSY_TRAINERS=true` to
     allow busy ones), plus the fallback instance. A tournament's builds are grouped on one builder while it has
     slots (BuildKit shares identical steps; identical layers dedupe in R2).
   - Attempts hold a 10 min lease renewed while the builder reports progress. A lost builder, a restarted
     trainer (404), or an attempt past 90 min becomes a retry on another builder.
   - Retries by failure class: `infrastructure`/`cancelled`/`lease_expired` up to 4 attempts, `build`/`timeout`/
     `stalled` 2, `dockerfile`/`clone` 1; backoff 60s, 180s, 600s.
3. `gate.attach_prepared_images` (in `orchestrator.process_pending_tournament_tasks`) attaches the ready image to a
   pending tournament training, or holds the training back while its image builds. A failed build, or one not
   ready 2h after its row was created, does not block: the trainer builds the repo itself as before. A failed
   prebuild never fails the miner.
4. Pruning: from Friday 15:00 UTC through Sunday, completed/cancelled tournaments' repositories are deleted from R2,
   then blobs no repository references (and older than 24h) are garbage-collected; GC also runs weekly.

## Fallback builder (Shadeform)

`shadeform.py` creates one CPU instance (cheapest available non-AWS/Azure type with >= 32 vCPU and 1 TB disk) when
builds have waited 10 min with no trainer slot free. It boots the trainer service in builder mode from
`SHADEFORM_BUILDER_GOD_REPO`@`SHADEFORM_BUILDER_GOD_REF`. It is deleted after 15 min idle, after 8h, or if it is
not active within 30 min; Shadeform's own `auto_delete` (8h / $40) backs that up, and instances are re-discovered by
tag after a validator restart.

## Settings (`.vali.env`)

- `TOURNAMENT_IMAGE_BUILDS_ENABLED=true`
- `IMAGE_REGISTRY_ENDPOINT`, `IMAGE_REGISTRY_BUCKET`, `IMAGE_REGISTRY_ACCESS_KEY`, `IMAGE_REGISTRY_SECRET_KEY`,
  `IMAGE_REGISTRY_PREFIX`: needed for pruning (and passed to the fallback builder).
- Fallback (optional): `SHADEFORM_BUILDER_ENABLED=true`, `SHADEFORM_API_KEY`, `SHADEFORM_BUILDER_ORCHESTRATOR_IPS`
  (validator public IPs the builder accepts), `SHADEFORM_BUILDER_GOD_REPO`, `SHADEFORM_BUILDER_GOD_REF`,
  `SHADEFORM_BUILDER_ALLOW_GPU_TYPES`, `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN`.

## Inspecting

```sql
SELECT hotkey, variant, status, attempts, builder, failure_class, left(error_message, 200)
FROM tournament_image_builds WHERE tournament_id = '<id>' ORDER BY status, hotkey;
SELECT * FROM tournament_image_build_attempts WHERE build_id = '<uuid>' ORDER BY attempt;
```
