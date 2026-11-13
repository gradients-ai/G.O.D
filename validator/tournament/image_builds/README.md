# Tournament image builds

When `DEPOT_IMAGE_BUILDS_ENABLED=true`, tournament activation writes one durable
build batch. The tournament runner claims each submission variant independently,
builds it in Depot, and marks it ready only after Depot Registry confirms the
tag and digest. Ready rows can train while other rows are still building.

Apply migration first:

```bash
dbmate --migrations-dir validator/db/migrations up
```

Required validator settings:

- `DEPOT_TOKEN`: organization API token.
- `DEPOT_ORGANIZATION_ID`: Depot organization/registry ID.
- `DEPOT_IMAGE_BUILDS_ENABLED=true`.

Optional settings:

- `DEPOT_REGION=us-east-1`
- `DEPOT_BUILD_CONCURRENCY=10`
- `DEPOT_BUILDS_PER_INSTANCE=1`
- `DEPOT_BUILD_ATTEMPT_TIMEOUT_SECONDS=1800`
- `DEPOT_BUILD_MAX_ATTEMPTS=2`
- `DEPOT_BUILD_RETRY_DELAY_SECONDS=15`
- `DEPOT_CACHE_KEEP_DAYS=14`
- `DEPOT_IMAGE_RETENTION_DAYS=7`

Retries use a fresh Depot build ID in the same per-submission project so cached
layers survive. The project is deleted only after tournament completion and the
retention window.

Trainer hosts must authenticate to `registry.depot.dev` with a pull-only token
in Docker's host credential store. Do not mount that credential store into
training containers. Trainer pull settings:

- `IMAGE_PREFETCH_CONCURRENCY=2`
- `IMAGE_PREFETCH_WAIT_TIMEOUT_SECONDS=3600`
- `IMAGE_CACHE_HIGH_WATERMARK=0.85`
- `IMAGE_CACHE_LOW_WATERMARK=0.75`
- `IMAGE_PREFETCH_STATE_PATH=trainer/image_prefetch_state.json`
