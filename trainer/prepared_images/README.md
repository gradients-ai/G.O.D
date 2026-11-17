# Prepared Images

Trainer side of tournament images that are built once per tournament and reused by every training of that miner.
The validator side (scheduling, retries, pruning) is `validator/tournament/image_builds/`.

## Contents

- `builder.py`: build jobs posted by the validator (`POST /v1/trainer/images/build`). Clone -> pick the variant's
  Dockerfile -> `docker buildx build --push` (zstd or gzip, see below) to the localhost registry -> publish to R2. Kills a build after
  `IMAGE_BUILD_TIMEOUT_SECONDS` (60 min) or `IMAGE_BUILD_STALL_SECONDS` (10 min) without output, and classifies
  failures (`dockerfile`, `clone`, `build`, `infrastructure`, `timeout`, `stalled`) for the validator's retry policy.
  `GET /v1/trainer/images/builder` reports free slots: `min(free_disk*0.8/80GB, vCPUs/4, RAM/8GB, 16)`.
- `r2_transfer.py`: parallel multipart upload / parallel ranged download (sha256-verified) of image blobs, written
  in the `registry:3` storage layout (see `core/image_registry.py`).
- `local_registry.py`: localhost `registry:3` (port `IMAGE_LOCAL_REGISTRY_PORT`, default 5050) used as staging;
  its storage is wiped whenever no build or pull holds it.
- `puller.py`: `ensure_prepared_image` used by `trainer/runtime.py`; skips the download when the image is already
  local (e.g. this trainer built it). On failure the runtime clones and builds the repo locally, as before.
- `cache.py`: last-use tracking and eviction of prepared images (TTL `PREPARED_IMAGE_TTL_HOURS`, LRU under disk
  pressure), plus pruning of build cache older than 24h. Run from `trainer/cleanup.py`.

## Host requirements

- Docker with buildx. Any image store works: the builder uses zstd on the containerd image store and gzip on
  overlay2 (which ignores zstd), detected per host and reported as `compression` in the builder status. Every
  trainer can pull both (Docker >= 23), so images from different hosts mix freely. zstd unpacks faster on pull, so
  prefer the containerd store (`"features": {"containerd-snapshotter": true}`, default on fresh Docker 29) where
  switching is convenient; switching hides existing images, which are pulled again once.
- Recommended `daemon.json`: `"dns": ["1.1.1.1", "8.8.8.8"]` (DNS failures inside `RUN` steps were the most common
  build failure under concurrency) and `"max-concurrent-downloads": 10`. Log in to Docker Hub to avoid base-image
  pull rate limits.

## Settings (`.trainer.env`)

- `IMAGE_REGISTRY_ENDPOINT`, `IMAGE_REGISTRY_BUCKET`, `IMAGE_REGISTRY_ACCESS_KEY`, `IMAGE_REGISTRY_SECRET_KEY`,
  optional `IMAGE_REGISTRY_PREFIX` (default `registry`): R2 bucket, token scoped to that bucket. Needed to pull.
- `IMAGE_BUILDER_ENABLED` (default `true`): set `false` to stop the validator scheduling builds on this host. Builds
  only happen when the registry settings are present and the host runs no GPU jobs.
- Optional: `IMAGE_BUILD_MAX_CONCURRENCY`, `IMAGE_BUILD_TIMEOUT_SECONDS`, `IMAGE_BUILD_STALL_SECONDS`,
  `IMAGE_LOCAL_REGISTRY_PORT`, `IMAGE_LOCAL_REGISTRY_STORAGE`.

Credentials stay on the host; they are never passed to training containers.
