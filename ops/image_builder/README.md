# Image build box

Builds the image of every training job before training and serves it to the trainers, which never build. Participants' images are built when the tournament starts; any other training row (a benchmark) gets its image built the first time it is scheduled.

## Contents

| Path | Purpose |
| --- | --- |
| `docker-compose.yml` | Zot (the registry) and Woodpecker (server and agent). |
| `zot-config.template.json` | Zot settings: bucket storage, pull redirects, users, retention. `setup.sh` fills it in from `.env`. |
| `buildkitd.toml` | BuildKit settings for the `god` builder: build cache limits. |
| `woodpecker.yml` | The build recipe. `setup.sh configure-repo` commits it to the snapshot repo as `.woodpecker.yml`. |
| `setup.sh` | Sets up or updates the box, renews its certificate, configures the snapshot repo. |
| `configure_repo.py` | The `configure-repo` step: Woodpecker repo settings and secret, and the recipe commit. |
| `env.example` | Every setting the box reads. |

## Pipeline

```
validator (validator/tournament/image_builds.py)     image build box
  tournament active: one row per participant's code    Woodpecker: queue, runs, logs
  and Dockerfile variant (tournament_images); other    BuildKit "god": builds, layer cache
  training rows add theirs when first scheduled
  snapshot the commit to t/<tournament>/<source>       Zot: registry on a bucket
  in the snapshot repo (deleted once built)
  start a Woodpecker build  ----------------------->   clone snapshot, buildx build, push
  poll, then read the digest from Zot  <------------   god/trainers:<tournament>-<source>-<variant>
scheduler: ready -> prebuilt_image, building -> hold the row, failed -> fail the row
trainer: pull <registry>/god/trainers@sha256:..., layers straight from the bucket
```

- Text and compression build the text Dockerfile. Image tournaments build both image Dockerfiles when present.
- All images share one registry repository, so layers participants have in common are uploaded and stored once.
- The same code is built once per tournament, whoever it is for. A build is retried three times; images are deleted `IMAGE_RETENTION_HOURS` after push; a snapshot branch is deleted once its images are built.

## Setup

1. Create a private snapshot repo on GitHub with a read-write token, a GitHub OAuth app (callback `https://<BUILD_BOX_HOST>/authorize`), and a bucket with a key limited to it.
2. On a box with Docker (ports 80, 443, 5000 open), copy this directory, fill `.env` from `env.example` and run `./setup.sh` as root.
3. Log in at `https://<BUILD_BOX_HOST>`, create an API token (User settings, CLI & API), set `WOODPECKER_TOKEN` in `.env` and run `./setup.sh configure-repo`. It prints the repo id.
4. Validator: apply the migrations (`task dbup`) and set the `IMAGE_BUILD_*` settings ([developer guide](../../docs/developer.md#validator-valienv)). Trainers: set `IMAGE_REGISTRY_USERNAME` and `IMAGE_REGISTRY_PASSWORD` (`trainer` and `state/trainer.pass`).
