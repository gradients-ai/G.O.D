# Image build box

Builds each tournament participant's trainer image once, when the tournament starts, and serves it to the trainers. Without it, every trainer builds the miner's Dockerfile for every job.

## Contents

| Path | Purpose |
| --- | --- |
| `docker-compose.yml` | Zot (the registry) and Woodpecker (server and agent). |
| `zot-config.template.json` | Zot settings: bucket storage, pull redirects, users, retention. `setup.sh` fills it in from `.env`. |
| `buildkitd.toml` | BuildKit settings for the `god` builder: build cache limits. |
| `woodpecker.yml` | The build recipe. `setup.sh configure-repo` commits it to the snapshot repo as `.woodpecker.yml`. |
| `setup.sh` | Sets up or updates the box, renews its certificate, configures the snapshot repo. |
| `configure_repo.py` | The `configure-repo` step: Woodpecker repo settings and secret, and the recipe commit. |
| `.env.example` | Every setting the box reads. |

## How it works

1. When a tournament is active, the validator (`validator/tournament/image_builds.py`) copies each participant's registered commit to a private snapshot repo, branch `t/<tournament_id>/<hotkey>`, as one commit without history.
2. For each Dockerfile variant the tournament uses, it starts a Woodpecker build through the API. Text and compression use the text Dockerfile. Image tournaments build both image Dockerfiles, `standalone-image-trainer` (FLUX tasks) and `standalone-image-toolkit-trainer` (other models), when the repository has them.
3. The Woodpecker agent on this box runs the recipe: clone the snapshot, `docker buildx build` with the `god` BuildKit builder, push to Zot as `god/trainers:<tournament_id>-<hotkey>-<variant>`. A failed build is retried up to three times within the run.
4. The validator reads the image digest from Zot and stores it in `tournament_images`. The scheduler holds a participant's training jobs until their image is ready, then sends `prebuilt_image` (`<registry>/god/trainers@sha256:...`) to the trainer. A failed image fails those jobs.
5. The trainer pulls the image by digest with a read-only login. Zot answers each layer request with a redirect to a signed bucket URL, so the layers come straight from the bucket.

All images share the `god/trainers` repository, so layers participants have in common (base images, frameworks) are uploaded and stored once. Images are deleted `IMAGE_RETENTION_HOURS` after they were pushed (default 120). The validator deletes the snapshot branches of finished tournaments once a day.

## Box requirements

Measured on the 2026-11-23 tournaments (35 images, text 22 GB compressed):

| Resource | Need |
| --- | --- |
| Disk | About 80 GB per build running at once, plus the BuildKit cache (capped at 150 GB). 500 GB runs 3 builds at once. |
| CPU, RAM | About 2 cores and 2 GB per build. |
| Network | Pushes are limited by the bucket: a layer is uploaded as one stream. Put the bucket in a region close to the box. |
| Ports | 80 (certificate renewal), 443 (Woodpecker: the validator's API calls and the admin login), 5000 (Zot). |

A build with a cached base takes about a minute. The first image with new layers takes 5-15 minutes, mostly uploading them.

## Setup

1. Create a private GitHub repo for the snapshots, and a fine-grained token with read and write on its contents.
2. Create a GitHub OAuth app (Settings, Developer settings, OAuth Apps) with callback URL `https://<BUILD_BOX_HOST>/authorize`.
3. Create a bucket and a key limited to it.
4. On the box, as root, with Docker installed, copy this directory to `/opt/god-image-builder`, fill in `.env` from `.env.example` (`chmod 600 .env`), and run:

   ```bash
   ./setup.sh
   ```

   It gets a Let's Encrypt certificate for `BUILD_BOX_HOST` (a short-lived one for a bare IP, renewed by a daily cron job), creates the registry users `builder` and `trainer` with passwords in `state/`, writes the Zot config, starts Zot and Woodpecker, creates the `god` BuildKit builder and logs it in to Zot. On a NATed box it also makes the public IP answer locally, so pushes do not leave the box.
5. Log in to `https://<BUILD_BOX_HOST>` with GitHub, open User settings, CLI & API, and copy the API token into `WOODPECKER_TOKEN` in `.env`. Then:

   ```bash
   ./setup.sh configure-repo
   ```

   It commits the recipe to the snapshot repo's `main`, enables the repo in Woodpecker (trusted for volumes, 60 minute timeout), stores the token as the `snapshot_read_token` secret, and prints the repo id for the validator.

Run `./setup.sh` again after changing `.env` or the files here, and `./setup.sh configure-repo` after changing `woodpecker.yml`.

## Validator settings

In `.vali.env`. All must be set; otherwise the feature is off and trainers build per job.

| Variable | Value |
| --- | --- |
| `IMAGE_BUILD_WOODPECKER_URL` | `https://<BUILD_BOX_HOST>` |
| `IMAGE_BUILD_WOODPECKER_TOKEN` | The Woodpecker API token. |
| `IMAGE_BUILD_WOODPECKER_REPO_ID` | Printed by `setup.sh configure-repo`. |
| `IMAGE_BUILD_SNAPSHOT_REPO` | `https://github.com/<owner>/<repo>.git` |
| `IMAGE_BUILD_SNAPSHOT_TOKEN` | The token with read and write on the snapshot repo. |
| `IMAGE_BUILD_REGISTRY` | `<BUILD_BOX_HOST>:5000` |
| `IMAGE_BUILD_REGISTRY_USERNAME`, `IMAGE_BUILD_REGISTRY_PASSWORD` | `trainer` and the password in `state/trainer.pass`. |

Apply the `tournament_images` migration before starting the validator with these settings.

## Trainer settings

In `.trainer.env`: `IMAGE_REGISTRY_USERNAME=trainer` and `IMAGE_REGISTRY_PASSWORD` from `state/trainer.pass`.

## Operations

| Task | How |
| --- | --- |
| See builds | Woodpecker UI at `https://<BUILD_BOX_HOST>`, or `tournament_images` in the validator database. |
| Rebuild a participant's image | Set its `tournament_images` rows back to `pending` (and `build_attempts` to 0). The next cycle builds it again. |
| Change how many builds run at once | `IMAGE_BUILD_CONCURRENCY` in `.env`, then `./setup.sh`. |
| Renew the certificate by hand | `./setup.sh renew`. Zot and Woodpecker restart only when the certificate changed. |
| Free disk | `docker buildx prune --builder god`. The cache refills from the next builds. |

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Pushes fail after about a minute with `i/o timeout` in the Zot log | Zot's request timeouts are back at the 60 second default. The template sets them to 2 hours. |
| Every image uploads its base layers again | The images are not in one repository. Zot skips only layers already in the same repository. |
| Trainers fail with `x509` errors | The certificate expired: check the cron job and `/var/log/god-image-builder-renew.log`. |
| A build stays `pending` in Woodpecker | The agent is not connected, or all its slots are busy: `docker logs woodpecker-agent`. |
