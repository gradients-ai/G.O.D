# Developer guide

How to set up, configure, run and change this repository, for validator and trainer operators and for developers.

[Layout](#repository-layout) · [Topology](#runtime-topology) · [Setup](#setup) · [Configuration](#configuration-files) · [Validator](#running-a-validator) · [Trainer](#running-a-trainer) · [Miner](#running-a-miner-locally) · [Observability](#observability) · [Evaluations](#evaluations-and-manual-tools) · [Testing](#testing-and-linting) · [Notes](#development-notes)

## Repository layout

The repo is organised by runtime boundary.

| Path | Purpose |
| --- | --- |
| `core/` | Contracts, constants, logging, dataset whitelist data and training templates used by more than one runtime. |
| `miner/` | The miner API. It answers the validator's training-repository requests. |
| `trainer/` | The trainer service. It runs miner repositories in Docker, manages GPUs and caches, and uploads results. |
| `validator/` | Validator API, tournaments, task creation, evaluation, scoring, persistence and transfer accounting. |
| `subnet_api/` | Public read-only API in front of the validator. See [subnet_api/README.md](../subnet_api/README.md). |
| `ops/` | Dockerfiles, compose files, config generation, observability, manual tools and operational scripts. |
| `tests/` | Unit, integration and end-to-end tests grouped by runtime. |
| `docs/` | Maintained guides. See [docs/README.md](README.md). |

Each directory's README lists its files. Start with [validator/README.md](../validator/README.md) for the validator's domains.

## Runtime topology

| Service | Command | Port | Role |
| --- | --- | --- | --- |
| Miner | `task miner` | `7999` | Returns the miner's training repository for each tournament type. |
| Trainer | `task trainer` | `8001` | Runs training jobs on GPUs and reports GPU availability to the validator. |
| Validator | `task validator` | `VALIDATOR_PORT` (default `9001`) | Creates tasks and tournaments, calls miners and trainers, evaluates, sets weights. |

The validator needs Postgres, Redis, S3-compatible storage, Docker and access to the Bittensor chain. Provision Postgres and the object store yourself: the compose stack starts Redis but no database. Set `DATABASE_URL` in `.vali.env`: dbmate (`task dbup`) reads only that variable. A trainer needs Docker with GPU support, Hugging Face credentials and disk for the model and dataset caches.

## Setup

Clone the repository:

```bash
git clone https://github.com/gradients-ai/G.O.D.git
cd G.O.D
```

Install system dependencies on a fresh Ubuntu machine. The script also installs `task`, which the other commands use:

```bash
sudo -E ./bootstrap.sh
. ~/.bashrc
```

If the script reboots the machine, reconnect and continue.

Install the Python package with the extras a node runs:

```bash
task install
```

For development, add the dev extra and the pre-commit hooks:

```bash
pip install -e '.[dev,organic,chain,evaluation-remote,validator]'
pre-commit install
```

Add the `gpu` extra to run evaluation code locally:

```bash
pip install -e '.[dev,organic,chain,evaluation-remote,validator,gpu]'
```

[organic-package.md](organic-package.md) explains what each extra covers.

## Configuration files

`ops/tools/config/create_config.py` asks a few questions and writes an env file in the repo root.

| Node | Command | Output |
| --- | --- | --- |
| Validator | `task config` | `.vali.env` |
| Validator, development | `python -m ops.tools.config.create_config --dev` | `.vali.env` |
| Miner | `task miner-config` | `.1.env` |
| Trainer | `task trainer-config` | `.trainer.env` |
| Auditor ([ops/auditing](../ops/auditing/README.md)) | `task auditor-config` | `.test-temp.env` |

### Validator: `.vali.env`

Written by `task config`:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `WALLET_NAME`, `HOTKEY_NAME` | `validator/app/config.py` | Wallet and hotkey the validator signs with. |
| `SUBTENSOR_NETWORK`, `SUBTENSOR_ADDRESS` | `validator/app/config.py` | Chain network (`finney` or `test`) and an optional websocket address. |
| `NETUID` | `core/constants/network.py` | Subnet id. The generator writes `56` for `finney` and `241` for `test`. |
| `ENV` | `validator/app/config.py` | `prod` or `dev`. |
| `DATABASE_URL` | `validator/db/database.py`, dbmate | Full Postgres URL. Takes precedence over the `POSTGRES_*` values. |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT` | `validator/db/database.py` | Postgres connection, used when `DATABASE_URL` is not set. |
| `S3_COMPATIBLE_ENDPOINT`, `S3_COMPATIBLE_ACCESS_KEY`, `S3_COMPATIBLE_SECRET_KEY`, `S3_REGION` | `validator/infrastructure/minio_client.py` | The S3-compatible store. |
| `S3_BUCKET_NAME` | `core/constants/credentials.py` | Bucket for datasets and artifacts. |
| `VALIDATOR_PORT` | `ops/validator_ops/start_validator.sh` | Port of the validator API. |
| `FRONTEND_API_KEY` | `validator/app/dependencies.py` | API key for the protected validator endpoints. Generated for you. |
| `TRAINER_API_KEY` | `validator/infrastructure/trainer_auth.py` | Shared key sent as the `X-Trainer-Auth` header on every request to a trainer. Every trainer must have the same value in its `.trainer.env`. Generated for you. |
| `REFRESH_NODES` | `validator/app/config.py` | Whether to refresh nodes from the metagraph. |
| `DISCORD_WEBHOOK` | `validator/app/config.py` | Optional webhook for alerts. |

Added by hand:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `HF_READ_ONLY_TOKEN` | `validator/evaluation/constants.py` | Hugging Face token passed to evaluation containers. |
| `TOURNAMENT_HF_USERNAME` | `core/constants/credentials.py` | Organisation the validator reads tournament uploads from. Default `gradients-io-tournaments`. |
| `EVAL_BACKEND` | `validator/evaluation/remote.py` | Remote evaluation backend: `basilica` (default) or `runpod`. |
| `EVAL_RUNPOD_*` | `validator/evaluation/runpod.py` | Runpod backend settings. See [the evaluation runbook](../ops/runbooks/evaluation_flow_debugging.md). |
| `NINETEEN_API_KEY` | `validator/tasks/synthetics/constants.py` | Key for the content service. |
| `ANTHROPIC_API_KEY` | `validator/infrastructure/`, `validator/tournament/repo_diff_report.py` | Key for code review, duplicate-submission review and repository diff reports. |
| `S3_SECURE` | `validator/infrastructure/minio_client.py` | Default `true` (HTTPS). Set `false` only for a plain-HTTP store. |
| `SUBNET_API_PORT` | `ops/validator_ops/start_validator.sh` | Starts the public API. See [subnet_api/README.md](../subnet_api/README.md). |
| `GRAFANA_PASSWORD` | `ops/compose/docker-compose.observability.yml` | Validator Grafana admin password. Required: compose refuses to start without it. `task validator` (`ops/validator_ops/setup_grafana.sh`) generates one when it is missing. |

```bash
echo "HF_READ_ONLY_TOKEN=<read-only-token>" >> .vali.env
```

The validator reads each tournament upload from `<TOURNAMENT_HF_USERNAME>/<expected_repo_name>`. Its trainers upload there, so each trainer's `HUGGINGFACE_USERNAME` must name the same organisation.

### Trainer: `.trainer.env`

Written by `task trainer-config`:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `HUGGINGFACE_TOKEN`, `HUGGINGFACE_USERNAME` | `trainer/runtime.py`, `trainer/containers/` | Account that downloads models and receives the uploaded results. |
| `WANDB_TOKEN` | `trainer/runtime.py` | Weights & Biases token. |
| `S3_COMPATIBLE_ENDPOINT`, `S3_COMPATIBLE_ACCESS_KEY`, `S3_COMPATIBLE_SECRET_KEY`, `S3_REGION` | `validator/infrastructure/minio_client.py` | The validator's S3-compatible store. Read when the trainer starts. |
| `QUANT_DECODE_WORK_ROOT` | `trainer/constants.py` | Optional. Directory for the freeform decode. Default `~/.local/share/god-trainer/quant_decode`. |
| `TRAINER_API_KEY` | `trainer/auth.py` | Required. The `TRAINER_API_KEY` from the validator's `.vali.env`. Every request must carry it in the `X-Trainer-Auth` header; without it set, the trainer rejects every request with 403 and logs an error at startup. |
| `ORCHESTRATOR_IPS` | `trainer/auth.py` | Required. Comma-separated validator IPs allowed to call the trainer. There is no default; `127.0.0.1` is always allowed (still with the key). `ORCHESTRATOR_IP` is read when `ORCHESTRATOR_IPS` is unset. |

Added by hand:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `MODEL_HASH_SALT` | `trainer/model_artifacts.py` | Salt for the anonymous model cache directory names. Keep it secret and stable. |
| `S3_SECURE` | `validator/infrastructure/minio_client.py` | Same value as on the validator. |
| `TRAINER_PRUNE_BUILD_CACHE` | `trainer/constants.py` | Default `true`. Set `false` to keep the host Docker build cache. |
| `TRAINER_IMAGE_BUILD_NO_CACHE` | `trainer/constants.py` | Default `false`. Set `true` to build every miner image from scratch. |
| `TRAINER_IMAGE_BUILD_PULL` | `trainer/constants.py` | Default `false`. Set `true` to check the registry for a newer base image on every build. |
| `TRAINER_IMAGE_BUILD_CACHE_MAX_AGE_HOURS` | `trainer/constants.py` | Default `24`. A build does not reuse a layer older than this, and the image layer cache is emptied this often. |
| `TRAINER_IMAGE_BUILD_CACHE_MAX_DISK_USED_PERCENT` | `trainer/constants.py` | Default `80`. The image layer cache is emptied when Docker's disk is this full. |
| `TRAINER_DOCKER_BUILD_TIMEOUT_MINUTES` | `trainer/constants.py` | Default `45`. Time limit of one miner image build. |
| `QUANT_DECODE_UPLOAD_RETRY_WINDOW_SECONDS` | `trainer/constants.py` | Optional. How long a decoded upload that fails on network errors is retried in the background before it counts as an infra error. Default `7200`; `0` turns the retry off. See [Pending uploads](../trainer/quant_decode/README.md#pending-uploads). |
| `QUANT_DECODE_UPLOAD_RETRY_MIN_FREE_DISK_GB` | `trainer/constants.py` | Optional. Decoded files are kept for an upload retry only while the decode work root has this much disk free. Default `50`. |
| `QUANT_DECODE_UPLOAD_PART_BYTES`, `QUANT_DECODE_UPLOAD_PARALLEL`, `QUANT_DECODE_UPLOAD_SEND_SLICE_BYTES`, `QUANT_DECODE_UPLOAD_HTTP_*`, `QUANT_DECODE_UPLOAD_MIN_BYTES_PER_SECOND` | `trainer/constants.py` | Optional. Tuning of decoded-file uploads for a slow or lossy link. Defaults and meaning: [`trainer/quant_decode/README.md`](../trainer/quant_decode/README.md#decode-job). |
| `LOKI_ENDPOINT`, `LOKI_USERNAME`, `LOKI_PASSWORD` | `task deploy-trainer-logs` | Log shipping. See [Observability](#observability). |
| `DISCORD_WEBHOOK` | `trainer/monitoring/monitor.py` | Optional. Where the GPU monitor posts alerts. See [GPU monitor](#gpu-monitor). |

```bash
echo "MODEL_HASH_SALT=$(openssl rand -hex 32)" >> .trainer.env
```

<details>
<summary>The freeform decode work root</summary>

`QUANT_DECODE_WORK_ROOT` must be a directory of its own on the trainer host. The trainer creates it, or adopts it if empty, writes a `.god-quant-decode-root` marker and sets mode `0700`. A non-empty directory without the marker is refused and left untouched, so no decode can run until the setting is fixed. Allow about three times the largest uncompressed model in free space. Internals: [trainer/quant_decode/README.md](../trainer/quant_decode/README.md).

</details>

<details>
<summary>Plain-HTTP object storage</summary>

With `S3_SECURE=false`, for example a local MinIO, set the value on the validator and on every trainer. The trainer's and the evaluator's containers must be able to resolve the store's endpoint.

</details>

### Miner: `.1.env`

`task miner-config` writes `WALLET_NAME`, `HOTKEY_NAME`, `SUBTENSOR_NETWORK`, `SUBTENSOR_ADDRESS` when given, `NETUID`, `ENV`, `REFRESH_NODES` and `MIN_STAKE_THRESHOLD`. Add `MINER_TOURNAMENT_TYPES` by hand: a comma-separated list of the tournaments the miner enters (`text`, `image`, `quant`). `miner/endpoints/training_repo.py` declines any type not listed, and the default is none.

## Running a validator

A validator needs the services in [Runtime topology](#runtime-topology): provision Postgres and the S3-compatible store before you start. Its hotkey needs subnet registration, with the `btcli` command in the [miner hub](miner.md#subnet-registration-and-the-miner-service): weight setting fails for a hotkey that is not on the subnet.

1. Clone and install.

   ```bash
   git clone https://github.com/gradients-ai/G.O.D.git
   cd G.O.D
   sudo -E ./bootstrap.sh
   . ~/.bashrc
   task install
   ```

2. Generate `.vali.env`, then add the hand-set values from [Configuration files](#validator-valienv).

   ```bash
   task config
   ```

3. Start the validator.

   ```bash
   task validator
   ```

`task validator` starts the compose stack in `ops/compose/docker-compose.yml`: Redis, dbmate and the observability services. The dbmate service applies pending migrations in the background. The task then runs `ops/validator_ops/start_validator.sh`, which starts the validator processes under pm2 without waiting for dbmate. When an update carries migrations, run `task dbup` first. See [Deploy order](../validator/db/migrations/README.md#deploy-order).

| pm2 process | Runs |
| --- | --- |
| `validator_api` | The validator API (`validator.asgi`). |
| `validator_cycle` | The task lifecycle loops (`validator.lifecycle.runner`). |
| `weight_setter` | Weight setting (`validator.scoring.weights`). |
| `tournament_orchestrator` | Training dispatch for tournaments (`validator.tournament.orchestrator`). |
| `tournament_cycle` | Tournament creation and advancement (`validator.tournament.runner`). |
| `subnet_api` | The public API, only when `SUBNET_API_PORT` is set. |

To update automatically under pm2:

```bash
task autoupdates
```

For local development:

```bash
task config
task validator_dev
```

`task validator_dev` adds `ops/compose/docker-compose.dev.yml` (Redis Commander and Adminer), applies migrations and starts the validator.

Database commands:

| Command | Does |
| --- | --- |
| `task dbup` | Applies pending migrations. |
| `task dbdown` | Rolls back the latest migration. |
| `task postgres` | Opens `psql` with the settings in `.vali.env`. |
| `task db-dump` | Dumps the database to `god-db-backup.dump`. |
| `task db-restore` | Restores `god-db-backup.dump`. |

Migrations and the order to deploy them in: [validator/db/migrations/README.md](../validator/db/migrations/README.md#deploy-order).

## Running a trainer

A trainer executes miner code. It needs Docker, NVIDIA drivers, the NVIDIA Container Toolkit and this package. `task bootstrap` and `task trainer` install DCGM (`datacenter-gpu-manager` and the `nvidia-dcgm` service) when it is not already running. Miner images are limited to a maximum CUDA version: see [miner.md](miner.md#training-repository).

1. Clone and install.

   ```bash
   git clone https://github.com/gradients-ai/G.O.D.git
   cd G.O.D
   sudo -E ./bootstrap.sh
   . ~/.bashrc
   task install
   ```

2. Generate `.trainer.env`, then add the hand-set values from [Configuration files](#trainer-trainerenv).

   ```bash
   task trainer-config
   ```

3. Start the trainer under pm2.

   ```bash
   task trainer
   ```

4. Check that the GPUs are visible.

   ```bash
   curl -H "X-Trainer-Auth: <TRAINER_API_KEY>" http://localhost:8001/v1/trainer/get_gpu_availability
   ```

5. Register the trainer with the validator. Run this where the validator API is reachable, with the `FRONTEND_API_KEY` from `.vali.env`.

   ```bash
   curl -X POST "http://<validator-host>:<VALIDATOR_PORT>/v1/tournament_orchestrator/add_trainer?trainer_ip=<trainer-ip>" \
     -H "Authorization: Bearer <FRONTEND_API_KEY>"
   ```

The validator then reads the trainer's GPUs from port `8001` (or from `<ip>:<port>` when you pass one) and stores them. The response lists the GPUs it registered.

| Request | Does |
| --- | --- |
| `POST /v1/tournament_orchestrator/add_trainer?trainer_ip=<ip>` | Registers a trainer and its GPUs. |
| `GET /v1/tournament_orchestrator/trainers` | Lists the registered trainers. |
| `DELETE /v1/tournament_orchestrator/remove_trainer/<ip>` | Removes a trainer. |

The trainer builds each miner's Dockerfile, runs the training container and streams its logs. Outputs go to the `checkpoints` Docker volume, models and datasets to the `cache` volume, and finished models to Hugging Face. A training container mounts only its own job's directory of the `checkpoints` volume (`<task_id>/<hotkey>`), and the `cache` volume read-only.

Endpoint paths are defined in `validator/infrastructure/service_constants.py` and registered in `trainer/endpoints.py`. Every trainer route requires both an allowed source IP (`ORCHESTRATOR_IPS`) and the `X-Trainer-Auth: <TRAINER_API_KEY>` header (`trainer/auth.py`); the validator adds the header from its own `TRAINER_API_KEY`.

### GPU monitor

The GPU throttle monitor starts with the trainer. `task trainer` and `ops/trainer_ops/trainer.service` both run `uvicorn trainer.asgi:app`, and that process starts the monitor. Set `DISCORD_WEBHOOK` in `.trainer.env` to post alerts; omit it to log only.

It ticks every 5 s (`GPU_MONITOR_POLL_SECONDS`). A clock alert needs a thermal, power-brake, or hardware-slowdown reason and an SM clock below 90% of max for two ticks (10 s), or 10 s of NVML thermal/reliability/board-limit throttle time inside 5 min. Temperature is included in the alert text and is not its own cutoff. Normal power-capping under load is ignored. `gpu_throttle_observation` logs every non-clean poll so the 90% line can be checked against a healthy loaded box. `task trainer` and the trainer systemd unit install and start DCGM when `dcgmi` or `nvidia-dcgm` is missing. Once the hostengine is up, that same tick uses `dcgmi health` for ECC, NVLink, PCIe, thermal, and power failures. Warnings are logged. A PCIe box with no DCGM, or no `nvidia-fabricmanager` unit, falls back to the NVML checks: a new uncorrected ECC error, pending page retirement, 10 new corrected ECC errors in 5 min (`GPU_MONITOR_ECC_SBE_THRESHOLD`), an NVLink error increase, or a hardware Xid. Application Xids are logged and left alone. Set `GPU_MONITOR_DCGM=0` to force the NVML path. If `nvidia-fabricmanager` is installed and not active, every GPU on the box is withdrawn; a machine without that unit is left alone.

A fault withdraws that GPU until the trainer process restarts and fails any training already on it. The trainer log line starts with `[GPU_THROTTLE]`. The orchestrator puts that training back to pending with the normal status update and does not add an `n_training_attempts`. Other GPUs on the box keep taking work, and the round is not held. A recovery message follows 5 min after a clock throttle clears; ECC, Xid, and fabric-manager withdrawals stay until restart.

## Running a miner locally

```bash
task miner-config
task miner
```

The validator calls `GET /training_repo/{task_type}` with `task_type` set to `text`, `image` or `quant`. The request is signed, and the miner rejects unsigned requests, so a plain `curl` does not return the repository.

The implementation is `miner/endpoints/training_repo.py`. The endpoint contract and how to enter are in [miner.md](miner.md#entering-a-tournament).

## Observability

The validator can host Grafana, Loki and Prometheus for trainer logs. Each trainer ships its container logs there with Vector.

On the validator:

```bash
task deploy-observability-server
```

`.vali.env` values for it (`GRAFANA_TRAINING_PASSWORD` and `LOKI_PASSWORD` are required):

| Variable | Meaning |
| --- | --- |
| `OBSERVABILITY_DOMAIN` | Domain or IP of the server. Detected from the public IP when unset. |
| `GRAFANA_TRAINING_USERNAME`, `GRAFANA_TRAINING_PASSWORD` | Grafana admin login. Required; there is no default. |
| `GRAFANA_ANONYMOUS_ENABLED` | Default `true`. Set `false` to require a login. |
| `LOKI_USERNAME`, `LOKI_PASSWORD` | Credentials the trainers push logs with. `LOKI_PASSWORD` is required. |

Grafana listens on port `3001`. Loki itself is not published: `loki-gateway` (nginx, `ops/observability/nginx/nginx.conf`) serves port `3101` over plain HTTP, and

- pushing logs (`POST /loki/api/v1/push`) needs `LOKI_USERNAME`/`LOKI_PASSWORD`;
- anyone may read: `GET` on `query`, `query_range`, `labels`, `label/<name>/values`, `series`, `index/stats` and `tail`;
- everything else (deleting logs, config, other push paths, any other method) is refused with `403`.

The deploy writes the push credentials to `ops/observability/nginx/.htpasswd` from `.vali.env` every time, so changing `LOKI_PASSWORD` there and redeploying is enough on the server.

On each trainer, set these in `.trainer.env`:

```bash
LOKI_ENDPOINT=http://<validator-host>:3101
LOKI_PASSWORD=<same-password-as-validator>
LOKI_USERNAME=trainer
```

`task deploy-trainer-logs` refuses to start without `LOKI_ENDPOINT` and `LOKI_PASSWORD`.

Then start the shipper:

```bash
task deploy-trainer-logs
```

Vector collects the logs of containers that carry the `task_id`, `hotkey`, `model` and `task_type` labels.

| Command | Does |
| --- | --- |
| `task logs-observability` | Follows the observability server logs. |
| `task logs-trainer-shipper` | Follows the Vector shipper logs. |
| `task status-trainer-logs` | Shows the state of both stacks and Vector's health. |
| `task test-trainer-logs` | Starts labelled test containers to check shipping. |
| `task stop-observability-server`, `task stop-trainer-logs` | Stops either stack. |

Where miners read their training logs: [miner.md](miner.md#troubleshooting).

## Evaluations and manual tools

`ops/validator_ops/run_evaluation.py` re-runs the evaluation of a task on your own GPU. It fetches the task from `https://api.sn56.io/auditing/tasks/<task_id>` (`SUBNET_API_URL`) and runs the same evaluator image as the validator.

Supported task types: instruct, chat, DPO, GRPO and image. Compression tasks are not supported: the helper exits with an error.

Use it on tasks at most 7 days old. Older tasks have expired dataset links.

The helper runs the image tags in `core/constants/docker.py`. Docker pulls a tag that is missing locally. To evaluate with this checkout's evaluator code, build the images first:

```bash
docker build -f ops/docker/validator.dockerfile -t gradientsio/text-evaluator:basilica .
docker build -f ops/docker/validator-diffusion.dockerfile -t gradientsio/image-evaluator:basilica .
```

Re-evaluate every submission of a task:

```bash
python -m ops.validator_ops.run_evaluation --task_id <task_id>
```

Evaluate models that were not submitted:

```bash
python -m ops.validator_ops.run_evaluation --task_id <task_id> --models <org/model> <org/model>
```

| Flag | Meaning |
| --- | --- |
| `--task_id` | Required. The task to fetch. |
| `--gpu_ids` | GPU ids to use. Default `0`. |
| `--models` | Evaluate these repositories instead of the task's submissions. |
| `--hotkeys` | Evaluate only the submissions of these hotkeys. Not combinable with `--models`. |

Results are written to the log.

Other tools:

| Path | Purpose |
| --- | --- |
| `ops/tools/tournament/` | Start, inspect and repair tournaments. |
| `ops/tools/evaluation/` | Evaluation probes. |
| `ops/tools/scoring/` | Scoring and weight checks. |
| `ops/tools/quant/` | Compression release checks. |
| `ops/tools/simulations/` | Offline simulations. |
| `ops/examples/` | Example API requests and local training scripts. |
| `ops/runbooks/` | Debugging runbooks. |

Most of these directories have a README that lists the commands.

## Testing and linting

Lint:

```bash
uv run --extra dev ruff check core miner trainer validator ops tests
```

Test:

```bash
uv run --extra dev pytest -q
```

Tests that need GPU and evaluation dependencies:

```bash
uv run --extra dev --extra gpu pytest -q
```

One file, in a fixed order:

```bash
uv run --extra dev pytest -q -o addopts='' tests/validator/scoring/test_tournament_scoring_pipeline.py
```

`pyproject.toml` runs tests in random order. `-o addopts=''` turns that off.

## Incoming changes for miners

`docs/incoming_changes.yaml` lists changes that affect miners and the UTC Monday each takes effect. The website reads it from `https://raw.githubusercontent.com/gradients-ai/G.O.D/main/docs/incoming_changes.yaml`, so write entries for a first-time miner reader: describe what will be, and never invent numbers that are not decided.

Add an entry in a PR:

```yaml
  - id: my-change               # kebab-case, unique
    title: Short name
    summary: One or two sentences for miners.
    why: Why the change is being made.
    areas: [text]               # any of: text, image, compression
    status: planned
    date: 2026-12-07            # UTC Monday it takes effect
    miner_action: Nothing.      # or what miners must do
    prs: []
    docs: []                    # repo-relative paths, e.g. [docs/miner.md]
    shipped_on: null
```

Statuses: `planned`, `in_progress`, `testing`, `shipped`, `postponed`. A `shipped` entry must have `shipped_on`; other statuses keep it `null`. Move `date` and set `postponed` when a change slips.

Validate locally with `python -m ops.tools.incoming_changes.validate`. CI runs the same check on PRs that touch the file.

To ship an entry, add the label `incoming:<id>` to the PR that delivers it (create the label once with `gh label create "incoming:<id>"`). When the PR merges, the `Incoming Changes Ship` workflow sets `status: shipped`, `shipped_on` to the merge date (UTC) and adds the PR number to `prs`. Because `main` requires reviewed PRs, the workflow pushes the update to `incoming-changes/ship-pr-<N>` and opens a PR for it; approve and merge that PR. If the repo does not allow Actions to open PRs and no `INCOMING_CHANGES_TOKEN` secret is set, the workflow comments a compare link on the merged PR instead.

## Development notes

- Design rules, the change checklist and the documentation standard: [AGENTS.md](../AGENTS.md#design-rules).
- Tournament mechanics: [validator/tournament/README.md](../validator/tournament/README.md).
- Miner contract: [miner.md](miner.md).
- Scoring: [validator/scoring/README.md](../validator/scoring/README.md).
