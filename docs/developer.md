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

Added by hand:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `ORCHESTRATOR_IPS` | `trainer/endpoints.py` | Comma-separated validator IPs allowed to call the trainer. Always set it. `127.0.0.1` is always allowed. |
| `MODEL_HASH_SALT` | `trainer/model_artifacts.py` | Salt for the anonymous model cache directory names. Keep it secret and stable. |
| `S3_SECURE` | `validator/infrastructure/minio_client.py` | Same value as on the validator. |
| `TRAINER_PRUNE_BUILD_CACHE` | `trainer/constants.py` | Default `true`. Set `false` to keep the host Docker build cache. |
| `LOKI_ENDPOINT`, `LOKI_USERNAME`, `LOKI_PASSWORD` | `task deploy-trainer-logs` | Log shipping. See [Observability](#observability). |

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
| `dstack_orchestrator` | Training dispatch through dstack (`validator.tournament.dstack_orchestrator`). |
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

A trainer executes miner code. It needs Docker, NVIDIA drivers, the NVIDIA Container Toolkit and this package. Miner images are limited to a maximum CUDA version: see [miner.md](miner.md#training-repository).

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
   curl http://localhost:8001/v1/trainer/get_gpu_availability
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

The trainer builds each miner's Dockerfile, runs the training container and streams its logs. Outputs go to the `checkpoints` Docker volume, models and datasets to the `cache` volume, and finished models to Hugging Face.

Endpoint paths are defined in `validator/infrastructure/service_constants.py` and registered in `trainer/endpoints.py`.

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

Optional `.vali.env` values for it:

| Variable | Meaning |
| --- | --- |
| `OBSERVABILITY_DOMAIN` | Domain or IP of the server. Detected from the public IP when unset. |
| `GRAFANA_TRAINING_USERNAME`, `GRAFANA_TRAINING_PASSWORD` | Grafana login. Set your own password. |
| `GRAFANA_ANONYMOUS_ENABLED` | Default `true`. Set `false` to require a login. |
| `LOKI_USERNAME`, `LOKI_PASSWORD` | Credentials the trainers push logs with. |

Grafana listens on port `3001` and the Loki push endpoint on `3101`. The stack serves Loki on that port over plain HTTP and without authentication. Restrict the port to your trainers.

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

`ops/validator_ops/run_evaluation.py` re-runs the evaluation of a task on your own GPU. It fetches the task from `https://api.gradients.io/auditing/tasks/<task_id>` and runs the same evaluator image as the validator.

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

## Development notes

- Design rules, the change checklist and the documentation standard: [AGENTS.md](../AGENTS.md#design-rules).
- Tournament mechanics: [validator/tournament/README.md](../validator/tournament/README.md).
- Miner contract: [miner.md](miner.md).
- Scoring: [validator/scoring/README.md](../validator/scoring/README.md).
