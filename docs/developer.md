# Developer Guide

This document is for people running or changing the Gradients on Demand repository. It covers the repo layout, the main runtime components, and the common commands for validators, trainers, miners, tests, and local development.

## Repository Layout

The repo is organized by runtime boundary:

| Path | Purpose |
| --- | --- |
| `core/` | Shared contracts, constants, logging, dataset whitelist data, training templates, and narrow helpers used by more than one runtime. |
| `miner/` | The public miner API. Miners answer tournament repo requests from validators. |
| `trainer/` | The trainer service that runs miner training repos inside Docker, manages GPU availability, caches models/datasets, and uploads finished models. |
| `validator/` | Validator API, tournament lifecycle, task creation, evaluation, scoring, persistence, infrastructure integrations, and transfer accounting. |
| `ops/` | Dockerfiles, compose files, config generation, observability, manual probes, tournament tools, and operational scripts. |
| `tests/` | Unit and integration tests grouped by domain. |
| `docs/` | The two maintained docs: this guide and `miner.md`. |

Inside `validator/`, the major boundaries are:

| Path | Purpose |
| --- | --- |
| `validator/app/` | App config and FastAPI dependencies. |
| `validator/db/` | Database connection, migrations, constants, and SQL access modules. |
| `validator/endpoints/` | Validator API endpoints. |
| `validator/evaluation/` | Evaluation runtimes, model checks, PvP evaluation, Docker/Basilica execution, and evaluator entrypoints. |
| `validator/infrastructure/` | MinIO, Redis/cache policy, substrate, content service, LLM, retries, and external service wrappers. |
| `validator/lifecycle/` | Main validator task lifecycle loops. |
| `validator/nodes/` | Metagraph/node refresh logic. |
| `validator/scoring/` | Task scoring, tournament scoring, scoring models, constants, and weight setting. |
| `validator/tasks/` | Task schemas, task config, task requests, task details, dataset prep, model prep, rewards, and synthetic task creation. |
| `validator/tournament/` | Tournament orchestration, participants, rounds, brackets, reports, GPU requirements, GitHub validation, and obfuscation detection. |
| `validator/transfers/` | Transfer and balance schemas. |

Avoid new `utils.py`, `shared/`, or cross-domain catch-all modules. If code belongs to scoring, put it in `validator/scoring`; if it belongs to task prep, put it in `validator/tasks/prep`; if both validator and trainer need it, consider `core/`.

## Runtime Topology

There are three services you usually care about:

| Service | Command | Default port | Role |
| --- | --- | --- | --- |
| Miner | `task miner` | `7999` | Returns a miner's training repository for each tournament type. |
| Trainer | `task trainer` | `8001` | Runs training jobs on GPUs, serves trainer status to the validator. |
| Validator | `task validator` or `task validator_dev` | `9001` | Creates tasks/tournaments, talks to miners and trainers, evaluates submissions, sets weights. |

The validator depends on Postgres, Redis, S3-compatible storage, Docker, and access to Bittensor/Fiber. Trainer nodes need Docker with GPU support, Hugging Face credentials, and enough disk for model/dataset/cache volumes.

## Setup

Clone the repo:

```bash
git clone https://github.com/rayonlabs/G.O.D.git
cd G.O.D
```

Install system dependencies on a fresh Ubuntu machine:

```bash
task bootstrap
```

Install the Python package:

```bash
task install
```

For development, install dev dependencies and pre-commit:

```bash
pip install -e '.[dev]'
pre-commit install
```

If you need GPU evaluation dependencies locally:

```bash
pip install -e '.[dev,gpu]'
```

## Configuration Files

Configuration is generated through `ops.tools.config.create_config` and written to env files in the repo root.

| Config | Command | Output |
| --- | --- | --- |
| Validator | `task config` | `.vali.env` |
| Validator dev | `python -m ops.tools.config.create_config --dev` | `.vali.env` |
| Miner | `task miner-config` | `.1.env` |
| Trainer | `task trainer-config` | `.trainer.env` |
| Auditor | `task auditor-config` | `.test-temp.env` |

Validator config prompts for wallet, hotkey, subtensor network, DB settings, S3-compatible storage, validator port, and whether to set metagraph weights. After generating `.vali.env`, add a model hash salt:

```bash
echo "MODEL_HASH_SALT=$(openssl rand -hex 32)" >> .vali.env
```

For local/dev validator features that call the content service, add:

```bash
echo "NINETEEN_API_KEY=<your-nineteen-api-key>" >> .vali.env
```

Trainer config needs at least:

```bash
HUGGINGFACE_TOKEN=<token>
HUGGINGFACE_USERNAME=<username>
WANDB_TOKEN=<token>
ORCHESTRATOR_IPS=<validator-ip-or-comma-separated-ips>
```

`ORCHESTRATOR_IPS` controls which validator IPs may call the trainer. Localhost is always allowed.

## Running A Validator

For a normal validator node:

```bash
task config
task install
task validator
```

`task validator` starts the compose stack in `ops/compose/docker-compose.yml`, refreshes Grafana, and launches `ops/validator_ops/start_validator.sh`.

For a validator with auto-updates managed by PM2:

```bash
task autoupdates
```

For local development:

```bash
task config
task validator_dev
```

`task validator_dev` starts the base compose stack plus `ops/compose/docker-compose.dev.yml`, applies DB migrations through dbmate, and starts the validator.

Useful DB commands:

```bash
task dbup
task dbdown
task postgres
task db-dump
task db-restore
```

## Running A Trainer On A New Machine

Trainer nodes execute miner code. They need Docker, NVIDIA drivers, NVIDIA Container Toolkit, the repo, and the Python package installed.

1. Clone and install:

```bash
git clone https://github.com/rayonlabs/G.O.D.git
cd G.O.D
task bootstrap
task install
```

2. Generate trainer config:

```bash
task trainer-config
```

3. Edit `.trainer.env` if needed:

```bash
ORCHESTRATOR_IPS=<validator-ip>
HUGGINGFACE_TOKEN=<token>
HUGGINGFACE_USERNAME=<username>
WANDB_TOKEN=<token>
```

4. Start the trainer:

```bash
task trainer
```

5. Verify GPU visibility:

```bash
curl http://localhost:8001/v1/trainer/get_gpu_availability
```

The trainer builds miner-provided Dockerfiles, starts training containers, streams logs, writes model outputs under the `checkpoints` Docker volume, caches models/datasets under the `cache` Docker volume, and uploads finished models to Hugging Face.

Trainer API endpoints are defined in `validator/infrastructure/service_constants.py` and registered in `trainer/endpoints.py`.

## Running A Miner Locally

Generate miner config:

```bash
task miner-config
```

Start the miner:

```bash
task miner
```

Check the tournament repo endpoint:

```bash
curl http://localhost:7999/training_repo/text
```

The endpoint implementation is `miner/endpoints/training_repo.py`. Tournament miners should read `docs/miner.md`.

## Observability

The validator can host Grafana/Loki/Prometheus for trainer logs.

On the validator:

```bash
task deploy-observability-server
```

Optional `.vali.env` overrides:

```bash
OBSERVABILITY_DOMAIN=<domain-or-ip>
GRAFANA_TRAINING_PASSWORD=<password>
LOKI_PASSWORD=<password>
GRAFANA_ANONYMOUS_ENABLED=false
```

On each trainer node:

```bash
task deploy-trainer-logs
```

The trainer log shipper requires these values in `.trainer.env`:

```bash
LOKI_ENDPOINT=https://<validator-host>:3101
LOKI_PASSWORD=<same-password-as-validator>
LOKI_USERNAME=trainer
```

Useful commands:

```bash
task logs-observability
task logs-trainer-shipper
task status-trainer-logs
task test-trainer-logs
```

Trainer log shipping uses Vector and collects containers named like `text-trainer-*`, `image-trainer-*`, `downloader-*`, and `hf-upload-*`.

## Evaluations And Manual Tools

Run the evaluation helper:

```bash
python -m ops.validator_ops.run_evaluation --help
```

Re-evaluate an existing task:

```bash
python -m ops.validator_ops.run_evaluation --task_id <task_id>
```

Evaluate a non-submitted model against a task:

```bash
python -m ops.validator_ops.run_evaluation --task_id <task_id> --models <model_name>
```

Tournament and debugging tools live under:

```text
ops/tools/tournament/
ops/tools/evaluation/
ops/tools/scoring/
ops/tools/simulations/
ops/examples/
ops/runbooks/
```

## Testing And Linting

Run lint:

```bash
uv run --extra dev ruff check core miner trainer validator ops tests
```

Run tests:

```bash
uv run --extra dev pytest -q
```

Run tests that need GPU/evaluation dependencies:

```bash
uv run --extra dev --extra gpu pytest -q
```

Run a focused test:

```bash
uv run --extra dev pytest -q -o addopts='' tests/validator/scoring/test_tournament_scoring_pipeline.py
```

`pytest` defaults to random ordering through `pyproject.toml`; use `-o addopts=''` when you need deterministic focused debugging.

## Development Notes

- Keep imports pointing at the owner module. Do not rely on compatibility imports from unrelated domains.
- Keep constants near the thing that owns them. Docker/service endpoints live under infrastructure constants, scoring constants under scoring, tournament shape under tournament, task prep constants under task prep.
- Keep miner-facing contracts in `core/models/payload_models.py` and the focused owner modules under `core/models/` when they cross process boundaries.
- Keep validator-only schemas in the validator domain that owns them.
- Prefer focused tests around changed behavior. For pure reorg work, run import smoke, `compileall`, and undefined-name lint.
