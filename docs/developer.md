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
| `docs/` | The maintained docs: this guide, `miner.md`, and its quant companion `miner_quant.md`. |

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

Install the Python package (node extras: `organic`, `chain`, `evaluation-remote`, `validator`):

```bash
task install
```

See [organic-package.md](organic-package.md) for what each extra covers.

For development, install the node extras plus dev dependencies and pre-commit:

```bash
pip install -e '.[dev,organic,chain,evaluation-remote,validator]'
pre-commit install
```

If you need GPU evaluation dependencies locally:

```bash
pip install -e '.[dev,organic,chain,evaluation-remote,validator,gpu]'
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

Evaluation containers authenticate to Hugging Face with `HF_READ_ONLY_TOKEN` from `.vali.env`. It must be a token
with read access to the `gradients-io-tournaments` organisation, where the tournament repos are uploaded.

```bash
echo "HF_READ_ONLY_TOKEN=<read-only-token-with-gradients-io-tournaments-access>" >> .vali.env
```

The validator reads every tournament upload from `<TOURNAMENT_HF_USERNAME>/<expected_repo_name>`; unset, that org is
`gradients-io-tournaments`. A test validator points it at a scratch org, and its trainers' `HUGGINGFACE_USERNAME` must
name the same org, since the trainer uploads there.

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
S3_COMPATIBLE_ENDPOINT=<endpoint>
S3_COMPATIBLE_ACCESS_KEY=<key>
S3_COMPATIBLE_SECRET_KEY=<secret>
S3_REGION=us-east-1
```

Optional: `QUANT_DECODE_WORK_ROOT=<dir>` is where the freeform quant decode runs (default
`~/.local/share/god-trainer/quant_decode`; `task trainer-config` prompts for it). It must be a directory of its own: the
trainer creates it (or adopts it if empty), writes a `.god-quant-decode-root` marker, and chmods it `0o700`; a non-empty
directory without the marker is refused untouched and every decode on it ends as `infra_error`. It needs about 3x the
largest bf16 model free, and it must be a path on the trainer host (the decoder sandbox bind-mounts host paths).

`ORCHESTRATOR_IPS` controls which validator IPs may call the trainer. Localhost is always allowed. The `S3_*` values
are the validator's S3-compatible store: the freeform quant decode uploads its decoded shards there, and
`validator/infrastructure/minio_client.py` reads them at import. `task trainer-config` prompts for them. The store is
reached over HTTPS; set `S3_SECURE=false` (validator and trainer alike) only for a plain-HTTP store such as a local
MinIO, whose endpoint the trainer's and the evaluator's containers must also be able to resolve.

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

### Deploy order

Apply the DB migrations (`task dbup`) before starting the new validator code. Text and image paths also read and write the
new schema objects (`tasks.carve_calibration`, `evaluations.infra_error_attempts`, `tournament_participants.decoder_*`,
the `tournament_seed_tasks` and `quant_task_sources` tables), so new code on an unmigrated database
fails for every task type. Rolling the code back after migrating is safe: every migration is additive, nullable or defaulted.

Three quant migrations (`20261019120000`, `20261019150000`, `20261020100000`) sort before the latest migration already
applied in production (`20261020040000`). `dbmate up`, as the Taskfile runs it, applies them; `dbmate --strict` would refuse.

`20261113120000_continuous_sft_chunk_scheme.sql` switches the continuous-SFT chunk layout: it doubles every lineage cursor once. Apply it (that is, deploy) between two tournaments, in the same window in which the content service switches layout; applied earlier or later, a task would read the wrong chunk. `20261113130000_text_seed_roles.sql` adds the text seed roles and the quant boss-round roles, `20261113140000_quant_task_sources.sql` adds `quant_tournament_sources` and `quant_task_sources`, `20261113150000_quant_round_seed_roles.sql` adds the quant round seed roles (`quant_r1`-`quant_r3`) and `20261113170000_tournament_boss_rank_comparisons.sql` adds `tournament_boss_rank_comparisons`.

### Task details responses

The public task endpoints (`get_task_details`, the by-account and completed-organic listings) return one details model per
task type. Quant tasks return `QuantFixedTaskDetails` (the common fields plus `base_model_repository` and the required
`quant_spec` format) or `QuantFreeformTaskDetails` (the common fields plus `base_model_repository`); neither exposes the
dataset, calibration or seed information, and seed tasks stay hidden by the same rule as other tasks.

## Running A Trainer On A New Machine

Trainer nodes execute miner code. They need Docker, NVIDIA drivers, NVIDIA Container Toolkit, the repo, and the Python package installed. Training containers should target CUDA **12.8 or lower**.

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
S3_COMPATIBLE_ENDPOINT=<endpoint>
S3_COMPATIBLE_ACCESS_KEY=<key>
S3_COMPATIBLE_SECRET_KEY=<secret>
S3_REGION=us-east-1
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
curl http://localhost:7999/training_repo/quant
```

The endpoint implementation is `miner/endpoints/training_repo.py`. Tournament miners should read `docs/miner.md`.

## Observability

The validator can host Grafana/Loki/Prometheus for trainer logs.

Live miner training logs (Grafana on the current observability host):

http://185.141.218.122:3001/

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
- Tournament types: `text`, `quant` and `image` (`core/models/tournament_models.py: TournamentType`; `environment` is retired). Each has its own schedule slot (Monday 09:00 quant, 11:00 text, 13:00 image UTC), fee (text 0.7, quant 0.6, image 0.4 TAO; `validator/tournament/constants.py`), and emission pool (minimums text 0.15, quant 0.35, image 0.225, flex 0.275; `validator/scoring/constants.py`); there are two bosses, the text champion and the quant champion. There is no feature flag: quant is scheduled weekly like the other two. The miner endpoint is `GET /training_repo/{task_type}` with the tournament type as its value; `GET /tournament/fees` returns `text_tournament_fee_rao`, `image_tournament_fee_rao` and the additive `quant_tournament_fee_rao`, and the tournament endpoints carry additive `quant` / `quant_*` fields next to the text and image ones.
- Tournament structure: a text tournament is three single-group instruct rounds (1-4B top 8, 1-4B top 4, 4-8B top 1; `TEXT_ROUND_PLAN`) and a four-task boss round (continuous-SFT, GRPO, DPO, then one big instruct task on a 30-72B model). A quant tournament is three single-group rounds (top 8, 4, 1; `QUANT_ROUND_PLAN`) and a four-task boss round (freeform on the text boss's round-2 model, fixed 2-3 bit on its round-3 model, freeform and fixed 2-3 bit on its continuous model). In both, the boss's entries are ranked in rounds 1-3 without taking an advancement slot, rounds 2 and 3 are always played, and the challenger is judged against the boss by one crown rule over 7 head-to-heads (`validator/tournament/crown.py`), with an early stop after the round-3 cut when the crown is already out of reach; without a crown a paid contender the boss outranks over rounds 1-3 is paid like the boss (`validator/tournament/boss_rank_shares.py`). The rules are in `validator/tournament/README.md`, "Crown, early stop and placements". `tournament_seed_tasks` holds every task a boss trains ahead of the round that uses it: the text roles `text_r1`, `text_r2`, `text_r3`, `text_continuous` (trained by the text boss at tournament start) and the quant roles `quant_r1`, `quant_r2`, `quant_r3`, `boss_r2_freeform`, `boss_r3_fixed`, `boss_continuous_freeform`, `boss_continuous_fixed` (trained by the quant boss as each source model lands). Each quant task is a task of its own, linked to its text task and model in `quant_task_sources` (`validator/tournament/quant_sources.py`); a quant tournament is bound to one text tournament in `quant_tournament_sources`. Continuous-SFT tasks train one chunk of their lineage per tournament.
- Quant task types: `QuantFixedTask` and `QuantFreeformTask` (`core/models/task_models.py`) are tournament-only; there is no organic create endpoint. They are the two tracks of the compression task: fixed names the weight/activation format (`QuantFixedSpec` in `core/models/quant_models.py`) and lowest test loss wins; freeform allows any format plus a miner decoder, entries must stay within `QUANT_BAR_CAP` of the uncompressed reference loss, and smallest total bytes wins. Both persist to `quant_tasks` and rank through `MinerResultsText` (`TEXT_TASK_TYPES`).
