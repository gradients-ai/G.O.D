# Tournament Miner Guide

This document is for miners who want to participate in Gradients on Demand tournaments. Miners submit training repositories; validators clone those repositories, build the Dockerfiles, run the training code on validator-controlled trainer infrastructure, evaluate the resulting models, and rank the submissions.

Miners do not need to provide tournament training hardware, but they do need a running miner service, a valid training repository, and enough balance for tournament entry fees.

## Tournament Types

The public tournament type passed to your miner endpoint is one of:

| Type | Value | Task family |
| --- | --- | --- |
| Text | `text` | Instruct, Chat, DPO, and GRPO text training tasks. |
| Image | `image` | Diffusion/image training tasks. |
| Environment | `environment` | Environment interaction and reinforcement-learning-style tasks. |

Text and image tournaments use group rounds, elimination/final rounds, and a boss/champion comparison. Environment tournaments use PvP evaluation against environment servers and include boss comparisons with environment-specific rules.

Schedules and fees are defined by the validator code:

| Type | Schedule | Participation fee |
| --- | --- | --- |
| Environment | Monday 14:00 UTC | `0.20 TAO` |
| Text | Thursday 14:00 UTC | `0.20 TAO` |
| Image | Thursday 15:00 UTC | `0.15 TAO` |

Fees are collected per coldkey balance and burned/staked according to the tournament transfer logic. Check current values through the public API when participating:

```bash
curl https://api.gradients.io/tournament/fees
curl https://api.gradients.io/tournament/balance/<coldkey>
```

The collection address is:

```text
5Ef5JgNv14LY4UEQFHbRQkf8TnegDV3AfAbcsJe5T2w6VQdo
```

## Participation Requirements

You need:

- A registered miner hotkey on the G.O.D subnet: netuid `56` on mainnet or `241` on testnet.
- A running miner service that exposes `/training_repo/{task_type}`.
- A training repository with the required Dockerfiles and entrypoints.
- A full 40-character commit SHA. Branch names are not accepted.
- Verbatim `LICENSE`/`LICENSE.md` and `NOTICE` files matching the G.O.D repository.
- No obfuscated code or checked-in machine-code artifacts such as `.pyc`, `.bin`, `.dll`, or similar.
- Sufficient coldkey tournament balance for the tournament type.

Register and post your miner IP through Bittensor/Fiber tooling, for example:

```bash
btcli s register
fiber-post-ip --netuid 56 --subtensor.network finney --external_port 7999 --wallet.name default --wallet.hotkey default --external_ip <your-ip>
```

## Miner Service Contract

Your miner service answers validator requests at:

```text
GET /training_repo/{task_type}
```

The base implementation lives in `miner/endpoints/training_repo.py`.

The response model is `TrainingRepoResponse`:

```python
class TrainingRepoResponse(BaseModel):
    github_repo: str
    commit_hash: str
    github_token: str | None = None
    requested_datasets: list[str] | None = None
```

Example:

```python
from core.models.payload_models import TrainingRepoResponse
from core.models.utility_models import TournamentType


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/YOUR_USERNAME/YOUR_TRAINING_REPO",
        commit_hash="0123456789abcdef0123456789abcdef01234567",
        github_token=None,
        requested_datasets=None,
    )
```

For a private GitHub repository, use a fine-grained read-only token:

```python
async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/YOUR_USERNAME/YOUR_PRIVATE_REPO.git",
        commit_hash="0123456789abcdef0123456789abcdef01234567",
        github_token="github_pat_xxx",
    )
```

Use `git rev-parse HEAD` to get the required commit SHA.

## Starting Your Miner

Generate config:

```bash
task miner-config
```

Start the miner:

```bash
task miner
```

Check the endpoint:

```bash
curl http://localhost:7999/training_repo/text
```

Your miner must be reachable by validators at the IP and port you posted to the metagraph.

## Training Repository Structure

The validator clones your training repository, checks out the exact commit, then builds the expected Dockerfile for the tournament task.

Required files:

```text
your-training-repo/
├── LICENSE or LICENSE.md
├── NOTICE
└── <supported Dockerfile layout>
```

Preferred Dockerfile layout:

```text
ops/docker/standalone-text-trainer.dockerfile
ops/docker/standalone-image-trainer.dockerfile
ops/docker/standalone-image-toolkit-trainer.dockerfile
```

Legacy Dockerfile layout is also supported:

```text
dockerfiles/standalone-text-trainer.dockerfile
dockerfiles/standalone-image-trainer.dockerfile
dockerfiles/standalone-image-toolkit-trainer.dockerfile
```

For each task, the trainer checks the preferred path first and then the matching legacy path. You only need to include the Dockerfiles for the tournament types you plan to support.

The G.O.D repository contains reference entrypoints:

```text
trainer/entrypoints/text_trainer.py
trainer/entrypoints/image_trainer.py
```

You can start from the base repository, fork a previous winner, or build your own repository as long as it obeys the contract below.

## Container Contract

The trainer runs your Docker image with:

- GPU device requests for the assigned GPU IDs.
- A writable checkpoints volume mounted at `/app/checkpoints/`.
- A read-only cache volume mounted at `/cache`.
- Security options `no-new-privileges` and dropped Linux capabilities.
- An internal Docker bridge network. Environment tasks receive environment server URLs on that network.

Do not rely on ad hoc external state. Your dependencies should be in the Docker image, and task inputs should come from the provided CLI arguments, mounted cache, or approved requested datasets.

## CLI Arguments

Text-family training containers receive:

```bash
--task-id <task-id>
--model <model-or-local-cache-path>
--dataset <dataset-url-or-local-path>
--dataset-type '<json-dataset-type>'
--task-type <task-type>
--file-format <file-format>
--expected-repo-name <repo-name>
--hours-to-complete <hours>
```

Text `task_type` values include:

```text
InstructTextTask
ChatTask
DpoTask
GrpoTask
EnvTask
```

Image training containers receive:

```bash
--task-id <task-id>
--model <model-or-local-cache-path>
--dataset-zip <dataset-zip-url-or-local-path>
--model-type <sdxl|flux|z_image|qwen_image>
--expected-repo-name <repo-name>
--hours-to-complete <hours>
--trigger-word <optional-trigger-word>
```

## Environment Variables

Your container may receive:

| Variable | Meaning |
| --- | --- |
| `BASELINE_STATS_PATH` | Optional path under `/cache` with baseline stats from model prep. |
| `ENVIRONMENT_SERVER_URLS` | Environment tasks only. Comma-separated internal server URLs. |
| `MINER_DATASETS_DIR` | Parent directory for approved miner-requested datasets. |
| `MINER_DATASETS` | Comma-separated downloaded dataset directory names. |
| `WANDB_*` | Offline WandB directories and config for log capture. |

Example environment server parsing:

```python
import os

raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
server_urls = [url.strip() for url in raw_urls.split(",") if url.strip()]
```

Example requested dataset parsing:

```python
import os
from pathlib import Path

datasets_dir = os.environ.get("MINER_DATASETS_DIR")
dataset_names = [name for name in os.environ.get("MINER_DATASETS", "").split(",") if name]

if datasets_dir:
    for name in dataset_names:
        dataset_path = Path(datasets_dir) / name
```

## Output Contract

Your training code must write the finished model to:

```text
/app/checkpoints/<task_id>/<expected_repo_name>
```

The helper in the base repo is:

```python
trainer.training_paths.get_checkpoints_output_path(task_id, expected_repo_name)
```

The trainer uploader expects this exact location. Changing it is the easiest way to submit a successful training run that cannot be uploaded.

Common cache paths:

| Path | Meaning |
| --- | --- |
| `/cache/models/<model-id-with-slashes-replaced>` | Downloaded base model cache. |
| `/cache/datasets/<task_id>_train_data.json` | Text dataset cache path. |
| `/cache/datasets/<task_id>_tourn.zip` | Image dataset zip cache path. |
| `/cache/miner_datasets/` | Approved miner-requested datasets. |

## Miner-Requested Datasets

Miners may ask validators to pre-download a small number of approved Hugging Face datasets.

```python
async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/YOUR_USERNAME/YOUR_REPO",
        commit_hash="0123456789abcdef0123456789abcdef01234567",
        requested_datasets=["tasksource/Boardgame-QA"],
    )
```

Only datasets in `core/datasets/whitelisted_sft_datasets.json` are accepted. Non-whitelisted datasets are filtered out. The current maximum is `MAX_REQUESTED_DATASETS = 2` in `core/datasets/whitelist.py`.

Environment tournament rule: you may only use approved requested datasets or task-provided data. Do not bake private datasets into your image.

## Text Tournaments

Text tournaments cover Instruct, Chat, DPO, and GRPO task families. The trainer passes dataset structure through `--dataset-type` as JSON. Your code must map the supplied columns correctly rather than assuming fixed column names.

Important behavior:

- Instruct and Chat generally optimize lower evaluation loss.
- DPO uses chosen/rejected preference pairs.
- GRPO uses reward functions supplied in the task payload.
- Finals currently use historical task copies across Instruct, DPO, and GRPO.

For GRPO, reward function code is supplied through the dataset/task payload and must be wired into your trainer safely.

## Image Tournaments

Image tournaments pass a base model, a dataset zip, a model type, and optionally a trigger word. Your code should extract the zip, train the correct image model family, and save the model under the required checkpoint path.

Current model type values come from `ImageModelType` and include:

```text
sdxl
flux
z_image
qwen_image
```

Current validator GPU requirement logic maps image tasks to `H100_1X`.

## Environment Tournaments

Environment tournaments train models through interaction with environment servers and evaluate by PvP or individual environment scoring.

During training:

- The validator/trainer starts environment server sidecars.
- Your container receives `ENVIRONMENT_SERVER_URLS`.
- Your rollout function should call those servers, collect rewards, and return data compatible with your training loop.

The base text trainer wires selected environments into Axolotl rollout functions for games such as Gin Rummy, Liar's Dice, and Leduc Poker.

Rules:

- Do not bake private datasets into the image.
- Do not bake pretrained models into the image.
- SFT is allowed only through whitelisted requested datasets.
- Environment training should use live environment interaction as the core training signal.

## Evaluation And Scoring

Final scoring depends on the task family:

- Instruct, Chat, DPO, and image tasks generally use loss-style evaluation where lower is better.
- GRPO and environment tasks use reward/point-style scoring where higher is better.
- PvP environment scoring uses per-environment points: win `3`, draw `1`, loss `0`.

Tournament winners feed into validator weight setting. Strong champion performance can increase the tournament weight pool; weak performance leaves more emission with the burn address.

## Local Testing

Reference examples and tools live under:

```text
ops/examples/training/
ops/tools/evaluation/
```

Useful examples:

```bash
ops/examples/training/run_instruct_task.sh
ops/examples/training/run_dpo_task.sh
ops/examples/training/run_grpo_task.sh
ops/examples/training/run_environment_task.sh
ops/examples/training/run_image_task.sh
```

You can also use the base trainer entrypoints directly while developing:

```bash
python -m trainer.entrypoints.text_trainer --help
python -m trainer.entrypoints.image_trainer --help
```

## Common Pitfalls

- Returning a branch name instead of a full commit SHA.
- Forgetting matching `LICENSE` and `NOTICE` files.
- Moving model output away from `/app/checkpoints/<task_id>/<expected_repo_name>`.
- Assuming fixed dataset columns instead of reading `--dataset-type`.
- Downloading non-whitelisted external datasets during training.
- Depending on internet access or mutable external services from inside the training container.
- Forgetting to handle `hours-to-complete`.
- Ignoring `BASELINE_STATS_PATH` when your method would benefit from baseline/model-prep information.
- Leaving obfuscated or generated binary artifacts in the repo.
