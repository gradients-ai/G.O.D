# Miner Guide

This guide is the single source of truth for miners who want to compete in G.O.D tournaments. It explains how tournament entry works, how to expose your repository, what your training repository must contain, how validators run your code, and what to expect from text, image, and environment tournaments.

## Table Of Contents

- [Quick Checklist](#quick-checklist)
- [What Miners Do](#what-miners-do)
- [Tournament Schedule](#tournament-schedule)
- [Registration Requirements](#registration-requirements)
- [Miner Setup](#miner-setup)
- [Submitting Your Training Repository](#submitting-your-training-repository)
- [Training Repository Requirements](#training-repository-requirements)
- [How Validators Run Your Code](#how-validators-run-your-code)
- [Runtime Arguments](#runtime-arguments)
- [Runtime Environment Variables](#runtime-environment-variables)
- [Cache And Output Paths](#cache-and-output-paths)
- [Requested Datasets](#requested-datasets)
- [Tournament Formats](#tournament-formats)
- [Scoring And Weights](#scoring-and-weights)
- [Environment Tournament Requirements](#environment-tournament-requirements)
- [Image Tournament Tips](#image-tournament-tips)
- [Local Testing](#local-testing)
- [Common Failure Modes](#common-failure-modes)
- [Useful References](#useful-references)

## Quick Checklist

Before a tournament starts, make sure:

- Your hotkey is registered on the subnet and your miner IP is posted to the metagraph.
- You have registered with Fiber at least 1 hour before the tournament you want to enter.
- `task miner` is running and reachable on port `7999`.
- `GET /training_repo/{task_type}` returns your GitHub repository, a full 40-character commit SHA, and an optional read-only GitHub token if the repo is private.
- Your submitted commit contains the required Dockerfiles, exact LICENSE/NOTICE files, readable source code, and no hidden datasets or pretrained models.
- Your training script writes the final model to `/app/checkpoints/{task_id}/{expected_repo_name}`.
- Your coldkey has enough tournament balance for the tournament type.
- You have tested the exact Dockerfile and entrypoint locally with the example scripts.

## What Miners Do

Tournament miners submit training code. Validators clone your submitted repository at the commit you provide, build your Docker image, run your training script on validator-managed GPUs, upload your model output to Hugging Face, then evaluate it against other miners.

You do not need to provide tournament compute. You do need a running miner endpoint so validators can ask which repository and commit you want to enter for each tournament type.

Winning tournament repositories are re-uploaded to the public winners organization at [github.com/gradients-opensource](https://github.com/gradients-opensource), so only submit code you are prepared to open-source if it wins.

## Tournament Schedule

Tournament scheduling is weekly and independent for each tournament type. Times are in UTC.

| Tournament type | Scheduled start |
| --- | --- |
| Environment | Monday at 14:00 UTC |
| Text | Thursday at 14:00 UTC |
| Image | Thursday at 15:00 UTC |

The scheduler only starts a new tournament when there is no active or pending tournament of the same type. It starts during the scheduled hour only. If the system misses that hour, it waits until the next scheduled weekly window instead of starting late.

The first tournament of a type can be created immediately when no previous tournament exists. After that, the previous tournament must be completed and the next scheduled window must arrive.

## Registration Requirements

### Subnet Registration

Register your hotkey on the G.O.D subnet:

```bash
# Mainnet
btcli s register --netuid 56

# Testnet
btcli s register --network test --netuid 241
```

Post your miner IP to the metagraph. The default miner server port is `7999`:

```bash
fiber-post-ip \
  --netuid 56 \
  --subtensor.network finney \
  --external_port 7999 \
  --wallet.name default \
  --wallet.hotkey default \
  --external_ip YOUR_PUBLIC_IP
```

Use `--network test` / netuid `241` equivalents for testnet.

Register with Fiber and post your miner IP at least 1 hour before the scheduled start of the tournament you want to enter.

### Participation Balance

Tournament fees are deducted from your coldkey balance after your repository passes validation. Current code constants are:

| Tournament type | Fee |
| --- | --- |
| Text | 0.25 TAO |
| Image | 0.20 TAO |
| Environment | 0.25 TAO |

Balances are tracked per coldkey, so hotkeys under the same coldkey share the same tournament balance. Transfer TAO from your coldkey to the collection address:

```text
5Ef5JgNv14LY4UEQFHbRQkf8TnegDV3AfAbcsJe5T2w6VQdo
```

Useful API endpoints:

```bash
curl https://api.gradients.io/tournament/fees
curl https://api.gradients.io/tournament/balance/{coldkey}
```

Collected fees are burned through the tournament balance system.

### Minimum Field Size

A pending tournament activates only after enough validated miners are available:

| Tournament type | Minimum validated miners |
| --- | --- |
| Text | 8 |
| Image | 8 |
| Environment | 5 |

If too few miners validate, the pending tournament waits and retries participant collection.

## Miner Setup

From this repository:

```bash
git clone https://github.com/rayonlabs/G.O.D.git
cd G.O.D
task bootstrap
task install
task miner-config
task miner
```

`task miner-config` writes `.1.env`. It prompts for:

- Wallet name.
- Hotkey name.
- Subtensor network or websocket address.
- Netuid, inferred as `56` for mainnet and `241` for testnet.
- Minimum validator stake threshold for requests to your miner.

`task miner` starts:

```bash
ENV=DEV uvicorn miner.server:app --reload --host 0.0.0.0 --port 7999 --env-file .1.env --log-level debug
```

The miner exposes:

```text
GET /training_repo/{task_type}
```

where `task_type` is one of `text`, `image`, or `environment`.

Requests are verified by Fiber dependencies in [miner/endpoints/tuning.py](../miner/endpoints/tuning.py), including low-stake validator blacklisting.

## Submitting Your Training Repository

Validators submit you automatically by querying your miner endpoint during participant registration. You control the response in [miner/endpoints/tuning.py](../miner/endpoints/tuning.py):

```python
from core.models.payload_models import TrainingRepoResponse
from core.models.tournament_models import TournamentType


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/YOUR_USERNAME/YOUR_REPO",
        commit_hash="0123456789abcdef0123456789abcdef01234567",
        github_token=None,
        requested_datasets=None,
    )
```

Important details:

- `commit_hash` must be a full 40-character hex commit SHA. Branch names such as `main` are rejected.
- Use `git rev-parse HEAD` to get the commit you want validators to run.
- The repository and commit must remain accessible until the tournament completes.
- You may return different repositories or commits for `text`, `image`, and `environment`.
- One miner is kept per duplicate IP address and one miner is kept per duplicate GitHub account. If duplicates exist, entries with a valid token are preferred.

### Private Repositories

Private repositories are supported if you return a GitHub fine-grained personal access token:

```python
async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/YOUR_USERNAME/YOUR_PRIVATE_REPO.git",
        commit_hash="0123456789abcdef0123456789abcdef01234567",
        github_token="github_pat_xxx",
        requested_datasets=None,
    )
```

Use a fine-grained token with access only to the submitted repository and `Contents: Read-only`. The validator checks the token against the GitHub repository API. If the token is invalid, it is ignored, which will make a private repository fail to clone.

## Training Repository Requirements

Your submitted commit is validated before entry and built before every assigned task.

### Required Files

At minimum, include:

```text
your-training-repo/
|-- LICENSE.md
|-- NOTICE
`-- dockerfiles/
    |-- standalone-text-trainer.dockerfile
    |-- standalone-image-trainer.dockerfile
    `-- standalone-image-toolkit-trainer.dockerfile
```

Required Dockerfiles by tournament type:

| Tournament type | Required Dockerfile |
| --- | --- |
| Text | `dockerfiles/standalone-text-trainer.dockerfile` |
| Environment | `dockerfiles/standalone-text-trainer.dockerfile` |
| Image, SDXL/Flux | `dockerfiles/standalone-image-trainer.dockerfile` |
| Image, Z-Image/Qwen-Image | `dockerfiles/standalone-image-toolkit-trainer.dockerfile` |

If you compete in image tournaments, include both image Dockerfiles because image tasks can use `sdxl`, `flux`, `z-image`, or `qwen-image`.

### License And Notice

Your repo must include a LICENSE file and NOTICE file matching this repository's files. Accepted LICENSE names are:

```text
LICENSE.md, LICENSE, license.md, license, License.md, License
```

Accepted NOTICE names are:

```text
NOTICE, NOTICE.txt, notice.txt, Notice.txt, notice, Notice
```

The validation compares normalized line content against this repository's LICENSE and NOTICE. Do not rewrite, summarize, or remove them.

### Readable Source

Repositories are scanned for obfuscation at the submitted commit. Obfuscated or compiled-only submissions are rejected. Avoid files such as `.pyc`, `.bin`, `.dll`, packed source, minified logic, or anything meant to hide how the method works.

Do not hide datasets, pretrained models, or private artifacts inside your Docker image. Environment tournaments explicitly allow supplementary SFT only through the requested dataset whitelist.

## How Validators Run Your Code

For each assigned task, the trainer:

1. Clones your submitted GitHub repository at `commit_hash`.
2. Downloads the base model and task dataset into a shared Docker cache volume.
3. Builds the task-specific Dockerfile from your repository root.
4. Starts your training container with assigned GPU IDs, CLI arguments, mounted volumes, and environment variables.
5. Waits up to `hours_to_complete`.
6. Uploads `/app/checkpoints/{task_id}/{expected_repo_name}` to the validator Hugging Face account.
7. Sends the uploaded model into evaluation and scoring.

Training containers run with:

- Assigned NVIDIA GPUs only.
- Docker security option `no-new-privileges`.
- All Linux capabilities dropped.
- `/cache` mounted read-only.
- `/app/checkpoints/` mounted read-write.
- An internal Docker bridge network. Treat this as no public internet; environment sidecars are reachable only when provided.
- Dynamic resource limits of 110 GB RAM and 24 CPU cores per GPU.

## Runtime Arguments

Your Dockerfile entrypoint receives standardized CLI arguments.

### Text And Environment Trainer

Used for `InstructTextTask`, `DpoTask`, `GrpoTask`, `ChatTask`, and `EnvTask`:

```bash
--task-id             # Unique task UUID/string
--model               # Base or starting model identifier
--dataset             # Original task dataset URL
--dataset-type        # JSON string describing columns, rewards, or environments
--task-type           # InstructTextTask, DpoTask, GrpoTask, ChatTask, or EnvTask
--file-format         # Always s3 for tournament tasks
--expected-repo-name  # Hugging Face repo name the uploader expects
--hours-to-complete   # Task timeout in hours
```

`--dataset-type` is one of the Pydantic schemas in [core/models/utility_models.py](../core/models/utility_models.py):

| Task type | Dataset type payload |
| --- | --- |
| `InstructTextTask` | Instruction/input/output column names and optional formatting fields. |
| `ChatTask` | Chat template, conversation column, role field, content field, user reference, and assistant reference. |
| `DpoTask` | Prompt/chosen/rejected fields plus optional formats. |
| `GrpoTask` | Prompt field, generated reward functions, reward weights, and optional extra column. |
| `EnvTask` | `environment_names`, such as `gin_rummy`, `liars_dice`, `leduc_poker`, `othello`, `clobber`, or `intercode`. |

For GRPO tasks, reward function code is passed inside `--dataset-type`. The base implementation writes those functions into the training environment before Axolotl starts.

### Image Trainer

Used for `ImageTask`:

```bash
--task-id             # Unique task UUID/string
--model               # Base model identifier or local cached model path
--dataset-zip         # Original task dataset zip URL
--model-type          # sdxl, flux, z-image, or qwen-image
--expected-repo-name  # Hugging Face repo name the uploader expects
--hours-to-complete   # Task timeout in hours
--trigger-word        # Optional, only when provided by task data
```

## Runtime Environment Variables

Your container may receive:

| Variable | Tasks | Meaning |
| --- | --- | --- |
| `BASELINE_STATS_PATH` | Text, image, environment | Optional path to model/dataset baseline stats JSON. Safe to ignore. |
| `WANDB_MODE` | Text, environment | Set to `offline`. |
| `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_ARTIFACT_DIR`, `WANDB_DATA_DIR`, `WANDB_CONFIG_DIR` | Text, environment | Point to the local WandB logs directory for later sync. |
| `TRANSFORMERS_CACHE` | Image | Points to the Hugging Face cache path. |
| `ENVIRONMENT_SERVER_URLS` | Environment | Comma-separated URLs for environment sidecars when the task needs live environment servers. |
| `MINER_DATASETS_DIR` | Text, environment | Parent directory for approved requested datasets. |
| `MINER_DATASETS` | Text, environment | Comma-separated downloaded dataset directory names. |
| `USE_KL` | Text (instruct) | Set to `1` when the task is KL-regularised. Scoring adds `KL_COEF · KL(your_model ‖ base_model)`, averaged over the completion (label) tokens of the eval set, to your eval loss — so drifting from the base model is penalised. Train with a KL term against the base model to stay competitive. Absent/unset means no KL penalty. |
| `KL_COEF` | Text (instruct) | The coefficient applied to the KL term in scoring (e.g. `0.1`). Match it in your training objective. Only set when `USE_KL=1`. |

For `intercode`-only environment tasks, `ENVIRONMENT_SERVER_URLS` may be absent because no separate training sidecar is started.

## Cache And Output Paths

Use these paths or the helpers in [trainer/utils/training_paths.py](../trainer/utils/training_paths.py). The uploader depends on the output path exactly.

| Purpose | Path |
| --- | --- |
| Final model output | `/app/checkpoints/{task_id}/{expected_repo_name}` |
| Text/environment dataset | `/cache/datasets/{task_id}_train_data.json` |
| Image dataset zip | `/cache/datasets/{task_id}_tourn.zip` |
| Cached models | `/cache/models/{model_id with "/" replaced by "--"}` |
| Requested datasets | `/cache/miner_datasets/{dataset_id with "/" replaced by "--"}` |
| WandB logs | `/app/checkpoints/wandb_logs` |

The cache volume is mounted read-only in your training container. Write checkpoints, adapters, configs that must be uploaded, and WandB logs under `/app/checkpoints`.

## Requested Datasets

Miners can ask validators to pre-download up to two approved Hugging Face datasets:

```python
async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/YOUR_USERNAME/YOUR_REPO",
        commit_hash="0123456789abcdef0123456789abcdef01234567",
        requested_datasets=[
            "SoelMgd/Poker_Dataset",
            "RZ412/PokerBench",
        ],
    )
```

Only datasets in [core/whitelisted_sft_datasets.json](../core/whitelisted_sft_datasets.json) are accepted. Non-whitelisted entries are filtered out, and at most two are kept.

If downloads succeed, your text/environment container receives:

```python
import os

root = os.environ.get("MINER_DATASETS_DIR")
names = [name for name in os.environ.get("MINER_DATASETS", "").split(",") if name]

for name in names:
    dataset_path = os.path.join(root, name)
```

Requested datasets are mounted read-only.

## Tournament Formats

### Text Tournaments

Text tournaments use `InstructTextTask`, `DpoTask`, and `GrpoTask`.

- If the field has more than 14 miners, the first round is a group round.
- Group rounds create one instruct task per group.
- The top 8 scored miners across group tasks advance.
- Once the field is 14 or fewer, rounds become pairwise knockout rounds.
- Knockout pairs receive one task, selected probabilistically from instruct, DPO, and GRPO.
- The final boss round creates 6 tasks: 2 instruct, 2 DPO, and 2 GRPO. Some final tasks may use larger models.
- The challenger must win a majority of boss-round tasks to dethrone the defending champion.

For instruct and DPO tasks, lower adjusted loss is better. For GRPO tasks, higher reward score is better.

### Image Tournaments

Image tournaments use `ImageTask`.

- If the field has more than 14 miners, the first round is a group round.
- Group rounds create one image task per group.
- The top 8 scored miners across group tasks advance.
- Once the field is 14 or fewer, rounds become pairwise knockout rounds.
- Knockout pairs receive one image task.
- The final boss round creates 6 image tasks. Up to 3 can be Z-Image or Qwen-Image tasks.
- The challenger must win a majority of boss-round tasks to dethrone the defending champion.

Image tasks currently use `1xH100` in the tournament GPU requirement code.

### Environment Tournaments

Environment tournaments use `EnvTask` and PvP or environment-specific evaluation.

- Participants are split into groups of 2 to 6.
- The defending champion is represented by the burn hotkey and auto-advances through non-final rounds.
- Non-final rounds create one environment task per group.
- Round 1 uses 2 environments per task, round 2 uses 4, round 3 uses 6, capped by the number of supported environments.
- Round 2 and later can continue from each miner's previous-round model.
- Up to one non-boss winner advances per group, with ties at the cutoff allowed.
- If the boss is in the only group and scores at least as well as the top challenger, the boss can retain without a final challenger.
- The final boss round has 3 tasks: continuation, from scratch, and previous-winner/target-model start.
- The contender wins the environment tournament only if they have no boss-round losses and at least one boss-round win. Draws are acceptable; any loss means the boss retains.

Environment group tasks use `ENV_TRAINING_HOURS = 1.5`. The from-scratch boss-round task uses `3.0` hours.

## Scoring And Weights

The subnet is tournament-based. Emissions are split between tournament champions, active tournament participants, and burn.

Current base and cap weights in [validator/core/constants.py](../validator/core/constants.py):

| Tournament type | Base weight | Max weight |
| --- | --- | --- |
| Text | 0.15 | 0.48 |
| Image | 0.125 | 0.32 |
| Environment | 0.15 | 0.16 |

Active tournament participants receive `0.0001` weight each. Undistributed weight goes to the burn hotkey.

Within a tournament, ranked participants are distributed by exponential decay using `TOURNAMENT_SIMPLE_DECAY_BASE = 0.3`.

Champions can earn boosted tournament allocation when boss-round performance exceeds the `0.05` performance threshold. The excess is multiplied by `2.0`, capped by tournament type, and reduced by time-based champion decay of `0.0033` per day after the configured decay start date.

Boss-round task wins use a progressive defense threshold:

```text
threshold = max(0.03, 0.05 * 0.8 ** (consecutive_wins - 1))
```

That means a defending champion starts with a 5% task advantage, then the threshold decays toward a 3% floor with consecutive wins.

## Environment Tournament Requirements

Supported environment names are defined in [core/constants.py](../core/constants.py):

- `gin_rummy`
- `liars_dice`
- `leduc_poker`
- `othello`
- `clobber`
- `intercode`

For OpenSpiel-style environments, one environment sidecar is usually started per assigned GPU. The URLs are passed through `ENVIRONMENT_SERVER_URLS`. Parse them as a comma-separated list:

```python
import os

server_urls = [
    url.strip()
    for url in os.environ.get("ENVIRONMENT_SERVER_URLS", "").split(",")
    if url.strip()
]
```

Environment tasks can use rollout logic that can interact with environment servers during GRPO training.

A rollout function:

- Generate completions from the model.
- Send actions or completions to the environment server.
- Collect rewards and trajectory data.
- Return the prompt tokens, completion tokens, logprobs, and reward values expected by your trainer.

The base repository includes reference rollout functions in [dockerfiles/environment_functions](../dockerfiles/environment_functions).

Rules for environment tournaments:

- Do not bundle your own dataset in the Docker image.
- Do not bundle a pretrained model in the Docker image.
- SFT is allowed only with whitelisted requested datasets.

## Image Tournament Tips

The base diffusion trainer supports SDXL, Flux, Z-Image, and Qwen-Image. The common tuning split is:

- Style LoRA tasks often prefer lower learning rates, more repeats, and less aggressive fitting.
- Person, object, or concept tasks often overfit faster and may need fewer repeats, fewer epochs, and a higher learning rate.
- SDXL defaults use more repeats than Flux in [core/constants.py](../core/constants.py).
- Z-Image and Qwen-Image use the AI Toolkit Dockerfile path.

These are starting points, not rules. Your tournament edge usually comes from detecting the task shape and adapting the training recipe.

## Local Testing

Example local runners are in [examples](../examples):

```bash
./examples/run_instruct_task.sh
./examples/run_dpo_task.sh
./examples/run_grpo_task.sh
./examples/run_image_task.sh
./examples/run_environment_task.sh
```

You can also re-evaluate recent tasks locally after building the validator images:

```bash
docker build -f dockerfiles/validator.dockerfile -t weightswandering/tuning_vali:latest .
docker build -f dockerfiles/validator-diffusion.dockerfile -t diagonalge/tuning_validator_diffusion:latest .

python -m utils.run_evaluation --help
python -m utils.run_evaluation --task_id TASK_ID
python -m utils.run_evaluation --task_id TASK_ID --models MODEL_REPO
python -m utils.run_evaluation --task_id TASK_ID --gpu_ids 0 1 --hotkeys HOTKEY_A HOTKEY_B
```

### Checking For Duplicate Submissions

Tournament submissions are de-duplicated to stop one entry being submitted many times under different identities. The check runs in three tiers: identical commit (T0), identical source after stripping whitespace/comments/ordering (T1), and a Claude pairwise review of the code deltas (T2). Cosmetic or evasive disguises — renaming, reordering, reformatting, dead code, noise hyperparameters, single-scalar tweaks, or "differences" that only trigger on inputs that are not provided at training time (an unset env var, an absent baseline-stats field, a config key the pinned library ignores) — are all treated as duplicates. Confirmed duplicates are eliminated.

You can run the exact same pairwise check locally on two repos before you submit, to confirm your work reads as genuinely distinct. It writes nothing and uploads nothing. The Claude step needs `ANTHROPIC_API_KEY`, and you should run it from a checkout of this repo (the reviewer is given this source to verify whether your claimed differentiators are real training-time inputs):

```bash
export ANTHROPIC_API_KEY=...
python -m validator.tournament.scripts.dedup_check <repo_a_url> <repo_b_url> \
    --hash-a <commit_a> --hash-b <commit_b> \
    [--token-a <gh_token>] [--token-b <gh_token>] [--out report.md]
```

It prints `DUPLICATE`, `DISTINCT`, or `DROP_EVASION` with a confidence and the reasoning. If you come back `DUPLICATE` against an existing submission, change the actual training behaviour — not just the surface.

## Common Failure Modes

- Returning a branch name instead of a full commit SHA.
- Letting a private repo token expire or omitting `Contents: Read-only`.
- Missing one of the required Dockerfiles for the tournament type.
- Changing the final output path away from `/app/checkpoints/{task_id}/{expected_repo_name}`.
- Writing outputs into `/cache`, which is read-only during training.
- Depending on public internet access from the training container.
- Bundling hidden datasets, weights, compiled code, or obfuscated code.
- Submitting a lightly-disguised copy of another entry — duplicates are detected and eliminated (see [Checking For Duplicate Submissions](#checking-for-duplicate-submissions)).
- Ignoring `hours_to_complete` and getting killed before a usable checkpoint exists.
- Not handling every model type in the tournament you enter.
- Assuming `--dataset` or `--dataset-zip` can be downloaded directly by your script instead of using the pre-downloaded cache path.

## Useful References

- [miner/endpoints/tuning.py](../miner/endpoints/tuning.py): miner repository response endpoint.
- [core/models/payload_models.py](../core/models/payload_models.py): `TrainingRepoResponse`, trainer request models, and runtime payloads.
- [core/models/utility_models.py](../core/models/utility_models.py): task types and dataset type schemas.
- [trainer/image_manager.py](../trainer/image_manager.py): Docker build/run logic and runtime arguments.
- [trainer/constants.py](../trainer/constants.py): Dockerfile paths, cache paths, and resource constants.
- [trainer/utils/training_paths.py](../trainer/utils/training_paths.py): path helpers used by base scripts.
- [validator/tournament/tournament_manager.py](../validator/tournament/tournament_manager.py): registration, validation, rounds, and scheduling.
- [validator/tournament/task_creator.py](../validator/tournament/task_creator.py): task creation by tournament type.
- [validator/tournament/gpu.py](../validator/tournament/gpu.py): GPU requirement logic.
- [validator/tournament/constants.py](../validator/tournament/constants.py): tournament structure, fees, and environment round constants.
- [validator/core/constants.py](../validator/core/constants.py): schedule, scoring weights, model/task constants, and emissions constants.
