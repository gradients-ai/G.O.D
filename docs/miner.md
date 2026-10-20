# Miner Guide

This guide is the single source of truth for miners who want to compete in G.O.D tournaments. It explains how tournament entry works, how to expose your repository, what your training repository must contain, how validators run your code, and what to expect from text and image tournaments.

Environment tournaments have been retired: they are no longer scheduled, no longer accept entries, and no longer contribute to emissions. Historical environment tournaments remain queryable through the analytics endpoints.

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
- Only one hotkey per coldkey will be enrolled in a given tournament.
- You have tested the exact Dockerfile and entrypoint locally with the example scripts.

## What Miners Do

Tournament miners submit training code. Validators clone your submitted repository at the commit you provide, build your Docker image, run your training script on validator-managed GPUs, upload your model output to Hugging Face, then evaluate it against other miners.

You do not need to provide tournament compute. You do need a running miner endpoint so validators can ask which repository and commit you want to enter for each tournament type.

Winning tournament repositories are re-uploaded to the public winners organization at [github.com/gradients-opensource](https://github.com/gradients-opensource), so only submit code you are prepared to open-source if it wins.

## Tournament Schedule

Tournament scheduling is weekly and independent for each tournament type. Times are in UTC.

| Tournament type | Scheduled start |
| --- | --- |
| Text | Monday at 11:00 UTC |
| Image | Monday at 13:00 UTC |

Environment tournaments are retired and are no longer scheduled.

The scheduler only starts a new tournament when there is no active or pending tournament of the same type. It starts during the scheduled hour only. If the system misses that hour, it waits until the next scheduled weekly window instead of starting late.

The first tournament of a type can be created immediately when no previous tournament exists. After that, the previous tournament must be completed and the next scheduled window must arrive.

### Weekly Completion Deadline

Every tournament, regardless of type, is expected to be completed by **Friday at 14:00 UTC** of the week it started. This deadline is independent of training time and GPU availability during the week, and a round being reverted does not push the deadline out — it only shrinks the time left to finish, as long as the tournament still fits before Friday.

Completion itself is a manual step: once a winner is decided, the team reviews the winning submission's code to check for exploits and confirm it reflects a genuine training improvement before marking the tournament completed. To make sure that review doesn't slip past the deadline, the validator posts a Discord reminder 30 minutes before the Friday 14:00 UTC cutoff for any tournament that is still active at that point.

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

Use `--network test` and netuid `241` equivalents for testnet.

Register with Fiber and post your miner IP at least 1 hour before the scheduled start of the tournament you want to enter.

### Participation Balance

Tournament fees are deducted from your coldkey balance after your repository passes validation. Current code constants are:

| Tournament type | Fee |
| --- | --- |
| Text | 0.7 TAO |
| Image | 0.4 TAO |

Balances are tracked per coldkey, so hotkeys under the same coldkey share the same tournament balance. Only **one tournament entry is allowed per coldkey**: if multiple hotkeys under the same coldkey respond during registration, one is kept and the rest are rejected (no second fee is charged). Transfer TAO from your coldkey to the collection address:

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
| Text | 4 |
| Image | 4 |

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
ENV=DEV uvicorn miner.asgi:app --reload --host 0.0.0.0 --port 7999 --env-file .1.env --log-level debug
```

The miner exposes:

```text
GET /training_repo/{task_type}
```

where `task_type` is one of `text` or `image`. (`environment` is retired; requests for it are no longer made.)

Requests are verified by Fiber dependencies in `miner/endpoints/training_repo.py`, including low-stake validator blacklisting.

## Submitting Your Training Repository

Validators submit you automatically by querying your miner endpoint during participant registration. You control the response in `miner/endpoints/training_repo.py`:

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
- You may return different repositories or commits for `text` and `image`.
- One miner is kept per coldkey, per IP address, and per GitHub account. Multiple hotkeys under the same coldkey cannot all enter the same tournament: only one entry is kept, and the others are rejected. If duplicates exist, entries with a valid GitHub token are preferred; otherwise the first kept candidate in the registration pass wins.

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
`-- <supported Dockerfile layout>
```

Preferred Dockerfile layout:

```text
ops/docker/standalone-text-trainer.dockerfile
ops/docker/standalone-image-toolkit-trainer.dockerfile
```

Legacy Dockerfile layout is also supported:

```text
dockerfiles/standalone-text-trainer.dockerfile
dockerfiles/standalone-image-toolkit-trainer.dockerfile
```

Required Dockerfiles by tournament type:

| Tournament type | Required Dockerfile |
| --- | --- |
| Text | `ops/docker/standalone-text-trainer.dockerfile` or legacy text path |
| Environment | `ops/docker/standalone-text-trainer.dockerfile` or legacy text path |
| Image | `ops/docker/standalone-image-toolkit-trainer.dockerfile` or legacy toolkit path |

If you compete in image tournaments, include the image toolkit Dockerfile. Image tasks can use `flux`, `z-image`, `qwen-image`, `ideogram4`, or `krea2`.

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
- An internal Docker bridge network. Treat this as no public internet.
- Dynamic resource limits of 110 GB RAM and 24 CPU cores per GPU.

## Runtime Arguments

Your Dockerfile entrypoint receives standardized CLI arguments.

### Text Trainer

Used for `InstructTextTask`, `DpoTask`, `GrpoTask`, and `ChatTask`:

```bash
--task-id             # Unique task UUID/string
--model               # Base or starting model identifier
--dataset             # Original task dataset URL
--dataset-type        # JSON string describing columns or rewards
--task-type           # InstructTextTask, DpoTask, GrpoTask, or ChatTask
--file-format         # Always s3 for tournament tasks
--expected-repo-name  # Hugging Face repo name the uploader expects
--hours-to-complete   # Task timeout in hours
```

`--dataset-type` is one of the Pydantic schemas in `core/models/dataset_models.py`:

| Task type | Dataset type payload |
| --- | --- |
| `InstructTextTask` | Instruction/input/output column names and optional formatting fields. |
| `ChatTask` | Chat template, conversation column, role field, content field, user reference, and assistant reference. |
| `DpoTask` | Prompt/chosen/rejected fields plus optional formats. |
| `GrpoTask` | Prompt field, generated reward functions, reward weights, and optional extra column. |

For GRPO tasks, reward function code is passed inside `--dataset-type`. The base implementation writes those functions into the training environment before Axolotl starts.

### Image Trainer

Used for `ImageTask`:

```bash
--task-id             # Unique task UUID/string
--model               # Base model identifier or local cached model path
--dataset-zip         # Original task dataset zip URL
--model-type          # flux, z-image, qwen-image, ideogram4, or krea2
--expected-repo-name  # Hugging Face repo name the uploader expects
--hours-to-complete   # Task timeout in hours
--trigger-word        # Optional, only when provided by task data
```

## Runtime Environment Variables

Your container may receive:

| Variable | Tasks | Meaning |
| --- | --- | --- |
| `BASELINE_STATS_PATH` | Text, image | Optional path to model/dataset baseline stats JSON. Safe to ignore. |
| `WANDB_MODE` | Text | Set to `offline`. |
| `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_ARTIFACT_DIR`, `WANDB_DATA_DIR`, `WANDB_CONFIG_DIR` | Text | Point to the local WandB logs directory for later sync. |
| `TRANSFORMERS_CACHE` | Image | Points to the Hugging Face cache path. |
| `MINER_DATASETS_DIR` | Text | Parent directory for approved requested datasets. |
| `MINER_DATASETS` | Text | Comma-separated downloaded dataset directory names. |
| `USE_KL` | Text (instruct) | Set to `1` when the task is KL-regularised. Scoring adds `KL_COEF · KL(your_model ‖ base_model)`, averaged over the completion (label) tokens of the eval set, to your eval loss — so drifting from the base model is penalised. Train with a KL term against the base model to stay competitive. Absent/unset means no KL penalty. |
| `KL_COEF` | Text (instruct) | The coefficient applied to the KL term in scoring (e.g. `0.1`). Match it in your training objective. Only set when `USE_KL=1`. |

## Cache And Output Paths

Use these paths or the helpers in `trainer/training_paths.py`. The uploader depends on the output path exactly.

| Purpose | Path |
| --- | --- |
| Final model output | `/app/checkpoints/{task_id}/{expected_repo_name}` |
| Text dataset | `/cache/datasets/{task_id}_train_data.json` |
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

Only datasets in `core/datasets/whitelisted_sft_datasets.json` are accepted. Non-whitelisted entries are filtered out, and at most two are kept by `core/datasets/whitelist.py`.

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

- Round 1 is always a single group with every participant. The group plays `SMALL_TOURNAMENT_GROUP_TASKS` (3) instruct tasks, and only the top `SMALL_TOURNAMENT_ADVANCE` (2) miners advance into the pre-boss knockout.
- Later rounds with 8 or fewer miners use pairwise knockout. Later rounds with more than 8 use a group round with groups sized around `EXPECTED_GROUP_SIZE` (32, min `MIN_GROUP_SIZE` 20), each playing one instruct task per group. Up to `TOP_WINNERS_TO_ADVANCE` (8) advance **per group**.
- Knockout pairs receive one task, selected probabilistically from instruct, DPO, and GRPO.
- **Knockout instruct, DPO and chat tasks are decided per held-out sample, not on mean loss.** Both submissions are scored on the identical held-out set, and the winner is whichever won more individual samples. A sample counts for neither side when the two losses are within `BOSS_ROUND_TIE_DEADZONE_NATS` (0.01 nats) of each other. If the sample wins are equal, or every sample falls inside that dead zone, the lower mean loss decides. The sample winner must also not be worse on the ranking loss — winning a majority of samples by a hair while losing the rest badly does not advance, and on KL-weighted tasks (every knockout instruct task from round 2) the per-sample comparison uses raw cross-entropy while the ranking loss carries the KL penalty. GRPO knockout tasks still rank on the mean score.
- The knockout round before the boss round (the last pair, whose winner becomes the boss challenger) plays a single task selected the same way as any other knockout pair — no forced model.
- The final boss round creates 6 tasks: 3 instruct (one of which always trains a large model, between `BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B` (30B) and `BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B` (72B)), 1 DPO, 1 GRPO, and one continuous-SFT chat task per lineage (currently 1: `qwen3-14b`). Each continuous-SFT lineage carries across tournaments — every round trains the next chunk starting from that lineage's previous winner (or a fixed seed model on the first run).
- **Boss-round instruct, DPO and continuous-SFT (chat) tasks are decided per held-out sample.** Both models are scored on the identical held-out set and compared sample by sample. To take a task the challenger must clear **both** of:
  - win at least `BOSS_ROUND_MIN_WIN_RATE` (55%) of *decided* samples — those where the two losses differ by more than the `BOSS_ROUND_TIE_DEADZONE_NATS` (0.01 nats) dead zone; and
  - be ahead on the mean by at least `max(BOSS_ROUND_MIN_MEAN_GAP_NATS, BOSS_ROUND_WIN_MARGIN x boss mean loss)` — a floor of 0.01 nats, the same value as the tie dead zone, that scales to 1% of the loss above a boss loss of 1.0 so it is never a weaker requirement than the old margin.

  Both are tested at the lower bound of a one-sided 99% bootstrap over 10,000 resamples of the held-out set, not on the point estimate, so a win has to survive a different draw of the data. The bootstrap seed is fixed, so every validator reaches the same verdict.

  There is no minimum number of decided samples. The mean gap is taken over the *whole* held-out set, so it is a requirement on total improvement rather than on a count: improving 20 of 800 samples by 0.25 nats averages 0.007 and fails, while improving the same 20 by 1.0 nats averages 0.026 and passes. A saturated task — where the models differ on few samples and by little — fails on that gap regardless.

  On KL-weighted instruct tasks the per-sample comparison uses raw cross-entropy, and the challenger must additionally not be worse once the KL penalty is applied.

  If *every* sample falls inside the dead zone the two models are indistinguishable on that task and it is recorded as a **draw**, not as a win for the defending champion. The defender still holds the task for the purposes of the win count, but no one outperformed anyone.
- **GRPO boss-round tasks** keep the previous rule: beat the boss's score by at least `BOSS_ROUND_WIN_MARGIN` (a fixed 1%, applied additively on the magnitude of the boss's score so it stays correct for zero/negative rewards).
- The challenger must win at least 5 of the 6 boss-round tasks, **and** win every continuous-SFT task, to dethrone the defending champion. Losing (or failing to complete) a continuous-SFT task blocks the dethrone regardless of the overall task count.

For instruct, DPO, and continuous-SFT (chat) tasks, lower adjusted loss is better. For GRPO tasks, higher reward score is better.

Your submission's `config.json` must not declare a smaller `max_position_embeddings` than the base model the task specifies. A smaller value shortens the evaluation sequence length, which drops the longer held-out rows and would score you on a different, shorter subset than everyone else — so it is treated as not a fine-tune of the task model and scores zero. A larger value is fine.

### Image Tournaments

Image tournaments use `ImageTask`.

- Round 1 always uses one group and three smaller image tasks: one Flux,
  one Ideogram, and one Z-Image task. These tasks use at most 30 source
  image/text pairs (at most 45 minutes of competition time and three held-out
  evaluation images). Average rank across the three tasks determines the top
  two miners.
- Round 2 is one pairwise knockout task, randomly using Qwen-Image or Krea.
  Its winner advances to the boss round.
- The final boss round remains 6 image tasks: Krea x2, Ideogram x2,
  Qwen-Image x1, and Z-Image x1.
- Each boss-round task is won by beating the boss's score by at least `BOSS_ROUND_WIN_MARGIN` (currently a fixed 1%, applied additively on the magnitude of the boss's score).
- The challenger must win all but at most one of the 6 boss-round tasks to dethrone the defending champion (no separate continuous-SFT gate for image).

Image tasks currently use `1xH100` in the tournament GPU requirement code.

## Scoring And Weights

The subnet is tournament-based. Emissions are split between tournament champions/runners-up and active tournament participants — nothing is burned.

Weights are set in `validator/scoring/constants.py`. Each type has a guaranteed minimum share that never moves; the remaining flex pool (`EMISSION_FLEX_POOL`, 0.20) is split between the types by a participation-driven balancer (an oversubscribed type cedes flex toward a starved one — see `validator/scoring/emission_balance.py`). A type's base therefore always lies between its minimum and its minimum plus the whole flex pool, and text + image always sum to exactly 1.0:

| Tournament type | Guaranteed minimum | Max weight (minimum + flex) |
| --- | --- | --- |
| Text | 0.50 | 0.70 |
| Image | 0.30 | 0.50 |

Active tournament participants receive `0.0001` weight each.

Within a tournament, up to the top three ranks are paid (`TOURNAMENT_PAID_RANKS`), distributed by exponential decay using `TOURNAMENT_SIMPLE_DECAY_BASE = 0.25` — roughly a 76/19/5 split across three paid ranks, or 80/20 if a clean 3rd place can't be identified (see "3rd Place" below).

Champions can earn boosted tournament allocation when boss-round performance exceeds the `0.10` performance threshold (`EMISSION_MULTIPLIER_THRESHOLD`). The excess is multiplied by `2.0` (`EMISSION_MULTIPLIER_RATE`) and capped by tournament type. A champion's weight does not decay over time — a long-reigning champion earns the same as a freshly-crowned one.

### 3rd Place

A tournament pays a 3rd place only when it can be identified unambiguously:

- **Text/Image**: the loser of the last single-pair knockout round before the boss round.

If no valid 3rd place exists (missing pre-boss round, a tie at the cutoff, or another degenerate case), the tournament pays only the top two ranks, renormalized.

## Image Tournament Tips

Image tournaments use the ai-toolkit trainer for Flux, Z-Image, Qwen-Image, Ideogram 4, and Krea 2. The common tuning split is:

- Style LoRA tasks often prefer lower learning rates, more repeats, and less aggressive fitting.
- Person, object, or concept tasks often overfit faster and may need fewer repeats, fewer epochs, and a higher learning rate.
- Tune per model type rather than assuming one recipe works across all toolkit architectures.

These are starting points, not rules. Your tournament edge usually comes from detecting the task shape and adapting the training recipe.

## Local Testing

Example local runners are in `ops/examples/training`:

```bash
./ops/examples/training/run_instruct_task.sh
./ops/examples/training/run_dpo_task.sh
./ops/examples/training/run_grpo_task.sh
./ops/examples/training/run_image_task.sh
```

You can also re-evaluate recent tasks locally after building the validator images:

```bash
docker build -f ops/docker/validator.dockerfile -t weightswandering/tuning_vali:latest .
docker build -f ops/docker/validator-diffusion.dockerfile -t diagonalge/tuning_validator_diffusion:latest .

python -m ops.validator_ops.run_evaluation --help
python -m ops.validator_ops.run_evaluation --task_id TASK_ID
python -m ops.validator_ops.run_evaluation --task_id TASK_ID --models MODEL_REPO
```

### Checking For Duplicate Submissions

Tournament submissions are de-duplicated to stop one entry being submitted many times under different identities. The check runs in three tiers: identical commit (T0), identical source after stripping whitespace/comments/ordering (T1), and a Claude pairwise review of the code deltas (T2). Cosmetic or evasive disguises — renaming, reordering, reformatting, dead code, noise hyperparameters, single-scalar tweaks, or "differences" that only trigger on inputs that are not provided at training time (an unset env var, an absent baseline-stats field, a config key the pinned library ignores) — are all treated as duplicates. Confirmed duplicates are eliminated.

You can run the exact same pairwise check locally on two repos before you submit, to confirm your work reads as genuinely distinct. It writes nothing and uploads nothing. The Claude step needs `ANTHROPIC_API_KEY`, and you should run it from a checkout of this repo (the reviewer is given this source to verify whether your claimed differentiators are real training-time inputs):

```bash
export ANTHROPIC_API_KEY=...
python -m ops.tools.tournament.dedup_check <repo_a_url> <repo_b_url> \
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

- `miner/endpoints/training_repo.py`: miner repository response endpoint.
- `core/models/payload_models.py`: `TrainingRepoResponse`, trainer request models, and runtime payloads.
- `core/models/dataset_models.py`: dataset type schemas.
- `core/models/task_models.py`: task type enum.
- `trainer/runtime.py`: Docker build/run logic and runtime arguments.
- `trainer/constants.py`: Dockerfile paths, cache paths, and resource constants.
- `trainer/training_paths.py`: path helpers used by base scripts.
- `validator/tournament/tournament_manager.py`: registration, validation, rounds, and scheduling.
- `validator/tournament/task_creator.py`: task creation by tournament type.
- `validator/tournament/gpu_requirements.py`: GPU requirement logic.
- `validator/tournament/constants.py`: tournament structure, fees, and environment round constants.
- `validator/scoring/constants.py`: scoring weights and emissions constants.
