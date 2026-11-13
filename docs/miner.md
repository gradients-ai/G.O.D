# Miner Guide

This guide is the single source of truth for miners who want to compete in G.O.D tournaments. It explains how tournament entry works, how to expose your repository, what your training repository must contain, how validators run your code, and what to expect from text, quant and image tournaments.

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
- [Training Logs](#training-logs)
- [Useful References](#useful-references)

## Quick Checklist

Before a tournament starts, make sure:

- Your hotkey is registered on the subnet and your miner IP is posted to the metagraph.
- You have registered with Fiber at least 1 hour before the tournament you want to enter.
- `task miner` is running and reachable on port `7999`.
- `GET /training_repo/{task_type}` returns your GitHub repository, a full 40-character commit SHA, and an optional read-only GitHub token if the repo is private.
- Your submitted commit contains the required Dockerfiles, exact LICENSE/NOTICE files, readable source code, and no hidden datasets or pretrained models.
- Your training image uses CUDA **12.8 or lower**.
- Your training script writes the final model to `/app/checkpoints/{task_id}/{expected_repo_name}`.
- For quant tournaments, also follow the [quant tournament checklist](miner_quant.md#checklist-for-miners) (decoder, output manifest).
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
| Quant | Monday at 09:00 UTC |
| Text | Monday at 11:00 UTC |
| Image | Monday at 13:00 UTC |

Environment tournaments are retired and are no longer scheduled.

The scheduler only starts a new tournament when there is no active or pending tournament of the same type. It starts during the scheduled hour only. If the system misses that hour, it waits until the next scheduled weekly window instead of starting late.

The first tournament of a type can be created immediately when no previous tournament of that type exists. After that, the previous tournament must be completed and the next scheduled window must arrive.

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
| Quant | 0.6 TAO |
| Image | 0.4 TAO |

There is one fee per entry per tournament type: entering text and quant is two entries and two fees. Balances are tracked per coldkey, so hotkeys under the same coldkey share the same tournament balance. Only **one tournament entry is allowed per coldkey**: if multiple hotkeys under the same coldkey respond during registration, one is kept and the rest are rejected (no second fee is charged). Transfer TAO from your coldkey to the collection address:

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
| Quant | 4 |
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

where `task_type` is one of `text`, `quant` or `image`. (`environment` is retired; requests for it are no longer made.) A miner opts into the quant tournament by answering `GET /training_repo/quant`; a miner that does not is simply not entered. See [Quant Tournament: Miner Guide](miner_quant.md#entering).

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
- You may return different repositories or commits for `text`, `quant` and `image`.
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
| Quant | `ops/docker/standalone-text-trainer.dockerfile` or legacy text path |
| Image | `ops/docker/standalone-image-toolkit-trainer.dockerfile` or legacy toolkit path |

If you compete in image tournaments, include the image toolkit Dockerfile. Image tasks can use `flux`, `z-image`, `qwen-image`, `ideogram4`, or `krea2`.

Quant-tournament entrants must also have `decoder/decode.py` at the repo root at registration; text and image entrants do not. See
[Freeform track: artifact and decoder](miner_quant.md#freeform-track-artifact-and-decoder).

### CUDA Compatibility

Keep your Dockerfile base image and PyTorch/CUDA wheels at CUDA **12.8 or lower** (`cu128`, `cu126`, `cu124`, and similar). Newer CUDA runtimes (12.9+ / 13.x) are outside the supported range for trainer hosts.

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
- Supported CUDA runtime up to **12.8**.
- Docker security option `no-new-privileges`.
- All Linux capabilities dropped.
- `/cache` mounted read-only.
- `/app/checkpoints/` mounted read-write.
- An internal Docker bridge network. Treat this as no public internet.
- Dynamic resource limits of 110 GB RAM and 24 CPU cores per GPU.

## Runtime Arguments

Your Dockerfile entrypoint receives standardized CLI arguments.

### Text Trainer

Used for `InstructTextTask`, `DpoTask`, `GrpoTask`, `ChatTask`, `QuantFixedTask`, and `QuantFreeformTask`:

```bash
--task-id             # Unique task UUID/string
--model               # Model to train: a directory name under /cache/models
--dataset             # Presigned train-split URL string (no network: read /cache/datasets/{task_id}_train_data.json)
--dataset-type        # JSON string describing columns or rewards
--task-type           # InstructTextTask, DpoTask, GrpoTask, ChatTask, QuantFixedTask, or QuantFreeformTask
--file-format         # Always s3 for tournament tasks
--expected-repo-name  # Hugging Face repo name the uploader expects
--hours-to-complete   # Task timeout in hours
--quant-spec          # QuantFixedTask only: QuantFixedSpec JSON (also in the QUANT_SPEC env var)
```

`--dataset-type` is one of the Pydantic schemas in `core/models/dataset_models.py`:

| Task type | Dataset type payload |
| --- | --- |
| `InstructTextTask` | Instruction/input/output column names and optional formatting fields. |
| `ChatTask` | Chat template, conversation column, role field, content field, user reference, and assistant reference. |
| `DpoTask` | Prompt/chosen/rejected fields plus optional formats. |
| `GrpoTask` | Prompt field, generated reward functions, reward weights, and optional extra column. |
| `QuantFixedTask`, `QuantFreeformTask` | The instruct or chat-template payload above, describing the train split passed as `--dataset`. |

For GRPO tasks, reward function code is passed inside `--dataset-type`. The base implementation writes those functions into the training environment before Axolotl starts.

### Quant Tasks

`QuantFixedTask` and `QuantFreeformTask` (quant tournament only) use the text trainer invocation above. `--dataset` is the task's train split;
the held-out split used for scoring is never given to miners. They add the `QUANT_TARGET` and `QUANT_CALIBRATION_PATH`
env vars (listed below) and require an output manifest, `quant_manifest.json`, in the output directory. See
[What your code receives](miner_quant.md#what-your-code-receives) and [Output Contract](miner_quant.md#output-contract).

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
| `QUANT_SPEC` | `QuantFixedTask` | The same `QuantFixedSpec` JSON as `--quant-spec`: the weight/activation format to produce. |
| `QUANT_TARGET` | `QuantFixedTask`, `QuantFreeformTask` | `QuantTarget` JSON (env only, never a flag): the reference loss and, on freeform, the knee and cap. Normally present: training waits for the reference. Absent only on organic tasks or if the reference measurement gave up after repeated failures. See [QUANT_TARGET](miner_quant.md#quanttarget). |
| `QUANT_CALIBRATION_PATH` | `QuantFixedTask`, `QuantFreeformTask` | Optional path to a calibration set JSON. See [The calibration set](miner_quant.md#calibration-set). |

## Cache And Output Paths

Use these paths or the helpers in `trainer/training_paths.py`. The uploader depends on the output path exactly.

| Purpose | Path |
| --- | --- |
| Final model output | `/app/checkpoints/{task_id}/{expected_repo_name}` |
| Text dataset | `/cache/datasets/{task_id}_train_data.json` |
| Image dataset zip | `/cache/datasets/{task_id}_tourn.zip` |
| Cached models | `/cache/models/<--model>` (use the `--model` argument verbatim; do not derive it from the repo id) |
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

If downloads succeed, your text/quant container receives:

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

Text tournaments use `InstructTextTask`, `DpoTask`, `GrpoTask`, and the continuous-SFT chat task.

Each miner round is one instruct task, played by every remaining miner in a single group. The boss (the defending text champion) trains the three round tasks and the continuous-SFT task with its pinned code when the tournament starts, so its results are in before the rounds are played. Round 1 starts at once, so miners train it in parallel with the boss.

| Round | Model size | Task | Advance |
| --- | --- | --- | --- |
| 1 | 1-4B | one instruct task, one group | top 8 |
| 2 | 1-4B | one instruct task, one group | top 4 |
| 3 | 4-8B | one instruct task, one group | top 1 |
| Boss round | | continuous-SFT, GRPO, DPO, then one large instruct task | |

- Standings in rounds 1-3 are taken from that round's task. The boss is not part of them.
- The boss round has four tasks. The continuous-SFT task is the one the boss already trained at the start; the challenger trains it now. One GRPO task and one DPO task are created with the round, and both the boss and the challenger train them.
- The large instruct task (a model between `BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B` (30B) and `BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B` (72B)) is always last. It is created only after the first three boss-round tasks are scored, and only if the challenger can still be crowned. Otherwise nobody trains it.
- The continuous-SFT task belongs to a lineage that carries across tournaments: each tournament trains the next chunk of the lineage's data, starting from that lineage's previous winner (or a fixed seed model on the first run). Currently there is one lineage, `qwen3-8b`, seeded from `Qwen/Qwen3-8B-Base`. The task trains for a fixed number of hours on 4xH100, and its held-out test set is sized to the chunk.

**Crown rule.** The challenger and the boss are compared on 7 tasks: the tasks of rounds 1-3 and the 4 boss-round tasks. On each task the two models are compared on their task score (adjusted loss for instruct, DPO and continuous-SFT tasks, where lower is better; reward for GRPO, where higher is better). A task is a win when the challenger is at least 1% better than the boss, measured against the boss's score. Any other task is a non-win with a deficit: how far the challenger is behind the boss, as a fraction of the boss's score, or 0 when it is ahead by less than 1%. The challenger is crowned when:

- it has at most two non-wins;
- no deficit is above 0.25%;
- with two non-wins, at least one deficit is at most 0.1%.

In short: the challenger wins 6 of 7 by at least 1% and loses the 7th by at most 0.25%, or wins 5 by at least 1%, loses one by at most 0.1% and one by at most 0.25%.

A task on which the challenger has no valid result is a non-win with an unbounded deficit. A task on which the boss has no valid result, while the challenger has one, is a win. A task on which neither has a valid result is a non-win with an unbounded deficit.

**Early stop.** The rule is checked after round 3 and after the first three boss-round tasks are scored, on the tasks decided so far, counting every task still to come as a win. When no result of the remaining tasks could crown the challenger, the tournament ends with the boss retained and no further task is created.

Your submission's `config.json` must not declare a smaller `max_position_embeddings` than the base model the task specifies. A smaller value shortens the evaluation sequence length, which drops the longer held-out rows and would score you on a different, shorter subset than everyone else — so it is treated as not a fine-tune of the task model and scores zero. A larger value is fine.

### Quant Tournaments

Quant tournaments are their own tournament type, with their own boss (the quant champion) and pool, and are entered separately by answering `GET /training_repo/quant`. They use `QuantFixedTask` and `QuantFreeformTask` only; no miner trains an instruct model.

- A quant tournament takes its models from a text tournament: the active text tournament that has its boss tasks ready and is not already feeding a quant tournament. Until one exists the quant tournament waits and nothing is created. Each quant task is its own task, linked to the text task and the model it compresses.
- There are three miner rounds, each a single group compressing the text boss's model from the matching text round: round 1 is freeform and compresses the round-1 model (it starts as soon as that model is available), round 2 is fixed-format (`W2A16-asym-g64`) on the round-2 model, round 3 is freeform on the round-3 model. The top 8, then the top 4, then 1 advance (round 3 is skipped when only one miner is left).
- The boss round has six tasks: a freeform and a fixed compression of each of the text boss's round-2 model, round-3 model and continuous-SFT model. The quant boss trains its six entries as soon as each source model is available. The challenger does no instruct training. The challenger must win 5 of the 6 tasks to be crowned. To win a freeform task it needs at least 3% fewer effective bytes than the boss (`validator/tournament/constants.py: QUANT_FREEFORM_BOSS_BYTE_MARGIN`). To win a fixed task it must pass the paired per-example test and be at least 1% better on the scalar loss. The fixed tasks of the boss round use 2-bit or 3-bit formats only.
- A quant tournament with no completed quant tournament of its own takes its boss from the latest completed text tournament.
- `decoder/decode.py` is required at registration.

Everything else is in the [Quant Tournament: Miner Guide](miner_quant.md).

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

Weights are set in `validator/scoring/constants.py`. Each type has a guaranteed minimum share that never moves; the remaining flex pool (`EMISSION_FLEX_POOL`, 0.275) is split between the types by a participation-driven balancer (an oversubscribed type cedes flex toward a starved one — see `validator/scoring/emission_balance.py`). A type's base therefore always lies between its minimum and its minimum plus the whole flex pool, and text + quant + image always sum to exactly 1.0:

| Tournament type | Guaranteed minimum | Max weight (minimum + flex) |
| --- | --- | --- |
| Text | 0.15 | 0.425 |
| Quant | 0.35 | 0.625 |
| Image | 0.225 | 0.50 |

Each pool pays its own type's latest completed tournament, until a newer tournament of that type completes. Text and quant each carry their own boss, advanced only by tournaments of their own type.

The minimums apply at all times. A type with no winner yet pays nobody, and its share goes to the burn address.

Active tournament participants receive `0.0001` weight each.

Within a tournament, up to the top three ranks are paid (`TOURNAMENT_PAID_RANKS`), distributed by exponential decay using `TOURNAMENT_SIMPLE_DECAY_BASE = 0.25` — roughly a 76/19/5 split across three paid ranks, or 80/20 if a clean 3rd place can't be identified (see "3rd Place" below).

Champions can earn boosted tournament allocation when boss-round performance exceeds the `0.10` performance threshold (`EMISSION_MULTIPLIER_THRESHOLD`). The excess is multiplied by `2.0` (`EMISSION_MULTIPLIER_RATE`) and capped by tournament type. A champion's weight does not decay over time — a long-reigning champion earns the same as a freshly-crowned one.

### 3rd Place

A tournament pays a 3rd place only when it can be identified unambiguously:

- **Text**: the next miner after the challenger in round 3's standings (`validator/tournament/round_results.py: get_seeded_text_runner_up`).
- **Image**: the loser of the last single-pair knockout round before the boss round.
- **Quant**: the best non-challenger, non-boss miner in the pre-boss round's standings by cumulative average rank; if round 3 was skipped, the ranking is round 1 + round 2 among the four that advanced (`validator/tournament/round_results.py: get_quant_pre_boss_runner_up`).

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
- Building on a CUDA runtime newer than 12.8.
- Changing the final output path away from `/app/checkpoints/{task_id}/{expected_repo_name}`.
- Writing outputs into `/cache`, which is read-only during training.
- Depending on public internet access from the training container.
- Bundling hidden datasets, weights, compiled code, or obfuscated code.
- Submitting a lightly-disguised copy of another entry — duplicates are detected and eliminated (see [Checking For Duplicate Submissions](#checking-for-duplicate-submissions)).
- Ignoring `hours_to_complete` and getting killed before a usable checkpoint exists.
- Not handling every model type in the tournament you enter.
- For quant tournaments, see the [quant tournament checklist](miner_quant.md#checklist-for-miners) (decoder, output manifest).
- Assuming `--dataset` or `--dataset-zip` can be downloaded directly by your script instead of using the pre-downloaded cache path.

## Training Logs

Live training-container logs for tournament jobs are on Grafana:

http://185.141.218.122:3001/

Use this when a job fails mid-training and you need the container output the trainer shipped to Loki.

## Useful References

- [Quant Tournament: Miner Guide](miner_quant.md): the quant tournament's entry, rounds, tasks, scoring, the checkpoint and decoder contracts, and evaluation.
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
