# Miner Guide

This guide is the single source of truth for miners who want to compete in G.O.D tournaments. It explains how tournament entry works, how to expose your repository, what your training repository must contain, how validators run your code, and what to expect from text, quant and image tournaments.

Environment tournaments are retired: they are not scheduled, accept no entries and carry no emissions. Historical environment tournaments remain queryable through the analytics endpoints.

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

Environment tournaments are retired and not scheduled.

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

where `task_type` is one of `text`, `quant` or `image`. (`environment` is retired and never requested.) A miner opts into the quant tournament by answering `GET /training_repo/quant`; a miner that does not is simply not entered. See [Quant Tournament: Miner Guide](miner_quant.md#entering).

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

Do not hide datasets, pretrained models, or private artifacts inside your Docker image. Extra datasets are allowed only through the [requested dataset whitelist](#requested-datasets).

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
| `USE_KL` | Text (instruct) | Set to `1` when the task is KL-regularised. Scoring adds `KL_COEF · KL(your_model ‖ base_model)`, averaged over the completion (label) tokens of the eval set, to your eval loss. Absent/unset means no KL penalty. |
| `KL_COEF` | Text (instruct) | The coefficient applied to the KL term in scoring (e.g. `0.1`). Only set when `USE_KL=1`. |
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

Each round is one instruct task on a model from the standard pool, played by every remaining miner in a single group. The boss (the defending text champion) trains the three round tasks and the continuous-SFT task with its pinned code when the tournament starts, so its results are in before the rounds are played. Round 1 starts at once, so miners train it in parallel with the boss.

| Round | Task | Model size | Advance |
| --- | --- | --- | --- |
| 1 | instruct, one group | 1-4B | top 8 |
| 2 | instruct, one group | 1-4B | top 4 |
| 3 | instruct, one group | 4-8B | top 1 |
| Boss round | continuous-SFT, GRPO, DPO, then the big instruct task | big task 30-72B | |

- Standings in rounds 1-3 are taken from that round's task. The boss's entry is ranked in them with the same ordering used between miners, but it never takes an advancement slot; a boss entry with no valid result ranks behind every valid entry.
- Every round 1-3 runs in full, and rounds 2 and 3 are played even with one miner left: the rounds decide 2nd and 3rd place, and each is one of the challenger's head-to-heads with the boss.
- The boss round has four tasks. The continuous-SFT task is the one the boss trained at the start; the challenger trains it now. One GRPO task and one DPO task are created with the round, and both the boss and the challenger train them.
- The big instruct task (a model between `BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B` (30B) and `BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B` (72B)) is always last. It is created only after the other three boss-round tasks are scored, and only if the challenger can still be crowned. Otherwise nobody trains it.
- The continuous-SFT task belongs to a lineage that carries across tournaments: each tournament trains the next chunk of the lineage's data, starting from that lineage's previous winner (or a fixed seed model on the first run). There is one lineage, `qwen3-8b`, seeded from `Qwen/Qwen3-8B-Base`.

**Crown rule.** The challenger is compared with the boss on 7 head-to-heads: its results on the tasks of rounds 1-3 against the boss's, and the 4 boss-round tasks. Each is a win, or a non-win within the tight allowance, within the looser allowance, or beyond both. The challenger is crowned when exactly these 7 are decided, with at most two non-wins, none beyond the looser allowance and, with two non-wins, at least one within the tight allowance.

| Task | Win | Tight allowance | Looser allowance |
| --- | --- | --- | --- |
| instruct, continuous-SFT, DPO | passes the paired per-example test against the boss and scores at least 1% better | at most 0.1% behind | at most 0.25% behind |
| GRPO | scores at least 1% better | at most 0.1% behind | at most 0.25% behind |

- The score is the adjusted loss for instruct, DPO and continuous-SFT (lower is better) and the reward for GRPO (higher is better). Percentages are relative to the boss's score. A non-win's deficit is how far you trail the boss, and 0 when you are level or ahead.
- The paired per-example test compares your per-example losses with the boss's on the identical held-out rows ([`validator/tournament/thresholds.py: compare_paired_losses`](../validator/tournament/thresholds.py)). It is used only against the boss; miners are never compared with each other on it.
- If per-example losses are missing on either side or cannot be paired (including when no example is comparable), the boss keeps the task: it is a non-win, judged against the allowances on its score.
- A task on which you have no valid result is beyond both allowances, so it rules out the crown. A task on which the boss has no valid result while you have one is a win.

**Early stop.** The crown is checked once the round-3 cut leaves the challenger: the rule is applied to its round 1-3 head-to-heads with the four boss-round tasks counted as wins. If the crown is already out of reach, no boss round is created and the boss is retained. In the boss round the big task is created last, and only while the crown is still possible.

**Placements.** Without a crown the boss keeps 1st and the contenders take 2nd and 3rd; a paid contender the boss outranks on average over rounds 1-3 has its share paid like the boss's: see [Placements](#placements).

The full mechanics, shared with the quant tournament, are in [Crown, early stop and placements](../validator/tournament/README.md#crown-early-stop-and-placements).

Your submission's `config.json` must not declare a smaller `max_position_embeddings` than the base model the task specifies. A smaller value shortens the evaluation sequence length, which drops the longer held-out rows and would score you on a different, shorter subset than everyone else — so it is treated as not a fine-tune of the task model and scores zero. A larger value is fine.

### Quant Tournaments

Quant tournaments are their own tournament type, with their own boss (the quant champion) and pool, and are entered separately by answering `GET /training_repo/quant`. They use `QuantFixedTask` and `QuantFreeformTask` only; no miner trains an instruct model.

- A quant tournament takes its models from a text tournament that has its boss tasks and is not already feeding a quant tournament: an active one, else one completed within the last 7 days. Until one exists the quant tournament waits and nothing is created. Each quant task is its own task, linked to the text task and the model it compresses.
- Three rounds, each a single group compressing the text boss's model from the matching text round: R1 freeform on the round-1 model (top 8), R2 fixed `W2A16-asym-g64` on the round-2 model (top 4), R3 freeform on the round-3 model (top 1). The quant boss's entries are ranked in each round but never take an advancement slot, and rounds 2 and 3 are played even with one miner left.
- The boss round has four tasks: freeform on the text boss's round-2 model, fixed 2-3 bit on its round-3 model, and freeform and fixed 2-3 bit on its continuous-SFT model.
- The crown uses the same 7 head-to-heads, early stop and placements as text, with compression wins and allowances: see [The boss round](miner_quant.md#the-boss-round).
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

The subnet is tournament-based. Emissions are split between tournament champions/runners-up and active tournament participants.

Weights are set in `validator/scoring/constants.py`. Each type has a guaranteed minimum share that never moves; the remaining flex pool (`EMISSION_FLEX_POOL`, 0.275) is split between the types by a participation-driven balancer (an oversubscribed type cedes flex toward a starved one — see `validator/scoring/emission_balance.py`). A type's base therefore always lies between its minimum and its minimum plus the whole flex pool, and text + quant + image always sum to exactly 1.0:

| Tournament type | Guaranteed minimum | Max weight (minimum + flex) |
| --- | --- | --- |
| Text | 0.15 | 0.425 |
| Quant (compression) | 0.35 | 0.625 |
| Image | 0.225 | 0.50 |

Each pool pays its own type's latest completed tournament, until a newer tournament of that type completes. Text and quant each carry their own boss, advanced only by tournaments of their own type.

The minimums apply at all times. A type with no winner yet pays nobody, and its share goes to the burn address.

Active tournament participants receive `0.0001` weight each.

Within a tournament, up to the top three ranks are paid (`TOURNAMENT_PAID_RANKS`), distributed by exponential decay using `TOURNAMENT_SIMPLE_DECAY_BASE = 0.25` — roughly a 76/19/5 split across three paid ranks, or 80/20 if a clean 3rd place can't be identified (see "Placements" below).

Champions can earn boosted tournament allocation when boss-round performance exceeds the `0.10` performance threshold (`EMISSION_MULTIPLIER_THRESHOLD`). The excess is multiplied by `2.0` (`EMISSION_MULTIPLIER_RATE`) and capped by tournament type. A champion's weight does not decay over time — a long-reigning champion earns the same as a freshly-crowned one.

### Placements

1st is the champion: the challenger if it was crowned, otherwise the boss. With a crown the boss is 2nd; without one the challenger is 2nd. A tournament pays a 3rd place only when it can be identified unambiguously:

- **Text**: the next miner after the challenger in round 3's standings (`validator/tournament/round_results.py: get_seeded_text_runner_up`).
- **Quant**: the next miner after the challenger in the cumulative standings after round 3 (`validator/tournament/round_results.py: get_quant_round_runner_up`). When the round-3 viability cut leaves the challenger alone, it is the best of the miners that advanced from round 2 on their round 1+2 standings, re-ranked among them.
- **Image**: the loser of the last single-pair knockout round before the boss round.

If no valid 3rd place exists (a tie at the cutoff or another degenerate case), the tournament pays only the top two ranks, renormalized.

The places are the same whether the boss round was played and lost or not played (early stop).

In text and quant, when there is no crown, each paid contender (2nd, 3rd) is compared with the boss on average rank over the rounds of 1-3 in which both are ranked (rounds 1 and 2 for a quant 3rd place taken from them). In these ranks an exact score tie, and every entry with no valid result, share one rank. If the boss's average is strictly better, that place's share is paid like the boss's own share; a tie or a better contender is paid, and so is a contender whose ranks cannot be computed. For example, if the boss outranks the 3rd-place miner but not the 2nd, 2nd is paid and 3rd's share goes to the boss. The comparison is fixed when the tournament completes, appears in the published audit data, and the tournament details endpoint applies it as the weight setter does. The boss's share, and every share paid like it, goes to the defending champion's hotkey; it is burned only when no defending champion hotkey is known or that hotkey is not registered. See [Crown, early stop and placements](../validator/tournament/README.md#crown-early-stop-and-placements).

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
