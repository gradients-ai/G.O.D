# Miner hub

Everything a miner needs that is the same in every tournament: schedule, fees, repository rules, how your code runs, entry, payouts, results and fair play.

Guides: [Text](miner_text.md) · [Image](miner_image.md) · [Compression](miner_compression.md) · [Developer guide](developer.md)

On this page: [Tournaments](#tournaments) · [Training repository](#training-repository) · [How your code runs](#how-your-code-runs) · [Local testing](#local-testing) · [Entering a tournament](#entering-a-tournament) · [The boss and the crown](#the-boss-and-the-crown) · [Payouts](#payouts) · [Results](#results) · [Fair play and review](#fair-play-and-review) · [Troubleshooting](#troubleshooting)

## What miners do

Miners submit code. You publish a GitHub repository, pin a commit, and run a small miner service that tells the validator which repository to use. Trainers build and run it on validator-managed GPUs, and the output is evaluated against the other miners.

> [!IMPORTANT]
> Repositories are published in the cases listed in [Fair play and review](#fair-play-and-review). Submit only code you are prepared to open-source.

Path to your first entry:

1. Pick a tournament in [Tournaments](#tournaments) and read its guide.
2. Build a repository that meets the [shared rules](#training-repository) and the guide's contract.
3. Test it locally (see [Local testing](#local-testing)).
4. Complete subnet registration, post your IP and start the miner service (see [Entering a tournament](#entering-a-tournament)).
5. Return your repository and full commit SHA from the endpoint for that tournament.
6. Fund your coldkey balance with at least the entry fee.

## Tournaments

Three tournaments run every week. Times are UTC.

| Tournament | API value | Endpoint on your miner | Weekly start | Entry fee | Guide |
| --- | --- | --- | --- | --- | --- |
| Text | `text` | `GET /training_repo/text` | Monday 09:00 | `0.7 TAO` | [Text guide](miner_text.md) |
| Image | `image` | `GET /training_repo/image` | Monday 11:00 | `0.4 TAO` | [Image guide](miner_image.md) |
| Compression | `quant` | `GET /training_repo/quant` | Monday 13:00 | `0.6 TAO` | [Compression guide](miner_compression.md) |

Each tournament is entered on its endpoint and charges one fee per entry. Entering two tournaments is two entries and two fees.

## Training repository

Every repository needs the following, whatever the tournament.

| Requirement | Rule |
| --- | --- |
| Licence | A `LICENSE` and a `NOTICE` file at the repository root, identical to this repository's. |
| Readable source | No obfuscated or compiled-only code. The scan runs at the pinned commit. |
| Dockerfile | Under `ops/docker/`. The same file name under `dockerfiles/` is also accepted. File names are in the guides. |
| CUDA | Trainer hosts support CUDA 12.8 or lower. Keep your base image and wheels within it. |
| Contents | What an image and repository may contain is policy: see [Fair play and review](#fair-play-and-review). |

The licence check reads the default branch for text and image, and the pinned commit for compression. Keep both files on the default branch and on the commit you pin.

<details>
<summary>Licence check details</summary>

Accepted licence file names: `LICENSE.md`, `LICENSE`, `license.md`, `license`, `License.md`, `License`.

Accepted notice file names: `NOTICE`, `NOTICE.txt`, `notice.txt`, `Notice.txt`, `notice`, `Notice`.

Each file is compared line by line with this repository's `LICENSE.md` and `NOTICE`, ignoring trailing whitespace. Any other difference fails.

</details>

## How your code runs

For each task the trainer does this:

1. Clones your repository at the pinned commit.
2. Downloads the task's model and data into the cache volume.
3. Builds the tournament's Dockerfile with the repository root as build context.
4. Starts your container with the task's CLI arguments and environment.
5. Waits until the container exits or the deadline passes.
6. Uploads your output to Hugging Face for evaluation.

The held-out split used for scoring is never given to your container.

### Container limits

| Limit | Value |
| --- | --- |
| Image build | No layer cache, 30 minute limit. |
| GPUs | Only the GPUs assigned to the task. Counts are in the guides. |
| Memory | 110 GB per GPU. |
| CPU | 24 cores per GPU, capped at the host's cores. |
| Shared memory | 8 GB, or 16 GB with four or more GPUs. |
| Network | An internal Docker bridge with no internet access. |
| Privileges | `no-new-privileges`, all Linux capabilities dropped. |
| `/cache` | Read-only. |
| `/app/checkpoints` | Read-write. |

### Arguments and environment

Containers receive these arguments. The guides list the rest.

| Argument | Tournaments | Meaning |
| --- | --- | --- |
| `--task-id` | All | The task's identifier. |
| `--model` | All | The model to work on, as a name under `/cache/models`. |
| `--expected-repo-name` | All | The name of your output directory. |
| `--hours-to-complete` | All | The deadline in hours, counted from container start. |
| `--dataset` | Text, compression | Presigned URL of the train split. The file is already at `/cache/datasets/{task_id}_train_data.json`. |
| `--file-format` | Text, compression | `s3`. |

| Variable | Tournaments | Meaning |
| --- | --- | --- |
| `BASELINE_STATS_PATH` | All | Path to a JSON file of model and dataset statistics. Set when the task has them. |
| `MINER_DATASETS_DIR` | Text, compression | Parent directory of your requested datasets. Set when you have any. |
| `MINER_DATASETS` | Text, compression | Comma-separated directory names under `MINER_DATASETS_DIR`. |

### Paths

| Purpose | Path |
| --- | --- |
| Your output | `/app/checkpoints/{task_id}/{expected_repo_name}` |
| The model | `/cache/models/{--model with "/" replaced by "--"}` |
| Requested datasets | `/cache/miner_datasets/{dataset id with "/" replaced by "--"}` |

The uploader reads the output path exactly. Task data paths are in the guides. `trainer/training_paths.py` has helpers for all of them.

### Deadline

| Outcome | Result |
| --- | --- |
| Container exits with code 0 | The output directory is uploaded. |
| Container exits with any other code | The task fails for you. Nothing is uploaded. |
| Deadline passes | The container is killed and the output directory is uploaded as it stands. |

### Upload

| Tournament | What is uploaded |
| --- | --- |
| Text, image | The output directory. If a direct subfolder holds a `.safetensors` file, that subfolder is uploaded in place of the directory. |
| Compression | `quant_manifest.json` and the files it lists, unmodified, once the output passes the check in its [output contract](miner_compression.md#output-contract). |

For text and image these files are not uploaded: `trainer_state.json`, `training_args.bin`, `optimizer.pt`, `optimizer.bin`, `scheduler.pt`, `scaler.pt`, `rng_state*.pth`.

Which evaluation failures count against you: [Whose fault an evaluation failure is](../validator/evaluation/README.md#whose-fault-an-evaluation-failure-is).

### Requested datasets

Text and compression entries can ask for extra data. Return up to two Hugging Face dataset ids in `requested_datasets`:

```python
requested_datasets=["OWNER/DATASET_A", "OWNER/DATASET_B"]
```

- Only ids listed in `core/datasets/whitelisted_sft_datasets.json` are accepted. Others are dropped.
- The first two accepted ids are kept.

## Local testing

Example scripts are in `ops/examples/training`. Each guide says what exists for its tournament and where to run it: [text](miner_text.md#local-testing), [image](miner_image.md#local-testing), [compression](miner_compression.md#local-testing).

To re-evaluate a finished task on your machine, see [Evaluations and manual tools](developer.md#evaluations-and-manual-tools).

## Entering a tournament

Two steps have similar names. Subnet registration puts your hotkey on the subnet, once. Enrolment adds you to one tournament, every week.

### Subnet registration and the miner service

Register your hotkey on the subnet:

```bash
btcli s register --netuid 56                   # mainnet
btcli s register --network test --netuid 241   # testnet
```

Post your miner IP to the metagraph. The miner service listens on port `7999`:

```bash
fiber-post-ip \
  --netuid 56 \
  --subtensor.network finney \
  --external_port 7999 \
  --wallet.name default \
  --wallet.hotkey default \
  --external_ip YOUR_PUBLIC_IP
```

Install the service from this repository:

```bash
git clone https://github.com/gradients-ai/G.O.D.git
cd G.O.D
sudo -E ./bootstrap.sh   # installs system dependencies and go-task
. ~/.bashrc
task install
task miner-config   # writes .1.env
```

Set your answer in `miner/endpoints/training_repo.py` (next section), then start the service:

```bash
task miner
```

Do this well before the weekly start. The validator asks only miners it already knows from the metagraph.

### The endpoint response

At enrolment the validator calls `GET /training_repo/{task_type}` on your miner. You control the answer in `miner/endpoints/training_repo.py`:

```python
async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/YOUR_USERNAME/YOUR_REPO",
        commit_hash="0123456789abcdef0123456789abcdef01234567",
        github_token=None,
        requested_datasets=None,
    )
```

| Field | Rule |
| --- | --- |
| `github_repo` | Repository URL. Must stay reachable until the tournament completes. |
| `commit_hash` | Full 40-character hex SHA (`git rev-parse HEAD`). A branch name is rejected. |
| `github_token` | Optional. Needed only for a private repository. |
| `requested_datasets` | Optional. See [Requested datasets](#requested-datasets). |

You may return a different repository or commit for each `task_type`. A miner that does not answer for a tournament is not enrolled in it.

<details>
<summary>Private repositories</summary>

Return a GitHub token that can read the repository. A fine-grained personal access token limited to that repository with `Contents: Read-only` is enough.

The validator tests the token against the GitHub repository API. A token that does not grant access is ignored, so the private repository then fails to clone.

</details>

### Enrolment checks

Enrolment checks the commit SHA, the obfuscation scan and the licence files ([Training repository](#training-repository)). Compression adds checks on the decoder: see its [enrolment checks](miner_compression.md#enrolment-checks).

Enrolment does not require a Dockerfile. A missing one fails when the trainer builds your image.

### One entry per miner

One entry is kept per IP address, per GitHub account and per coldkey, checked in that order. Among duplicate entries, one with a working GitHub token is preferred, then the first one seen.

Duplicate entries are dropped before the enrolment checks run. The kept entry must then pass them.

### Balance and fee

- Fees are paid from a balance held per coldkey. All hotkeys under a coldkey share it.
- The fee is charged when you are enrolled, after your repository passes the checks. Enrolment and the charge happen together.
- With too little balance you are not enrolled and nothing is charged.
- A coldkey is charged at most once per tournament.

Top up by transferring TAO from your coldkey to the collection address:

```text
5Ef5JgNv14LY4UEQFHbRQkf8TnegDV3AfAbcsJe5T2w6VQdo
```

Check the fees and your balance with the endpoints in [Results](#results).

### Scheduling

| Rule | Value |
| --- | --- |
| Minimum field | 4 enrolled miners. A pending tournament retries enrolment every 30 minutes until it has them. |
| One at a time | A tournament is created only when none of its kind is pending or active. |
| Start window | Creation happens during the tournament's start hour. If the hour is missed, the next chance is the following week. |
| First tournament | Created at once when none of its kind has ever run. |
| Completion target | Friday 14:00 UTC of the week it started. Operators complete a tournament by hand. |

## The boss and the crown

The boss is a tournament's defending champion, and the challenger is the one miner left after the rounds. The challenger is compared with the boss task by task and must beat it to take the crown.

The rule is in [Crown, early stop and placements](../validator/tournament/README.md#crown-early-stop-and-placements). What counts as a win on each tournament's tasks is in its guide: [text](miner_text.md#the-crown), [image](miner_image.md#the-crown), [compression](miner_compression.md#the-crown).

## Payouts

Each tournament pays the top places of its latest completed run until the next run of its kind completes.

### Share per tournament

| Tournament | Base minimum | Maximum |
| --- | --- | --- |
| Text | 0.15 | 0.40 |
| Image | 0.25 | 0.50 |
| Compression | 0.35 | 0.60 |

The minimums sum to 0.75. The other 0.25 floats between the tournaments by participation. A maximum is the minimum plus the whole 0.25, and it caps the share with or without a boost. How the floating share is computed: [Tournament emission split](../validator/scoring/README.md#tournament-emission-split).

### Split over the paid places

| Paid places | 1st | 2nd | 3rd |
| --- | --- | --- | --- |
| Three | 76.2% | 19.0% | 4.8% |
| Two | 80% | 20% | |

Up to three places are paid, each a quarter of the one above. Who holds each place is in [Crown, early stop and placements](../validator/tournament/README.md#crown-early-stop-and-placements). When the boss keeps the crown of a text or compression tournament, the [per-contender rule](../validator/tournament/README.md#per-contender-rule) decides whether 2nd and 3rd are paid.

### Other rules

| Rule | Value |
| --- | --- |
| Participation weight | `0.0001` to each miner not yet eliminated from an active tournament. The shares scale down by the total. |
| Champion boost | Text and image only. A champion's winning margin above `0.10` adds twice the excess to its tournament's share, up to the maximum. All three shares are then rescaled to sum to 1. |
| No winner | A tournament with no completed run has no payee. Its share is not paid out. |

While a boost is active, another tournament can fall under its base minimum.

## Results

| Endpoint | Returns |
| --- | --- |
| `GET /tournament/fees` | Entry fee per tournament. |
| `GET /tournament/balance/{coldkey}` | Your coldkey balance. |
| `GET /tournament/latest/details` | Rounds, tasks and scores of the latest tournaments. |
| `GET /tournament/{tournament_id}/details` | The same for one tournament. |
| `GET /v1/performance/latest-tournament-weights` | The weights from the latest tournaments. |

The API base is `https://api.gradients.io`. None of these needs a key.

## Fair play and review

These rules apply to every tournament. Compression adds rules for the decoder in its [guide](miner_compression.md#freeform-track-artifact-and-decoder).

### Duplicate-submission check

A duplicate submission is code that matches another entry's.

| Catches | When | Result |
| --- | --- | --- |
| The identical commit to another miner's. | Before round 1 trains. | Every entry in the matching group is eliminated automatically. |
| Source identical to another miner's after normalisation. | Before round 1 trains. | Every entry in the matching group is eliminated automatically. |
| A pairwise review by a language model of what differs, among the miners entering round 2 and against the boss's code. | When round 1 completes. | A flag holds the tournament until an operator decides. |

The boss is never eliminated by this check.

### Code review before round 2

Every miner entering round 2 is checked once by an automated code review when round 1 completes. The boss is not reviewed.

A flag, or a review that fails to run, holds the tournament until an operator decides. The operator either eliminates the miner or lets it play.

A miner eliminated at the round-2 duplicate review or by the code review is replaced by the next miner in the round 1 standings. The replacement goes through both checks.

### Publication

| Repository | Published at [github.com/gradients-opensource](https://github.com/gradients-opensource) |
| --- | --- |
| The winner's | When the tournament completes. |
| One confirmed as a duplicate submission at the round-2 review | When the operator approves the flag. |

### Policy

These lines are tournament policy. The automated checks are a floor, not the full definition of what is allowed.

- Given data only. Use the task's model and data, and for text and compression your [requested datasets](#requested-datasets). No outside copies of a task's dataset or its test rows.
- Produced inside the job. No datasets, pretrained models, pre-computed weights or private artifacts in your repository or image.
- Readable code. Nothing meant to hide how the method works.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Not enrolled | A branch name returned as `commit_hash`, a failed licence or obfuscation check, too little balance, or a duplicate entry by IP, GitHub account or coldkey. |
| Clone fails | The private repository's token expired or cannot read the repository. |
| Build fails | A Dockerfile the tournament needs is missing, or the build passed its time limit. Image needs two: see [Dockerfiles](miner_image.md#dockerfiles). |
| Download fails inside the container | The container has no internet. Read the task's files from `/cache`. |
| Write fails | The path is under `/cache`, which is read-only. Write under `/app/checkpoints`. |
| Nothing to evaluate | The output is not at the [output path](#paths), or nothing was there at the deadline. |
| Eliminated as a duplicate submission | The code matched another entry. See [Duplicate-submission check](#duplicate-submission-check). |
| Need the training logs | Grafana: <http://185.141.218.122:3001/> |
