<h1 align="center">G.O.D Subnet</h1>

Welcome to the [Gradients on Demand](https://gradients.io) subnet.

G.O.D is the subnet runtime behind Gradients.io training jobs and tournaments. Validators create tasks, coordinate miner and trainer infrastructure, evaluate results, and set weights. Miners expose training repositories for tournament tasks; trainers run those repositories on validator-controlled GPU infrastructure.

## Tournaments

Tournaments are recurring competitions where miners submit open-source training code. The validator asks each miner for a repository and exact commit, validates the repository, runs the code on dedicated trainers, evaluates the resulting models, and advances winners through tournament rounds.

Tournament types are scheduled independently:

| Type | Scheduled start | Task family | Participation fee | Minimum miners |
| --- | --- | --- | --- | --- |
| Quant | Monday 09:00 UTC | Model compression (freeform and fixed-format) of the models the text boss trained | `0.6 TAO` | 4 |
| Text | Monday 11:00 UTC | Instruct rounds; a boss round of continuous (chat), GRPO, DPO and a big instruct task | `0.7 TAO` | 4 |
| Image | Monday 13:00 UTC | Diffusion/image tasks | `0.4 TAO` | 4 |

Miners enter a type by answering `GET /training_repo/{text|quant|image}` and pay that type's fee; there are two champions (the text champion and the quant champion), and each type has its own emission pool (guaranteed minimums text 0.15, quant (compression) 0.35, image 0.225, plus a 0.275 flex pool split by participation). The minimums always apply: a type with no winner yet pays nobody and its share goes to the burn address. A quant tournament with no completed quant history takes its boss from the latest completed text tournament.

Environment tournaments are retired: they are not scheduled or emission-weighted. Historical
environment tournaments remain queryable through the tournament analytics endpoints.

Scheduling notes:

- The scheduler creates a new pending tournament only when there is no pending or active tournament of the same type. The first tournament of a type is created at once when none of that type exists; a quant tournament then waits, with nothing created, until an active text tournament has its boss tasks ready.
- Follow-up tournaments are created only during the configured UTC start hour. If the window is missed, the next opportunity is the following week's window.
- Tournament length depends on the rounds and tasks played. Pending tournaments collect and validate participants, deduct fees, and activate after the minimum miner count is met; active tournaments advance as training and evaluation rounds finish.
- Text tournaments use three single-group instruct rounds on standard-pool models (1-4B top 8, 1-4B top 4, 4-8B top 1) and a boss round of four tasks: continuous (the carried lineage), GRPO, DPO and a big instruct task (30-72B), which runs last and only while a crown is still possible. The boss trains its round 1-3 tasks and the continuous task at tournament start, so miners' round 1 runs in parallel. See the [Miner Guide](docs/miner.md#text-tournaments).
- Quant tournaments use three rounds on the text boss's round 1-3 models (freeform top 8, fixed `W2A16-asym-g64` top 4, freeform top 1) and a four-task boss round (freeform on the round-2 model, fixed 2-3 bit on the round-3 model, freeform and fixed 2-3 bit on the continuous model). See the [Quant Tournament Miner Guide](docs/miner_quant.md).
- In text and quant the boss's entries are ranked in rounds 1-3 (never taking an advancement slot), and the challenger is compared with the boss on 7 head-to-heads: rounds 1-3 plus the 4 boss-round tasks. It is crowned with at most two non-wins, every non-win within the looser allowance and, with two, one within the tight allowance. Every round runs in full, rounds 2 and 3 even with one miner left; after the round-3 cut, the boss round is not created if the last contender's round 1-3 head-to-heads already rule out the crown. Without a crown the boss keeps 1st, and a paid contender the boss outranks on average over rounds 1-3 is paid like the boss. Full rules: [Crown, early stop and placements](validator/tournament/README.md#crown-early-stop-and-placements).
- Image tournaments use a group round, a knockout round, and a boss round.
- Winning repositories are published to [github.com/gradients-opensource](https://github.com/gradients-opensource), and tournament performance feeds validator weight setting.

Useful public endpoints:

```bash
curl https://api.gradients.io/v1/tournaments/next-dates
curl https://api.gradients.io/v1/tournaments/active
curl https://api.gradients.io/v1/tournaments/latest/details
curl https://api.gradients.io/tournament/fees
```

## Documentation

- [Developer Guide](docs/developer.md): repo layout, setup, validator/trainer/miner operations, tests, and common development workflows.
- [Miner Guide](docs/miner.md): miner participation requirements, training repository contract, tournament types, and scoring expectations.
- [Quant Tournament Miner Guide](docs/miner_quant.md): the quant tournament's entry, rounds, tasks, scoring, the fixed checkpoint and freeform decoder contracts, and evaluation.

## Running Evaluations

You can re-evaluate existing tasks on your own machine. Or you can run non-submitted models to check if they are good.
This works for tasks not older than 7 days.

Make sure to build the latest docker images before running the evaluation.

```bash
docker build -f ops/docker/validator.dockerfile -t weightswandering/tuning_vali:latest .
docker build -f ops/docker/validator-diffusion.dockerfile -t diagonalge/tuning_validator_diffusion:latest .
```

To see the available options, run:

```bash
python -m ops.validator_ops.run_evaluation --help
```

To re-evaluate a task, run:

```bash
python -m ops.validator_ops.run_evaluation --task_id <task_id>
```

To re-evaluate a PvP environment task for selected hotkeys, run:

```bash
python -m ops.validator_ops.run_evaluation --task_id <task_id> --gpu_ids 0 1 --hotkeys <hotkey_a> <hotkey_b>
```

For continuation-round PvP (previous-round adapter as `base_chain`), run:

```bash
python -m ops.validator_ops.run_evaluation --task_id <task_id> --gpu_ids 0 1 \
  --hotkeys <hotkey_a> <hotkey_b> \
  --base_chain <hotkey_a>=<prev_repo_a> \
  --base_chain <hotkey_b>=<prev_repo_b>
```

To run a non-submitted model, run:

```bash
python -m ops.validator_ops.run_evaluation --task_id <task_id> --models <model_name>
```
