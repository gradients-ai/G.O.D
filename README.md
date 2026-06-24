<h1 align="center">G.O.D Subnet</h1>

Welcome to the [Gradients on Demand](https://gradients.io) subnet.

G.O.D is the subnet runtime behind Gradients.io training jobs and tournaments. Validators create tasks, coordinate miner and trainer infrastructure, evaluate results, and set weights. Miners expose training repositories for tournament tasks; trainers run those repositories on validator-controlled GPU infrastructure.

## Tournaments

Tournaments are recurring competitions where miners submit open-source training code. The validator asks each miner for a repository and exact commit, validates the repository, runs the code on dedicated trainers, evaluates the resulting models, and advances winners through tournament rounds.

Tournament types are scheduled independently:

| Type | Scheduled start | Task family | Participation fee | Minimum miners |
| --- | --- | --- | --- | --- |
| Environment | Monday 11:00 UTC | Environment interaction and PvP tasks | `0.25 TAO` | 5 |
| Text | Monday 13:00 UTC | Instruct, Chat, DPO, and GRPO tasks | `0.25 TAO` | 8 |
| Image | Monday 15:00 UTC | Diffusion/image tasks | `0.20 TAO` | 8 |

Scheduling notes:

- The scheduler creates a new pending tournament only when there is no pending or active tournament of the same type.
- Follow-up tournaments are created only during the configured UTC start hour. If the window is missed, the next opportunity is the following week's window.
- Tournament length is round- and task-dependent rather than a fixed 4-7 day duration. Pending tournaments collect and validate participants, deduct fees, and activate after the minimum miner count is met; active tournaments advance as training and evaluation rounds finish.
- Text and image tournaments use group, knockout/final, and boss/champion rounds. Environment tournaments use PvP-style evaluation and environment-specific boss comparisons.
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

To run a non-submitted model, run:

```bash
python -m ops.validator_ops.run_evaluation --task_id <task_id> --models <model_name>
```
