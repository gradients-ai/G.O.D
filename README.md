<h1 align="center">G.O.D Subnet</h1>

🚀 Welcome to the [Gradients on Demand](https://gradients.io) Subnet

> Distributed intelligence for LLM and diffusion model training. Where the world's best AutoML minds compete.

**Tournaments** 🏆
Competitive events where the validator executes miners' open-source training scripts on dedicated infrastructure.

- **Duration**: 4-7 days per tournament
- **Frequency**: New tournaments start 72 hours after the previous one ends
- **Rewards**: Exponentially higher weight potential for top performers
- **Open Source**: Winning AutoML scripts are released when tournaments complete
- **Winners Repository**: First place tournament scripts is uploaded to [github.com/gradients-opensource](https://github.com/gradients-opensource) 🤙

## Documentation

- [Developer Guide](docs/developer.md): repo layout, setup, validator/trainer/miner operations, tests, and common development workflows.
- [Tournament Miner Guide](docs/miner.md): miner participation requirements, training repository contract, tournament types, and scoring expectations.

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
