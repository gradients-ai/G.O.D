# Tournament Overview

## What Are Tournaments?

Tournaments are competitive training events where miners submit their open-source training repositories and validators execute them on standardized infrastructure. The system runs continuous cycles managed by three main processes:

- `process_pending_tournaments()` - Populates participants and activates tournaments
- `process_pending_rounds()` - Creates tasks and assigns nodes
- `process_active_tournaments()` - Advances tournaments through completion

Unlike real-time serving where miners train models on their own hardware for [Gradients.io](https://gradients.io) customers, tournaments focus purely on the quality of training methodologies.

## Why Tournaments Matter

- **🔍 Transparency**: Enterprise customers can see exactly how models are trained
- **⚡ Innovation**: Cutting-edge techniques are implemented within hours of publication
- **🏆 Competition**: Only the best AutoML approaches survive the tournament structure
- **📖 Open Source**: Winning methodologies become available to the entire AI community

## Tournament Schedule & Timing

Based on `TOURNAMENT_INTERVAL_HOURS = 72`, the system automatically schedules tournaments:

- **Duration**: 4-7 days per tournament
- **Gap Between Tournaments**: 72 hours after completion before next tournament starts
- **Tournament Types**: Text and Image tournaments run independently via `TournamentType.TEXT` and `TournamentType.IMAGE`
- **Auto-Creation**: `process_tournament_scheduling()` creates new tournaments when previous ones complete

## Tournament Lifecycle

### 1. Tournament Creation (`TournamentStatus.PENDING`)

- System creates basic tournament with `generate_tournament_id()`
- Adds base contestant (defending champion) if available
- Begins participant registration process

### 2. Participant Registration

- System pings miners via `/training_repo/{task_type}` endpoint
- All responses are checked for obfuscation and for sufficient tournament fee balance
- Requires minimum `MIN_MINERS_FOR_TOURN = 8` to proceed

### 3. Tournament Activation (`TournamentStatus.ACTIVE`)

- First round created with `_create_first_round()`
- Round structure:
  - **Group Stage**: All miners compete in one large group
  - **Top 8 advance**: The top 8 performers from the group stage advance to knockout rounds
  - **Knockout Stage**: Single elimination format for the top 8

### 4. Task Creation & Assignment

#### Group Stage Tasks

- **Text tournaments**: 1 Instruct task
- **Image tournaments**: 1 image task

#### Knockout Stage Tasks

- **Text tournaments**: 1 probabilistically selected task per pair (Instruct/DPO/GRPO)
- **Image tournaments**: 1 task per pair (SDXL or Flux)

#### Final Round Tasks

- **Text tournaments**: 2 of each type (Instruct + DPO + GRPO) with big models
- **Image tournaments**: 6 image tasks all assigned to same pair

Tasks assigned to trainer nodes via `assign_nodes_to_tournament_tasks()` with expected repo names: `tournament-{tournament_id}-{task_id}-{hotkey[:8]}`

## Tournament Compute Allocation

### GPU Requirements by Model Size

Dynamic allocation based on `get_tournament_gpu_requirement()`:

**Text Tasks:**

```python
params_b = model_params_count / 1_000_000_000

# Task type multipliers
if task_type == TaskType.DPOTASK:
    params_b *= TOURNAMENT_DPO_GPU_MULTIPLIER  # 3x
elif task_type == TaskType.GRPOTASK:
    params_b *= TOURNAMENT_GRPO_GPU_MULTIPLIER  # 2x

# GPU allocation thresholds
if params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100:  # 4.0B
    return GpuRequirement.H100_1X
elif params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100:  # 12.0B
    return GpuRequirement.H100_2X
elif params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100:  # 40.0B
    return GpuRequirement.H100_4X
else:
    return GpuRequirement.H100_8X
```

**Image Tasks:**

- All image tasks (SDXL, Flux) receive `GpuRequirement.A100`

### Trainer Node Execution

- Tournament orchestrator finds suitable GPUs via `_check_suitable_gpus()`
- Training executed using miner's Docker containers and scripts
- Memory limit: `DEFAULT_TRAINING_CONTAINER_MEM_LIMIT = "24g"`
- CPU limit: `DEFAULT_TRAINING_CONTAINER_NANO_CPUS = 8`
- Network isolation: `--network none` for security

## Round Management

### Round Status Flow

```
PENDING → ACTIVE → COMPLETED
```

### Task Execution Monitoring

- Progress tracked through `monitor_training_tasks()`
- Training status: `PENDING → TRAINING → SUCCESS/FAILURE`
- GPU availability updated when tasks complete
- Failed tasks moved back to `PENDING` for retry (up to `MAX_TRAINING_ATTEMPTS = 2`)

### Round Completion & Advancement

- System waits for all tasks to reach `TaskStatus.SUCCESS` or `TaskStatus.FAILURE`
- Winners calculated using tournament scoring system
- Losers eliminated
- Next round created with winners, or tournament completes

## Boss Round Mechanics

When tournament reaches final round with single winner:

1. **Historical Task Selection**: Boss round uses proven historical tasks from the database with at least 2 successful quality scores
   - Text tournaments: 2 of each type (InstructText, DPO, GRPO) = 6 total tasks
   - Image tournaments: 6 image tasks
   - Tasks are copied with new IDs while preserving original training data
2. **Score Comparison**: Tournament miners' results are compared against the best historical scores from general miners
3. **Winning Requirements**: Challenger wins by **majority rule** (4+ out of 6 tasks) for **both text and image tournaments**
4. **Champion Defense**: Previous winner retains title unless challenger wins the majority of tasks

## Scoring & Weight Distribution

Tournament results feed into exponential weight decay system:

- Round winners get `round_number * type_weight` points
- Type weights defined by `TOURNAMENT_TEXT_WEIGHT` and `TOURNAMENT_IMAGE_WEIGHT` constants
- Final weights calculated using `exponential_decline_mapping()` with `TOURNAMENT_WEIGHT_DECAY_RATE`
- Previous winners get special placement based on boss round performance

## Technical Integration Points

### For Miners

- Implement `/training_repo/{task_type}` endpoint
- Ensure training scripts accept standardized CLI arguments
- Include WandB logging for validator monitoring (`wandb_mode = "offline"`)
- Output models to exact paths: `/app/checkpoints/{task_id}/{expected_repo_name}`
- Handle all task types within your tournament category

## Viewing Tournament Results

After tournaments complete, view detailed results and rankings at: https://gradients.io/app/research/tournament/{TOURNAMENT_ID}

Replace `{TOURNAMENT_ID}` with the specific tournament ID.

## Getting Started

Ready to compete? Check out:

- [Tournament Miner Guide](tourn_miner.md) - Complete setup instructions and technical details
- [Example Scripts](../examples/) - Reference implementations for testing
