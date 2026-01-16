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

The system automatically schedules tournaments on a weekly basis with separate schedules for each type:

- **Duration**: 4-7 days per tournament
- **Tournament Types**:
  - `TournamentType.ENVIRONMENT` - Starts every Monday at 14:00 UTC
  - `TournamentType.TEXT` - Starts every Thursday at 14:00 UTC
  - `TournamentType.IMAGE` - Starts every Thursday at 14:00 UTC
- **Auto-Creation**: `process_tournament_scheduling()` creates new tournaments when previous ones complete and the scheduled time arrives
- **Missed Windows**: If a scheduled start time is missed (e.g., system was down), the tournament will NOT start late - it waits until the next scheduled occurrence


## Tournament Lifecycle

### 1. Tournament Creation (`TournamentStatus.PENDING`)

- System creates basic tournament with `generate_tournament_id()`
- Adds base contestant (defending champion) if available
- Begins participant registration process

### 2. Participant Registration

- System pings miners via `/training_repo/{task_type}` endpoint
- All responses are checked for obfuscation and for sufficient tournament fee balance
- Requires minimum `MIN_MINERS_FOR_TOURN = 8` to proceed (text/image tournaments)
- Requires minimum `MIN_MINERS_FOR_ENV_TOURN = 5` to proceed (environment tournaments)

### 3. Tournament Activation (`TournamentStatus.ACTIVE`)

- First round created with `_create_first_round()`
- Round structure varies by tournament type:
  - **Text/Image Tournaments**:
    - **Group Stage**: All miners compete in groups of 6-8 participants
    - **Top 8 advance**: The top 8 performers from the group stage advance to knockout rounds
    - **Knockout Stage**: Single elimination format for the top 8
  - **Environment Tournaments**:
    - **Single Group Stage**: All participants (including boss) compete in one large group
    - **No Knockout Rounds**: Environment tournaments only have the group stage
    - **One Winner**: The highest-scoring participant wins the tournament

### 4. Task Creation & Assignment

#### Group Stage Tasks

- **Text tournaments**: 1 Instruct task
- **Image tournaments**: 1 image task
- **Environment tournaments**: 1 environment task (all participants compete on the same task)

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

**Environment Tasks:**

- Environment tasks use the same GPU allocation logic as GRPO tasks (2x multiplier)
- GPU requirements determined by model size with `TOURNAMENT_GRPO_GPU_MULTIPLIER` applied

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

## Environment Tournament Mechanics

Environment tournaments have a simplified structure focused on environment-based reinforcement learning:

### Structure

- **Single Round**: Only one group stage round (no knockout rounds)
- **All Participants Together**: Boss and all participants compete in a single large group
- **Minimum Participants**: Requires at least 5 participants to start
- **Participation Fee**: 0.20 TAO per tournament

### Scoring System

- **GRPO-Based Scoring**: Uses GRPO (Group Relative Policy Optimization) scoring where **higher scores are better**
- **Progressive Threshold**: Defending champion benefits from a progressive threshold system
  - Champion starts with 10% advantage on first defense
  - Threshold decays exponentially with consecutive wins (same formula as text/image tournaments)
  - Minimum threshold floor of 3%
- **Winner Selection**: The participant with the highest GRPO score wins
  - Challengers must beat the boss threshold score to be eligible: `challenger_score >= boss_score * (1 + threshold_percentage)`
  - If no challenger beats the threshold, boss retains title
  - Among eligible participants, highest score wins

### Task Assignment

- All participants (including boss) are assigned to the same environment task
- Task uses environment rollout functions (e.g., `alfworld_rollout_first_prompt_and_completion`)
- Environment servers are provisioned during training
- Evaluation: 250 episodes post-training to determine final scores

## Scoring & Weight Distribution

Tournament results feed into exponential weight decay system:

- Round winners get `round_number * type_weight` points
- Type weights defined by `TOURNAMENT_TEXT_WEIGHT`, `TOURNAMENT_IMAGE_WEIGHT`, and `TOURNAMENT_ENVIRONMENT_WEIGHT` constants
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
