# Tournament Miner Documentation 🏆

This guide covers everything you need to know about creating and running a training repository for G.O.D (Gradient-On-Demand) tournaments.

## Overview

The tournament system builds and runs your training code in Docker containers. Your code will compete against other miners by training models on provided datasets within time and resource constraints.

## Prerequisites for Tournament Participation

To compete in tournaments, miners must meet the following requirements:

1. **Subnet Registration**: You must be registered on the G.O.D subnet (netuid 56 on mainnet, 241 on testnet)

   - Register using: `btcli s register` (mainnet) or `btcli s register --network test` (testnet)
   - Post your IP to the metagraph using fiber: `fiber-post-ip --netuid 56 --subtensor.network finney --external_port 7999 --wallet.name default --wallet.hotkey default --external_ip [YOUR-IP]`

2. **Active Miner**: Your miner must be running and listening for training repository requests

   - The miner must expose the `/training_repo/{task_type}` endpoint that returns your training repository details
   - Start your miner with: `task miner`

3. **Sufficient Balance** The tournaments require a small fee to participate. All collected fees are burned

   - The current fee for participanting is 0.2 TAO for text tournaments, 0.15 TAO for image tournaments, and 0.20 TAO for environment tournaments. You can always check the current fee using the api endpoint:
     `curl https://api.gradients.io/tournament/fees`
   - To get balance you need to transfer TAO from your coldkey to the collection address: `5Ef5JgNv14LY4UEQFHbRQkf8TnegDV3AfAbcsJe5T2w6VQdo`
   - The balance is credited per coldkey, so all miner hotkeys with the same coldkey will have a shared balance
   - To get current balance for a coldkey you can use the API:
     `curl 'https://api.gradients.io/tournament/balance/{coldkey}'`
   - All collected TAO will be staked for alpha and the **alpha burned**

4. **Valid License Files**: Your training repository must include verbatim LICENSE and NOTICE files

   - The repository must contain a LICENSE file (accepted names: `LICENSE.md`, `LICENSE`, `license.md`, `license`, `License.md`, or `License`) that matches exactly the LICENSE file from the G.O.D validator repository
   - The repository must contain a `NOTICE` file that matches exactly the NOTICE file from the G.O.D validator repository
   - These files are validated verbatim (character-for-character match) during tournament registration
   - Repositories without matching LICENSE and NOTICE files will be excluded from tournament participation
   - You can find the required LICENSE and NOTICE files in the [G.O.D repository](https://github.com/rayonlabs/G.O.D)

5. **No Code Obfuscation**: Your training repository must not contain obfuscated code

   - All code in your repository must be readable and not obfuscated
   - The validator automatically checks repositories for obfuscation during tournament registration
   - examples of obfuscated code files are: .bin, .pyc, .dll and similar files containing machine code or anything resembling that.
   - Repositories with obfuscated code will be excluded from tournament participation
   - This ensures transparency and allows the community to learn from winning implementations

## Initial Miner Setup

Before you can participate in tournaments, you need to configure your miner with the required credentials and settings.

### Running the Configuration Script

The repository includes a configuration generator to help you set up your miner:

```bash
python core/create_config.py --miner
```

This interactive script will prompt you for:

- **Wallet name** - Your Bittensor wallet name (default: "default")
- **Hotkey name** - Your hotkey name (default: "default")
- **Subtensor network** - "finney" for mainnet (netuid 56) or "test" for testnet (netuid 241)
- **Minimum validator stake threshold** - Default: 1000 for mainnet, 0 for testnet. Requests to your miner from nodes with stake less than this will be blocked to protect you and reduce spam.

The script will generate a `.1.env` file with your configuration.

### Starting Your Miner

After configuration, start your miner:

```bash
task miner
```

Your miner will:

- Start listening for training repository requests from validators
- Respond to tournament participation queries

## Choosing Your Training Repository

You have two options for your tournament training repository:

### Option 1: Start from the Base Miner (Recommended for Beginners)

Use the official G.O.D repository as your starting point:

- **GitHub Repository**: `https://github.com/rayonlabs/G.O.D`
- **Commit Hash**: Use `"main"` for the latest version, or a specific commit hash for consistency

The base repository includes functional training scripts that you can modify and improve.

### Option 2: Fork a Previous Winner's Repository

Browse winning tournament implementations at the [Gradients Open Source Organization](https://github.com/orgs/gradients-opensource/repositories)

These repositories contain proven AutoML techniques that won previous tournaments. You can:

- Study their approaches to understand what works
- Fork and improve upon winning strategies
- Combine techniques from multiple winners

## Tournament Registration

Configure the `/training_repo/{task_type}` endpoint in your miner to point to your chosen repository.

**Location**: `miner/endpoints/tuning.py` (lines 10-13)

Update the `get_training_repo()` function:

```python
async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/YOUR_USERNAME/YOUR_REPO",  # Your repo URL
        commit_hash="YOUR_COMMIT_HASH"  # Specific commit or "main"
    )
```

**Example configurations:**

```python
# Using the base miner
github_repo="https://github.com/rayonlabs/G.O.D"
commit_hash="main"

# Using your own fork
github_repo="https://github.com/yourname/my-training-repo"
commit_hash="a1b2c3d4e5f6..."

# Using a previous winner's approach
github_repo="https://github.com/gradients-opensource/position-1-tournament-xyz"
commit_hash="main"
```

Where `task_type` can be:

- `"text"` - For text-based tournaments (Instruct, DPO, GRPO, Chat)
- `"image"` - For image-based tournaments (SDXL, Flux)
- `"environment"` - For environment-based tournaments (e.g., AlfWorld)

**Important**: The repository and commit hash you configure will be used by validators to build and run your training code during tournaments.

## Docker-Based Architecture

### Recommended Base Images

**For Text Tasks (Instruct, DPO, GRPO, Chat):**

```dockerfile
FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1
```

**For Image Tasks (SDXL, Flux):**

```dockerfile
FROM diagonalge/kohya_latest:latest
```

### Required Repository Structure

```
your-training-repo/
├── dockerfiles/
│   ├── standalone-text-trainer.dockerfile
│   └── standalone-image-trainer.dockerfile
```

**Important:** The dockerfile paths must be exactly:

- `dockerfiles/standalone-text-trainer.dockerfile`
- `dockerfiles/standalone-image-trainer.dockerfile`

## CLI Arguments

Your training scripts accept these standardised CLI arguments:

### Text Training Arguments

```bash
--task-id             # Unique task identifier
--model               # Base model to finetune
--dataset             # S3 dataset URL
--dataset-type        # JSON structure of dataset (columns, format)
--task-type           # "InstructTextTask", "DpoTask", "ChatTask", "GrpoTask", or "EnvTask"
--expected-repo-name  # Expected HuggingFace repository name for upload
--hours-to-complete   # Time limit in hours for the job to finish
```

**Note:** For GRPO tasks, the reward functions used for training can be found in [`core/manual_reward_funcs.py`](https://github.com/rayonlabs/G.O.D/blob/main/core/manual_reward_funcs.py).

**Note:** For Environment tasks (EnvTask), you must implement rollout functions. See the [Environment Tasks documentation](environment_tasks.md) for details.

### Image Training Arguments

```bash
--task-id             # Unique task identifier
--model               # Base model to finetune (e.g., stabilityai/stable-diffusion-xl-base-1.0)
--dataset-zip         # S3 URL to dataset zip file
--model-type          # "sdxl" or "flux"
--expected-repo-name  # Expected HuggingFace repository name for upload
--hours-to-complete   # Time limit in hours for the job to finish
```

## Training Logs and Monitoring

### Grafana Dashboard

View real-time training logs and metrics at: http://185.141.218.59:3001/d/training-runs/training-runs-dashboard

### Tournament Results

After tournaments complete, view detailed results and rankings at: https://gradients.io/app/research/tournament/{TOURNAMENT_ID}

Replace `{TOURNAMENT_ID}` with the specific tournament ID you participated in.

## WandB Logging for Your Training Analysis

Include WandB logging so you can analyze your training runs after tournaments complete:

```python
def create_config(task_id, model, dataset, dataset_type, file_format, output_dir, expected_repo_name=None, log_wandb=True):
    if log_wandb:
        config["wandb_runid"] = f"{task_id}_{expected_repo_name}"
        config["wandb_name"] = f"{task_id}_{expected_repo_name}"
        config["wandb_mode"] = "offline"  # Logs saved locally
        os.makedirs(train_cst.WANDB_LOGS_DIR, exist_ok=True)

def patch_wandb_symlinks(base_dir: str):
    """Handle WandB symlinks by converting to real files."""
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)
            if os.path.islink(full_path):
                target_path = os.readlink(full_path)
                try:
                    os.unlink(full_path)
                    if os.path.exists(target_path):
                        shutil.copy(target_path, full_path)
                    else:
                        pathlib.Path(full_path).touch()
                except Exception as e:
                    print(f"Symlink patch failed: {e}")

# Call after training completes
patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)
```

## Dataset Handling

### Text Datasets

- Always provided as S3 URLs
- Format: JSON
- Dataset type parameter describes the structure (columns, format)

### Image Datasets

- Provided as S3 URLs to zip files
- Should contain images and metadata (captions)
- Your script must handle extraction and preparation

## Output Structure Requirements

**Critical:** The output paths are standardised and MUST NOT be changed. The uploader expects models at these exact locations.

For your reference, all the paths used in training can be found at:

`trainer/constants.py`

And the functions to construct the paths can be found at:

`trainer/utils/training_paths.py`

Here are some most important paths:

### Model Output Path

```python
output_dir = f"/app/checkpoints/{task_id}/{expected_repo_name}"
```

### Input Model Cache Path

```python
# Models are pre-downloaded to this location by the downloader container
model_path = f"/cache/models/{model.replace('/', '--')}"
```

### Image Dataset Path

```python
# Datasets are pre-downloaded to this location by the downloader container
model_path = f"/cache/datasets/{task_id}_tourn.zip"
```

### Text Dataset Path

```python
# Datasets are pre-downloaded to this location by the downloader container
model_path = f"/cache/datasets/{task_id}_train_data.json"
```

## Utility Functions in Trainer Scripts

### Text Trainer

```python
def patch_wandb_symlinks(base_dir:str)
```

Fixes the local wandb logs that are later synced to cloud. Offline saves are prone to broken files and symlinks, causing issues while syncing. This function patches those files, which has to be done in the training context.

```python
def patch_model_metadata(output_dir: str, base_model_id: str)
```

Huggingface verifies the base model id when a finetune is uploaded. That gets broken at times due to the nature of our training with localized paths and separate uploads. This function patches the model metadata to deal with that, fixes the model name back to the original huggingface link.

## Example Entrypoint Script

Create a simple entrypoint that passes arguments to your training script:

```bash
#!/bin/bash
set -e

# For text training
python3 /workspace/scripts/text_trainer.py "$@"

# For image training
python3 /workspace/scripts/image_trainer.py "$@"
```

## Testing Your Setup

Test scripts are provided to validate your implementation locally:

```bash
# Text task examples
./examples/run_instruct_task.sh
./examples/run_dpo_task.sh
./examples/run_grpo_task.sh

# Image task examples
./examples/run_image_task.sh
```

## Tournament Structure

Tournaments run continuously with 4-7 day duration and 72-hour gaps between tournaments. There are separate tournaments for:

- **Text**: Instruct, DPO, GRPO tasks
- **Image**: SDXL and Flux diffusion tasks
- **Environment**: Environment-based reinforcement learning tasks (e.g., AlfWorld)

### Group Stage

**Text/Image Tournaments:**
- Miners are organized into groups of 6-8 participants
- Each group competes on 3 tasks (text tournaments: 1 Instruct task; image tournaments: 1 image task; total of 3 required)
- Top 8 performers overall across all groups advance to knockout rounds

**Environment Tournaments:**
- All participants (including boss) compete in a single large group
- Minimum 5 participants required to start
- All participants compete on the same environment task
- Only one round (group stage) - no knockout rounds
- One winner selected based on highest GRPO score

### Knockout Rounds

**Text/Image Tournaments:**
- Single elimination format
- Runs when field is reduced to 14 or fewer miners
- Head-to-head competition

**Environment Tournaments:**
- No knockout rounds - tournament ends after group stage

### Boss Round

**Text/Image Tournaments:**
- Tournament winner must face defending champion
- Uses progressive threshold system with exponential decay based on consecutive wins
- **Winning Requirements**: Challenger wins by **majority rule** (4+ out of 6 tasks) for **both text and image tournaments**
- Defending champion retains title unless challenger wins the majority of tasks

**Environment Tournaments:**
- Boss (defending champion) competes directly in the single group stage
- Uses progressive threshold system: challengers must beat `boss_score * (1 + threshold_percentage)` to be eligible
- Winner is the participant with the highest GRPO score among eligible participants
- If no challenger beats the threshold, boss retains title automatically

#### Championship Defense Thresholds

The advantage required to dethrone a champion decreases with each successful defense using an exponential decay formula:

**Formula:** `threshold = max(EXPONENTIAL_MIN_THRESHOLD, EXPONENTIAL_BASE_THRESHOLD × EXPONENTIAL_DECAY_RATE^(consecutive_wins - 1))`

**Implementation:** See `validator/tournament/utils.py:83-88` - `get_progressive_threshold()`

**Constants** (defined in `validator/tournament/constants.py:63-65`):

- `EXPONENTIAL_BASE_THRESHOLD = 0.10` - Starting threshold (10%) for champions on their first defense
- `EXPONENTIAL_DECAY_RATE = 0.8` - Decay factor applied per consecutive win
- `EXPONENTIAL_MIN_THRESHOLD = 0.03` - Minimum threshold floor (3%)

**Example Thresholds:**

- 1st consecutive win: 10.0% threshold
- 2nd consecutive win: 8.0% threshold
- 3rd consecutive win: 6.4% threshold
- 4th consecutive win: 5.1% threshold
- 5th consecutive win: 4.1% threshold
- 6th+ consecutive wins: 3.0% threshold (floor)

This system ensures:

- New champions start with a significant 10% advantage on their first defense
- Long-reigning champions become progressively more vulnerable as the threshold decays
- Minimum 3% threshold floor prevents complete stagnation while still rewarding excellence

### GPU Requirements

- Determined by model size and task type
- Resource limits are enforced (memory, CPU)
- Plan for efficient resource usage

## Common Pitfalls to Avoid

1. **Don't change output paths** - The uploader expects exact locations
2. **Don't hardcode paths** - Use provided constants
3. **Don't ignore time limits** - Respect the hours-to-complete parameter
4. **Don't skip validation** - Test with various model sizes and datasets
5. **Don't upload/download in the training script** - Training container is run with no access to internet or host machine

## Environment Tournaments

Environment tournaments are a specialized tournament type focused on reinforcement learning with environment interactions.

### Key Differences

- **Single Round Structure**: Only one group stage round (no knockout rounds)
- **All Participants Together**: Boss and all participants compete in a single large group on the same task
- **Minimum Participants**: Requires at least 5 participants (vs 8 for text/image tournaments)
- **Participation Fee**: 0.20 TAO per tournament
- **Schedule**: Starts every Monday at 14:00 UTC

### Scoring System

- **GRPO-Based**: Uses Group Relative Policy Optimization scoring where **higher scores are better**
- **Progressive Threshold**: Defending champion benefits from progressive threshold (same system as text/image tournaments)
  - Challengers must beat: `boss_score * (1 + threshold_percentage)` to be eligible
  - If no challenger beats threshold, boss automatically wins
- **Winner Selection**: Highest GRPO score among eligible participants wins

### Technical Requirements

- **Rollout Functions**: Must implement custom rollout functions for environment interaction
- **Environment Servers**: Environment servers are provisioned during training (typically one per GPU)
- **Environment Variable**: Server URLs provided via `ENVIRONMENT_SERVER_URLS` environment variable
- **Evaluation**: Post-training evaluation uses 250 episodes to determine final scores

For detailed information on implementing environment tasks, see the [Environment Tasks documentation](environment_tasks.md).

## Reference Implementation

The G.O.D repository provides base training scripts that you can customize:

- `scripts/text_trainer.py` - Base implementation for text tasks (including environment tasks)
- `scripts/image_trainer.py` - Base implementation for image tasks

These scripts handle all required functionality including dataset preparation, training configuration, and model saving. You can modify and enhance these scripts to improve performance.
