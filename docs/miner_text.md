# Text tournament: miner guide

The text tournament ranks miners on how well their training code fine-tunes a given language model in a fixed time. This guide is the contract for miners who enter it.

[Miner hub](miner.md) · [Image guide](miner_image.md) · [Compression guide](miner_compression.md)

[At a glance](#at-a-glance) · [Rounds](#rounds) · [The boss round](#the-boss-round) · [What your code receives](#what-your-code-receives) · [Output contract](#output-contract) · [Scoring](#scoring) · [The crown](#the-crown) · [Local testing](#local-testing) · [Checklist](#checklist)

Schedule, fee and endpoint are in [Tournaments](miner.md#tournaments). Entry rules are in [Entering a tournament](miner.md#entering-a-tournament).

## At a glance

| | |
| --- | --- |
| You compete on | fine-tuning a given model on a given dataset, within the task's time budget |
| You submit | a training repository. Its container writes one model per task |
| Dockerfile | the trainer builds `ops/docker/standalone-text-trainer.dockerfile` from your repository |
| Enrolment | the shared checks. `LICENSE` and `NOTICE` are read on your repository's default branch |
| Rounds | three, one instruct task each |
| Boss round | four tasks: continuous, GRPO, DPO, big instruct |
| Task score | loss on held-out data, lower is better. On GRPO, reward, higher is better |

## Rounds

Each round is one task. Every miner in the round, and the boss, trains the same model on the same data.

| Round | Task | Model size | Advance |
| --- | --- | --- | --- |
| R1 | instruct | 1-4B | top 8 |
| R2 | instruct | 1-4B | top 4 |
| R3 | instruct | 4-8B | top 1 |

Source: [`TEXT_ROUND_PLAN`](../validator/tournament/constants.py).

- Models are drawn from the standard pool, the validator's list of supported language models, within the round's size band, among models that can also be compressed.
- Standings come from the round's task alone, ordered by [task score](#scoring).
- With fewer miners than slots, all advance.
- A miner with no valid result ranks behind every miner with one.

The miner that advances from round 3 is the challenger.

## The boss round

The challenger and the boss meet on four tasks.

The continuous task belongs to a lineage that carries across tournaments. There is one lineage, `qwen3-8b`, which starts from `Qwen/Qwen3-8B-Base`. Each tournament trains the lineage's current model on the lineage's next data. The lowest-loss model on the task becomes the lineage's model for the following tournament ([`CONTINUOUS_SFT_LINEAGES`](../validator/tournament/constants.py)).

| Task | Task type | Model | Trained in the boss round by |
| --- | --- | --- | --- |
| Continuous | `ChatTask` | the lineage's current model | the challenger |
| GRPO | `GrpoTask` | standard pool | both |
| DPO | `DpoTask` | standard pool | both |
| Big instruct | `InstructTextTask` | 30-72B | both |

Source: [`TEXT_BOSS_ROUND_STAGE_ONE_TASK_TYPES`, `BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B`, `BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B`](../validator/tournament/constants.py), [`task_creator.py`](../validator/tournament/task_creator.py).

- The boss trains its round 1-3 tasks and the continuous task when the tournament starts.
- The GRPO and DPO tasks are created with the boss round.
- The big instruct task runs last. It is created once the other three are scored, and only if the challenger can then take the [crown](miner.md#the-boss-and-the-crown) by winning it.

<details>
<summary>When another model is carried</summary>

- If the boss round is not played, or the challenger is disqualified, the boss's model is carried.
- With no scored model, the lineage keeps its current one.

</details>

## What your code receives

Every task runs one Docker image, built from `ops/docker/standalone-text-trainer.dockerfile` in your repository at your pinned commit ([`TEXT_DOCKERFILE_PATHS`](../core/constants/docker.py)). The shared container contract (limits, no network, the shared arguments and variables, paths, the deadline) is in [How your code runs](miner.md#how-your-code-runs). This section covers what is specific to text.

### Arguments

| Argument | Meaning |
| --- | --- |
| `--task-type` | `InstructTextTask`, `ChatTask`, `DpoTask` or `GrpoTask` |
| `--dataset` | presigned URL of the train split. The file is already at `/cache/datasets/{task_id}_train_data.json` |
| `--file-format` | `s3` |
| `--dataset-type` | JSON that names the dataset's columns, per task type below |

Source: [`trainer/runtime.py: run_trainer_container_text`](../trainer/runtime.py), [`trainer/training_paths.py`](../trainer/training_paths.py).

- The train split is one JSON file of rows.
- Budgets are between 0.25 and 5 hours and vary by task. A task of rounds 1-3 gets at most 2.5 hours.

### Dataset type

| Task type | Fields in `--dataset-type` |
| --- | --- |
| `InstructTextTask` | `field_instruction`, `field_input`, `field_output`, `field_system`, `system_prompt`, `system_format`, `format`, `no_input_format`, `field` |
| `ChatTask` | `chat_template`, `chat_column`, `chat_role_field`, `chat_content_field`, `chat_user_reference`, `chat_assistant_reference` |
| `DpoTask` | `field_prompt`, `field_system`, `field_chosen`, `field_rejected`, `prompt_format`, `chosen_format`, `rejected_format` |
| `GrpoTask` | `field_prompt`, `reward_functions`, `extra_column` |

Source: [`core/models/dataset_models.py`](../core/models/dataset_models.py), [`core/models/reward_models.py`](../core/models/reward_models.py).

- A field the dataset does not set keeps its default from `dataset_models.py`. That default is `null` for most fields.
- Each entry of `reward_functions` carries `reward_func`, the Python source of one reward function, and its `reward_weight`, plus metadata fields (`reward_id`, `func_hash`, `is_generic`, `is_manual`).
- The continuous task is a `ChatTask`. Its `--model` is the lineage's current model.

### Environment variables

| Variable | Set on | Meaning |
| --- | --- | --- |
| `USE_KL` | KL-regularised instruct tasks | `1`. Scoring adds a KL term to the loss (see [Scoring](#scoring)) |
| `KL_COEF` | with `USE_KL` | the coefficient of that KL term |
| `WANDB_MODE` | every task | `offline` |
| `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_ARTIFACT_DIR`, `WANDB_DATA_DIR`, `WANDB_CONFIG_DIR` | every task | `/app/checkpoints/wandb_logs/{task_id}_{hotkey}` |

The tasks of rounds 1-3 carry no KL term. The big instruct task may carry one.

### GPUs

The GPU count follows the task's effective size: the model's parameter count, times 3 on DPO, times 2 on GRPO, and times 2 again on a KL-regularised task.

| Effective size | GPUs |
| --- | --- |
| up to 4B | 1 |
| above 4B, up to 12B | 2 |
| above 12B, up to 40B | 4 |
| above 40B | 8 |
| the continuous task, any size | 4 |

Source: [`get_tournament_gpu_requirement`](../validator/tournament/gpu_requirements.py).

## Output contract

Write the model to your output directory ([Paths](miner.md#paths)). The trainer uploads it as follows.

| Upload rule | Detail |
| --- | --- |
| What is uploaded | the output directory. If a direct subfolder holds a `.safetensors` file, that subfolder is uploaded in its place |
| Not uploaded | at the top level of what is uploaded: `trainer_state.json`, `training_args.bin`, `optimizer.pt`, `optimizer.bin`, `scheduler.pt`, `scaler.pt`, `rng_state*.pth` |

Source: [`trainer/containers/uploader.py`](../trainer/containers/uploader.py).

| Rule | Detail |
| --- | --- |
| Form | a full model, or a LoRA adapter with its `adapter_config.json` |
| Loading | with `trust_remote_code` off |
| Architecture | a full model keeps the task model's architecture fields |
| Context window | a full model's `max_position_embeddings` is not below the task model's |

Source: [`model_is_a_finetune`](../validator/evaluation/model_checks.py).

A full model that fails the architecture or context-window rule is not a fine-tune of the task model. It gets no valid result.

<details>
<summary>Architecture fields compared</summary>

`architectures`, `hidden_size`, `n_layer`, `intermediate_size`, `head_dim`, `hidden_act`, `model_type`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`. A field is compared when the task model's config sets it.

</details>

## Scoring

Each model is scored on the task's held-out split. Evaluation uses the task model's tokenizer.

| Task type | Score | Better |
| --- | --- | --- |
| `InstructTextTask` | cross-entropy loss on the output tokens | lower |
| `InstructTextTask` with `USE_KL` | that loss plus `KL_COEF` x the KL divergence of your model from the task model, averaged over the same tokens | lower |
| `ChatTask` | cross-entropy loss on the assistant turns | lower |
| `DpoTask` | DPO loss over the held-out pairs | lower |
| `GrpoTask` | the weighted sum of the reward functions' mean rewards on generated completions, minus the evaluation loss | higher |

Source: [`validator/evaluation/evaluators/`](../validator/evaluation/evaluators/), [`calculate_miner_ranking_and_scores`](../validator/scoring/tasks.py).

- Entries on a task are ranked by this score.
- An entry with no model, a failed evaluation, or a model that is not a fine-tune has no valid result.
- On the continuous task the tokenizer and chat template are those of `Qwen/Qwen3-8B-Base`.
- Which evaluation failures are scored and which are retried is in [Whose fault an evaluation failure is](../validator/evaluation/README.md#whose-fault-an-evaluation-failure-is).

## The crown

The crown rule is in [The boss and the crown](miner.md#the-boss-and-the-crown). This table says what a win is on a text task and which value the two sides are compared on.

| Task | Compared on | Win |
| --- | --- | --- |
| Instruct, continuous, DPO | the [task score](#scoring) | a score at least 1% better, and a pass on the [paired per-example test](../validator/tournament/README.md#the-paired-per-example-test) against the boss |
| GRPO | the [task score](#scoring) | a score at least 1% better |

- The paired test only decides whether a result is a win. A result that is not a win is placed on the score alone, whether the test was failed or the per-example losses could not be paired.

Source: [`TEXT_CROWN_WIN_MARGIN`](../validator/tournament/constants.py), [`crown.py: text_head_to_head`](../validator/tournament/crown.py).

## Local testing

Three scripts in `ops/examples/training` run one task each through the downloader, your trainer image and the uploader:

```bash
./ops/examples/training/run_instruct_task.sh
./ops/examples/training/run_dpo_task.sh
./ops/examples/training/run_grpo_task.sh
```

Each script builds `ops/docker/standalone-text-trainer.dockerfile` from the current directory. This repository does not contain that file: it comes from your repository. The script also builds the downloader and uploader images, which this repository does contain. Run it from the root of a checkout of this repository with your repository's files copied in. The script does not look under `dockerfiles/`. Set the task values and credentials at the top of the script first.

## Checklist

- [ ] Your repository holds `ops/docker/standalone-text-trainer.dockerfile` at your pinned commit.
- [ ] The entrypoint accepts the shared arguments and the four text arguments.
- [ ] All four task types are handled: `InstructTextTask`, `ChatTask`, `DpoTask`, `GrpoTask`.
- [ ] The train split is read from `/cache/datasets/{task_id}_train_data.json`, with no download.
- [ ] The model is in your output directory before the deadline.
- [ ] A full model keeps the task model's architecture and context window.
- [ ] The repository meets the shared rules in [Training repository](miner.md#training-repository) and [Fair play and review](miner.md#fair-play-and-review).
