# Text tournament: miner guide

The text tournament ranks miners on how well their training code fine-tunes a given language model in a fixed time. This guide is the contract for miners who enter it.

[Miner hub](miner.md) · [Image guide](miner_image.md) · [Compression guide](miner_compression.md)

[At a glance](#at-a-glance) · [Rounds](#rounds) · [The boss round](#the-boss-round) · [What your code receives](#what-your-code-receives) · [Output contract](#output-contract) · [Scoring](#scoring) · [The crown](#the-crown) · [Local testing](#local-testing) · [Checklist](#checklist)

Schedule, fee and endpoint are in [Tournaments](miner.md#tournaments). Entry rules are in [Entering a tournament](miner.md#entering-a-tournament). Rules below cite the code that enforces them. Where this guide and the code differ, the code is right.

## At a glance

| | |
| --- | --- |
| You compete on | fine-tuning a given model on a given dataset, within the task's time budget |
| You submit | a training repository. Its container writes one model per task |
| Dockerfile | `ops/docker/standalone-text-trainer.dockerfile` |
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

- Models are drawn from the standard pool, the validator's list of supported language models, within the round's size band and limited to models the compression tournament supports.
- Standings come from the round's task alone, ordered by [task score](#scoring).
- A miner with no valid result ranks behind every miner with one.

The miner that advances from round 3 is the challenger.

## The boss round

The challenger and the boss meet on four tasks.

| Task | Task type | Model | Trained in the boss round by |
| --- | --- | --- | --- |
| Continuous | `ChatTask` | the lineage's current model | the challenger |
| GRPO | `GrpoTask` | standard pool | both |
| DPO | `DpoTask` | standard pool | both |
| Big instruct | `InstructTextTask` | 30-72B | both |

Source: [`TEXT_BOSS_ROUND_STAGE_ONE_TASK_TYPES`, `BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B`, `BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B`](../validator/tournament/constants.py), [`task_creator.py`](../validator/tournament/task_creator.py).

- The boss trains its round 1-3 tasks and the continuous task when the tournament starts.
- The GRPO and DPO tasks are created with the boss round.
- The big instruct task runs last. It is created once the other three are scored, if the crown is in reach ([early stop](../validator/tournament/README.md#crown-early-stop-and-placements)).

The continuous task belongs to a lineage that carries across tournaments. There is one lineage, `qwen3-8b`, which starts from `Qwen/Qwen3-8B-Base`. Each tournament trains the lineage's current model on the lineage's next data. The lowest-loss model on the task becomes the lineage's model for the following tournament ([`CONTINUOUS_SFT_LINEAGES`](../validator/tournament/constants.py)).

<details>
<summary>When another model is carried</summary>

- If the boss round is not played, or the challenger is disqualified, the boss's model is carried.
- With no scored model, the lineage keeps its current one.

</details>

## What your code receives

Every task runs one image, built from your text Dockerfile at your pinned commit. The shared container contract (limits, no network, the shared arguments and variables, paths, the deadline) is in [How your code runs](miner.md#how-your-code-runs). This section covers what is specific to text.

| Argument | Meaning |
| --- | --- |
| `--task-type` | `InstructTextTask`, `ChatTask`, `DpoTask` or `GrpoTask` |
| `--dataset-type` | JSON that names the dataset's columns, per task type below |

Source: [`trainer/runtime.py: run_trainer_container_text`](../trainer/runtime.py), [`trainer/training_paths.py`](../trainer/training_paths.py).

The train split named by `--dataset` is one JSON file of rows.

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

The tasks of rounds 1-3 carry no KL term. The big instruct task may carry one. The variables shared by all tournaments are in [How your code runs](miner.md#how-your-code-runs).

### GPUs

Every task trains on H100s. The count follows the task's effective size: the model's parameter count, times 3 on DPO, times 2 on GRPO, and times 2 again on a KL-regularised task.

| Effective size | GPUs |
| --- | --- |
| up to 4B | 1 |
| above 4B, up to 12B | 2 |
| above 12B, up to 40B | 4 |
| above 40B | 8 |
| the continuous task, any size | 4 |

Source: [`get_tournament_gpu_requirement`](../validator/tournament/gpu_requirements.py).

## Output contract

Write the model to your output directory ([Paths](miner.md#paths)). The trainer uploads it to a public Hugging Face repo, where it is evaluated.

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
| `GrpoTask` | the sum over the reward functions of each function's mean reward times its weight, on generated completions, less the evaluation loss | higher |

Source: [`validator/evaluation/evaluators/`](../validator/evaluation/evaluators/), [`calculate_miner_ranking_and_scores`](../validator/scoring/tasks.py).

- Entries on a task are ranked by this score.
- An entry with no model, a failed evaluation, or a model that is not a fine-tune has no valid result.
- On the continuous task the tokenizer and chat template are those of `Qwen/Qwen3-8B-Base`.
- Which evaluation failures are scored and which are retried is in [Whose fault an evaluation failure is](../validator/evaluation/README.md#whose-fault-an-evaluation-failure-is).

## The crown

The crown goes to the challenger when it beats the boss under the rule in [Crown, early stop and placements](../validator/tournament/README.md#crown-early-stop-and-placements). Each head-to-head is a win or a non-win, and a non-win is within the tight allowance, within the looser allowance, or beyond both. This table says what each is on a text task.

| Task | Win | Tight allowance | Looser allowance |
| --- | --- | --- | --- |
| Instruct, continuous, DPO | passes the [paired per-example test](../validator/tournament/README.md#the-paired-per-example-test) against the boss, with a score at least 1% better | at most 0.1% behind the boss | at most 0.25% behind |
| GRPO | a score at least 1% better | at most 0.1% behind the boss | at most 0.25% behind |

- Percentages are relative to the boss's score.
- A non-win that is level with the boss or ahead of it is within the tight allowance.

Source: [`TEXT_CROWN_WIN_MARGIN`, `TEXT_CROWN_TIGHT_ALLOWANCE`, `TEXT_CROWN_LOOSER_ALLOWANCE`](../validator/tournament/constants.py), [`crown.py: text_head_to_head`](../validator/tournament/crown.py).

## Local testing

Three scripts in `ops/examples/training` run one task each through the downloader, your trainer image and the uploader:

```bash
./ops/examples/training/run_instruct_task.sh
./ops/examples/training/run_dpo_task.sh
./ops/examples/training/run_grpo_task.sh
```

Each script builds `ops/docker/standalone-text-trainer.dockerfile` from the current directory. It also builds the downloader and uploader images from `ops/docker/` in the same tree. Run it from the root of a checkout of this repository with your repository's files copied in, so that `ops/docker/` holds your Dockerfile. Set the task values and credentials at the top of the script first.

## Checklist

- [ ] `ops/docker/standalone-text-trainer.dockerfile` exists at your pinned commit.
- [ ] The entrypoint accepts the shared arguments and the two text arguments.
- [ ] All four task types are handled: `InstructTextTask`, `ChatTask`, `DpoTask`, `GrpoTask`.
- [ ] The train split is read from `/cache/datasets/{task_id}_train_data.json`, with no download.
- [ ] The model is in your output directory before the deadline.
- [ ] A full model keeps the task model's architecture and context window.
- [ ] The repository meets the shared rules in [Training repository](miner.md#training-repository) and [Fair play and review](miner.md#fair-play-and-review).
