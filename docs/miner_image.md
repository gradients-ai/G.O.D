# Image tournament: miner guide

The image tournament ranks miners on how well their training code fits an image model to a small captioned image set in a fixed time. This guide is the contract for miners who enter it.

[Miner hub](miner.md) · [Text guide](miner_text.md) · [Compression guide](miner_compression.md)

[At a glance](#at-a-glance) · [Rounds](#rounds) · [The boss round](#the-boss-round) · [What your code receives](#what-your-code-receives) · [Output contract](#output-contract) · [Scoring](#scoring) · [The crown](#the-crown) · [Local testing](#local-testing) · [Checklist](#checklist)

Schedule, fee and endpoint are in [Tournaments](miner.md#tournaments). Entry rules are in [Entering a tournament](miner.md#entering-a-tournament). Rules below cite the code that enforces them. Where this guide and the code differ, the code is right.

## At a glance

| | |
| --- | --- |
| You compete on | training a LoRA for a given image model on a given set of captioned images, within the task's time budget |
| You submit | a training repository. Its container writes one LoRA per task |
| Dockerfiles | two: `ops/docker/standalone-image-trainer.dockerfile` for Flux tasks, `ops/docker/standalone-image-toolkit-trainer.dockerfile` for every other family |
| Model families | Flux, Z-Image, Qwen-Image, Ideogram 4, Krea 2 |
| Rounds | two: a three-task group round, then a one-task knockout |
| Boss round | six tasks |
| Task score | denoising loss on held-out images, lower is better |

## Rounds

Round 1 is three tasks for every miner. Round 2 is one task between the two miners that advance.

| Round | Format | Tasks | Model family | Advance |
| --- | --- | --- | --- | --- |
| R1 | one group, every miner | 3 | one Flux, one Ideogram 4, one Z-Image | top 2 |
| R2 | knockout, one pair | 1 | Qwen-Image or Krea 2, drawn at random | the winner |

Source: [`ROUND_ONE_IMAGE_MODEL_TYPES`, `ROUND_TWO_IMAGE_MODEL_TYPES`, `SMALL_TOURNAMENT_ADVANCE`](../validator/tournament/constants.py), [`task_creator.py`](../validator/tournament/task_creator.py).

Round 1 standings use average rank over the three tasks ([`round_results.py`](../validator/tournament/round_results.py)):

- On each task, miners are ranked by [task score](#scoring).
- A miner with no valid result on a task ranks one place below that task's last ranked miner.
- A miner missing a result on any counted task sorts below every miner with a result on each of them.
- A tie on average rank goes to the lower summed score, then to the hotkey that sorts first.

In round 2 the lower score wins. The winner is the challenger.

<details>
<summary>Edge cases</summary>

- If only one miner comes out of round 1, it is the challenger and round 2 is not played.
- A round-1 task on which nobody has a valid result is left out of the average.

</details>

## The boss round

The challenger and the boss both train six tasks.

| Model family | Tasks |
| --- | --- |
| Krea 2 | 2 |
| Ideogram 4 | 2 |
| Qwen-Image | 1 |
| Z-Image | 1 |

Source: [`FINAL_ROUND_IMAGE_TASK_DISTRIBUTION`](../validator/tournament/constants.py).

All six tasks are created with the round. Each task draws a base model and an image set.

## What your code receives

Each task runs one image, built at your pinned commit from the Dockerfile for the task's model family. The shared container contract (limits, no network, the shared arguments and variables, paths, the deadline) is in [How your code runs](miner.md#how-your-code-runs). This section covers what is specific to image.

### Dockerfiles

| Model family | `--model-type` | Dockerfile |
| --- | --- | --- |
| Flux | `flux` | `ops/docker/standalone-image-trainer.dockerfile` |
| Z-Image | `z-image` | `ops/docker/standalone-image-toolkit-trainer.dockerfile` |
| Qwen-Image | `qwen-image` | same as Z-Image |
| Ideogram 4 | `ideogram4` | same as Z-Image |
| Krea 2 | `krea2` | same as Z-Image |

Source: [`get_dockerfile_path`](../trainer/runtime.py), [`trainer/constants.py`](../trainer/constants.py).

> [!IMPORTANT]
> An entry needs both Dockerfiles. Round 1 has a Flux task, and every later task uses the toolkit Dockerfile.

### Arguments

| Argument | Meaning |
| --- | --- |
| `--model-type` | the model family: `flux`, `z-image`, `qwen-image`, `ideogram4` or `krea2` |
| `--dataset-zip` | presigned URL of the training zip. The file is already at `/cache/datasets/{task_id}_tourn.zip` |
| `--trigger-word` | the task's trigger word. Passed only when the task has one |

Source: [`trainer/runtime.py: run_trainer_container_image`](../trainer/runtime.py), [`trainer/training_paths.py`](../trainer/training_paths.py).

### The dataset

| Property | Value |
| --- | --- |
| Contents | images, each with a caption in a `.txt` file of the same name |
| Image formats | `.png`, `.jpg`, `.jpeg` |
| Layout | all files inside one folder, `train_data/` |
| Task size | 10 to 50 pairs. 10 to 30 in round 1 |
| Held out | a tenth of the pairs, rounded up. They are not in the zip |

Source: [`train_test_split_image`](../validator/tasks/datasets/preparation.py), [`validator/tasks/synthetics/constants.py`](../validator/tasks/synthetics/constants.py).

A task is created with a budget of 0.5 to 1 hour, by image count, plus 0.5 hour on Qwen-Image. `--hours-to-complete` carries the budget your container gets.

### Environment variables

| Variable | Set on | Meaning |
| --- | --- | --- |
| `HF_HOME`, `TRANSFORMERS_CACHE` | every task | `/cache/hf_cache` |
| `FLUX_T5_PATH` | Flux tasks | `/cache/hf_cache/t5xxl_fp16.safetensors` |
| `FLUX_CLIP_PATH` | Flux tasks | `/cache/hf_cache/clip_l.safetensors` |
| `FLUX_VAE_PATH` | Flux tasks | `/cache/hf_cache/ae.safetensors` |

The variables shared by all tournaments are in [How your code runs](miner.md#how-your-code-runs).

<details>
<summary>Supporting files in the cache</summary>

The base model is at the model path in [Paths](miner.md#paths). A Flux model published as a single checkpoint is a directory holding that one `.safetensors` file. Other files the families need are under `/cache/hf_cache`:

| Model family | Files |
| --- | --- |
| Every family | tokenizer files of `openai/clip-vit-large-patch14`, `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` and `google/t5-v1_1-xxl`, in Hugging Face cache layout |
| Flux, single checkpoint | the three files named by the `FLUX_*` variables |
| Z-Image | `zimage_turbo_training_adapter_v2.safetensors` |
| Qwen-Image | `qwen_image_torchao_uint3.safetensors` |
| Ideogram 4 | `Qwen--Qwen3-VL-8B-Instruct/` (text encoder), `ideogram_4_unconditional_lora_r16.safetensors` |
| Krea 2 | `Qwen--Qwen3-VL-4B-Instruct/` (text encoder) |

Source: [`trainer/containers/downloader.py`](../trainer/containers/downloader.py).

</details>

### GPUs

Every image task trains on one H100 ([`get_tournament_gpu_requirement`](../validator/tournament/gpu_requirements.py)).

## Output contract

Write the LoRA to your output directory ([Paths](miner.md#paths)). The trainer uploads your output to a public Hugging Face repo, under `checkpoints/`, by the rule in [Upload](miner.md#upload).

The evaluator loads one `.safetensors` file from `checkpoints/`, chosen in this order:

1. `last.safetensors`, if exactly one file has that name.
2. The file with the highest number at the end of its name, as in `name-000500.safetensors` or `name_500.safetensors`.
3. The only `.safetensors` file.

| Rule | Detail |
| --- | --- |
| Form | a LoRA in `.safetensors`, applied to the task's base model |
| No match | no `.safetensors` file under `checkpoints/`, or several and none matches the order above: the evaluation fails |

Source: [`select_lora`](../validator/evaluation/image_artifacts.py).

## Scoring

Your LoRA is applied to the task's base model and scored on the held-out images ([`evaluators/diffusion.py`](../validator/evaluation/evaluators/diffusion.py)).

| Step | Rule |
| --- | --- |
| Per image | the model's denoising error (mean squared error of its prediction) over a fixed set of noise levels |
| Two passes | once with the image's caption, once with an empty prompt |
| Task score | 0.5 x the mean captioned loss + 0.5 x the mean empty-prompt loss |
| Better | lower |

Source: [`DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT`](../validator/evaluation/constants.py), [`_calculate_weighted_loss_for_image_eval`](../validator/scoring/tasks.py).

- Entries on a task are ranked by this score.
- An entry with no upload or a failed evaluation has no valid result.
- Which evaluation failures are scored and which are retried is in [Whose fault an evaluation failure is](../validator/evaluation/README.md#whose-fault-an-evaluation-failure-is).

## The crown

The crown goes to the challenger when it beats the boss on the six boss-round tasks, under the rule in this table. Placements are in [Crown, early stop and placements](../validator/tournament/README.md#crown-early-stop-and-placements).

| Rule | Value |
| --- | --- |
| Win on a task | a score at least 1% better than the boss's |
| Crown | at most one decided task not won, and at least one win. With six decided, five wins |
| Otherwise | the boss keeps the crown |

Source: [`BOSS_ROUND_WIN_MARGIN`](../validator/tournament/constants.py), [`determine_boss_round_winner`](../validator/tournament/round_results.py).

The 1% is relative to the boss's score. All six tasks are played.

A task with a missing result goes as follows.

| Case | Task goes to |
| --- | --- |
| You have a valid result, the boss has none | you |
| The boss has a valid result, you have none | the boss |
| Both trainings failed | the boss |
| Both trained, neither has an evaluation result | the boss |

## Local testing

One script in `ops/examples/training` runs an image task through the downloader, your trainer image and the uploader:

```bash
./ops/examples/training/run_image_task.sh
```

The script builds `ops/docker/standalone-image-toolkit-trainer.dockerfile` from the current directory. It also builds the downloader and uploader images from `ops/docker/` in the same tree. Run it from the root of a checkout of this repository with your repository's files copied in, so that `ops/docker/` holds your Dockerfiles. Set the task values and credentials at the top of the script first.

## Checklist

- [ ] Both Dockerfiles exist at your pinned commit.
- [ ] The entrypoint of each accepts the shared arguments and the three image arguments, with `--trigger-word` optional.
- [ ] Every model family is handled: Flux, Z-Image, Qwen-Image, Ideogram 4, Krea 2.
- [ ] The training zip is read from `/cache/datasets/{task_id}_tourn.zip`, with no download.
- [ ] The LoRA is in your output directory before the deadline.
- [ ] The evaluator's file choice in [Output contract](#output-contract) lands on the file you intend.
- [ ] The repository meets the shared rules in [Training repository](miner.md#training-repository) and [Fair play and review](miner.md#fair-play-and-review).
