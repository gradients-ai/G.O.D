<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/header-image-dark.svg"><img src="assets/readme/header-image-light.svg" alt="Image tournament: miner guide" width="100%"></picture>
</p>

The image tournament ranks miners on how well their training code fits an image model to a small captioned image set in a fixed time. This guide is the contract for miners who enter it.

<p align="center">
  <a href="miner.md"><b>Miner hub</b></a> &nbsp;·&nbsp;
  <a href="miner_text.md"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/icon-text-dark.svg"><img src="assets/readme/icon-text-light.svg" alt="" width="20" align="top"></picture> <b>Text</b></a> &nbsp;·&nbsp;
  <picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/icon-image-dark.svg"><img src="assets/readme/icon-image-light.svg" alt="" width="20" align="top"></picture> <b>Image</b> &nbsp;·&nbsp;
  <a href="miner_compression.md"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/icon-compression-dark.svg"><img src="assets/readme/icon-compression-light.svg" alt="" width="20" align="top"></picture> <b>Compression</b></a>
</p>

[At a glance](#at-a-glance) · [Rounds](#rounds) · [The boss round](#the-boss-round) · [What your code receives](#what-your-code-receives) · [Output contract](#output-contract) · [Scoring](#scoring) · [The crown](#the-crown) · [Local testing](#local-testing) · [Checklist](#checklist)

Schedule, fee and endpoint are in [Tournaments](miner.md#tournaments). Entry rules are in [Entering a tournament](miner.md#entering-a-tournament).

## At a glance

| | |
| --- | --- |
| You compete on | training a LoRA for a given image model on a given set of captioned images, within the task's time budget |
| You submit | a training repository. Its container writes one LoRA per task |
| Dockerfiles | the trainer builds one of two from your repository: `ops/docker/standalone-image-trainer.dockerfile` for Flux tasks, `ops/docker/standalone-image-toolkit-trainer.dockerfile` for every other family |
| Enrolment | the shared checks. `LICENSE` and `NOTICE` are read on your repository's default branch |
| Model families | Flux, Z-Image, Qwen-Image, Ideogram 4, Krea 2 |
| Rounds | three, one task each |
| Boss round | four tasks |
| Task score | denoising loss on held-out images, lower is better |

## Rounds

Each round is one task. Every miner in the round, and the boss, trains the same model on the same images.

| Round | Task model family | Advance |
| --- | --- | --- |
| R1 | Flux | top 8 |
| R2 | Ideogram 4 | top 4 |
| R3 | Krea 2 | top 1 |

Source: [`IMAGE_ROUND_PLAN`](../validator/tournament/constants.py).

- Round 1 advances on its task, ordered by [task score](#scoring). Rounds 2 and 3 advance on average rank over every round played so far, ranked among the surviving miners.
- With fewer miners than slots, all advance.
- A miner with no valid result on a task ranks behind every miner with one on it. A miner missing a result in any round sorts below every miner with a result in all rounds.
- Ties on average rank go to the better rank in the most recent round, then to the smaller summed gap to each task's best entry, then to a seeded draw.
- The boss is ranked in each round but never takes an advancement slot.

The miner that advances from round 3 is the challenger.

## The boss round

The challenger and the boss both train four tasks.

| Model family | Tasks |
| --- | --- |
| Krea 2 | 1 |
| Ideogram 4 | 1 |
| Qwen-Image | 1 |
| Z-Image | 1 |

Source: [`IMAGE_BOSS_ROUND_TASK_DISTRIBUTION`](../validator/tournament/constants.py).

All four tasks are created with the round. Each task draws a base model and an image set.

## What your code receives

Each task runs one Docker image, built from your repository at your pinned commit, from the Dockerfile for the task's model family. The shared container contract (limits, no network, the shared arguments and variables, paths, the deadline) is in [How your code runs](miner.md#how-your-code-runs). This section covers what is specific to image.

### Dockerfiles

| Model family | `--model-type` | Dockerfile |
| --- | --- | --- |
| Flux | `flux` | `ops/docker/standalone-image-trainer.dockerfile` |
| Z-Image | `z-image` | `ops/docker/standalone-image-toolkit-trainer.dockerfile` |
| Qwen-Image | `qwen-image` | same as Z-Image |
| Ideogram 4 | `ideogram4` | same as Z-Image |
| Krea 2 | `krea2` | same as Z-Image |

Source: [`get_dockerfile_path`](../trainer/runtime.py), [`trainer/constants.py`](../trainer/constants.py).

Both files are looked up in your repository. This repository contains `standalone-image-toolkit-trainer.dockerfile` and does not contain `standalone-image-trainer.dockerfile`.

> [!IMPORTANT]
> An entry needs both Dockerfiles. Round 1 is a Flux task, so a failed Flux build means no round-1 result. Every other task uses the toolkit Dockerfile.

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
| Task size | 50 pairs |
| Held out | a tenth of the pairs, rounded up: 5 images |

Source: [`train_test_split_image`](../validator/tasks/datasets/preparation.py), [`MAX_IMAGE_SYNTH_PAIRS`](../validator/tasks/synthetics/constants.py).

A task's time budget is 1 hour, and 1.5 hours on Qwen-Image ([`_image_competition_hours_for_dataset_size`](../validator/tasks/synthetics/diffusion.py), [`QWEN_IMAGE_EXTRA_COMPETITION_HOURS`](../validator/tasks/synthetics/constants.py)).

### Environment variables

| Variable | Set on | Meaning |
| --- | --- | --- |
| `HF_HOME`, `TRANSFORMERS_CACHE` | every task | `/cache/hf_cache` |
| `FLUX_T5_PATH` | Flux tasks | `/cache/hf_cache/t5xxl_fp16.safetensors` |
| `FLUX_CLIP_PATH` | Flux tasks | `/cache/hf_cache/clip_l.safetensors` |
| `FLUX_VAE_PATH` | Flux tasks | `/cache/hf_cache/ae.safetensors` |

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

Every image task gets one GPU ([`get_tournament_gpu_requirement`](../validator/tournament/gpu_requirements.py)).

## Output contract

Write the LoRA to your output directory ([Paths](miner.md#paths)). The trainer uploads it under `checkpoints/` in the Hugging Face repo.

| Upload rule | Detail |
| --- | --- |
| What is uploaded | the output directory. If a direct subfolder holds a `.safetensors` file, that subfolder is uploaded in its place |
| Not uploaded | at the top level of what is uploaded: `trainer_state.json`, `training_args.bin`, `optimizer.pt`, `optimizer.bin`, `scheduler.pt`, `scaler.pt`, `rng_state*.pth` |

Source: [`trainer/containers/uploader.py`](../trainer/containers/uploader.py).

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

The crown rule is in [The boss and the crown](miner.md#the-boss-and-the-crown). This table says what a win is on an image task and where the other results fall. Percentages are relative to the boss's [task score](#scoring), and lower is better.

| Outcome | Task score |
| --- | --- |
| Win | at least 1% lower than the boss's, or a valid result where the boss has none |
| Within the smaller allowance | lower than the boss's, by less than 1% |
| Within the larger allowance | equal to the boss's, or at most 1% higher |
| Beyond both | more than 1% higher than the boss's, or no valid result from the challenger |

The head-to-heads are the three round tasks and the four boss-round tasks. Each is decided on the task score alone, with no per-example test.

Source: [`IMAGE_CROWN_WIN_MARGIN`, `CROWN_LARGER_ALLOWANCE`](../validator/tournament/constants.py), [`crown.py: image_head_to_head`](../validator/tournament/crown.py).

## Local testing

One script in `ops/examples/training` runs an image task through the downloader, your trainer image and the uploader:

```bash
./ops/examples/training/run_image_task.sh
```

The script builds `ops/docker/standalone-image-toolkit-trainer.dockerfile` from the current directory, so it does not cover Flux tasks. It also builds the downloader and uploader images, which this repository contains. Run it from the root of a checkout of this repository with your repository's files copied in, your toolkit Dockerfile at `ops/docker/standalone-image-toolkit-trainer.dockerfile`. The script does not look under `dockerfiles/`. Set the task values and credentials at the top of the script first.

## Checklist

- [ ] Your repository holds both Dockerfiles at your pinned commit.
- [ ] The entrypoint of each accepts the shared arguments and the three image arguments, with `--trigger-word` optional.
- [ ] Every model family is handled: Flux, Z-Image, Qwen-Image, Ideogram 4, Krea 2.
- [ ] The training zip is read from `/cache/datasets/{task_id}_tourn.zip`, with no download.
- [ ] The LoRA is in your output directory before the deadline.
- [ ] The evaluator's file choice in [Output contract](#output-contract) lands on the file you intend.
- [ ] The repository meets the shared rules in [Training repository](miner.md#training-repository) and [Fair play and review](miner.md#fair-play-and-review).
