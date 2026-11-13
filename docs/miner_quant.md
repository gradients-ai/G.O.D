# Quant Tournament: Miner Guide

The quant tournament is its own weekly tournament type, next to the text and image tournaments. It is about compression: smaller, faster models that keep their quality. Its champion (the boss) is carried forward only by quant tournaments, and starts as the latest text tournament's champion. The text champion and the quant champion are the two bosses of the subnet.

This guide covers what is specific to the quant tournament. Registration, the miner endpoint, repository layout and licence are in the [Miner Guide](miner.md). Rules cite the code that enforces them as `path: symbol`, relative to the repository root; where this guide and the code differ, the code is right.

[Entering](#entering) · [Overview](#overview) · [What your code receives](#what-your-code-receives) · [Output contract](#output-contract) · [Scoring](#scoring) · [Fixed track](#fixed-track-the-checkpoint) · [Freeform track](#freeform-track-artifact-and-decoder) · [Evaluation](#evaluation) · [Registration checks](#registration-checks) · [Emissions](#emissions) · [Fair play](#fair-play-and-review) · [Checklist](#checklist-for-miners)

## Entering

| | |
| --- | --- |
| Endpoint | your miner answers `GET /training_repo/quant` with your repo and a full commit SHA. A miner that does not answer for `quant` is simply not entered |
| Schedule | Monday 09:00 UTC, weekly (text starts at 11:00, image at 13:00). The first quant tournament is created at once when no quant tournament exists; after that, one starts in the Monday 09:00 UTC window once the previous one is complete. A quant tournament takes its tasks from an active text tournament that has its boss tasks ready; until one exists it waits and nothing is created ([`validator/tournament/quant_sources.py: bind_quant_source_tournament`](../validator/tournament/quant_sources.py)) |
| Fee | 0.6 TAO per entry (text 0.7, image 0.4). One fee per entry per tournament type; entering text and quant is two entries |
| Decoder | `decoder/decode.py` at your pinned commit is required for quant only ([Registration checks](#registration-checks)). The text tournament needs none |
| Field size | at least 4 validated miners ([`MIN_MINERS_FOR_TOURN`](../validator/tournament/constants.py)) |

You may return a different repo or commit for `quant` than for `text` and `image`. The generic rules (one entry per coldkey, IP and GitHub account, fee balance, licence, obfuscation) are in the [Miner Guide](miner.md).

## Overview

Rounds 1 to 3 give every miner the same bf16 model to compress: the text boss's model from the matching text round. The boss round gives the challenger and the boss six compression tasks, on the text boss's round-2, round-3 and continuous-SFT models. No miner trains an instruct model in a quant tournament. Each quant task is its own task, linked to the text task and the model it compresses.

```text
SOURCES      the text tournament's boss trains its text tasks
             ├── round-1 model ────────► what every miner compresses in R1
             ├── round-2 model ────────► what every miner compresses in R2; also two boss-round tasks
             ├── round-3 model ────────► what every miner compresses in R3; also two boss-round tasks
             └── continuous-SFT model ─► two boss-round tasks

ROUNDS       R1 freeform ──top 8──► R2 fixed ──top 4──► R3 freeform ──top 1──► BOSS ROUND
             every entrant,         W2A16-asym-g64      survivors of the       challenger vs boss,
             boss excluded                              viability cut          six tasks, wins 5 of 6
```

> [!IMPORTANT]
> The champion (boss) does not play rounds 1 to 3. Its pinned code trains the text tasks whose outputs you compress ([Source models](#source-models-and-the-boss-entries)).

There are two tracks:

| Track | You upload | Ranked by |
| --- | --- | --- |
| Fixed | a compressed-tensors checkpoint in a prescribed format | loss |
| Freeform | any bytes, plus a `decoder/` in your repo that expands them to bf16 | total bytes, with a penalty on loss |

### Rounds

| Round | Task | Input model | Who plays | Advance |
| --- | --- | --- | --- | --- |
| R1 | `QuantFreeformTask` | text boss's round-1 model (starts as soon as that model lands) | every entrant, boss excluded | top 8 |
| R2 | `QuantFixedTask`, always `W2A16-asym-g64` | text boss's round-2 model | the 8 | top 4, then a viability cut |
| R3 | `QuantFreeformTask` | text boss's round-3 model | survivors | top 1 |
| Boss | three `QuantFreeformTask` and three `QuantFixedTask` | text boss's round-2, round-3 and continuous-SFT models, one freeform and one fixed task on each | challenger vs boss | win 5 of 6 |

Source: [`validator/tournament/constants.py: QUANT_ROUND_PLAN`](../validator/tournament/constants.py), `FINAL_ROUND_QUANT_TASK_DISTRIBUTION`.

### Source models and the boss entries

The text tournament's boss trains its text tasks when the text tournament starts. A source model counts as available when that training run succeeded and its output is a full model on the hub; an adapter output is first merged into a full model with the model-prep merge, and the merged repo is the source. Each quant task is created from one of those models and released as soon as its model is available. The link between each quant task and its text task, text tournament and model repo is stored in `quant_task_sources`, and the link between a quant tournament and its text tournament in `quant_tournament_sources` ([`validator/tournament/quant_sources.py`](../validator/tournament/quant_sources.py)).

| Source | Text task it comes from | Used for |
| --- | --- | --- |
| round-1 model | the text round-1 task (instruct, 1-4 B) | the input of R1 |
| round-2 model | the text round-2 task (instruct, 1-4 B) | the input of R2, and of a freeform and a fixed boss-round task |
| round-3 model | the text round-3 task (instruct, 4-8 B) | the input of R3, and of a freeform and a fixed boss-round task |
| continuous-SFT model | the text continuous-SFT task | the input of a freeform and a fixed boss-round task |

In rounds 1 to 3 your input model is the text boss's output on a text task, never the raw base. You also get that task's train split and calibration set. In the boss round the challenger does no instruct training: it compresses the same model the boss compresses.

**Boss baseline.** The quant boss trains its six boss-round entries with its pinned code as soon as each source model is available, and they are evaluated and stored. The boss round trains and evaluates the challenger's six entries and compares them with the stored ones.

### Advancement

Cuts use average rank over every round played so far, within the surviving cohort ([`validator/tournament/round_results.py: get_group_winners`](../validator/tournament/round_results.py)).

- A scored fail counts as played and ranks last on that task.
- A training failure leaves no result. A miner missing a result in any round sorts below every miner with a result in all rounds, whatever its average rank.
- After R2, a miner whose best possible R3 rank cannot put it top or joint top overall is dropped. If one miner is left, R3 is skipped and it meets the boss.
- A tournament needs at least 4 entrants ([`MIN_MINERS_FOR_TOURN`](../validator/tournament/constants.py)).

### The boss round

The boss round has six tasks: a freeform compression and a fixed compression of each of the text boss's round-2 model, round-3 model and continuous-SFT model. The boss's entries were trained as each source model became available; the challenger trains its six entries with its own code.

| Task | Type | What each player does | Decided by |
| --- | --- | --- | --- |
| Freeform compression (3 tasks) | `QuantFreeformTask` | compresses the shared input into an artifact and decoder | effective size `S` |
| Fixed compression (3 tasks) | `QuantFixedTask` | compresses the shared input into a drawn format | paired per-example test |

The fixed tasks of the boss round use 2-bit or 3-bit formats only: each draws `W3A16-asym-g128` or `W2A16-asym-g64` ([`validator/tournament/constants.py: QUANT_BOSS_ROUND_FIXED_FORMATS`](../validator/tournament/constants.py)). Both players get the same format on a task; the three fixed tasks may differ. Each task has one bf16 reference, the input model on the same GPU type.

The challenger is crowned if it wins at least 5 of the 6 tasks. A task that could not be decided is not a win.

| Task type | Challenger wins when | Otherwise |
| --- | --- | --- |
| freeform | under the cap, and the boss is not under the cap or its `S` is at least 3 % smaller than the boss's | boss keeps it; never a draw |
| fixed | passes the paired gate below and its scalar loss `r` is at least 1 % better than the boss's | boss keeps it |

The margins, in [`validator/tournament/constants.py`](../validator/tournament/constants.py):

| Margin | Value | Meaning | Why |
| --- | --- | --- | --- |
| Freeform win margin (`QUANT_FREEFORM_BOSS_BYTE_MARGIN`) | 3 % | to beat the boss, your `S` must be at least 3 % smaller | the title should change hands for a real improvement, not a rounding difference |
| Fixed minimum improvement | 1 % | your scalar loss must be at least 1 % better than the boss's, on top of the paired test | a statistically clean but tiny difference should not change the title |

**The fixed task** is decided example by example: to win you must be reliably better than the boss, ahead on at least 55 % of examples with a mean gap of at least 0.01 nats and 1 % of the boss's loss, each holding at 99 % confidence ([`validator/tournament/thresholds.py: compare_paired_losses`](../validator/tournament/thresholds.py)). Why: two quantised models differ by small amounts on every example, so a single averaged number can flip on noise; the paired test only crowns a difference that holds across the eval set.

The crown logic is [`validator/tournament/round_results.py: determine_boss_round_winner`](../validator/tournament/round_results.py).

## What Your Code Receives

Every quant task runs one image, built from `ops/docker/standalone-text-trainer.dockerfile` (else `dockerfiles/standalone-text-trainer.dockerfile`) in your repo at your pinned commit. The container has only the assigned H100s ([`validator/tournament/gpu_requirements.py: get_tournament_gpu_requirement`](../validator/tournament/gpu_requirements.py)), no network and no credentials. The general contract is in [Runtime Arguments](miner.md#runtime-arguments) and [Cache And Output Paths](miner.md#cache-and-output-paths).

### Invocation

| Input | What it is |
| --- | --- |
| `--task-type` | `QuantFixedTask` or `QuantFreeformTask` |
| `--model` | directory name under `/cache/models/`: the input model with its tokenizer and chat template, read-only |
| `--dataset`, `--dataset-type` | the train split, already at `/cache/datasets/<task_id>_train_data.json`, and its column description. The held-out split is never shown |
| `--hours-to-complete` | your wall-clock budget in hours, as a float |
| `--expected-repo-name` | your output directory is `/app/checkpoints/<task_id>/<expected_repo_name>/`; only that is uploaded |
| `--quant-spec`, `QUANT_SPEC` | fixed tasks only: the format to produce, as `QuantFixedSpec` JSON |
| `QUANT_TARGET` | the reference loss and, on freeform, the knee and cap ([QuantTarget](#quanttarget)) |
| `QUANT_CALIBRATION_PATH` | tasks that publish a set: path of the calibration file ([Calibration set](#calibration-set)) |

Source: [`trainer/runtime.py: run_trainer_container_text`](../trainer/runtime.py).

> [!WARNING]
> The deadline is a SIGKILL with no grace, `--hours-to-complete` after container start. If a valid manifest is in the output directory at that moment, the output is uploaded; otherwise the job fails. A container that exits non-zero fails even with a valid manifest.

Quant budgets are between 0.5 and 5 hours and vary by task ([`validator/tasks/synthetics/scheduler.py: quant_training_hours`](../validator/tasks/synthetics/scheduler.py)). The argument is the only place the budget is visible.

### QuantTarget

`QUANT_TARGET` is JSON ([`core/models/quant_models.py: QuantTarget`](../core/models/quant_models.py)).

| Field | Meaning |
| --- | --- |
| `reference_model`, `reference_loss`, `reference_sku` | the bf16 model you were given, its loss on the task's eval set, and the GPU it was measured on |
| `knee`, `cap` | freeform only, as ratios of the reference loss: bytes are unpenalised up to the knee; above the cap the entry is out |
| `knee_loss`, `cap_loss` | the same two as absolute losses |
| `penalty_curvature` | freeform only: curvature of the penalty |

It is absent on organic tasks and if the reference measurement failed repeatedly.

### Calibration set

Tokenised rows that resemble the scored rows, each with the input model's bf16 loss on it. They are carved from the train side, are disjoint from the scored set, and are never scored. You may train on them.

The file is one JSON document ([`core/models/quant_models.py: QuantCalibrationSet`](../core/models/quant_models.py)) with `input_ids`, `labels` (`-100` where masked) and `reference_loss` per example. The set, or the per-row reference losses, can be missing; the job runs anyway.

## Output Contract

The uploaded bytes are the score, so the trainer uploads exactly what you declare. Write `quant_manifest.json` at the output root, last ([`core/models/quant_artifact_models.py: QuantManifest`](../core/models/quant_artifact_models.py)):

```json
{
  "version": 1,
  "task_id": "<--task-id>",
  "track": "freeform",
  "files": [
    {"name": "weights.bin", "bytes": 1031204864},
    {"name": "meta.json", "bytes": 1412}
  ]
}
```

- Every listed file must exist with exactly the listed size. Files not listed are deleted before upload.
- No symlinks, no exec bits, and no forbidden suffix such as `.py` or `.so` ([`trainer/constants.py: QUANT_ARTIFACT_FORBIDDEN_SUFFIXES`](../trainer/constants.py)).
- The total may not exceed 1.25 x the input model's bf16 size.
- The directory is uploaded as is to a public Hugging Face repo.

A missing, invalid or rejected manifest is a training failure ([`trainer/quant_upload_check.py: check_quant_output`](../trainer/quant_upload_check.py)).

## Scoring

Each task is ranked once and rounds combine by average rank. Losses are normalised: `r = test_loss / reference_loss`, where the reference is the bf16 input model scored on the same GPU type as your entry. Rank key: [`validator/scoring/quant_ranking.py: quant_rank_key`](../validator/scoring/quant_ranking.py).

### Freeform

```text
S = total_bytes x exp(penalty_curvature x max(0, r - knee)^2)        # knee is a RATIO, e.g. 1.04
```

Entries under the cap rank first, by lowest `S`. Entries over the cap come after all of them, by lower `r`. Scored fails come last.

| `r` | Multiplier on bytes |
| --- | --- |
| at or under 1.040 | x1.000 |
| 1.050 | x1.284 |
| 1.060 | x2.72 |
| 1.070 | x9.5 |
| 1.080 | x54.6 |
| above 1.080 | out |

The knee (1.04), cap (1.08) and curvature (2500) are in [`validator/scoring/constants.py`](../validator/scoring/constants.py). They are pinned on each task at creation and published in `QUANT_TARGET`.

`total_bytes` is every file in your uploaded repo plus the `xz -9` size of the `git archive` of `decoder/` at your registered commit ([`validator/tournament/decoder_snapshot.py: xz_bytes_of`](../validator/tournament/decoder_snapshot.py)). To reproduce the decoder part:

```bash
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
git archive --format=tar <commit> decoder/ | xz -9 -T1 --format=xz --check=crc64 -c | wc -c
```

### Fixed

Ranked by `r`, lowest first; scored fails last. There is no knee, cap or byte count.

### Scored fail or infrastructure error

A broken submission is a scored fail and ranks last: wrong layout or config, a model that fails to load or decode, a non-finite loss, or the uncompressed base model. Validator-side problems (hub, storage, GPU, deployment crashes) are re-queued and never scored. A failed training job is a training failure, which sorts below a scored fail at the cut. Source: [`validator/evaluation/errors.py`](../validator/evaluation/errors.py).

## Fixed Track: The Checkpoint

You are scored on the model vLLM 0.31.0 serves from your repo. Only `config.json` and the safetensors weights are read; the tokenizer is the input model's. The format name reads `W<weight_bits>A<activation bits>-<sym|asym>-g<group_size>`, and `A16` means bf16 activations.

`config.json` is the input model's with a `quantization_config` added that matches the spec. Apart from a few bookkeeping keys ([`CONFIG_KEYS_FREE_TO_DIFFER`](../validator/evaluation/quant_config_check.py)), any other changed, missing or added field is a scored fail ([`validator/evaluation/quant_config_check.py`](../validator/evaluation/quant_config_check.py)).

```json
"quantization_config": {
  "quant_method": "compressed-tensors",
  "format": "pack-quantized",
  "ignore": ["lm_head"],
  "kv_cache_scheme": null,
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "format": "pack-quantized",
      "weights": {"type": "int", "strategy": "group", "num_bits": 2, "group_size": 64,
                  "symmetric": false, "actorder": null, "scale_dtype": "torch.bfloat16"},
      "input_activations": null,
      "output_activations": null
    }
  }
}
```

The tensors must be exactly the set the input model implies under the spec ([`validator/evaluation/quant_structure.py: expected_fixed_tensor_set`](../validator/evaluation/quant_structure.py)). Every 2-D `*.weight` except `embed_tokens` and `lm_head.weight` is replaced by:

| Tensor | dtype | Shape for a Linear `[out, in]` |
| --- | --- | --- |
| `weight_packed` | `I32` | `[out, ceil(in x bits / 32)]` |
| `weight_scale` | `BF16` | `[out, in / group_size]` |
| `weight_shape` | `I64` | `[2]` |
| `weight_zero_point` (asymmetric only) | `I32` | `[ceil(out x bits / 32), in / group_size]` |

Every other tensor stays `BF16` at its input name and shape, and is yours to train.

> [!WARNING]
> Packing is dense with no padding bits. The compressed-tensors packer pads 3-bit to 10 values per int32, and that layout is rejected as a shape mismatch ([`quant_structure.py: packed_layout`](../validator/evaluation/quant_structure.py)).

After loading, every linear layer must be quantised with the spec's scheme and run the expected kernel ([`validator/evaluation/quant_vllm.py`](../validator/evaluation/quant_vllm.py)).

## Freeform Track: Artifact And Decoder

A freeform submission is the artifact your training job uploads plus the `decoder/` directory of your repo at the registered commit. The decoder turns the artifact into the input model's tensors in bf16. The artifact's file names and format are yours.

- `decoder/decode.py` is required at registration for every quant entrant, whichever track a round uses. The decoder is frozen at registration.
- It runs on the trainer after upload as `python /decoder/decode.py /artifact /output`. An optional `decoder/build.sh` runs first under POSIX `sh`, with `gcc`, `g++` and `make`.
- The sandbox has no GPU, no network and no base model: 16 CPUs, 900 s per phase, memory of 2.5 x the input model's bf16 bytes (at least 8 GiB), a read-only root and 1 GB at `/tmp` ([`trainer/quant_decode/sandbox.py`](../trainer/quant_decode/sandbox.py)).
- The image is Python 3.11 with numpy, safetensors, zstandard and CPU torch ([`ops/docker/quant-decoder-run.dockerfile`](../ops/docker/quant-decoder-run.dockerfile)).
- `/output` must hold only top-level `*.safetensors` files with exactly the input model's tensor names and shapes, all `BF16`, all finite ([`trainer/quant_decode/output_check.py`](../trainer/quant_decode/output_check.py)).
- The decoder may be at most 1,048,576 bytes after xz ([`core/constants/quant.py: DECODER_MAX_XZ_BYTES`](../core/constants/quant.py)).
- Decoding must be deterministic on CPU. At handover the champion's decoder is published and anyone can re-run it against the public artifact.

A decoder fault does not fail the training job; it is a scored fail on the task. Decoding time does not come out of your training budget.

Your artifact repo is public from upload. Your decoder stays with the validator until the champion handover, when the champion's code is published. Other decoders are never released.

## Evaluation

Entries are scored with vLLM 0.31.0 on an H100 (SXM or PCIe), against a bf16 reference measured on the same GPU type ([`validator/evaluation/evaluators/quant.py`](../validator/evaluation/evaluators/quant.py)).

The text instruct tasks hold out `min(10 % of rows, 1,000)` ([`validator/tasks/datasets/constants.py: TRAIN_TEST_SPLIT_PERCENTAGE`](../validator/tasks/datasets/constants.py), [`MAX_TEST_DATA_POINTS`](../validator/tasks/datasets/constants.py)); every compression task reuses the split of the text task it compresses. The held-out split is tokenised once with the input model's tokenizer, so every entry and the reference are scored on identical token ids ([`validator/tasks/quant_eval_set.py`](../validator/tasks/quant_eval_set.py), [`validator/evaluation/chat_labels.py`](../validator/evaluation/chat_labels.py)).

The loss is the mean over examples of each example's mean token loss; examples longer than 4096 tokens, or the model's context, are dropped, not truncated ([`validator/evaluation/quant_loss.py`](../validator/evaluation/quant_loss.py)).

## Registration Checks

On top of the generic checks in the [Miner Guide](miner.md#training-repository-requirements):

- `decoder/` with `decoder/decode.py` must exist at the pinned commit, within the size limits ([`validator/tournament/decoder_snapshot.py`](../validator/tournament/decoder_snapshot.py)). Otherwise you are excluded from the whole quant tournament.
- Dependencies must be immutable: hashed pip installs, git pinned to a full SHA, digest-pinned base images ([`validator/tournament/registration_lint.py`](../validator/tournament/registration_lint.py)). This is currently warn-only.
- Large files in the repo are logged for manual review.

## Emissions

Each tournament type has its own pool. The guaranteed minimums are text 0.15, quant 0.35 and image 0.225; the remaining 0.275 is a flex pool that moves with how many miners entered the recent tournaments of each type ([`validator/scoring/constants.py`](../validator/scoring/constants.py), [`validator/scoring/emission_balance.py`](../validator/scoring/emission_balance.py)). The quant pool therefore lies between 0.35 and 0.625.

The pool pays the latest completed quant tournament's top three, in the same ratio as the other types, until a newer quant tournament completes. The quant boss is carried forward only by quant tournaments, and is paid from this pool; the text boss is paid from the text pool.

The minimums apply at all times. A type with no winner yet pays nobody, and its share goes to the burn address. See [Scoring And Weights](miner.md#scoring-and-weights).

## Fair Play And Review

Submissions are reviewed at the end of each tournament, first by an automated agent check, then by a person. An entry that did not genuinely train or compress the given model on the given data within the given budget may be removed from that tournament's results. The automated checks in this guide are a floor, not a full definition of what is allowed; if you find a gap, report it rather than use it.

The given inputs are:

- the input model directory passed as `--model`;
- the training split passed as `--dataset`;
- the calibration set at `QUANT_CALIBRATION_PATH`, when there is one;
- `QUANT_TARGET`, `--quant-spec` and the other documented arguments and environment variables;
- extra datasets the validator itself fetches for you through its documented mechanism, where that applies;
- your own code and its open-source dependencies, pinned.

Everything else is out of bounds: other files on the trainer, anything downloaded, anything computed beforehand that depends on the models or datasets a task might draw.

- **Train only on the data you are given.** No outside copies of the task's dataset or its test rows, and nothing that memorises them.
- **Produce everything inside the job, from the given model.** No pre-computed or downloaded weights, and no model-specific payloads carried in your repository, image or decoder.
- **Let the decoder do the work.** A freeform decoder must genuinely rebuild the model from the counted bytes, and give the same result every time.
- **Keep to the required format.** On the fixed track, the values inside the layout are yours to train.

If this list does not cover your situation, the rule above still applies. If in doubt, ask before the tournament.

## Checklist For Miners

- [ ] `decoder/decode.py` exists at your registered commit, with no symlinks, hardlinks or LFS files, and the decoder is under the xz size limit.
- [ ] Your Dockerfile is at `ops/docker/standalone-text-trainer.dockerfile` with every dependency pinned.
- [ ] The entrypoint handles both task types and reads `--task-type`, `--hours-to-complete`, `--dataset-type` and `--quant-spec`.
- [ ] Your code runs when `QUANT_TARGET`, `QUANT_CALIBRATION_PATH` or the per-row reference losses are absent.
- [ ] A valid output and `quant_manifest.json` are in place before the SIGKILL, and the container exits 0 if it finishes early.
- [ ] Fixed: the input model's `config.json` with only `quantization_config` added, and the exact packed tensor set for the format in `--quant-spec`.
- [ ] Freeform: `decode.py` writes only top-level bf16 `*.safetensors` with the input model's names and shapes, inside the sandbox limits, deterministically.
