# Compression tournament: miner guide

The compression tournament ranks miners on compressing a given model: lowest loss in a prescribed format, or smallest size at a bounded loss. This guide is the contract for miners who enter it.

[Miner hub](miner.md) · [Text guide](miner_text.md) · [Image guide](miner_image.md)

[At a glance](#at-a-glance) · [Rounds](#rounds) · [The boss round](#the-boss-round) · [What your code receives](#what-your-code-receives) · [Output contract](#output-contract) · [Scoring](#scoring) · [The crown](#the-crown) · [Local testing](#local-testing) · [Checklist](#checklist)

Schedule, fee and endpoint are in [Tournaments](miner.md#tournaments). Entry rules are in [Entering a tournament](miner.md#entering-a-tournament).

## At a glance

| | |
| --- | --- |
| You compete on | compressing a given bfloat16 (bf16) model, within the task's time budget |
| You submit | a training repository. Its container writes one compressed model per task. The repository must also contain a `decoder/` directory, which is required to enter and is used on freeform tasks only |
| Enrolment | the shared checks, read at your pinned commit, plus a [decoder check](#enrolment-checks) |
| Tracks | fixed: a checkpoint in a prescribed format, loaded as it is. Freeform: any bytes, which your `decoder/` expands to bf16 |
| Dockerfile | the trainer builds `ops/docker/standalone-text-trainer.dockerfile` from your repository |
| Rounds | three, one task each: freeform, fixed, freeform |
| Boss round | four tasks: two freeform, two fixed |
| Task score | fixed: loss, lower is better. Freeform: total bytes with a penalty on loss, lower is better |

## Rounds

Each round is one task. Every miner in the round, and the boss, compresses the same input model.

| Round | Track | Input model | Advance |
| --- | --- | --- | --- |
| R1 | freeform | text boss's round-1 model | top 8 |
| R2 | fixed, `W2A16-asym-g64` | text boss's round-2 model | top 4, then the viability cut |
| R3 | freeform | text boss's round-3 model | top 1 |

Source: [`QUANT_ROUND_PLAN`](../validator/tournament/constants.py), [`round_results.py`](../validator/tournament/round_results.py).

The format name reads `W<weight bits>A<activation bits>-<sym|asym>-g<group size>`. `A16` means bf16 activations.

- The input models are the ones the text boss trains in a text tournament, at the sizes in the [text guide](miner_text.md#rounds). A compression tournament takes all its tasks from one text tournament: an active one, else one completed in the last 7 days. Until one is available it waits.
- Advancement uses average rank over every round played so far, ranked among the surviving miners.
- Viability cut: after round 2, a miner is dropped if no round-3 result, its own or the others', lets it finish top or joint top overall.
- The miner that advances from round 3 is the challenger.

<details>
<summary>Ranking details</summary>

- A [scored fail](#scored-fail-or-infrastructure-error) counts as played and ranks last on that task.
- A miner missing a result in any round sorts below every miner with a result in all rounds.
- Ties on average rank go to the better rank in the most recent round, then to the smaller summed gap to each task's best entry, then to a seeded draw.
- If the text boss produced no usable model (its run failed, or its output is not a full model), the task uses that text task's base model.

</details>

## The boss round

The challenger and the boss meet on four tasks. Both compress the same input model on each.

| Task | Track | Input model | Format |
| --- | --- | --- | --- |
| 1 | freeform | text boss's round-2 model | yours |
| 2 | fixed | text boss's round-3 model | `W3A16-asym-g128` or `W2A16-asym-g64` |
| 3 | freeform | text boss's continuous model | yours |
| 4 | fixed | text boss's continuous model | `W3A16-asym-g128` or `W2A16-asym-g64` |

Source: [`FINAL_ROUND_QUANT_TASK_DISTRIBUTION`, `QUANT_BOSS_ROUND_FIXED_FORMATS`](../validator/tournament/constants.py), [`seed_tasks.py`](../validator/tournament/seed_tasks.py).

- The boss's entries are run as each input model becomes available. The challenger's entries are run in the boss round.
- Each fixed task draws one of the two formats. Both players get the same format on a task. The two fixed tasks may differ.

## What your code receives

Every task runs one Docker image, built from `ops/docker/standalone-text-trainer.dockerfile` in your repository at your pinned commit ([`TEXT_DOCKERFILE_PATHS`](../core/constants/docker.py)). The shared container contract (limits, no network, the shared arguments and variables, paths, the deadline) is in [How your code runs](miner.md#how-your-code-runs). This section covers what is specific to compression.

### Arguments

| Argument | Meaning |
| --- | --- |
| `--task-type` | `QuantFixedTask` or `QuantFreeformTask` |
| `--dataset` | presigned URL of the train split. The file is already at `/cache/datasets/{task_id}_train_data.json` |
| `--file-format` | `s3` |
| `--dataset-type` | JSON that names the dataset's columns |
| `--quant-spec` | fixed tasks only: the format to produce, as JSON |

Source: [`trainer/runtime.py: run_trainer_container_text`](../trainer/runtime.py), [`core/models/quant_models.py`](../core/models/quant_models.py).

- `--model` is the input model to compress.
- Budgets are between 0.5 and 5 hours and vary by task. `--hours-to-complete` carries the budget.
- The train split is one JSON file of rows. `--dataset-type` carries the `InstructTextTask` or `ChatTask` fields listed in [Dataset type](miner_text.md#dataset-type).

This is `--quant-spec` for `W2A16-asym-g64`:

```json
{"weight_bits": 2, "activation": "bf16", "group_size": 64, "symmetric": false,
 "scale_dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "act_order": false}
```

### Environment variables

| Variable | Set on | Meaning |
| --- | --- | --- |
| `QUANT_SPEC` | fixed tasks | the same JSON as `--quant-spec` |
| `QUANT_TARGET` | tasks that carry a target | JSON: the reference loss and, on freeform, the knee and cap |
| `QUANT_CALIBRATION_PATH` | tasks that have a calibration set | path of the calibration file |
| `WANDB_MODE` | every task | `offline` |
| `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_ARTIFACT_DIR`, `WANDB_DATA_DIR`, `WANDB_CONFIG_DIR` | every task | `/app/checkpoints/wandb_logs/{task_id}_{hotkey}` |

### The target

`QUANT_TARGET` is set when the task carries a target, so your code must run without it.

| Field | Meaning |
| --- | --- |
| `reference_model` | the bf16 model you were given |
| `reference_loss` | its loss on the task's eval set |
| `reference_sku` | the GPU type that loss was measured on |
| `knee`, `cap` | freeform only, as ratios of the reference loss: bytes are unpenalised up to the knee, and an entry above the cap ranks behind every entry at or under it |
| `knee_loss`, `cap_loss` | freeform only: the same two as absolute losses |
| `penalty_curvature` | freeform only: curvature of the penalty |
| `bar_cap` | freeform only: `knee - 1`, the relative loss increase that carries no penalty |
| `bar` | freeform only: equal to `knee_loss` |

### The calibration set

The calibration set is tokenised rows, each with the bf16 input model's loss on it. The rows are carved from the train side of the split. They are disjoint from the scored set and are never scored.

The file holds 100 to 1,000 rows. The set can be absent, and the reference losses can be `null`. The job runs either way.

<details>
<summary>File format and how the rows are chosen</summary>

The file is one JSON document ([`QuantCalibrationSet`](../core/models/quant_models.py)):

| Field | Meaning |
| --- | --- |
| `task_id`, `eval_set_fingerprint` | identity of the set |
| `reference_model`, `reference_loss`, `reference_sku` | the model scored, its mean loss over the set, and the GPU type |
| `examples[].input_ids` | token ids |
| `examples[].labels` | same length, `-100` where masked |
| `examples[].reference_loss` | the reference's loss on that example |

At least 250 rows are carved, at most 1,000 and at most 10% of train, aiming at about 50,000 supervised tokens. Over-long rows and rows equal to a scored example are then dropped. With fewer than 100 rows left, no set is published ([`validator/tasks/datasets/constants.py`](../validator/tasks/datasets/constants.py), [`validator/tasks/requests.py`](../validator/tasks/requests.py)).

</details>

### GPUs

The GPU count follows the input model's parameter count.

| Model size | GPUs |
| --- | --- |
| up to 4B | 1 |
| above 4B, up to 12B | 2 |

Source: [`get_tournament_gpu_requirement`](../validator/tournament/gpu_requirements.py).

## Output contract

Write your output to your output directory ([Paths](miner.md#paths)). Once your output passes the check below, the trainer uploads `quant_manifest.json` and exactly the files it lists, unmodified.

### Manifest

Write `quant_manifest.json` at the output root, last ([`QuantManifest`](../core/models/quant_artifact_models.py)):

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

| Rule | Limit |
| --- | --- |
| `track` | `fixed` or `freeform`, matching the task type |
| `files` | 1 to 2,048 entries, unique relative paths, no `..`, not the manifest itself |
| Listed file | exists as a regular file with exactly the listed size |
| Unlisted files | deleted before upload |
| Total size, manifest included | at most 1.25 x the input model's bf16 bytes |
| File types | no symlinks, no exec bits, no forbidden suffix |

Source: [`check_quant_output`](../trainer/quant_upload_check.py), [`QUANT_ARTIFACT_FORBIDDEN_SUFFIXES`](../trainer/constants.py).

> [!IMPORTANT]
> A missing, invalid or rejected manifest is a training failure with no upload. This holds at the deadline too: the output is uploaded only if a valid manifest is in place when the container is killed.

<details>
<summary>Forbidden suffixes</summary>

`.py`, `.pyc`, `.pyo`, `.pyw`, `.so`, `.sh`, `.bash`, `.exe`, `.dll`, `.dylib`, `.wasm`, `.js`, `.jar`, `.class`, `.dockerfile`.

</details>

### Fixed track: the checkpoint

You are scored on the model the evaluator serves from your upload. Only `config.json` and the safetensors weights are read. The tokenizer is the input model's.

**Config.** `config.json` is the input model's, with a `quantization_config` added that matches the spec. Configs are compared as the evaluator resolves them. Five keys may differ from the input model's: `quantization_config`, `transformers_version`, `_name_or_path`, `dtype` and `torch_dtype`. Any other changed, missing or added field is a scored fail ([`quant_config_check.py`](../validator/evaluation/quant_config_check.py)).

<details>
<summary>The <code>quantization_config</code> block for <code>W2A16-asym-g64</code></summary>

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

`scale_dtype` may be `"torch.bfloat16"` or omitted.

</details>

**Tensors.** The tensors must be exactly the set the input model implies under the spec ([`expected_fixed_tensor_set`](../validator/evaluation/quant_structure.py)). Every 2-D `*.weight` except `embed_tokens` and `lm_head.weight` is replaced by:

| Tensor | dtype | Shape for a Linear `[out, in]` |
| --- | --- | --- |
| `weight_packed` | `I32` | `[out, ceil(in x bits / 32)]` |
| `weight_scale` | `BF16` | `[out, in / group_size]` |
| `weight_shape` | `I64` | `[2]` |
| `weight_zero_point` (asymmetric only) | `I32` | `[ceil(out x bits / 32), in / group_size]` |

- Every other tensor is `BF16` at its input name and shape.
- Packing is dense: values form a contiguous bitstream across 32-bit words, with no padding bits.
- An extra tensor, a missing tensor or a different shape is a scored fail.
- When the input model ties `lm_head.weight` to the embedding, a stored `lm_head.weight` is optional, as `BF16` at the embedding's shape.
- After loading, every linear layer must be quantised with the spec's scheme and run the expected kernel ([`quant_vllm.py`](../validator/evaluation/quant_vllm.py)).

### Freeform track: artifact and decoder

A freeform submission is the artifact your training job uploads, plus the `decoder/` directory of your repo at the pinned commit. The decoder runs on freeform tasks only, and turns the artifact into the input model's tensors in bf16. The artifact's file names and format are yours.

The decoder is snapshotted at enrolment and that snapshot is what runs. It runs on the trainer after your upload, in two phases:

| Phase | Command | Sees |
| --- | --- | --- |
| build (optional) | `/bin/sh /decoder/build.sh` | `/decoder` read-only, `/build` writable. `gcc`, `g++` and `make` are available |
| run | `python /decoder/decode.py /artifact /output` | `/artifact`, `/decoder` and `/build` read-only, `/output` writable. No compiler |

| Limit | Value |
| --- | --- |
| Decoder size after xz | 1,048,576 bytes |
| Decoder size extracted | 64 MiB |
| Wall clock | 900 s per phase |
| CPUs | 16 |
| Memory | run: 2.5 x the input model's bf16 bytes, at least 8 GiB. Build: 8 GiB |
| Processes | 256 |
| Writable scratch | 1 GiB at `/tmp`. The root filesystem is read-only |
| GPU, network, base model, environment | none (`HOME=/tmp` is the only variable) |

Source: [`core/constants/quant.py`](../core/constants/quant.py), [`sandbox.py`](../trainer/quant_decode/sandbox.py), [`quant-decoder-run.dockerfile`](../ops/docker/quant-decoder-run.dockerfile).

- The run image is Python 3.11 with `numpy` 2.1.3, `safetensors` 0.4.5, `zstandard` 0.23.0, CPU `torch` 2.5.1, and the `zstd` and `xz` binaries.
- `/output` must hold only top-level `*.safetensors` files, and optionally `model.safetensors.index.json`. Together they carry exactly the input model's tensor names and shapes, all `BF16`, all finite ([`output_check.py`](../trainer/quant_decode/output_check.py)).
- A decoder fault does not fail the training job. It is a scored fail on the task.
- Decoding time does not come out of your training budget.

Two policy rules apply to the decoder, on top of the shared ones in [Fair play and review](miner.md#fair-play-and-review):

- Counted bytes only: the decoder rebuilds the model from the artifact and decoder bytes alone.
- Deterministic decode: the decoder gives the same output on every run.

<details>
<summary>What is published</summary>

- Your decoder is held by the validator. The champion's decoder is published with its code after the tournament. Decoders of other miners are not published.
- The decoded bf16 models of boss-round freeform entries are published as public model repos.

</details>

Sandbox internals: [`trainer/quant_decode/README.md`](../trainer/quant_decode/README.md).

### Enrolment checks

Enrolment adds a decoder check to the shared checks in [Entering a tournament](miner.md#entering-a-tournament). It is made at your pinned commit for every entry, whatever tasks the tournament draws ([`decoder_snapshot.py`](../validator/tournament/decoder_snapshot.py)). Failing it excludes you from the tournament.

- `decoder/decode.py` exists as a regular file.
- `decoder/` holds no symlinks or hardlinks.
- The decoder is inside the two size limits above.

<details>
<summary>Checks that log and do not exclude</summary>

- Immutable dependencies ([`registration_lint.py`](../validator/tournament/registration_lint.py)): hash-pinned pip installs and git pinned to a full SHA. Downloads from untrusted URLs carry a checksum. Base images are pinned by digest unless they come from a trusted publisher namespace. `QUANT_REGISTRATION_LINT_ENFORCE` is `False`, so a violation is logged and the miner is enrolled.
- Files over 1,000,000 bytes in your repository are listed for manual review, at enrolment and at every image build.

</details>

## Scoring

Losses are normalised: `r = test_loss / reference_loss`. The reference is the bf16 input model, scored on the same GPU type as your entry. Rank key: [`quant_rank_key`](../validator/scoring/quant_ranking.py).

### Evaluation

| Property | Value |
| --- | --- |
| Engine | vLLM 0.31.0 on an H100 (SXM or PCIe) |
| Held-out split | the held-out split of the text task whose model you compress: `min(10% of rows, 1,000)` |
| Held-out split, continuous-model tasks | up to 1,000 rows sampled from the given test split |
| Tokenisation | once, with the input model's tokenizer, so every entry and the reference see identical token ids |
| Loss | mean over examples of each example's mean token loss |
| Long examples | over 4,096 tokens, or at or over the model's context length, are dropped, not truncated |

Source: [`evaluators/quant.py`](../validator/evaluation/evaluators/quant.py), [`validator/tasks/requests.py`](../validator/tasks/requests.py), [`quant_eval_set.py`](../validator/tasks/quant_eval_set.py), [`quant_loss.py`](../validator/evaluation/quant_loss.py).

### Freeform

```text
S = total_bytes x exp(penalty_curvature x max(0, r - knee)^2)
```

Entries at or under the cap rank first, by lowest effective size `S`, then by `r`. Entries over the cap rank after all of them, by lower `r`. Scored fails rank last.

The knee (1.04), cap (1.08) and curvature (2500) are in [`validator/scoring/constants.py`](../validator/scoring/constants.py). They are pinned on each task at creation and published in `QUANT_TARGET`.

<details>
<summary>The multiplier at sample values of <code>r</code></summary>

| `r` | Multiplier on bytes |
| --- | --- |
| at or under 1.040 | x1.000 |
| 1.050 | x1.284 |
| 1.060 | x2.72 |
| 1.070 | x9.5 |
| 1.080 | x54.6 |
| above 1.080 | behind every entry at or under the cap |

</details>

`total_bytes` is every file you upload plus the size of your decoder after xz. The decoder is measured as the `git archive` of `decoder/` at your pinned commit. To reproduce that number:

```bash
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
git archive --format=tar <commit> decoder/ | xz -9 -T1 --format=xz --check=crc64 -c | wc -c
```

### Fixed

Entries rank by `r`, lowest first. Scored fails rank last. There is no knee, cap or byte count.

### Scored fail or infrastructure error

A submission the evaluator cannot score is a scored fail and ranks last: a wrong layout or config, a model that fails to load or decode, a non-finite loss. A failed training job is a training failure, which ranks below a scored fail. Which failures are retried and never scored is in [Whose fault an evaluation failure is](../validator/evaluation/README.md#whose-fault-an-evaluation-failure-is).

## The crown

The crown goes to the challenger when it beats the boss under the rule in [Crown, early stop and placements](../validator/tournament/README.md#crown-early-stop-and-placements). Each task is a head-to-head with four outcomes: a win, or a non-win within the tight allowance, within the looser allowance, or beyond both. This table says what each is on a compression task.

| Track | Win | Tight allowance | Looser allowance |
| --- | --- | --- | --- |
| Freeform | at least 3% fewer effective bytes `S` than the boss | at most 1% more `S` than the boss | at most 2% more `S` |
| Fixed | passes the [paired per-example test](../validator/tournament/README.md#the-paired-per-example-test), with a loss at least 1% better | a draw on the paired test | a draw on the paired test |

- Percentages are relative to the boss's value.
- Freeform: fewer effective bytes by less than 3% is a non-win within the tight allowance.
- Fixed: passing the paired test without the 1% minimum is a draw.
- Fixed: a task where the boss passes the paired test against the challenger is beyond both allowances.

Source: [`QUANT_FREEFORM_BOSS_BYTE_MARGIN`, `QUANT_CROWN_FREEFORM_TIGHT_ALLOWANCE`, `QUANT_CROWN_FREEFORM_LOOSER_ALLOWANCE`, `QUANT_FIXED_BOSS_MIN_IMPROVEMENT`](../validator/tournament/constants.py), [`crown.py: freeform_head_to_head`, `fixed_head_to_head`](../validator/tournament/crown.py).

Payout notes for compression:

- When the boss keeps the crown, the [per-contender rule](../validator/tournament/README.md#per-contender-rule) decides whether 2nd and 3rd are paid.

## Local testing

No script in `ops/examples/training` runs a compression task. Two parts can be checked by hand:

- The decoder's counted size is reproduced by the `git archive | xz` command in [Freeform](#freeform).
- The decoder is run as `python decoder/decode.py <artifact dir> <output dir>`, under the limits in [Freeform track: artifact and decoder](#freeform-track-artifact-and-decoder).

## Checklist

- [ ] Your repository holds `ops/docker/standalone-text-trainer.dockerfile` at your pinned commit.
- [ ] `decoder/decode.py` exists at your pinned commit, with no symlinks or hardlinks, inside both size limits.
- [ ] The entrypoint accepts the shared arguments and the compression arguments, `--quant-spec` among them.
- [ ] Both task types are handled: `QuantFixedTask`, `QuantFreeformTask`.
- [ ] Your code runs when `QUANT_TARGET`, `QUANT_CALIBRATION_PATH` or the per-row reference losses are absent.
- [ ] A complete output and `quant_manifest.json` are in your output directory before the deadline.
- [ ] Fixed: the input model's `config.json` with `quantization_config` added, and the exact tensor set for the format in `--quant-spec`.
- [ ] Freeform: `decode.py` writes only top-level bf16 `*.safetensors` with the input model's names and shapes, inside the sandbox limits.
- [ ] The repository meets the shared rules in [Training repository](miner.md#training-repository) and [Fair play and review](miner.md#fair-play-and-review).
