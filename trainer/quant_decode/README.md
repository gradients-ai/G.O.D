# Trainer quant decode

The sandbox that runs a freeform submission's decoder on the trainer, and the check on what it writes. The miner-facing contract is in [Freeform track: artifact and decoder](../../docs/miner_compression.md#freeform-track-artifact-and-decoder).

## Contents

| Path | Purpose |
| --- | --- |
| `sandbox.py` | Container specs for the build and run phases, and `run_phase`, which runs one phase under a wall-clock cap and removes the container. |
| `output_check.py` | `check_decoded_output`: a numpy-only safetensors reader that checks the decoded tensors against the base model. |
| `job.py` | `run_freeform_decode`, the end-to-end decode of one submission, and `check_artifact_files`. |
| `rerun.py` | CLI that re-runs one job's decode from task history. |

## Phases

| Phase | Image | Mounts | Command |
| --- | --- | --- | --- |
| build | `QUANT_DECODER_BUILD_IMAGE` (the run image plus `gcc`, `g++`, `make` and `libc6-dev`) | decoder source read-only at `/decoder`, empty writable `/build` | `/bin/sh /decoder/build.sh` if present |
| run | `QUANT_DECODER_RUN_IMAGE` (no compiler) | `/artifact`, `/decoder` and `/build` read-only, empty writable `/output` | `python /decoder/decode.py /artifact /output` |

The build phase never sees the artifact, so nothing compiled is derived from artifact bytes.

A `build.sh` that exits 0 but leaves `/build` empty, or logs a line matching `sandbox.BUILD_FAILURE_LINE`, gets a `build_warning`. The warning is logged, carried on `QuantDecodeResult.build_warning` and stored in `task_nodes.decoder_build_warning`. It never fails the submission.

## Limits

Both phases share these limits. The constants are in `core/constants/quant.py` and `trainer/constants.py`. Their values are listed once, in the [miner contract](../../docs/miner_compression.md#freeform-track-artifact-and-decoder).

| Limit | Value |
| --- | --- |
| User | the trainer process's uid:gid (`sandbox.sandbox_user`) |
| Network | disabled (`network_disabled`) |
| GPU | none |
| Environment | `HOME=/tmp` only. Credential-like keys are rejected |
| Filesystem | read-only root, tmpfs at `/tmp` (`sandbox.DECODE_TMPFS`) |
| Privileges | all capabilities dropped, `no-new-privileges` |
| CPUs | `DECODE_CPUS` |
| Processes | `DECODE_PIDS_LIMIT` |
| Wall clock | `DECODE_TIMEOUT_SECONDS` per phase |
| Memory, build | `QUANT_DECODE_MIN_MEMORY_GB` |
| Memory, run | `DECODE_MEMORY_MULTIPLIER` x the bf16 model bytes, at least the build value |

The sandbox has no network, cache or base model, and its image packages are pinned. Every bit of information about the weights therefore comes from the counted bytes: the artifact and the decoder.

## Output check

`check_decoded_output` returns the first failure, in this order:

1. an entry that is not a `*.safetensors` file or `model.safetensors.index.json`
2. no safetensors output
3. a corrupt header (including a header length above `MAX_SAFETENSORS_HEADER_BYTES` or an unreadable file)
4. a tensor present in two files
5. names that differ from the base model
6. a dtype other than `BF16`
7. a wrong shape
8. a data length other than `prod(shape) * 2`
9. a non-finite value

The index file is accepted and ignored. The job deletes it, so only checked shards reach the evaluator. The evaluator loads the output with the base config and tokenizer, so miner code never runs at inference time.

## Decode job

`run_freeform_decode` works in `$QUANT_DECODE_WORK_ROOT/<task_id>_<hotkey[:8]>` and removes that directory afterwards.

1. Fetch the decoder snapshot from its presigned URL, capped at `DECODER_MAX_XZ_BYTES` plus 1 MiB, and check its sha256.
2. Extract it. Members must sit under `decoder/`, with no symlinks or hardlinks, at most `DECODER_MAX_EXTRACTED_BYTES` in total, and `decode.py` must exist.
3. Download the artifact repo. Reject symlinks, exec bits and `QUANT_ARTIFACT_FORBIDDEN_SUFFIXES`. Count every file, `.gitattributes` included, as `artifact_bytes`.
4. Run the build and run phases, then `check_decoded_output` against the base model the training job used.
5. Upload each checked shard to `<bucket>/<object_prefix>/`, then `index.json` (`QuantDecodedIndex`) last. Its presence marks the upload complete.

Each stage and each uploaded file is appended to the job's log in `task_history.json`.

| Outcome | Field | Examples |
| --- | --- | --- |
| Miner's fault, scored | `quant_failure` | unsafe decoder archive, extraction past the cap, no `decode.py`, executable in the artifact, build or run exit or timeout, bad output |
| Trainer-side, never scored | `infra_error` | snapshot unreachable or sha mismatch, artifact download failure, base model unreadable, `container error:`, upload failure or timeout, unusable work root |

<details>
<summary>Work directory permissions</summary>

The trainer owns the work directories, and both phases run as the trainer's uid:gid.

- The work root is `QUANT_DECODE_WORK_ROOT`, by default `~/.local/share/god-trainer/quant_decode`. It is not on the cache volume, because every training container mounts that volume.
- The sandbox bind-mounts the work root from the Docker host, so the path must exist on the host under the same name.
- The root and every work subdirectory are `0o700`. Files the job writes and the downloaded artifact files are `0o600`.
- A missing root is created with the marker file `.god-quant-decode-root`. An existing root is adopted only if it has the marker or is empty. A non-empty directory without the marker is never touched, and the decode ends as `infra_error`.
- Everything the decoder writes is owned by the trainer. `remove_work_tree` restores owner permissions on directories before removing them, and never follows a symlink.
- `_WorkDirs.create` clears what an earlier attempt left. A leftover it cannot remove is renamed to `<name>.stale-<hex>`. Stale directories are not swept automatically.
- Disk: the root holds the decoder, the artifact and the bf16 output at once. Keep about 3x the largest bf16 model free.

</details>

<details>
<summary>Uploads</summary>

Decoded shards go to S3-compatible storage as multipart uploads: 32 MiB parts, 16 in flight. Each file has a deadline of 300 s plus its size at 2 MB/s. A stalled upload is an `infra_error`. The upload uses the S3-compatible storage settings in `.trainer.env` (see [`docs/developer.md`](../../docs/developer.md#running-a-trainer)).

</details>

## Re-running a decode

```bash
python -m trainer.quant_decode.rerun <task_id> <hotkey>
```

Run it on the trainer that ran the job. It needs the base model in the cache volume and the job in `task_history.json`. It runs the same decode as the training job, stores the result on the job and prints it as JSON.

| Exit code | Meaning |
| --- | --- |
| 0 | `ok` or `quant_failure` |
| 1 | `infra_error` |
| 2 | the job is unknown or has no decode request |

The validator holds a freeform task until every successful hotkey has a decode outcome, and re-polls the trainer. After an `infra_error`, this re-run is the whole fix. The presigned snapshot URL in the stored request expires 7 days after dispatch.

## See also

- [Freeform track: artifact and decoder](../../docs/miner_compression.md#freeform-track-artifact-and-decoder): the contract miners see.
- [Compression jobs](../README.md#compression-jobs): where the decode sits in a job.
