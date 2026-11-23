# Trainer quant decode

The sandbox that runs a freeform submission's decoder on the trainer, and the check on what it writes. The miner-facing contract is in [Freeform track: artifact and decoder](../../docs/miner_compression.md#freeform-track-artifact-and-decoder).

## Contents

| Path | Purpose |
| --- | --- |
| `sandbox.py` | Container specs for the build and run phases, and `run_phase`, which runs one phase under a wall-clock cap and removes the container. |
| `output_check.py` | `check_decoded_output`: a numpy-only safetensors reader that checks the decoded tensors against the base model. |
| `job.py` | `run_freeform_decode`, the end-to-end decode of one submission, and `check_artifact_files`. |
| `upload.py` | The storage client for decoded files: multipart upload, whole-file retry, and the check that a file is already stored. |
| `upload_retry.py` | Background retry of a decoded upload that the network keeps failing, its resume after a trainer restart, and the startup sweep of the work root. |
| `job_lock.py` | The per-job lock that keeps decodes and upload retries of one job apart, across processes. |
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
| Environment | `HOME=/tmp` and the thread-pool variables (`host_resources.THREAD_ENV_VARS`). Credential-like keys are rejected |
| Filesystem | read-only root, tmpfs at `/tmp` (`sandbox.DECODE_TMPFS`) |
| Privileges | all capabilities dropped, `no-new-privileges` |
| CPUs | `DECODE_CPUS`, pinned to the job's cores (`cpuset_cpus`); fewer when the job's share is smaller |
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

`run_freeform_decode` works in `$QUANT_DECODE_WORK_ROOT/<task_id>_<hotkey[:8]>` and removes that directory afterwards, unless the upload is still pending (see [Pending uploads](#pending-uploads)).

1. Fetch the decoder snapshot from its presigned URL, capped at `DECODER_MAX_XZ_BYTES` plus 1 MiB, and check its sha256.
2. Extract it. Members must sit under `decoder/`, with no symlinks or hardlinks, at most `DECODER_MAX_EXTRACTED_BYTES` in total, and `decode.py` must exist.
3. Download the artifact repo. Reject symlinks, exec bits and `QUANT_ARTIFACT_FORBIDDEN_SUFFIXES`. Count every file, `.gitattributes` included, as `artifact_bytes`.
4. Run the build and run phases, then `check_decoded_output` against the base model the training job used.
5. Upload each checked shard to `<bucket>/<object_prefix>/`, then `index.json` (`QuantDecodedIndex`) last. Its presence marks the upload complete.

Each stage and each uploaded file is appended to the job's log in `task_history.json`.

| Outcome | Field | Examples |
| --- | --- | --- |
| Miner's fault, scored | `quant_failure` | unsafe decoder archive, extraction past the cap, no `decode.py`, executable in the artifact, build or run exit or timeout, bad output |
| Trainer-side, never scored | `infra_error` | snapshot unreachable or sha mismatch, artifact download failure, base model unreadable, `container error:`, an upload error no retry can fix, an upload still failing when its retry window closes, unusable work root |
| Not final | `upload_pending` | the decode passed its checks and the upload of the decoded files is being retried |

## Pending uploads

A decode whose output passed the check, but whose files cannot be uploaded because of a transient error (a dropped connection, a timeout, throttling, a 5xx), does not fail. The job finishes with `quant_decode.upload_pending` set and `upload_retry_until` at the end of the retry window. `upload_retry.py` then retries in the background.

- The decoded files stay in the work directory, next to `pending_upload.json`. The decoder, its build output and the artifact are removed.
- The files are kept only while the work root's disk has `QUANT_DECODE_UPLOAD_RETRY_MIN_FREE_DISK_GB` (default 50) free after that. Below it the failed upload is an `infra_error` at once and the work directory is removed.
- The retry runs in rounds until `QUANT_DECODE_UPLOAD_RETRY_WINDOW_SECONDS` (default 2 hours) after the first failure. A round uploads only the files that are not in storage yet, one attempt per file, and stops at the first file that fails. The wait between rounds doubles from `QUANT_DECODE_UPLOAD_RETRY_ROUND_BASE_SECONDS` to `QUANT_DECODE_UPLOAD_RETRY_ROUND_MAX_SECONDS`, with jitter. The last round starts as the window closes.
- A file counts as stored when an earlier attempt recorded it, or when storage holds an object of the same size whose `sha256` metadata matches. The second case covers an upload whose final response was lost.
- When a round completes, the result becomes `ok`. When the window closes, or an error turns up that no retry can fix, it becomes `infra_error`. The result is stored on the job first and the work directory removed after it. Each round is logged on the job.
- A restarted trainer resumes every pending upload it finds in the work root, up to its original deadline. A job stored as `upload_pending` whose files are gone becomes an `infra_error`.
- A new decode of the same job, or a re-run, replaces the pending upload. The retry it replaces stops and stores nothing. See [Job lock](#job-lock).
- `QUANT_DECODE_UPLOAD_RETRY_WINDOW_SECONDS=0` turns the background retry off: the failed upload is an `infra_error` at once.

`upload_retry_until` is when the last round starts. `upload_settle_deadline` adds the per-file deadline of every file that round may still send: by then the result is final.

The validator treats `upload_pending` as unsettled. It polls the trainer again and does not alert, unless the trainer still reports the upload as pending 30 minutes after `upload_settle_deadline`.

The task-details endpoint serves an `upload_pending` result only to a caller that sends `accepts_upload_pending=true`. Any other caller gets no decode result while the upload is pending, which a validator without this state reads as "not decoded yet".

While an upload is pending, its bf16 output occupies the work root. Other decodes on the trainer need their own space on top of it.

## Job lock

A decode, a round of the upload retry and the startup sweep each change a job's work directory. Each first takes the job's lock: an exclusive `flock` on `$QUANT_DECODE_WORK_ROOT/.locks/<work dir name>.lock`. The trainer and the re-run CLI both take it, so they never work in the same directory at once. The kernel releases the lock when its holder exits, so a crash leaves nothing locked.

- A decode waits up to `QUANT_DECODE_JOB_LOCK_WAIT_SECONDS` (1 hour) for the lock, then gives up without touching the directory. That outcome is not stored on the job: the holder of the lock stores its own. A decode holds the lock until its result is known.
- A retry round takes the lock without waiting. If the lock is taken, a decode of the job is running and replaces the retry, which stops. Between rounds the lock is free.
- Under the lock, a round checks that `pending_upload.json` still carries its token: before it starts, after every uploaded file and before it stores a result. A different token means another decode or a restarted trainer owns the directory now, and the round stops without writing anything.

## Startup sweep

When the trainer starts, it goes through the work root (only if it carries the `.god-quant-decode-root` marker):

- a work directory with a pending upload is resumed under a new token;
- a work directory whose `pending_upload.json` exists but cannot be read is kept and logged as an error, and its job gets an `infra_error`. Its decoded files are never deleted on a state file the trainer cannot read; a re-run of the job clears it;
- any other work directory is removed, including `<name>.stale-<hex>` leftovers and a decode the restart cut short;
- a directory whose lock is held is left alone, because the re-run CLI is working in it;
- lock files without a work directory are removed.

<details>
<summary>Work directory permissions</summary>

The trainer owns the work directories, and both phases run as the trainer's uid:gid.

- The work root is `QUANT_DECODE_WORK_ROOT`, by default `~/.local/share/god-trainer/quant_decode`. It is not on the cache volume, because every training container mounts that volume.
- The sandbox bind-mounts the work root from the Docker host, so the path must exist on the host under the same name.
- The root and every work subdirectory are `0o700`. Files the job writes and the downloaded artifact files are `0o600`.
- A missing root is created with the marker file `.god-quant-decode-root`. An existing root is adopted only if it has the marker or is empty. A non-empty directory without the marker is never touched, and the decode ends as `infra_error`.
- Everything the decoder writes is owned by the trainer. `remove_work_tree` restores owner permissions on directories before removing them, and never follows a symlink.
- `_WorkDirs.create` clears what an earlier attempt left. A leftover it cannot remove is renamed to `<name>.stale-<hex>`. Stale directories are removed at the next trainer start.
- Disk: the root holds the decoder, the artifact and the bf16 output at once. Keep about 3x the largest bf16 model free.

</details>

<details>
<summary>Uploads</summary>

Decoded shards go to S3-compatible storage as multipart uploads: 16 MiB parts, 4 in flight. The HTTP client retries each part on connection errors and 429/5xx; when a part still fails on a transient error (dropped connection, TLS EOF, throttling, 5xx) the whole file is uploaded again to the same key, with exponential backoff and jitter, up to 4 attempts. Auth, permission and local-file errors fail at once. Each file has a deadline of 300 s plus its size at 2 MB/s, covering every attempt. A file that still fails on a transient error or runs out of time leaves the upload pending; any other failure is an `infra_error`. The upload uses the S3-compatible storage settings in `.trainer.env` (see [`docs/developer.md`](../../docs/developer.md#running-a-trainer)).

Each object carries the file's sha256 as `sha256` user metadata.

An upload that runs out of time is told to stop: its thread ends before the next slice instead of sending the rest of the file, it can no longer complete the object, and minio aborts the multipart upload. A new upload of the same object waits for that thread to leave, so abandoned uploads do not pile up on the link.

The HTTP client keeps its connect timeout on the socket while a request body is sent, and that timeout covers one socket write. A part's body is therefore sent in slices of `QUANT_DECODE_UPLOAD_SEND_SLICE_BYTES` (1 MiB): the timeout then limits how long one slice may stall, not how long the whole part takes, so a slow link that keeps moving does not time a part out.

| Setting | Default | Meaning |
| --- | --- | --- |
| `QUANT_DECODE_UPLOAD_PART_BYTES` | 16 MiB | Multipart part size. A failed part is sent again in full. |
| `QUANT_DECODE_UPLOAD_PARALLEL` | 4 | Parts in flight per file. |
| `QUANT_DECODE_UPLOAD_SEND_SLICE_BYTES` | 1 MiB | Slice size of a part's body. `0` sends each part in one write, which must then finish within the connect timeout. |
| `QUANT_DECODE_UPLOAD_HTTP_CONNECT_TIMEOUT_SECONDS` | 30 | Connecting, and each socket write of a request body. |
| `QUANT_DECODE_UPLOAD_HTTP_READ_TIMEOUT_SECONDS` | 90 | Waiting for the response to one request. |
| `QUANT_DECODE_UPLOAD_HTTP_RETRIES` | 5 | HTTP-level retries of one request. |
| `QUANT_DECODE_UPLOAD_MIN_BYTES_PER_SECOND` | 2,000,000 | Slowest rate the per-file deadline allows for. |

All of them, and the retry settings above, are environment variables read from `.trainer.env` at start. A value that is not a number falls back to its default, and one below its minimum (5 MiB parts, 1 part in flight) is raised to it; both log a warning.

</details>

## Re-running a decode

```bash
python -m trainer.quant_decode.rerun <task_id> <hotkey>
```

Run it on the trainer that ran the job. It needs the base model in the cache volume and the job in `task_history.json`. It runs the same decode as the training job, stores the result on the job and prints it as JSON. It replaces a pending upload of the same job. If the trainer is uploading for that job at that moment, the command waits for the attempt to end first. If its own upload ends up pending, the command keeps retrying in the foreground for the retry window before it exits.

| Exit code | Meaning |
| --- | --- |
| 0 | `ok` or `quant_failure` |
| 1 | `infra_error` |
| 2 | the job is unknown or has no decode request |
| 3 | the trainer restarted during the command's upload retry and took it over |
| 4 | the job stayed locked by the trainer for the whole lock wait; nothing was stored and the trainer's result stands |
| 130 | interrupted (Ctrl-C or SIGTERM) |

An interrupted command that had already replaced a pending upload removes its work directory and stores an `infra_error`, so the job is not left pending with nobody retrying it. Interrupted before its decode started, it changes nothing.

The validator holds a freeform task until every successful hotkey has a decode outcome, and re-polls the trainer. A pending upload needs no action. After an `infra_error`, this re-run is the whole fix. The presigned snapshot URL in the stored request expires 7 days after dispatch.

## See also

- [Freeform track: artifact and decoder](../../docs/miner_compression.md#freeform-track-artifact-and-decoder): the contract miners see.
- [Compression jobs](../README.md#compression-jobs): where the decode sits in a job.
