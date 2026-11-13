# Trainer Quant Decode

Sandbox for a freeform quant submission's miner-written decoder, and the check on what it writes.
The decoder only decompresses: `decode(artifact_dir)` writes the base model's bf16 tensors as safetensors.
The submission's `total_bytes` is the HF artifact plus the decoder's xz size, and the sandbox exists to keep that
number honest: every bit of information about the weights must come from those counted bytes.

## Contents

- `sandbox.py`: `DecodeContainerSpec` for the build and run phases (`build_phase_spec`, `run_phase_spec`) and
  `run_phase`, which runs one phase with a wall-clock cap and always removes the container.
- `output_check.py`: `check_decoded_output`, a numpy-only safetensors reader that checks names, shapes, dtype and
  finiteness against `ExpectedTensors` taken from the base-model snapshot.
- `job.py`: `run_freeform_decode`, the end-to-end decode of one freeform submission (below), and
  `check_artifact_files`, the executable-free check on the downloaded HF artifact.
- `rerun.py`: ops CLI that re-runs one job's freeform decode from task history (below).
- `__init__.py`: package marker.

## Decoder contract

| Phase | Image | Mounted | Command |
| --- | --- | --- | --- |
| build | `QUANT_DECODER_BUILD_IMAGE` (run image + gcc/g++/make) | `decoder/` source ro at `/decoder`, empty rw `/build` | `/bin/sh /decoder/build.sh` if present, else nothing |
| run | `QUANT_DECODER_RUN_IMAGE` (numpy, CPU torch, safetensors, zstd, xz; no compiler) | artifact ro at `/artifact`, decoder source ro at `/decoder`, build output ro at `/build`, empty rw `/output` | `python /decoder/decode.py /artifact /output` |

- `build.sh` is optional and only ever sees decoder source; it writes binaries to `/build`, which the run phase sees
  read-only.
- A build that exits 0 but leaves `/build` empty or logs a failure line (`sandbox.BUILD_FAILURE_LINE`: `fail`,
  `failed`, `failure`, `error`, any case) gets a `build_warning` (`sandbox.build_phase_warning`): logged as
  `!!! QUANT DECODER BUILD WARNING ... !!!` and carried on `QuantDecodeResult.build_warning` to the validator, which
  stores it in `task_nodes.decoder_build_warning`. It never fails the submission: what the run phase does with
  an empty `/build` is the miner's choice. Only the build phase's log is read on success (the last
  `QUANT_DECODE_LOG_TAIL_LINES` lines, stdout and stderr).
- `decode.py` must write only `*.safetensors` (plus an optional `model.safetensors.index.json`) into `/output`.

## Limits

Both phases: the trainer process's own uid:gid (`sandbox.sandbox_user`, never root on a prod trainer), no network
(`network_disabled`, not the internal bridge), no GPU, no environment variables but `HOME=/tmp` (`DECODE_ENVIRONMENT`;
credential-like keys are rejected), read-only root filesystem with a 1 GB tmpfs (mode 1777) at `/tmp`, all
capabilities dropped, `no-new-privileges`, `DECODE_CPUS` cores, `DECODE_PIDS_LIMIT` PIDs and `DECODE_TIMEOUT_SECONDS` wall clock
(`core/constants/quant.py`). Memory is `QUANT_DECODE_MIN_MEMORY_GB` for build and
`max(floor, DECODE_MEMORY_MULTIPLIER x bf16 model bytes)` for run.

Why each limit: no network, cache or base model, because any of them would let weight information bypass the
counted bytes; pinned run-image packages for the same reason (no decoder logic can live in a dependency); no GPU and a
900 s cap, so the decoder can unpack but not retrain. The build phase gets a compiler but never the artifact, so
nothing compiled is derived from artifact bytes.

## Output check

First failure wins: unexpected file, no safetensors output, corrupt header, tensor duplicated across files, names
differ from the base model, dtype not `BF16`, wrong shape, data length not `prod(shape) * 2`, non-finite value
(bf16 exponent bits all set, scanned in 64 MiB chunks).

An accepted `model.safetensors.index.json` is ignored by the check; the job deletes it, so only checked shards reach
vLLM. A header length above `MAX_SAFETENSORS_HEADER_BYTES` or an unreadable file is a corrupt header, never an
exception.

A failed phase or check is a scored fail for the submission, never a training failure. The output is plain bf16 at
base names and shapes because the evaluator loads it with the base config and tokenizer: miner code never runs at
inference time.

## Decode job

`run_freeform_decode` runs in a work dir under `$QUANT_DECODE_WORK_ROOT/<task_id>_<hotkey[:8]>`
(host paths, so the sandbox can bind-mount them) that is always removed afterwards:

1. Fetch the decoder snapshot from its presigned URL (capped at `DECODER_MAX_XZ_BYTES` + 1 MiB) and check its sha256.
2. Extract it: members must sit under `decoder/` (stripped), no symlinks or hardlinks, the `data` tar filter, at most
   `DECODER_MAX_EXTRACTED_BYTES` in total, and `decode.py` (`DECODER_ENTRYPOINT`) must exist. Both limits live in
   `core/constants/quant.py` and are checked again at registration, so an entrant should only fail them here if the
   snapshot was tampered with.
3. `snapshot_download` the HF artifact; reject symlinks, exec bits and `QUANT_ARTIFACT_FORBIDDEN_SUFFIXES`, then count
   every file (`.gitattributes` included) as `artifact_bytes`.
4. Build and run phases, then `check_decoded_output` against the base model the training job used.
5. Upload each checked `*.safetensors` to `<bucket>/<object_prefix>/`, then `index.json` (`QuantDecodedIndex`) last:
   its presence marks the upload complete. Uploads use 32 MiB parts x 16 in flight under a per-file deadline (`upload_timeout_seconds`); a stalled upload is an `infra_error`.

Each stage, and each uploaded file with its rate, is appended to the job's log in task_history.json, so a long decode
is visible without the Vector log sink.

### Work dir permissions

The trainer runs as a non-root user and owns the work dirs. Both sandbox phases run as that same
uid:gid (`DecodeContainerSpec.user`), with every capability dropped, so:

- The work root is `QUANT_DECODE_WORK_ROOT` (default `~/.local/share/god-trainer/quant_decode`), **not** on the cache
  volume: every training container mounts that volume at `/cache`, so a decode dir there would let a co-tenant read
  the miner's decoder. The root and every work subdir are `0o700`, chmodded explicitly (whatever the umask); a missing
  root is created with the marker file `.god-quant-decode-root` (`0o600`), and parents we create keep default modes
  (a `0o700` root alone blocks traversal). An existing root is chmodded only if it has the marker or is empty (then it
  is adopted and marked); an existing **non-empty** directory without the marker, such as a shared `/data`, is never
  touched: the decode ends as `infra_error` with a message naming `QUANT_DECODE_WORK_ROOT`. The marker is not a task
  dir and is never removed by the work-dir cleanup. Files we write (`decoder.tar.xz`, `index.json`) and the
  downloaded artifact files are `0o600` (dirs `0o700`), applied after `check_artifact_files` so its exec-bit check
  still sees the original modes. Co-tenant containers run as root with `cap_drop=ALL` (no `CAP_DAC_OVERRIDE` /
  `CAP_DAC_READ_SEARCH`), so they only get the "other" bits, which are empty. If the root cannot be
  created, chmodded or written to, the decode ends as `infra_error`, never `quant_failure`.
- Disk: the root holds the decoder, the artifact and the bf16 output at once, so keep roughly 3x the largest bf16
  model free there; set `QUANT_DECODE_WORK_ROOT` on boxes with a small home disk. The base model is still read from
  the cache volume.
- `build/` and `output/` need nothing beyond the owner bits: no 0o777, no ACL, no host setup. They are writable to
  the sandbox (same uid) and closed to everyone else.
- Everything the decoder writes, nested dirs included, is owned by the trainer, so it can check, upload and remove
  it. A dir the decoder `chmod 0`s is still the trainer's: `remove_work_tree` restores owner rwx on every dir it owns
  (never following a symlink) before `rmtree`.
- The uid has no passwd entry in the images, so docker would set `HOME=/` on the read-only root; `HOME=/tmp` points
  it at the tmpfs. The images need nothing else: python, torch and the compiler only read world-readable paths and
  write `/tmp`, `/build` or `/output`.


`_WorkDirs.create` is idempotent: it removes whatever an earlier attempt for the same `<task_id>_<hotkey[:8]>` left
and, if something still survives (a leftover the trainer cannot remove), renames it
aside to `<name>.stale-<hex>` (`job.STALE_WORK_DIR_MARKER`), which only needs write access to the parent. A re-run
therefore never fails with `FileExistsError`; an unusable work root is an `infra_error`. Stale dirs are not swept
automatically: remove `$QUANT_DECODE_WORK_ROOT/*.stale-*` as root.

| Outcome | Field | Examples |
| --- | --- | --- |
| miner's fault, scored | `quant_failure` | unsafe decoder archive, zip bomb, no `decode.py`, executable in the artifact, build/run exit or timeout, bad output |
| ours, never scored | `infra_error` | snapshot unreachable or sha mismatch, HF download failure, base model unreadable, `container error:` (image missing), MinIO upload failure or timeout |

## Re-running a decode

```bash
python -m trainer.quant_decode.rerun <task_id> <hotkey>
```

Run on the trainer that ran the job (it needs the base model in the cache volume and the job in
`task_history.json`). It rebuilds the request from the stored `TrainerTaskLog`, runs the same decode as the training
job (`trainer.runtime.freeform_decode_result`), stores the new `QuantDecodeResult` on the job and prints it as JSON.
Exit code 0 for `ok` or `quant_failure`, 1 for `infra_error`, 2 when the job is unknown or has no decode request.

The validator's orchestrator holds a freeform task until every successful hotkey has a decode outcome and re-polls
the trainer, so after an `infra_error` this re-run is the whole fix. The presigned decoder snapshot URL in the stored
request expires 7 days after dispatch; past that the re-run itself ends in an `infra_error`.

The trainer needs `S3_COMPATIBLE_ENDPOINT`, `S3_COMPATIBLE_ACCESS_KEY`, `S3_COMPATIBLE_SECRET_KEY` and `S3_REGION` in
`.trainer.env` for the upload.
