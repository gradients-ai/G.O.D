# Ops Tools

Python and shell tools grouped by operational domain.

## Contents

- `config/`: interactive env-file generation.
- `db/`: database inspection snippets and helpers.
- `datasets/`: dataset generation/formatting tools.
- `evaluation/`: manual evaluation and reward-function tools.
- `god_monitor/`: read-only tournament monitoring CLI (`python -m ops.tools.god_monitor`).
- `observability/`: observability smoke tools.
- `quant/`: `reconcile_freeform_publications.py`, the validator-side CLI for the freeform decoder releases and boss-round decoded models (read-only by default; see below), and `verify_freeform_release.py`, the third-party verifier of a released freeform quant result (decoder snapshot sha256, tree sha vs the released repo, artifact hashes, byte arithmetic, optional decoder re-run); library + CLI, public inputs only.
- `scoring/`: scoring and weight calculation probes.
- `simulations/`: scoring and threshold simulations.
- `tasks/`: task creation and preparation probes.
- `trainer/`: trainer-specific smoke tests and probes.
- `tournament/`: tournament control and inspection tools.
- `__init__.py`: package marker.

## quant/reconcile_freeform_publications.py

Purpose: inspect and re-run the two public freeform publications (the champion's decoder `.tar.xz` + `release.json` in
`<org>/freeform-releases`, and the boss-round `-decoded` HF model repos). These run in the background from the tournament
hooks and hourly from the validator; this is the by-hand entry point and the way to see what a pass would do.

Environment: the validator's `.vali.env` (database, HF token with write on the tournament org, MinIO credentials,
`DISCORD_URL`), run from the repo root with the validator venv.

```
python -m ops.tools.quant.reconcile_freeform_publications                  # report only, changes nothing
python -m ops.tools.quant.reconcile_freeform_publications --apply          # reconcile for real, wait for completion
python -m ops.tools.quant.reconcile_freeform_publications --release --tournament-id T --hotkey H [--allow-boss]
python -m ops.tools.quant.reconcile_freeform_publications --publish-decoded --tournament-id T
```

Add `--json` for machine-readable output. The report lists decoder releases pending, champions blocked for lack of a
`backup_repo` (the code push comes first and is not retried here) and tournaments with unpublished decoded models.

Side effects: only `--apply`, `--release` and `--publish-decoded` write, and they are outward-facing: they create or
update public HF repos/datasets, download GBs from MinIO, write `decoder_released_*` and `task_nodes.decoded_hf_repo`, and
may alert Discord. `--release` goes through the same winner guard as the hooks: a hotkey that is not the tournament's
winner, or that has no `backup_repo`, is refused (exit 1); `--allow-boss` is only for the burn hotkey (the reigning
champion's one-off release). The CLI takes no cross-process lock, so do not run `--apply` while the validator is mid-publication
for the same item; a repeat is harmless (idempotent) but wastes the upload.
