# Continuation-round train/eval base-model mismatch (env / PvP)

## The bug

Tournament round ≥ 2 env tasks use `TrainingStartPoint.CONTINUATION`. A miner is
**trained** on one base but **evaluated** on a different one, silently dropping
their earlier-round contribution.

Mechanism, end to end:

1. **Assignment.** Each continuation participant's per-miner `starting_model_repo`
   is set to their previous-round output adapter
   (`validator/tournament/tournament_manager.py`, the CONTINUATION block in node
   assignment → `task_sql.set_starting_model_repo`).

2. **Training.** `orchestrator.py` sets
   `training_model = starting_model or task.augmented_model_id or task.model_id`, so
   the trainer receives the miner's previous-round adapter `R_{N-1}`. The trainer
   merges it into the foundation → a full model `M1 = foundation + ΔR_{N-1}`
   (`trainer/utils/trainer_downloader.py:_detect_and_merge_lora`), then trains the
   new LoRA `R_N` **on top of M1**. `R_N`'s weights are therefore relative to `M1`.

3. **Upload.** `trainer/utils/hf_upload.py:patch_model_metadata` only rewrites the
   adapter's `adapter_config.base_model_name_or_path` **string** to the resolved
   foundation. It does **not** re-base the weights. (This same flattening defeats
   the trainer's own chain walk on the *next* round: round N+1 loads `R_N`, follows
   its base pointer straight to the foundation, and merges only `R_N` onto the
   foundation. So the base any round actually trains on is
   `foundation + merge(starting_model_repo)` — a **single** adapter, never a deep
   chain.)

4. **Eval.** `validator/evaluation/scoring.py` sets
   `base_model = task.augmented_model_id or task.model_id` (the bare foundation for
   env tasks) and passes that single shared base to
   `run_evaluation_pvp_pair`. In the container,
   `validator/evaluation/pvp/__main__.py:_prepare_model` serves
   `base_model` + `R_N` adapter via `--enable-lora`, **dropping `ΔR_{N-1}`**.

**Net:** trained on `foundation + ΔR_{N-1} + ΔR_N`, evaluated on
`foundation + ΔR_N`. Earlier-round deltas often establish the tool-call / output
format the later adapter assumes, so a model that plays correctly with its full
weights can forfeit every turn once `R_{N-1}` is dropped. This can flip PvP results.

## The fix (eval-side reconstruction)

Reconstruct, at eval time, the exact base the trainer used:
`foundation + merge(starting_model_repo)`, and serve the miner's own adapter on top.

- `core/models/pvp_models.py` — `PvPModelSpec.base_chain: list[str]`: adapter repos
  to merge onto `original_model` before applying `repo`. Empty for round-1 models.
- `validator/evaluation/scoring.py:_get_continuation_base_chains` — per miner, reads
  `starting_model_repo`; if present and ≠ foundation, sets `base_chain=[starting_repo]`.
  Threaded through `_eval_pvp_envs` → `_get_or_run_pvp_pairs` →
  `run_evaluation_pvp_pair` as per-miner `base_chain_a` / `base_chain_b`.
- `validator/evaluation/pvp/materialize.py` — downloads the foundation and merges the
  chain (reusing the env-eval merge primitives), returning a local base dir.
- `validator/evaluation/pvp/__main__.py:_prepare_model` — for a continuation spec,
  serves the materialized base + the miner's adapter (and resolves the tool-call
  parser from the materialized dir's `config.json`, since a merged dir has no family
  substring in its path).

`base_chain` is typed as a list (not a single string) so eval stays correct if the
upload-side flattening is ever removed and genuine multi-adapter chains appear; in
practice scoring populates it with the single `starting_model_repo`.

### Why eval-side, not upload-side

The alternative — merge `R_N` into `M1 → M2` and upload the full model — makes eval
trivially correct but multiplies upload size every round and requires reworking
winner-lineage handling (`_resolve_winner_base_model`). The eval-side reconstruction
keeps uploads as small adapters and reuses the existing per-model PvP serving path
(each model already gets its own SGLang on its own GPU, and `_prepare_model` already
accepts a per-model `original_model`). The cost is a per-miner merge at eval time —
bounded to a single adapter merge because of the upload-side flattening above.

## Blast radius

| Path | Affected? | Status |
|---|---|---|
| **Env / PvP continuation** (`_create_environment_group_tasks`, round > 1) | Yes | **Fixed here** |
| **Boss round task 1 — env CONTINUATION** (`_create_environment_boss_round_tasks`) | Yes | **Fixed here** — sets per-miner `starting_model_repo` and runs through the same `_run_env_tournament_eval` PvP path |
| **Boss round task 2 — FROM_SCRATCH** | No | round-1-like, served on the foundation |
| **Boss round task 3 — PREVIOUS_WINNER** | Yes, *if* the stored winner repo is a LoRA adapter | **Not fixed** — see below |
| **Text tournaments (instruct / DPO / GRPO)** | Latent (same root cause) | **Not triggered today** — see below |

### Boss round PREVIOUS_WINNER (not fixed)

The `PREVIOUS_WINNER` boss task carries lineage through `task.model_id`
(= `prev_tournament.winner_model_repo`), **not** through per-miner
`starting_model_repo`, so `_get_continuation_base_chains` does not pick it up. If the
stored winner repo is a full merged model (the common case via
`_save_winner_model_repo` / `_resolve_winner_base_model`), eval serves it directly and
there is no mismatch. If the winner repo is a LoRA adapter, the same drop occurs *and*
serving an adapter repo as the SGLang base is itself wrong.

Follow-up (deliberately out of scope for this PR — non-trivial): when `task.model_id`
is itself an adapter, set `original_model` to its resolved foundation and
`base_chain=[task.model_id]` for every miner on the task. This reuses the exact
machinery added here.

### Text tournaments (latent, not triggered)

`validator/evaluation/eval_instruct_text.py` (and DPO/GRPO) load a LoRA submission via
`load_finetuned_model` → `AutoPeftModelForCausalLM.from_pretrained`, which attaches the
adapter to whatever `adapter_config.base_model_name_or_path` says — i.e. the *flattened*
foundation. So **if** text tournaments ran LoRA continuation rounds, they would have the
identical mismatch. They do not: the text task creators
(`_create_group_text_tasks`, `_create_new_text_boss_round_tasks`, …) never set
`TrainingStartPoint.CONTINUATION` or a per-miner `starting_model_repo`, so no text task
is trained on a merged base today. This is documented as latent; if text continuation is
ever enabled, the text eval load path needs the same foundation-reconstruction before
attaching the new adapter.

## Reproduction & regression tests

- `tests/test_continuation_base_mismatch.py` — deterministic, GPU-free proof that
  dropping the round-1 delta flips the model's argmax token, plus a real-code check
  (`core.pvp.chat._parse_tool_calls`) that a missing/garbled tool call forfeits.
- `tests/test_continuation_eval_base.py` — regression tests for the fix: per-miner
  chain derivation, `_prepare_model` serving the reconstructed base (not the bare
  foundation), and a real-transformer LoRA-merge check that sequential merge
  reconstructs the trained model while dropping the prior adapter diverges.
