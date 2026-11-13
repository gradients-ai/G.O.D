# Validator Scoring

Task scoring, tournament scoring, scoring schemas, and metagraph weight setting.

## Contents

- `constants.py`: scoring and emission constants.
- `models.py`: task/tournament scoring models.
- `quant_ranking.py`: quant rank keys and the freeform effective-size penalty.
- `scorecard.py`: the public scorecard of a freeform result (`build_quant_scorecard`, pure; `quant_scorecards_for_task`, the I/O). See "Freeform hub listing and scorecard" below.
- `tasks.py`: miner result scoring and task result processing.
- `tournaments.py`: tournament-specific score-to-weight and pairwise scoring logic. `get_boss_round_pair_weights` pays the paid places; `_resolve_burn_placeholder` maps the boss's place to the defending champion's hotkey when one is known, and `_positions_paid_to_boss` pays a place like the boss's when the stored boss rank comparison says so (rule in `validator/tournament/README.md`, "Crown, early stop and placements").
- `emission_balance.py`: the participation-driven split of the flex pool over text, quant and image.
- `weights.py`: metagraph weight setting entrypoint and allocation logic. The quant champion earns only its base share: no loss-ratio emission boost is computed for quant (freeform is decided on effective bytes, so a loss ratio says nothing about how the crown was won). A share left on the burn hotkey (a kept boss with no known defending champion, and any place paid like it) is left undistributed (`_paid_node_id`) and reaches the burn address with the rest of the undistributed weight, as is a share whose hotkey is not registered. `stored_boss_rank_comparisons` reads the stored boss rank comparisons for the weight setter and the tournament details endpoint alike. `get_tournament_burn_details` raises instead of returning a non-finite pool, so weights are never set from `inf`/`nan`; the weight setter logs it and retries.
- `__init__.py`: package marker.

## Tournament emission split

Three pools: text, quant and image. Constants are in `constants.py`, the allocator in `emission_balance.py`.

- **Minimums and flex.** `TOURNAMENT_TEXT_WEIGHT` 0.15, `TOURNAMENT_QUANT_WEIGHT` 0.35, `TOURNAMENT_IMAGE_WEIGHT` 0.225; the
  remaining `EMISSION_FLEX_POOL` (0.275) is split by `compute_dynamic_base_weights` from the rolling mean entrant count of
  each type's recent completed tournaments (`EMISSION_BALANCE_ALPHA`, `EMISSION_BALANCE_ROLLING_WINDOW`; a type without usable
  history counts as average-subscribed). Each base lies in `[minimum, minimum + flex]` and the bases sum to 1.0. The
  caps `MAX_*_TOURNAMENT_WEIGHT` are minimum + flex (text 0.425, quant 0.625, image 0.50).
- **Always three pools.** The three-way minimums apply at all times; there is no fold of one type's share into another.
  A type with no completed tournament and winner yet has no payee: its share is not paid out and goes to the burn
  address (the undistributed weight added to `EMISSION_BURN_HOTKEY` in `weights.py`).
- **Payees.** Each pool pays its own type's latest completed tournament's top three (`TOURNAMENT_PAID_RANKS`, decay base
  `TOURNAMENT_SIMPLE_DECAY_BASE`) until a newer one of that type completes. Who holds those places, and when a
  contender's share goes to the boss, is in `validator/tournament/README.md`, "Crown, early stop and placements". `get_quant_tournament_weights_from_data` in
  `tournaments.py` is quant's counterpart of the text and image readers; quant fields sit on `TournamentAuditData` and
  `TournamentBurnData` (`validator/tournament/models.py`).

## Quant scoring

- Every quant loss comparison is on the normalised loss, `test_loss / reference_loss`, where `reference_loss` is the
  bf16 reference (the task's shared input model) scored on the row's own GPU SKU (see "Quant SKU normalisation" below). Fixed track ranks by
  normalised loss. Freeform ranks entries at or under the cap, `(1 + bar_cap) x` the reference loss on their SKU,
  by effective size `S = total_bytes x exp(curvature x max(0, r - knee)^2)` (knee 1.04, curvature 2500, cap 1.08) then
  normalised loss; entries over the cap follow by normalised loss, so if nobody is under the cap the closest entry
  wins. The penalty is smooth with zero slope at the knee, so a tiny difference in ratio cannot decide a task, and it
  prices loss in bytes with one pinned curve; byte counts are exact, so the boss margin on `S` needs no bootstrap. A task with no curvature pinned keeps the
  cliff at its `bar_cap`.
- The freeform policy is built per task by `freeform_penalty_policy(task)` (knee, curvature and cap pinned on
  `QuantFreeformRawTask`) in `finalize_task_scores_from_raw_losses`, and applied per row against the row's
  `reference_loss`. A row with a finite loss and no reference on its SKU raises `QuantReferenceUnavailable`;
  lifecycle re-evaluates such rows first (`validator/lifecycle/quant_reference.py`).
- **Chained quant tasks.** Every quant tournament task compresses one shared input: the text boss's output on the
  text task it is linked to (`quant_task_sources`): R1-R3 on the text round tasks, the four boss-round tasks on the
  round-2, round-3 and continuous-SFT tasks. They need nothing special here: the knee
  and cap are those multiples of that input's bf16 loss on each row's SKU, one task-common reference per task, and every
  fixed head-to-head with the boss (R2 and the boss-round fixed tasks) takes the paired gate on `sku_aligned_losses`. The per-hotkey own-model references
  (`OWN_UPSTREAM_MODEL`, `OWN_R1_MODEL`) are supported for tasks that carry such a start point; no quant tournament
  creates one.
- **Fixed-track structural pre-check.** `_evaluate_quant_submissions` computes the expected tensor set once per
  task from `augmented_model_id or model_id` (`validator/evaluation/quant_structure.py`: compressed-tensors
  pack-quantized layout from the base header and config) and compares each repo's hub safetensors headers to it
  before dispatch. A mismatch (missing/extra tensor, wrong dtype or shape, no safetensors) is a scored fail with a
  `structural: ...` reason and is never sent to a GPU; an unreadable hub is an evaluation error (retried). A base we cannot derive a tensor set from (`ValueError`) is an evaluation error for every repo of the task, logged with the base and reason, never a scored fail. A tied base's stored `lm_head.weight` is an optional tensor (absent or BF16 at the embedding shape). Config semantics stay with the evaluator's config check.
- A **scored fail** (`quant_failure` set: the submission did not load, failed the structural check, or is the base
  model) is a result, not an error: `is_finetune` stays True, it ranks after every valid entry, takes
  `SCORE_PENALTY` and a `Scored fail: ` reason. A **no-show** (no repo on the hub, the evaluator's error about the
  repo) keeps the existing `is_finetune=False` failed-result path. An evaluation error is never a result:
  `process_miners_pool` raises `EvaluationInfraError` and lifecycle re-queues the rows
  (`validator/evaluation/README.md`, "Whose fault an evaluation failure is").
- **Scored fail == evaluated.** `evaluate_and_score_hotkeys` lists a scored fail in `evaluated` only, never in
  `failed`, so lifecycle marks it `evaluation_status='success'` and finalize ranks it last; it never trips the
  'no evaluation succeeded' branch. With no valid entry at all, scored fails keep 0.0 and no rank like everyone else.
- `process_miners_pool` reads persisted `task_nodes` rows for quant tasks: a decode-job `quant_failure` becomes a
  scored fail without an HF check or GPU, and the decode job's `decoder_bytes` is carried onto the evaluated
  result (persisting the result rewrites the bytes columns).
- **`quant_failure` is sticky.** Any persisted `task_nodes.quant_failure` - written by the trainer's decode job, by
  the evaluator (e.g. 'submission is the uncompressed base model' or a `structural: ...` mismatch) or by the orchestrator (`NO_DECODER_SNAPSHOT_FAILURE`, "no decoder/ snapshot at
  registration") - makes `process_miners_pool` skip that hotkey on every later run. To re-evaluate one, clear the
  column first: `UPDATE task_nodes SET quant_failure = NULL WHERE task_id = ... AND hotkey = ...`.
- **The finalize write only adds a failure.** `_update_scores` passes each result's `quant_failure` to
  `set_task_node_quality_score`, which writes `COALESCE(new, existing)`: a scored fail set during finalize (a
  reference duplicate) leaves a self-describing row, and a result without one never NULLs a decode-job failure.

- **Calibration rows are never scored.** A freeform task's `calibration_data` (the set published to its miners, see
  `docs/miner_quant.md`) is not read by scoring: the quant eval request builders take `test_data` only. Its one
  evaluation is the bf16 reference pre-pass, stored in `quant_calibration_references` and never ranked on.

## Quant SKU normalisation

Quant evals land on H100 SXM or PCIe (`validator/evaluation/README.md`, "Quant eval GPU SKUs") and each submission
is scored in its own deployment, so two rows on a task may come from different SKUs. Kernel drift between SKUs is
small but lands directly in a loss comparison; every quant comparison therefore divides by a bf16 reference scored on
the same SKU, which cancels a drift that scales served and reference loss alike to first order.

- **Reference model** (`tournament/quant_reference.quant_reference_model(task, hotkey, db)`): the common input
  (`augmented_model_id or model_id`) on both tracks; a task with an own-model start point (`OWN_UPSTREAM_MODEL` or
  `OWN_R1_MODEL`, which no quant tournament creates) uses the miner's own `starting_model_repo` instead.
  `reference_hotkey_for(task, hotkey)` is the hotkey the row's reference is stored under: `''` for a common input, the
  miner's on an own-model task.
- **Cache** (`quant_sku_references`, `validator/db/sql/quant_references.py`): one row per `(task_id, hotkey, sku)`, keyed by
  `(hotkey, QuantEvalGpu.reference_key)` (the SKU label, or `override:<name>/<sm_count>` for a test-box override, which never
  shares a prod SKU's reference). First writer wins. `reference_on(hotkey, eval_gpu, references)` looks up the
  row's own `(hotkey, sku)` and falls back to the task-common `('', sku)`.
- **Same-pod measurement.** `_evaluate_quant_submissions` passes each repo's reference model and its cached `{sku: loss}` map
  to the dispatch (a repo with no reference model is not sent: on an own-model task it is that miner's scored fail
  `NO_REFERENCE_MODEL_FAILURE`, as finalize records it; otherwise an evaluation error); a deployment whose SKU is not cached scores the reference first and returns it as
  `EvaluationResultText.quant_reference`. `_record_quant_references` inserts it; if the SKU was cached meanwhile (two
  pods raced), the stored row is kept and a mismatch (model, loss, vector or fingerprint) is logged and alerted, since
  per-SKU kernels are deterministic.
- **Missing reference = evaluation error.** At finalize, `ensure_quant_references` resets every successful row with
  a finite loss and no reference on its SKU to pending (up to `QUANT_REFERENCE_EVAL_RETRIES`, sharing
  `n_eval_attempts`), so its re-dispatch measures it; spent retries, or no reference model at all, raise
  `QuantReferenceUnavailable` and the task is swapped (`PREP_TASK_FAILURE`). It is never a scored fail.
- **Ranking and the penalty.** `MinerResults.reference_loss` is the reference on the row's SKU, attached by
  `_result_from_persisted_row` and `tournament/task_results.get_task_results_for_ranking`. `adjusted_loss` and
  `rank_metric` (for loss-ranked rows) are the normalised loss, so every reader of `adjusted_loss` (group rounds,
  boss scalars, the emission projection) compares across SKUs. `passed_bar` (under the cap) and `freeform_penalty` (`ln P` as scored) are persisted per row; the tournament
  re-ranks from those two columns, so a retuned constant never moves a scored task. Ties on `S` break on normalised
  loss.
- **Paired gates** (`tournament/thresholds.sku_aligned_losses`): the fixed head-to-heads with the boss, quant
  fixed knockouts, the emission projection and the paired analytics view express the other side's per-example losses
  on the anchor's SKU (the boss; in a knockout, the side with the lower normalised loss):
  `x_i * ref_i(anchor) / ref_i(own)`. That is each side's per-example normalisation by its own SKU reference,
  multiplied back by the anchor's reference so the dead zone and mean-gap floor stay in nats; two rows on one SKU are
  compared unchanged. Without the references the boss keeps the task (`no same-SKU reference to pair on`) and a
  knockout falls back to the mean-loss winner. The scalar check in `challenger_takes_paired_task` uses the normalised
  `adjusted_loss`. The freeform boss gate decides on `passed_bar` and effective size, so it follows from the per-row penalty.

## Freeform hub listing and scorecard

**Hub listing at scoring.** For a freeform row, `_hub_artifact_listing` in `tasks.py` makes one hub call per miner
(`artifact_listing_from_hub` in `validator/evaluation/quant_artifacts.py`: `model_info(files_metadata=True)`). From that
single response it derives:

- `bytes`: the counted artifact size. Every file in the repo counts, `.gitattributes` included.
- `files`: `[{name, bytes, sha256, sha256_kind}]`. For an LFS file `sha256` is the LFS content hash
  (`sha256_kind="sha256"`). A plain git file has no content hash on the hub, so its git blob id stands in
  (`sha256_kind="git-blob-sha1"`), which is not a sha256 of the content.
- `revision`: the hub commit sha the listing was taken at.

`artifact_files` and `artifact_revision` are persisted on `task_nodes` (`validator/db/sql/submissions_and_scoring.py`).
A missing listing never overwrites a stored value with NULL. The fixed track keeps the byte count only. A listing
failure is logged; the byte count then falls back to `_hub_artifact_bytes`.

**`decode_matches_hub`.** `decode_matches_hub(hub_files, decoded_inputs)` in `scorecard.py` is true when the trainer's
decode downloaded and counted exactly the files the hub lists: the same names and bytes, and the same sha256 for every
LFS file. A plain file is compared on name and bytes only, since the hub gives a git blob sha1, not a content hash. It
is false on any difference, and None when either list is unknown (a row with no stored listing).

**The scorecard.** `build_quant_scorecard` builds `QuantFreeformScorecard` (`validator/tasks/models.py`) from one scored
row, the hotkey's participant row and the bf16 reference loss on its SKU. It is pure, so the handover release can reuse
it. `quant_scorecards_for_task` does the I/O and runs only for `QUANTFREEFORMTASK`:

- Only rows that are scored are included: a persisted `test_loss` or `quant_failure` (`is_scored`).
- Decoded shard URLs are presigned on each read, and only once the tournament is COMPLETED. A signing failure logs
  and leaves `urls` null; it never fails the response.
- `release` is set only when the participant has `decoder_released_at` (after the handover release).
- The builder copies participant fields one by one. It never reads or emits `decoder_object_name`, a presigned decoder
  URL, `github_token` or the private training repo URL.

`get_task_with_hotkey_details` (`validator/db/sql/auditing.py`) attaches the scorecard to each `HotkeyDetails` as
`quant_scorecard` for freeform tasks. It returns the quant fixed and freeform response models; the fixed track gets no
scorecard.
