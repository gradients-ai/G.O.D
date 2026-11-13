# Validator scoring

Task scoring, tournament weights and metagraph weight setting. For developers changing how results become scores and scores become emissions.

The compression tournament is `quant` in code. Miner-facing scoring rules are in the miner guides. This file covers the internals.

## Contents

| Path | Purpose |
| --- | --- |
| `constants.py` | Scoring and emission constants. |
| `emission_balance.py` | Splits the floating emission share over the three tournaments by participation. |
| `models.py` | Scoring models for task and tournament results. |
| `organic.py` | Database-free evaluation of a single organic training result. |
| `quant_ranking.py` | Compression rank keys, the normalised loss and the freeform effective-size penalty. |
| `scorecard.py` | The public scorecard of a freeform result. |
| `tasks.py` | Miner result scoring and task result processing. |
| `tournaments.py` | Rank weights for the paid places and pairwise point helpers. |
| `weights.py` | Weight setting: per-tournament pools, paid nodes, burn, audit data. |
| `__init__.py` | Package marker. |

## Tournament emission split

Each tournament's base share is a minimum plus a part of the share that floats by participation. The numbers, the champion boost and the split over paid places are in [Payouts](../../docs/miner.md#payouts). Who holds the paid places is in [Crown, early stop and placements](../tournament/README.md#crown-early-stop-and-placements).

**How the floating share is computed** (`emission_balance.py`):

1. For each of text, image and compression, take the participant counts of its most recent completed tournaments. The boss's participant row is counted.
2. Skip the latest one, so the split a cohort competes under is fixed before it enters.
3. Average the counts. A tournament with no usable history counts as average-subscribed.
4. Give each tournament a part of the floating share proportional to `minimum_share x (minimum_share / participants) ^ alpha`.

A tournament with more participants than its minimum share implies cedes floating share to one with fewer. Each base lies between its minimum and its minimum plus the whole floating share, and the three bases sum to 1.0.

These bounds apply to the base share. How the champion boost composes with them is in [Payouts](../../docs/miner.md#payouts).

| Constant | Value | Meaning |
| --- | --- | --- |
| `EMISSION_BALANCE_ROLLING_WINDOW` | 4 | Completed tournaments averaged, after the lag. |
| Lag | 1 | The latest non-cancelled tournament (pending, active or completed) is excluded. |
| `EMISSION_MIN_ENTRANTS_FOR_BALANCE` | 3 | Tournaments with fewer participants are ignored. |
| `EMISSION_BALANCE_ALPHA` | 1.25 | Reaction strength. At 0 the floating share splits at the minimums' ratio. |

**Undistributed weight goes to the burn node.** That covers:

- the pool of text, image or compression when it has no completed tournament and winner
- the part of a pool not paid out to its places
- a share left on the burn hotkey because no defending champion is known
- a share whose hotkey is not registered

`get_tournament_burn_details` raises on a non-finite pool, so weights are never set from `inf` or `nan`. The weight setter logs it and retries.

## Compression scoring

How a compression task is ranked, and the effective-size formula, are in [Scoring](../../docs/miner_compression.md#scoring). What the code adds:

| Rule | Detail |
| --- | --- |
| Reference | The loss each comparison uses is divided by the input model's bf16 loss on the row's GPU SKU ([Compression SKU normalisation](#compression-sku-normalisation)). |
| Policy | `QUANT_PENALTY_KNEE`, `QUANT_PENALTY_CURVATURE` and `QUANT_BAR_CAP` are pinned on the task at creation. |
| Cap | A freeform row is under the cap when its loss is at most `(1 + QUANT_BAR_CAP)` times the reference loss on its SKU, using the value pinned on the task's `bar_cap` column. In the target published to miners this is `cap_loss`. The published `bar_cap` and `bar` are the knee (`penalty_knee`). |
| Shared input | Every tournament task compresses one input model, so each task has one reference per SKU. |
| Calibration rows | Never scored. Scoring reads `test_data` only. |

`passed_bar` and the penalty are stored per row, and tournament ranking reads the stored values. Retuning a constant does not move a scored task.

A fixed head-to-head against the boss uses the paired per-example test. Its definition is in [the tournament README](../tournament/README.md#the-paired-per-example-test).

### Result kinds

| Kind | Cause | Effect |
| --- | --- | --- |
| Scored fail | `quant_failure` is set: the submission did not load, failed the structural check, is the base model, or failed its decode. Also a freeform entry under the cap with no byte count. | A result. It ranks after every valid entry and takes `SCORE_PENALTY`. |
| No-show | No repo on the hub. | The failed-result path, `is_finetune=False`. |
| Evaluation error | Infrastructure trouble. | Never a result. The rows are re-queued, and held for an operator after the retry cap. |

Failure attribution is in [Whose fault an evaluation failure is](../evaluation/README.md#whose-fault-an-evaluation-failure-is).

<details>
<summary>Scored fail handling</summary>

- A scored fail counts as evaluated, not failed, so lifecycle marks it `evaluation_status='success'` and finalisation ranks it last.
- With no valid entry at all on a task, scored fails keep 0.0 and no rank.
- A persisted `quant_failure` is sticky: `process_miners_pool` skips that hotkey on every later run, without a hub check or a GPU.
- To re-evaluate such a row, clear the column first: `UPDATE task_nodes SET quant_failure = NULL WHERE task_id = ... AND hotkey = ...`.
- The write at finalisation only adds a failure. It writes `COALESCE(new, existing)`, so a result without one never clears a decode-job failure.
- The decode job's `decoder_bytes` is carried onto the evaluated result.

</details>

<details>
<summary>Fixed-track structural pre-check</summary>

- The expected tensor set is computed once per task from the input model (`validator/evaluation/quant_structure.py`).
- Each repo's hub safetensors headers are compared with it before dispatch.
- A mismatch (missing or extra tensor, wrong dtype or shape, no safetensors) is a scored fail with a `structural: ...` reason. It is never sent to a GPU.
- An unreadable hub is an evaluation error and is retried.
- An input model no tensor set can be derived from is an evaluation error for every repo of the task.
- On an input with tied embeddings, a stored `lm_head.weight` is optional.

</details>

## Compression SKU normalisation

Compression evaluations run on more than one H100 SKU ([Compression eval GPU SKUs](../evaluation/README.md#compression-eval-gpu-skus)), and each submission is scored in a deployment of one SKU. Kernel differences between SKUs are small but land directly in a loss comparison. Dividing by a bf16 reference scored on the same SKU cancels them to first order.

| Part | Rule |
| --- | --- |
| Reference model | The task's input model (`tournament/quant_reference.py`). |
| Cache | `quant_sku_references`: one row per tournament task and SKU. The first writer wins. |
| Measurement | A deployment whose SKU is not cached scores the reference first and returns it with the result. |
| Missing reference | An evaluation error, never a scored fail. |
| Ranking | `adjusted_loss` is the normalised loss, so every reader of it compares across SKUs. |
| Paired comparisons | The other side's per-example losses are expressed on the anchor's SKU: `x_i * ref_i(anchor) / ref_i(own)`. |

<details>
<summary>Details</summary>

- **Raced deployments.** If two deployments measure one SKU, the stored row is kept. A mismatch between them is logged and alerted, since per-SKU kernels are deterministic.
- **Missing at finalisation.** Rows with a finite loss and no reference on their SKU are reset to pending so their re-dispatch measures it, up to `QUANT_REFERENCE_EVAL_RETRIES`. After that a non-seed task is swapped (`PREP_TASK_FAILURE`). A seed task is held and alerted once.
- **Paired alignment.** Used by the fixed head-to-heads with the boss, the emission projection and the paired analytics view. In a head-to-head the anchor is the boss.
- **Why multiply back.** Each side is normalised by the reference of the SKU it ran on, then multiplied by the anchor's, so the dead zone and mean-gap floor stay in nats. Two rows on one SKU are compared unchanged.
- **No reference to align on.** The boss keeps the task.
- **Freeform.** The freeform head-to-head is decided on `passed_bar` and effective size, both stored per row.

</details>

## Freeform hub listing and scorecard

**Hub listing.** For a freeform row, scoring makes one hub call per miner and stores what it returns on `task_nodes`.

| Field | Content |
| --- | --- |
| `bytes` | The counted artifact size: the evaluator's count, else the listing's. In the listing every file in the repo counts, `.gitattributes` included. |
| `artifact_files` | `name`, `bytes`, `sha256`, `sha256_kind` per file. |
| `artifact_revision` | The hub commit the listing was taken at. |

For an LFS file `sha256` is the content hash. A plain git file has no content hash on the hub, so its git blob id stands in (`sha256_kind="git-blob-sha1"`). A missing listing never overwrites a stored value. The fixed track stores the byte count only.

**`decode_matches_hub`.** True when the trainer's decode downloaded exactly the files the hub lists: the same names and bytes, and the same sha256 for every LFS file. False on any difference. `None` when either list is unknown.

**The scorecard.** `build_quant_scorecard` builds the public record of one scored freeform row from the row, the participant's decoder fields and the reference loss on its SKU.

- Only scored rows are included: a stored `test_loss` or `quant_failure`.
- Decoded shard URLs are presigned on each read, and only once the tournament is completed.
- `release` is set once the participant's decoder has been released.
- It never emits the decoder object name, a presigned decoder URL, the GitHub token or the private training repo URL.
- The auditing task endpoint attaches it to each hotkey of a freeform task as `quant_scorecard`. The fixed track has no scorecard.

## See also

- [Validator tournament](../tournament/README.md): crown rule, placements, seeds, orchestration.
- [Validator evaluation](../evaluation/README.md): evaluation internals and GPU SKUs.
- [Miner payouts](../../docs/miner.md#payouts): emission numbers and the split over places.
