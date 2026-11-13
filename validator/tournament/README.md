# Validator tournament

The tournament state machine: scheduling, enrolment, rounds, the boss's seed tasks, trainer orchestration, the crown rule and placements. For developers and operators.

The compression tournament is `quant` in code (files, constants, task types). Miner-facing rules are in the miner guides: [text](../../docs/miner_text.md), [image](../../docs/miner_image.md), [compression](../../docs/miner_compression.md).

## Contents

| Path | Purpose |
| --- | --- |
| `benchmark_utils.py` | Benchmark task copies for a tournament winner. |
| `boss_rank_shares.py` | Per-contender rule: each paid contender's average rank against the boss's. |
| `brackets.py` | Bracket creation and advancement helpers. |
| `chaining.py` | Read-only helpers for chained compression tasks (`upstream_task_id`). |
| `challenger_code_review.py` | Automated code review of every miner entering round 2. |
| `constants.py` | Schedule, round plans, fees, crown and paired-test constants, retry limits. |
| `cost_calculator.py` | Weekly training, prep and evaluation cost roll-up. |
| `cost_constants.py` | GPU hourly prices and fee conversions for the cost roll-up. |
| `crown.py` | The crown rule as pure functions: one head-to-head, and the verdict over them. |
| `decoder_snapshot.py` | Canonical snapshot of a miner's `decoder/` at its pinned commit. |
| `dedup_gate.py` | Duplicate-submission gates before round 1 training and before round 2. |
| `dstack_orchestrator.py` | dstack-backed training orchestration. |
| `freeform_release.py` | Public release of the champion's decoder and the boss-round decoded models. |
| `github_validation.py` | Repository, licence, commit and obfuscation checks at enrolment. |
| `gpu_requirements.py` | GPU requirement calculation for tournament tasks. |
| `models.py` | Tournament schemas: rounds, seeds, source links, rank comparisons, responses. |
| `notifications.py` | Discord notifications. |
| `obfuscation_detection/` | Bundled detector binary used by repository validation. |
| `orchestrator.py` | Trainer assignment, scheduling, monitoring, chained release, adapter merges. |
| `participants.py` | Participant lookups: the boss, the challenger, the pre-boss knockout loser. |
| `performance_calculator.py` | Boss-versus-challenger performance differences for analytics and projections. |
| `performance_utils.py` | Shared performance helpers. |
| `quant_reference.py` | Which bf16 model a compression result is normalised by. |
| `quant_reference_prepass.py` | Measures each task's bf16 reference before miners train. |
| `quant_sources.py` | The link between a compression tournament and its text tournament. |
| `registration_lint.py` | Build lint for immutable dependencies at compression enrolment. |
| `repo_diff_report.py` | Repository diff report for a completed tournament. |
| `repo_diff_report_config.json` | Diff report configuration. |
| `repo_uploader.py` | Publishes the winning repository. |
| `reports.py` | Tournament report generation. |
| `round_results.py` | Round standings, advancement, head-to-heads against the boss, third place. |
| `runner.py` | Tournament loop runner. |
| `seed_tasks.py` | Seed tasks: creation, status, replacement, trainer priority. |
| `task_creator.py` | Task creation per round and task replacement. |
| `task_results.py` | Loads task results for ranking. |
| `thresholds.py` | The paired per-example test, SKU alignment, win-margin helpers. |
| `tournament_manager.py` | Lifecycle: scheduling, enrolment, round creation, advancement, completion. |
| `transfer_monitoring.py` | Fee transfer monitoring. |
| `utils.py` | Remaining shared helpers. |
| `__init__.py` | Package marker. |

## Text tournament flow

1. **Seeds.** When the first round is created, the four text seeds are created and the boss trains them with its pinned code ([Seeds](#seeds)).
2. **Round 1.** The round attaches `text_r1`, so miners train it in parallel with the boss.
3. **Entry gates.** Before round 2 is built, the advancing miners pass the duplicate-submission gate and a code review. A disqualified miner is replaced by the next miner in the round-1 standings, who is checked too. A duplicate flag, or a review that needs an operator, holds the tournament until it is resolved.
4. **Rounds 2 and 3.** Each attaches its seed task and is played by one group.
5. **Early stop.** The check after round 3 can end the tournament ([Early stop](#early-stop)).
6. **Boss round, stage one.** The continuous seed is attached and one GRPO and one DPO task are drawn. Who trains each task is in the [text guide](../../docs/miner_text.md#the-boss-round).
7. **Boss round, stage two.** The big instruct task is created last, under the same early-stop check.
8. **Crown and placements.** See [Crown, early stop and placements](#crown-early-stop-and-placements).

Round 1 advances on its task. Rounds 2 and 3 advance on average rank over every round played so far, ranked among the surviving miners, with the same tie-breaks as compression. Model sizes and how many miners advance are in the [text guide](../../docs/miner_text.md#rounds) (`TEXT_ROUND_PLAN`).

## Image tournament flow

1. **Round 1.** One group plays one text-to-image task, with the boss assigned to it alongside the miners. Its model type is drawn from `ROUND_ONE_IMAGE_MODEL_TYPES` and its size is capped at `ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS` pairs. The best 8 miners advance (`IMAGE_ROUND_PLAN`).
2. **Entry gates.** As in the text flow, before round 2 is built.
3. **Rounds 2 and 3.** Each is one group on one task, with the boss assigned to it: a slot drawn from `ROUND_TWO_IMAGE_TASK_SPECS` (top 4) in round 2, a slot drawn from `FINAL_ROUND_IMAGE_TASK_SPECS` (top 1) in round 3. A slot is a (model type, training mode) pair.
4. **Early stop.** The check after round 3 can end the tournament ([Early stop](#early-stop)).
5. **Boss round.** Four tasks are created together: `IMAGE_BOSS_ROUND_TASKS_PER_MODE` of each training mode, two edit and two text-to-image, drawn from the `FINAL_ROUND_IMAGE_TASK_SPECS` slots other than round 3's. The round-3 slot is read from round 3's task when the boss round is created; a rerun keeps the slots the round already holds and draws only for missing tasks. The boss and the challenger both train them.
6. **Crown and placements.** See [Crown, early stop and placements](#crown-early-stop-and-placements).

Round 1 advances on its task. Rounds 2 and 3 advance on average rank over every round played so far, ranked among the surviving miners, with the same tie-breaks as text.

The image tournament has no seed tasks: the boss trains each round's task in the same window as the miners. Task details are in the [image guide](../../docs/miner_image.md#rounds).

## Compression tournament flow

1. **Binding.** The compression tournament binds to one text tournament ([Sources](#sources)). Until one qualifies it waits.
2. **Seeds.** When the first round is created, the seven compression seeds are created. Each is held until the text boss's model for it has landed, then trained by the compression boss ([Seeds](#seeds)).
3. **Round 1.** A freeform task on the text boss's round-1 model. All enrolled miners play.
4. **Entry gates.** As in the text flow, before round 2 is built.
5. **Round 2.** A fixed task on the round-2 model. After the cut, the viability cut drops any miner that cannot finish top or joint top after round 3.
6. **Round 3.** A freeform task on the round-3 model.
7. **Early stop.** The check after round 3 can end the tournament ([Early stop](#early-stop)).
8. **Boss round.** The four boss-round seeds are attached ([Seeds](#seeds)). The tasks and who trains them are in the [compression guide](../../docs/miner_compression.md#the-boss-round).
9. **Crown and placements.** See [Crown, early stop and placements](#crown-early-stop-and-placements).

Each round is one task attached from its seed, so the miners and the boss compress the same model. Formats, advancement and tie-breaks are in the [compression guide](../../docs/miner_compression.md#rounds) (`QUANT_ROUND_PLAN`).

## Crown, early stop and placements

The crown rule below decides every tournament. Miners read the rule in plain words in [The boss and the crown](../../docs/miner.md#the-boss-and-the-crown). What counts as a win on each task is in the miner guides: [text](../../docs/miner_text.md#the-crown), [image](../../docs/miner_image.md#the-crown), [compression](../../docs/miner_compression.md#the-crown).

### The boss in rounds 1-3

The boss is the tournament's defending champion. It has an entry on the task of each of rounds 1-3: the seed task it trained in text and compression, the round's own task in image, which it trains with the miners. That entry is ranked in the round's standings with the ordering used between miners. A boss entry with no valid result ranks behind every valid entry. The boss never takes an advancement slot. Its three entries are also its side of the first three head-to-heads.

### Seven head-to-heads

The challenger is the one miner left after the round-3 cut. It is compared with the boss on 7 tasks (`CROWN_TASKS`): the tasks of rounds 1-3 and the 4 boss-round tasks. Each head-to-head has one of four outcomes (`crown.HeadToHeadOutcome`).

| Outcome | Meaning |
| --- | --- |
| `win` | The challenger takes the task. |
| `within_smaller` | A non-win where the challenger is better than the boss, short of a win. |
| `within_larger` | A non-win where the challenger is level with the boss, or worse by at most `CROWN_LARGER_ALLOWANCE` of the boss's value. |
| `beyond` | A non-win where the challenger is worse by more than that. |

The challenger is crowned when exactly these 7 are decided and all of the following hold:

- it has at most two non-wins (`CROWN_MAX_NON_WINS`)
- no non-win is beyond the larger allowance
- with two non-wins, at least one is within the smaller allowance

So a crown is one of three results:

| Wins | Non-wins |
| --- | --- |
| 7 | none |
| 6 | one, within either allowance |
| 5 | two, each within an allowance, at least one of them within the smaller |

Any other set of head-to-heads leaves the boss in place.

What a win is depends on the task. A non-win is banded the same way on every task, on the compared value alone (score, scalar loss or effective bytes): the paired test and the reason the task was not won do not move it. The constants that hold each tournament's numbers, all in `constants.py`:

| Tournament | Task | Win | Non-win bands |
| --- | --- | --- | --- |
| Text | instruct, continuous (chat), DPO | paired test and `TEXT_CROWN_WIN_MARGIN` on the score | `CROWN_LARGER_ALLOWANCE` on the score deficit |
| Text | GRPO | `TEXT_CROWN_WIN_MARGIN` on the score | as above |
| Image | image | `IMAGE_CROWN_WIN_MARGIN` on the weighted loss | `CROWN_LARGER_ALLOWANCE` on the loss |
| Compression | freeform | `QUANT_FREEFORM_BOSS_BYTE_MARGIN` fewer effective bytes | `CROWN_LARGER_ALLOWANCE` on extra effective bytes |
| Compression | fixed | paired test and `QUANT_FIXED_BOSS_MIN_IMPROVEMENT` on the loss | `CROWN_LARGER_ALLOWANCE` on the scalar loss |

An image crown task on which both entries trained and neither has a valid result is not decided: the evaluation data is missing (`round_results.CrownEvaluationFault`). The round is held and an alert is sent once, at the round-3 early-stop check and in the boss round, and it resolves on its own once the results exist.

### The paired per-example test

`thresholds.compare_paired_losses` compares the two entries example by example on the identical held-out rows. The gap on an example is the boss's loss minus the challenger's.

| Quantity | Rule | Constant |
| --- | --- | --- |
| Decided example | the gap exceeds 0.01 nats either way | `BOSS_ROUND_TIE_DEADZONE_NATS` |
| Win rate | the challenger wins at least 55% of decided examples | `BOSS_ROUND_MIN_WIN_RATE` |
| Mean gap | at least 0.01 nats, or 1% of the boss's mean loss if larger, over all comparable examples | `BOSS_ROUND_MIN_MEAN_GAP_NATS`, `BOSS_ROUND_WIN_MARGIN` |
| Confidence | both statistics must hold at the one-sided 99% bootstrap lower bound | `BOSS_ROUND_BOOTSTRAP_CONFIDENCE` |

The bootstrap resamples examples 10,000 times with a fixed seed, so every validator reaches the same verdict. Examples without a finite loss on both sides are dropped. A draw is a comparison in which no example is decided.

The paired test is applied against the boss only. Miners are ranked against each other on each round's standings.

For a fixed compression task scored on two GPU SKUs, the challenger's per-example losses are first expressed on the boss's SKU ([Compression SKU normalisation](../scoring/README.md#compression-sku-normalisation)).

### Early stop

Every one of rounds 1-3 runs in full. The rounds decide second and third place, and each round's task is one of the crown's head-to-heads, so a round is played even with one miner left. The crown is checked once the round-3 cut leaves the challenger. The rule is applied to its three head-to-heads, with the four boss-round tasks counted as wins. If the crown is out of reach, no boss round is created and the tournament ends with the boss retained.

In the text boss round the big task runs last. It is created only if the crown is in reach after the continuous, GRPO and DPO tasks.

### Placements

| Result | 1st | 2nd | 3rd |
| --- | --- | --- | --- |
| Crown | challenger | boss | third-place miner |
| No crown | boss | challenger | third-place miner |

The placements are the same whether the boss round was played and lost or skipped by the early stop.

| Tournament | Third place |
| --- | --- |
| Text | The next miner after the challenger in the cumulative standings after round 3. |
| Image | The next miner after the challenger in the cumulative standings after round 3. |
| Compression | The next miner after the challenger in the cumulative standings after round 3. |

When the compression viability cut left the challenger alone in round 3, the round-2 advance set is used. Third is its best other miner on the round 1 and 2 standings, re-ranked within that set.

How the paid places split a tournament's emission is in [Payouts](../../docs/miner.md#payouts).

### Per-contender rule

A contender is a miner in a paid place (2nd or 3rd). When the boss keeps the crown, each contender is compared with the boss on average rank over the rounds of 1-3 that rank both.

- If the boss's average rank is strictly better, that place's share goes to the defending champion's hotkey.
- On a tie, or if the contender ranks better, the contender is paid.

It is decided per contender. If the boss outranks the 3rd-place contender and not the 2nd, 2nd is paid and 3rd's share goes to the defending champion.

The boss's 1st-place share goes to the same hotkey (`base_winner_hotkey`). A share reaches the burn address only when no defending champion hotkey is known or that hotkey is not registered.

The comparison is computed once at completion and stored in `tournament_boss_rank_comparisons`. The weight setter and the tournament details endpoint both read the stored rows.

<details>
<summary>Edge cases</summary>

- **No valid challenger result.** The task is beyond both allowances, so it rules out the crown.
- **No valid boss result.** With a valid challenger result the task is a win. On freeform, a boss entry over the cap, a scored fail or a missing byte count is not valid.
- **Boss-side seed failure.** The boss side of a compression boss-round task has failed in two cases: the task failed prep after attach with no successful boss run, or the task failed with the boss's run failed. Such a task is a win for the challenger, without a comparison, when the challenger's run there succeeded with no scored fail. If neither side produced a result, the challenger has no valid result and the task is beyond both allowances.
- **Failed seed role.** A round 1-3 slot or a compression boss-round slot plays a task drawn with no boss entry, so any valid challenger result wins it. For `text_continuous`, the boss and the challenger both train the drawn task and it is judged as usual. Either way the crown is decided over seven.
- **Unpairable losses, text.** Missing vectors, different eval sets or no comparable example: the task is a non-win banded on the score.
- **Unpairable losses, fixed.** The same cases, or no same-SKU reference to align on: the task is a non-win banded on the scalar loss, unless the boss has no valid result. A task the boss takes on the paired test is banded the same way.
- **Boundaries.** A value exactly on a win margin or exactly `CROWN_LARGER_ALLOWANCE` worse than the boss meets it (`CROWN_BOUNDARY_RELATIVE_TOLERANCE`), as does the freeform byte bar. Level with the boss is within the larger allowance, never the smaller one: the smaller allowance needs a strictly better value, with no tolerance. The paired test's bounds have no tolerance.
- **Wrong task set.** More than 7 decided, or a compression boss round that is not rounds 1-3 plus its four tasks, never crowns.
- **Undecided round, compression.** A round 1-3 that does not hold exactly one task cannot be decided, so the early stop retains the boss.
- **Missing round task, text.** Treated as a data fault: the check raises and the tournament does not advance.
- **Ranks with the boss.** An exact score tie shares one rank, and so do all entries with no valid result. A tie with the boss therefore pays the contender.
- **Comparison unavailable.** If no round ranks both, or the comparison cannot be computed or stored, the contender is paid.
- **Third place.** If no other miner is ranked, or the computed miner already holds 1st or 2nd, third place is empty.
- **No miner left after the entry gates.** The boss retains.

</details>

## Seeds

A seed is a task the boss trains with its pinned repo and commit, created when the tournament's first round is created (`seed_tasks.py`, table `tournament_seed_tasks`). A round attaches its seed as the round's task, which adds the miners.

| Role | Tournament | Task | Attached by |
| --- | --- | --- | --- |
| `text_r1`, `text_r2`, `text_r3` | text | instruct | rounds 1, 2 and 3 |
| `text_continuous` | text | continuous task | boss round |
| `quant_r1` | compression | compression of the `text_r1` model | round 1 |
| `quant_r2` | compression | compression of the `text_r2` model | round 2 |
| `quant_r3` | compression | compression of the `text_r3` model | round 3 |
| `boss_r2_freeform` | compression | compression of the `text_r2` model | boss round |
| `boss_r3_fixed` | compression | compression of the `text_r3` model | boss round |
| `boss_continuous_freeform`, `boss_continuous_fixed` | compression | compression of the `text_continuous` model | boss round |

**Lifecycle.** `training` (the boss's run is not finished), `trained` (the boss's run is over and the seed is held for its round), `ready` (the task has completed). A superseded attempt is `replaced`. A role out of replacements is `failed`.

**Limits.**

| Limit | Constant | Value |
| --- | --- | --- |
| Charged replacements per role | `SEED_MAX_REPLACEMENTS` | 2 |
| Uncharged prep-stage dataset rejections of a text round seed | `SEED_MAX_PREP_REJECTIONS` | 20 |
| Task-level model-prep failures before a seed is redrawn | `SEED_MAX_PREP_ATTEMPTS` | 3 |

The 20th prep-stage rejection marks the role `failed`.

**Visibility.** Seed data is hidden on public and auditing endpoints.

- A text seed is hidden until a round attaches it or its text tournament ends, and while a running compression tournament has a live task on it.
- Compression seeds, and compression tasks chained on a text seed, stay hidden until their compression tournament is completed or cancelled.

<details>
<summary>Replacement and failure handling</summary>

- A seed whose task dies before its round attaches it is redrawn as the next attempt, and held tasks chained on it are re-pointed to the replacement.
- A prep-stage dataset rejection of a text round seed (`text_r1` to `text_r3`) is not charged. The redraw skips every dataset already rejected in the tournament.
- A failed boss training run on a compression seed is not redrawn. It is the boss's failed entry and ranks behind every valid entry.
- A failed boss run on a text seed is redrawn while the seed is unattached and replacements are left. Otherwise it is the boss's failed entry.
- Once attached, a round seed is replaced only for a prep failure, and the replacement takes its place in the round.
- Once attached, a boss-round compression seed is never replaced. See "Boss-side seed failure" above.
- A boss upload that is a LoRA adapter is not a failure. The seed waits for its merge into a full model.
- A `failed` role never halts the tournament. Its round draws a task for that slot with no boss entry. For `text_continuous`, the boss round draws a continuous task that the boss and the challenger both train.
- While a role's latest row is `replaced`, its round waits in `PENDING`.
- Once a compression tournament is decided, its unattached seeds that the boss has not started are retired. The others finish and close with the boss's stored result. None is replaced.
- A role out of replacements raises one Discord alert per tournament and role, per validator process.
- The fixed format of a compression boss-round seed is drawn from `QUANT_BOSS_ROUND_FIXED_FORMATS`, seeded by tournament and role, so a redraw gets the same format.

</details>

## Sources

`quant_sources.py` links a compression tournament to the text tournament whose models it compresses (tables `quant_tournament_sources`, `quant_task_sources`).

| Rule | Detail |
| --- | --- |
| Binding | The most recent text tournament that has its four seed roles and feeds no other compression tournament. |
| Order | An active text tournament first, else one completed within 7 days (`QUANT_SOURCE_COMPLETED_WINDOW_SECONDS`). |
| Waiting | Nothing is created until a binding exists. One Discord alert after 2 hours (`QUANT_SOURCE_WAIT_ALERT_SECONDS`), measured from when the validator process first saw it waiting. |
| Task link | Each compression task records its slot, text seed task and, once released, the exact model repo. |
| Fallback | When the text boss produced no usable model, the task compresses the text task's base model (`source_is_fallback`). |

<details>
<summary>When the fallback is taken</summary>

- At once for a `failed` text seed role. A full model the boss did land is used first.
- At once when the text tournament is over.
- A failed text boss run takes the fallback once its seed row will not be replaced. Before that the task holds and follows the replacement.
- While the text tournament runs and the boss's run succeeded with a missing or non-full-model output: only after the seed row is `trained` or `ready` and unchanged for 30 minutes (`QUANT_SOURCE_FALLBACK_GRACE_SECONDS`).
- An adapter awaiting its merge, and a run that has not finished, hold the task.
- For the continuous task the fallback is its base model if that is a full model, else the lineage's seed model.
- If no fallback candidate is a full model, the task holds and one alert goes out.
- A text seed role whose replacement is pending makes binding and task creation wait.
- A task on a failed seed whose text task was never prepped is released on a freshly drawn instruct dataset.

</details>

## Orchestration

`orchestrator.py` runs these cycles side by side.

| Mechanism | Rule |
| --- | --- |
| Queue buckets | Tournament tasks reach training through three buckets: image, compression, and everything else. Each is fed while it has fewer than 8 pending assignments (`PENDING_QUEUE_THRESHOLD_PER_TYPE`). |
| Reference pre-pass | Each compression task's bf16 reference is measured once before miners train, or copied from a task with the same model and a byte-identical eval set. Training waits for it. |
| Chained release | A compression task is held in `AWAITING_UPSTREAM` until the text boss's model has landed and its nodes are assigned. That repo becomes its `model_id`. |
| Seed priority | The compression boss's boss-round entries rank last. In a cycle where an earlier row found no GPUs, they are not tried. |
| Seed hold | A seed is not finalised until its round has attached it and every assigned miner has finished. |
| Adapter merge | A boss output that is a LoRA adapter is merged into a full model by a trainer model-prep job, and the merged repo becomes the seed's output. |

<details>
<summary>Details and failure handling</summary>

- **Released tasks.** A released task copies the text task's prepped split, calibration data and dataset type, and its training budget is recomputed. The boss and the challenger get the same budget on a boss-round task.
- **Input repo.** The model is always the validator-assigned repo of the text boss's run, never a miner-supplied id.
- **Reference failures.** A failed measurement is an infra error, retried every cycle for as long as it takes; the rows wait. An alert goes out at every third consecutive failure.
- **Calibration reference.** The prepass measures the reference bf16 model on the calibration rows after its test-set reference (per hotkey on an own-model task, once for the task otherwise, or copied from the previous round when the model and the calibration set are identical). It is never used to score. A tournament row waits for it. A stored reference measured on another calibration set than the task's current one counts as missing and is replaced by the re-measurement. A row is never dispatched without its target, its calibration set and the reference losses on it; one that cannot be given all three is held and alerted.
- **Counters.** The failure counters are in memory. A validator restart resets them.
- **Priority.** A challenger's row is never deferred by the seed priority rule, and a running job is never preempted.
- **Seed-time evaluation.** The boss's entry is evaluated as soon as it has trained, so the round evaluates only the miners. Retry rules are in `validator/lifecycle/boss_seed_eval.py`.
- **Challenger rows.** A cycle adds training rows for the miners assigned to each attached seed. A boss entry that has not finished is logged at WARNING. The round does not wait for it.
- **Merge failures.** The seed waits and is never redrawn or charged. Retries back off from 10 to 60 minutes. Three consecutive failures raise an alert. After six the merge is parked until the orchestrator restarts.
- **Decode results.** A freeform training's decode result is stored on `task_nodes`. Only a miner fault is scored: no decoder snapshot, or the trainer's `quant_failure`. An infra error, a missing result or a snapshot mismatch holds the task and alerts once.
- **Isolation.** A compression row that raises during scheduling is dropped for that cycle, so it does not hold text and image rows.
- **Logging.** Trainer requests are logged with the GitHub token and the presigned decoder URL masked, and URL queries stripped.

</details>

## Enrolment checks

Checks shared by every tournament are in [Entering a tournament](../../docs/miner.md#entering-a-tournament). Compression adds two steps on one clone of the pinned commit. The miner-facing rules are in [Enrolment checks](../../docs/miner_compression.md#enrolment-checks).

| Step | Module | Rule |
| --- | --- | --- |
| Build lint | `registration_lint.py` | Checks the build for hash-pinned pip, SHA-pinned git, checksummed fetches and pinned base images. `QUANT_REGISTRATION_LINT_ENFORCE` is `False`: violations are logged and the miner is enrolled. |
| Decoder snapshot | `decoder_snapshot.py` | Archives `decoder/` with `git archive`, compresses it with single-threaded `xz`, and uploads it to MinIO. A miner without a usable snapshot is excluded. |

<details>
<summary>Decoder snapshot rules</summary>

- The snapshot is taken once, at enrolment. A freeform decode only ever runs that tar, never a fresh clone.
- A miner is excluded for: no `decoder/`, a symlink or hardlink inside it, a compressed size over `DECODER_MAX_XZ_BYTES`, no `decoder/decode.py`, extracted members over `DECODER_MAX_EXTRACTED_BYTES`, or a clone or `xz` timeout.
- At most `DECODER_SNAPSHOT_CONCURRENCY` (4) snapshots run at a time.
- The boss is snapshotted and never linted. A boss without a usable decoder keeps empty snapshot fields and scored-fails every freeform task.
- A transient boss snapshot failure is retried while the tournament is pending, no more often than every 10 minutes. A boss with no usable decoder raises one alert and is not retried.
- `decoder_tree_sha` is the git tree id of `decoder/`. It depends on content only, so it matches the published repository after the winner's code is pushed.
- Files over `QUANT_LARGE_FILE_REVIEW_BYTES` are listed for manual review and never fail the lint.

</details>

## Design notes

- **Boss-trained inputs are fair.** The models the compression tournament compresses are trained by the text boss on the validator's draw of base model and dataset. The text boss's code is public, so every miner can read what trained the input. The compression boss and the miners compress the same model.
- **Boss entries are visible, not usable.** The compression boss's entries are public on the hub before the challenger trains. Training containers have no network and run pinned code, so this allows offline study only.
- **Hold rule.** A seed is a `TRAINING` task whose only row is the boss's until its round attaches it. It is ranked and closed once, when every assigned hotkey has a terminal training row or a persisted `quant_failure`. A seed of a decided tournament that no round attached closes with the boss's stored result.
- **Chained release does not wait on the hold.** It keys on the text boss's model having landed as a full model. The compression boss trains each entry as soon as its source exists.
- **Seed-time evaluation is order-independent.** Ranking reads only what the evaluation stored: loss, per-example losses, eval-set fingerprint, eval GPU, bytes and `quant_failure`. Penalty, cap verdict and adjusted loss are derived at ranking from those and the task's pinned policy. The verdict does not depend on when the boss was scored.

## Decoder columns

`TournamentParticipant` (`models.py`) carries the decoder snapshot taken at enrolment.

| Field | Meaning | Public |
| --- | --- | --- |
| `decoder_object_name` | MinIO key of the snapshot `.tar.xz` | never |
| `decoder_xz_bytes`, `decoder_sha256` | counted size and sha256 of the `.tar.xz` | yes, in the scorecard |
| `decoder_tar_sha256` | sha256 of the uncompressed canonical tar | yes, in the scorecard |
| `decoder_tree_sha` | git tree id of `decoder/` | yes, in the scorecard and `release.json` |
| `decoder_xz_version` | `xz --version` line that compressed the snapshot | yes, in the scorecard |
| `decoder_released_url`, `decoder_released_at` | the public release | yes, in the scorecard after release |

Public participant lists have `github_token` and `decoder_object_name` set to null (`_public_participants` in `validator/endpoints/tournament_analytics.py`). The training repo URL stays, because results are labelled by it. The object name can be derived from the commit and sha256 in the same response. The control is that the bucket is private and every MinIO link is presigned.

## Freeform publications

Two publications follow a compression tournament's completion (`freeform_release.py`).

| Publication | What | When |
| --- | --- | --- |
| Decoder release | The champion's snapshot `.tar.xz` and a `release.json`, to a public HF dataset | After the champion's code is pushed |
| Decoded models | The bf16 model of every boss-round freeform submission, as public HF model repos | At completion |

- Both run as background tasks under one lock, so the tournament loop and the code push never wait on uploads.
- Both are idempotent. A failure is logged and alerted, and never undoes completion or the code push.
- A decoder is public only for the tournament's champion. A size or sha256 mismatch against the enrolment snapshot refuses the release.
- `reconcile_freeform_publications` re-derives unpublished work from the database. It runs at validator start and then hourly (`FREEFORM_PUBLICATION_RECONCILE_INTERVAL`).

Run the reconcile by hand. It is read-only unless `--apply` is passed. Flags are in [`ops/tools/README.md`](../../ops/tools/README.md).

```bash
python -m ops.tools.quant.reconcile_freeform_publications
```

## See also

- [Validator scoring](../scoring/README.md): emission split, compression scoring, SKU normalisation.
- [Validator evaluation](../evaluation/README.md): evaluation internals and failure attribution.
- [Migrations](../db/migrations/README.md): schema changes and deploy order.
