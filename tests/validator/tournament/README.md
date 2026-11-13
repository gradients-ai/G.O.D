# Validator tournament tests

Tests for `validator/tournament`: scheduling, enrolment, seeds, rounds, the crown rule, orchestration and analytics. The rules they cover are in the [tournament README](../../../validator/tournament/README.md).

Run them from the repository root:

```bash
pytest tests/validator/tournament
```

Some tests skip without their tools on `PATH`: `test_decoder_snapshot.py` and `test_decoder_runnable.py` need `git`, `tar` and `xz`, and the database tests in `test_compression_rounds.py` and `test_text_seed_rounds.py` need `initdb` and `pg_ctl`. `test_github_validation.py` runs `git` and fails without it.

## Contents

**Scheduling and enrolment**

| Path | Covers |
| --- | --- |
| `test_quant_tournament_type.py` | The compression tournament: scheduling, the first boss lookup, the entry fee, and the licence and decoder steps that apply to compression miners only. |
| `test_next_tournament_dates_flag.py` | The next start date is advertised for every tournament. |
| `test_pending_tournaments_nonblocking.py` | A pending tournament short of miners, or one that raises, does not block the others. |
| `test_base_contestant_datasets.py` | The boss participant of an environment-type tournament carries the last winner's requested datasets. |
| `test_github_validation.py` | The licence check at a pinned commit, and token-free logging of the obfuscation check. |
| `test_registration_lint.py` | The build lint, rule by rule, and the large-file review listing. |
| `test_decoder_registration_gate.py` | Compression enrolment: miners excluded or enrolled by decoder snapshot and build lint. |
| `test_decoder_runnable.py` | A decoder with no entrypoint or over the extraction cap is rejected. |
| `test_decoder_snapshot.py` | Snapshot determinism, decoder errors, the MinIO upload and the lint hook. |
| `test_boss_decoder_snapshot_retry.py` | The boss's decoder snapshot: retries, repair while pending, the boss row always present. |
| `test_r2_dedup_includes_boss.py` | The round-2 duplicate gate compares the advancing field against the boss's code. |
| `test_repo_dedup_reporting.py` | Duplicate report rendering and verdict parsing. |

**Seeds and sources**

| Path | Covers |
| --- | --- |
| `test_seed_tasks.py` | Seed creation, the boss pin, task filters, hours routing, trainer priority, the hold and the replacement ladder. |
| `test_text_seed_rounds.py` | Text seeds and rounds 1-3: attach, standings without the boss, the finalisation hold, seed replacement, adapter merge and seed hiding. |
| `test_text_seed_failed_role.py` | A failed text seed role never halts or blocks the text tournament. |
| `test_boss_round_seeds.py` | The compression boss round attaches its four seeds and adds challenger rows only. |
| `test_boss_seed_hold.py` | The seed hold on every finalising path. |
| `test_boss_seed_time_eval.py` | The boss's entries are evaluated when trained. The round evaluates the challenger. |
| `test_boss_seed_prep_failure_completion.py` | A prep-failed boss-round seed is not replaced, and who takes the slot. |
| `test_prep_failed_seed_slots.py` | A slot nobody produced a result on never credits the challenger. |
| `test_seed_adapter_merge.py` | A boss adapter output is merged, and the seed is never replaced for it. |
| `test_quant_text_linkage.py` | Compression tasks are created from the text boss's tasks with a persisted link. |
| `test_quant_source_failed_role.py` | A failed text seed role never stalls a compression tournament. The slot takes the fallback. |
| `quant_source_fakes.py` | In-memory stand-in for the source tables, with their constraints. |

**Rounds and standings**

| Path | Covers |
| --- | --- |
| `test_compression_rounds.py` | Compression rounds 1-3 on the text boss's models: tasks, release, cuts, the boss's rank, third place. |
| `test_quant_round_shape.py` | Group sizes through rounds 1-3 and into the boss round. |
| `test_round3_viability.py` | The viability cut after round 2, and third place when one miner survives. |
| `test_text_cumulative_advancement.py` | Advancement by mean rank over the rounds played, within the surviving miners. |
| `test_group_standings_tiebreak.py` | Standings tie-breaks: average rank, summed gap, seeded shuffle. |
| `test_image_round_plan.py` | The image round plan, the boss-round mix, `round_plan`, and telling a planned image tournament from one with a three-task round 1. |
| `test_image_rounds.py` | Image rounds 1-3: the boss assigned to each round's task, standings without the boss, advancement 8 / 4 / 1 on average rank, the boss's rank, third place and the empty-score check. |
| `test_image_task_creation.py` | Image task creation: one task per planned round, the four boss-round tasks, the three-task round 1 structure, replacements, and the exact pair count of a synthetic image task. |
| `test_text_r1_single_group.py` | Round 1 is one group for text and image, and how many tasks it holds. |
| `test_empty_score_group_guard.py` | A group with no valid challenger score halts advancement. |
| `test_majority_failure_gate.py` | A task where most trainings failed is held for manual review. |
| `test_metric_direction.py` | Sort directions: rewards descend, losses ascend. |
| `test_pre_boss_third_place.py` | Third place from the round before the boss round. |
| `test_knockout_paired.py` | Per-example win rate between two knockout miners (`paired_head_to_head_winner`). |
| `test_knockout_resolution.py` | Knockout task resolution end to end with stubbed reads. |

**Crown and boss round**

| Path | Covers |
| --- | --- |
| `test_crown.py` | The crown rule and each task kind's win and non-win bands, as pure functions. |
| `test_text_boss_round_crown.py` | The text boss round: its tasks, the crown, the early stop and the flow around them. |
| `test_image_boss_round_crown.py` | The planned image boss round: the crown, the early stop after round 3, the evaluation-fault hold and third place. |
| `test_quant_boss_crown_rule.py` | The compression crown over seven head-to-heads. |
| `test_quant_early_stop.py` | The compression early stop, and retirement of unattached seeds. |
| `test_paired_boss_round.py` | The paired per-example test. |
| `test_paired_example_reporting.py` | The per-example split reported to analytics. |
| `test_boss_round_task_resolution.py` | Paired test versus relative margin on a boss-round task. |
| `test_quant_boss_round_task_resolution.py` | Fixed on the paired test, freeform on the cap and effective bytes. |
| `test_quant_sku_paired_gates.py` | Paired comparisons across GPU SKUs. |
| `test_chained_fixed_judging.py` | A fixed boss task on a shared input, on one SKU. |
| `test_boss_allowed_loss.py` | What the boss-battle analytics count as a narrow loss. |
| `test_boss_round_draws.py` | A draw on a fixed task, freeform never drawing, the built size of the compression boss round, and a round deferred for a task with no nodes. |
| `test_boss_round_draw_deciders.py` | Draws and decider tasks in a boss round with draws. |
| `test_boss_round_large_instruct_model.py` | The large-model instruct task of a text boss round. |
| `test_quant_boss_round_constants.py` | Boss-round composition and the continuous dataset-id encoding. |
| `test_quant_boss_round_tasks.py` | The four compression boss-round tasks, and creation and log redaction of the tasks on the continuous model. |

**Task creation and release**

| Path | Covers |
| --- | --- |
| `test_quant_task_creation.py` | Compression task fields, budgets and replacement copies, and resolution of a miner's round-1 repo. |
| `test_calibration_task_creation.py` | Where the calibration set is carved or inherited. |
| `test_text_augmentation_r1_only.py` | Compression tasks other than the input seeds never roll model augmentation. |
| `test_chained_boss_tasks.py` | Held chained tasks, replacements that keep the chain, and the choice of the shared input. |
| `test_chained_release.py` | Release of a held chained task: the shared input and its fallbacks, the copied data and budget, model prep after release, the near-duplicate exemption. |
| `test_chained_training_requests.py` | A released chained task's training requests carry the shared input. |
| `test_own_model_assignment.py` | Node assignment for a task with a per-miner start model. |
| `test_replacement_invariants.py` | A replacement task keeps the original's pool, lineage and model type, and the original is detached, not deleted. |
| `test_carry_forward_continuous_sft.py` | Carrying each continuous lineage's winner to its state row. |
| `test_continuous_sft_chunk_scheme.py` | Continuous-task position bookkeeping. |
| `test_continuous_sft_constants.py` | Continuous-SFT constants, the text boss-round task mix, and the dataset-id and lineage helpers. |

**Orchestration and compute**

| Path | Covers |
| --- | --- |
| `test_quant_dispatch_bucket.py` | The three queue buckets. A full bucket never blocks another. |
| `test_quant_scheduling_isolation.py` | A compression row that raises does not hold text and image rows. |
| `test_quant_reference.py` | Which bf16 model and reference key a compression row is normalised by. |
| `test_quant_reference_prepass.py` | The reference pre-pass: candidates, copies, measurement, failures. |
| `test_quant_reference_gating.py` | Training waits for its bf16 reference and gets its target. |
| `test_quant_training_request.py` | The compression training request and the freeform decode request. |
| `test_quant_decode_plumbing.py` | Decode-result persistence and the hold for unsettled decodes. |
| `test_training_request_logging.py` | Trainer requests are logged without secrets. |
| `test_quant_gpu_hours.py` | Compression GPU and hours sizing. |
| `test_quant_prep_reuse.py` | One prep per model and dataset, shared between sibling tasks. |
| `test_continuous_sft_gpu_hours.py` | Continuous task compute sizing. |
| `test_multigpu_interconnect.py` | Multi-GPU tasks are placed on high-interconnect trainers. |

**Publication, analytics and weights**

| Path | Covers |
| --- | --- |
| `test_freeform_release.py` | The decoder release and the boss-round decoded model publication. |
| `test_no_public_decoder_path.py` | No public response model serialises the decoder object name, a presigned decoder URL or the GitHub token. |
| `test_tournament_analytics.py` | The tournament details endpoint, including the stored boss rank comparisons. |
| `test_projection_alpha.py` | Weight projections are constant over the projection horizon. |
| `test_separated_burn_dynamics.py` | Per-tournament weights and their application to node weights. |

**Support**

| Path | Covers |
| --- | --- |
| `fixtures/registration_lint/` | Fixture repositories for the build lint. |
| `__init__.py` | Package marker. |
