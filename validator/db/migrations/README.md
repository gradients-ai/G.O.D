# Database migrations

Timestamped [dbmate](https://github.com/amacneil/dbmate) migrations for the validator database schema.

## Contents

| Path | Purpose |
| --- | --- |
| `*.sql` | One schema migration each, with `-- migrate:up` and `-- migrate:down` sections. Applied in filename order. |

## Adding a migration

1. Add a file named `<YYYYMMDDHHMMSS>_<what_it_does>.sql`.
2. Update `validator/db/constants.py` and the SQL modules in `validator/db/sql/` in the same change.
3. Add it to the list below.

Never edit a migration that has been applied. Add another one.

## Deploy order

Apply migrations first, then start the code that needs them.

`task validator` starts the compose stack, whose dbmate service runs `dbmate up` in the background, and then starts the validator processes without waiting for it. The auto-updater runs the same task.

To guarantee the order, apply the migrations in the foreground before `task validator`:

```bash
task dbup
```

`dbmate up` applies every pending file, including one whose timestamp sorts before a migration already applied.

## Migrations

| File | Change |
| --- | --- |
| `20241013103031_initial_migration.sql` | Initial schema: `tasks`, `nodes`, `task_nodes`, `submissions`, `nodes_history`. |
| `20241201000001_rename_hf_training_repo.sql` | Rename `tasks.hf_training_repo` to `training_data`. |
| `20241206000001_task_type_n_delay.sql` | Add `is_organic` and `delay_times` to `tasks`. |
| `20241211141227_rename_task_columns.sql` | Rename task field and timestamp columns. Add `system_format` and `trained_model_repository`. |
| `20241223120235_add_training_repo_backup.sql` | Add `tasks.training_repo_backup`. |
| `20241226000001_add_file_format.sql` | Add `tasks.file_format`. |
| `20250108000001_add_losses.sql` | Add `test_loss` and `synth_loss` to `task_nodes`. |
| `20250108221935_add_expected_repo_name.sql` | Add `task_nodes.expected_repo_name`. |
| `20250113154733_add_score_reason.sql` | Add `task_nodes.score_reason`. |
| `20250113154734_offer_responses.sql` | Create `offer_responses`. |
| `20250120123629_latest_scores_url.sql` | Create `latest_scores_url`. |
| `20250130160001_add_n_eval_attempts.sql` | Add `tasks.n_eval_attempts`. |
| `20250201222021_split_tasks_table.sql` | Split task-type fields out of `tasks` into text and image task tables. Add `tasks.task_type`. |
| `20250202163618_add-image-text-pairs.sql` | Create `image_text_pairs`. |
| `20250203215716_remove_model_filename.sql` | Drop `image_tasks.model_filename`. |
| `20250210145838_add_result_model_name.sql` | Add `tasks.result_model_name`. |
| `20250212124717_add_index_on_model_name.sql` | Add `tasks.result_model_name_lower` and its index. |
| `20250219153232_new_nodes_fields.sql` | Add `alpha_stake` and `tao_stake` to `nodes`. |
| `20250220225032_new_nodes_fields_for_history_nodes.sql` | Add `alpha_stake` and `tao_stake` to `nodes_history`. |
| `20250224140828_add_indexes_for_tasks.sql` | Indexes on `task_nodes` and `tasks`. |
| `20250224155501_add_model_params_count.sql` | Add `tasks.model_params_count`. |
| `20250310150111_only_one_submission_per_miner.sql` | One submission per miner and task on `submissions`. |
| `20250403235532_integrate_dpo.sql` | DPO task type and `dpo_tasks`. |
| `20250410150115_image_model_type.sql` | Add `image_tasks.model_type` and its enum. |
| `20250424150000_add_hotkey_performance_indexes.sql` | Indexes for per-hotkey performance queries. |
| `20250429183400_fix_overindexing.sql` | Drop redundant indexes. |
| `20250507133000_integrate_grpo.sql` | GRPO task type, `grpo_tasks`, `reward_functions`, `grpo_task_functions`. |
| `20250523180000_add_manual_reward_functions.sql` | Add `reward_functions.is_manual`. |
| `20250603024700_integrate_chat_tasks.sql` | Chat task type and `chat_tasks`. |
| `20250611060905_add_tournament_tables.sql` | Tournament tables: `tournaments`, rounds, participants, groups, group members, pairs, tasks. |
| `20250625000000_add_training_repo_to_tournament_participants.sql` | Add `training_repo` and `training_commit_hash` to `tournament_participants`. |
| `20250625131001_add_trainers_gpus_table.sql` | Create `trainers_gpus`. |
| `20250626120301_add_tournaments_training_status.sql` | Create `tournament_task_hotkey_trainings`. |
| `20250626183701_add_tournaments_github_submissions.sql` | Ensure `training_repo` and `training_commit_hash` exist on `tournament_participants`. |
| `20250627000000_add_base_winner_hotkey_to_tournaments.sql` | Add `tournaments.base_winner_hotkey`. |
| `20250628000000_add_winner_hotkey_to_tournaments.sql` | Add `tournaments.winner_hotkey`. |
| `20250713120000_add_boss_round_synced_tasks.sql` | Create `boss_round_synced_tasks`. |
| `20250720000000_add_stake_required_to_tournament_participants.sql` | Add `tournament_participants.stake_required`. |
| `20250730000000_add_backup_repo_to_tournament_participants.sql` | Add `tournament_participants.backup_repo`. |
| `20250731000001_add_benchmark_root_tasks.sql` | Create `benchmark_root_tasks`. |
| `20250731000002_add_benchmark_task_copies.sql` | Create `benchmark_task_copies`. |
| `20250731000003_add_training_priority.sql` | Add `tournament_task_hotkey_trainings.priority`. |
| `20250805193501_add_grpo_extra_column.sql` | Add `grpo_tasks.extra_column`. |
| `20250919142043_fix_benchmark_task_copies_constraint.sql` | Adjust the `benchmark_task_copies` constraint. |
| `20251007112652_add_training_fields_to_tournament_task_hotkey_trainings.sql` | Add `training_repo` and `training_commit_hash` to `tournament_task_hotkey_trainings`. |
| `20251010000000_add_winning_performance_difference_to_tournaments.sql` | Add `tournaments.winning_performance_difference`. |
| `20251016000000_add_transfer_tables.sql` | Transfer tables: `transfers`, `coldkey_balances`, `transfer_processing_state`. |
| `20251022000000_add_balance_events.sql` | Create `balance_events`. |
| `20251115194338_add_backend_to_tasks.sql` | Add `tasks.backend`. |
| `20251115194339_add_dstack_runname_to_tournament_task_hotkey_trainings.sql` | Add `tournament_task_hotkey_trainings.dstack_runname`. |
| `20251203170741_add_trainer_ip_to_tournament_task_hotkey_trainings.sql` | Add `tournament_task_hotkey_trainings.trainer_ip`. |
| `20251211000000_add_yarn_factor_to_tasks.sql` | Add `tasks.yarn_factor`. |
| `20251216180000_add_zimage_qwen_image_model_types.sql` | Image model types `z-image` and `qwen-image`. |
| `20251227221903_integrate_environment_tasks.sql` | Environment task type and `env_tasks`. |
| `20260129000000_add_eval_seed_to_env_tasks.sql` | Add `env_tasks.eval_seed`. |
| `20260303120000_add_evaluations_table.sql` | Create `evaluations`. |
| `20260311120000_add_deployment_ids_to_evaluations.sql` | Add deployment ids to `evaluations`. |
| `20260317120000_add_github_token_for_private_repos.sql` | Add `github_token` for private repositories to participants and training rows. |
| `20260409120000_tasks_hours_to_complete_float.sql` | Make `tasks.hours_to_complete` a float. |
| `20260427120000_add_augmentation_config_to_tasks.sql` | Add `tasks.augmentation_config`. |
| `20260428120000_add_model_prep_fields_to_tasks.sql` | Add `augmented_model_id` and `baseline_stats` to `tasks`. |
| `20260430000000_add_diff_report_to_tournaments.sql` | Add `tournaments.diff_report`. |
| `20260504141200_add_requested_datasets.sql` | Add `requested_datasets` to participants and training rows. |
| `20260506120000_env_tasks_multiple_environments.sql` | Add `environment_names` and `environment_weights` to `env_tasks`, drop `environment_name`, add `tasks.training_start_point`. |
| `20260506130000_add_starting_model_repo.sql` | Add `task_nodes.starting_model_repo`. |
| `20260506140000_add_winner_model_repo.sql` | Add `winner_model_repo` and `winner_model_base` to `tournaments`. |
| `20260518161000_add_eval_loop_indexes.sql` | Indexes for the evaluation loop. |
| `20260521120000_add_pvp_pair_results.sql` | Create `pvp_pair_results`. |
| `20260524120000_add_pvp_individual_scores.sql` | Create `pvp_individual_scores`. |
| `20260530120000_add_baseline_stats_to_task_nodes.sql` | Add `task_nodes.baseline_stats`. |
| `20260602120000_add_evaluation_gpu_count.sql` | Add `evaluations.gpu_count` and an index, drop `deployment_env_id`. |
| `20260602120001_add_kl_to_instruct_tasks.sql` | Add `use_kl` and `kl_coef` to `instruct_text_tasks`. |
| `20260605120000_add_tournament_dedup_reviews.sql` | Create `tournament_dedup_reviews`. |
| `20260609110000_add_trigger_word_to_image_tasks.sql` | Add `image_tasks.trigger_word`. |
| `20260623120000_add_pvp_pair_deployment_id.sql` | Add `pvp_pair_results.deployment_id`. |
| `20260628120000_add_continuous_sft_state.sql` | Create `continuous_sft_state`. |
| `20260629120000_add_continuous_sft_source_round.sql` | Add `continuous_sft_state.last_source_round_id`. |
| `20260629180000_add_pvp_individual_deployment_id.sql` | Add `pvp_individual_scores.deployment_id`. |
| `20260706163000_add_ideogram4_krea2_image_model_types.sql` | Image model types `ideogram4` and `krea2`. |
| `20260709120000_add_pvp_pair_gpu_count.sql` | Add `pvp_pair_results.gpu_count`. |
| `20260710140000_add_tournament_code_review.sql` | Add `tournaments.code_review`. |
| `20260717120000_add_tournament_gpu_costs.sql` | GPU cost tables: `gpu_usage_runs`, `trainer_gpu_capacity_intervals`. |
| `20260720160000_add_pvp_pair_deployment_verified.sql` | Add `pvp_pair_results.deployment_verified`. |
| `20260808120000_add_per_example_eval_losses.sql` | Add `per_example_losses` and `eval_set_fingerprint` to `task_nodes`. |
| `20260810234500_add_tournament_task_failure_reviews.sql` | Create `tournament_task_failure_reviews`. |
| `20260907120000_add_gpu_interconnect_to_trainers_gpus.sql` | Add `product_name`, `interconnect` and `nvlink` to `trainers_gpus`. |
| `20260907120100_add_interconnect_to_capacity_intervals.sql` | Add `trainer_gpu_capacity_intervals.interconnect`. |
| `20261005140000_add_participant_code_review.sql` | Add `tournament_participants.code_review`. |
| `20261019120000_integrate_quant_tasks.sql` | Compression task types, `quant_tasks`, and the artifact and decoder columns on `task_nodes`. |
| `20261019150000_add_participant_decoder_snapshot.sql` | Decoder snapshot columns on `tournament_participants`. |
| `20261020040000_add_nvlink_peers_to_trainers_gpus.sql` | Add `trainers_gpus.nvlink_peers`. |
| `20261020100000_quant_references.sql` | Reference columns and `upstream_task_id` on `quant_tasks`. |
| `20261027120000_add_quant_eval_gpu.sql` | Add `eval_gpu` and `decoder_build_warning` to `task_nodes`, and `quant_tasks.reference_eval_gpu`. |
| `20261027180000_add_quant_sku_references.sql` | Create `quant_sku_references`, copy the reference values into it and drop the reference columns on `quant_tasks`. |
| `20261029100000_quant_sku_references_per_hotkey.sql` | Add `quant_sku_references.hotkey`. |
| `20261106120000_quant_freeform_soft_penalty.sql` | Freeform penalty columns on `quant_tasks` and `task_nodes`. |
| `20261107120000_quant_calibration_set.sql` | Calibration columns on `tasks`, `dataset_calibration_rows`, `quant_calibration_references`. |
| `20261108120000_quant_freeform_exposure.sql` | Freeform artifact and decoder release columns on `task_nodes` and `tournament_participants`. |
| `20261109120000_tournament_seed_tasks.sql` | Create `tournament_seed_tasks`. |
| `20261110120000_evaluation_infra_error_attempts.sql` | Add `evaluations.infra_error_attempts`. |
| `20261112120000_tournament_seed_merged_output_repo.sql` | Add `tournament_seed_tasks.merged_output_repo`. |
| `20261113120000_continuous_sft_chunk_scheme.sql` | Record of the lineage position each continuous task was created at (`continuous_sft_task_chunks`). |
| `20261113130000_text_seed_roles.sql` | Text seed roles and compression boss-round roles on `tournament_seed_tasks`. |
| `20261113140000_quant_task_sources.sql` | Create `quant_tournament_sources` and `quant_task_sources`. |
| `20261113150000_quant_round_seed_roles.sql` | Round seed roles for compression on `tournament_seed_tasks`. |
| `20261113170000_tournament_boss_rank_comparisons.sql` | Create `tournament_boss_rank_comparisons`. |
| `20261114120000_quant_calibration_required.sql` | `tasks_quant_calibration_required` CHECK (NOT VALID): a compression task past prep has its calibration set. |

## See also

- Tables and columns: [validator/db/README.md](../README.md).
- Operator commands: [docs/developer.md](../../../docs/developer.md#running-a-validator).
