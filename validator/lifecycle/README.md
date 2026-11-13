# Validator Lifecycle

Main validator loops for creating, preparing, assigning, evaluating, and finalizing tasks.

## Contents

- `constants.py`: lifecycle loop constants.
- `eval_infra_errors.py`: `record_infra_error`, called by `_evaluate_and_update_hotkeys` for an `EvaluationInfraError`,
  any other `EvaluationRetryableError` that is not a capacity wait (`EvaluationCapacityUnavailable`,
  `DeploymentNotReadyError`: re-queued uncounted; for quant tasks a pod that never becomes ready IS counted, so a
  crash-looping evaluator is held instead of re-launched every cycle), or any exception out of evaluation (miner faults come back as
  results, never as exceptions). It counts the error in
  `evaluations.infra_error_attempts`, re-queues the rows to `pending` and alerts on the first one; at
  `MAX_EVAL_INFRA_ATTEMPTS` it holds them in `infra_held` and alerts with the SQL to re-queue them. A held row is never
  dispatched or scored and finalize does not finalise the task around it. A successful evaluation resets the count.
  A re-queued row is not dispatched again for `EVAL_INFRA_REQUEUE_BACKOFF_SECONDS` (10 min, in memory), only rows still
  `pending` or `evaluating` are counted (settled rows are never reopened), and a task with held rows is re-alerted every
  `EVAL_INFRA_HELD_REMINDER_SECONDS` (4 h) until a human re-queues or fails them.
- `quant_reference.py`: `ensure_quant_references`, called by finalize for quant tasks (both tracks). It measures nothing itself: each evaluation deployment scores the bf16 reference on its own SKU and scoring caches it in `quant_sku_references`. It checks that every successful row with a finite loss has a reference on its SKU; rows that lack one are reset to pending (re-evaluated, which measures it) up to `QUANT_REFERENCE_EVAL_RETRIES`, alerting once per task. Spent retries, or a task with no reference model, raise `QuantReferenceUnavailable`.
- `boss_seed_eval.py`: seed-time evaluation of the boss's seed entries (text seeds and the quant boss-round entries). An evaluation error of the boss's entry before the boss round attaches the seed (the pass raised, no result, or a scorer-recorded `score_reason`) is put back to pending, never scored, up to `SEED_TIME_BOSS_EVAL_RETRIES` (each retry waits `SEED_TIME_BOSS_EVAL_BACKOFF_SECONDS` after the last failure), then parked and alerted (evaluated in the boss round instead); a quant entry with no bf16 reference on its eval SKU is re-evaluated at seed time; `alert_if_boss_evaluation_missing` (called by the orchestrator's boss-round seed cycle) alerts once when the round opens without the boss's result; `alert_on_eval_set_mismatch` alerts once at finalisation when the two stored entries carry different eval-set fingerprints.
- `runner.py`: lifecycle runner orchestration.
- `tasks.py`: task lifecycle processing. Finalize of any quant task first waits for `quant_reference.ensure_quant_references`; a `QuantReferenceUnavailable` from it or from finalize marks the task `PREP_TASK_FAILURE` and the tournament replaces it. `_finalize_task_status_from_evaluations` returns early for a TRAINING seed whose hold (`seeds_sql.boss_round_seed_ready_to_finalise`: attached, another player assigned, every row terminal; or its tournament decided, see `validator/tournament/README.md`) is not satisfied, so a seed a round attached is never ranked on the boss's row alone; its rows are still evaluated as they train (the boss's at seed time). `_get_tasks_ready_for_evaluation` takes such a seed up only for its pending or running evaluation rows (never from the completed-training query, where it has nothing to finalise), and not at all while its seed-time evaluation is parked.
- `__init__.py`: package marker.
