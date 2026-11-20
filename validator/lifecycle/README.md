# Validator lifecycle

The validator loops that create, prepare, assign, evaluate and finalise tasks.

## Contents

| Path | Purpose |
| --- | --- |
| `runner.py` | Starts the lifecycle loops. |
| `tasks.py` | Task processing: prep, miner assignment, evaluation dispatch and finalisation. |
| `constants.py` | Concurrency caps, the evaluation GPU cap and the retry and hold constants. |
| `eval_infra_errors.py` | `record_infra_error`: re-queues, counts, alerts on and finally holds evaluation rows that failed through no fault of the miner. |
| `quant_reference.py` | `ensure_quant_references`: the check at finalisation that every scored compression row has a bf16 reference on its GPU SKU. |
| `boss_seed_eval.py` | Evaluation of the boss's seed entries as soon as they train, with retries and alerts. |

## Data prep failures

A task's data prep fails in one of two ways.

| Failure | Effect |
| --- | --- |
| Prep could not fetch or store the task's data: an HTTP error, status or timeout on a data URL, an unreachable or erroring Hugging Face hub, or an object-store upload that did not go through (`DataFetchError`, `validator/tasks/datasets/fetch.py`) | The task returns to `PENDING` unchanged and is prepped again after a backoff. The backoff starts at `PREP_FETCH_RETRY_BASE_SECONDS` (1 min), doubles per consecutive failure and is capped at `PREP_FETCH_RETRY_MAX_SECONDS` (30 min). The `PREP_FETCH_FAILURES_BEFORE_ALERT`-th (3rd) consecutive failure sends one alert. The task is never failed for it, so the tournament does not replace it, and the retries go on until the data can be fetched. The count is per validator process. |
| Anything else: a near-duplicate dataset, too few rows, no calibration set, a dataset or model that cannot be used | The task is marked `PREP_TASK_FAILURE`. A tournament replaces it. |

## Finalising a compression task

Finalisation of a compression task waits for `ensure_quant_references`. The function measures nothing. Each evaluation deployment scores the reference on its SKU.

| Situation | Result |
| --- | --- |
| A successful row lacks a reference on its SKU | The row returns to `pending` and is evaluated again. The budget is `QUANT_REFERENCE_EVAL_RETRIES` (1), counted on the task's `n_eval_attempts`. |
| The budget is spent, or a common-input task has no reference model | `QuantReferenceUnavailable`: the task is marked `PREP_TASK_FAILURE` and the tournament replaces it. |
| The same, on a seed task | The task is held and an alert is sent once. It is not marked `PREP_TASK_FAILURE`. |

## Seed tasks

A seed task is not finalised on the boss's row alone. The boss's row is evaluated as soon as it trains.

| Condition | Effect |
| --- | --- |
| A round has attached the seed, a non-boss player is assigned, and every assigned player's training row is terminal or carries a `quant_failure` | Finalisation proceeds. |
| The tournament is decided, and the seed is unattached or is a text seed or a compression round seed | The seed is released and closes with the boss's stored result. |
| Anything else | Finalisation waits. |

<details>
<summary>Seed-time evaluation failures</summary>

An evaluation error on the boss's entry before a round attaches the seed is retried, never scored. It is retried up to `SEED_TIME_BOSS_EVAL_RETRIES` times, each after `SEED_TIME_BOSS_EVAL_BACKOFF_SECONDS`. After that the entry is parked, an alert is sent, and the entry is evaluated when its round attaches the seed. `alert_on_eval_set_mismatch` alerts once at finalisation when the two stored entries carry different eval-set fingerprints.

</details>

## See also

- [Whose fault an evaluation failure is](../evaluation/README.md#whose-fault-an-evaluation-failure-is): the error classes and the hold.
- [Tournament design notes](../tournament/README.md#design-notes): seed tasks.
