TASK_TIME_DELAY = 15
MAX_DELAY_TIMES = 6

MAX_CONCURRENT_MINER_ASSIGNMENTS = 5
MAX_CONCURRENT_TASK_PREPS = 3
EVAL_MAX_GPUS = 50
# How often the eval loop reconciles live Basilica deployments against the reservation ledger.
EVAL_RECONCILE_INTERVAL_SECONDS = 120
MAX_CONCURRENT_JOBS = 60

MODEL_SIZE_REQUIRING_2_GPUS = 30 * 10**9
# Runpod only sells 1/2/4/8 GPU pods. Never request 3. 70B-class bf16 evals fit on
# 2xA100-80GB (with expandable CUDA segments); jump straight to 4 above this threshold.
MODEL_SIZE_REQUIRING_4_GPUS = 75 * 10**9

# Seed-time evaluation of the boss's boss-round entries (lifecycle/boss_seed_eval.py): how many times in all an evaluation
# error (or a missing same-SKU bf16 reference) of the boss's entry is retried before the boss round attaches the task.
# After that the row is parked and evaluated in the boss round.
SEED_TIME_BOSS_EVAL_RETRIES = 3
# A retried seed-time evaluation is not dispatched again until this long after its last failure, so the budget above
# outlasts a short hub or deployment-service outage instead of being spent within a few 30 s cycles.
SEED_TIME_BOSS_EVAL_BACKOFF_SECONDS = 10 * 60

# A task whose data prep could not fetch or store its data (validator.tasks.datasets.fetch.DataFetchError) stays PENDING
# and is prepped again after a backoff that doubles per consecutive failure from the base to the cap (seconds), for as
# long as the failure lasts: it is never failed or replaced for it. The failure that reaches the alert count alerts once.
PREP_FETCH_RETRY_BASE_SECONDS = 60
PREP_FETCH_RETRY_MAX_SECONDS = 30 * 60
PREP_FETCH_FAILURES_BEFORE_ALERT = 3

# Evaluation rows whose evaluation keeps failing for reasons that are not the miner's (infrastructure, or our own
# evaluation error: validator.evaluation.errors.EvaluationInfraError). Each such failure re-queues the row and counts
# in evaluations.infra_error_attempts; at this many the row is held in EVALUATION_STATUS_INFRA_HELD, which is never
# dispatched, never scored and keeps the task from finalising, until a human re-queues or fails it.
MAX_EVAL_INFRA_ATTEMPTS = 3
EVALUATION_STATUS_INFRA_HELD = "infra_held"
# A row re-queued after such a failure is not dispatched again until this long after it, so the attempts above outlast a
# short hub or deployment-service outage instead of being spent within a few 30 s cycles.
EVAL_INFRA_REQUEUE_BACKOFF_SECONDS = 10 * 60
# A task with held rows is re-alerted this often (the hold itself alerts once), so a held row cannot block its task, round
# and tournament unnoticed.
EVAL_INFRA_HELD_REMINDER_SECONDS = 4 * 60 * 60
