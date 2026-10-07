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
