"""Landscape sweep of the deployed tournament training-hours formula.

Mirrors validator/tasks/synthetics/scheduler.py compute_training_hours so we
can inspect where real tasks land across model size, row count, and tokens per
row. Standalone: no validator import.

Keep constants in sync with validator/tasks/datasets/constants.py.

Run: python -m ops.tools.simulations.training_hours_landscape
"""

TRAINING_HOURS_MIN = 0.75
MAX_TRAINING_HOURS = 6.0
TARGET_TRAINING_EPOCHS = 2.0
H100_BF16_TFLOPS = 989.0
ASSUMED_TRAINING_MFU = 0.15
EFFECTIVE_MIN_TOKENS_PER_ROW = 64
TRAINING_OVERHEAD_HOURS = 0.5
TYPE_MULT = {"instruct": 1.0, "dpo": 1.4, "grpo": 1.3}
GPU_TYPE_MULT = {"instruct": 1, "dpo": 3, "grpo": 2}
GPU_THRESHOLDS = [(4.0, 1), (12.0, 2), (40.0, 4), (float("inf"), 8)]

MODELS = [
    ("0.35B", 0.35e9),
    ("1.1B", 1.1e9),
    ("3B", 3e9),
    ("7B", 7e9),
    ("9B", 9.2e9),
    ("14B", 14e9),
    ("32B", 32e9),
]
ROWS = [8_000, 20_000, 40_000, 90_000, 175_000]
TOK_PER_ROW = [100, 200, 400, 700, 1000, 1500]


def gpu_count(params: float, task_type: str = "instruct") -> int:
    params_b = params / 1e9 * GPU_TYPE_MULT[task_type]
    for threshold, gpus in GPU_THRESHOLDS:
        if params_b <= threshold:
            return gpus
    return 8


def analytic_tps_per_gpu(params: float) -> float:
    return H100_BF16_TFLOPS * 1e12 * ASSUMED_TRAINING_MFU / (6.0 * params)


def hours(tokens_per_epoch: float, params: float, task_type: str = "instruct") -> float:
    gpus = gpu_count(params, task_type)
    tps = analytic_tps_per_gpu(params)
    secs = TARGET_TRAINING_EPOCHS * tokens_per_epoch * TYPE_MULT[task_type] / (tps * gpus)
    h = secs / 3600 + TRAINING_OVERHEAD_HOURS
    h = max(TRAINING_HOURS_MIN, round(h * 4) / 4)
    return min(h, MAX_TRAINING_HOURS)


def print_throughput_table():
    print(f"\n=== analytic tok/s per H100 (MFU={ASSUMED_TRAINING_MFU:.2f}) ===")
    print(f"{'model':>6} | {'tok/s/gpu':>9} | {'gpus':>4}")
    print("-" * 28)
    for name, params in MODELS:
        print(f"{name:>6} | {analytic_tps_per_gpu(params):>9.0f} | {gpu_count(params):>4}")


def print_grid(task_type: str, tokens_per_row: int):
    eff_tpr = max(tokens_per_row, EFFECTIVE_MIN_TOKENS_PER_ROW)
    print(f"\n=== hours [{task_type}, {tokens_per_row} tok/row (effective {eff_tpr})]: rows x model ===")
    header = "rows\\model | " + " | ".join(f"{name:>6}" for name, _ in MODELS)
    print(header)
    print("-" * len(header))
    for rows in ROWS:
        cells = [f"{hours(rows * eff_tpr, params, task_type):>6}" for _, params in MODELS]
        print(f"{rows:>10} | " + " | ".join(cells))


def print_cap_saturation():
    print("\n=== cap saturation (% of rows x tok/row grid at 6.0h cap, instruct) ===")
    total = len(ROWS) * len(TOK_PER_ROW)
    for name, params in MODELS:
        n = sum(1 for rows in ROWS for tpr in TOK_PER_ROW if hours(rows * tpr, params) >= MAX_TRAINING_HOURS)
        print(f"{name:>6} | {n:>2}/{total} = {100 * n / total:>5.1f}%")


def print_epochs_at_cap():
    print("\n=== epochs that fit at the granted hours (instruct, 700 tok/row, 175k rows) ===")
    tokens = 175_000 * 700
    for name, params in MODELS:
        h = hours(tokens, params)
        train_h = h - TRAINING_OVERHEAD_HOURS
        epochs = train_h * 3600 * analytic_tps_per_gpu(params) * gpu_count(params) / tokens
        print(f"{name:>6} | {h:>5.2f}h granted | {epochs:>4.2f} epochs fit")


if __name__ == "__main__":
    print_throughput_table()
    for tokens_per_row in [200, 400, 700, 1000]:
        print_grid("instruct", tokens_per_row)
    print_grid("dpo", 600)
    print_grid("grpo", 200)
    print_cap_saturation()
    print_epochs_at_cap()
