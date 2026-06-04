"""Simulate the tournament training-hours formula across model size, rows, and
context length, and compare formula variants (old / branch / proposed-gentle).

Standalone — reimplements the formula so we can A/B variants without importing
the validator. Keep the constants in sync with validator/core/constants.py.

Run:  python scripts/training_hours_sim.py
"""

from collections.abc import Callable
from dataclasses import dataclass

# --- constants (mirror validator/core/constants.py) ---
TRAINING_HOURS_SCALE_START_ROWS = 75_000
TRAINING_HOURS_MAX_ROWS = 500_000
TRAINING_HOURS_MIN = 1.0
TRAINING_HOURS_MAX_BASE = 6.0
MAX_TRAINING_HOURS = 6.0
FULL_HOURS_MODEL_PARAMS = 14e9
MIN_HOURS_SCALE = 0.5
CTX_SCALE_MIN = 0.25
CTX_SCALE_MAX = 3.0
TYPE_MULT = {"instruct": 1.0, "dpo": 1.4, "grpo": 1.3}


def _half(x: float) -> float:
    return round(x * 2) / 2


# --- base hours (rows x model size x task type) ---
def base_hours_old(rows: int, params: float, mult: float = 1.0) -> float:
    """Round to half at every stage (current deployed)."""
    t = max(0.0, min(1.0, (rows - TRAINING_HOURS_SCALE_START_ROWS) / (TRAINING_HOURS_MAX_ROWS - TRAINING_HOURS_SCALE_START_ROWS)))
    hours = _half(TRAINING_HOURS_MIN + t * (TRAINING_HOURS_MAX_BASE - TRAINING_HOURS_MIN))
    if params < FULL_HOURS_MODEL_PARAMS:
        scale = MIN_HOURS_SCALE + (params / FULL_HOURS_MODEL_PARAMS) * (1.0 - MIN_HOURS_SCALE)
        hours = max(TRAINING_HOURS_MIN, _half(hours * scale))
    hours = max(TRAINING_HOURS_MIN, _half(hours * mult))
    return min(hours, MAX_TRAINING_HOURS)


def base_hours_new(rows: int, params: float, mult: float = 1.0) -> float:
    """Accumulate factors, round once (branch fix)."""
    t = max(0.0, min(1.0, (rows - TRAINING_HOURS_SCALE_START_ROWS) / (TRAINING_HOURS_MAX_ROWS - TRAINING_HOURS_SCALE_START_ROWS)))
    hours = TRAINING_HOURS_MIN + t * (TRAINING_HOURS_MAX_BASE - TRAINING_HOURS_MIN)
    if params < FULL_HOURS_MODEL_PARAMS:
        hours *= MIN_HOURS_SCALE + (params / FULL_HOURS_MODEL_PARAMS) * (1.0 - MIN_HOURS_SCALE)
    hours *= mult
    return min(max(TRAINING_HOURS_MIN, _half(hours)), MAX_TRAINING_HOURS)


# --- context scale variants ---
def ctx_old(packed: int) -> float:
    """Quadratic, pivot 1024 (current deployed)."""
    return max(CTX_SCALE_MIN, min(CTX_SCALE_MAX, (packed / 1024) ** 2))


def ctx_branch(packed: int) -> float:
    """Linear, pivot 512, span 512 (currently on the branch)."""
    return max(CTX_SCALE_MIN, min(CTX_SCALE_MAX, packed / 512))


def ctx_gentle(packed: int, ref: int = 512, span: int = 1024) -> float:
    """Linear, pivot `ref`, gentler slope: 1024->1.5, 2048->2.5."""
    return max(CTX_SCALE_MIN, min(CTX_SCALE_MAX, 1.0 + (packed - ref) / span))


@dataclass
class Variant:
    name: str
    base: callable
    ctx: callable


def full_hours(rows: int, params: float, packed: int, v: Variant, mult: float = 1.0) -> float:
    h = v.base(rows, params, mult)
    return min(max(TRAINING_HOURS_MIN, _half(h * v.ctx(packed))), MAX_TRAINING_HOURS)


# --- grids ---
MODELS = [("0.35B", 0.35e9), ("1.1B", 1.1e9), ("3B", 3e9), ("7B", 7e9), ("9B", 9.2e9), ("14B", 14e9), ("32B", 32e9)]
ROWS = [20_000, 75_000, 150_000, 300_000, 500_000]
CTX = [256, 512, 768, 1024, 1536, 2048, 3072]


def print_ctx_table():
    print("\n=== ctx_scale mapping ===")
    print(f"{'packed':>7} | {'old quad/1024':>13} | {'branch lin/512':>14} | {'gentle span1024':>15}")
    print("-" * 60)
    for p in [256, 512, 768, 1024, 1280, 1500, 1536, 2048, 2560, 3072]:
        print(f"{p:>7} | {ctx_old(p):>13.2f} | {ctx_branch(p):>14.2f} | {ctx_gentle(p):>15.2f}")


def print_base_grid():
    print("\n=== base hours (instruct, pre-ctx): rows x model ===   [old -> new]")
    header = "rows\\model | " + " | ".join(f"{n:>11}" for n, _ in MODELS)
    print(header)
    print("-" * len(header))
    for r in ROWS:
        cells = []
        for _, p in MODELS:
            cells.append(f"{base_hours_old(r,p):>4} ->{base_hours_new(r,p):>4}")
        print(f"{r:>9} | " + " | ".join(f"{c:>11}" for c in cells))


def print_full_grid(v: Variant, params: float, model_name: str, mult: float = 1.0):
    print(f"\n=== FULL hours [{v.name}] — model {model_name}: rows x packed_len ===")
    header = "rows\\ctx | " + " | ".join(f"{c:>6}" for c in CTX)
    print(header)
    print("-" * len(header))
    for r in ROWS:
        cells = [f"{full_hours(r, params, c, v, mult):>6}" for c in CTX]
        print(f"{r:>8} | " + " | ".join(cells))


if __name__ == "__main__":
    OLD = Variant("OLD: per-stage round + quad ctx", base_hours_old, ctx_old)
    BRANCH = Variant("BRANCH: single round + linear/512", base_hours_new, ctx_branch)
    GENTLE = Variant("GENTLE: single round + linear span1024", base_hours_new, ctx_gentle)

    print_ctx_table()
    print_base_grid()

    # full grids for a small and a large model under each ctx variant
    for v in (OLD, BRANCH, GENTLE):
        print_full_grid(v, 9.2e9, "9B")
    print_full_grid(GENTLE, 0.35e9, "0.35B")
    print_full_grid(GENTLE, 7e9, "7B-DPO", mult=TYPE_MULT["dpo"])
