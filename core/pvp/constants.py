"""Constants for the shared PvP harness.

These live in core/ (not validator/) so the model-prep image — which ships only
core/ + trainer/model_prep/ — can run the harness. validator/core/constants.py
re-exports them so existing vcst.PVP_* references keep working.
"""

# Game / instance sampling
PVP_SEED_RANGE_MAX = 1_000_000
PVP_CONFIG_ID_DIVISOR = 100_000_000

# Per-turn wall-clock forfeit budget. Covers the whole multi-step tool loop;
# a healthy turn is a few seconds, this is the "stuck/too slow" cutoff.
PVP_TURN_TIMEOUT_SECONDS = 15
PVP_RETRY_BACKOFF_CAP_SECONDS = 32

# Tool-calling memory harness
# Max tool round-trips per turn before forfeiting (bounds the agentic loop).
PVP_MAX_INNER_STEPS = 4
# Generation cap per inner step. A step writes at most one slot (<=128 tokens)
# plus brief reasoning + tool-call JSON, so this is comfortably sufficient.
PVP_PER_STEP_MAX_TOKENS = 256
PVP_WORKING_MEM_SLOTS = 4
PVP_WORKING_SLOT_TOKENS = 128
PVP_LONGTERM_MEM_SLOTS = 8
PVP_LONGTERM_SLOT_TOKENS = 128
