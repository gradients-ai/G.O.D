"""Constants for the shared PvP harness.

These live in core/ (not validator/) so the model-prep image — which ships only
core/ + trainer/model_prep/ — can run the harness. validator/core/constants.py
re-exports them so existing vcst.PVP_* references keep working.
"""

# Game / instance sampling
PVP_SEED_RANGE_MAX = 1_000_000
PVP_CONFIG_ID_DIVISOR = 100_000_000

# Per-turn wall-clock forfeit budget. A turn is a SINGLE model call (memory
# edits + the move in one response); this is the "stuck/too slow" cutoff.
# Sustained throughput on the eval GPU is ~57 tok/s; 512 output tokens takes ~9s.
# 15s gives 6s headroom over max generation, down from 30s.
PVP_TURN_TIMEOUT_SECONDS = 15
# Reflection generates up to 384 tokens (~7s); 12s gives reasonable headroom.
PVP_REFLECTION_TIMEOUT_SECONDS = 12
PVP_RETRY_BACKOFF_CAP_SECONDS = 32

# HTTP read timeout + retries for in-turn/reflection calls. Kept under the turn
# budget so a hung connection is caught (and at most one retry attempted) before
# the wall-clock alarm forfeits — the old 30s/10-retry defaults could never fit.
# Raised 12 -> 24: must cover a full ~18-20s generation so a slow-but-valid turn
# isn't aborted mid-stream (12s aborted before 512 tokens could finish).
PVP_HTTP_READ_TIMEOUT_SECONDS = 24
PVP_HTTP_MAX_RETRIES = 1

# Tool-calling memory harness.
# Generation cap for a turn. A turn bundles memory edits AND the move in one
# response, so this must fit two full slot writes (~128 tokens of content each,
# plus tool-call JSON) + brief reasoning + the game_action call without
# truncating — a cut-off tool call parses to no move and forfeits.
PVP_TURN_MAX_TOKENS = 512
# Reflection writes a couple of long-term slots and makes no move.
PVP_REFLECTION_MAX_TOKENS = 384
PVP_MATCHUP_TIME_BUDGET_SECONDS = 900  # 15 min per environment per pair

PVP_WORKING_MEM_SLOTS = 4
PVP_WORKING_SLOT_TOKENS = 128
PVP_LONGTERM_MEM_SLOTS = 8
PVP_LONGTERM_SLOT_TOKENS = 128
