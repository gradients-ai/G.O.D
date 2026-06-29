"""Constants for the shared PvP harness.

These live in core/ so the model-prep image, which ships only core/ plus
trainer/model_prep/, can run the harness.
"""

# Game / instance sampling
PVP_SEED_RANGE_MAX = 1_000_000
PVP_CONFIG_ID_DIVISOR = 100_000_000

# Per-turn wall-clock forfeit budget. A turn is a SINGLE model call (memory
# edits + the move in one response); this is the "stuck/too slow" cutoff.
# Qwen3-family tool calls can run into the low tens of seconds on busy eval GPUs,
# so keep this high enough for healthy generations but still bounded.
PVP_TURN_TIMEOUT_SECONDS = 30
# Reflection is best-effort and non-scoring, but it uses the same slow tool-call
# path as turns. Keep it bounded to avoid wedged post-game memory consolidation.
PVP_REFLECTION_TIMEOUT_SECONDS = 30
PVP_RETRY_BACKOFF_CAP_SECONDS = 32

# HTTP read timeout + retries for in-turn/reflection calls. This intentionally
# sits above the bot wall-clock alarm, so slow scored turns are classified by
# TurnTimeoutError and become forfeits before the transport layer abandons the
# in-flight SGLang request.
PVP_HTTP_READ_TIMEOUT_SECONDS = 35
PVP_HTTP_MAX_RETRIES = 1

# Tool-calling memory harness.
# Generation cap for a turn. A turn bundles memory edits AND the move in one
# response, so this must fit two full slot writes (~128 tokens of content each,
# plus tool-call JSON) + brief reasoning + the game_action call without
# truncating — a cut-off tool call parses to no move and forfeits.
PVP_TURN_MAX_TOKENS = 512
# Reflection writes a couple of long-term slots and makes no move.
PVP_REFLECTION_MAX_TOKENS = 384
PVP_MATCHUP_TIME_BUDGET_SECONDS = 900
PVP_WORKING_MEM_SLOTS = 4
PVP_WORKING_SLOT_TOKENS = 128
PVP_LONGTERM_MEM_SLOTS = 8
PVP_LONGTERM_SLOT_TOKENS = 128
