EMISSION_BURN_HOTKEY = "5GU4Xkd3dCGTU3s8VLcHGc5wsD5M8XyxDca5yDQhYm1mVXFu"

# General miner pool sizes
MIN_IDEAL_NUM_MINERS_IN_POOL = 8

# Evaluation retry limits
MAX_EVAL_ATTEMPTS = 4
MAX_TOURNAMENT_EVAL_ATTEMPTS = 3

# Fixed-track structural pre-check: hub metadata reads are retried in place this many times (backoff doubles
# from the base), then the whole evaluation is re-queued; an unreadable hub is never a scored fail.
STRUCTURAL_CHECK_READ_ATTEMPTS = 3
STRUCTURAL_CHECK_READ_BACKOFF_SECONDS = 5.0

# Task scoring
SCORE_PENALTY = -1
FIRST_PLACE_SCORE = 3

# Freeform quant penalty, all three pinned per task at creation (QuantFreeformRawTask) so retuning here
# never moves an old task. Every ratio is against the UNCOMPRESSED same-SKU bf16 reference loss of the
# input model, which is never compressed and never competes. An entry's effective size is
# total_bytes x exp(QUANT_PENALTY_CURVATURE x max(0, r - (1 + QUANT_PENALTY_KNEE))^2), lowest wins;
# above (1 + QUANT_BAR_CAP) x the reference the entry is out (ranks after every under-cap entry, closest
# ratio first). A task with NULL knee/curvature keeps the cliff at its bar_cap.
QUANT_PENALTY_KNEE = 0.04
QUANT_PENALTY_CURVATURE = 2500.0
QUANT_BAR_CAP = 0.08

# PvP and individual environment scoring
INDIVIDUAL_WIN_MARGIN = 0.015
PVP_ENV_WIN_POINTS = 3
PVP_ENV_DRAW_POINTS = 1
PVP_ENV_LOSS_POINTS = 0
PVP_WIN_PCT_THRESHOLD = 0.60
# Env win-rate -> performance-difference mapping, stored for analytics (not used for weights).
PVP_PERF_DIFF_SLOPE = 0.125
PVP_PERF_DIFF_AT_THRESHOLD = 0.10

# Tournament score and weight distribution
TOURNAMENT_PARTICIPATION_WEIGHT = 0.0001
TOURNAMENT_PAID_RANKS = 3
TOURNAMENT_SIMPLE_DECAY_BASE = 0.25
# Guaranteed minimum share per tournament type (text / quant / image). Each type always receives at
# least this much of the emission pie; the remainder (EMISSION_FLEX_POOL) is split between the
# types by participation. Also the fallback split's floor when a type has no completed history.
TOURNAMENT_TEXT_WEIGHT = 0.15
TOURNAMENT_QUANT_WEIGHT = 0.35
TOURNAMENT_IMAGE_WEIGHT = 0.225
EMISSION_FLEX_POOL = 1.0 - (TOURNAMENT_TEXT_WEIGHT + TOURNAMENT_QUANT_WEIGHT + TOURNAMENT_IMAGE_WEIGHT)  # 0.275

# Hard cap on any single tournament type's final (base + boost) weight: its minimum plus the
# whole flex pool, i.e. the most participation alone can ever hand it.
MAX_TEXT_TOURNAMENT_WEIGHT = TOURNAMENT_TEXT_WEIGHT + EMISSION_FLEX_POOL  # 0.425
MAX_QUANT_TOURNAMENT_WEIGHT = TOURNAMENT_QUANT_WEIGHT + EMISSION_FLEX_POOL  # 0.625
MAX_IMAGE_TOURNAMENT_WEIGHT = TOURNAMENT_IMAGE_WEIGHT + EMISSION_FLEX_POOL  # 0.50

# Participation-driven auto-balancer for the flex pool. Each cycle the flex pool is split from the
# rolling entrant counts of the last N completed tournaments of each type: an oversubscribed type
# cedes flex toward a starved one. Minimums are never touched. See validator.scoring.emission_balance.
EMISSION_BALANCE_ALPHA = 1.25            # rate of change: 0 = flex split at the minimums' ratio, higher = sharper reaction
# The window is LAGGED by one tournament: the most recent completed-or-active tournament (position
# -1) is excluded, and the average is taken over positions -2..-5. This locks the split a cohort
# competes under before they enter, so it can't move against them (rug them) when their tournament
# finishes. Window size therefore = number of completed tournaments averaged after the lag.
EMISSION_BALANCE_ROLLING_WINDOW = 4      # positions -2..-5 (one-tournament lag applied in the query)
EMISSION_MIN_ENTRANTS_FOR_BALANCE = 3    # ignore degenerate tournaments below this entrant count

# Burn and emission weight dynamics
BURN_REDUCTION_RATE = 5.0
MAX_BURN_REDUCTION = 0.8
EMISSION_MULTIPLIER_THRESHOLD = 0.10
EMISSION_MULTIPLIER_RATE = 2.0
SECONDS_PER_DAY = 86400.0

ALPHA_PER_SECOND = 1.0 / 12.0
MINER_ALPHA_SHARE = 0.41
DAILY_ALPHA_TO_MINERS = ALPHA_PER_SECOND * SECONDS_PER_DAY * MINER_ALPHA_SHARE
