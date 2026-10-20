EMISSION_BURN_HOTKEY = "5GU4Xkd3dCGTU3s8VLcHGc5wsD5M8XyxDca5yDQhYm1mVXFu"

# General miner pool sizes
MIN_IDEAL_NUM_MINERS_IN_POOL = 8

# Evaluation retry limits
MAX_EVAL_ATTEMPTS = 4
MAX_TOURNAMENT_EVAL_ATTEMPTS = 3

# Task scoring
SCORE_PENALTY = -1
FIRST_PLACE_SCORE = 3

# PvP and individual environment scoring
INDIVIDUAL_WIN_MARGIN = 0.015
PVP_ENV_WIN_POINTS = 3
PVP_ENV_DRAW_POINTS = 1
PVP_ENV_LOSS_POINTS = 0
PVP_WIN_PCT_THRESHOLD = 0.60
PVP_PERF_DIFF_SLOPE = 0.125

# Tournament score and weight distribution
TOURNAMENT_PARTICIPATION_WEIGHT = 0.0001
TOURNAMENT_PAID_RANKS = 3
TOURNAMENT_SIMPLE_DECAY_BASE = 0.25
# Guaranteed minimum share per tournament type (text / image). Each type always receives at
# least this much of the emission pie; the remainder (EMISSION_FLEX_POOL) is split between the
# types by participation. Also the fallback split's floor when a type has no completed history.
TOURNAMENT_TEXT_WEIGHT = 0.50
TOURNAMENT_IMAGE_WEIGHT = 0.30
EMISSION_FLEX_POOL = 1.0 - (TOURNAMENT_TEXT_WEIGHT + TOURNAMENT_IMAGE_WEIGHT)  # 0.20

# Hard cap on any single tournament type's final (base + boost) weight: its minimum plus the
# whole flex pool, i.e. the most participation alone can ever hand it.
MAX_TEXT_TOURNAMENT_WEIGHT = TOURNAMENT_TEXT_WEIGHT + EMISSION_FLEX_POOL  # 0.70
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
