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
# Env win-rate -> reported performance-difference mapping (analytics only).
PVP_PERF_DIFF_SLOPE = 0.125
PVP_PERF_DIFF_AT_THRESHOLD = 0.10

# Tournament score and weight distribution
TOURNAMENT_PARTICIPATION_WEIGHT = 0.0001
TOURNAMENT_PAID_RANKS = 3
# Share of a type's pool paid to 1st place. It starts at TOURNAMENT_FIRST_PLACE_SHARE_START when a new boss takes the
# crown and stays there for the boss's first defence. From the second defence it falls linearly by
# (START - FLOOR) / TOURNAMENT_FIRST_PLACE_DECLINE_DEFENCES per defence and reaches FLOOR at the 17th defence, where it
# stays until the boss is dethroned. The rest of the pool goes to 2nd and 3rd at TOURNAMENT_SECOND_PLACE_REST_FRACTION
# and 1 - TOURNAMENT_SECOND_PLACE_REST_FRACTION of it. See validator.scoring.tournaments.first_place_share.
TOURNAMENT_FIRST_PLACE_SHARE_START = 0.85
TOURNAMENT_FIRST_PLACE_SHARE_FLOOR = 0.50
TOURNAMENT_FIRST_PLACE_DECLINE_DEFENCES = 16
TOURNAMENT_SECOND_PLACE_REST_FRACTION = 0.75
# Guaranteed minimum share per tournament type (text / quant / image). Each type always receives at
# least this much of the emission pie; the remainder (EMISSION_FLEX_POOL) is split between the
# types by participation. Also the fallback split's floor when a type has no completed history.
TOURNAMENT_TEXT_WEIGHT = 0.25
TOURNAMENT_QUANT_WEIGHT = 0.15
TOURNAMENT_IMAGE_WEIGHT = 0.15
EMISSION_FLEX_POOL = 1.0 - (TOURNAMENT_TEXT_WEIGHT + TOURNAMENT_QUANT_WEIGHT + TOURNAMENT_IMAGE_WEIGHT)  # 0.45

# Upper bound of any single tournament type's pool: its minimum plus the whole flex pool, i.e. the
# most participation alone can ever hand it. The dynamic base always lies within [minimum, this].
MAX_TEXT_TOURNAMENT_WEIGHT = TOURNAMENT_TEXT_WEIGHT + EMISSION_FLEX_POOL  # 0.70
MAX_QUANT_TOURNAMENT_WEIGHT = TOURNAMENT_QUANT_WEIGHT + EMISSION_FLEX_POOL  # 0.60
MAX_IMAGE_TOURNAMENT_WEIGHT = TOURNAMENT_IMAGE_WEIGHT + EMISSION_FLEX_POOL  # 0.60

# Participation-driven auto-balancer for the flex pool. Each cycle the flex pool is split from the
# rolling contender counts of the last N completed tournaments of each type (a contender
# ranked strictly above the boss in round 1; see emission_balance.tournament_contenders): an oversubscribed type
# cedes flex toward a starved one. Minimums are never touched. See validator.scoring.emission_balance.
EMISSION_BALANCE_ALPHA = 1.25            # rate of change: 0 = flex split at the minimums' ratio, higher = sharper reaction
EMISSION_BALANCE_ROLLING_WINDOW = 4      # last N completed tournaments of each type, the latest included
EMISSION_MIN_ENTRANTS_FOR_BALANCE = 3    # ignore degenerate tournaments below this contender count
EMISSION_CONTENDER_SCAN_LIMIT = 12       # completed tournaments of a type looked at to fill the window

# Burn and emission weight dynamics
BURN_REDUCTION_RATE = 5.0
MAX_BURN_REDUCTION = 0.8
SECONDS_PER_DAY = 86400.0

ALPHA_PER_SECOND = 1.0 / 12.0
MINER_ALPHA_SHARE = 0.41
DAILY_ALPHA_TO_MINERS = ALPHA_PER_SECOND * SECONDS_PER_DAY * MINER_ALPHA_SHARE
