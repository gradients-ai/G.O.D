from datetime import date


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
TOURNAMENT_PAID_RANKS = 2
TOURNAMENT_SIMPLE_DECAY_BASE = 0.25
# Uniform hard cap on any single tournament type's final (base + boost) weight.
MAX_TEXT_TOURNAMENT_WEIGHT = 0.50
MAX_IMAGE_TOURNAMENT_WEIGHT = 0.50
MAX_ENVIRONMENT_TOURNAMENT_WEIGHT = 0.50

# Anchor base split (text / image / env). These are the *neutral* bases used when
# participation is balanced across the anchor ratio; they double as the fallback base
# for a type with no completed-tournament history. Their sum is the balancer pool.
TOURNAMENT_TEXT_WEIGHT = 0.35
TOURNAMENT_IMAGE_WEIGHT = 0.25
TOURNAMENT_ENVIRONMENT_WEIGHT = 0.25

# Participation-driven auto-balancer. The base split is recomputed each cycle from the
# rolling entrant counts of the last N completed tournaments of each type: an oversubscribed
# type sheds weight (down to the floor) toward a starved one. See validator.scoring.emission_balance.
EMISSION_BALANCE_ALPHA = 1.25            # rate of change: 0 = static anchors, higher = sharper reaction
EMISSION_BALANCE_FLOOR = 0.05            # no type's base may drop below this
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
EMISSION_BOOST_DECAY_PER_WIN = 0.01
# Champion emission decays on a piecewise-linear *retention* curve (multiplier on
# the champion's full day-0 emission weight = base + boost). Fast early drop, a
# plateau, then a floor of 15% held from day 40 onwards — a champion's emission
# never drops below this. (days_since_reign_start, retention).
EMISSION_TIME_DECAY_CURVE: tuple[tuple[float, float], ...] = (
    (0.0, 1.00),
    (7.0, 0.50),
    (30.0, 0.30),
    (40.0, 0.15),
)
EMISSION_TIME_DECAY_START_DATE = date(2025, 11, 26)
SECONDS_PER_DAY = 86400.0

ALPHA_PER_SECOND = 1.0 / 12.0
MINER_ALPHA_SHARE = 0.41
DAILY_ALPHA_TO_MINERS = ALPHA_PER_SECOND * SECONDS_PER_DAY * MINER_ALPHA_SHARE
