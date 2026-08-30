from datetime import date

from core.constants.environments import EnvironmentName
from core.constants.environments import TrainingStartPoint
from core.models.image_models import ImageModelType
from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType


TOURNAMENT_INTERVAL_HOURS = 120

# A runner-up holds their base-pool share only until the next tournament's
# results replace the standings: one interval plus roughly the run itself.
RUNNER_UP_EMISSION_DAYS = 7

# Tournament scheduling settings
TOURNAMENT_SCHEDULE_ENVIRONMENT_DAY_OF_WEEK = 0
TOURNAMENT_SCHEDULE_ENVIRONMENT_HOUR = 9
TOURNAMENT_SCHEDULE_TEXT_DAY_OF_WEEK = 0
TOURNAMENT_SCHEDULE_TEXT_HOUR = 11
TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK = 0
TOURNAMENT_SCHEDULE_IMAGE_HOUR = 13

# Tournament start requirements
MIN_MINERS_FOR_ENV_TOURN = 5
MIN_MINERS_FOR_TOURN = 4  # within the small-tournament band (3..9): round 1 is a single group, top 2 advance

# Boss round historical task selection
BOSS_ROUND_HISTORICAL_START_DATE = date(2025, 6, 1)
BOSS_ROUND_HISTORICAL_END_DATE = date(2025, 8, 1)
MIN_SUCCESSFUL_SCORES_FOR_HISTORICAL_TASK = 2

MAX_TRAINING_ATTEMPTS = 2

# Smart prioritization thresholds for task fetching
PENDING_QUEUE_THRESHOLD_PER_TYPE = 8  # Fetch tournament tasks when pending per type < this
PENDING_QUEUE_THRESHOLD_FOR_BENCHMARK = 5  # Fetch benchmark tasks when pending < this

# Orchestrator cycle intervals (in seconds)
FETCH_TASKS_CYCLE_INTERVAL = 60  # 1 minute for testing
PROCESS_PENDING_TASKS_CYCLE_INTERVAL = 60
MONITOR_TRAINING_TASKS_CYCLE_INTERVAL = 60
MOVE_COMPLETED_TASKS_CYCLE_INTERVAL = 60
PERIODIC_GPU_AVAILABILITY_UPDATE_INTERVAL = 60
MODEL_PREP_CYCLE_INTERVAL = 30
MODEL_PREP_GPU_RESERVE_HOURS = 1.0

# Reject a task whose dataset near-duplicate rate (from baseline_stats) is at or above this
# fraction. Only applies to text tasks (instruct/dpo/grpo); env tasks have no dataset stats.
MAX_NEAR_DUPLICATE_RATE = 0.20

TOURNAMENT_PENDING_CYCLE_INTERVAL = 60  # 1 minute
TOURNAMENT_ACTIVE_CYCLE_INTERVAL = 60
TOURNAMENT_PENDING_ROUND_CYCLE_INTERVAL = 60


# Retry intervals (in seconds)
TRAINING_START_RETRY_INTERVAL = 1 * 60  # 1 minute

# Dstack orchestrator retry settings
DSTACK_RETRY_DELAY_MINUTES = 30
DSTACK_MAX_RETRIES = 3

# Dstack regions
DSTACK_IMAGE_REGIONS = ["CA-MTL-3", "CA-MTL-1", "AP-JP-1", "US-KS-2", "US-GA-2", "US-CA-2", "EUR-IS-1", "US-MO-1"]
DSTACK_TEXT_REGIONS = ["CA-MTL-1", "AP-JP-1", "US-KS-2", "US-GA-2", "US-CA-2", "EUR-IS-1", "US-MO-1"]

# Trainer requests
TRAINER_HTTP_TIMEOUT = 30.0  # seconds
# Grace period after GPU reservation before trusting trainer "available" reports.
# Covers the gap between dispatch and container startup (clone, docker build, etc).
GPU_RESERVATION_GRACE_PERIOD_SECONDS = 10 * 60  # 10 minutes
EXPECTED_TRAINING_START_MESSAGE = "Started Training!"
NO_RETRY_RESULT = "No Retry"


# Tournament structure constants
MAX_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND = 8
EXPECTED_GROUP_SIZE = 32
MIN_GROUP_SIZE = 20

# Small tournament (text/image) round-1 format.
# When a tournament starts with fewer than 15 competitors we don't want a thin
# knockout or a tiny group that still advances 8. Instead round 1 is a single
# group that plays SMALL_TOURNAMENT_GROUP_TASKS matches, and only the best
# SMALL_TOURNAMENT_ADVANCE advance (into the knockout that decides the boss
# challenger). Below SMALL_TOURNAMENT_MIN_PARTICIPANTS there aren't enough
# competitors to make this worthwhile, so we fall back to the normal knockout.
SMALL_TOURNAMENT_MIN_PARTICIPANTS = 3
SMALL_TOURNAMENT_MAX_PARTICIPANTS = 14  # i.e. fewer than 15 at tournament start
SMALL_TOURNAMENT_GROUP_TASKS = 3
SMALL_TOURNAMENT_ADVANCE = 2
MIN_ENVIRONMENT_GROUP_SIZE = 2
# Cap includes the injected boss. With 5 members, a group evaluates at most
# C(5, 2) = 10 PvP pairs.
MAX_ENVIRONMENT_GROUP_SIZE = 5
# Small env tournaments collapse too fast (one big group advancing 1 contender). When the
# field is smaller than SMALL_ENVIRONMENT_MAX_PARTICIPANTS, cap the group size lower so there
# are more groups, more contenders survive each round, and the bracket plays out over more rounds.
SMALL_ENVIRONMENT_MAX_PARTICIPANTS = 7  # i.e. fewer than 8
SMALL_ENVIRONMENT_GROUP_SIZE = 4


# Environment tournament round structure
ENV_ADVANCE_PER_GROUP = 1
ENV_FINAL_ROUND_TASK_COUNT = 3
ENV_ENVS_PER_ROUND_MULTIPLIER = 2  # R1=2, R2=4, R3=6 (capped at total available)
ENV_TRAINING_HOURS = 1.5
ENV_TRAINING_HOURS_SWE_INFINITE = ENV_TRAINING_HOURS + 1.0
ENV_TRAINING_HOURS_BOSS_ROUND_FROM_SCRATCH = 3.0
# Clean copy of the retired qwen (Qwen3-8B-Base) continuous-SFT final winner; base for the
# guaranteed swe_infinite boss task (TrainingStartPoint.PREVIOUS_WINNER / target-model start).
ENV_TARGET_TOURN_MODEL = "gradients-io-tournaments/swe-base-qwen3-8b-continuous"
# If set, forces this game to be the boss (final) round task and excludes it from earlier rounds.
# Set to None to let any game randomly be the boss round.
FORCED_BOSS_ENVIRONMENT: EnvironmentName | None = EnvironmentName.SWE_INFINITE

TOURNAMENT_PARTICIPANT_PING_BATCH_SIZE = 50
DEFAULT_PARTICIPANT_REPO = "https://github.com/rayonlabs/G.O.D"
DEFAULT_PARTICIPANT_COMMIT = "8631451156e2915070f77e5547ca0d5ed3d0eb8a"

LATEST_TOURNAMENTS_CACHE_TTL = 3600
LATEST_TOURNAMENTS_CACHE_KEY = "latest_tournaments_details"

CLAUDE_REPO_DIFF_MODEL = "claude-sonnet-4-5"
CLAUDE_REPO_DIFF_MAX_TURNS = 30
CLAUDE_REPO_DIFF_MAX_BUDGET_USD = 2
CLAUDE_REPO_DIFF_MAX_FOCUS_FILES = 180

TOURN_DEDUP_ENABLED = True
TOURN_DEDUP_CLAUDE_MODEL = "claude-opus-4-8"
TOURN_DEDUP_CLAUDE_MAX_TURNS = 60
TOURN_DEDUP_CLAUDE_MAX_BUDGET_USD = 15
TOURN_DEDUP_CONCURRENCY = 8

R1_TEXT_DATASET_BIN = (20_000, 75_000)

# Tournament task allocation
TEXT_TASKS_PER_GROUP = 1
IMAGE_TASKS_PER_GROUP = 1
ENVIRONMENT_TASKS_PER_GROUP = 1

# Round 1 uses one randomly selected high-capacity image architecture for every group.
ROUND_ONE_IMAGE_MODEL_TYPES = (
    ImageModelType.KREA2,
    ImageModelType.IDEOGRAM4,
)

# Final round task counts
FINAL_ROUND_IMAGE_TASKS = 6
FINAL_ROUND_IMAGE_TASK_DISTRIBUTION = {
    ImageModelType.KREA2: 2,
    ImageModelType.IDEOGRAM4: 2,
    ImageModelType.QWEN_IMAGE: 1,
    ImageModelType.Z_IMAGE: 1,
}

# Explicit text boss-round mix; FINAL_ROUND_TEXT_TASKS below = these + continuous-SFT.
FINAL_ROUND_TEXT_TASK_DISTRIBUTION: dict[TaskType, int] = {
    TaskType.INSTRUCTTEXTTASK: 2,
    TaskType.DPOTASK: 1,
    TaskType.GRPOTASK: 1,
}

PROBABILITY_OF_A_BIG_TEXT_MODEL = 0.2
# Number of boss-round text tasks (from FINAL_ROUND_TEXT_TASK_DISTRIBUTION) to force onto
# OVERSAMPLED_LATER_MODELS. Set to 0 to disable without removing the wiring.
FINAL_ROUND_OVERSAMPLED_TASKS = 0

# --- Continuous-SFT boss task ---------------------------------------------------------------
# Chat-SFT lineages carried across tournaments; each round trains the next stage-1 chunk from the
# lineage's previous winner (or seed on first run).
#
# CONTINUOUS_SFT_LINEAGES maps lineage slug (the continuous_sft_state PK, encoded into the task ds
# for carry-forward routing) -> seed model, from which eval reads the tokenizer + chat template.
# Seeds must be standard architectures: eval and model-prep load miner-controlled repos with
# trust_remote_code off, so a custom-arch seed would not load.
CONTINUOUS_SFT_LINEAGES: dict[str, str] = {
    "qwen3-14b": "Qwen/Qwen3-14B",
}
FINAL_ROUND_CONTINUOUS_SFT_TASKS = len(CONTINUOUS_SFT_LINEAGES)

# Derived so the boss-round completeness gate (task_creator) matches the real mix: a content-service
# failure dropping one lineage would otherwise weaken the "win all continuous-SFT tasks" dethrone rule.
FINAL_ROUND_TEXT_TASKS = sum(FINAL_ROUND_TEXT_TASK_DISTRIBUTION.values()) + FINAL_ROUND_CONTINUOUS_SFT_TASKS

# ds field is encoded as "{prefix}:{lineage}:{label}" so the completion hook can recover lineage.
CONTINUOUS_SFT_DS_PREFIX = "continuous-sft"


def continuous_sft_ds(lineage: str, label: str) -> str:
    """Encode the lineage slug into a continuous-SFT task ds field."""
    return f"{CONTINUOUS_SFT_DS_PREFIX}:{lineage}:{label}"


def continuous_sft_lineage_from_ds(ds: str | None) -> str | None:
    """Recover the lineage slug from a continuous-SFT task ds, or None if ds isn't one."""
    if not ds:
        return None
    parts = ds.split(":", 2)
    if len(parts) >= 2 and parts[0] == CONTINUOUS_SFT_DS_PREFIX:
        return parts[1]
    return None


def is_continuous_sft_task(task) -> bool:
    """True if a task is a continuous-SFT boss task (CHATTASK + CONTINUOUS_SFT start point)."""
    return task.task_type == TaskType.CHATTASK and task.training_start_point == TrainingStartPoint.CONTINUOUS_SFT


def continuous_sft_seed_repo(lineage: str | None) -> str | None:
    """The lineage's immutable seed model; eval pins the tokenizer/chat template here. None otherwise."""
    return CONTINUOUS_SFT_LINEAGES.get(lineage) if lineage else None


def continuous_sft_seed_repo_for_ds(ds: str | None) -> str | None:
    """Seed model for a task's ds (pins the eval tokenizer); None for non-continuous tasks."""
    return continuous_sft_seed_repo(continuous_sft_lineage_from_ds(ds))


# --- Pre-boss task --------------------------------------------------------------------------
# The last knockout before the boss round (a single pair — its winner becomes the boss challenger)
# is a standard instruct task with a normal dataset pull, computed hours and param-based GPU
# sizing, where only the model is forced to PRE_BOSS_MODEL. Augmentation, KL and YaRN stay off so
# both competitors train the exact published model.
PRE_BOSS_MODEL = "Qwen/Qwen3-32B"


def is_pre_boss_task(task) -> bool:
    """True for the pre-boss forced-model instruct task."""
    return task.task_type == TaskType.INSTRUCTTEXTTASK and task.model_id == PRE_BOSS_MODEL


# Initial/fallback budget only: GPUs stay forced at 4xH100 (gpu_requirements.py), but hours are
# resized post-prep by the general throughput pipeline (2-epoch budget over the measured chunk,
# capped at MAX_TRAINING_HOURS). This value survives only if prep produces no baseline stats.
CONTINUOUS_SFT_TRAINING_HOURS = 4.0

# Knockout round task counts
KNOCKOUT_PAIR_TASKS = 1

# Model size constants (in billions)
DEFAULT_MODEL_MIN_SIZE_B = 1
DEFAULT_MODEL_MAX_SIZE_B = 10
MODEL_SIZE_RANGE_MULTIPLIER_MIN = 0.8
MODEL_SIZE_RANGE_MULTIPLIER_MAX = 1.2

# Model parameter conversion
MODEL_PARAMS_TO_BILLIONS = 1e9

# Margin a challenger must beat the boss by to win a boss-round task (text/image
# only; env uses PVP_WIN_PCT_THRESHOLD). Applied additively on the boss score's
# magnitude (see challenger_beats_boss) so it stays correct for negative/zero GRPO
# rewards. Also used by the emission projection and boss-round analytics so they
# agree with crowning. See challenger_beats_boss in thresholds.py.
BOSS_ROUND_WIN_MARGIN = 0.01

# Paired per-example boss-round gate (instruct + DPO only; GRPO/env keep the relative margin
# above). A relative margin is the wrong scale for a log-likelihood loss: a difference of D nats
# is D nats of evidence whether the loss is 0.02 or 2.0, so `abs(boss) * 1%` collapses to nothing
# exactly where the task is most saturated. Worse, a scalar mean carries no information about its
# own uncertainty, so no threshold on it can separate a real win from held-out sampling noise.
#
# Instead both models are scored on the identical held-out set and compared example by example.
# Example difficulty dominates the variance in both losses and cancels in the pairing.
# See compare_paired_losses in thresholds.py.

# Per-example gap below which neither side won that example. 0.01 nats ~= 1% more probability
# assigned to it — trivial. Without a dead zone the win count is decided at the 5th decimal.
BOSS_ROUND_TIE_DEADZONE_NATS = 0.01
# Challenger must win this share of *decided* examples, at the bootstrap lower bound - so the
# observed rate has to run a little above it. Not a noise threshold: at ~1000 examples the standard
# error on a win rate is ~1.6%, and the bootstrap below is what handles noise. This is the policy
# statement of how much more *consistent* a challenger has to be.
#
# It deliberately does not carry the "never weaker than the old rule" job - the mean-gap floor
# below does that, since it scales to the old relative margin on high-loss tasks. That leaves this
# free to be tuned for consistency alone. A genuinely better model with a wide per-example spread
# can sit near 55% and still be the better model; demanding much more of it selects for
# low-variance submissions rather than good ones.
#
# Provisional - see the win-rate logging in round_results.py, which records the observed rate on
# every task, won or lost, so this can be recalibrated against real matchups rather than reasoning.
BOSS_ROUND_MIN_WIN_RATE = 0.55
# Challenger must also be better on average by this much, so it cannot win on a majority of
# hairline examples while being materially worse where it loses. Deliberately the same value as
# the tie dead zone: that constant already defines the smallest per-sample difference the rule
# treats as meaningful, and a second, larger answer to the same question is what made a model
# better by 0.011 nats on ALL 800 samples - 100% win rate - lose the task.
#
# Calibrated against 62 boss-round text tasks over 120 days, where the median separation between
# the two competitors was 0.0094 nats: at 0.02 only a quarter of matchups were close enough to
# even be winnable, which compounded over 4-of-5 tasks makes the crown near-permanent.
#
# Saturated tasks stay unwinnable through the dead zone rather than through this floor - at losses
# around 0.02 essentially no per-sample difference clears 0.01, so nothing is decided.
BOSS_ROUND_MIN_MEAN_GAP_NATS = 0.01
# Below this many decided examples the task cannot distinguish the two models. Saturated tasks
# land here naturally. Interlocks with the bootstrap: at exactly this many decided examples
# the standard error is ~5%, so a bare 65% will fail the bound and a much larger observed margin
# is needed — the gate is automatically stricter when it has less to go on.
# Both statistics must clear their bar at this one-sided bootstrap bound, not on the point estimate.
BOSS_ROUND_BOOTSTRAP_CONFIDENCE = 0.99
BOSS_ROUND_BOOTSTRAP_RESAMPLES = 10_000
# Fixed so two validators scoring the same boss round always agree. Never vary this.
BOSS_ROUND_BOOTSTRAP_SEED = 20260808
# Task types whose loss is a log-likelihood, and so eligible for the paired gate. GRPO rewards and
# env scores are on an arbitrary scale where a relative margin is not obviously wrong, and keep it.
# CHATTASK is here because the continuous-SFT boss tasks are ChatRawTask and route to the instruct
# evaluator - and they carry the "challenger must win EVERY continuous-SFT task" dethrone gate, so
# leaving them on the relative margin would leave the strictest rule resting on the weakest test.
PAIRED_BOSS_ROUND_TASK_TYPES = (TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.CHATTASK)

def expected_boss_round_task_count(tournament_type: TournamentType) -> int:
    """How many tasks a boss round is built with, for the types that can draw. 0 for the rest.

    Only text qualifies: PAIRED_BOSS_ROUND_TASK_TYPES is text-only, so image and environment boss
    rounds are decided on the relative margin, which has no draw. They return 0 and every
    draw-related rule keyed off this value stays switched off for them - in particular the dethrone
    bar keeps deriving from their own resolved task count, unchanged by any of this.

    Two uses, both needing the round's *built* size rather than its current one: capping deciders
    (a round holding more tasks than this has already had them, and prep-failure replacement swaps
    tasks without changing the count), and pinning the dethrone bar so a drawn task cannot lower it.
    """
    return FINAL_ROUND_TEXT_TASKS if tournament_type == TournamentType.TEXT else 0


# Obfuscation detection constants
OBFUSCATION_DETECTION_PATH = "./validator/tournament/obfuscation_detection/anti_obfuscation"

# Round Sanity Check
PERCENTAGE_OF_TASKS_SHOULD_BE_SUCCESS = 0.5

# Tournament participation fees (in RAO)
TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO = 350_000_000  # 0.35 TAO = 350,000,000 RAO
TOURNAMENT_ENVIRONMENT_PARTICIPATION_FEE_RAO = 300_000_000  # 0.3 TAO = 300,000,000 RAO
TOURNAMENT_IMAGE_PARTICIPATION_FEE_RAO = 200_000_000  # 0.2 TAO = 200,000,000 RAO
