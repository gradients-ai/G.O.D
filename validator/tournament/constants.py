from datetime import date

from core.constants.environments import EnvironmentName
from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskType


TOURNAMENT_INTERVAL_HOURS = 120

# Tournament scheduling settings
TOURNAMENT_SCHEDULE_ENVIRONMENT_DAY_OF_WEEK = 0
TOURNAMENT_SCHEDULE_ENVIRONMENT_HOUR = 11
TOURNAMENT_SCHEDULE_TEXT_DAY_OF_WEEK = 0
TOURNAMENT_SCHEDULE_TEXT_HOUR = 13
TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK = 0
TOURNAMENT_SCHEDULE_IMAGE_HOUR = 15

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

TOURNAMENT_PENDING_CYCLE_INTERVAL = 15 * 60
TOURNAMENT_ACTIVE_CYCLE_INTERVAL = 15 * 60
TOURNAMENT_PENDING_ROUND_CYCLE_INTERVAL = 15 * 60


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
# Cap includes the injected boss. With 4 members, a group evaluates at most
# C(4, 2) = 6 PvP pairs.
MAX_ENVIRONMENT_GROUP_SIZE = 4
# Small env tournaments collapse too fast (one big group advancing 1 contender). When the
# field is smaller than SMALL_ENVIRONMENT_MAX_PARTICIPANTS, cap the group size lower so there
# are more groups, more contenders survive each round, and the bracket plays out over more rounds.
SMALL_ENVIRONMENT_MAX_PARTICIPANTS = 7  # i.e. fewer than 8
SMALL_ENVIRONMENT_GROUP_SIZE = 3


# Environment tournament round structure
ENV_ADVANCE_PER_GROUP = 1
ENV_FINAL_ROUND_TASK_COUNT = 3
ENV_ENVS_PER_ROUND_MULTIPLIER = 2  # R1=2, R2=4, R3=6 (capped at total available)
ENV_TRAINING_HOURS = 1.5
ENV_TRAINING_HOURS_BOSS_ROUND_FROM_SCRATCH = 3.0
ENV_TARGET_TOURN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# If set, forces this game to be the boss (final) round task and excludes it from earlier rounds.
# Set to None to let any game randomly be the boss round.
FORCED_BOSS_ENVIRONMENT: EnvironmentName | None = None

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

# Final round task counts
FINAL_ROUND_IMAGE_TASKS = 6
FINAL_ROUND_IMAGE_QWEN_ZIMAGE_TASKS = 3

# Explicit text boss-round mix (the two continuous-SFT lineages replace one GRPO + one DPO slot);
# FINAL_ROUND_TEXT_TASKS below = these + continuous-SFT.
FINAL_ROUND_TEXT_TASK_DISTRIBUTION: dict[TaskType, int] = {
    TaskType.INSTRUCTTEXTTASK: 2,
    TaskType.DPOTASK: 1,
    TaskType.GRPOTASK: 1,
}

PROBABILITY_OF_A_BIG_TEXT_MODEL = 0.2

# --- Continuous-SFT boss task ---------------------------------------------------------------
# Two parallel chat-SFT lineages carried across tournaments; each round trains the next stage-1
# chunk from the lineage's previous winner (or seed on first run).
#
# CONTINUOUS_SFT_LINEAGES maps lineage slug (the continuous_sft_state PK, encoded into the task ds
# for carry-forward routing) -> seed model, from which eval reads the tokenizer + chat template.
# The quasar seed is an HF mirror = teutonic king weights + silx-ai/Quasar-10B tokenizer
# (eos <|endoftext|>=248044) + quasar HUMAN/ASSISTANT template — NOT Quasar-Preview/157184.
CONTINUOUS_SFT_LINEAGES: dict[str, str] = {
    "quasar": "gradients-io-tournaments/continuous-sft-seed-quasar-king",
    "qwen": "Qwen/Qwen3-8B-Base",
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


# Lineages whose model ships custom-arch code (needs trust_remote_code); eval pins the modeling *.py
# to the audited seed mirror so miner code never runs. Standard-arch lineages (qwen) don't.
CONTINUOUS_SFT_REMOTE_CODE_LINEAGES: set[str] = {"quasar"}


def continuous_sft_remote_code_repo(lineage: str | None) -> str | None:
    """The audited custom-code source for a lineage (its fixed seed mirror), or None if the lineage
    is a standard architecture that should load without remote code."""
    if lineage in CONTINUOUS_SFT_REMOTE_CODE_LINEAGES:
        return CONTINUOUS_SFT_LINEAGES.get(lineage)
    return None


def continuous_sft_remote_code_repo_for_ds(ds: str | None) -> str | None:
    """Audited custom-code mirror for a task's ds: None for standard-arch or non-continuous tasks."""
    return continuous_sft_remote_code_repo(continuous_sft_lineage_from_ds(ds))


def continuous_sft_seed_repo(lineage: str | None) -> str | None:
    """The lineage's immutable seed model; eval pins the tokenizer/chat template here. None otherwise."""
    return CONTINUOUS_SFT_LINEAGES.get(lineage) if lineage else None


def continuous_sft_seed_repo_for_ds(ds: str | None) -> str | None:
    """Seed model for a task's ds (pins the eval tokenizer); None for non-continuous tasks."""
    return continuous_sft_seed_repo(continuous_sft_lineage_from_ds(ds))


# --- Pre-boss quasar task -------------------------------------------------------------------
# The last knockout before the boss round (a single pair — its winner becomes the boss
# challenger) also plays on quasar: a standard instruct task with a normal dataset pull, computed
# hours and param-based GPU sizing, where only the model is forced to the quasar seed mirror.
# No augmentation: perturbing and re-uploading the custom-arch seed is unexercised, and
# remote-code pinning keys off the exact seed repo.
PRE_BOSS_QUASAR_MODEL = CONTINUOUS_SFT_LINEAGES["quasar"]

# Seed mirrors that ship custom-arch code; any task whose BASE MODEL is one of these needs the
# same audited remote-code pinning as the corresponding continuous-SFT lineage.
_CUSTOM_ARCH_SEED_REPOS: set[str] = {
    CONTINUOUS_SFT_LINEAGES[lineage] for lineage in CONTINUOUS_SFT_REMOTE_CODE_LINEAGES
}


def is_custom_arch_seed_model(model_id: str | None) -> bool:
    """True when a task's base model is itself a custom-arch seed mirror (the pre-boss quasar task)."""
    return model_id in _CUSTOM_ARCH_SEED_REPOS


def is_pre_boss_quasar_task(task) -> bool:
    """True for the pre-boss forced-quasar instruct task (base model pinned to the quasar seed)."""
    return task.task_type == TaskType.INSTRUCTTEXTTASK and task.model_id == PRE_BOSS_QUASAR_MODEL


def remote_code_repo_for_task(model_id: str | None, ds: str | None) -> str | None:
    """Audited custom-code mirror for a task, or None for standard-arch tasks.

    Continuous-SFT tasks key by ds (the base is the carried winner, not the seed); the pre-boss
    forced-quasar instruct task keys by its base model being a custom-arch seed mirror itself.
    """
    repo = continuous_sft_remote_code_repo_for_ds(ds)
    if repo:
        return repo
    if is_custom_arch_seed_model(model_id):
        return model_id
    return None


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

# Obfuscation detection constants
OBFUSCATION_DETECTION_PATH = "./validator/tournament/obfuscation_detection/anti_obfuscation"

# Round Sanity Check
PERCENTAGE_OF_TASKS_SHOULD_BE_SUCCESS = 0.5

# Tournament participation fees (in RAO)
TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO = 250_000_000  # 0.25 TAO = 250,000,000 RAO
TOURNAMENT_ENVIRONMENT_PARTICIPATION_FEE_RAO = 250_000_000  # 0.25 TAO = 250,000,000 RAO
TOURNAMENT_IMAGE_PARTICIPATION_FEE_RAO = 200_000_000  # 0.2 TAO = 200,000,000 RAO
