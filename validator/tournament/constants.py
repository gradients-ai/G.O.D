from datetime import date

from core.constants.environments import TrainingStartPoint
from core.models.image_models import ImageModelType
from core.models.quant_models import QuantFormat
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.tournament.models import QUANT_SLOT_SPECS
from validator.tournament.models import ChunkScheme
from validator.tournament.models import ContinuousRowsPart
from validator.tournament.models import QuantRoundSpec
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import SeedRole
from validator.tournament.models import TextRoundSpec


TOURNAMENT_INTERVAL_HOURS = 120

# A runner-up holds their base-pool share only until the next tournament's
# results replace the standings: one interval plus roughly the run itself.
RUNNER_UP_EMISSION_DAYS = 7

# Tournament scheduling settings
TOURNAMENT_SCHEDULE_TEXT_DAY_OF_WEEK = 0
TOURNAMENT_SCHEDULE_TEXT_HOUR = 11
TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK = 0
TOURNAMENT_SCHEDULE_IMAGE_HOUR = 13
# Monday 09:00 UTC (text is 11:00, image 13:00).
TOURNAMENT_SCHEDULE_QUANT_DAY_OF_WEEK = 0
TOURNAMENT_SCHEDULE_QUANT_HOUR = 9

# Every tournament (any type) is expected to be completed by this day/hour, independent of
# training time, GPU availability, and round reverts during the week. Completion is a manual
# step (see advance_tournament's "update DB manually" note), so this deadline only drives the
# Discord reminder below - it never forces a tournament to finish.
TOURNAMENT_DEADLINE_DAY_OF_WEEK = 4  # Friday
TOURNAMENT_DEADLINE_HOUR = 14
TOURNAMENT_DEADLINE_ALERT_MINUTES_BEFORE = 30

# Tournament start requirements
MIN_MINERS_FOR_TOURN = 4  # text, quant and image minimum

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
# Freeform publication reconcile. Slow on purpose: every failing item re-downloads GBs from MinIO and re-alerts Discord.
FREEFORM_PUBLICATION_RECONCILE_INTERVAL = 3600
MODEL_PREP_GPU_RESERVE_HOURS = 1.0

# Reject a task whose dataset near-duplicate rate (from baseline_stats) is at or above this
# fraction. Only applies to text tasks (instruct/dpo/grpo); env tasks have no dataset stats.
MAX_NEAR_DUPLICATE_RATE = 0.15

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

# Multi-match round-1 format. Text and image round 1 are always a single group
# that plays SMALL_TOURNAMENT_GROUP_TASKS matches, and only the best
# SMALL_TOURNAMENT_ADVANCE advance into the knockout that decides the boss
# challenger.
SMALL_TOURNAMENT_GROUP_TASKS = 3
SMALL_TOURNAMENT_ADVANCE = 2

TOURNAMENT_PARTICIPANT_PING_BATCH_SIZE = 50
# The pending loop runs every TOURNAMENT_PENDING_CYCLE_INTERVAL; a tournament short of miners only re-pings the
# network this often, so a thin tournament neither blocks the others nor hammers every miner every minute.
TOURNAMENT_PARTICIPANT_RETRY_INTERVAL_SECONDS = 30 * 60
# Boss decoder snapshot: a transient failure (clone timeout, MinIO) is retried a couple of times in place, then by the
# pending loop no more often than the cooldown. A boss with genuinely no usable decoder is never retried.
BOSS_DECODER_SNAPSHOT_ATTEMPTS = 2
BOSS_DECODER_SNAPSHOT_RETRY_SLEEP_SECONDS = 20
BOSS_DECODER_REPAIR_COOLDOWN_SECONDS = 10 * 60
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

# Tournament task allocation
TEXT_TASKS_PER_GROUP = 1
IMAGE_TASKS_PER_GROUP = 1

# Image R1 has one smaller task for each family. Thirty source pairs map to at
# most three test images and at most 0.75 competition hours.
ROUND_ONE_IMAGE_MODEL_TYPES = (
    ImageModelType.FLUX,
    ImageModelType.IDEOGRAM4,
    ImageModelType.Z_IMAGE,
)
ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS = 30

# Text R1 instruct tasks keep the throughput-based hours formula unchanged.
# A draw longer than this is discarded and another task is created (creation
# estimate, and again after model prep from measured tokens). Hours are never
# shortened to fit.
ROUND_ONE_TEXT_MAX_TRAINING_HOURS = 2.5

# The single R2 knockout task randomly uses one of these larger families.
ROUND_TWO_IMAGE_MODEL_TYPES = (
    ImageModelType.QWEN_IMAGE,
    ImageModelType.KREA2,
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
# One of the 3 instruct-text tasks is always forced onto a large model (see
# BOSS_ROUND_LARGE_INSTRUCT_MIN/MAX_SIZE_B below); the other tasks draw from the standard pool.
FINAL_ROUND_TEXT_TASK_DISTRIBUTION: dict[TaskType, int] = {
    TaskType.INSTRUCTTEXTTASK: 3,
    TaskType.DPOTASK: 1,
    TaskType.GRPOTASK: 1,
}

# One boss-round instruct-text task is always forced onto a model in this size band,
# instead of the normal standard-pool draw. Augmentation is skipped for this slot:
# publishing a full 70B copy during model-prep regularly exceeds the prep timeout.
# Prep-failure replacement redraws from this same band (it does not pin the failed model).
BOSS_ROUND_LARGE_INSTRUCT_MIN_SIZE_B = 30.0
BOSS_ROUND_LARGE_INSTRUCT_MAX_SIZE_B = 72.0

# --- Continuous-SFT boss task ---------------------------------------------------------------
# Chat-SFT lineages carried across tournaments; each round trains the next stage-1 chunk from the
# lineage's previous winner (or seed on first run).
#
# CONTINUOUS_SFT_LINEAGES maps lineage slug (the continuous_sft_state PK, encoded into the task ds
# for carry-forward routing) -> seed model, from which eval reads the tokenizer + chat template.
# Seeds must be standard architectures: eval and model-prep load miner-controlled repos with
# trust_remote_code off, so a custom-arch seed would not load.
# New slug on purpose: the retired `qwen` row still holds the old Qwen3-8B-Base winner, and
# reusing that slug would resume it. A slug with no continuous_sft_state row starts at this seed, chunk 0.
CONTINUOUS_SFT_LINEAGES: dict[str, str] = {
    "qwen3-8b": "Qwen/Qwen3-8B-Base",
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


# Fixed training budget: every continuous-SFT task gets exactly this many hours, pinned as both floor and
# cap after prep (compute_hours_from_baseline_stats ignores the measured stats for these tasks). GPUs stay
# a forced 4xH100 (gpu_requirements.py).
CONTINUOUS_SFT_TRAINING_HOURS = 2.5

# The chunk scheme of every continuous-SFT task this code creates; its train_index counts chunks of that scheme.
CONTINUOUS_SFT_CHUNK_SCHEME = ChunkScheme.FIFTY_K

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

    Only text and quant qualify: PAIRED_BOSS_ROUND_TASK_TYPES is text-only, so image and environment boss
    rounds are decided on the relative margin, which has no draw. They return 0 and every
    draw-related rule keyed off this value stays switched off for them - in particular the dethrone
    bar keeps deriving from their own resolved task count, unchanged by any of this.

    Two uses, both needing the round's *built* size rather than its current one: capping deciders
    (a round holding more tasks than this has already had them, and prep-failure replacement swaps
    tasks without changing the count), and pinning the dethrone bar so a drawn task cannot lower it.
    """
    if tournament_type == TournamentType.QUANT:
        return FINAL_ROUND_QUANT_TASKS
    return FINAL_ROUND_TEXT_TASKS if tournament_type == TournamentType.TEXT else 0


# Obfuscation detection constants
OBFUSCATION_DETECTION_PATH = "./validator/tournament/obfuscation_detection/anti_obfuscation"

# Round Sanity Check
PERCENTAGE_OF_TASKS_SHOULD_BE_SUCCESS = 0.5

# Tournament participation fees (in RAO)
TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO = 700_000_000  # 0.7 TAO = 700,000,000 RAO
TOURNAMENT_IMAGE_PARTICIPATION_FEE_RAO = 400_000_000  # 0.4 TAO = 400,000,000 RAO
TOURNAMENT_QUANT_PARTICIPATION_FEE_RAO = 600_000_000  # 0.6 TAO = 600,000,000 RAO

# Multi-GPU (>1 GPU) tasks must land on high-interconnect trainers (SXM or NVLink):
# split across PCIe GPUs the cross-device traffic bottlenecks and the run stalls/OOMs.
# When True, the scheduler skips PCIe/unknown trainers for >1-GPU requirements and waits
# for a suitable box rather than placing a multi-GPU job somewhere it will struggle.
MULTI_GPU_REQUIRES_HIGH_INTERCONNECT = True


# =====================================================================================================
# Quant tournament
# =====================================================================================================

# Quant tournament rounds. Each of rounds 1-3 is one quant task (one group, the boss excluded) on the text boss's model of the
# matching text round: round 1 compresses the model the text boss trained on the bound text tournament's text_r1 seed task,
# round 2 its text_r2 model and round 3 its text_r3 model. Every miner compresses that one model. Standings are the
# average rank over every round played, ranks within the surviving cohort; the lone survivor of round 3 plays the boss round.
QUANT_ROUND_PLAN = (
    QuantRoundSpec(
        round_number=1,
        slot=QuantSourceSlot.R1,
        task_type=QUANT_SLOT_SPECS[QuantSourceSlot.R1].task_type,
        input_role=QUANT_SLOT_SPECS[QuantSourceSlot.R1].text_role,
        advance=8,
    ),
    QuantRoundSpec(
        round_number=2,
        slot=QuantSourceSlot.R2,
        task_type=QUANT_SLOT_SPECS[QuantSourceSlot.R2].task_type,
        input_role=QUANT_SLOT_SPECS[QuantSourceSlot.R2].text_role,
        advance=4,
        quant_format=QuantFormat.W2A16_ASYM_G64,
    ),
    QuantRoundSpec(
        round_number=3,
        slot=QuantSourceSlot.R3,
        task_type=QUANT_SLOT_SPECS[QuantSourceSlot.R3].task_type,
        input_role=QUANT_SLOT_SPECS[QuantSourceSlot.R3].text_role,
        advance=1,
    ),
)
QUANT_ROUND_ONE_ADVANCE = QUANT_ROUND_PLAN[0].advance
QUANT_ROUND_TWO_ADVANCE = QUANT_ROUND_PLAN[1].advance
QUANT_ROUND_THREE_ADVANCE = QUANT_ROUND_PLAN[2].advance
# Pause before re-listing a round-2 quant input repo after a hub error, so a transient one can clear.
QUANT_INPUT_LISTING_RETRY_DELAY_SECONDS = 5
# Salts the per-round seed that breaks group-standing ties left after average rank, most-recent-round rank and summed gap.
# Every validator derives the same seed from the round id; bump the version to reshuffle all rounds.
R2_TIEBREAK_SEED_SALT = "quant-r2-tiebreak-v1"

# The quant boss round is six compression tasks, three freeform and three fixed: the freeform and the fixed compression of the
# text boss's round-2, round-3 and continuous models. The quant boss trains its six entries as soon as each source model has
# landed (they are seed tasks); the challenger trains the same tasks in the boss round and compresses the same model.
# The challenger is crowned by winning 5 of the 6 (QUANT_BOSS_ROUND_ALLOWED_LOSSES). A freeform task is won with at least
# QUANT_FREEFORM_BOSS_BYTE_MARGIN fewer effective bytes; a fixed task needs the paired per-example test and, on top of it,
# a scalar loss QUANT_FIXED_BOSS_MIN_IMPROVEMENT better. The fixed tasks use 2-bit or 3-bit formats only
# (QUANT_BOSS_ROUND_FIXED_FORMATS).
FINAL_ROUND_QUANT_TASK_DISTRIBUTION: dict[TaskType, int] = {
    TaskType.QUANTFREEFORMTASK: 3,
    TaskType.QUANTFIXEDTASK: 3,
}
FINAL_ROUND_QUANT_TASKS = sum(FINAL_ROUND_QUANT_TASK_DISTRIBUTION.values())

# A tournament type with no completed tournament of its own takes its first boss from the latest completed tournament of the
# mapped type. After its first completed tournament the mapping is never consulted again.
BOSS_SEED_TOURNAMENT_TYPE: dict[TournamentType, TournamentType] = {TournamentType.QUANT: TournamentType.TEXT}

# The formats the three fixed tasks of the quant boss round draw from: 2-bit and 3-bit only, so the win margin stays reachable.
# Drawn per tournament and role (a restart or a replacement redraws the same format; the three tasks may differ).
QUANT_BOSS_ROUND_FIXED_FORMATS = (QuantFormat.W3A16_ASYM_G128, QuantFormat.W2A16_ASYM_G64)

# quant_failure for every player of a chained task when no upstream instruct output is usable.
UPSTREAM_FAILURE_REASON = "upstream instruct run failed"
# quant_failure for a player of an own-model quant round (OWN_R1_MODEL) whose round-1 repo is missing or not a full model.
NO_USABLE_R1_MODEL_REASON = "no usable round-1 model"

# The challenger takes the freeform boss task only by being under the cap with at least this fraction
# fewer effective bytes than the boss: total bytes (artifact + decoder) x the loss penalty
# exp(curvature x max(0, r - knee)^2). Byte counts and the penalty are deterministic, so no bootstrap.
QUANT_FREEFORM_BOSS_BYTE_MARGIN = 0.03

# Fixed quant is judged by the same paired per-example test as text; freeform is decided on effective bytes, not loss.
QUANT_PAIRED_TASK_TYPES = (TaskType.QUANTFIXEDTASK,)

# Fixed quant boss-round task: the challenger needs the paired per-example test and, on top of it, a scalar loss at least
# this fraction better than the boss's.
QUANT_FIXED_BOSS_MIN_IMPROVEMENT = 0.01

# The quant boss round is built with FINAL_ROUND_QUANT_TASKS tasks and the challenger is crowned by winning all but this many
# of them (determine_boss_round_winner): 5 of 6. The bar is always the built size, so a task that was never attached (its seed
# failed) or never decided is simply not a win. Nothing is asked of the lost tasks.
QUANT_BOSS_ROUND_ALLOWED_LOSSES = 1

# A quant task held on a text seed is released on the text task's own base model (the fallback) when the text boss's run
# succeeded but its output is not usable (a hub lookup can fail or lag right after the run). The fallback is final, so while
# the text tournament runs it is taken only once the seed row is trained or ready (the seed loop has seen the finished run)
# and has not changed for this long; before that the task keeps holding.
QUANT_SOURCE_FALLBACK_GRACE_SECONDS = 30 * 60

# A quant tournament can bind to a text tournament that completed within this many seconds (7 days): its seed state is final.
QUANT_SOURCE_COMPLETED_WINDOW_SECONDS = 7 * 24 * 3600

# A quant tournament that has waited this long for a text tournament to bind to raises one Discord alert (the wait is measured
# from the first time the process saw it waiting).
QUANT_SOURCE_WAIT_ALERT_SECONDS = 2 * 3600

# Analytics only (the boss-battle projection, never crowning): a lost boss task counts as lost narrowly when the challenger is
# under the cap and its effective bytes are at most this much above the boss's on freeform; on fixed there is no constant,
# the loss is narrow unless the boss would have beaten the challenger under the same paired bar with the sides swapped.
BOSS_ROUND_FREEFORM_ALLOWED_LOSS_WITHIN = 0.01

# Re-evaluations of rows whose SKU has no cached bf16 reference before the task is swapped.
QUANT_REFERENCE_EVAL_RETRIES = 1

# --- Own-model quant reference pre-pass (validator/tournament/quant_reference_prepass.py) ----------------------
# Concurrent bf16 reference measurements (one Basilica deployment each) across all (task, hotkey) pairs.
QUANT_REFERENCE_PREPASS_CONCURRENCY = 4
# Consecutive infra failures of one (task, hotkey) measurement before a human is pinged (it keeps retrying).
QUANT_REFERENCE_PREPASS_FAILURES_BEFORE_ALERT = 3
# Consecutive failures of one (task, hotkey) bf16 measurement on the calibration rows before training stops waiting for
# it and publishes the calibration examples without reference losses.
QUANT_CALIBRATION_REFERENCE_MAX_FAILURES = 3
# After this many consecutive prepass failures of a task-common task's '' bf16 reference the orchestrator stops waiting and
# the miners train with no published target (scoring measures the SKU in-deployment). The alert already fires at
# QUANT_REFERENCE_PREPASS_FAILURES_BEFORE_ALERT. Own-model tasks keep waiting.
QUANT_TASK_COMMON_REFERENCE_MAX_FAILURES = 6

# --- Quant registration build lint (validator/tournament/registration_lint.py) -------------------------------
# Immutable-dependency rule: hash-pinned pip, SHA-pinned git, no fetch of miner-controlled URLs.
# True excludes a violating entrant from the quant tournament; False is warn-only (violations are logged with
# their reasons and the entrant is enrolled).
QUANT_REGISTRATION_LINT_ENFORCE = False
# Base images a quant Dockerfile may name by tag. These namespaces belong to their publishers, so a miner cannot
# move the tag; any other image must be digest-pinned (image@sha256:...). A bare name ("python") is a Docker
# official image and always trusted.
QUANT_TRUSTED_BASE_IMAGE_PREFIXES = (
    "runpod/",
    "nvidia/",
    "nvcr.io/nvidia/",
    "pytorch/",
    "axolotlai/",
    "gradientsio/",
    "docker.io/library/",
    "library/",
)
# URL prefixes a build may curl/wget/ADD without a checksum: publisher-owned hosts no miner controls. Anything
# else needs `ADD --checksum=sha256:...` or a `sha256sum -c` in the same RUN (same build script).
QUANT_TRUSTED_FETCH_URL_PREFIXES = (
    "https://download.pytorch.org/",
    "https://developer.download.nvidia.com/",
    "https://files.pythonhosted.org/",
    "https://astral.sh/uv/",
)

# --- Continuous-data tasks of the boss round: the continuous instruct task and its two compression tasks -----------
# The continuous instruct task is a CHATTASK on a raw quant-pool base trained on one continuous-SFT chat chunk; its compressions
# (its freeform / fixed compression tasks) are quant tasks on the SAME chunk. The chunk is pulled once per tournament. Their
# train rows are two disjoint, tournament-seeded shares of that chunk's train file (select_continuous_rows at prep):
# the instruct task the first QUANT_CONTINUOUS_INSTRUCT_TRAIN_ROWS, the compressions the next ones. Neither is a continuous-SFT
# task: the ds prefixes are on purpose not parsed by continuous_sft_lineage_from_ds (which would pin a seed tokenizer and size
# hours off a seed), and the start point stays DEFAULT / OWN_UPSTREAM_MODEL (CONTINUOUS_SFT forces 4xH100).
# Compression tasks: ds is "{prefix}:{train_index}:{tournament_id}:{label}".
QUANT_CONTINUOUS_DS_PREFIX = "continuous-quant"
# Continuous instruct task: ds is "{prefix}:{train_index}:{tournament_id}:{label}". The tournament id seeds the row split and
# the prefix marks it for the GPU tier; continuous_sft_lineage_from_ds does not match either prefix.
CONTINUOUS_CHAT_DS_PREFIX = "continuous-chat"
# A cursor row in continuous_sft_state (not a lineage: it is not in CONTINUOUS_SFT_LINEAGES). One advance per tournament
# hands out the chunk index; once a task exists its ds is the truth, not the row.
QUANT_CONTINUOUS_LINEAGE = "quant-random-base"
# Row-count estimate of one chunk, used only for the creation-time hours estimate; prep resizes.
QUANT_CONTINUOUS_CHUNK_ROWS = 95_000
# Each of the two shares is this many rows (halves of the chunk if it has fewer than twice as many).
QUANT_CONTINUOUS_INSTRUCT_TRAIN_ROWS = 40_000
# The continuous instruct task trains on this multiple of the GPUs its size would normally take (2 up to 4B, 4 above:
# 1x -> 2x, 2x -> 4x).
CONTINUOUS_INSTRUCT_GPU_MULTIPLIER = 2


def continuous_quant_ds(train_index: int, tournament_id: str, label: str) -> str:
    """A compression task's ds: the chunk index, the tournament that seeds the row split, and the content service label."""
    return f"{QUANT_CONTINUOUS_DS_PREFIX}:{train_index}:{tournament_id}:{label}"


def continuous_quant_train_index(ds: str | None) -> int | None:
    """Recover the chunk index from a compression task's ds, or None if ds isn't one."""
    if not ds:
        return None
    parts = ds.split(":", 2)
    if len(parts) == 3 and parts[0] == QUANT_CONTINUOUS_DS_PREFIX and parts[1].isdigit():
        return int(parts[1])
    return None


def continuous_quant_tournament(ds: str | None) -> str | None:
    """The tournament id carried in a compression task's ds, or None if ds isn't one (or carries none)."""
    if continuous_quant_train_index(ds) is None:
        return None
    parts = ds.split(":", 3)
    return parts[2] if len(parts) == 4 and parts[2] else None


def is_continuous_quant_task(task) -> bool:
    """True if a task is a quant task on a continuous-SFT chunk (one of the continuous instruct task's two compressions)."""
    return task.task_type in QUANT_TASK_TYPES and continuous_quant_train_index(getattr(task, "ds", None)) is not None


def continuous_chat_ds(train_index: int, tournament_id: str, label: str) -> str:
    """The continuous instruct task's ds: the chunk index, the tournament id that seeds the row split, the content label."""
    return f"{CONTINUOUS_CHAT_DS_PREFIX}:{train_index}:{tournament_id}:{label}"


def continuous_chat_parts(ds: str | None) -> tuple[int, str] | None:
    """(chunk index, tournament id) of the continuous instruct task's ds, or None if ds isn't its."""
    if not ds:
        return None
    parts = ds.split(":", 3)
    if len(parts) == 4 and parts[0] == CONTINUOUS_CHAT_DS_PREFIX and parts[1].isdigit():
        return int(parts[1]), parts[2]
    return None


def is_continuous_instruct_task(task) -> bool:
    """True if a task is the continuous instruct task, the boss round's continuous-chunk instruct task."""
    return task.task_type == TaskType.CHATTASK and continuous_chat_parts(getattr(task, "ds", None)) is not None


def continuous_rows_part(task) -> tuple[ContinuousRowsPart, str] | None:
    """(which share of the chunk's train rows, split seed) of the continuous instruct task or a compression task, else None.

    The seed is "{tournament_id}:{train_index}", the same for all three tasks, which is what makes their shares of one
    chunk disjoint and reproducible."""
    chat = continuous_chat_parts(getattr(task, "ds", None))
    if chat is not None and task.task_type == TaskType.CHATTASK:
        return ContinuousRowsPart.INSTRUCT, f"{chat[1]}:{chat[0]}"
    train_index, tournament_id = continuous_quant_train_index(getattr(task, "ds", None)), continuous_quant_tournament(task.ds)
    if task.task_type in QUANT_TASK_TYPES and train_index is not None and tournament_id is not None:
        if getattr(task, "training_start_point", None) == TrainingStartPoint.DEFAULT:
            # A shared-input compression of the text boss's continuous model: no instruct task takes the other share, and the
            # text task trained on the whole chunk, so the quant task takes every row (a chunk is about 47.5k rows, under
            # the 80k a split into two 40k shares needs).
            return None
        return ContinuousRowsPart.COMPRESSION, f"{tournament_id}:{train_index}"
    return None


def gpu_count_multiplier(task_type: TaskType, ds: str | None) -> int:
    """How many times its normal GPU tier a task trains on: CONTINUOUS_INSTRUCT_GPU_MULTIPLIER for the continuous instruct
    task, 1 otherwise."""
    if task_type == TaskType.CHATTASK and continuous_chat_parts(ds) is not None:
        return CONTINUOUS_INSTRUCT_GPU_MULTIPLIER
    return 1


# Seed tasks: everything the boss trains, created and trained upfront when the first round is created
# (validator/tournament/seed_tasks.py).
# Model-size bands in B params for the two instruct input seeds, both inside QUANT_MODEL_SIZE_B.
SEED_INPUT_SMALL_SIZE_B = (1.0, 4.0)
SEED_INPUT_LARGE_SIZE_B = (4.0, 8.0)
# The one switch for the 7-8B case: when True the large input seed draws from the small band instead.
SEED_INPUT_LARGE_FALLS_BACK_TO_SMALL = False
# Replacements per role after a seed fails in training (the boss's row failed, a FAILURE status, a non-full-model output, an
# input seed that scored no fine-tune); one more such failure marks the row failed and halts the tournament. A prep-stage
# dataset rejection of a dataset-drawn role (PREP_TASK_FAILURE: near-duplicate or any other prep rejection) does not count:
# it redraws like prod's uncapped replacement path, bounded only by SEED_MAX_PREP_REJECTIONS.
SEED_MAX_REPLACEMENTS = 2
# Loop guard for those uncharged redraws: the Nth prep rejection of one role in one tournament marks the row failed, halts
# the tournament and alerts (20 in a row is a bug).
SEED_MAX_PREP_REJECTIONS = 20
# Task-level model-prep failures of a seed before it is given up and redrawn as a prep rejection (prep otherwise retries forever).
SEED_MAX_PREP_ATTEMPTS = 3
# The continuous-chunk cursor is advanced once per tournament, keyed on f"{tournament_id}{SEED_CURSOR_KEY_SUFFIX}": the chunk
# is fixed at the first draw of the continuous instruct task or its compressions and every later draw (a retry, a
# replacement) reuses it.
SEED_CURSOR_KEY_SUFFIX = "_seed"

# Rounds 1-3 of a seeded text tournament are one instruct task each, in one group: the task is trained by the boss at
# tournament start and by the round's miners when the round is reached. The standings of a round come from that task
# alone, the boss excluded.
TEXT_ROUND_PLAN = (
    TextRoundSpec(round_number=1, seed_role=SeedRole.TEXT_R1, model_size_b=SEED_INPUT_SMALL_SIZE_B, advance=8),
    TextRoundSpec(round_number=2, seed_role=SeedRole.TEXT_R2, model_size_b=SEED_INPUT_SMALL_SIZE_B, advance=4),
    TextRoundSpec(round_number=3, seed_role=SeedRole.TEXT_R3, model_size_b=SEED_INPUT_LARGE_SIZE_B, advance=1),
)

# --- Boss round of a seeded text tournament ---------------------------------------------------------------------------
# Four tasks: the continuous-SFT seed (attached), one GRPO and one DPO task (created with the round), then the big task
# (an instruct task on a 30-72B model, drawn from the BOSS_ROUND_LARGE_INSTRUCT_* pool) which is created only when the
# challenger can still be crowned after those three. FINAL_ROUND_TEXT_TASK_DISTRIBUTION and FINAL_ROUND_TEXT_TASKS above
# belong to the boss round of a text tournament that has no seeds and are left as they are.
TEXT_BOSS_ROUND_STAGE_ONE_TASK_TYPES = (TaskType.GRPOTASK, TaskType.DPOTASK)

# Relative slack on the "at least" and "at most" crown boundaries, so that scores exactly on a decimal boundary count as
# meeting it despite floating-point rounding (a pair exactly 1% apart can compute as 0.99999999999999%).
CROWN_BOUNDARY_RELATIVE_TOLERANCE = 1e-9

# Head-to-heads of the crown rule: the tasks of rounds 1-3 plus the four boss-round tasks.
TEXT_CROWN_TASKS = 7

# A task is a win when the challenger beats the boss by at least this fraction of the boss's score magnitude (see
# challenger_beats_boss). Any other task is a non-win whose deficit is how far the challenger trails the boss, as a
# fraction of the same magnitude (0 when it is level, or ahead by less than this margin).
TEXT_CROWN_WIN_MARGIN = 0.01
# The challenger is crowned when it has at most this many non-wins, no deficit above TEXT_CROWN_MAX_DEFICIT, and, with
# exactly two non-wins, at least one deficit at most TEXT_CROWN_SECOND_NON_WIN_MAX_DEFICIT.
TEXT_CROWN_MAX_NON_WINS = 2
TEXT_CROWN_MAX_DEFICIT = 0.0025
TEXT_CROWN_SECOND_NON_WIN_MAX_DEFICIT = 0.001
