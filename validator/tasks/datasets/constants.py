from core.models.quant_models import QuantWorkload
from core.models.task_models import TaskType


TRAIN_TEST_SPLIT_PERCENTAGE = 0.1
MAX_TEST_DATA_POINTS = 1000

# Calibration set carved from the train side of a freeform quant task's split (see calibration.py).
QUANT_CALIBRATION_MIN_ROWS = 250
QUANT_CALIBRATION_TARGET_TOKENS = 50_000  # supervised (completion) tokens
QUANT_CALIBRATION_MAX_ROWS = 1_000
QUANT_CALIBRATION_MAX_TRAIN_FRACTION = 0.10  # under MIN_ROWS of this share of train: carve nothing
QUANT_CALIBRATION_CHARS_PER_TOKEN = 4  # sizes the carve only; the tokenised count is logged at quant prep

QUANT_CONTINUOUS_CARVE_CALIBRATION: bool = True
"""Whether the two continuous-chunk compression tasks (freeform and fixed) carve a calibration set from the given chunk.

When on, both tasks (fixed and freeform, same chunk) carve the usual QUANT_CALIBRATION_* rows
(at least QUANT_CALIBRATION_MIN_ROWS and ~QUANT_CALIBRATION_TARGET_TOKENS supervised tokens, capped at
QUANT_CALIBRATION_MAX_ROWS and QUANT_CALIBRATION_MAX_TRAIN_FRACTION of the chunk's train rows) from the given train
file, seeded by the chunk index so the two sibling tasks carve identical rows, and re-upload the train file without
them. About 1,000 of the chunk's ~95,000 train rows leave the train side: negligible for the fine-tune.
"""

IMAGE_TRAIN_SPLIT_ZIP_NAME = "train_data.zip"
IMAGE_TEST_SPLIT_ZIP_NAME = "test_data.zip"
TEMP_PATH_FOR_IMAGES = "/tmp/validator/temp_images"
SUPPORTED_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_FILE_SIZE_BYTES = 2_147_483_646  # pyarrow max json load size
MAX_ARCHIVE_ENTRIES = 10_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 10 * 1024 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 100
MINIMUM_DATASET_ROWS = 8_000  # Minimum number of rows required in a dataset
MAXIMUM_DATASET_ROWS = 175_000  # Above this, 2 epochs cannot fit the training-hours cap

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

# we sample datasets with these num_rows ranges equally
DATASET_BINS_TO_SAMPLE = [
    (MINIMUM_DATASET_ROWS, 40_000),
    (40_000, 90_000),
    (90_000, MAXIMUM_DATASET_ROWS),
]

# Training hours: throughput-based budget targeting TARGET_TRAINING_EPOCHS.
# One quarter-hour step: the size-band overhead below is the real floor for small tasks.
TRAINING_HOURS_MIN = 0.25
MAX_TRAINING_HOURS = 5.0
# Cap for quant-tournament instruct inputs (INPUT_SMALL, INPUT_LARGE, BOSS_INSTRUCT) whatever the epoch budget says.
NON_BOSS_INSTRUCT_MAX_TRAINING_HOURS = 2.5
TARGET_TRAINING_EPOCHS = 2.5
H100_BF16_TFLOPS = 989.0
ASSUMED_TRAINING_MFU = 0.15
ASSUMED_TOKENS_PER_ROW = 400
EFFECTIVE_MIN_TOKENS_PER_ROW = 64
DEFAULT_MODEL_PARAMS_FOR_HOURS = 8e9
# Fixed overhead (setup, model load, final save/upload) added on top of the token budget, by model
# size band: (exclusive upper bound in B params, hours). Small models load and save in minutes.
TRAINING_OVERHEAD_HOURS_BY_PARAMS_B: list[tuple[float, float]] = [
    (0.5, 0.4),
    (1.0, 0.6),
    (float("inf"), 0.75),
]
MEASURED_THROUGHPUT_MINER_RATIO = 1.0
MEASURED_THROUGHPUT_CLAMP = (0.33, 3.0)
# Quant hours come from prep's measured throughput (see scheduler.quant_hours_breakdown); the flat values below
# are the no-stats fallback used at creation and whenever baseline stats are unusable.
# Freeform flat fallback, the same in every round: the boss round gets no extra quant time.
QUANT_FREEFORM_EARLY_HOURS = 2.0
# Fixed hours by model size: linear between the (params_b, hours) anchors, flat below the first and above the last.
QUANT_FIXED_HOURS_BY_PARAMS_B: tuple[tuple[float, float], tuple[float, float]] = ((4.0, 1.25), (8.0, 2.0))
# Cost of a forward-only token relative to the probe's fwd+bwd token.
QUANT_FORWARD_TOKEN_FACTOR = 0.25
# Cost of a budgeted training token relative to the probe's fwd+bwd token.
QUANT_TRAIN_TOKEN_FACTOR = 3.0
# Floor on quant hours.
QUANT_TRAINING_HOURS_MIN = 0.5
# Boss multiplier on the raw hours: 1.0 for both tracks (the boss round gets no extra quant time).
QUANT_BOSS_HOURS_MULTIPLIER: dict[TaskType, float] = {TaskType.QUANTFREEFORMTASK: 1.0, TaskType.QUANTFIXEDTASK: 1.0}
# Rows per budgeted evaluation pass.
QUANT_EVAL_ROWS_PER_PASS = 768
# Cap on train_tokens, in epochs of the train split.
QUANT_TRAIN_MAX_EPOCHS = 1.0
# Headroom on the summed seconds.
QUANT_HOURS_SLACK = 1.1
# Freeform budget terms. The decoded-shard upload runs after the container exits and is not inside the budget.
QUANT_FREEFORM_WORKLOAD = QuantWorkload(
    eval_passes=32,
    forward_tokens=1_048_576,
    train_tokens=8_000_000,
    train_max_epochs=QUANT_TRAIN_MAX_EPOCHS,
    model_seconds_per_b_params=500.0,
    host_seconds_per_b_params=40.0,
    host_fixed_seconds=760.0,
    hours_slack=QUANT_HOURS_SLACK,
)
# A larger budget.
QUANT_FREEFORM_EXTENDED_WORKLOAD = QUANT_FREEFORM_WORKLOAD.model_copy(
    update={"eval_passes": 40, "train_max_epochs": 2.0, "host_fixed_seconds": 1_400.0}
)
# Fixed budget terms.
QUANT_FIXED_WORKLOAD = QuantWorkload(
    eval_passes=20,
    forward_tokens=1_048_576,
    train_tokens=7_000_000,
    train_max_epochs=None,
    model_seconds_per_b_params=80.0,
    host_seconds_per_b_params=60.0,
    host_fixed_seconds=600.0,
    hours_slack=1.0,
)
# Per-token FLOPs multiplier. GRPO is excluded because it is step-budgeted,
# not token-proportional. Quant tasks are priced by their own workloads above and have no multiplier.
TASK_TYPE_HOURS_MULTIPLIER: dict[TaskType, float] = {
    TaskType.INSTRUCTTEXTTASK: 1.0,
    TaskType.CHATTASK: 1.0,
    TaskType.DPOTASK: 1.4,
}
GRPO_HOURS_BY_PARAMS_B: list[tuple[float, float]] = [
    (4.0, 1.5),
    (12.0, 2.5),
    (40.0, 4.0),
    (float("inf"), 6.0),
]
GRPO_MIN_SYNTH_ROWS = 20_000

STANDARD_INSTRUCT_COLUMN = "instruct"
STANDARD_INPUT_COLUMN = "input"
STANDARD_OUTPUT_COLUMN = "output"
STANDARD_SYSTEM_COLUMN = "system"
STANDARD_GRPO_PROMPT_COLUMN = "prompt"
STANDARD_GRPO_EXTRA_COLUMN = "extra_data"
STANDARD_DPO_PROMPT_COLUMN = "prompt"
STANDARD_DPO_CHOSEN_COLUMN = "chosen"
STANDARD_DPO_REJECTED_COLUMN = "rejected"
STANDARD_CHAT_MESSAGES_COLUMN = "conversations"
