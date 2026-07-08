from core.models.task_models import TaskType


TRAIN_TEST_SPLIT_PERCENTAGE = 0.1
MAX_TEST_DATA_POINTS = 1000

IMAGE_TRAIN_SPLIT_ZIP_NAME = "train_data.zip"
IMAGE_TEST_SPLIT_ZIP_NAME = "test_data.zip"
TEMP_PATH_FOR_IMAGES = "/tmp/validator/temp_images"
SUPPORTED_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_FILE_SIZE_BYTES = 2_147_483_646  # pyarrow max json load size
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
TRAINING_HOURS_MIN = 0.75
MAX_TRAINING_HOURS = 6.0
TARGET_TRAINING_EPOCHS = 2.0
H100_BF16_TFLOPS = 989.0
ASSUMED_TRAINING_MFU = 0.15
ASSUMED_TOKENS_PER_ROW = 400
EFFECTIVE_MIN_TOKENS_PER_ROW = 64
DEFAULT_MODEL_PARAMS_FOR_HOURS = 8e9
TRAINING_OVERHEAD_HOURS = 0.75
MEASURED_THROUGHPUT_MINER_RATIO = 1.0
MEASURED_THROUGHPUT_CLAMP = (0.33, 3.0)
# Per-token FLOPs multiplier. GRPO is excluded because it is step-budgeted,
# not token-proportional.
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
