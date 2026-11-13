from core.models.pvp_models import PvPStatus


# Connection Pool Constants
MIN_POOL_SIZE = 10  # Minimum number of connections to stay open
MAX_POOL_SIZE = 90  # Maximum number of connections to reach if needed
COMMAND_TIMEOUT = 20.0  # If sql query takes longer than this, raise an error
TIMEOUT = 10.0  # If no connection is available after this time, raise an error
MAX_QUERIES = 1000  # Maximum number of queries to execute before closing a connection in the pool ( and opening a new one)

# Tables
NODES_TABLE = "nodes"
NODES_HISTORY_TABLE = "nodes_history"
TASKS_TABLE = "tasks"
INSTRUCT_TEXT_TASKS_TABLE = "instruct_text_tasks"
CHAT_TASKS_TABLE = "chat_tasks"
IMAGE_TASKS_TABLE = "image_tasks"
DPO_TASKS_TABLE = "dpo_tasks"
TASK_NODES_TABLE = "task_nodes"
EVALUATIONS_TABLE = "evaluations"
SUBMISSIONS_TABLE = "submissions"
OFFER_RESPONSES_TABLE = "offer_responses"
LATEST_SCORES_URL_TABLE = "latest_scores_url"
IMAGE_TEXT_PAIRS_TABLE = "image_text_pairs"
GRPO_TASKS_TABLE = "grpo_tasks"
ENV_TASKS_TABLE = "env_tasks"
QUANT_TASKS_TABLE = "quant_tasks"
QUANT_SKU_REFERENCES_TABLE = "quant_sku_references"
QUANT_CALIBRATION_REFERENCES_TABLE = "quant_calibration_references"
DATASET_CALIBRATION_ROWS_TABLE = "dataset_calibration_rows"
REWARD_FUNCTIONS_TABLE = "reward_functions"
GRPO_TASK_FUNCTIONS_TABLE = "grpo_task_functions"


# Tournament Tables
TOURNAMENTS_TABLE = "tournaments"
TOURNAMENT_ROUNDS_TABLE = "tournament_rounds"
TOURNAMENT_PARTICIPANTS_TABLE = "tournament_participants"
TOURNAMENT_GROUPS_TABLE = "tournament_groups"
TOURNAMENT_GROUP_MEMBERS_TABLE = "tournament_group_members"
TOURNAMENT_PAIRS_TABLE = "tournament_pairs"
TOURNAMENT_TASKS_TABLE = "tournament_tasks"
BENCHMARK_ROOT_TASKS_TABLE = "benchmark_root_tasks"
BENCHMARK_TASK_COPIES_TABLE = "benchmark_task_copies"
TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE = "tournament_task_hotkey_trainings"
PVP_PAIR_RESULTS_TABLE = "pvp_pair_results"
TOURNAMENT_DEDUP_REVIEWS_TABLE = "tournament_dedup_reviews"
TOURNAMENT_TASK_FAILURE_REVIEWS_TABLE = "tournament_task_failure_reviews"
PVP_INDIVIDUAL_SCORES_TABLE = "pvp_individual_scores"

# Continuous-SFT lineage state (one row per lineage slug): the monotonic train_index cursor passed
# to the content service + the carried-forward winner repo. Chunk data itself lives in the service.
CONTINUOUS_SFT_STATE_TABLE = "continuous_sft_state"
CONTINUOUS_SFT_LINEAGE = "lineage"
CONTINUOUS_SFT_TRAIN_INDEX = "train_index"
CONTINUOUS_SFT_LAST_WINNER_REPO = "last_winner_repo"
CONTINUOUS_SFT_LAST_SOURCE_ROUND_ID = "last_source_round_id"
CONTINUOUS_SFT_UPDATED_AT = "updated_at"
CONTINUOUS_SFT_CHUNK_SCHEME = "chunk_scheme"

# One row per continuous-SFT task: the cursor value and chunk scheme the task was created with.
CONTINUOUS_SFT_TASK_CHUNKS_TABLE = "continuous_sft_task_chunks"

# PvP Pair Results Table Columns
PVP_HOTKEY_A = "hotkey_a"
PVP_HOTKEY_B = "hotkey_b"
PVP_ENVIRONMENT_NAME = "environment_name"
PVP_MODEL_A_WINS = "model_a_wins"
PVP_MODEL_B_WINS = "model_b_wins"
PVP_DRAWS = "draws"
PVP_TOTAL_GAMES = "total_games"
PVP_N_ATTEMPTS = "n_attempts"
PVP_DEPLOYMENT_ID = "deployment_id"
PVP_DEPLOYMENT_VERIFIED = "deployment_verified"
PVP_STATUS_PENDING = PvPStatus.PENDING
PVP_STATUS_COMPLETE = PvPStatus.COMPLETE

# PvP Individual Scores Table Columns
PVP_INDIVIDUAL_SCORE = "score"

# Tournament Task Hotkey Trainings Table Columns
PRIORITY = "priority"

# Benchmark Task Copies Table Columns
COPY_TASK_ID = "copy_task_id"
ROOT_TASK_ID = "root_task_id"
PARTICIPANT_HOTKEY = "participant_hotkey"
TOURNAMENT_ID = "tournament_id"

# Node Table Columns
NODE_ID = "node_id"
HOTKEY = "hotkey"
COLDKEY = "coldkey"
IP = "ip"
IP_TYPE = "ip_type"
PORT = "port"
NETUID = "netuid"
ALPHA_STAKE = "alpha_stake"
TAO_STAKE = "tao_stake"
STAKE = "stake"
TRUST = "trust"
VTRUST = "vtrust"
INCENTIVE = "incentive"
LAST_UPDATED = "last_updated"
PROTOCOL = "protocol"
CREATED_TIMESTAMP = "created_timestamp"
ASSIGNED_MINERS = "assigned_miners"

# Task Table Columns
TASK_ID = "task_id"
ACCOUNT_ID = "account_id"
MODEL_ID = "model_id"
DS = "ds"
STATUS = "status"
HOURS_TO_COMPLETE = "hours_to_complete"
TEST_DATA = "test_data"
CALIBRATION_DATA = "calibration_data"
CALIBRATION_FINGERPRINT = "calibration_fingerprint"
CALIBRATION_N_EXAMPLES = "calibration_n_examples"
CARVE_CALIBRATION = "carve_calibration"
TRAINING_DATA = "training_data"
CREATED_AT = "created_at"
NEXT_DELAY_AT = "next_delay_at"
UPDATED_AT = "updated_at"
STARTED_AT = "started_at"
COMPLETED_AT = "completed_at"
TERMINATION_AT = "termination_at"
IS_ORGANIC = "is_organic"
ASSIGNED_MINERS = "assigned_miners"
TASK_TYPE = "task_type"
TRAINING_REPO_BACKUP = "training_repo_backup"
RESULT_MODEL_NAME = "result_model_name"
MODEL_PARAMS_COUNT = "model_params_count"
BACKEND = "backend"
YARN_FACTOR = "yarn_factor"
AUGMENTATION_CONFIG = "augmentation_config"
AUGMENTED_MODEL_ID = "augmented_model_id"
BASELINE_STATS = "baseline_stats"

# Instruct Text Tasks Table Columns
FIELD_SYSTEM = "field_system"
FIELD_INSTRUCTION = "field_instruction"
FIELD_INPUT = "field_input"
FIELD_OUTPUT = "field_output"
FORMAT = "format"
NO_INPUT_FORMAT = "no_input_format"
FILE_FORMAT = "file_format"
USE_KL = "use_kl"
KL_COEF = "kl_coef"

# Image Text Pairs Table Columns
IMAGE_URL = "image_url"
TEXT_URL = "text_url"
SOURCE_IMAGE_URL = "source_image_url"
ID = "id"
MODEL_TYPE = "model_type"
TRAINING_MODE = "training_mode"
TRIGGER_WORD = "trigger_word"

# DPO Tasks Table Columns
FIELD_PROMPT = "field_prompt"
FIELD_CHOSEN = "field_chosen"
FIELD_REJECTED = "field_rejected"
PROMPT_FORMAT = "prompt_format"
CHOSEN_FORMAT = "chosen_format"
REJECTED_FORMAT = "rejected_format"
FIELD_EXTRA_COLUMN = "extra_column"

# Chat Tasks Table Columns
CHAT_TEMPLATE = "chat_template"
CHAT_COLUMN = "chat_column"
CHAT_ROLE_FIELD = "chat_role_field"
CHAT_CONTENT_FIELD = "chat_content_field"
CHAT_USER_REFERENCE = "chat_user_reference"
CHAT_ASSISTANT_REFERENCE = "chat_assistant_reference"

# Reward Functions Table Columns
REWARD_ID = "reward_id"
REWARD_FUNC = "reward_func"
FUNC_HASH = "func_hash"
IS_GENERIC = "is_generic"
IS_MANUAL = "is_manual"

# GRPO Task Functions Table Columns
REWARD_WEIGHT = "reward_weight"

# Environment Task Functions Table Columns
ENVIRONMENT_NAMES = "environment_names"
ENVIRONMENT_WEIGHTS = "environment_weights"
TRAINING_START_POINT = "training_start_point"
EVAL_SEED = "eval_seed"

# Quant Tasks Table Columns (file_format reuses FILE_FORMAT)
DATASET_TYPE = "dataset_type"
QUANT_SPEC = "quant_spec"
BAR_CAP = "bar_cap"  # freeform: the hard cap as the excess over the reference (the bar before the penalty)
PENALTY_KNEE = "penalty_knee"  # freeform: unpenalised excess over the reference; NULL = cliff at bar_cap
PENALTY_CURVATURE = "penalty_curvature"  # freeform: ln P = penalty_curvature x max(0, r - knee)^2; NULL = cliff at bar_cap
UPSTREAM_TASK_ID = "upstream_task_id"
SYNTHETIC_DATA = "synthetic_data"

# Quant SKU References Table Columns (task_id, hotkey, per_example_losses, eval_set_fingerprint and eval_gpu
# reuse the shared names; hotkey '' is the task-common reference)
SKU = "sku"
REFERENCE_MODEL = "model"
LOSS = "loss"
MEASURED_AT = "measured_at"

# Dataset Calibration Rows Table Columns (ds and task_id reuse the shared names)
ROW_HASH = "row_hash"

# Submissions Table Columns
SUBMISSION_ID = "submission_id"
REPO = "repo"
CREATED_ON = "created_on"

# Task Nodes Table Columns
TASK_NODE_QUALITY_SCORE = "quality_score"

EXPECTED_REPO_NAME = "expected_repo_name"
STARTING_MODEL_REPO = "starting_model_repo"
EVALUATION_STATUS = "evaluation_status"
DEPLOYMENT_ID = "deployment_id"
GPU_COUNT = "gpu_count"
INFRA_ERROR_ATTEMPTS = "infra_error_attempts"

TEST_LOSS = "test_loss"
SYNTH_LOSS = "synth_loss"
PER_EXAMPLE_LOSSES = "per_example_losses"
EVAL_SET_FINGERPRINT = "eval_set_fingerprint"
SCORE_REASON = "score_reason"

# Task Nodes quant columns
ARTIFACT_BYTES = "artifact_bytes"
DECODER_BYTES = "decoder_bytes"
DECODER_SHA256 = "decoder_sha256"  # also the tournament_participants decoder snapshot sha
PASSED_BAR = "passed_bar"  # freeform: under the cap (the bar before the penalty)
FREEFORM_PENALTY = "freeform_penalty"  # freeform: ln P as scored; effective bytes = total bytes x exp(it)
QUANT_FAILURE = "quant_failure"
EVAL_GPU = "eval_gpu"  # QuantEvalGpu JSON: the GPU (and, for quant, the SKU) the eval ran on
DECODER_BUILD_WARNING = "decoder_build_warning"

# Tournament Participants decoder snapshot columns (DECODER_SHA256 above is shared)
DECODER_OBJECT_NAME = "decoder_object_name"
DECODER_XZ_BYTES = "decoder_xz_bytes"
DECODER_TAR_SHA256 = "decoder_tar_sha256"  # sha256 of the uncompressed canonical tar
DECODER_TREE_SHA = "decoder_tree_sha"  # git tree id of decoder/, reproducible from the released repo
DECODER_XZ_VERSION = "decoder_xz_version"
DECODER_RELEASED_URL = "decoder_released_url"  # public .tar.xz URL, set at champion handover
DECODER_RELEASED_AT = "decoder_released_at"

# Task Nodes freeform exposure columns
ARTIFACT_FILES = "artifact_files"  # JSONB: hub files [{name, bytes, sha256, sha256_kind}] read at scoring
ARTIFACT_REVISION = "artifact_revision"  # hub commit sha the files were listed at
DECODE_ARTIFACT_FILES = "decode_artifact_files"  # JSONB: files the trainer's decode downloaded and counted
DECODED_FILES = "decoded_files"  # JSONB: the decoded bf16 shards [{name, bytes, sha256}]
DECODED_HF_REPO = "decoded_hf_repo"  # public HF copy of the decoded model (boss-round submissions)

# Offer Responses Table Columns
OFFER_RESPONSE = "offer_response"


# Tournament Table Columns
TOURNAMENT_ID = "tournament_id"
TOURNAMENT_TYPE = "tournament_type"
TOURNAMENT_STATUS = "status"
BASE_WINNER_HOTKEY = "base_winner_hotkey"
WINNER_HOTKEY = "winner_hotkey"
CODE_REVIEW = "code_review"
DIFF_REPORT = "diff_report"
WINNER_MODEL_REPO = "winner_model_repo"
WINNER_MODEL_BASE = "winner_model_base"
ROUND_ID = "round_id"
ROUND_NUMBER = "round_number"
ROUND_TYPE = "round_type"
IS_FINAL_ROUND = "is_final_round"
ROUND_STATUS = "status"
STARTED_AT = "started_at"
COMPLETED_AT = "completed_at"
GROUP_ID = "group_id"
PAIR_ID = "pair_id"
HOTKEY1 = "hotkey1"
HOTKEY2 = "hotkey2"
ELIMINATED_IN_ROUND_ID = "eliminated_in_round_id"
FINAL_POSITION = "final_position"
TRAINING_STATUS = "training_status"
N_TRAINING_ATTEMPTS = "n_training_attempts"
TRAINING_REPO = "training_repo"
TRAINING_COMMIT_HASH = "training_commit_hash"
GITHUB_TOKEN = "github_token"
BACKUP_REPO = "backup_repo"
REQUESTED_DATASETS = "requested_datasets"
DSTACK_RUNNAME = "dstack_runname"

# Trainer GPUs Table
TRAINERS_GPUS_TABLE = "trainers_gpus"
TRAINER_IP = "trainer_ip"
GPU_ID = "gpu_id"
GPU_TYPE = "gpu_type"
VRAM_GB = "vram_gb"
USED_UNTIL = "used_until"
PRODUCT_NAME = "product_name"
INTERCONNECT = "interconnect"
NVLINK = "nvlink"
NVLINK_PEERS = "nvlink_peers"


# Common Column Names (shared between tables)
QUALITY_SCORE = "quality_score"  # Used in both submissions and task_nodes


# Tournament Seed Tasks Table (the boss's upfront tasks; see validator/db/sql/tournament_seeds.py)
TOURNAMENT_SEED_TASKS_TABLE = "tournament_seed_tasks"
SEED_ROLE = "role"
SEED_ATTEMPT = "attempt"
SEED_STATUS = "status"
SEED_BOSS_REPO = "boss_repo"
SEED_BOSS_COMMIT = "boss_commit"
SEED_ATTACHED_ROUND_ID = "attached_round_id"
SEED_PREP_REJECTED = "prep_rejected"
SEED_MERGED_OUTPUT_REPO = "merged_output_repo"


# Quant task sources (which text tournament and text seed task a quant tournament's tasks come from; see
# validator/db/sql/quant_sources.py)
QUANT_TOURNAMENT_SOURCES_TABLE = "quant_tournament_sources"
QUANT_TASK_SOURCES_TABLE = "quant_task_sources"
QUANT_TOURNAMENT_ID = "quant_tournament_id"
QUANT_TASK_ID = "quant_task_id"
QUANT_SOURCE_TOURNAMENT_ID = "source_tournament_id"
QUANT_SOURCE_SLOT = "slot"
QUANT_SOURCE_ROLE = "source_role"
QUANT_SOURCE_TASK_ID = "source_task_id"
QUANT_SOURCE_MODEL_REPO = "source_model_repo"
QUANT_SOURCE_IS_FALLBACK = "source_is_fallback"
QUANT_SOURCE_SUPERSEDED = "superseded"


# Boss rank comparisons: a paid contender of a tournament the boss kept, compared with the boss on average round rank
# (see validator/tournament/boss_rank_shares.py)
TOURNAMENT_BOSS_RANK_COMPARISONS_TABLE = "tournament_boss_rank_comparisons"
CONTENDER_AVERAGE_RANK = "contender_average_rank"
BOSS_AVERAGE_RANK = "boss_average_rank"
COMPARED_ROUND_NUMBERS = "compared_round_numbers"
SHARE_TO_BOSS = "share_to_boss"
