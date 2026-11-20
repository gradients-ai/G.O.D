# Quant-task artifact and decoder-sandbox limits shared by the validator and trainer.

# Directory inside a submitted artifact repo that holds the miner's decoder.
DECODER_DIR_NAME = "decoder"
# Decoder size cap, measured after xz compression.
DECODER_MAX_XZ_BYTES = 1_048_576

# Files over this size in a text entrant's build context are flagged for manual review, never auto-rejected: the rules
# ban weights-derived payloads over about 1 MB (for example precomputed model weights or deltas). Logged at registration
# (tracked files at the pinned commit) and at every image build (the trainer's checkout, LFS objects included).
QUANT_LARGE_FILE_REVIEW_BYTES = 1_000_000

# Decoder sandbox limits.
DECODE_TIMEOUT_SECONDS = 900
DECODE_CPUS = 16
DECODE_MEMORY_MULTIPLIER = 2.5  # memory limit = multiplier x decoded bf16 model size
DECODE_PIDS_LIMIT = 256

# What the trainer's freeform decode requires of a decoder snapshot (trainer/quant_decode/job.py): the entrypoint file
# and the cap on the snapshot's extracted size (a zip-bomb guard). Registration checks both
# (validator/tournament/decoder_snapshot.py), so a decoder that would scored-fail every freeform task is turned away.
DECODER_ENTRYPOINT = "decode.py"
DECODER_MAX_EXTRACTED_BYTES = 64 * 1024 * 1024

# safetensors dtype string every decoded tensor must carry.
DECODED_TENSOR_DTYPE = "BF16"

# Files inspected on Hugging Face: a full model has config.json; a LoRA adapter repo has adapter_config.json.
QUANT_MODEL_CONFIG_FILENAME = "config.json"
QUANT_ADAPTER_CONFIG_FILENAME = "adapter_config.json"

# Size band in B params for every quant task's base model; validator/tasks/quant_shapes.py checks the rest (dense, shapes).
QUANT_MODEL_SIZE_B = (1.0, 8.0)

# Miner-written completion marker at the root of a quant output dir; the trainer uploads exactly the files it lists.
QUANT_MANIFEST_NAME = "quant_manifest.json"
QUANT_MANIFEST_MAX_FILES = 2048
# Upload cap = multiplier x the base model's bf16 bytes: an uncompressed bf16 fine-tune still uploads (a pass that never wins).
QUANT_ARTIFACT_MAX_BYTES_MULTIPLIER = 1.25
