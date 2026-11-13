from enum import Enum
from pathlib import PurePosixPath


FLUX_T5_ARTIFACT = ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors")
FLUX_CLIP_ARTIFACT = ("comfyanonymous/flux_text_encoders", "clip_l.safetensors")
FLUX_TEXT_ENCODER_ARTIFACTS = (FLUX_T5_ARTIFACT, FLUX_CLIP_ARTIFACT)
FLUX_VAE_ARTIFACT = ("rayonlabs/FLUX.1-dev", "ae.safetensors")

# Standalone FLUX repos sometimes ship several root checkpoints (bf16 + fp8 + nf4).
# Evaluation and the trainer downloader must pick the same full-precision file.
_FLUX_FULL_PRECISION_NAME_TOKENS = ("bf16", "bfloat16", "fp16", "float16")
_FLUX_QUANTIZED_NAME_TOKENS = ("fp8", "nf4", "int4", "int8", "gguf", "bnb", "quanto")


class ImageModelType(str, Enum):
    FLUX = "flux"
    SDXL = "sdxl"
    Z_IMAGE = "z-image"
    QWEN_IMAGE = "qwen-image"
    IDEOGRAM4 = "ideogram4"
    KREA2 = "krea2"
    FLUX2_KLEIN_4B = "flux2-klein-4b"
    QWEN_IMAGE_EDIT_2511 = "qwen-image-edit-2511"
    HIDREAM_O1 = "hidream-o1"
    MOSS_IMAGE_BASE = "moss-image-base"


class ImageTrainingMode(str, Enum):
    TEXT_TO_IMAGE = "text-to-image"
    EDIT = "edit"


IMAGE_MODEL_CAPABILITIES = {
    ImageModelType.FLUX: frozenset({ImageTrainingMode.TEXT_TO_IMAGE}),
    ImageModelType.SDXL: frozenset({ImageTrainingMode.TEXT_TO_IMAGE}),
    ImageModelType.Z_IMAGE: frozenset({ImageTrainingMode.TEXT_TO_IMAGE}),
    ImageModelType.QWEN_IMAGE: frozenset({ImageTrainingMode.TEXT_TO_IMAGE}),
    ImageModelType.IDEOGRAM4: frozenset({ImageTrainingMode.TEXT_TO_IMAGE}),
    ImageModelType.KREA2: frozenset({ImageTrainingMode.TEXT_TO_IMAGE}),
    ImageModelType.FLUX2_KLEIN_4B: frozenset({ImageTrainingMode.TEXT_TO_IMAGE, ImageTrainingMode.EDIT}),
    ImageModelType.QWEN_IMAGE_EDIT_2511: frozenset({ImageTrainingMode.EDIT}),
    ImageModelType.HIDREAM_O1: frozenset({ImageTrainingMode.TEXT_TO_IMAGE, ImageTrainingMode.EDIT}),
    ImageModelType.MOSS_IMAGE_BASE: frozenset({ImageTrainingMode.TEXT_TO_IMAGE, ImageTrainingMode.EDIT}),
}

# Hub repos AI Toolkit loads by id besides the base model, as (repo_id, allow_patterns).
# Trainers have no internet, so the downloader caches these into the shared hub cache.
AI_TOOLKIT_RUNTIME_REPOS = {
    ImageModelType.FLUX2_KLEIN_4B: (
        ("Qwen/Qwen3-4B", None),
        ("ai-toolkit/flux2_vae", ("ae.safetensors",)),
    ),
    ImageModelType.MOSS_IMAGE_BASE: (("Qwen/Qwen3.5-9B", ("*.json", "*.txt", "*.jinja", "*.model")),),
}
# Comfy-layout files AI Toolkit resolves under its MODELS_PATH, as (repo_id, filename).
AI_TOOLKIT_COMFY_FILES = {
    ImageModelType.MOSS_IMAGE_BASE: (("Comfy-Org/MOSS-Image", "text_encoders/qwen3.5_9b_int8_convrot.safetensors"),),
}


def supports_image_training_mode(model_type: ImageModelType, mode: ImageTrainingMode) -> bool:
    return mode in IMAGE_MODEL_CAPABILITIES.get(model_type, frozenset())


def _flux_checkpoint_precision_rank(filename: str) -> int:
    """0 = named bf16/fp16, 1 = unmarked, 2 = named quantized."""
    stem = PurePosixPath(filename).name.lower()
    if any(token in stem for token in _FLUX_FULL_PRECISION_NAME_TOKENS):
        return 0
    if any(token in stem for token in _FLUX_QUANTIZED_NAME_TOKENS):
        return 2
    return 1


def select_standalone_flux_checkpoint(
    candidates: list[tuple[str, int | None]],
) -> str:
    """Pick one root FLUX `.safetensors` when a repo ships several precision variants.

    Prefers filenames that name bf16/fp16, then unmarked files, and never chooses a
    quantized sibling when a fuller-precision option exists. Size breaks remaining ties
    (larger first), then the filename.
    """
    if not candidates:
        raise ValueError("Base model requires a root-level ComfyUI checkpoint")
    if len(candidates) == 1:
        return candidates[0][0]

    best_rank = min(_flux_checkpoint_precision_rank(name) for name, _ in candidates)
    if best_rank == 2:
        names = ", ".join(sorted(name for name, _ in candidates))
        raise ValueError(
            "Base model has multiple root-level checkpoints but none look like bf16/fp16: "
            f"{names}"
        )

    preferred = [(name, size) for name, size in candidates if _flux_checkpoint_precision_rank(name) == best_rank]
    preferred.sort(key=lambda item: (-(item[1] or 0), item[0]))
    return preferred[0][0]
