from enum import Enum


FLUX_T5_ARTIFACT = ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors")
FLUX_CLIP_ARTIFACT = ("comfyanonymous/flux_text_encoders", "clip_l.safetensors")
FLUX_TEXT_ENCODER_ARTIFACTS = (FLUX_T5_ARTIFACT, FLUX_CLIP_ARTIFACT)
FLUX_VAE_ARTIFACT = ("rayonlabs/FLUX.1-dev", "ae.safetensors")


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


def supports_image_training_mode(model_type: ImageModelType, mode: ImageTrainingMode) -> bool:
    return mode in IMAGE_MODEL_CAPABILITIES.get(model_type, frozenset())
