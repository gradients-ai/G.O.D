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
