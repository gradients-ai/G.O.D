from enum import Enum


class ImageModelType(str, Enum):
    FLUX = "flux"
    SDXL = "sdxl"
    Z_IMAGE = "z-image"
    QWEN_IMAGE = "qwen-image"
