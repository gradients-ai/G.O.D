"""Production image-family encoders and strict flow/LoRA contracts."""

import logging
from dataclasses import dataclass

import numpy as np
import torch

from validator.evaluation.image_artifacts import materialize_model


@dataclass(frozen=True)
class FlowFamily:
    clip_type: str
    encoders: tuple
    vae: tuple
    shift: float | None = None
    guidance: float | None = None


FAMILIES = {
    "z-image": FlowFamily(
        "lumina2",
        (("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors"),),
        ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
    ),
    "flux": FlowFamily(
        "flux",
        tuple(("comfyanonymous/flux_text_encoders", f) for f in ("t5xxl_fp16.safetensors", "clip_l.safetensors")),
        ("rayonlabs/FLUX.1-dev", "ae.safetensors"),
        guidance=3.5,
    ),
    "qwen-image": FlowFamily(
        "qwen_image",
        (("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),),
        ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
        shift=3.0,
    ),
    "krea2": FlowFamily(
        "krea2",
        (("Comfy-Org/Krea-2", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"),),
        ("Comfy-Org/Krea-2", "vae/qwen_image_vae.safetensors"),
    ),
    "ideogram4": FlowFamily(
        "ideogram4",
        (("Comfy-Org/Ideogram-4", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors"),),
        ("Comfy-Org/Ideogram-4", "vae/flux2-vae.safetensors"),
    ),
}


class ComfyImageEncoder:
    """Use native VAE shapes and posterior means, including single-frame video VAEs."""

    def __init__(self, vae):
        self.vae = vae
        self.vae.first_stage_model.eval().requires_grad_(False)
        regularizer = getattr(self.vae.first_stage_model, "regularization", None)
        if regularizer is not None and getattr(regularizer, "sample", False):
            raise ValueError("Evaluation VAE must encode posterior means")

    @torch.inference_mode()
    def encode(self, image):
        pixels = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)
        latent = self.vae.encode(pixels).float().cpu()
        if not torch.isfinite(latent).all():
            raise ValueError("Non-finite evaluation latent")
        return latent


class ImageFlowAdapter:
    def __init__(self, family, api, root):
        if family not in FAMILIES:
            raise ValueError(f"No validated flow contract for {family}")
        self.family = family
        self.spec = FAMILIES[family]
        self.encoders = []
        for repo, filename in self.spec.encoders:
            name, _ = materialize_model(api, repo, filename, root / "models/text_encoders")
            self.encoders.append(name)
        self.vae_name, _ = materialize_model(api, *self.spec.vae, root / "models/vae")

    def load_clip(self):
        import nodes

        if len(self.encoders) == 2:
            return nodes.DualCLIPLoader().load_clip(*self.encoders, self.spec.clip_type)[0]
        return nodes.CLIPLoader().load_clip(self.encoders[0], self.spec.clip_type)[0]

    def load_vae(self):
        import comfy.sd
        import comfy.utils
        import folder_paths

        path = folder_paths.get_full_path_or_raise("vae", self.vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path), dtype=torch.float32)
        return ComfyImageEncoder(vae)

    def conditioning(self, clip, caption):
        import node_helpers
        import nodes

        cond = nodes.CLIPTextEncode().encode(clip, caption)[0]
        if self.spec.guidance is not None:
            cond = node_helpers.conditioning_set_values(cond, {"guidance": self.spec.guidance})
        return cond

    def configure(self, model):
        import comfy.model_sampling

        if self.spec.shift is not None:
            from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

            model = ModelSamplingAuraFlow().patch_aura(model, self.spec.shift)[0]
        sampling = model.model.model_sampling
        if not isinstance(sampling, comfy.model_sampling.CONST) or getattr(sampling, "noise_scale", 1.0) != 1.0:
            raise ValueError("Model is not unit-noise rectified flow; refusing incorrect targets")
        return model

    def apply_lora(self, model, clip, name):
        import nodes

        missing = []

        class CaptureMissing(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                if message.startswith(("lora key not loaded:", "NOT LOADED ")):
                    missing.append(message)

        handler = CaptureMissing()
        logger = logging.getLogger()
        logger.addHandler(handler)
        try:
            patched, patched_clip = nodes.LoraLoader().load_lora(model, clip, name, 1.0, 1.0)
        finally:
            logger.removeHandler(handler)
        if missing or not patched.patches:
            raise ValueError(
                f"Incomplete LoRA application: {len(missing)} unmatched entries; {len(patched.patches)} model patches"
            )
        return patched, patched_clip
