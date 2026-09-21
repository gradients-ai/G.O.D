"""Deterministic float32 VAE encoding for image prediction evaluation."""

import numpy as np
import torch


class DeterministicVAE:
    """Use a single fixed encoder and its posterior mean; never sample the posterior.

    Inputs are RGB in [-1, 1]. Returns unscaled encoder coordinates; the caller
    applies model-native scaling. The exact weights/config are recorded by the
    caller. Candidate LoRAs never modify this evaluation encoder.
    """

    def __init__(self, vae, device: str = "cuda"):
        self.device = device
        self.vae = vae.eval().requires_grad_(False).to(device=device, dtype=torch.float32)

    @torch.inference_mode()
    def encode(self, image):
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 127.5 - 1.0
        if array.shape[0] % 8 or array.shape[1] % 8:
            raise ValueError("Image dimensions must be divisible by eight")
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        latent = self.vae.encode(tensor).latent_dist.mode()
        if not torch.isfinite(latent).all():
            raise ValueError("Non-finite evaluation latent")
        return latent.float().cpu()
