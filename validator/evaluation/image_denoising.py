"""Direct shared-case prediction evaluation; no image-generation server."""

import torch

from validator.evaluation.denoising_mse import flow_prediction_mse


class FlowPredictionSampler:
    """Run cached cases inside Comfy's model-loading and conditioning lifecycle."""

    def __init__(self, cases, noise_batch_size=4, loss_function=flow_prediction_mse):
        if noise_batch_size < 1:
            raise ValueError("Noise batch size must be positive")
        self.cases = cases
        self.noise_batch_size = noise_batch_size
        self.rows = []
        self.loss_function = loss_function

    @torch.inference_mode()
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        device = noise.device
        for case in self.cases:
            for start in range(0, len(case["seeds"]), self.noise_batch_size):
                stop = start + self.noise_batch_size
                noisy = case["noisy"][start:stop].to(device)
                target = case["target"][start:stop].to(device)
                timestep = torch.full((noisy.shape[0],), case["sigma"], device=device, dtype=torch.float32)
                denoised = model_wrap(noisy, timestep, **extra_args)
                losses = self.loss_function(noisy, denoised, target, case["sigma"]).cpu().tolist()
                for seed, loss in zip(case["seeds"][start:stop], losses, strict=True):
                    self.rows.append({"stratum": case["stratum"], "sigma": case["sigma"], "seed": seed, "mse": loss})
        return latent_image

