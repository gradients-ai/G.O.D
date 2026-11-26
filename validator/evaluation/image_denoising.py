"""Direct shared-case prediction evaluation; no image-generation server."""

import torch

from validator.evaluation.denoising_mse import flow_prediction_mse


class FlowPredictionSampler:
    """Run cached cases inside Comfy's model-loading and conditioning lifecycle."""

    def __init__(self, cases, noise_batch_size=4, loss_function=flow_prediction_mse, recorder=None):
        if noise_batch_size < 1:
            raise ValueError("Noise batch size must be positive")
        self.cases = cases
        self.noise_batch_size = noise_batch_size
        self.rows = []
        self.loss_function = loss_function
        self.recorder = recorder

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
                parts = {}
                if self.recorder is not None:
                    uncond = self.recorder.take()
                    parts["mse_empty"] = self.loss_function(noisy, uncond, target, case["sigma"]).cpu().tolist()
                for index, (seed, loss) in enumerate(zip(case["seeds"][start:stop], losses, strict=True)):
                    row = {"stratum": case["stratum"], "sigma": case["sigma"], "seed": seed, "mse": loss}
                    row.update({key: values[index] for key, values in parts.items()})
                    self.rows.append(row)
        return latent_image



class CFGRecorder:
    """Comfy post-CFG hook that keeps the empty-prompt prediction behind each guided prediction."""

    def __init__(self):
        self._last = None

    def __call__(self, args):
        self._last = args["uncond_denoised"]
        return args["denoised"]

    def take(self):
        if self._last is None:
            raise ValueError("Guided prediction did not pass through the CFG hook")
        last, self._last = self._last, None
        return last
