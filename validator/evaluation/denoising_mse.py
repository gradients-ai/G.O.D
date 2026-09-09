"""Shared flow-prediction cases and L2 error.

The initial adapter is explicitly restricted to unit-noise rectified flow.
Other prediction parameterizations must supply their own targets and conversion.
"""

import hashlib

import torch


def case_seed(image_hash: str, stratum: int, noise_index: int, master_seed: int = 42) -> int:
    key = f"{master_seed}:{image_hash}:{stratum}:{noise_index}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


def flow_cases(latent, image_hash: str, strata: int = 16, noises: int = 4):
    if strata < 1 or noises < 1:
        raise ValueError("Strata and noise counts must be positive")
    if latent.shape[0] != 1 or latent.device.type != "cpu":
        raise ValueError("Cache one CPU latent per image")
    cases = []
    for band in range(strata):
        # Midpoints exclude unstable endpoints while covering uniform flow time.
        sigma = (band + 0.5) / strata
        seeds = [case_seed(image_hash, band, sample) for sample in range(noises)]
        noise = torch.cat(
            [torch.randn(latent.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32) for seed in seeds]
        )
        cases.append(
            {
                "stratum": band,
                "sigma": sigma,
                "seeds": seeds,
                "noisy": (1 - sigma) * latent + sigma * noise,
                "target": noise - latent,
            }
        )
    return cases


def flow_prediction_mse(noisy, denoised, target, sigma):
    if not 0 < sigma < 1:
        raise ValueError("Flow sigma must lie strictly between zero and one")
    if noisy.shape != denoised.shape or target.shape != noisy.shape:
        raise ValueError("Prediction and target dimensions differ")
    # Comfy returns x0 = x_t - sigma * velocity, so recover the velocity target.
    predicted = (noisy.float() - denoised.float()) / sigma
    losses = (predicted - target.float()).square().flatten(1).mean(1)
    if not torch.isfinite(losses).all():
        raise ValueError("Non-finite denoising MSE")
    return losses
