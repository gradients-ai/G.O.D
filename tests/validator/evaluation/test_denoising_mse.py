import torch

from validator.evaluation.denoising_mse import case_seed
from validator.evaluation.denoising_mse import flow_cases


def test_cases_share_noise_under_one_task_seed():
    latent = torch.zeros(1, 4, 8, 8)
    first = flow_cases(latent, "imagehash", 123, strata=2, noises=2)
    again = flow_cases(latent, "imagehash", 123, strata=2, noises=2)
    for a, b in zip(first, again):
        assert torch.equal(a["noisy"], b["noisy"])


def test_task_seed_and_image_hash_both_change_the_noise():
    latent = torch.zeros(1, 4, 8, 8)
    base = flow_cases(latent, "imagehash", 123, strata=1, noises=1)[0]["noisy"]
    other_seed = flow_cases(latent, "imagehash", 124, strata=1, noises=1)[0]["noisy"]
    other_image = flow_cases(latent, "otherhash", 123, strata=1, noises=1)[0]["noisy"]
    assert not torch.equal(base, other_seed)
    assert not torch.equal(base, other_image)
    assert case_seed("imagehash", 0, 0, 123) != case_seed("imagehash", 0, 0, 124)


def test_cases_cover_only_the_requested_noise_band():
    latent = torch.zeros(1, 4, 8, 8)
    cases = flow_cases(latent, "imagehash", 1, strata=8, noises=2, sigma_min=0.8, sigma_max=1.0)
    sigmas = [case["sigma"] for case in cases]
    assert len(sigmas) == 8
    assert all(0.8 < sigma < 1.0 for sigma in sigmas)
    assert sigmas == sorted(sigmas)


def test_invalid_noise_band_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        flow_cases(torch.zeros(1, 4, 8, 8), "imagehash", 1, sigma_min=0.9, sigma_max=0.8)


def test_comfy_noise_draws_hold_per_miner_compute_fixed():
    from validator.evaluation.evaluators.diffusion import noises_per_level

    assert noises_per_level(8, 2) == 16
    assert noises_per_level(8, 4) == 16
    assert noises_per_level(8, 8) == 8
    assert noises_per_level(8, 40) == 8
