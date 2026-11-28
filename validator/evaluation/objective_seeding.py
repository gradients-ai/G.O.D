"""Order-independent RNG seeding shared by the AI Toolkit and Musubi objective evaluators.

Both trainers shuffle their data, so a seed derived from visit order would give
the same held-out sample different noise between runs and between candidates.
Seeds here depend only on the sample identity and how often it has been scored.
"""

import hashlib
import os
import random

import numpy as np
import torch


CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def case_seed(base_seed: int, sample_key: str, repeat: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_key}:{repeat}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def stratified_case(base_seed: int, sample_key: str, repeat: int, repeats: int, lo: int, hi: int) -> tuple[int, float, int]:
    """Seed, noise sign and timestep index for one case of a stratified, antithetic evaluation.

    Repeats come in pairs: repeat 2k and 2k+1 share one seed (the same noise draw) with the noise sign flipped, so the
    pair's errors cancel most of the noise sample's own variance. Pair k gets the k-th of repeats // 2 equal slices of
    the timestep index range [lo, hi), so every sample covers the whole noise band evenly instead of at random.
    """
    if repeats < 2 or repeats % 2:
        raise ValueError("Stratified antithetic evaluation needs an even number of repeats")
    if not 0 <= repeat < repeats or hi <= lo:
        raise ValueError("Case repeat or timestep range out of bounds")
    pair, odd = divmod(repeat, 2)
    pairs = repeats // 2
    seed = case_seed(base_seed, sample_key, 2 * pair)
    position = random.Random(case_seed(base_seed, sample_key, 10_000 + pair)).random()
    index = lo + min(hi - lo - 1, int((pair + position) * (hi - lo) / pairs))
    return seed, -1.0 if odd else 1.0, index


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_deterministic_kernels() -> None:
    # cuBLAS reads this when its first handle is created, so set it before any CUDA work.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class CaseSeeder:
    """Hands out one seed per (sample, repeat) regardless of the order samples arrive in."""

    def __init__(self, base_seed: int):
        self.base_seed = base_seed
        self.counts: dict[str, int] = {}

    def next(self, sample_key: str) -> tuple[int, int]:
        repeat = self.counts.get(sample_key, 0)
        self.counts[sample_key] = repeat + 1
        seed = case_seed(self.base_seed, sample_key, repeat)
        seed_everything(seed)
        return seed, repeat
