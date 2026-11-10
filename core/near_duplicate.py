"""Near-duplicate rate of a text dataset, shared by validator data prep and model prep."""

import random

from datasketch import MinHash
from datasketch import MinHashLSH


# Token-length stats are computed on a random sample and scaled to the full record count.
LENGTH_SAMPLE_SIZE = 2_000
# Model forward passes and the content stats (including this rate) use the first rows of that sample.
FORWARD_SAMPLE_SIZE = 1_000
LENGTH_SAMPLE_SEED = 42
NEAR_DUP_MAX_WORDS = 512


def near_duplicate_sample_indices(num_rows: int) -> list[int]:
    """Row indices the near-duplicate rate is computed on."""
    if num_rows <= LENGTH_SAMPLE_SIZE:
        indices = list(range(num_rows))
    else:
        indices = random.Random(LENGTH_SAMPLE_SEED).sample(range(num_rows), LENGTH_SAMPLE_SIZE)
    return indices[:FORWARD_SAMPLE_SIZE]


def compute_near_duplicate_rate(texts: list[str], num_perm: int = 128, threshold: float = 0.5) -> float:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    for i, text in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for word in text.lower().split()[:NEAR_DUP_MAX_WORDS]:
            m.update(word.encode("utf-8"))
        minhashes.append(m)
        try:
            lsh.insert(str(i), m)
        except ValueError:
            pass
    dup_count = sum(1 for m in minhashes if len(lsh.query(m)) > 1)
    return dup_count / max(len(texts), 1)
