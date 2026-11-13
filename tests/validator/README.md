# Validator Tests

Tests for validator evaluation, scoring, and tournament behavior.

## Contents

- `db/`: SQL helper tests (continuous SFT state, placement persistence, quant tasks and the per-SKU reference cache).
- `lifecycle/`: finalize gating (quant same-SKU references, the reference swap) and Basilica cleanup.
- `evaluation/`: local evaluation and PvP tests.
- `infrastructure/`: S3 client (`S3_SECURE`, path- and virtual-host-style URL parsing).
- `scoring/`: task/tournament scoring pipeline tests.
- `tournament/`: tournament rounds, boss rounds, weight distribution, and analytics tests.
- `__init__.py`: package marker.
