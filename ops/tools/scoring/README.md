# Scoring Tools

Scoring and weight calculation probes.

## Contents

- `burn_mock_cycle.py`: mock tournament weight cycle probe. Stale — imports the removed
  champion time-decay constant (`EMISSION_BOOST_DECAY_PER_WIN`); needs updating before use.
- `check_scoring.py`: scoring inspection script.
- `debug_weight_calculation.py`: weight calculation debugging helper.
- `tournament_burn_probe.py`: pytest-style tournament weight probe for ad hoc scoring investigation.
  Stale — its assertions expect a non-zero `burn_weight`, which is always `0.0` now that emission
  decay and burn have been removed; needs updating before use.
