# Validator Tournament Tests

Tests for tournament lifecycle, boss rounds, tournament weight distribution, and analytics.

## Contents

- `test_pre_boss_third_place.py`: 3rd-place identification tests (pre-boss knockout loser). Placement persistence of that 3rd place is covered in `tests/validator/db/test_tournament_placements.py`.
- `test_separated_burn_dynamics.py`: per-type tournament weight calculation and node-weight application tests.
- `test_text_r1_single_group.py`: text round 1 always uses a single group with three tasks; image keeps the field-size band.
- `test_tournament_analytics.py`: tournament analytics tests.
- `__init__.py`: package marker.
