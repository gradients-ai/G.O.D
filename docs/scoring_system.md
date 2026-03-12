# Scoring System and Weight Distribution

## Overview

The Gradient subnet is **100% tournament-based**. Emissions are distributed between:

1. **Tournament champions** (text, image, and environment, calculated separately)
2. **Tournament participants** (0.01% per active participant)
3. **Burn address** (all remaining weight)

## Key Constants

From [`validator/core/constants.py`](../validator/core/constants.py):

```python
# Base allocations
TOURNAMENT_TEXT_WEIGHT = 0.15          # 15% base
TOURNAMENT_IMAGE_WEIGHT = 0.10         # 10% base
TOURNAMENT_ENVIRONMENT_WEIGHT = 0.10   # 10% base
MAX_TEXT_TOURNAMENT_WEIGHT = 0.48      # 48% maximum
MAX_IMAGE_TOURNAMENT_WEIGHT = 0.32     # 32% maximum
MAX_ENVIRONMENT_TOURNAMENT_WEIGHT = 0.20  # 20% maximum

# Performance boosts
EMISSION_MULTIPLIER_THRESHOLD = 0.05   # Must exceed 5% to get boost
EMISSION_MULTIPLIER_RATE = 2.0         # 2x multiplier on excess

# Time-based decay (replaces consecutive wins decay)
EMISSION_DAILY_TIME_DECAY_RATE = 0.0033  # -0.33% per day as champion
EMISSION_TIME_DECAY_START_DATE = 2025-11-26  # When time-based decay began

# Within-tournament distribution
TOURNAMENT_SIMPLE_DECAY_BASE = 0.3     # Exponential decay: 1.0, 0.3, 0.09...

# Participation
TOURNAMENT_PARTICIPATION_WEIGHT = 0.0001  # 0.01% per participant
```

## How Weights Are Calculated

### 1. Base Allocation

```python
text_weight = 0.15
image_weight = 0.10
environment_weight = 0.10
burn_weight = 0.65
```

### 2. Performance Boost (if winner exceeds threshold)

```python
if performance_diff > 0.05:
    emission_increase = (performance_diff - 0.05) * 2.0

    # Apply time-based decay
    days_as_champion = (current_time - first_championship_time).days
    decay = days_as_champion * 0.0033  # 0.33% per day
    emission_increase = emission_increase - decay

    # Apply MAX cap
    text_weight = min(0.15 + emission_increase, 0.48)
```

**Result:** Strong performance = higher allocation, weak performance = more burn

### 3. Dual Weight System

**Critical:** Champions and non-champions use different weight pools.

- **Champion:** Uses the boosted tournament weight pool (e.g., 0.35 if earned 20% boost)
- **Non-champions:** Share the base weight pool (0.15 for text, 0.10 for image, 0.10 for environment)
- Both are then distributed by rank using exponential decay (see below)
- **Undistributed:** Goes to burn address

### 4. Within-Tournament Distribution

Tournament participants are ranked, then weights are distributed using exponential decay:

```python
weight[rank] = 0.3^(rank - 1)
# 1st: 1.0, 2nd: 0.3, 3rd: 0.09, 4th: 0.027...
```

**Example:** Champion (1st place) with 0.35 boosted pool:

- Champion weight = 1.0 \* 0.35 = 0.35

If 3 non-champions share 0.15 base pool at ranks 2nd, 3rd, 4th:

- 2nd place = 0.3 \* (0.15 / sum_of_decay_weights)
- 3rd place = 0.09 \* (0.15 / sum_of_decay_weights)
- 4th place = 0.027 \* (0.15 / sum_of_decay_weights)

Where sum_of_decay_weights = 0.3 + 0.09 + 0.027 = 0.417

## Examples

### Strong Performance (first win)

Text champion outperformed runner-up by 15% (performance_diff = 0.15):

```
Base: 0.15
Boost: (0.15 - 0.05) * 2.0 = 0.20
Text champion weight: 0.35
```

### Long-Reigning Champion (30 days)

Text champion outperformed runner-up by 20% (performance_diff = 0.20):

```
Base: 0.15
Raw boost: (0.20 - 0.05) * 2.0 = 0.30
Decay: 30 days * 0.0033 = 0.099
Final boost: 0.30 - 0.099 = 0.201
Text champion weight: 0.351
```

Despite 20% performance, time-based decay reduces boost from 0.30 to 0.201.

### Weak Performance (below threshold)

Champion outperformed runner-up by only 3% (performance_diff = 0.03):

```
Text champion weight: 0.15 (no boost - below 5% threshold)
```

## Key Mechanisms

**Performance Incentives:** Champions with >5% advantage get 2x boost on excess performance

**Balance Controls:**

- MAX caps prevent domination (48% text, 32% image, 20% environment)
- Time-based decay (-0.33% per day) prevents indefinite reign
- Dual weights ensure only champions benefit from boosts

**Burn as Quality Signal:** Weak performance = high burn rate

**Tournament Independence:** Text, image, and environment tournaments are calculated separately

## Implementation

Main entry point: [`get_node_weights_from_tournament_audit_data`](../validator/core/weight_setting.py)

Key functions:

- [`get_tournament_burn_details`](../validator/core/weight_setting.py) - Calculates allocations with decay and caps
- [`apply_tournament_weights`](../validator/core/weight_setting.py) - Applies dual weight system
- [`tournament_scores_to_weights`](../validator/evaluation/tournament_scoring.py) - Within-tournament distribution
