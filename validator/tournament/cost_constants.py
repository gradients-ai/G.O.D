import os
from decimal import Decimal


# Provider prices are deployment-specific. Zero defaults keep accounting safe
# until operators configure the real rates.
H100_8X_HOURLY_USD = Decimal(os.getenv("H100_8X_HOURLY_USD", "0"))
A100_HOURLY_USD = Decimal(os.getenv("A100_HOURLY_USD", "0"))

H100_HOURLY_USD = H100_8X_HOURLY_USD / Decimal(8)

COST_CATEGORIES = frozenset({"training", "prep", "evaluation"})
WEEK_PIVOT_HOUR_UTC = 11
