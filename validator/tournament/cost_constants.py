from decimal import Decimal


H100_8X_HOURLY_USD = Decimal("29.12")
A100_HOURLY_USD = Decimal("1.77")

H100_HOURLY_USD = H100_8X_HOURLY_USD / Decimal(8)

COST_CATEGORIES = frozenset({"training", "prep", "evaluation"})
WEEK_PIVOT_HOUR_UTC = 11
