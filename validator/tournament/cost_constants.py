from decimal import Decimal

from validator.tournament import constants as tourn_cst


# 8x node hourly rates by H100 interconnect. Per-GPU = rate / 8.
H100_SXM_8X_HOURLY_USD = Decimal("34.4")
H100_NVL_8X_HOURLY_USD = Decimal("32")
H100_OTHER_8X_HOURLY_USD = Decimal("29")  # PCIe and anything else

# Backward-compatible alias: "default" H100 8x rate is the non-SXM/non-NVL tier.
H100_8X_HOURLY_USD = H100_OTHER_8X_HOURLY_USD

A100_HOURLY_USD = Decimal("1.77")

H100_SXM_HOURLY_USD = H100_SXM_8X_HOURLY_USD / Decimal(8)
H100_NVL_HOURLY_USD = H100_NVL_8X_HOURLY_USD / Decimal(8)
H100_OTHER_HOURLY_USD = H100_OTHER_8X_HOURLY_USD / Decimal(8)
H100_HOURLY_USD = H100_OTHER_HOURLY_USD

COST_CATEGORIES = frozenset({"training", "prep", "evaluation"})
WEEK_PIVOT_HOUR_UTC = 9

RAO_PER_TAO = Decimal(1_000_000_000)

# Participation fee per tournament type, in RAO (mirrors the canonical values used
# when charging miners). Keyed by TournamentType value.
TOURNAMENT_PARTICIPATION_FEE_RAO_BY_TYPE = {
    "text": Decimal(tourn_cst.TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO),
    "image": Decimal(tourn_cst.TOURNAMENT_IMAGE_PARTICIPATION_FEE_RAO),
    "environment": Decimal(tourn_cst.TOURNAMENT_ENVIRONMENT_PARTICIPATION_FEE_RAO),
}


def h100_hourly_usd_for_interconnect(interconnect: str | None = None) -> Decimal:
    """Per-GPU H100 hourly USD for an interconnect class (SXM / NVL / other)."""
    normalized = (interconnect or "").strip().upper()
    if normalized == "SXM":
        return H100_SXM_HOURLY_USD
    if normalized == "NVL":
        return H100_NVL_HOURLY_USD
    return H100_OTHER_HOURLY_USD
