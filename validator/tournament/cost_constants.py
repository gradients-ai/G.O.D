from decimal import Decimal

from validator.tournament import constants as tourn_cst


H100_8X_HOURLY_USD = Decimal("29.12")
A100_HOURLY_USD = Decimal("1.77")

H100_HOURLY_USD = H100_8X_HOURLY_USD / Decimal(8)

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
