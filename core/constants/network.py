import os


VERSION_KEY = 61_000
DEFAULT_NETUID = 56

try:
    NETUID = int(os.getenv("NETUID", DEFAULT_NETUID))
except (TypeError, ValueError):
    NETUID = DEFAULT_NETUID

IS_PROD_ENV = NETUID == DEFAULT_NETUID

# Public subnet data (auditing, tournaments, emissions). api.gradients.io serves the training platform.
SUBNET_API_URL = os.getenv("SUBNET_API_URL", "https://api.sn56.io").rstrip("/")
