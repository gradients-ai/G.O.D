import os
from dataclasses import dataclass
from dataclasses import field


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    validator_url: str
    validator_api_key: str = field(repr=False)
    request_timeout: float = 30.0
    slow_request_timeout: float = 120.0
    cache_seconds: float = 30.0
    cors_origins: tuple[str, ...] = ()
    public_url: str = "https://api.sn56.io"

    @classmethod
    def from_env(cls) -> "Settings":
        url = os.getenv("VALIDATOR_URL", "").rstrip("/")
        key = os.getenv("VALIDATOR_API_KEY", "")
        if not url or not key:
            raise RuntimeError("VALIDATOR_URL and VALIDATOR_API_KEY must be set")
        raw_origins = os.getenv("SUBNET_API_CORS_ORIGINS", "")
        origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
        return cls(
            validator_url=url,
            validator_api_key=key,
            request_timeout=_float("SUBNET_API_REQUEST_TIMEOUT", 30.0),
            slow_request_timeout=_float("SUBNET_API_SLOW_REQUEST_TIMEOUT", 120.0),
            cache_seconds=_float("SUBNET_API_CACHE_SECONDS", 30.0),
            cors_origins=origins,
            public_url=os.getenv("SUBNET_API_PUBLIC_URL", "https://api.sn56.io").rstrip("/"),
        )
