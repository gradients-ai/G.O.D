from dataclasses import dataclass


@dataclass
class BaseConfig:
    wallet_name: str
    hotkey_name: str
    subtensor_network: str
    netuid: int
    env: str
    subtensor_address: str | None


@dataclass
class MinerConfig(BaseConfig):
    min_stake_threshold: str
    refresh_nodes: bool


@dataclass
class ValidatorConfig(BaseConfig):
    s3_compatible_endpoint: str
    s3_compatible_access_key: str
    s3_compatible_secret_key: str
    s3_bucket_name: str
    frontend_api_key: str
    validator_port: str
    set_metagraph_weights: bool
    validator_port: str
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str | None = None
    postgres_host: str | None = None
    postgres_port: str | None = None
    gpu_server: str | None = None
    localhost: bool = False
    env_file: str = ".vali.env"
    hf_datasets_trust_remote_code = True
    s3_region: str = "us-east-1"
    refresh_nodes: bool = True
    database_url: str | None = None
    postgres_profile: str = "default"
    discord_webhook: str | None = None


@dataclass
class TrainerConfig:
    wandb_token: str
    huggingface_username: str
    huggingface_token: str


@dataclass
class AuditorConfig(BaseConfig): ...
