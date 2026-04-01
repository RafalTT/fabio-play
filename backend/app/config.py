from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    eodhd_api_key: str = ""
    alpha_vantage_api_key: str = ""
    fred_api_key: str = ""

    default_symbol: str = "NQ.INDX"
    default_exchange: str = "INDX"
    default_account_size: float = 100_000.0
    default_risk_per_trade: float = 0.0025  # 0.25%

    cache_dir: str = "./cache"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
