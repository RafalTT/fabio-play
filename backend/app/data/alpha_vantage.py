"""
Alpha Vantage fallback fetcher.
Used when EODHD doesn't have the needed symbol or as a cross-check.
Supports intraday (1min, 5min, 15min, 30min, 60min) and daily.
"""
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from app.config import settings

logger = logging.getLogger(__name__)

AV_BASE = "https://www.alphavantage.co/query"
AV_INTERVAL = Literal["1min", "5min", "15min", "30min", "60min"]


def _cache_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace(".", "_")
    return settings.cache_path / f"av_{safe}_{interval}.parquet"


def fetch_intraday(
    symbol: str,
    interval: AV_INTERVAL = "5min",
    outputsize: str = "full",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch intraday data from Alpha Vantage.

    symbol: e.g. 'QQQ', 'SPY' (AV doesn't support futures directly)
    interval: '1min', '5min', '15min', '30min', '60min'

    Returns DataFrame with [open, high, low, close, volume]
    """
    cache_file = _cache_path(symbol, interval)
    if use_cache and cache_file.exists():
        logger.info("AV cache hit: %s", cache_file)
        return pd.read_parquet(cache_file)

    if not settings.alpha_vantage_api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set in .env")

    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": settings.alpha_vantage_api_key,
        "datatype": "json",
        "extended_hours": "false",
    }

    logger.info("Fetching AV intraday %s %s", symbol, interval)
    resp = requests.get(AV_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    key = f"Time Series ({interval})"
    if key not in data:
        raise ValueError(f"AV response missing key '{key}'. Response: {data}")

    df = pd.DataFrame.from_dict(data[key], orient="index")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.astype(float)

    if use_cache:
        df.to_parquet(cache_file)

    return df
