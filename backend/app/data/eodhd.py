"""
EODHD data fetcher for intraday OHLCV.
Supports 1m, 5m, 1h intervals.
NQ futures continuous contract: NQ.INDX
"""
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from app.config import settings

logger = logging.getLogger(__name__)

EODHD_BASE = "https://eodhistoricaldata.com/api"
INTERVAL_TYPE = Literal["1m", "5m", "1h", "1d"]


def _cache_path(symbol: str, interval: str, from_dt: str, to_dt: str) -> Path:
    safe = symbol.replace(".", "_")
    return settings.cache_path / f"{safe}_{interval}_{from_dt}_{to_dt}.parquet"


def fetch_intraday(
    symbol: str,
    interval: INTERVAL_TYPE = "5m",
    from_dt: str | None = None,
    to_dt: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV from EODHD.

    symbol: e.g. 'NQ.INDX', 'QQQ.US', 'NQ2506.CME'
    interval: '1m', '5m', '1h'
    from_dt / to_dt: 'YYYY-MM-DD' strings

    Returns DataFrame with columns: [open, high, low, close, volume]
    index: DatetimeTZAware (UTC)
    """
    if not from_dt:
        from_dt = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not to_dt:
        to_dt = datetime.utcnow().strftime("%Y-%m-%d")

    cache_file = _cache_path(symbol, interval, from_dt, to_dt)
    if use_cache and cache_file.exists():
        logger.info("Loading from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    if not settings.eodhd_api_key:
        raise ValueError("EODHD_API_KEY not set in .env")

    url = f"{EODHD_BASE}/intraday/{symbol}"
    params = {
        "api_token": settings.eodhd_api_key,
        "interval": interval,
        "from": from_dt,
        "to": to_dt,
        "fmt": "json",
    }

    logger.info("Fetching EODHD intraday %s %s [%s → %s]", symbol, interval, from_dt, to_dt)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        logger.warning("EODHD returned empty data for %s", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    if use_cache:
        df.to_parquet(cache_file)
        logger.info("Cached to %s", cache_file)

    return df


def fetch_eod(
    symbol: str,
    from_dt: str | None = None,
    to_dt: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch end-of-day OHLCV from EODHD.
    Useful for daily context (prior-day POC/VAH/VAL).
    """
    if not from_dt:
        from_dt = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not to_dt:
        to_dt = datetime.utcnow().strftime("%Y-%m-%d")

    cache_file = _cache_path(symbol, "1d", from_dt, to_dt)
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    if not settings.eodhd_api_key:
        raise ValueError("EODHD_API_KEY not set in .env")

    url = f"{EODHD_BASE}/eod/{symbol}"
    params = {
        "api_token": settings.eodhd_api_key,
        "from": from_dt,
        "to": to_dt,
        "fmt": "json",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    if use_cache:
        df.to_parquet(cache_file)

    return df
