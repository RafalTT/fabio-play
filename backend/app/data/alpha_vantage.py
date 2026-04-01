"""
Alpha Vantage data fetcher.
Primary source for intraday OHLCV (QQQ as NQ proxy).

Alpha Vantage TIME_SERIES_INTRADAY with month parameter returns
full month of data. We iterate months to build longer history.
Rate limit: 75 req/min on paid plan.
"""
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from app.config import settings

logger = logging.getLogger(__name__)

AV_BASE = "https://www.alphavantage.co/query"
AV_INTERVAL = Literal["1min", "5min", "15min", "30min", "60min"]


def _cache_path(symbol: str, interval: str, month: str) -> Path:
    safe = symbol.replace(".", "_")
    return settings.cache_path / f"av_{safe}_{interval}_{month}.parquet"


def fetch_intraday(
    symbol: str,
    interval: AV_INTERVAL = "5min",
    from_dt: str | None = None,
    to_dt: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV from Alpha Vantage for a date range.

    Iterates month-by-month (AV returns one full month per request).
    Caches each month separately.

    symbol: e.g. 'QQQ', 'SPY'
    interval: '1min', '5min', '15min', '30min', '60min'
    from_dt / to_dt: 'YYYY-MM-DD'

    Returns DataFrame with [open, high, low, close, volume], UTC index.
    """
    if not settings.alpha_vantage_api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set in .env")

    from_date = pd.Timestamp(from_dt or "2024-01-01").date()
    to_date = pd.Timestamp(to_dt or datetime.utcnow().strftime("%Y-%m-%d")).date()

    months = _month_range(from_date, to_date)
    frames = []

    for month_str in months:
        df_month = _fetch_month(symbol, interval, month_str, use_cache)
        if df_month is not None and not df_month.empty:
            frames.append(df_month)

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Slice to requested range
    from_ts = pd.Timestamp(from_date, tz="UTC")
    to_ts = pd.Timestamp(to_date, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df.index >= from_ts) & (df.index < to_ts)]

    return df


def _fetch_month(
    symbol: str,
    interval: str,
    month: str,
    use_cache: bool,
) -> pd.DataFrame | None:
    """Fetch a single month of intraday data."""
    cache_file = _cache_path(symbol, interval, month)

    # Don't re-cache current month (data still incoming)
    current_month = datetime.utcnow().strftime("%Y-%m")
    is_current = month == current_month

    if use_cache and cache_file.exists() and not is_current:
        logger.debug("AV cache hit: %s", cache_file.name)
        return pd.read_parquet(cache_file)

    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "month": month,
        "outputsize": "full",
        "apikey": settings.alpha_vantage_api_key,
        "datatype": "json",
        "extended_hours": "false",
    }

    logger.info("AV fetch: %s %s %s", symbol, interval, month)
    try:
        resp = requests.get(AV_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("AV request failed for %s %s: %s", symbol, month, e)
        return None

    ts_key = f"Time Series ({interval})"
    if ts_key not in data:
        if "Note" in data:
            logger.warning("AV rate limit hit: %s", data["Note"])
            time.sleep(12)  # wait and skip this month
        elif "Information" in data:
            logger.warning("AV info: %s", data["Information"])
        else:
            logger.warning("AV unexpected response for %s %s: %s", symbol, month, str(data)[:200])
        return None

    raw = data[ts_key]
    if not raw:
        return None

    df = pd.DataFrame.from_dict(raw, orient="index")
    # AV columns: "1. open", "2. high", "3. low", "4. close", "5. volume"
    df.columns = ["open", "high", "low", "close", "volume"]
    # AV returns timestamps in US/Eastern — localize then convert to UTC
    df.index = (
        pd.to_datetime(df.index)
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    df = df.sort_index().astype(float)

    if use_cache and not is_current:
        df.to_parquet(cache_file)

    return df


def fetch_daily(
    symbol: str,
    from_dt: str | None = None,
    to_dt: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Alpha Vantage (TIME_SERIES_DAILY_ADJUSTED).
    Useful for prior-day context (POC reference).
    """
    if not settings.alpha_vantage_api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set in .env")

    cache_file = settings.cache_path / f"av_{symbol}_daily.parquet"
    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": settings.alpha_vantage_api_key,
            "datatype": "json",
        }
        resp = requests.get(AV_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        key = "Time Series (Daily)"
        if key not in data:
            raise ValueError(f"AV daily: unexpected response: {str(data)[:300]}")

        df = pd.DataFrame.from_dict(data[key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(columns={
            "1. open": "open", "2. high": "high", "3. low": "low",
            "4. close": "close", "6. volume": "volume",
        })[["open", "high", "low", "close", "volume"]].astype(float)

        if use_cache:
            df.to_parquet(cache_file)

    # Slice to range
    if from_dt:
        df = df[df.index >= pd.Timestamp(from_dt)]
    if to_dt:
        df = df[df.index <= pd.Timestamp(to_dt)]

    return df


def _month_range(from_date: date, to_date: date) -> list[str]:
    """Generate list of 'YYYY-MM' strings between two dates."""
    months = []
    current = date(from_date.year, from_date.month, 1)
    end = date(to_date.year, to_date.month, 1)
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        # Advance by one month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months
