"""
Order Flow simulation from OHLCV data.

Since we don't have tick-level bid/ask data, we simulate:
- CVD (Cumulative Volume Delta): estimated from close position within bar
- Aggression signals: bars with strong directional volume
- Absorption: high volume bars with small range (absorption of orders)

These are approximations of what Fabio looks for on a footprint chart.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OrderFlowSignal:
    timestamp: pd.Timestamp
    direction: str            # 'buy', 'sell', 'neutral'
    aggression_score: float   # 0.0 – 1.0
    delta_estimate: float     # positive = net buying, negative = net selling
    is_aggressive: bool       # True = big print / imbalance detected
    absorption: bool          # True = high vol, small range (absorption)
    details: dict


def compute_cvd(df: pd.DataFrame) -> pd.Series:
    """
    Estimate Cumulative Volume Delta from OHLCV.

    Formula: delta per bar = volume * ((close - low) - (high - close)) / (high - low)
    This estimates net buying pressure based on close position in bar's range.

    Returns Series aligned with df.index
    """
    hi = df["high"]
    lo = df["low"]
    close = df["close"]
    vol = df["volume"]

    bar_range = (hi - lo).replace(0, np.nan)
    # Fraction of range that close is in (0 = bottom, 1 = top)
    close_position = (close - lo) / bar_range
    # Delta: +volume if close at top, -volume if close at bottom
    delta = vol * (2 * close_position - 1)
    delta = delta.fillna(0)

    cvd = delta.cumsum()
    return cvd


def compute_bar_delta(df: pd.DataFrame) -> pd.Series:
    """Per-bar estimated delta (not cumulative)."""
    hi = df["high"]
    lo = df["low"]
    close = df["close"]
    vol = df["volume"]

    bar_range = (hi - lo).replace(0, np.nan)
    close_position = (close - lo) / bar_range
    delta = vol * (2 * close_position - 1)
    return delta.fillna(0)


def detect_aggression(
    df: pd.DataFrame,
    volume_threshold_pct: float = 1.5,
    delta_threshold_pct: float = 0.6,
) -> pd.DataFrame:
    """
    Detect aggressive bars: bars where volume and directional delta are both elevated.

    volume_threshold_pct: bar volume > X * rolling mean volume = high volume
    delta_threshold_pct: |delta| / volume > X = strong directional bar

    Returns df with added columns:
    - bar_delta: estimated delta per bar
    - is_aggressive_buy: True if strong buy aggression
    - is_aggressive_sell: True if strong sell aggression
    - aggression_score: 0.0 – 1.0
    """
    result = df.copy()
    result["bar_delta"] = compute_bar_delta(df)

    vol_mean = df["volume"].rolling(20, min_periods=5).mean()
    result["vol_mean"] = vol_mean
    result["high_volume"] = df["volume"] > (vol_mean * volume_threshold_pct)

    # Directional ratio: how much of the bar's volume went one way
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    close_pos = (df["close"] - df["low"]) / bar_range
    result["delta_ratio"] = (2 * close_pos - 1).fillna(0)  # -1 to +1

    result["is_aggressive_buy"] = (
        result["high_volume"] &
        (result["delta_ratio"] > delta_threshold_pct)
    )
    result["is_aggressive_sell"] = (
        result["high_volume"] &
        (result["delta_ratio"] < -delta_threshold_pct)
    )

    # Aggression score: combine volume ratio and delta ratio
    vol_ratio = (df["volume"] / vol_mean).fillna(1).clip(0, 5)
    result["aggression_score"] = (
        (vol_ratio / 5) * 0.5 +
        result["delta_ratio"].abs() * 0.5
    ).clip(0, 1)

    return result


def detect_absorption(
    df: pd.DataFrame,
    volume_threshold_pct: float = 2.0,
    range_threshold_pct: float = 0.5,
) -> pd.Series:
    """
    Detect absorption bars: high volume + small range.
    Indicates one side is absorbing the other's aggression.

    Returns boolean Series.
    """
    vol_mean = df["volume"].rolling(20, min_periods=5).mean()
    atr = (df["high"] - df["low"]).rolling(14, min_periods=5).mean()

    high_vol = df["volume"] > (vol_mean * volume_threshold_pct)
    small_range = (df["high"] - df["low"]) < (atr * range_threshold_pct)

    return high_vol & small_range


def get_order_flow_signal_at(
    df: pd.DataFrame,
    bar_idx: int,
    direction_filter: str = "both",
) -> OrderFlowSignal:
    """
    Get order flow signal at a specific bar index.

    direction_filter: 'buy', 'sell', 'both'
    Used by the setup scanner to confirm entry triggers.
    """
    aggr_df = detect_aggression(df)
    absorption = detect_absorption(df)

    row = aggr_df.iloc[bar_idx]
    ts = df.index[bar_idx]

    is_agg_buy = bool(row["is_aggressive_buy"])
    is_agg_sell = bool(row["is_aggressive_sell"])
    is_abs = bool(absorption.iloc[bar_idx])

    if direction_filter == "buy":
        is_aggressive = is_agg_buy
        direction = "buy" if is_agg_buy else "neutral"
    elif direction_filter == "sell":
        is_aggressive = is_agg_sell
        direction = "sell" if is_agg_sell else "neutral"
    else:
        is_aggressive = is_agg_buy or is_agg_sell
        direction = "buy" if is_agg_buy else "sell" if is_agg_sell else "neutral"

    return OrderFlowSignal(
        timestamp=ts,
        direction=direction,
        aggression_score=float(row["aggression_score"]),
        delta_estimate=float(row["bar_delta"]),
        is_aggressive=is_aggressive,
        absorption=is_abs,
        details={
            "volume": float(df["volume"].iloc[bar_idx]),
            "vol_mean": float(row.get("vol_mean", 0)),
            "delta_ratio": float(row["delta_ratio"]),
            "high_volume": bool(row["high_volume"]),
        },
    )


def cvd_trend(cvd: pd.Series, lookback: int = 5) -> str:
    """
    Returns the recent CVD trend direction.
    'up' = net buying pressure, 'down' = net selling, 'flat' = neutral.
    """
    if len(cvd) < lookback:
        return "flat"
    recent = cvd.iloc[-lookback:]
    slope = recent.iloc[-1] - recent.iloc[0]
    std = recent.std()
    if std == 0:
        return "flat"
    normalized = slope / std
    if normalized > 0.5:
        return "up"
    if normalized < -0.5:
        return "down"
    return "flat"
