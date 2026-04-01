"""
Market State Classifier
Determines whether the market is in BALANCE or IMBALANCE (out-of-balance).

Logic based on Auction Market Theory:
- BALANCE: price rotates within a range, low directional momentum
- IMBALANCE: displacement, strong directional move away from value

Uses rolling ATR, price range relative to VP, and momentum indicators.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from app.strategy.volume_profile import VolumeProfile, build_volume_profile


class MarketState(str, Enum):
    BALANCE = "balance"
    IMBALANCE = "imbalance"
    TRANSITION = "transition"  # unclear — no trade


@dataclass
class StateResult:
    state: MarketState
    confidence: float          # 0.0 – 1.0
    directional_bias: str      # 'up', 'down', 'neutral'
    range_high: float
    range_low: float
    atr: float
    displacement_score: float  # 0.0 = full balance, 1.0 = strong imbalance
    details: dict


def classify_market_state(
    df: pd.DataFrame,
    lookback_bars: int = 20,
    atr_period: int = 14,
    displacement_threshold: float = 1.5,
    range_contraction_threshold: float = 0.4,
) -> StateResult:
    """
    Classify current market state from recent OHLCV bars.

    df: DataFrame with [open, high, low, close, volume] — recent bars
    lookback_bars: how many bars to analyze
    atr_period: ATR calculation period
    displacement_threshold: price move > X * ATR = imbalance
    range_contraction_threshold: range < X * ATR = balance

    Returns StateResult
    """
    if len(df) < max(lookback_bars, atr_period + 1):
        return StateResult(
            state=MarketState.TRANSITION,
            confidence=0.0,
            directional_bias="neutral",
            range_high=df["high"].max() if not df.empty else 0,
            range_low=df["low"].min() if not df.empty else 0,
            atr=0.0,
            displacement_score=0.5,
            details={"reason": "insufficient data"},
        )

    recent = df.iloc[-lookback_bars:]
    atr = _compute_atr(df, atr_period)

    range_high = recent["high"].max()
    range_low = recent["low"].min()
    price_range = range_high - range_low

    # Displacement score: how far has price moved directionally
    net_move = abs(recent["close"].iloc[-1] - recent["close"].iloc[0])
    displacement_score = net_move / atr if atr > 0 else 0

    # Range score: tight range = balance, wide range + direction = imbalance
    range_ratio = price_range / atr if atr > 0 else 0

    # Directional momentum (slope of closes)
    closes = recent["close"].values
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    slope_norm = slope / atr if atr > 0 else 0

    # Volume trend (increasing = imbalance signal)
    vol_trend = _volume_trend(recent)

    # --- Classification logic ---
    details = {
        "displacement_score": round(displacement_score, 3),
        "range_ratio": round(range_ratio, 3),
        "slope_norm": round(slope_norm, 3),
        "vol_trend": round(vol_trend, 3),
        "atr": round(atr, 4),
    }

    if displacement_score > displacement_threshold and abs(slope_norm) > 0.15:
        state = MarketState.IMBALANCE
        confidence = min(0.95, displacement_score / (displacement_threshold * 2))
        bias = "up" if slope > 0 else "down"
    elif range_ratio < range_contraction_threshold and abs(slope_norm) < 0.08:
        state = MarketState.BALANCE
        confidence = min(0.90, 1 - range_ratio / range_contraction_threshold)
        bias = "neutral"
    else:
        state = MarketState.TRANSITION
        confidence = 0.5
        bias = "up" if slope > 0 else "down" if slope < 0 else "neutral"

    return StateResult(
        state=state,
        confidence=round(confidence, 3),
        directional_bias=bias,
        range_high=round(range_high, 4),
        range_low=round(range_low, 4),
        atr=round(atr, 4),
        displacement_score=round(displacement_score, 3),
        details=details,
    )


def detect_balance_range(
    df: pd.DataFrame,
    min_bars: int = 10,
    max_range_atr_multiple: float = 2.0,
    atr_period: int = 14,
) -> tuple[float, float] | None:
    """
    Detect a balance range: a period where price consolidates.
    Returns (range_low, range_high) if balance is found, else None.
    Used by Mean Reversion setup to define the balance reference.
    """
    if len(df) < min_bars:
        return None

    atr = _compute_atr(df, atr_period)
    recent = df.iloc[-min_bars:]
    high = recent["high"].max()
    low = recent["low"].min()
    price_range = high - low

    if price_range <= max_range_atr_multiple * atr:
        return (round(low, 4), round(high, 4))
    return None


def detect_impulse_leg(
    df: pd.DataFrame,
    min_move_atr: float = 1.5,
    atr_period: int = 14,
    lookback: int = 30,
) -> dict | None:
    """
    Detect the most recent impulse leg that broke structure.
    Returns dict with: {start_idx, end_idx, direction, start_price, end_price, magnitude_atr}
    Used by Trend Model to identify the impulse for VP application.
    """
    if len(df) < lookback:
        return None

    atr = _compute_atr(df, atr_period)
    recent = df.iloc[-lookback:]

    # Find the strongest directional move in the lookback window
    # Sliding window of 5–15 bars
    best = None
    best_score = 0.0

    for window in range(5, min(20, len(recent))):
        for i in range(len(recent) - window):
            segment = recent.iloc[i: i + window]
            net = segment["close"].iloc[-1] - segment["close"].iloc[0]
            score = abs(net) / atr if atr > 0 else 0

            # Check it's directional (not choppy)
            retracements = _count_direction_changes(segment["close"].values)
            if score > best_score and score >= min_move_atr and retracements <= 2:
                best_score = score
                direction = "up" if net > 0 else "down"
                best = {
                    "start_idx": i,
                    "end_idx": i + window - 1,
                    "direction": direction,
                    "start_price": round(float(segment["close"].iloc[0]), 4),
                    "end_price": round(float(segment["close"].iloc[-1]), 4),
                    "magnitude_atr": round(score, 3),
                    "start_time": segment.index[0],
                    "end_time": segment.index[-1],
                }

    return best


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else float(tr.mean())


def _volume_trend(df: pd.DataFrame) -> float:
    """Returns normalized slope of volume (positive = increasing volume)."""
    vols = df["volume"].values
    if len(vols) < 2:
        return 0.0
    x = np.arange(len(vols))
    slope = np.polyfit(x, vols, 1)[0]
    return float(slope / (vols.mean() + 1e-9))


def _count_direction_changes(closes: np.ndarray) -> int:
    """Count how many times the price reverses direction."""
    diffs = np.diff(closes)
    signs = np.sign(diffs[diffs != 0])
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))
