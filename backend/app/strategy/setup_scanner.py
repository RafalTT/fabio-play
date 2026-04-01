"""
Setup Scanner — implements Fabio's two trade models.

Setup 1: Trend Continuation (Out-of-Balance → Seek New Balance)
- Market out of balance (imbalance)
- Impulse leg identified
- VP applied to impulse → LVN found
- Price pulls back to LVN
- Order flow confirms aggression in trend direction
- Stop: below LVN, Target: previous balance POC

Setup 2: Mean Reversion (Failed Breakout → Back Into Balance)
- Market in balance/consolidation
- Price breaks out of range but fails to hold
- Price reclaims inside balance
- Pullback to reclaim LVN
- Order flow confirms snap-back aggression
- Stop: beyond failed high/low, Target: balance POC
"""
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from app.strategy.market_state import (
    MarketState,
    StateResult,
    classify_market_state,
    detect_balance_range,
    detect_impulse_leg,
)
from app.strategy.order_flow import (
    compute_cvd,
    detect_aggression,
    cvd_trend,
)
from app.strategy.volume_profile import build_volume_profile


class SetupType(str, Enum):
    TREND_CONTINUATION = "trend_continuation"
    MEAN_REVERSION = "mean_reversion"
    NO_SETUP = "no_setup"


@dataclass
class TradeSetup:
    setup_type: SetupType
    direction: str            # 'long' or 'short'
    entry_price: float
    stop_loss: float
    target: float
    risk_reward: float
    confidence: float         # 0.0 – 1.0
    timestamp: pd.Timestamp
    session: str              # 'ny' or 'london'
    details: dict = field(default_factory=dict)

    @property
    def risk_points(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_points(self) -> float:
        return abs(self.target - self.entry_price)

    def is_valid(self, min_rr: float = 1.5) -> bool:
        return (
            self.risk_reward >= min_rr and
            self.risk_points > 0 and
            self.reward_points > 0 and
            self.confidence >= 0.3
        )


def scan_for_setups(
    df: pd.DataFrame,
    current_bar_idx: int,
    session: str = "ny",
    min_rr: float = 1.5,
    atr_period: int = 14,
    sl_buffer_ticks: int = 2,
    tick_size: float = 0.25,
) -> TradeSetup:
    """
    Main entry point: scan for a valid setup at the current bar.

    df: full DataFrame with OHLCV up to and including current bar
    current_bar_idx: index of the current bar (no lookahead)
    session: 'ny' or 'london'
    min_rr: minimum reward:risk ratio to consider setup valid

    Returns TradeSetup (with setup_type=NO_SETUP if nothing found)
    """
    data = df.iloc[: current_bar_idx + 1]
    if len(data) < 30:
        return _no_setup(df.index[current_bar_idx])

    state_result = classify_market_state(data, lookback_bars=20, atr_period=atr_period)
    current_price = float(data["close"].iloc[-1])

    # Session gate: NY for trend, London for mean reversion
    # (We still scan both but weight confidence)
    if state_result.state == MarketState.IMBALANCE:
        setup = _scan_trend_continuation(
            data, state_result, current_price, session,
            sl_buffer_ticks, tick_size, atr_period,
        )
    elif state_result.state == MarketState.BALANCE:
        setup = _scan_mean_reversion(
            data, state_result, current_price, session,
            sl_buffer_ticks, tick_size, atr_period,
        )
    else:
        return _no_setup(data.index[-1])

    if setup and setup.is_valid(min_rr):
        return setup

    return _no_setup(data.index[-1])


# ── Setup 1: Trend Continuation ────────────────────────────────────────────

def _scan_trend_continuation(
    data: pd.DataFrame,
    state: StateResult,
    current_price: float,
    session: str,
    sl_buffer_ticks: int,
    tick_size: float,
    atr_period: int,
) -> TradeSetup | None:
    """
    Trend Continuation logic:
    1. Find impulse leg that broke structure
    2. Apply VP to that leg → identify LVNs
    3. Check if price has pulled back to an LVN
    4. Check order flow aggression in trend direction
    5. Compute entry/stop/target
    """
    impulse = detect_impulse_leg(data, min_move_atr=1.5, atr_period=atr_period, lookback=30)
    if not impulse:
        return None

    # Apply VP to the impulse leg
    impulse_data = data.iloc[impulse["start_idx"]: impulse["end_idx"] + 1]
    if len(impulse_data) < 3:
        return None

    try:
        vp = build_volume_profile(impulse_data, num_bins=60)
    except ValueError:
        return None

    lvn_zones = vp.lvn_zones
    if not lvn_zones:
        return None

    direction = impulse["direction"]  # 'up' or 'down'
    sl_buffer = sl_buffer_ticks * tick_size

    # Find nearest LVN that price is near (pullback zone)
    nearest_lvn = None
    lvn_midpoint = None
    for lo, hi in lvn_zones:
        mid = (lo + hi) / 2
        # Price should be near the LVN (within 0.5 ATR)
        if abs(current_price - mid) < state.atr * 0.5:
            nearest_lvn = (lo, hi)
            lvn_midpoint = mid
            break

    if nearest_lvn is None:
        return None

    # Check order flow confirmation
    aggr_df = detect_aggression(data)
    cvd = compute_cvd(data)
    cvd_dir = cvd_trend(cvd, lookback=5)

    of_direction = "buy" if direction == "up" else "sell"
    last_bar = aggr_df.iloc[-1]

    has_aggression = (
        (direction == "up" and bool(last_bar["is_aggressive_buy"])) or
        (direction == "down" and bool(last_bar["is_aggressive_sell"]))
    )
    cvd_confirms = (
        (direction == "up" and cvd_dir == "up") or
        (direction == "down" and cvd_dir == "down")
    )

    # Require at least CVD confirmation if no direct aggression print
    if not has_aggression and not cvd_confirms:
        return None

    # Compute entry, stop, target
    if direction == "up":
        entry = current_price
        stop_loss = nearest_lvn[0] - sl_buffer
        # Target: look for prior balance POC above entry
        target = _find_prior_poc_target(data, entry, "up", state.atr)
    else:
        entry = current_price
        stop_loss = nearest_lvn[1] + sl_buffer
        target = _find_prior_poc_target(data, entry, "down", state.atr)

    if target is None:
        # Fallback: 2.5 * risk
        risk = abs(entry - stop_loss)
        target = entry + (2.5 * risk) if direction == "up" else entry - (2.5 * risk)

    risk = abs(entry - stop_loss)
    reward = abs(target - entry)
    rr = reward / risk if risk > 0 else 0

    confidence = _compute_confidence(
        state.confidence, has_aggression, cvd_confirms,
        session == "ny",  # NY is preferred for trend
    )

    return TradeSetup(
        setup_type=SetupType.TREND_CONTINUATION,
        direction="long" if direction == "up" else "short",
        entry_price=round(entry, 4),
        stop_loss=round(stop_loss, 4),
        target=round(target, 4),
        risk_reward=round(rr, 2),
        confidence=round(confidence, 3),
        timestamp=data.index[-1],
        session=session,
        details={
            "impulse_direction": direction,
            "impulse_magnitude_atr": impulse["magnitude_atr"],
            "lvn_zone": nearest_lvn,
            "has_aggression": has_aggression,
            "cvd_direction": cvd_dir,
            "market_state": state.state.value,
        },
    )


# ── Setup 2: Mean Reversion ────────────────────────────────────────────────

def _scan_mean_reversion(
    data: pd.DataFrame,
    state: StateResult,
    current_price: float,
    session: str,
    sl_buffer_ticks: int,
    tick_size: float,
    atr_period: int,
) -> TradeSetup | None:
    """
    Mean Reversion logic:
    1. Detect balance range (prior consolidation)
    2. Detect failed breakout (price went outside, came back)
    3. Wait for reclaim + pullback to LVN inside balance
    4. Check order flow confirms snap-back direction
    5. Entry/stop/target: POC of balance
    """
    balance_range = detect_balance_range(data, min_bars=10, atr_period=atr_period)
    if not balance_range:
        return None

    range_low, range_high = balance_range
    sl_buffer = sl_buffer_ticks * tick_size

    # Check for failed breakout
    # Look at the last N bars to see if we pushed out and came back
    lookback_bars = data.iloc[-15:]
    recent_high = lookback_bars["high"].max()
    recent_low = lookback_bars["low"].min()

    failed_up = recent_high > range_high and current_price < range_high
    failed_down = recent_low < range_low and current_price > range_low

    if not failed_up and not failed_down:
        return None

    # Don't take the first touch — wait for reclaim confirmation
    # Check that price is now inside balance and making a pullback
    inside_balance = range_low <= current_price <= range_high
    if not inside_balance:
        return None

    # Build VP of the balance range to find POC
    # Use data within the balance price range
    balance_mask = (data["close"] >= range_low * 0.999) & (data["close"] <= range_high * 1.001)
    balance_data = data[balance_mask].iloc[-40:]
    if len(balance_data) < 5:
        return None

    try:
        vp = build_volume_profile(balance_data, num_bins=50)
    except ValueError:
        return None

    # Build VP of the reclaim leg to find LVNs
    reclaim_data = data.iloc[-8:]
    try:
        reclaim_vp = build_volume_profile(reclaim_data, num_bins=30)
        lvn_zones = reclaim_vp.lvn_zones
    except ValueError:
        lvn_zones = []

    # Order flow check
    aggr_df = detect_aggression(data)
    cvd = compute_cvd(data)
    cvd_dir = cvd_trend(cvd, lookback=5)
    last_bar = aggr_df.iloc[-1]

    if failed_up:
        # Short: price failed above balance, now reclaimed inside → short to POC
        direction = "short"
        has_aggression = bool(last_bar["is_aggressive_sell"])
        cvd_confirms = cvd_dir == "down"
        entry = current_price
        stop_loss = recent_high + sl_buffer  # just above the failed high
        target = vp.poc
    else:
        # Long: price failed below balance, now reclaimed inside → long to POC
        direction = "long"
        has_aggression = bool(last_bar["is_aggressive_buy"])
        cvd_confirms = cvd_dir == "up"
        entry = current_price
        stop_loss = recent_low - sl_buffer  # just below the failed low
        target = vp.poc

    if not has_aggression and not cvd_confirms:
        return None

    risk = abs(entry - stop_loss)
    reward = abs(target - entry)
    if risk <= 0:
        return None

    rr = reward / risk
    confidence = _compute_confidence(
        state.confidence, has_aggression, cvd_confirms,
        session in ("london", "ny"),
    )

    return TradeSetup(
        setup_type=SetupType.MEAN_REVERSION,
        direction=direction,
        entry_price=round(entry, 4),
        stop_loss=round(stop_loss, 4),
        target=round(target, 4),
        risk_reward=round(rr, 2),
        confidence=round(confidence, 3),
        timestamp=data.index[-1],
        session=session,
        details={
            "balance_range": balance_range,
            "failed_up": failed_up,
            "failed_down": failed_down,
            "balance_poc": round(vp.poc, 4),
            "has_aggression": has_aggression,
            "cvd_direction": cvd_dir,
            "market_state": state.state.value,
        },
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _find_prior_poc_target(
    data: pd.DataFrame,
    entry: float,
    direction: str,
    atr: float,
) -> float | None:
    """
    Look for a prior balance area's POC in the target direction.
    Uses a 20-bar rolling window shifted back.
    """
    if len(data) < 40:
        return None

    prior_data = data.iloc[-40:-20]
    try:
        prior_vp = build_volume_profile(prior_data, num_bins=50)
        poc = prior_vp.poc
        if direction == "up" and poc > entry + atr * 0.5:
            return poc
        if direction == "down" and poc < entry - atr * 0.5:
            return poc
    except ValueError:
        pass
    return None


def _compute_confidence(
    state_confidence: float,
    has_aggression: bool,
    cvd_confirms: bool,
    session_match: bool,
) -> float:
    score = state_confidence * 0.4
    score += 0.3 if has_aggression else 0.0
    score += 0.15 if cvd_confirms else 0.0
    score += 0.15 if session_match else 0.0
    return min(1.0, score)


def _no_setup(timestamp: pd.Timestamp) -> TradeSetup:
    return TradeSetup(
        setup_type=SetupType.NO_SETUP,
        direction="neutral",
        entry_price=0.0,
        stop_loss=0.0,
        target=0.0,
        risk_reward=0.0,
        confidence=0.0,
        timestamp=timestamp,
        session="",
    )
