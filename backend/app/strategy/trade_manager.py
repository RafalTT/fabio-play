"""
Trade Manager — manages open position lifecycle.

Rules from Fabio's playbook:
- Partial take-profit at 1:2 R:R (close 50% of position)
- Move stop to break-even after partial TP
- Full exit at target (POC)
- Trail on strong trend days (optional)
- Daily loss cap: max 3 stop-losses per session
"""
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED_TARGET = "closed_target"
    CLOSED_STOP = "closed_stop"
    CLOSED_PARTIAL_BE = "closed_partial_be"  # partial TP then BE hit
    CLOSED_EOD = "closed_eod"               # end-of-day close


@dataclass
class Trade:
    setup_type: str
    direction: str          # 'long' or 'short'
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float
    target: float
    account_size: float
    risk_pct: float         # e.g. 0.0025 = 0.25%
    session: str
    confidence: float

    # Computed on entry
    position_size: float = field(init=False)
    risk_amount: float = field(init=False)
    risk_points: float = field(init=False)

    # Mutable state
    status: TradeStatus = TradeStatus.OPEN
    exit_price: float | None = None
    exit_time: pd.Timestamp | None = None
    partial_exit_price: float | None = None
    partial_exit_time: pd.Timestamp | None = None
    current_stop: float = field(init=False)
    be_activated: bool = False
    partial_done: bool = False
    pnl: float = 0.0
    pnl_r: float = 0.0  # P&L in R multiples

    def __post_init__(self):
        self.risk_points = abs(self.entry_price - self.stop_loss)
        self.risk_amount = self.account_size * self.risk_pct
        # Position size in "units" (for NQ: points * $20/point)
        self.position_size = self.risk_amount / self.risk_points if self.risk_points > 0 else 0
        self.current_stop = self.stop_loss

    @property
    def partial_target(self) -> float:
        """Price for partial take-profit at 1:2 R:R."""
        risk = abs(self.entry_price - self.stop_loss)
        if self.direction == "long":
            return self.entry_price + 2 * risk
        else:
            return self.entry_price - 2 * risk

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    def to_dict(self) -> dict:
        return {
            "setup_type": self.setup_type,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": str(self.entry_time),
            "stop_loss": self.stop_loss,
            "target": self.target,
            "partial_target": self.partial_target,
            "exit_price": self.exit_price,
            "exit_time": str(self.exit_time) if self.exit_time else None,
            "status": self.status.value,
            "pnl": round(self.pnl, 2),
            "pnl_r": round(self.pnl_r, 3),
            "risk_amount": round(self.risk_amount, 2),
            "position_size": round(self.position_size, 4),
            "session": self.session,
            "confidence": self.confidence,
        }


class TradeManager:
    """
    Manages a single open trade through price updates.
    Call update(bar) on each new bar until trade is closed.
    """

    def __init__(self, trade: Trade):
        self.trade = trade

    def update(self, bar: pd.Series) -> Trade:
        """
        Process a new OHLCV bar. Returns updated Trade.
        Order of checks: stop first, then partial TP, then full target.
        """
        t = self.trade
        if not t.is_open:
            return t

        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        ts = bar.name

        if t.direction == "long":
            self._update_long(high, low, close, ts)
        else:
            self._update_short(high, low, close, ts)

        return t

    def _update_long(self, high: float, low: float, close: float, ts: pd.Timestamp):
        t = self.trade

        # 1. Check stop hit
        if low <= t.current_stop:
            self._close(t.current_stop, ts, TradeStatus.CLOSED_STOP)
            return

        # 2. Partial TP at 1:2 (if not done)
        if not t.partial_done and high >= t.partial_target:
            t.partial_exit_price = t.partial_target
            t.partial_exit_time = ts
            t.partial_done = True
            # Move stop to break-even
            t.current_stop = t.entry_price
            t.be_activated = True

        # 3. Full target
        if high >= t.target:
            exit_price = t.target
            if t.partial_done:
                # Only 50% of position was still open
                self._close_partial_remainder(exit_price, ts)
            else:
                self._close(exit_price, ts, TradeStatus.CLOSED_TARGET)

    def _update_short(self, high: float, low: float, close: float, ts: pd.Timestamp):
        t = self.trade

        # 1. Check stop hit
        if high >= t.current_stop:
            self._close(t.current_stop, ts, TradeStatus.CLOSED_STOP)
            return

        # 2. Partial TP at 1:2
        if not t.partial_done and low <= t.partial_target:
            t.partial_exit_price = t.partial_target
            t.partial_exit_time = ts
            t.partial_done = True
            t.current_stop = t.entry_price
            t.be_activated = True

        # 3. Full target
        if low <= t.target:
            exit_price = t.target
            if t.partial_done:
                self._close_partial_remainder(exit_price, ts)
            else:
                self._close(exit_price, ts, TradeStatus.CLOSED_TARGET)

    def _close(self, exit_price: float, ts: pd.Timestamp, status: TradeStatus):
        t = self.trade
        t.exit_price = exit_price
        t.exit_time = ts
        t.status = status
        if t.direction == "long":
            t.pnl = (exit_price - t.entry_price) * t.position_size
        else:
            t.pnl = (t.entry_price - exit_price) * t.position_size
        t.pnl_r = t.pnl / t.risk_amount if t.risk_amount > 0 else 0

    def _close_partial_remainder(self, exit_price: float, ts: pd.Timestamp):
        t = self.trade
        # Full 100% P&L: first 50% at partial_target, second 50% at full target
        if t.direction == "long":
            partial_pnl = (t.partial_target - t.entry_price) * t.position_size * 0.5
            full_pnl = (exit_price - t.entry_price) * t.position_size * 0.5
        else:
            partial_pnl = (t.entry_price - t.partial_target) * t.position_size * 0.5
            full_pnl = (t.entry_price - exit_price) * t.position_size * 0.5

        t.pnl = partial_pnl + full_pnl
        t.pnl_r = t.pnl / t.risk_amount if t.risk_amount > 0 else 0
        t.exit_price = exit_price
        t.exit_time = ts
        t.status = TradeStatus.CLOSED_TARGET

    def force_close(self, price: float, ts: pd.Timestamp):
        """Force close at end of session or daily limit hit."""
        self._close(price, ts, TradeStatus.CLOSED_EOD)


class SessionController:
    """
    Controls daily session limits:
    - Max 3 stop-losses per day (Fabio's rule)
    - Tracks session P&L and trade count
    """

    def __init__(self, max_daily_losses: int = 3):
        self.max_daily_losses = max_daily_losses
        self.current_date: str | None = None
        self.daily_losses: int = 0
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.session_active: bool = True

    def new_bar(self, timestamp: pd.Timestamp):
        """Call at start of each bar to check session reset."""
        date_str = timestamp.strftime("%Y-%m-%d")
        if date_str != self.current_date:
            self.current_date = date_str
            self.daily_losses = 0
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.session_active = True

    def can_trade(self) -> bool:
        return self.session_active and self.daily_losses < self.max_daily_losses

    def register_trade_result(self, trade: Trade):
        self.daily_trades += 1
        self.daily_pnl += trade.pnl
        if trade.status == TradeStatus.CLOSED_STOP:
            self.daily_losses += 1
            if self.daily_losses >= self.max_daily_losses:
                self.session_active = False

    def stats(self) -> dict:
        return {
            "date": self.current_date,
            "daily_losses": self.daily_losses,
            "daily_trades": self.daily_trades,
            "daily_pnl": round(self.daily_pnl, 2),
            "session_active": self.session_active,
        }
