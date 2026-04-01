"""
Walk-Forward Backtest Engine.

Simulates the strategy bar-by-bar with zero lookahead bias.
Each bar represents a decision point: scan for setup, manage open trade.

Walk-forward windows: 3-month rolling (matching Fabio's WTC quarter).
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from app.strategy.setup_scanner import scan_for_setups, SetupType
from app.strategy.trade_manager import (
    Trade, TradeManager, SessionController, TradeStatus
)
from app.backtest.metrics import compute_metrics, compute_equity_curve, BacktestMetrics
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    account_size: float = 100_000.0
    risk_per_trade: float = 0.0025     # 0.25%
    min_rr: float = 1.5
    max_daily_losses: int = 3
    tick_size: float = 0.25            # NQ tick size
    sl_buffer_ticks: int = 2
    atr_period: int = 14
    # Session filtering
    ny_start_utc: int = 14             # 14:30 UTC = 9:30 ET
    ny_end_utc: int = 20               # 20:00 UTC = 15:00 ET
    london_start_utc: int = 8
    london_end_utc: int = 12
    # Warm-up bars before first trade
    warmup_bars: int = 50


@dataclass
class BacktestResult:
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: list[dict]
    equity_curve: list[dict]
    scan_log: list[dict]          # all bars scanned (for debugging/visualization)
    window_start: str
    window_end: str
    total_bars_scanned: int


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> BacktestResult:
    """
    Run a single-window backtest on the provided DataFrame.

    df: intraday OHLCV DataFrame (DatetimeTZ index, UTC)
    config: backtest configuration
    window_start/end: optional date strings to slice the data
    """
    if config is None:
        config = BacktestConfig(
            account_size=settings.default_account_size,
            risk_per_trade=settings.default_risk_per_trade,
        )

    # Slice to window
    data = df.copy()
    if window_start:
        data = data[data.index >= pd.Timestamp(window_start, tz="UTC")]
    if window_end:
        data = data[data.index <= pd.Timestamp(window_end, tz="UTC")]

    if data.empty:
        raise ValueError("No data in the specified window")

    session_ctrl = SessionController(max_daily_losses=config.max_daily_losses)
    closed_trades: list[Trade] = []
    scan_log: list[dict] = []
    active_trade: Trade | None = None
    active_manager: TradeManager | None = None

    bars = list(data.iterrows())
    total_bars = len(bars)

    for bar_idx, (ts, bar) in enumerate(bars):
        if bar_idx < config.warmup_bars:
            continue

        session_ctrl.new_bar(ts)
        session = _get_session(ts, config)

        # --- Manage open trade ---
        if active_trade and active_manager:
            active_manager.update(bar)
            if not active_trade.is_open:
                session_ctrl.register_trade_result(active_trade)
                closed_trades.append(active_trade)
                logger.debug(
                    "Trade closed: %s %s pnl=%.2f R=%.2f",
                    active_trade.status.value,
                    active_trade.direction,
                    active_trade.pnl,
                    active_trade.pnl_r,
                )
                active_trade = None
                active_manager = None
            else:
                # Force close at end of session
                if session == "closed" and active_trade.is_open:
                    active_manager.force_close(float(bar["close"]), ts)
                    session_ctrl.register_trade_result(active_trade)
                    closed_trades.append(active_trade)
                    active_trade = None
                    active_manager = None

        # --- Scan for new setup (only if no open trade and session active) ---
        if active_trade is None and session != "closed" and session_ctrl.can_trade():
            setup = scan_for_setups(
                df=data.iloc[: bar_idx + 1],
                current_bar_idx=bar_idx,
                session=session,
                min_rr=config.min_rr,
                atr_period=config.atr_period,
                sl_buffer_ticks=config.sl_buffer_ticks,
                tick_size=config.tick_size,
            )

            log_entry = {
                "timestamp": str(ts),
                "session": session,
                "setup_type": setup.setup_type.value,
                "direction": setup.direction,
                "entry": setup.entry_price,
                "stop": setup.stop_loss,
                "target": setup.target,
                "rr": setup.risk_reward,
                "confidence": setup.confidence,
            }
            scan_log.append(log_entry)

            if setup.setup_type != SetupType.NO_SETUP and setup.is_valid(config.min_rr):
                active_trade = Trade(
                    setup_type=setup.setup_type.value,
                    direction=setup.direction,
                    entry_price=setup.entry_price,
                    entry_time=ts,
                    stop_loss=setup.stop_loss,
                    target=setup.target,
                    account_size=config.account_size,
                    risk_pct=config.risk_per_trade,
                    session=session,
                    confidence=setup.confidence,
                )
                active_manager = TradeManager(active_trade)
                logger.debug(
                    "Trade opened: %s %s @ %.4f SL=%.4f TP=%.4f",
                    setup.setup_type.value, setup.direction,
                    setup.entry_price, setup.stop_loss, setup.target,
                )

    # Close any remaining open trade at end of data
    if active_trade and active_manager and active_trade.is_open:
        last_price = float(data["close"].iloc[-1])
        active_manager.force_close(last_price, data.index[-1])
        closed_trades.append(active_trade)

    trade_dicts = [t.to_dict() for t in closed_trades]
    trading_days = max(1, len(data.index.normalize().unique()))

    metrics = compute_metrics(trade_dicts, config.account_size, trading_days)
    equity_curve = compute_equity_curve(trade_dicts, config.account_size)

    logger.info(
        "Backtest complete: %d trades, WR=%.1f%%, PF=%.2f, DD=%.1f%%",
        metrics.total_trades,
        metrics.win_rate * 100,
        metrics.profit_factor,
        metrics.max_drawdown_pct * 100,
    )

    return BacktestResult(
        config=config,
        metrics=metrics,
        trades=trade_dicts,
        equity_curve=equity_curve,
        scan_log=scan_log[-500:],  # keep last 500 entries
        window_start=str(data.index[0].date()),
        window_end=str(data.index[-1].date()),
        total_bars_scanned=total_bars,
    )


def run_walk_forward(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
    window_months: int = 3,
) -> list[BacktestResult]:
    """
    Run walk-forward backtest: split data into rolling windows,
    run independent backtest on each window.

    window_months: size of each window (default 3 months = WTC quarter)
    """
    if config is None:
        config = BacktestConfig()

    results = []
    start = df.index[0].to_pydatetime()
    end = df.index[-1].to_pydatetime()
    step = timedelta(days=window_months * 30)

    current = start
    while current + step <= end:
        window_end = current + step
        window_start_str = current.strftime("%Y-%m-%d")
        window_end_str = window_end.strftime("%Y-%m-%d")

        logger.info("Walk-forward window: %s → %s", window_start_str, window_end_str)

        try:
            result = run_backtest(
                df, config,
                window_start=window_start_str,
                window_end=window_end_str,
            )
            results.append(result)
        except (ValueError, Exception) as e:
            logger.warning("Window %s failed: %s", window_start_str, e)

        current = window_end

    return results


def _get_session(ts: pd.Timestamp, config: BacktestConfig) -> str:
    """
    Determine which session the bar belongs to.
    Converts UTC timestamp to US/Eastern to handle DST correctly.
    """
    import pytz
    et = pytz.timezone("America/New_York")
    ts_et = ts.astimezone(et)
    time_minutes = ts_et.hour * 60 + ts_et.minute

    # NY session: 9:30 – 16:00 ET
    ny_start = 9 * 60 + 30
    ny_end = 16 * 60

    # London session: 03:00 – 08:00 ET (08:00–13:00 UTC)
    london_start = 3 * 60
    london_end = 8 * 60

    if london_start <= time_minutes < london_end:
        return "london"
    if ny_start <= time_minutes < ny_end:
        return "ny"
    return "closed"
