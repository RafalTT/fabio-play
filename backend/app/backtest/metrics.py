"""
Performance metrics calculator.
Benchmarks against Fabio's declared targets:
- Win rate ~50%
- Min R:R 1:2
- Max drawdown < 20%
- Profit factor > 1.5
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    avg_rr: float
    profit_factor: float
    total_pnl: float
    max_drawdown_pct: float
    sharpe_ratio: float
    net_pnl_r: float              # total P&L in R multiples
    trades_per_day: float
    # Benchmarks vs Fabio's targets
    benchmark: dict

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate * 100, 1),
            "avg_win_r": round(self.avg_win_r, 2),
            "avg_loss_r": round(self.avg_loss_r, 2),
            "avg_rr": round(self.avg_rr, 2),
            "profit_factor": round(self.profit_factor, 2),
            "total_pnl": round(self.total_pnl, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "net_pnl_r": round(self.net_pnl_r, 2),
            "trades_per_day": round(self.trades_per_day, 1),
            "benchmark": self.benchmark,
        }


def compute_metrics(
    trades: list[dict],
    account_size: float,
    trading_days: int,
) -> BacktestMetrics:
    """
    Compute performance metrics from a list of closed trade dicts.
    Each trade dict must have: pnl, pnl_r, status
    """
    if not trades:
        return _empty_metrics(account_size)

    df = pd.DataFrame(trades)
    closed = df[df["status"].isin(["closed_target", "closed_stop", "closed_partial_be", "closed_eod"])]

    if closed.empty:
        return _empty_metrics(account_size)

    # Exclude EOD closes from win/loss stats (not real signal exits)
    signal_exits = closed[closed["status"] != "closed_eod"]
    eod_exits = closed[closed["status"] == "closed_eod"]

    wins = signal_exits[signal_exits["pnl_r"] > 0]
    losses = signal_exits[signal_exits["pnl_r"] <= 0]

    total = len(closed)
    n_wins = len(wins)
    n_losses = len(losses)
    signal_total = len(signal_exits)
    win_rate = n_wins / signal_total if signal_total > 0 else 0

    avg_win_r = float(wins["pnl_r"].mean()) if n_wins > 0 else 0
    avg_loss_r = float(losses["pnl_r"].mean()) if n_losses > 0 else 0
    avg_rr = abs(avg_win_r / avg_loss_r) if avg_loss_r != 0 else 0

    gross_profit = float(wins["pnl"].sum()) if n_wins > 0 else 0
    gross_loss = abs(float(losses["pnl"].sum())) if n_losses > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = float(closed["pnl"].sum())
    net_pnl_r = float(closed["pnl_r"].sum())

    # Equity curve and drawdown
    equity_curve = account_size + closed["pnl"].cumsum()
    max_dd = _max_drawdown(equity_curve.values)

    # Sharpe (daily returns)
    sharpe = _sharpe_ratio(closed["pnl"].values, trading_days)

    trades_per_day = total / trading_days if trading_days > 0 else 0

    # Benchmark vs Fabio targets
    benchmark = {
        "win_rate_target": "~50%",
        "win_rate_ok": win_rate >= 0.45,
        "rr_target": "min 1:2",
        "rr_ok": avg_rr >= 2.0,
        "drawdown_target": "< 20%",
        "drawdown_ok": max_dd < 0.20,
        "profit_factor_target": "> 1.5",
        "profit_factor_ok": profit_factor > 1.5,
    }

    return BacktestMetrics(
        total_trades=total,
        winning_trades=n_wins,
        losing_trades=n_losses,
        win_rate=win_rate,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        avg_rr=avg_rr,
        profit_factor=profit_factor,
        total_pnl=total_pnl,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        net_pnl_r=net_pnl_r,
        trades_per_day=trades_per_day,
        benchmark=benchmark,
    )


def compute_equity_curve(
    trades: list[dict],
    account_size: float,
) -> list[dict]:
    """Returns equity curve as list of {time, equity, drawdown}."""
    if not trades:
        return []

    df = pd.DataFrame(trades)
    closed = df[df["status"].isin(["closed_target", "closed_stop", "closed_partial_be", "closed_eod"])]
    closed = closed.sort_values("exit_time")

    equity = account_size
    peak = account_size
    curve = []

    for _, row in closed.iterrows():
        equity += row["pnl"]
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        curve.append({
            "time": row["exit_time"],
            "equity": round(equity, 2),
            "drawdown": round(dd * 100, 2),
            "trade_pnl": round(row["pnl"], 2),
            "pnl_r": round(row["pnl_r"], 3),
        })

    return curve


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown as fraction of peak equity."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(dd.max()) if len(dd) > 0 else 0.0


def _sharpe_ratio(pnls: np.ndarray, trading_days: int) -> float:
    """Simplified Sharpe ratio (annualized, risk-free = 0)."""
    if len(pnls) < 2:
        return 0.0
    mean = np.mean(pnls)
    std = np.std(pnls)
    if std == 0:
        return 0.0
    # Annualize: assume ~252 trading days, scale by trades/day
    trades_per_day = len(pnls) / trading_days if trading_days > 0 else 1
    annual_factor = np.sqrt(252 * trades_per_day)
    return float((mean / std) * annual_factor)


def _empty_metrics(account_size: float) -> BacktestMetrics:
    return BacktestMetrics(
        total_trades=0, winning_trades=0, losing_trades=0,
        win_rate=0, avg_win_r=0, avg_loss_r=0, avg_rr=0,
        profit_factor=0, total_pnl=0, max_drawdown_pct=0,
        sharpe_ratio=0, net_pnl_r=0, trades_per_day=0,
        benchmark={},
    )
