from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.data.eodhd import fetch_intraday
from app.backtest.engine import run_backtest, run_walk_forward, BacktestConfig
from app.config import settings

router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    symbol: str = "NQ.INDX"
    interval: str = "5m"
    from_dt: str
    to_dt: str
    account_size: float = 100_000.0
    risk_per_trade: float = 0.0025
    min_rr: float = 1.5
    max_daily_losses: int = 3
    walk_forward: bool = False
    window_months: int = 3


@router.post("/run")
async def run_backtest_endpoint(req: BacktestRequest):
    """Run a backtest on given symbol and date range."""
    try:
        df = fetch_intraday(
            symbol=req.symbol,
            interval=req.interval,
            from_dt=req.from_dt,
            to_dt=req.to_dt,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data fetch failed: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail="No data returned for given parameters")

    config = BacktestConfig(
        account_size=req.account_size,
        risk_per_trade=req.risk_per_trade,
        min_rr=req.min_rr,
        max_daily_losses=req.max_daily_losses,
    )

    try:
        if req.walk_forward:
            results = run_walk_forward(df, config, window_months=req.window_months)
            return {
                "windows": len(results),
                "results": [
                    {
                        "window": f"{r.window_start} → {r.window_end}",
                        "metrics": r.metrics.to_dict(),
                        "trades_count": len(r.trades),
                    }
                    for r in results
                ],
                "aggregated": _aggregate_wf(results),
            }
        else:
            result = run_backtest(df, config, req.from_dt, req.to_dt)
            return {
                "metrics": result.metrics.to_dict(),
                "trades": result.trades,
                "equity_curve": result.equity_curve,
                "window_start": result.window_start,
                "window_end": result.window_end,
                "total_bars_scanned": result.total_bars_scanned,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _aggregate_wf(results) -> dict:
    """Aggregate walk-forward metrics across all windows."""
    if not results:
        return {}

    all_trades = sum(r.metrics.total_trades for r in results)
    all_pnl = sum(r.metrics.total_pnl for r in results)
    avg_wr = sum(r.metrics.win_rate for r in results) / len(results)
    avg_pf = sum(r.metrics.profit_factor for r in results) / len(results)
    max_dd = max(r.metrics.max_drawdown_pct for r in results)

    return {
        "total_trades": all_trades,
        "total_pnl": round(all_pnl, 2),
        "avg_win_rate": round(avg_wr * 100, 1),
        "avg_profit_factor": round(avg_pf, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
    }
