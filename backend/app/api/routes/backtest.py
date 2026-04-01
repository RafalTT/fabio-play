import asyncio
import json
import logging
from functools import partial
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.data.alpha_vantage import fetch_intraday, _month_range, _fetch_month
from app.backtest.engine import run_backtest, run_walk_forward, BacktestConfig
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    symbol: str = "QQQ"
    interval: str = "5min"
    from_dt: str
    to_dt: str
    account_size: float = 100_000.0
    risk_per_trade: float = 0.0025
    min_rr: float = 1.5
    max_daily_losses: int = 3
    walk_forward: bool = False
    window_months: int = 3


def _event(type: str, **kwargs) -> str:
    return f"data: {json.dumps({'type': type, **kwargs})}\n\n"


async def _stream(req: BacktestRequest) -> AsyncGenerator[str, None]:
    """Stream backtest progress as SSE events."""
    import pandas as pd
    from datetime import datetime

    try:
        # --- Step 1: Calculate months to fetch ---
        from_date = pd.Timestamp(req.from_dt).date()
        to_date = pd.Timestamp(req.to_dt).date()
        months = _month_range(from_date, to_date)
        total_months = len(months)

        yield _event("start", msg=f"Zaczynam: {total_months} miesięcy danych do pobrania", total=total_months)

        # --- Step 2: Fetch month by month with progress ---
        frames = []
        for i, month in enumerate(months):
            pct = int((i / total_months) * 60)
            yield _event("progress", msg=f"Pobieranie danych: {month}", month=month, pct=pct, step=i+1, total=total_months)

            loop = asyncio.get_event_loop()
            df_month = await loop.run_in_executor(
                None, partial(_fetch_month, req.symbol, req.interval, month, True)
            )
            if df_month is not None and not df_month.empty:
                frames.append(df_month)

        if not frames:
            yield _event("error", msg="Brak danych dla podanego okresu i symbolu")
            return

        yield _event("progress", msg="Łączę dane...", pct=62)

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        from_ts = pd.Timestamp(from_date, tz="UTC")
        to_ts = pd.Timestamp(to_date, tz="UTC") + pd.Timedelta(days=1)
        df = df[(df.index >= from_ts) & (df.index < to_ts)]

        yield _event("progress", msg=f"Dane gotowe: {len(df)} barów ({req.from_dt} → {req.to_dt})", pct=65, bars=len(df))

        # --- Step 3: Run backtest ---
        config = BacktestConfig(
            account_size=req.account_size,
            risk_per_trade=req.risk_per_trade,
            min_rr=req.min_rr,
            max_daily_losses=req.max_daily_losses,
        )

        yield _event("progress", msg="Uruchamiam silnik backtestowy...", pct=70)

        if req.walk_forward:
            from app.backtest.engine import run_walk_forward
            from datetime import timedelta

            months_list = _month_range(from_date, to_date)
            total_windows = max(1, len(months_list) // req.window_months)

            yield _event("progress", msg=f"Walk-forward: ~{total_windows} okien po {req.window_months} mies.", pct=72)

            results = await loop.run_in_executor(
                None, partial(run_walk_forward, df, config, req.window_months)
            )

            yield _event("progress", msg=f"Obliczam metryki walk-forward...", pct=90)

            yield _event("result", data={
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
            })
        else:
            yield _event("progress", msg="Skanuję setupy bar po barze...", pct=75)

            result = await loop.run_in_executor(
                None, partial(run_backtest, df, config, req.from_dt, req.to_dt)
            )

            yield _event("progress", msg=f"Gotowe! {result.metrics.total_trades} tradów znalezionych", pct=95)

            yield _event("result", data={
                "metrics": result.metrics.to_dict(),
                "trades": result.trades,
                "equity_curve": result.equity_curve,
                "window_start": result.window_start,
                "window_end": result.window_end,
                "total_bars_scanned": result.total_bars_scanned,
            })

        yield _event("done", msg="Backtest zakończony", pct=100)

    except Exception as e:
        logger.exception("Backtest stream error")
        yield _event("error", msg=str(e))


@router.post("/run")
async def run_backtest_stream(req: BacktestRequest):
    """Stream backtest progress via Server-Sent Events."""
    return StreamingResponse(
        _stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


def _aggregate_wf(results) -> dict:
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
