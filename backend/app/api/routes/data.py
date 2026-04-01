from fastapi import APIRouter, HTTPException, Query

from app.data.eodhd import fetch_intraday, fetch_eod
from app.strategy.volume_profile import build_volume_profile, build_session_profiles
from app.strategy.market_state import classify_market_state
from app.strategy.order_flow import compute_cvd, detect_aggression

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/ohlcv")
async def get_ohlcv(
    symbol: str = Query("NQ.INDX"),
    interval: str = Query("5m"),
    from_dt: str = Query(...),
    to_dt: str = Query(...),
):
    """Fetch OHLCV data from EODHD."""
    try:
        df = fetch_intraday(symbol=symbol, interval=interval, from_dt=from_dt, to_dt=to_dt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")

    return {
        "symbol": symbol,
        "interval": interval,
        "from": from_dt,
        "to": to_dt,
        "bars": len(df),
        "data": df.reset_index().rename(columns={"datetime": "time"}).to_dict(orient="records"),
    }


@router.get("/volume-profile")
async def get_volume_profile(
    symbol: str = Query("NQ.INDX"),
    interval: str = Query("5m"),
    from_dt: str = Query(...),
    to_dt: str = Query(...),
    num_bins: int = Query(100),
    session: str = Query("full"),
):
    """Compute volume profile for given period."""
    try:
        df = fetch_intraday(symbol=symbol, interval=interval, from_dt=from_dt, to_dt=to_dt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")

    try:
        vp = build_volume_profile(df, num_bins=num_bins)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "poc": vp.poc,
        "vah": vp.vah,
        "val": vp.val,
        "lvn_zones": vp.lvn_zones,
        "hvn_zones": vp.hvn_zones,
        "profile": [
            {"price": float(p), "volume": float(v)}
            for p, v in zip(vp.prices, vp.volumes)
        ],
    }


@router.get("/market-state")
async def get_market_state(
    symbol: str = Query("NQ.INDX"),
    interval: str = Query("5m"),
    from_dt: str = Query(...),
    to_dt: str = Query(...),
):
    """Classify current market state (balance/imbalance)."""
    try:
        df = fetch_intraday(symbol=symbol, interval=interval, from_dt=from_dt, to_dt=to_dt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")

    result = classify_market_state(df)
    return {
        "state": result.state.value,
        "confidence": result.confidence,
        "directional_bias": result.directional_bias,
        "range_high": result.range_high,
        "range_low": result.range_low,
        "atr": result.atr,
        "displacement_score": result.displacement_score,
        "details": result.details,
    }
