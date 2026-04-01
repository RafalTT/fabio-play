"""
Volume Profile engine.
Computes: POC, VAH, VAL, LVN zones, HVN zones from OHLCV bars.

Since we don't have tick data, volume is distributed uniformly
across each bar's price range (TPO approximation).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolumeProfile:
    prices: np.ndarray       # price levels (bin centers)
    volumes: np.ndarray      # volume at each price level
    poc: float               # Point of Control
    vah: float               # Value Area High (70% of volume)
    val: float               # Value Area Low (70% of volume)
    value_area_pct: float    # typically 0.70

    @property
    def lvn_zones(self) -> list[tuple[float, float]]:
        """Return price ranges that are Low Volume Nodes."""
        return _find_lvn_zones(self.prices, self.volumes)

    @property
    def hvn_zones(self) -> list[tuple[float, float]]:
        """Return price ranges that are High Volume Nodes."""
        return _find_hvn_zones(self.prices, self.volumes)

    def nearest_lvn(self, price: float, direction: str = "both") -> float | None:
        """
        Find the nearest LVN to a given price.
        direction: 'above', 'below', 'both'
        """
        zones = self.lvn_zones
        if not zones:
            return None
        midpoints = [(lo + hi) / 2 for lo, hi in zones]
        if direction == "above":
            candidates = [m for m in midpoints if m > price]
        elif direction == "below":
            candidates = [m for m in midpoints if m < price]
        else:
            candidates = midpoints

        if not candidates:
            return None
        return min(candidates, key=lambda m: abs(m - price))

    def is_in_value_area(self, price: float) -> bool:
        return self.val <= price <= self.vah

    def is_above_value_area(self, price: float) -> bool:
        return price > self.vah

    def is_below_value_area(self, price: float) -> bool:
        return price < self.val


def build_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 100,
    value_area_pct: float = 0.70,
) -> VolumeProfile:
    """
    Build a Volume Profile from OHLCV bars.

    Distributes each bar's volume uniformly across its price range
    (classical TPO/volume profile approximation for OHLCV data).

    df: DataFrame with [open, high, low, close, volume] columns
    num_bins: number of price bins (resolution)
    value_area_pct: percentage of volume to define the value area (default 70%)
    """
    if df.empty:
        raise ValueError("Cannot build volume profile from empty DataFrame")

    price_min = df["low"].min()
    price_max = df["high"].max()

    if price_min == price_max:
        raise ValueError("Price range is zero — cannot build volume profile")

    bins = np.linspace(price_min, price_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_volumes = np.zeros(num_bins)

    for _, row in df.iterrows():
        lo, hi, vol = row["low"], row["high"], row["volume"]
        if vol <= 0 or lo == hi:
            continue
        # Find bins that overlap with this bar's range
        overlapping = np.where((bins[1:] >= lo) & (bins[:-1] <= hi))[0]
        if len(overlapping) == 0:
            continue
        # Distribute volume proportionally by overlap
        for idx in overlapping:
            bin_lo = bins[idx]
            bin_hi = bins[idx + 1]
            overlap = min(hi, bin_hi) - max(lo, bin_lo)
            fraction = overlap / (hi - lo)
            bin_volumes[idx] += vol * fraction

    # POC = bin with highest volume
    poc_idx = int(np.argmax(bin_volumes))
    poc = float(bin_centers[poc_idx])

    # Value Area: expand from POC outward until 70% of total volume
    vah, val = _compute_value_area(bin_centers, bin_volumes, poc_idx, value_area_pct)

    return VolumeProfile(
        prices=bin_centers,
        volumes=bin_volumes,
        poc=poc,
        vah=vah,
        val=val,
        value_area_pct=value_area_pct,
    )


def build_session_profiles(
    df: pd.DataFrame,
    session: str = "ny",
    num_bins: int = 80,
) -> dict[str, VolumeProfile]:
    """
    Build a Volume Profile per trading session/day.

    session: 'ny' (14:30-21:00 UTC), 'london' (08:00-12:00 UTC), 'full'
    Returns dict: {date_str: VolumeProfile}
    """
    session_hours = {
        "ny": (14, 30, 21, 0),       # 9:30-16:00 ET in UTC
        "london": (8, 0, 12, 0),     # 08:00-12:00 UTC
        "full": (0, 0, 23, 59),
    }
    s_start_h, s_start_m, s_end_h, s_end_m = session_hours.get(session, session_hours["ny"])

    profiles: dict[str, VolumeProfile] = {}

    for day, day_df in df.groupby(df.index.date):
        mask = (
            (day_df.index.hour > s_start_h) |
            ((day_df.index.hour == s_start_h) & (day_df.index.minute >= s_start_m))
        ) & (
            (day_df.index.hour < s_end_h) |
            ((day_df.index.hour == s_end_h) & (day_df.index.minute <= s_end_m))
        )
        session_df = day_df[mask]
        if len(session_df) < 5:
            continue
        try:
            profiles[str(day)] = build_volume_profile(session_df, num_bins=num_bins)
        except ValueError:
            continue

    return profiles


def _compute_value_area(
    prices: np.ndarray,
    volumes: np.ndarray,
    poc_idx: int,
    target_pct: float,
) -> tuple[float, float]:
    """Expand outward from POC until target_pct of total volume is captured."""
    total = volumes.sum()
    target = total * target_pct
    accumulated = volumes[poc_idx]

    lo_idx = poc_idx
    hi_idx = poc_idx

    while accumulated < target:
        can_go_up = hi_idx + 1 < len(volumes)
        can_go_down = lo_idx - 1 >= 0

        if not can_go_up and not can_go_down:
            break

        vol_up = volumes[hi_idx + 1] if can_go_up else -1
        vol_down = volumes[lo_idx - 1] if can_go_down else -1

        if vol_up >= vol_down:
            hi_idx += 1
            accumulated += volumes[hi_idx]
        else:
            lo_idx -= 1
            accumulated += volumes[lo_idx]

    return float(prices[hi_idx]), float(prices[lo_idx])


def _find_lvn_zones(
    prices: np.ndarray,
    volumes: np.ndarray,
    threshold_pct: float = 0.30,
) -> list[tuple[float, float]]:
    """
    Find Low Volume Nodes: contiguous bins below threshold_pct of mean volume.
    These are key reaction zones in Fabio's strategy.
    """
    mean_vol = volumes.mean()
    threshold = mean_vol * threshold_pct

    zones = []
    in_zone = False
    zone_start = 0

    for i, (price, vol) in enumerate(zip(prices, volumes)):
        if vol < threshold:
            if not in_zone:
                in_zone = True
                zone_start = i
        else:
            if in_zone:
                zones.append((float(prices[zone_start]), float(prices[i - 1])))
                in_zone = False

    if in_zone:
        zones.append((float(prices[zone_start]), float(prices[-1])))

    return zones


def _find_hvn_zones(
    prices: np.ndarray,
    volumes: np.ndarray,
    threshold_pct: float = 1.5,
) -> list[tuple[float, float]]:
    """
    Find High Volume Nodes: contiguous bins above threshold_pct of mean volume.
    These are support/resistance and target areas.
    """
    mean_vol = volumes.mean()
    threshold = mean_vol * threshold_pct

    zones = []
    in_zone = False
    zone_start = 0

    for i, (price, vol) in enumerate(zip(prices, volumes)):
        if vol >= threshold:
            if not in_zone:
                in_zone = True
                zone_start = i
        else:
            if in_zone:
                zones.append((float(prices[zone_start]), float(prices[i - 1])))
                in_zone = False

    if in_zone:
        zones.append((float(prices[zone_start]), float(prices[-1])))

    return zones
