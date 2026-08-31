"""Layer-2 grid-congestion *regime* signal for the equal-weight grid-equipment
basket.

The basket's bull case is "the electricity grid is the bottleneck for the AI
buildout." Whether that bottleneck is **tightening or easing** is measurable in
the physical wholesale-power market -- transmission congestion and reserve-margin
tightness in the ISO zones where data-center load concentrates -- *before* it
shows up in equipment-maker margins, and it is not in prices or momentum. This
module turns that read into one **monthly exposure multiplier** on the basket
(step 8 of the spec), which layer 1's `apply_overlay_l2` applies in place of the
price trend gate:

    exposure = regime_multiplier(grid_regime)  x  vol_target_scalar(basket_returns)

Design / gate / honesty caveats:
    docs/superpowers/specs/2026-08-31-grid-regime-layer2-design.md

Data is the already-cached PJM zone-level LMP (real congestion component) and
per-zone load, read through `grid_resilience.data.grid_data` public functions --
NOT the heavy `build_multi_iso_gsi` pipeline. The gated backtest does zero
network I/O.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from grid_equipment_basket import config
from grid_equipment_basket.overlay import _month_hold


# ── per-zone daily signals ───────────────────────────────────────────────────

def _daily_zone_congestion(lmp_df: pd.DataFrame) -> pd.DataFrame:
    """Hourly zone LMP rows -> daily mean of ``|congestion $|`` per zone.

    Absolute dollars, *not* the ``|congestion| / |LMP|`` ratio -- the ratio
    blows up to >300% on divide-by-near-zero LMP hours (seen at COMED). Returns
    a frame indexed by ``date`` with one sorted column per zone; empty in ->
    empty out.
    """
    if lmp_df is None or lmp_df.empty:
        return pd.DataFrame()
    df = lmp_df.copy()
    df["_date"] = pd.to_datetime(df["time"]).dt.normalize()
    df["_abs_cong"] = df["congestion"].abs()
    daily = (df.groupby(["_date", "location"])["_abs_cong"].mean()
               .unstack("location").sort_index(axis=1))
    daily.index.name = "date"
    return daily


def _daily_zone_peak_load(load_df: pd.DataFrame) -> pd.DataFrame:
    """Hourly per-zone load rows -> daily peak MW per zone (date x zone)."""
    if load_df is None or load_df.empty:
        return pd.DataFrame()
    df = load_df.copy()
    df["_date"] = pd.to_datetime(df["time"]).dt.normalize()
    daily = (df.groupby(["_date", "zone"])["load_mw"].max()
               .unstack("zone").sort_index(axis=1))
    daily.index.name = "date"
    return daily


def _trailing_zscore(s: pd.Series, window: int, min_periods: int,
                     winsor: float) -> pd.Series:
    """``clip((s - trailing_mean) / trailing_std, +/- winsor)`` using a rolling
    window that includes only past-and-current observations (point-in-time). A
    zero-variance window -> NaN, not inf."""
    s = s.astype(float)
    roll = s.rolling(window, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std().replace(0.0, np.nan)
    return ((s - mean) / std).clip(-winsor, winsor)


def _combine_subsignals(subsignals: dict[str, tuple[pd.Series, float]]) -> pd.Series:
    """Weight-normalised mean of named series. Entries with weight <= 0 are
    dropped. At each timestamp the weights are renormalised over the subsignals
    that are present (non-NaN) there, so a not-yet-warm subsignal simply doesn't
    contribute rather than NaN-ing the whole composite. All subsignals absent ->
    NaN at that timestamp."""
    active = {k: (s, float(w)) for k, (s, w) in subsignals.items() if w > 0}
    if not active:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({k: s.astype(float) for k, (s, _) in active.items()})
    w = pd.Series({k: wt for k, (_, wt) in active.items()}).reindex(frame.columns)
    present = frame.notna()
    wsum = present.mul(w, axis=1).sum(axis=1)
    num = frame.mul(w, axis=1).sum(axis=1, min_count=1)
    return (num / wsum.replace(0.0, np.nan)).rename("composite")


# ── daily cross-zone composite ───────────────────────────────────────────────

def _default_lmp_fn(zones, start, end):
    from grid_resilience.data.grid_data import fetch_lmp
    df = fetch_lmp("PJM", start, end, location_type="ZONE", use_cache=True)
    if not df.empty and "location" in df.columns:
        df = df[df["location"].isin(zones)]
    return df


def _default_load_fn(zones, start, end):
    from grid_resilience.data.grid_data import fetch_zonal_load
    df = fetch_zonal_load(start, end, use_cache=True)
    if not df.empty and "zone" in df.columns:
        df = df[df["zone"].isin(zones)]
    return df


def regime_composite(start: str, end: str, *, zones: list[str],
                     w_cong: float = config.REGIME_W_CONG,
                     w_reserve: float = config.REGIME_W_RESERVE,
                     zone_weight: str = "equal",
                     zscore_window: int = config.REGIME_ZSCORE_WINDOW,
                     zscore_minp: int = config.REGIME_ZSCORE_MINP,
                     winsor: float = config.REGIME_ZSCORE_WINSOR,
                     lmp_fn=None, load_fn=None) -> pd.Series:
    """Daily cross-zone congestion/scarcity composite z (spec s4.1 steps 1-5).

    Per zone: z-scored daily ``|congestion $|`` (weight ``w_cong``) blended with
    z-scored daily reserve-tightness = peak-load / trailing-p99-peak-load
    (weight ``w_reserve``). Zones are then combined with equal weights, or
    weights proportional to trailing mean peak load when ``zone_weight="load"``
    (falls back to equal, with a warning, when load data is unavailable).

    ``lmp_fn(zones, start, end) -> hourly LMP frame`` and
    ``load_fn(zones, start, end) -> hourly load frame`` are injectable for tests;
    the defaults read the cached PJM parquet via ``grid_resilience.data.grid_data``.
    """
    lmp_fn = lmp_fn or _default_lmp_fn
    load_fn = load_fn or _default_load_fn

    cong = _daily_zone_congestion(lmp_fn(zones, start, end))
    if cong.empty:
        warnings.warn("grid_regime: no LMP/congestion data for the window; "
                      "composite is empty (signal inactive).", RuntimeWarning, stacklevel=2)
        return pd.Series(dtype=float, name="composite")
    cong_z = cong.apply(lambda col: _trailing_zscore(col, zscore_window, zscore_minp, winsor))

    peak = pd.DataFrame()
    reserve_z = pd.DataFrame()
    if w_reserve > 0 or zone_weight == "load":
        peak = _daily_zone_peak_load(load_fn(zones, start, end))
    if w_reserve > 0:
        if peak.empty:
            warnings.warn("grid_regime: w_reserve>0 but no load data; using congestion only.",
                          RuntimeWarning, stacklevel=2)
        else:
            tight = peak / peak.rolling(zscore_window, min_periods=zscore_minp).quantile(0.99)
            reserve_z = tight.apply(
                lambda col: _trailing_zscore(col, zscore_window, zscore_minp, winsor))

    zone_cols = list(cong_z.columns)
    per_zone = {}
    for z in zone_cols:
        subs = {"cong": (cong_z[z], w_cong)}
        if z in getattr(reserve_z, "columns", []):
            subs["reserve"] = (reserve_z[z], w_reserve)
        per_zone[z] = _combine_subsignals(subs)

    if zone_weight == "load" and not peak.empty:
        wt = peak.reindex(columns=zone_cols).mean()
        wt = (wt / wt.sum()).to_dict()
    else:
        if zone_weight == "load":
            warnings.warn("grid_regime: zone_weight='load' but no load data; equal-weighting zones.",
                          RuntimeWarning, stacklevel=2)
        wt = {z: 1.0 / len(zone_cols) for z in zone_cols}

    return _combine_subsignals({z: (per_zone[z], wt[z]) for z in zone_cols}).rename("regime_composite")


# ── daily -> monthly exposure multiplier ─────────────────────────────────────

def regime_multiplier(composite: pd.Series, *, mode: str = "discrete",
                      thresh: float = config.REGIME_THRESH,
                      hi: float = config.REGIME_HI, lo: float = config.REGIME_LO,
                      k: float = config.REGIME_K,
                      month_lookback: int = config.REGIME_MONTH_LOOKBACK) -> pd.Series:
    """Daily exposure multiplier (spec s4.2 steps 6-9): smooth the composite with
    a trailing ``month_lookback``-day mean, map to a multiplier, then hold each
    month-end's verdict through the *following* calendar month (decision known at
    the prior month-end). Warm-up / missing months -> 1.0 (neutral, invested)."""
    comp = composite.astype(float).sort_index()
    if comp.empty:
        return pd.Series(dtype=float, name="regime_mult")
    smoothed = comp.rolling(month_lookback, min_periods=1).mean()

    if mode == "continuous":
        raw = (1.0 + k * smoothed).clip(0.5, 1.5)
    elif mode == "discrete":
        raw = pd.Series(1.0, index=smoothed.index)
        raw[smoothed >= thresh] = hi
        raw[smoothed <= -thresh] = lo
        raw[smoothed.isna()] = np.nan
    else:
        raise ValueError(f"mode must be 'discrete' or 'continuous', got {mode!r}")

    return _month_hold(raw, comp.index, fill=1.0).rename("regime_mult")
