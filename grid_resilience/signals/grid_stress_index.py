"""
Grid Stress Index (GSI) construction.

The GSI is a daily composite score per ISO that quantifies how much the
grid is under stress on a given day.  A higher score = more stress.

Four sub-signals (weights defined in config.GSI_WEIGHTS):

  1. lmp_zscore        — Daily max LMP normalised to a rolling z-score.
                         Captures absolute price level and volatility.

  2. congestion_frac   — Share of LMP that is the congestion component.
                         Where data are available (ERCOT, PJM), this
                         distinguishes physical scarcity from transmission
                         constraint stress.

  3. reserve_tightness — Daily max load / estimated available capacity.
                         Proxied from the load series; calibrated per ISO
                         using peak historical load as a capacity reference.

  4. event_flag        — Continuous [0,1] decay around named and algo-
                         detected stress events.

The composite GSI is a weighted average of the z-scored sub-signals,
winsorised at [-3, 3] and rescaled to [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_resilience.config import GSI_WEIGHTS, LMP_ZSCORE_WINDOW
from grid_resilience.signals.stress_events import build_event_flag_series


def build_gsi(
    daily_lmp: pd.DataFrame,
    daily_load: pd.DataFrame,
    events_df: pd.DataFrame,
    iso: str,
) -> pd.DataFrame:
    """
    Construct the daily Grid Stress Index for a single ISO.

    Parameters
    ----------
    daily_lmp   : output of grid_data.daily_lmp_summary() — indexed by date
    daily_load  : output of grid_data.daily_load_summary() — indexed by date
    events_df   : output of stress_events.build_full_event_calendar()
    iso         : ISO label (used only for logging)

    Returns
    -------
    DataFrame indexed by date with columns:
        lmp_zscore, congestion_frac_z, reserve_tightness_z,
        event_flag, gsi  (composite score, [0, 1])
    """
    if daily_lmp.empty:
        print(f"  [gsi] No LMP data for {iso} — returning empty GSI.")
        return pd.DataFrame()

    index = daily_lmp.index

    # ── 1. LMP z-score ────────────────────────────────────────────────────────
    lmp_max = daily_lmp["lmp_max"]
    roll_mean = lmp_max.rolling(LMP_ZSCORE_WINDOW, min_periods=20).mean()
    roll_std  = lmp_max.rolling(LMP_ZSCORE_WINDOW, min_periods=20).std().replace(0, np.nan)
    lmp_z = ((lmp_max - roll_mean) / roll_std).clip(-5, 5)

    # ── 2. Congestion fraction z-score ────────────────────────────────────────
    if "congestion_frac" in daily_lmp.columns and daily_lmp["congestion_frac"].notna().any():
        cf = daily_lmp["congestion_frac"].fillna(0.0)
        cf_mean = cf.rolling(LMP_ZSCORE_WINDOW, min_periods=20).mean()
        cf_std  = cf.rolling(LMP_ZSCORE_WINDOW, min_periods=20).std().replace(0, np.nan)
        cong_z  = ((cf - cf_mean) / cf_std).clip(-5, 5)
    else:
        # Fall back to lmp_zscore as congestion proxy when component data absent
        cong_z = lmp_z.copy()

    # ── 3. Reserve tightness ──────────────────────────────────────────────────
    if not daily_load.empty and "load_max_mw" in daily_load.columns:
        load_aligned = daily_load["load_max_mw"].reindex(index).ffill()
        # Capacity proxy: 110 % of the 99th-percentile historical load
        # (conservative estimate; replace with ICAP data if available)
        capacity_proxy = load_aligned.quantile(0.99) * 1.10
        reserve_ratio  = load_aligned / capacity_proxy
        rt_mean = reserve_ratio.rolling(LMP_ZSCORE_WINDOW, min_periods=20).mean()
        rt_std  = reserve_ratio.rolling(LMP_ZSCORE_WINDOW, min_periods=20).std().replace(0, np.nan)
        reserve_z = ((reserve_ratio - rt_mean) / rt_std).clip(-5, 5)
    else:
        reserve_z = pd.Series(0.0, index=index)

    # ── 4. Event flag ─────────────────────────────────────────────────────────
    event_flag = build_event_flag_series(index, events_df)

    # ── Composite GSI ─────────────────────────────────────────────────────────
    gsi_raw = (
        GSI_WEIGHTS["lmp_zscore"]        * _rescale(lmp_z)
        + GSI_WEIGHTS["congestion_frac"] * _rescale(cong_z)
        + GSI_WEIGHTS["reserve_tightness"] * _rescale(reserve_z)
        + GSI_WEIGHTS["event_flag"]       * event_flag
    )
    gsi = gsi_raw.clip(0, 1).rename("gsi")

    result = pd.DataFrame({
        "lmp_zscore":          lmp_z,
        "congestion_frac_z":   cong_z,
        "reserve_tightness_z": reserve_z,
        "event_flag":          event_flag,
        "gsi":                 gsi,
    })
    result.index.name = "date"
    return result


def build_multi_iso_gsi(
    daily_lmp_by_iso:  dict[str, pd.DataFrame],
    daily_load_by_iso: dict[str, pd.DataFrame],
    events_df:         pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build GSI for each ISO and return a dict keyed by ISO name.
    """
    return {
        iso: build_gsi(
            daily_lmp  = daily_lmp_by_iso.get(iso, pd.DataFrame()),
            daily_load = daily_load_by_iso.get(iso, pd.DataFrame()),
            events_df  = events_df,
            iso        = iso,
        )
        for iso in daily_lmp_by_iso
    }


def gsi_for_ticker(
    ticker: str,
    gsi_by_iso: dict[str, pd.DataFrame],
    ticker_iso_map: dict[str, str],
) -> pd.Series:
    """
    Return the GSI series relevant to a specific ticker based on its primary ISO.
    """
    iso = ticker_iso_map.get(ticker)
    if iso and iso in gsi_by_iso:
        return gsi_by_iso[iso]["gsi"].rename(f"gsi_{ticker}")
    return pd.Series(dtype=float, name=f"gsi_{ticker}")


def stress_days(
    gsi_series: pd.Series,
    threshold: float = 0.6,
) -> pd.DatetimeIndex:
    """Return dates where GSI >= threshold (i.e., high-stress days)."""
    return gsi_series.index[gsi_series >= threshold]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rescale(s: pd.Series, lo: float = -3.0, hi: float = 3.0) -> pd.Series:
    """Min-max rescale a winsorised series to [0, 1]."""
    clipped = s.clip(lo, hi)
    span = hi - lo
    return (clipped - lo) / span
