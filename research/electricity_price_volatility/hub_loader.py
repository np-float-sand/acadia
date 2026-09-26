"""
Daily hub price loader -- reads the already-cached ERCOT / PJM hourly DA-LMP
parquet files and resamples each hub to a daily mean price series.

Cache layout (grid_resilience/data/cache/):
  ercot_lmp_settlement point_YYYY-MM.parquet   columns: time, location, lmp
  pjm_lmp_DAY_AHEAD_HOURLY_YYYY-MM.parquet     columns: time, ..., location, lmp, ...

No new fetcher needed -- this module only reads local parquet.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "grid_resilience" / "data" / "cache"

_ERCOT_GLOB = str(CACHE_DIR / "ercot_lmp_settlement point_*.parquet")
_PJM_GLOB = str(CACHE_DIR / "pjm_lmp_DAY_AHEAD_HOURLY_*.parquet")


def _month_files(iso: str, start: str, end: str) -> list[str]:
    pattern = _ERCOT_GLOB if iso == "ERCOT" else _PJM_GLOB
    files = sorted(glob.glob(pattern))
    start_ym = pd.Timestamp(start).strftime("%Y-%m")
    end_ym = pd.Timestamp(end).strftime("%Y-%m")
    out = []
    for f in files:
        ym = f[-13:-8] if iso == "ERCOT" else f[-13:-8]
        # filename ends "..._YYYY-MM.parquet" -- extract the last 7 chars before ".parquet"
        ym = Path(f).stem[-7:]
        if start_ym <= ym <= end_ym:
            out.append(f)
    return out


def load_hub_hourly(iso: str, hub: str, start: str, end: str) -> pd.Series:
    """Return the raw hourly LMP series for one hub, indexed by timestamp."""
    files = _month_files(iso, start, end)
    if not files:
        return pd.Series(dtype=float)
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["time", "location", "lmp"])
        df = df[df["location"] == hub]
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.Series(dtype=float)
    full = pd.concat(frames, ignore_index=True)
    full["time"] = pd.to_datetime(full["time"])
    full = full.sort_values("time").drop_duplicates(subset="time")
    s = full.set_index("time")["lmp"]
    s = s.loc[str(pd.Timestamp(start).date()):str(pd.Timestamp(end).date())]
    return s


def load_hub_daily(iso: str, hub: str, start: str, end: str) -> pd.Series:
    """
    Resample one hub's hourly LMP to a daily mean price.

    Daily mean of the day-ahead hourly LMP is the standard way to collapse an
    hourly settlement series to one observation per day -- it is influenced by
    the full daily shape (including price spikes) without letting a single
    hour dominate the way a single settlement-period snapshot would.
    """
    hourly = load_hub_hourly(iso, hub, start, end)
    if hourly.empty:
        return hourly
    daily = hourly.resample("D").mean()
    daily = daily.dropna()
    daily.name = f"{iso}:{hub}"
    return daily


def load_all_hubs_daily(start: str, end: str, hubs: list[tuple[str, str]]) -> pd.DataFrame:
    """Return a DataFrame of daily mean price, columns = 'ISO:HUB'."""
    series = {}
    for iso, hub in hubs:
        s = load_hub_daily(iso, hub, start, end)
        if not s.empty:
            series[f"{iso}:{hub}"] = s
    return pd.DataFrame(series)
