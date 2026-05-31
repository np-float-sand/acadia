from __future__ import annotations

"""
ERCOT historical LMP downloader using gridstatus annual archive methods.

gridstatus.Ercot.get_dam_spp(year) downloads ERCOT Data Product NP4-180-ER
(Historical DAM Load Zone and Hub Prices) as an annual Excel archive covering
Load Zones and Hubs from 2011 onwards.  One file per year (~12 MB) is far
more efficient than one API call per month.

This module is invoked by grid_data.fetch_lmp() whenever the requested
date range pre-dates the live API's ~90-day rolling window.
"""

from pathlib import Path

import pandas as pd

# Days of history available via the live gridstatus get_lmp() API.
# Months older than this are fetched from annual bulk archives instead.
_LIVE_API_WINDOW_DAYS = 90


def is_historical(date_str: str) -> bool:
    """Return True if date_str is older than the live API's rolling window."""
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=_LIVE_API_WINDOW_DAYS)
    return pd.Timestamp(date_str) < cutoff


def backfill_lmp(
    start: str,
    end: str,
    cache_base: Path,
) -> None:
    """
    Populate monthly LMP cache chunks for ERCOT historical date ranges.

    Uses gridstatus.Ercot.get_dam_spp(year) which hits ERCOT's public CDR
    API — no B2C authentication required.  One annual Excel archive (~12 MB)
    covers all 12 months, so we download once per year rather than once per
    month.  Already-cached months are skipped; no redundant requests.

    Falls back to RTM 15-min data (resampled to hourly) if DAM unavailable.
    """
    try:
        import gridstatus
    except ImportError:
        raise ImportError("gridstatus is not installed.  Run: pip install gridstatus")

    from grid_resilience.data.cache_utils import monthly_chunks, chunk_path, save_monthly_chunks

    missing = [
        (y, m) for y, m in monthly_chunks(start, end)
        if not chunk_path(cache_base, y, m).exists()
    ]
    if not missing:
        return

    # CDR API is public — no token needed
    ercot = gridstatus.Ercot()

    years_needed = sorted({y for y, _ in missing})
    missing_set = set(missing)

    for year in years_needed:
        df = _download_year(ercot, year)
        if df is None or df.empty:
            continue

        # Save only the months that were actually missing for this year
        times = pd.to_datetime(df["time"])
        missing_months_in_year = {m for y, m in missing_set if y == year}
        mask = (times.dt.year == year) & times.dt.month.isin(missing_months_in_year)
        subset = df[mask].reset_index(drop=True)
        if not subset.empty:
            save_monthly_chunks(subset, cache_base, "time")
            print(
                f"  [ercot_bulk] {year}: saved {len(missing_months_in_year)} months,"
                f" {len(subset):,} rows"
            )


def _download_year(ercot, year: int) -> pd.DataFrame | None:
    """Download one year of ERCOT SPP data; DAM first, RTM 15-min as fallback."""
    try:
        print(f"[ercot_bulk] Downloading DAM SPP archive {year}…")
        df = ercot.get_dam_spp(year=year, verbose=False)
        if not df.empty:
            return _normalize(df, resample_to_hourly=False)
    except Exception as exc:
        print(f"  [ercot_bulk] DAM SPP {year} failed: {exc}")

    try:
        print(f"[ercot_bulk] Falling back to RTM SPP archive {year}…")
        df = ercot.get_rtm_spp(year=year, verbose=False)
        if not df.empty:
            return _normalize(df, resample_to_hourly=True)
    except Exception as exc:
        print(f"  [ercot_bulk] RTM SPP {year} also failed: {exc}")

    return None


def _normalize(df: pd.DataFrame, *, resample_to_hourly: bool) -> pd.DataFrame:
    """
    Map gridstatus DAM/RTM SPP columns to the standard schema (time, location, lmp).

    Expected input columns (gridstatus output):
        Interval Start, Interval End, Location, Location Type, Market, SPP
    """
    rename: dict[str, str] = {}

    # Time column — prefer "Interval Start" over bare "Time"
    for c in ("Interval Start", "Time"):
        if c in df.columns:
            rename[c] = "time"
            break

    if "Location" in df.columns:
        rename["Location"] = "location"

    for c in ("SPP", "Settlement Point Price"):
        if c in df.columns:
            rename[c] = "lmp"
            break

    result = df.rename(columns=rename)
    keep = [c for c in ("time", "location", "lmp") if c in result.columns]
    result = result[keep].copy()

    if "time" in result.columns:
        result["time"] = pd.to_datetime(result["time"])
        # Strip timezone so comparisons with plain date strings work consistently
        if result["time"].dt.tz is not None:
            result["time"] = result["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    if resample_to_hourly and "time" in result.columns and "lmp" in result.columns:
        # RTM is 15-min — resample to hourly by averaging to match DAM resolution
        result["time"] = result["time"].dt.floor("h")
        group_cols = [c for c in ("time", "location") if c in result.columns]
        result = result.groupby(group_cols)["lmp"].mean().reset_index()

    return result
