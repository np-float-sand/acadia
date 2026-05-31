from __future__ import annotations

import calendar
import threading
from pathlib import Path

import pandas as pd

# Lock protecting concurrent writes to the same monthly chunk file.
# Different months write to different files so are inherently safe;
# this lock only matters if the same month is written twice concurrently
# (shouldn't happen in practice but cheap to guard).
_CHUNK_WRITE_LOCK = threading.Lock()


def read_cached(path: Path) -> pd.DataFrame:
    """Read a parquet file; return an empty DataFrame if it doesn't exist."""
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def compute_date_gaps(
    cached_df: pd.DataFrame,
    time_col: str,
    requested_start: str,
    requested_end: str,
) -> list[tuple[str, str]]:
    """
    Return date ranges in [requested_start, requested_end] not covered by cached_df.

    Returns up to two gaps: a head gap if requested_start precedes cached data,
    and a tail gap if requested_end extends beyond cached data.
    Returns [(requested_start, requested_end)] when cached_df is empty.
    """
    if cached_df.empty or time_col not in cached_df.columns:
        return [(requested_start, requested_end)]

    times = pd.to_datetime(cached_df[time_col])
    cached_min = times.min().strftime("%Y-%m-%d")
    cached_max = times.max().strftime("%Y-%m-%d")

    gaps = []
    if requested_start < cached_min:
        gaps.append((requested_start, cached_min))
    if requested_end > cached_max:
        gaps.append((cached_max, requested_end))
    return gaps


def merge_and_save(
    cached_df: pd.DataFrame,
    new_df: pd.DataFrame,
    time_col: str,
    path: Path,
) -> pd.DataFrame:
    """
    Concatenate cached and new data, drop completely duplicate rows (keeping the
    newer fetch), sort by time_col, write to path, and return the merged DataFrame.
    """
    combined = pd.concat([cached_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(keep="last")
    combined = combined.sort_values(time_col).reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined


# ── Monthly chunk helpers ─────────────────────────────────────────────────────

def monthly_chunks(start: str, end: str) -> list[tuple[int, int]]:
    """Return (year, month) pairs covering every month in [start, end]."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    result: list[tuple[int, int]] = []
    cur = s.replace(day=1)
    while cur <= e:
        result.append((cur.year, cur.month))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return result


def chunk_path(base_path: Path, year: int, month: int) -> Path:
    """Return the parquet path for a single monthly chunk."""
    return base_path.parent / f"{base_path.stem}_{year:04d}-{month:02d}{base_path.suffix}"


def month_bounds(year: int, month: int) -> tuple[str, str]:
    """Return (first_day, last_day) strings for the given year/month."""
    _, last = calendar.monthrange(year, month)
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last:02d}"


# ── Long-format monthly cache (grid data: time column in body) ────────────────

def find_missing_months(base_path: Path, start: str, end: str) -> list[tuple[int, int]]:
    """Return (year, month) pairs whose chunk files are absent from disk."""
    return [
        (y, m) for y, m in monthly_chunks(start, end)
        if not chunk_path(base_path, y, m).exists()
    ]


def read_monthly_cache(base_path: Path, start: str, end: str) -> pd.DataFrame:
    """Read and concatenate all existing monthly chunk files overlapping [start, end]."""
    frames = [
        pd.read_parquet(chunk_path(base_path, y, m))
        for y, m in monthly_chunks(start, end)
        if chunk_path(base_path, y, m).exists()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_monthly_chunks(df: pd.DataFrame, base_path: Path, time_col: str) -> None:
    """Split df by calendar month and write each month to its chunk file."""
    if df.empty:
        return
    df = df.copy()
    times = pd.to_datetime(df[time_col])
    df["_year"] = times.dt.year
    df["_month"] = times.dt.month
    for (y, m), grp in df.groupby(["_year", "_month"]):
        grp = grp.drop(columns=["_year", "_month"])
        p = chunk_path(base_path, int(y), int(m))
        with _CHUNK_WRITE_LOCK:
            if p.exists():
                existing = pd.read_parquet(p)
                grp = (
                    pd.concat([existing, grp], ignore_index=True)
                    .drop_duplicates(keep="last")
                    .sort_values(time_col)
                    .reset_index(drop=True)
                )
            grp.to_parquet(p, index=False)


# ── Wide-format monthly cache (equity prices: date-indexed, tickers as cols) ──

def find_missing_months_wide(
    base_path: Path, start: str, end: str, tickers: list[str]
) -> list[tuple[int, int]]:
    """
    Return (year, month) pairs whose chunk files are absent or lack any of tickers.
    """
    missing = []
    for y, m in monthly_chunks(start, end):
        p = chunk_path(base_path, y, m)
        if not p.exists():
            missing.append((y, m))
        else:
            existing_cols = pd.read_parquet(p).columns.tolist()
            if not all(t in existing_cols for t in tickers):
                missing.append((y, m))
    return missing


def read_monthly_cache_wide(base_path: Path, start: str, end: str) -> pd.DataFrame:
    """Read and concatenate wide monthly chunk files; returns date-indexed DataFrame."""
    frames = [
        pd.read_parquet(chunk_path(base_path, y, m))
        for y, m in monthly_chunks(start, end)
        if chunk_path(base_path, y, m).exists()
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def save_monthly_chunks_wide(df: pd.DataFrame, base_path: Path) -> None:
    """Split wide (date-indexed) DataFrame by month and write each chunk."""
    if df.empty:
        return
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    for period, grp in df.groupby(df.index.to_period("M")):
        p = chunk_path(base_path, period.year, period.month)
        with _CHUNK_WRITE_LOCK:
            if p.exists():
                existing = pd.read_parquet(p)
                grp = pd.concat([existing, grp]).groupby(level=0).last().sort_index()
            grp.to_parquet(p)
