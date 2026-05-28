"""
Grid stress event detection — two complementary approaches:

1. Named events  : curated calendar of known extreme events (from utility_node_map)
2. LMP-derived   : algorithmic detection from daily LMP summary statistics

The combined output is a DataFrame of stress windows, one row per event,
used by:
  - grid_stress_index.py  (event_flag component)
  - conditional_beta.py   (event study windows)
  - backtest.py           (stress-period performance attribution)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_resilience.data.utility_node_map import STRESS_EVENTS


# ── Named event helpers ───────────────────────────────────────────────────────

def get_named_events(isos: list[str] | None = None) -> pd.DataFrame:
    """
    Return the curated stress event calendar as a tidy DataFrame.

    Columns: name, type, window_start, window_end, context_start, context_end, iso_list
    """
    rows = []
    for ev in STRESS_EVENTS:
        if isos and not any(i in ev["iso"] for i in isos):
            continue
        rows.append({
            "name":          ev["name"],
            "type":          ev["type"],
            "window_start":  pd.Timestamp(ev["window"][0]),
            "window_end":    pd.Timestamp(ev["window"][1]),
            "context_start": pd.Timestamp(ev["context"][0]),
            "context_end":   pd.Timestamp(ev["context"][1]),
            "iso_list":      ev["iso"],
            "notes":         ev.get("notes", ""),
        })
    return pd.DataFrame(rows)


def event_date_mask(
    dates: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    use_context: bool = False,
) -> pd.Series:
    """
    Return a boolean Series (indexed by `dates`) that is True on days
    falling inside any event window.

    Parameters
    ----------
    use_context : if True, use the wider context window instead of the core window
    """
    mask = pd.Series(False, index=dates)
    start_col = "context_start" if use_context else "window_start"
    end_col   = "context_end"   if use_context else "window_end"
    for _, row in events_df.iterrows():
        mask |= (dates >= row[start_col]) & (dates <= row[end_col])
    return mask


# ── LMP-based event detection ─────────────────────────────────────────────────

def detect_lmp_spike_events(
    daily_lmp: pd.DataFrame,
    iso: str,
    spike_pct: float = 0.95,
    min_duration_days: int = 1,
    merge_gap_days: int = 2,
) -> pd.DataFrame:
    """
    Identify stress windows where LMP exceeded the `spike_pct` percentile
    of the full historical distribution.

    Parameters
    ----------
    daily_lmp       : output of grid_data.daily_lmp_summary() — indexed by date
    iso             : ISO label (used for metadata)
    spike_pct       : percentile threshold (0–1) above which a day is a spike day
    min_duration_days : minimum consecutive spike days to qualify as an event
    merge_gap_days  : merge events separated by fewer than this many days

    Returns DataFrame with same schema as get_named_events().
    """
    if daily_lmp.empty or "lmp_max" not in daily_lmp.columns:
        return pd.DataFrame()

    threshold = daily_lmp["lmp_max"].quantile(spike_pct)
    spike_days = daily_lmp.index[daily_lmp["lmp_max"] >= threshold]

    if spike_days.empty:
        return pd.DataFrame()

    # Group consecutive spike days into events
    windows = _group_consecutive(spike_days, gap_days=merge_gap_days)

    # Filter by minimum duration
    windows = [(s, e) for s, e in windows if (e - s).days + 1 >= min_duration_days]

    rows = []
    for start, end in windows:
        rows.append({
            "name":          f"{iso} LMP Spike {start.date()}",
            "type":          "lmp_spike",
            "window_start":  start,
            "window_end":    end,
            "context_start": start - pd.Timedelta(days=14),
            "context_end":   end   + pd.Timedelta(days=14),
            "iso_list":      [iso],
            "notes":         f"Algo-detected: lmp_max ≥ {threshold:.0f} $/MWh",
        })
    return pd.DataFrame(rows)


def detect_congestion_events(
    daily_lmp: pd.DataFrame,
    iso: str,
    congestion_threshold: float = 0.30,
    min_duration_days: int = 3,
) -> pd.DataFrame:
    """
    Identify periods of persistent high congestion (congestion_frac above threshold).
    These are the subtler transmission stress events referenced in the pitch.

    Requires that daily_lmp contains a 'congestion_frac' column (available for
    ERCOT and PJM where LMP components are published; NaN for MISO/CAISO/SPP).
    """
    if daily_lmp.empty or "congestion_frac" not in daily_lmp.columns:
        return pd.DataFrame()

    cong = daily_lmp["congestion_frac"].dropna()
    if cong.empty:
        return pd.DataFrame()

    high_days = cong.index[cong >= congestion_threshold]
    if high_days.empty:
        return pd.DataFrame()

    windows = _group_consecutive(high_days, gap_days=1)
    windows = [(s, e) for s, e in windows if (e - s).days + 1 >= min_duration_days]

    rows = []
    for start, end in windows:
        rows.append({
            "name":          f"{iso} Congestion {start.date()}",
            "type":          "congestion",
            "window_start":  start,
            "window_end":    end,
            "context_start": start - pd.Timedelta(days=7),
            "context_end":   end   + pd.Timedelta(days=7),
            "iso_list":      [iso],
            "notes":         f"Algo-detected: congestion_frac ≥ {congestion_threshold:.0%}",
        })
    return pd.DataFrame(rows)


def build_full_event_calendar(
    daily_lmp_by_iso: dict[str, pd.DataFrame],
    isos: list[str] | None = None,
) -> pd.DataFrame:
    """
    Combine named events + algo-detected events into one deduplicated calendar.

    Parameters
    ----------
    daily_lmp_by_iso : {iso: daily_lmp_df} from grid_data.daily_lmp_summary()
    isos             : filter to these ISOs; None = all

    Returns sorted DataFrame of all stress event windows.
    """
    frames = [get_named_events(isos=isos)]

    for iso, df in daily_lmp_by_iso.items():
        if isos and iso not in isos:
            continue
        frames.append(detect_lmp_spike_events(df, iso))
        frames.append(detect_congestion_events(df, iso))

    calendar = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    calendar = calendar.sort_values("window_start").reset_index(drop=True)
    return calendar


# ── Event-flag time series ────────────────────────────────────────────────────

def build_event_flag_series(
    dates: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    decay_days: int = 5,
) -> pd.Series:
    """
    Build a continuous [0, 1] event-flag series for use in the Grid Stress Index.

    On days inside an event window the flag = 1.0.
    Outside, it decays exponentially from the nearest window edge with a
    half-life of `decay_days` days.
    """
    flag = pd.Series(0.0, index=dates)
    for _, row in events_df.iterrows():
        start = row["window_start"]
        end   = row["window_end"]
        for date in dates:
            if start <= date <= end:
                flag[date] = 1.0
            else:
                dist = min(abs((date - start).days), abs((date - end).days))
                decay = np.exp(-np.log(2) * dist / decay_days)
                flag[date] = max(flag[date], decay)
    return flag.rename("event_flag")


# ── Utility ───────────────────────────────────────────────────────────────────

def _group_consecutive(
    dates: pd.DatetimeIndex,
    gap_days: int = 1,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group a sorted DatetimeIndex into (start, end) tuples of consecutive runs."""
    if len(dates) == 0:
        return []
    groups: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = end = dates[0]
    for d in dates[1:]:
        if (d - end).days <= gap_days:
            end = d
        else:
            groups.append((start, end))
            start = end = d
    groups.append((start, end))
    return groups
