from __future__ import annotations

"""
Data-center load growth signal for the regulated-utility factor path.

Proxies data-center demand via co-located generation interconnection activity
in a utility's PJM zones (PJM has no separate load-interconnection queue —
data centers connect at the distribution level through the local utility).
See docs/superpowers/specs/2026-07-06-dc-load-signal-design.md.
"""

import pandas as pd

from grid_resilience.config import (
    CACHE_DIR,
    DC_QUEUE_MW_MIN,
    DC_QUEUE_PROJECT_TYPES,
    DC_MOMENTUM_WINDOW_DAYS,
    DC_LEVEL_WEIGHT,
    DC_MOMENTUM_WEIGHT,
)

# PJM interconnection queue "Transmission Owner" strings that don't uppercase
# to their matching TICKER_NODE_MAP zone name.
_TO_ZONE_OVERRIDES = {
    "DOMINION": "DOM",
}

_QUEUE_CACHE_FILE = CACHE_DIR / "pjm_interconnection_queue.parquet"


def normalize_transmission_owner(to) -> str | None:
    """Map a raw PJM 'Transmission Owner' string to a TICKER_NODE_MAP zone name."""
    if not isinstance(to, str):
        return None
    first = to.split(";")[0].strip()
    upper = first.upper()
    return _TO_ZONE_OVERRIDES.get(upper, upper)


def _clean_queue(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw PJM interconnection queue rows into the columns used by
    queued_mw()/in_queue(). Filters on static (non-date) fields only —
    Status is deliberately NOT filtered here (see design spec: it has no
    associated date, so filtering on it would corrupt point-in-time
    reconstruction for historical dates).
    """
    df = raw.copy()
    df["zone"] = df["Transmission Owner"].apply(normalize_transmission_owner)
    df["mw_capacity"] = pd.to_numeric(df["MW Capacity"], errors="coerce")
    df["submitted_date"] = pd.to_datetime(df["Submitted Date"], errors="coerce")
    df["withdrawal_date"] = pd.to_datetime(df["Withdrawal Date"], errors="coerce")
    df["actual_in_service_date"] = pd.to_datetime(df["Actual In Service Date"], errors="coerce")

    mask = (
        df["zone"].notna()
        & df["mw_capacity"].notna()
        & (df["mw_capacity"] >= DC_QUEUE_MW_MIN)
        & df["Project Type"].isin(DC_QUEUE_PROJECT_TYPES)
        & df["submitted_date"].notna()
    )
    cols = ["zone", "mw_capacity", "submitted_date", "withdrawal_date", "actual_in_service_date"]
    return df.loc[mask, cols].reset_index(drop=True)


def _load_raw_queue() -> pd.DataFrame:
    """Pull PJM's raw interconnection queue Excel export via gridstatus.
    Uses get_raw_interconnection_queue(), NOT get_interconnection_queue() —
    the latter raises AssertionError on gridstatus==0.34.0 (PJM removed a
    'Revised In Service Date' column this gridstatus version still expects)."""
    import os
    import gridstatus

    pjm = gridstatus.PJM(api_key=os.environ.get("PJM_API_KEY"))
    raw_bytes = pjm.get_raw_interconnection_queue()
    return pd.read_excel(raw_bytes)


def fetch_interconnection_queue(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch and clean the PJM interconnection queue.
    Cached for 1 day (queue updates weekly; daily freshness is more than enough).
    """
    today = pd.Timestamp.now().normalize()
    if use_cache and _QUEUE_CACHE_FILE.exists():
        age_days = (today - pd.Timestamp(_QUEUE_CACHE_FILE.stat().st_mtime, unit="s").normalize()).days
        if age_days < 1:
            return pd.read_parquet(_QUEUE_CACHE_FILE)

    cleaned = _clean_queue(_load_raw_queue())
    if use_cache:
        cleaned.to_parquet(_QUEUE_CACHE_FILE)
    return cleaned
