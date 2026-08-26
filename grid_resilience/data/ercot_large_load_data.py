from __future__ import annotations

from pathlib import Path

import pandas as pd

_SEED_DIR = Path(__file__).parent / "seed"
_DEFAULT_SEED_PATH = _SEED_DIR / "ercot_large_load_monthly.csv"


def load_ercot_large_load_seed(path: Path | None = None) -> pd.DataFrame:
    """Load the checked-in ERCOT large-load seed CSV (see Tasks 4-5 for why
    this is a maintained seed file, not a live fetcher)."""
    df = pd.read_csv(path or _DEFAULT_SEED_PATH)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def compute_ercot_signal(
    tickers: list[str],
    tsp_map: dict[str, str],
    ercot_history: pd.DataFrame,
    as_of_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Raw (pre-z-score) ERCOT large-load score per ticker per date: latest
    known TSP-level total_mw as of `as_of` (level only in v1 — no
    zone-size denominator exists yet for ERCOT TSPs, unlike the PJM
    signal's zone_size()). Only tickers in `tsp_map` get real values.
    """
    tsp_rows = ercot_history[ercot_history["scope"] == "tsp_level"]
    rows = {}
    for date in as_of_dates:
        row = {}
        for ticker in tickers:
            tsp = tsp_map.get(ticker)
            if tsp is None:
                row[ticker] = float("nan")
                continue
            known = tsp_rows[(tsp_rows["tsp"] == tsp) & (tsp_rows["snapshot_date"] <= date)]
            row[ticker] = float(known.sort_values("snapshot_date")["total_mw"].iloc[-1]) if not known.empty else float("nan")
        rows[date] = row
    return pd.DataFrame.from_dict(rows, orient="index")[tickers]
