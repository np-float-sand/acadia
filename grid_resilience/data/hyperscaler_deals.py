from __future__ import annotations

from pathlib import Path

import pandas as pd

_SEED_DIR = Path(__file__).parent / "seed"
_DEFAULT_SEED_PATH = _SEED_DIR / "hyperscaler_deals.csv"


def load_hyperscaler_deals(path: Path | None = None, include_unverified: bool = False) -> pd.DataFrame:
    """
    Load the curated hyperscaler-PPA seed table. Excludes
    date_confidence='needs_verification' rows by default — see Task 7
    Step 2; pass include_unverified=True only for exploratory analysis,
    never for a result reported as final.
    """
    df = pd.read_csv(path or _DEFAULT_SEED_PATH)
    df["first_disclosure_date"] = pd.to_datetime(df["first_disclosure_date"])
    if not include_unverified:
        df = df[df["date_confidence"] == "confirmed_primary_source"]
    return df.reset_index(drop=True)


def compute_hyperscaler_signal(
    tickers: list[str],
    deals: pd.DataFrame,
    as_of_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Raw (pre-z-score) cumulative disclosed DC-linked MW under contract per
    ticker, as of each date: running sum of sign*mw across all rows up to
    that date. `event_type` is a descriptive label only, not read by this
    function — a `regulatory_setback` row's sign=-1 subtracts its own mw,
    but nothing here assumes later rows "resolve" it or that adjacent rows
    for the same ticker form a clean +/-/+ cycle of one MW figure. Real
    deal chains can be several distinct, differently-sized instruments in
    sequence (see grid_resilience/data/seed/hyperscaler_deals.csv's TLN
    rows) — this function sums whatever `sign*mw` values it's given, and
    it's each row's own accuracy (not this function) that has to earn that.

    NaN vs 0.0 semantics (changed 2026-08-26, whole-branch review finding #3
    — the previous behaviour returned 0.0 in BOTH cases below):

      * NaN  = this layer has no information about the ticker on this date.
               Either the ticker appears nowhere in `deals` at all, or the
               date is strictly before that ticker's earliest
               `first_disclosure_date`. "No disclosed deal had happened
               yet" is an absence of information, not a measured zero.
      * 0.0  = the ticker IS disclosed as of this date and its running
               sign*mw sum genuinely nets to zero (e.g. a setback row fully
               offsets an earlier announcement). This is a real measurement.

    The distinction matters downstream: dc_demand_combined.apply_layer_precedence()
    claims a (date, ticker) cell for this layer only where the value is
    non-NaN, so returning 0.0 pre-disclosure made this layer claim (and
    thereby displace the PJM generation-queue layer's real, dispersed data
    for) every date back to 2018 for CEG/TLN, on the strength of deals that
    had not been announced yet.
    """
    if deals.empty:
        first_disclosure = pd.Series(dtype="datetime64[ns]")
    else:
        first_disclosure = deals.groupby("ticker")["first_disclosure_date"].min()

    rows = {}
    for date in as_of_dates:
        row = {}
        for ticker in tickers:
            if ticker not in first_disclosure.index:
                row[ticker] = float("nan")          # ticker not covered by this layer at all
                continue
            if date < first_disclosure[ticker]:
                row[ticker] = float("nan")          # covered ticker, but nothing disclosed yet as of `date`
                continue
            ticker_deals = deals[(deals["ticker"] == ticker) & (deals["first_disclosure_date"] <= date)]
            row[ticker] = float((ticker_deals["sign"] * ticker_deals["mw"]).sum())
        rows[date] = row
    return pd.DataFrame.from_dict(rows, orient="index")[tickers].astype(float)
