"""
Price fetch + effective-start truncation for the satellite-sleeve probe.

Fetcher choice: `grid_equipment_basket.data.prices.fetch_prices` is used
throughout (it is what `electrification_strategy/__main__.py` itself calls for
the base book). Unlike `grid_resilience.data.equity_prices.fetch_prices` --
the module that silently dropped ticker columns under a threaded per-month
cache warmer in `docs/electricity-price-volatility-miner-results.md` §3 --
`grid_equipment_basket.data.prices.fetch_prices` issues ONE whole-span
`yf.download` call per missing-months gap; it never fans work out across a
`ThreadPoolExecutor`, so it does not hit that bug. No new fetcher is written
here and no non-threaded workaround is needed for this probe -- confirmed by
inspecting both modules side by side before writing this file.
"""
from __future__ import annotations

import pandas as pd

from grid_equipment_basket.data.prices import fetch_prices
from research.satellite_sleeve_diversification import universe as U


def fetch_all(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Thin pass-through -- kept as a named entry point so run_probe.py has one
    place to point at if the fetcher ever needs to change."""
    return fetch_prices(sorted(set(tickers)), start, end)


def effective_start(prices: pd.DataFrame, ticker: str) -> pd.Timestamp | None:
    """First date this ticker's price series reflects the business it is held
    for today (see universe.py's EFFECTIVE_START_OVERRIDES docstring)."""
    if ticker not in prices.columns:
        return None
    override = U.EFFECTIVE_START_OVERRIDES.get(ticker)
    if override is not None:
        return pd.Timestamp(override)
    return prices[ticker].first_valid_index()


def sleeve_common_start(prices: pd.DataFrame, tickers: list[str]) -> pd.Timestamp | None:
    """The date after which EVERY name in `tickers` has genuine (business-
    relevant, post-shell) price history -- the earliest date a plain
    equal-weight basket of all of them can legitimately be formed."""
    starts = [effective_start(prices, t) for t in tickers]
    starts = [s for s in starts if s is not None]
    if len(starts) < len(tickers):
        return None
    return max(starts)
