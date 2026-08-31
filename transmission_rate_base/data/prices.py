from __future__ import annotations

import pandas as pd

from grid_resilience.data.equity_prices import fetch_prices


def _fetch_prices(tickers: list[str], start: str, end: str, offline: bool) -> pd.DataFrame:
    """Isolated call into the grid_resilience yfinance/parquet cache so tests can
    monkeypatch it. ``offline`` is threaded through for callers/tests; the
    underlying cache returns immediately on a hit and only tries the network on a
    miss (which report.run_pipeline catches)."""
    return fetch_prices(sorted(set(tickers)), start, end, use_cache=True)


def daily_prices(tickers, start, end, *, offline: bool = False) -> pd.DataFrame:
    px = _fetch_prices(list(tickers), start, end, offline)
    cols = [t for t in tickers if t in px.columns]
    return px[cols].loc[start:end]


def daily_returns(tickers, start, end, *, offline: bool = False) -> pd.DataFrame:
    return daily_prices(tickers, start, end, offline=offline).pct_change().dropna(how="all")


def monthly_returns(tickers, start, end, *, offline: bool = False) -> pd.DataFrame:
    px = daily_prices(tickers, start, end, offline=offline)
    return px.resample("ME").last().pct_change().dropna(how="all")
