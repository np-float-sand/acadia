from __future__ import annotations

"""
Equity price fetcher using yfinance, with local parquet caching.

Usage:
    from grid_resilience.data.equity_prices import fetch_prices, fetch_returns

    prices  = fetch_prices(["VST", "NRG", "CNP"], "2018-01-01", "2024-12-31")
    returns = fetch_returns(["VST", "NRG", "CNP"], "2018-01-01", "2024-12-31")
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from grid_resilience.config import CACHE_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

_PRICE_CACHE = CACHE_DIR / "equity_prices.parquet"


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily adjusted close prices.

    Columns = tickers, index = date (business days).
    Missing dates are forward-filled up to 5 days then dropped.

    Data source: Yahoo Finance via yfinance (free, no API key).
    """
    tickers = sorted(set(tickers))

    if use_cache and not force_refresh and _PRICE_CACHE.exists():
        cached = pd.read_parquet(_PRICE_CACHE)
        cached_tickers = set(cached.columns)
        date_ok = (
            not cached.empty
            and str(cached.index.min().date()) <= start
            and str(cached.index.max().date()) >= end
        )
        if cached_tickers >= set(tickers) and date_ok:
            return cached[tickers].loc[start:end]

    print(f"[equity] Downloading prices for {len(tickers)} tickers ({start} → {end})…")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Close", axis=1, level=0)
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    prices = prices.ffill(limit=5).dropna(how="all")

    if use_cache:
        prices.to_parquet(_PRICE_CACHE)

    return prices[tickers] if set(tickers) <= set(prices.columns) else prices


def fetch_returns(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return daily log returns (ln(P_t / P_{t-1}))."""
    prices = fetch_prices(tickers, start, end, use_cache=use_cache)
    return np.log(prices / prices.shift(1)).dropna(how="all")


def fetch_sector_return(
    tickers: list[str],
    start: str,
    end: str,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Return equal-weighted (or custom-weighted) sector return series.
    Used to demean individual stock returns in the factor layer.
    """
    returns = fetch_returns(tickers, start, end)
    if weights:
        w = pd.Series(weights).reindex(returns.columns).fillna(0)
        w = w / w.sum()
        sector = returns.mul(w.values, axis=1).sum(axis=1)
    else:
        sector = returns.mean(axis=1)
    return sector.rename("sector_return")


def normalize_prices(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Normalize each column to `base` at its first non-NaN observation."""
    first_valid = prices.apply(lambda s: s.first_valid_index())
    result = prices.copy()
    for col, idx in first_valid.items():
        if idx is not None:
            result[col] = prices[col] / prices[col].loc[idx] * base
    return result
