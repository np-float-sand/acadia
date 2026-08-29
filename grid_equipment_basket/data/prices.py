from __future__ import annotations

"""Adjusted-close price fetcher for the grid-equipment basket.

Thin wrapper over yfinance reusing grid_resilience's monthly-parquet WIDE cache
helpers (public functions only). Returns adjusted-close prices; every downstream
maths in this package works in SIMPLE returns, not log returns — see README.
"""

import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

from grid_equipment_basket.config import CACHE_DIR
from grid_resilience.data.cache_utils import (
    chunk_path,
    find_missing_months_wide,
    month_bounds,
    read_monthly_cache_wide,
    save_monthly_chunks_wide,
)

warnings.filterwarnings("ignore", category=FutureWarning)

_PRICE_CACHE_BASE = CACHE_DIR / "prices.parquet"
_RETRY_DELAY_SECONDS = 2


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    # yfinance `end` is exclusive — advance one day so `end` is included.
    end_exc = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers, start=start, end=end_exc, auto_adjust=True,
        group_by="column", progress=False, threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Close", axis=1, level=0)
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    return prices.ffill(limit=5).dropna(how="all")


def _download_with_retry(
    tickers: list[str], start: str, end: str, max_attempts: int = 2
) -> pd.DataFrame:
    """Retry once when yfinance silently drops a ticker column under load.

    A column can also be legitimately absent (ticker not listed in the window),
    which looks identical here, so retries are capped, not looped.
    """
    frame = _download(tickers, start, end)
    for _ in range(max_attempts - 1):
        if all(t in frame.columns for t in tickers):
            break
        time.sleep(_RETRY_DELAY_SECONDS)
        frame = _download(tickers, start, end)
    return frame


def fetch_prices(
    tickers: list[str], start: str, end: str,
    use_cache: bool = True, force_refresh: bool = False,
) -> pd.DataFrame:
    tickers = sorted(set(tickers))

    if use_cache and not force_refresh:
        missing = find_missing_months_wide(_PRICE_CACHE_BASE, start, end, tickers)
        if missing:
            def _fetch_month(ym: tuple[int, int]) -> None:
                ms, me = month_bounds(*ym)
                frame = _download_with_retry(tickers, ms, me)
                if not frame.empty:
                    save_monthly_chunks_wide(frame, _PRICE_CACHE_BASE)
                else:
                    p = chunk_path(_PRICE_CACHE_BASE, *ym)
                    if not p.exists():
                        pd.DataFrame().to_parquet(p, index=False)

            with ThreadPoolExecutor(max_workers=min(len(missing), 4)) as ex:
                list(ex.map(_fetch_month, missing))

        cached = read_monthly_cache_wide(_PRICE_CACHE_BASE, start, end)
        if cached.empty:
            return pd.DataFrame()
        available = [t for t in tickers if t in cached.columns]
        return cached[available].loc[start:end].sort_index()

    prices = _download(tickers, start, end)
    if use_cache and not prices.empty:
        save_monthly_chunks_wide(prices, _PRICE_CACHE_BASE)
    available = [t for t in tickers if t in prices.columns]
    return prices[available].sort_index() if available else pd.DataFrame()
