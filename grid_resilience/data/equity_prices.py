from __future__ import annotations

"""
Equity price fetcher using yfinance, with monthly-chunked local parquet caching.

Each calendar month is stored in its own parquet file
(e.g. equity_prices_2020-01.parquet) so partial progress survives interruptions
and only missing months need to be fetched on re-runs.

Usage:
    from grid_resilience.data.equity_prices import fetch_prices, fetch_returns

    prices  = fetch_prices(["VST", "NRG", "CNP"], "2018-01-01", "2024-12-31")
    returns = fetch_returns(["VST", "NRG", "CNP"], "2018-01-01", "2024-12-31")
"""

import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

from grid_resilience.config import CACHE_DIR
from grid_resilience.data.cache_utils import (
    find_missing_months_wide,
    read_monthly_cache_wide,
    save_monthly_chunks_wide,
    month_bounds,
    chunk_path,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Base path — actual files are <stem>_YYYY-MM<suffix>
_PRICE_CACHE_BASE = CACHE_DIR / "equity_prices.parquet"


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    # yfinance `end` is exclusive; advance by one day so the requested end date is included
    end_exc = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers,
        start=start,
        end=end_exc,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Close", axis=1, level=0)
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    return prices.ffill(limit=5).dropna(how="all")


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
    Monthly parquet chunks are written as each month completes so
    interrupted runs resume from where they left off.

    Data source: Yahoo Finance via yfinance (free, no API key).
    """
    tickers = sorted(set(tickers))

    if use_cache and not force_refresh:
        missing = find_missing_months_wide(_PRICE_CACHE_BASE, start, end, tickers)

        if missing:
            def _fetch_month(ym: tuple[int, int]) -> None:
                ms, me = month_bounds(*ym)
                print(f"[equity] Downloading {len(tickers)} tickers {ms[:7]}…")
                frame = _download(tickers, ms, me)
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
        return cached[available].loc[start:end]

    print(f"[equity] Downloading prices for {len(tickers)} tickers ({start} → {end})…")
    prices = _download(tickers, start, end)
    if use_cache and not prices.empty:
        save_monthly_chunks_wide(prices, _PRICE_CACHE_BASE)
    return prices[tickers] if set(tickers) <= set(prices.columns) else prices


def fetch_returns(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
    max_log_move: float = 0.40,
) -> pd.DataFrame:
    """Return daily log returns (ln(P_t / P_{t-1})).

    max_log_move caps any single-day log return at ±0.40 (≈ ±33% simple).
    Utility stocks physically cannot move more than ~20% in a day; the cap
    guards against stale-then-corrected cache entries that create artificial
    multi-hundred-percent spikes in the backtest.
    """
    prices = fetch_prices(tickers, start, end, use_cache=use_cache)
    log_ret = np.log(prices / prices.shift(1))

    extreme = log_ret.abs() > max_log_move
    if extreme.any(axis=None):
        flagged = [(col, str(idx.date())) for col in log_ret.columns
                   for idx in log_ret.index[extreme[col]]]
        if flagged:
            import warnings
            warnings.warn(
                f"[equity] Clipped {len(flagged)} extreme daily returns "
                f"(|log| > {max_log_move}): {flagged[:5]}{'...' if len(flagged) > 5 else ''}",
                RuntimeWarning, stacklevel=2,
            )
        log_ret = log_ret.clip(-max_log_move, max_log_move)

    return log_ret.dropna(how="all")


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


def fetch_icr(
    tickers: list[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch quarterly interest coverage ratio (EBIT / |Interest Expense|) per ticker.

    Returns a DataFrame indexed by quarter-end date with ticker columns.
    Values are clipped to [0, 20] — a cap of 20x is applied so zero-debt names
    don't dominate the cross-section. NaN where financials are unavailable.

    Callers should apply a ~45-day reporting lag (SEC 10-Q deadline for large
    accelerated filers) before treating a reading as observable.

    Data source: Yahoo Finance via yfinance (free, no API key).
    Depth: typically 4–12 quarters of history per ticker.
    """
    key = hashlib.md5("-".join(sorted(tickers)).encode()).hexdigest()[:12]
    cache_file = CACHE_DIR / f"icr_{key}.parquet"
    today = pd.Timestamp.now().normalize()

    if use_cache and cache_file.exists():
        age_days = (today - pd.Timestamp(cache_file.stat().st_mtime, unit="s").normalize()).days
        if age_days < 1:
            return pd.read_parquet(cache_file)

    frames: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            fin = yf.Ticker(t).quarterly_financials
            if fin is None or fin.empty:
                continue

            ebit_row = next(
                (r for r in fin.index if "ebit" in r.lower()
                 and "ebitda" not in r.lower()),
                None,
            )
            if ebit_row is None:
                ebit_row = next(
                    (r for r in fin.index if "operating income" in r.lower()), None
                )
            int_row = next(
                (r for r in fin.index if "interest expense" in r.lower()), None
            )
            if ebit_row is None or int_row is None:
                continue

            ebit = pd.to_numeric(fin.loc[ebit_row], errors="coerce")
            interest = pd.to_numeric(fin.loc[int_row], errors="coerce").abs()
            icr = (ebit / interest.replace(0, float("nan"))).clip(0, 20)
            frames[t] = icr
        except Exception as exc:
            print(f"  [icr] {t} fetch failed: {exc}")

    if not frames:
        print("  [icr] No ICR data retrieved.")
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index, utc=True).tz_localize(None)
    result = result.sort_index()

    if use_cache:
        result.to_parquet(cache_file)
    return result


def normalize_prices(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Normalize each column to `base` at its first non-NaN observation."""
    first_valid = prices.apply(lambda s: s.first_valid_index())
    result = prices.copy()
    for col, idx in first_valid.items():
        if idx is not None:
            result[col] = prices[col] / prices[col].loc[idx] * base
    return result
