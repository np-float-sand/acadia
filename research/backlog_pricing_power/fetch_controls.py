"""
Length/size-control fetcher for docs/backlog-pricing-power-length-control-
preregistration.md §2. Does NOT touch filing_scores.parquet or any already-
scored data -- this is a new, separate join.

Control 1 (filing word count, `n_words`) needs no fetch: it's already a
column in _cache/filing_scores.parquet, joined in directly by residualize.py.

Control 2 (log point-in-time market cap) needs two new fetches, both cached:
  1. SEC EDGAR XBRL `companyfacts` (not `companyconcept` -- see pre-reg §2.2,
     that endpoint silently returns empty `units` for HUBB/JCI even though
     the same tag has full data via companyfacts for the same CIK) for the
     `dei:EntityCommonStockSharesOutstanding` tag: the cover-page shares-
     outstanding figure every 10-K/10-Q tags, one fact per filing, indexed by
     that filing's own accession number.
  2. Raw (NOT split/dividend-adjusted) daily close price per ticker via
     yfinance (`auto_adjust=False`) -- deliberately not the project's
     standard adjusted-close cache (grid_equipment_basket.data.prices),
     because adjusted close would mismatch a contemporaneous point-in-time
     shares-outstanding figure for any ticker with a split in the window
     (5 of 27 names did: APG, IESC, JCI, POWL, TT -- see pre-reg §2.2).

Output: _cache/filing_controls.parquet, one row per (ticker, accessionNumber)
already present in filing_scores.parquet, columns:
    ticker, accessionNumber, filingDate, shares_out, price_at_filing,
    market_cap, log_market_cap

Not production code. Throwaway research script. Run:
    python3 research/backlog_pricing_power/fetch_controls.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from cik_map import CIK_BY_TICKER, SCREEN_27

SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
_CACHE = Path(__file__).parent / "_cache"
_CONTROLS_CACHE = _CACHE / "controls"
_CONTROLS_CACHE.mkdir(parents=True, exist_ok=True)

SCORES_PATH = _CACHE / "filing_scores.parquet"
RAW_PX_CACHE = _CACHE / "raw_close_prices.parquet"
OUT_PATH = _CACHE / "filing_controls.parquet"

SHARES_TAG = "EntityCommonStockSharesOutstanding"


def _get(url: str, timeout: int = 30) -> requests.Response:
    r = None
    for i in range(5):
        r = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
        if r.status_code == 429:
            time.sleep(1.5 * (i + 1))
            continue
        return r
    return r


def fetch_shares_outstanding(ticker: str, cik: str) -> pd.DataFrame:
    """
    Per-filing shares outstanding via EDGAR XBRL companyfacts (cached).
    Columns: accessionNumber, shares_out. One row per accession number that
    tags the dei:EntityCommonStockSharesOutstanding fact.
    """
    cache_path = _CONTROLS_CACHE / f"{ticker}_shares.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    cik10 = f"{int(cik):010d}"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    r = _get(url, timeout=60)
    rows = []
    if r is not None and r.status_code == 200:
        data = r.json()
        node = data.get("facts", {}).get("dei", {}).get(SHARES_TAG)
        if node:
            facts = node.get("units", {}).get("shares", [])
            for f in facts:
                accn = f.get("accn")
                val = f.get("val")
                if accn and val is not None:
                    rows.append({"accessionNumber": accn, "shares_out": float(val)})
    df = pd.DataFrame(rows).drop_duplicates("accessionNumber")
    df.to_parquet(cache_path)
    time.sleep(0.2)
    return df


def fetch_raw_close_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Raw (unadjusted) daily close prices, wide DataFrame, cached to disk."""
    if RAW_PX_CACHE.exists():
        return pd.read_parquet(RAW_PX_CACHE)

    end_exc = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers, start=start, end=end_exc, auto_adjust=False,
        group_by="column", progress=False, threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw.xs("Close", axis=1, level=0)
    else:
        px = raw[["Close"]] if "Close" in raw.columns else raw
    px = px.dropna(how="all")
    px.to_parquet(RAW_PX_CACHE)
    return px


def price_at_or_before(px: pd.DataFrame, ticker: str, date: pd.Timestamp,
                        max_lookback_days: int = 5) -> float | None:
    """Raw close on the nearest trading day at-or-before `date`."""
    if ticker not in px.columns:
        return None
    window = px[ticker].loc[:date].dropna()
    if window.empty:
        return None
    last_date = window.index[-1]
    if (date - last_date).days > max_lookback_days:
        return None
    return float(window.iloc[-1])


def main() -> None:
    scores = pd.read_parquet(SCORES_PATH)
    scores["filingDate"] = pd.to_datetime(scores["filingDate"])

    raw_px = fetch_raw_close_prices(SCREEN_27, "2016-06-01", "2026-08-31")

    frames = []
    for t in SCREEN_27:
        print(f"controls: {t} ...", flush=True)
        shares = fetch_shares_outstanding(t, CIK_BY_TICKER[t])
        sub = scores[scores.ticker == t][["ticker", "accessionNumber", "filingDate"]].copy()
        sub = sub.merge(shares, on="accessionNumber", how="left")
        sub["price_at_filing"] = sub["filingDate"].apply(
            lambda d: price_at_or_before(raw_px, t, d))
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)
    out["market_cap"] = out["shares_out"] * out["price_at_filing"]
    out["log_market_cap"] = np.log(out["market_cap"].where(out["market_cap"] > 0))
    out.to_parquet(OUT_PATH)

    n = len(out)
    n_shares = out["shares_out"].notna().sum()
    n_px = out["price_at_filing"].notna().sum()
    n_both = out["log_market_cap"].notna().sum()
    print(f"\nDONE. {n} filing rows.")
    print(f"  shares_out coverage: {n_shares}/{n} ({n_shares/n:.1%})")
    print(f"  price_at_filing coverage: {n_px}/{n} ({n_px/n:.1%})")
    print(f"  log_market_cap coverage (both): {n_both}/{n} ({n_both/n:.1%})")
    print("\nPer-ticker log_market_cap coverage:")
    cov = out.groupby("ticker").apply(
        lambda g: g["log_market_cap"].notna().mean(), include_groups=False)
    print(cov.sort_values().to_string())


if __name__ == "__main__":
    main()
