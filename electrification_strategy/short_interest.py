"""Short-interest snapshot cache -- a cost/liquidity diagnostic for the hedge
short leg (config.HEDGE_SHORT_TICKERS), not a signal input.

yfinance exposes only a live snapshot (no historical short-interest endpoint),
so this accumulates history from the first pipeline run forward. See
docs/technical-reference.md and the known-limitations section of the Phase 2
findings report.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from electrification_strategy import config


def _fetch_one(ticker: str) -> dict:
    return yf.Ticker(ticker).info


def fetch_short_interest(tickers: list[str], use_cache: bool = True) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    rows = []
    for t in tickers:
        info = _fetch_one(t)
        rows.append({
            "ticker": t,
            "shares_short": info.get("sharesShort"),
            "short_ratio": info.get("shortRatio"),
            "short_pct_float": info.get("shortPercentOfFloat"),
            "date_short_interest": pd.to_datetime(info.get("dateShortInterest"), unit="s", utc=True)
                if info.get("dateShortInterest") else pd.NaT,
            "fetched_at": now,
        })
    snapshot = pd.DataFrame(rows)

    if use_cache:
        cache_path = config.CACHE_DIR / config.SHORT_INTEREST_CACHE_FILE
        if cache_path.exists():
            existing = pd.read_parquet(cache_path)
            combined = pd.concat([existing, snapshot], ignore_index=True)
            combined = combined.drop_duplicates(subset=["ticker", "date_short_interest"], keep="last")
        else:
            combined = snapshot
        combined.to_parquet(cache_path, index=False)

    return snapshot.set_index("ticker")


def crowding_diagnostic(history: pd.DataFrame, tickers: tuple[str, ...] = config.HEDGE_SHORT_TICKERS) -> pd.DataFrame:
    latest = (history[history["ticker"].isin(tickers)]
              .sort_values("fetched_at")
              .drop_duplicates(subset="ticker", keep="last")
              .set_index("ticker"))
    return latest.sort_values("short_pct_float", ascending=False)
