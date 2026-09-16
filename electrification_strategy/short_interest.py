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


COLUMNS = ["ticker", "shares_short", "short_ratio", "short_pct_float",
           "date_short_interest", "fetched_at", "data_quality_flag"]

#: Below this many shares short, a reported 0%/missing short-%-of-float is treated as a
#: stale or missing yfinance field rather than a genuine "nobody is short this name".
MIN_PLAUSIBLE_SHARES_SHORT = 100_000


def _fetch_one(ticker: str) -> dict:
    return yf.Ticker(ticker).info


def _implausible(shares_short, short_pct_float) -> bool:
    """True when a snapshot row carries the signature of a stale/missing yfinance field.

    The pattern seen in production (DLR, 2026-08): `shortPercentOfFloat` reported as 0 (or
    absent) *together with* a near-zero `sharesShort`. For a large, liquid, widely-held name
    that combination is not a real market fact -- some shares are always short -- so it must
    not be presented as a clean zero. A missing % of float on its own (with a plausible share
    count) is NOT flagged; neither is a genuinely small but non-trivial short interest.
    """
    pct_missing = short_pct_float is None or pd.isna(short_pct_float) or float(short_pct_float) == 0.0
    if not pct_missing:
        return False
    if shares_short is None or pd.isna(shares_short):
        return True
    return float(shares_short) < MIN_PLAUSIBLE_SHARES_SHORT


def _empty_snapshot() -> pd.DataFrame:
    """Correctly-shaped empty frame, for the no-tickers case (mirrors
    risk_exposures.load_risk_exposures's empty-file contract)."""
    return pd.DataFrame(columns=COLUMNS).set_index("ticker")


def fetch_short_interest(tickers: list[str], use_cache: bool = True) -> pd.DataFrame:
    """Fetch a live short-interest snapshot per ticker and append it to the on-disk cache."""
    if not list(tickers):
        return _empty_snapshot()

    now = datetime.now(timezone.utc)
    rows = []
    for t in tickers:
        info = _fetch_one(t)
        shares_short = info.get("sharesShort")
        short_pct_float = info.get("shortPercentOfFloat")
        rows.append({
            "ticker": t,
            "shares_short": shares_short,
            "short_ratio": info.get("shortRatio"),
            "short_pct_float": short_pct_float,
            "date_short_interest": pd.to_datetime(info.get("dateShortInterest"), unit="s", utc=True)
                if info.get("dateShortInterest") else pd.NaT,
            "fetched_at": now,
            "data_quality_flag": _implausible(shares_short, short_pct_float),
        })
    snapshot = pd.DataFrame(rows)[COLUMNS]

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
    """Return each hedge short leg's latest short-interest-as-%-of-float, ranked most crowded first.

    Carries `data_quality_flag` through from the snapshot; a history cached before that column
    existed has it recomputed here, so a stale/implausible row is never reported as a clean zero.
    """
    latest = (history[history["ticker"].isin(tickers)]
              .sort_values("fetched_at")
              .drop_duplicates(subset="ticker", keep="last")
              .set_index("ticker"))
    if "data_quality_flag" not in latest.columns:
        latest["data_quality_flag"] = [
            _implausible(s, p) for s, p in zip(latest["shares_short"], latest["short_pct_float"])
        ]
    return latest.sort_values("short_pct_float", ascending=False)
