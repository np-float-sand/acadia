from __future__ import annotations

"""Backlog / book-to-bill collection for the Step 2 weight tilt.

Structured-first: SEC XBRL companyconcept RevenueRemainingPerformanceObligation.
Everything else is hand-collected into data/backlog_quarterly.csv with an explicit
disclosure_type per row — definitions are NOT coerced to a common metric (spec §6).
"""

import io
import time

import pandas as pd
import requests

from grid_equipment_basket.config import CACHE_DIR

_SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
_VALID_DISCLOSURE = {
    "xbrl_rpo", "nongaap_backlog_total", "nongaap_backlog_segment", "book_to_bill_only",
}
_CSV_PATH = CACHE_DIR.parent / "backlog_quarterly.csv"

# Resolved once from https://www.sec.gov/files/company_tickers.json (Task 6 Step 4).
CIK_BY_TICKER: dict[str, str] = {
    "ETN": "0001551182",   # Eaton Corp plc
    "HUBB": "0000048898",  # HUBBELL INC
    "GEV": "0001996810",   # GE Vernova Inc.
    "VRT": "0001674101",   # Vertiv Holdings Co
    "PWR": "0001050915",   # QUANTA SERVICES, INC.
    "MYRG": "0000700923",  # MYR GROUP INC.
    "NVT": "0001720635",   # nVent Electric plc
    "FLNC": "0001868941",  # Fluence Energy, Inc.
    "PRIM": "0001361538",  # Primoris Services Corp
}


def _companyconcept(cik: str, tag: str) -> pd.DataFrame:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    r = requests.get(url, headers=_SEC_HEADERS, timeout=30)
    if r.status_code == 404:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "metric_value"])
    r.raise_for_status()
    units = r.json().get("units", {})
    rows = []
    for _unit, facts in units.items():
        for f in facts:
            end = f.get("end")
            filed = f.get("filed")
            val = f.get("val")
            if end and filed and val is not None:
                rows.append((end, filed, float(val)))
    if not rows:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "metric_value"])
    df = pd.DataFrame(rows, columns=["quarter_end", "availability_date", "metric_value"])
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    df["availability_date"] = pd.to_datetime(df["availability_date"])
    df = (df.sort_values("availability_date")
            .drop_duplicates("quarter_end", keep="first")
            .sort_values("quarter_end")
            .reset_index(drop=True))
    return df


def fetch_rpo(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE_DIR / f"rpo_{ticker}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    cik = CIK_BY_TICKER.get(ticker)
    if cik is None:
        raise KeyError(f"no CIK for {ticker}; add it to CIK_BY_TICKER")
    df = _companyconcept(cik, "RevenueRemainingPerformanceObligation")
    if df.empty:
        cur = _companyconcept(cik, "RevenueRemainingPerformanceObligationCurrent")
        non = _companyconcept(cik, "RevenueRemainingPerformanceObligationNoncurrent")
        if not cur.empty and not non.empty:
            df = cur.merge(non, on="quarter_end", suffixes=("_c", "_n"))
            df["availability_date"] = df[["availability_date_c", "availability_date_n"]].max(axis=1)
            df["metric_value"] = df["metric_value_c"] + df["metric_value_n"]
            df = df[["quarter_end", "availability_date", "metric_value"]]
        time.sleep(0.2)
    if use_cache and not df.empty:
        df.to_parquet(cache)
    return df


def load_backlog_csv(path: str | None = None) -> pd.DataFrame:
    p = path or _CSV_PATH
    df = pd.read_csv(p, dtype={"ticker": str})
    bad = set(df["disclosure_type"]) - _VALID_DISCLOSURE
    if bad:
        raise ValueError(f"invalid disclosure_type(s): {sorted(bad)}")
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    df["availability_date"] = pd.to_datetime(df["availability_date"])
    if df.duplicated(["ticker", "quarter_end"]).any():
        raise ValueError("duplicate (ticker, quarter_end) rows in backlog_quarterly.csv")
    return df.sort_values(["ticker", "quarter_end"]).reset_index(drop=True)


def backlog_growth_signal(df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    vis = df[df["availability_date"] <= asof]
    out: dict[str, float] = {}
    for tkr, g in vis.groupby("ticker"):
        g = g.sort_values("quarter_end")
        b2b = g[g["disclosure_type"] == "book_to_bill_only"]
        if len(b2b) == len(g):
            # Fire only when the ticker discloses *exclusively* book-to-bill.
            # A mixed ticker falls through to the non-B2B growth logic below.
            out[tkr] = float(b2b.iloc[-1]["metric_value"]) - 1.0
            continue
        if len(g) < 5:
            out[tkr] = float("nan")
            continue
        latest_row = g.iloc[-1]
        year_ago_row = g.iloc[-5]
        span_days = (latest_row["quarter_end"] - year_ago_row["quarter_end"]).days
        if not (300 <= span_days <= 430):
            # Positional iloc[-5] only lands a true year ago on a clean, gap-free
            # quarterly series. A gappy series (e.g. VRT) would inflate the "YoY"
            # ratio here — NaN it instead of ranking it for the wrong reason.
            out[tkr] = float("nan")
            continue
        if (pd.Timestamp(asof) - latest_row["quarter_end"]).days > 200:
            # Latest disclosed quarter is too old to still describe the name —
            # a stopped-disclosing series would otherwise be ranked on stale data.
            out[tkr] = float("nan")
            continue
        latest = float(latest_row["metric_value"])
        year_ago = float(year_ago_row["metric_value"])
        out[tkr] = (latest / year_ago - 1.0) if year_ago else float("nan")
    return pd.Series(out, dtype=float)


def backlog_ranks(signal: pd.Series) -> pd.Series:
    return signal.rank(method="dense", ascending=True)
