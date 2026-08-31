from __future__ import annotations

"""SEC XBRL RevenueRemainingPerformanceObligation discovery + per-name fetch.
Mirrors grid_equipment_basket/backlog_data.py (instant concept, filing-dated,
per-CIK parquet cache)."""

import time

import pandas as pd
import requests

from backlog_factor.config import CACHE_DIR, IN_SCOPE_SIC_PREFIXES, SEC_HEADERS

_FRAMES = "https://data.sec.gov/api/xbrl/frames/us-gaap/RevenueRemainingPerformanceObligation/USD/{q}.json"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"


def _get(url: str) -> requests.Response:
    return requests.get(url, headers=SEC_HEADERS, timeout=30)


def discover_rpo_filers(quarters: list[str], use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE_DIR / "rpo_filers.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    ciks: set[int] = set()
    for q in quarters:
        r = _get(_FRAMES.format(q=q))
        if r.status_code == 404:
            continue
        r.raise_for_status()
        for row in r.json().get("data", []):
            if row.get("cik") is not None:
                ciks.add(int(row["cik"]))
        time.sleep(0.2)

    rows = []
    for cik in sorted(ciks):
        cik10 = f"{cik:010d}"
        r = _get(_SUBMISSIONS.format(cik=cik10))
        if r.status_code != 200:
            continue
        body = r.json()
        sic = str(body.get("sic") or "")
        if not sic.startswith(IN_SCOPE_SIC_PREFIXES):
            continue
        tickers = body.get("tickers") or []
        rows.append({
            "cik": cik10,
            "ticker": tickers[0] if tickers else "",
            "sic": sic,
            "sic_description": body.get("sicDescription", ""),
            "company": body.get("name", ""),
        })
        time.sleep(0.15)

    df = pd.DataFrame(rows, columns=["cik", "ticker", "sic", "sic_description", "company"])
    df = df[df["ticker"] != ""].sort_values("ticker").reset_index(drop=True)
    if use_cache and not df.empty:
        df.to_parquet(cache)
    return df


_CONCEPT = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"


def _companyconcept(cik: str, tag: str) -> pd.DataFrame:
    r = _get(_CONCEPT.format(cik=cik, tag=tag))
    if r.status_code == 404:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "metric_value"])
    r.raise_for_status()
    rows = []
    for _unit, facts in r.json().get("units", {}).items():
        for f in facts:
            end, filed, val = f.get("end"), f.get("filed"), f.get("val")
            if end and filed and val is not None:
                rows.append((pd.Timestamp(end), pd.Timestamp(filed), float(val)))
    if not rows:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "metric_value"])
    df = pd.DataFrame(rows, columns=["quarter_end", "availability_date", "metric_value"])
    return (df.sort_values("availability_date")
              .drop_duplicates("quarter_end", keep="first")
              .sort_values("quarter_end")
              .reset_index(drop=True))


def fetch_rpo(ticker: str, cik: str, use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE_DIR / f"rpo_{ticker}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    df = _companyconcept(cik, "RevenueRemainingPerformanceObligation")
    if df.empty:
        cur = _companyconcept(cik, "RevenueRemainingPerformanceObligationCurrent")
        non = _companyconcept(cik, "RevenueRemainingPerformanceObligationNoncurrent")
        if not cur.empty and not non.empty:
            m = cur.merge(non, on="quarter_end", suffixes=("_c", "_n"))
            m["availability_date"] = m[["availability_date_c", "availability_date_n"]].max(axis=1)
            m["metric_value"] = m["metric_value_c"] + m["metric_value_n"]
            df = m[["quarter_end", "availability_date", "metric_value"]].sort_values("quarter_end")
        time.sleep(0.2)
    if use_cache and not df.empty:
        df.to_parquet(cache)
    return df


def load_rpo_panel(universe: list[dict], use_cache: bool = True) -> pd.DataFrame:
    parts = []
    for row in universe:
        d = fetch_rpo(row["ticker"], row["cik"], use_cache=use_cache)
        if d.empty:
            continue
        d = d.copy()
        d["ticker"] = row["ticker"]
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["ticker", "quarter_end", "availability_date", "metric_value"])
    return (pd.concat(parts, ignore_index=True)
              .sort_values(["ticker", "quarter_end"]).reset_index(drop=True))
