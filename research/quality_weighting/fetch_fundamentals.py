"""Point-in-time fundamentals pull for the 27-name `screen` universe -- leverage,
margin stability, FCF conversion -- RESEARCH-LOG open item 13,
docs/quality-weighting-preregistration.md.

Reuses grid_equipment_basket.margin_data's XBRL companyconcept fetch/union logic
(revenue + gross-profit tags, filing-date dating) UNCHANGED for margin stability,
just re-pointed at the 27-ticker CIK map already built for the sibling
backlog-pricing-power probe (research/backlog_pricing_power/cik_map.py) instead of
the original 9-name grid_equipment_basket.backlog_data.CIK_BY_TICKER.

Adds two new concept classes margin_data.py doesn't fetch:
  - balance-sheet *instant* facts (debt, equity) for leverage
  - *annual* (350-380 day span) flow facts (CFO, capex, revenue) for FCF conversion

Throwaway research script -- not production code. Own cache dir under
research/quality_weighting/_cache/; never touches grid_equipment_basket's or
electrification_strategy's production caches.

Run: python3 research/quality_weighting/fetch_fundamentals.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))

from grid_equipment_basket import margin_data as _md  # noqa: E402
from cik_map_27 import CIK_BY_TICKER, SCREEN_27  # noqa: E402

_SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
CACHE_DIR = Path(__file__).parent / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_SLEEP = 0.2

_DEBT_TAGS = ("LongTermDebtNoncurrent", "LongTermDebtCurrent", "DebtCurrent",
              "ShortTermBorrowings", "LongTermDebt")
_EQUITY_TAGS = ("StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
_CFO_TAGS = ("NetCashProvidedByUsedInOperatingActivities",)
_CAPEX_TAGS = ("PaymentsToAcquirePropertyPlantAndEquipment",
               "PaymentsForCapitalImprovements", "PaymentsToAcquireProductiveAssets")
_REVENUE_TAGS = _md._REVENUE_TAGS


# -- generic XBRL companyconcept fetch helpers -----------------------------------
def _get_json(cik: str, tag: str) -> dict | None:
    # SEC requires the 10-digit zero-padded CIK in the URL; cik_map_27's CIK_BY_TICKER
    # (copied from research/backlog_pricing_power/cik_map.py) stores unpadded strings.
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{str(cik).zfill(10)}/us-gaap/{tag}.json"
    r = requests.get(url, headers=_SEC_HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def _instant_facts(cik: str, tag: str) -> pd.DataFrame:
    """Point-in-time balance-sheet snapshot facts (instant concepts: no start date)."""
    data = _get_json(cik, tag)
    cols = ["quarter_end", "availability_date", "val"]
    if data is None:
        return pd.DataFrame(columns=cols)
    rows = []
    for _unit, facts in data.get("units", {}).items():
        for f in facts:
            end, filed, val = f.get("end"), f.get("filed"), f.get("val")
            if end and filed and val is not None:
                rows.append((pd.Timestamp(end), pd.Timestamp(filed), float(val)))
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    return (df.sort_values("availability_date")
              .drop_duplicates("quarter_end", keep="first")
              .sort_values("quarter_end").reset_index(drop=True))


def _annual_flow_facts(cik: str, tag: str) -> pd.DataFrame:
    """Annual (350-380 day span) flow facts -- 10-K figures only, not TTM/quarterly."""
    data = _get_json(cik, tag)
    cols = ["quarter_end", "availability_date", "val"]
    if data is None:
        return pd.DataFrame(columns=cols)
    rows = []
    for _unit, facts in data.get("units", {}).items():
        for f in facts:
            start, end, filed, val = f.get("start"), f.get("end"), f.get("filed"), f.get("val")
            if not (start and end and filed and val is not None):
                continue
            span = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if not (350 <= span <= 380):
                continue
            rows.append((pd.Timestamp(end), pd.Timestamp(filed), float(val)))
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    return (df.sort_values("availability_date")
              .drop_duplicates("quarter_end", keep="first")
              .sort_values("quarter_end").reset_index(drop=True))


def _first_nonempty_annual(cik: str, tags) -> pd.DataFrame:
    for tag in tags:
        df = _annual_flow_facts(cik, tag)
        if not df.empty:
            return df
        time.sleep(_SLEEP)
    return pd.DataFrame(columns=["quarter_end", "availability_date", "val"])


def _union_annual_tags(cik: str, tags) -> pd.DataFrame:
    """Union annual facts across every tag, deduped per quarter_end with the
    earliest filing kept. Same rationale as margin_data._union_tags: a filer
    with a few stale pre-ASC-606 `Revenues` facts AND a modern
    `RevenueFromContractWithCustomerExcludingAssessedTax` series would otherwise
    stop at the stale tag under first-nonempty and never reach the modern one
    (confirmed live for ETN: `Revenues` annual has only 7 rows ending 2016-12-31;
    `RevenueFromContractWithCustomerExcludingAssessedTax` has the real 2017-2025
    data) -- first-nonempty is the wrong choice here, exactly as it would be for
    margin_data's revenue fetch."""
    parts = []
    for i, tag in enumerate(tags):
        if i:
            time.sleep(_SLEEP)
        df = _annual_flow_facts(cik, tag)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "val"])
    return (pd.concat(parts, ignore_index=True)
              .sort_values("availability_date")
              .drop_duplicates("quarter_end", keep="first")
              .sort_values("quarter_end").reset_index(drop=True))


def _sum_instant_tags(cik: str, tags) -> pd.DataFrame:
    """Sum whichever instant tags report a value at each quarter_end (a tag missing
    that date contributes 0 for that date only if >=1 other tag DOES report; a
    date where NO tag reports is NaN, not silently zero)."""
    per_tag = {}
    avail_per_tag = {}
    for i, tag in enumerate(tags):
        if i:
            time.sleep(_SLEEP)
        df = _instant_facts(cik, tag)
        if not df.empty:
            per_tag[tag] = df.set_index("quarter_end")["val"]
            avail_per_tag[tag] = df.set_index("quarter_end")["availability_date"]
    if not per_tag:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "val"])
    wide = pd.concat(per_tag, axis=1, sort=True)
    avail_wide = pd.concat(avail_per_tag, axis=1, sort=True)
    total = wide.sum(axis=1, min_count=1)
    avail = avail_wide.max(axis=1)
    out = pd.DataFrame({"val": total, "availability_date": avail}).dropna(subset=["val"])
    out = out.reset_index().rename(columns={"index": "quarter_end"})
    return out.sort_values("quarter_end").reset_index(drop=True)


# -- per-ticker fetch + cache ------------------------------------------------------
def fetch_leverage(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """quarter_end, availability_date, debt, equity, leverage (=debt/equity)."""
    cache = CACHE_DIR / f"leverage_{ticker}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    cik = CIK_BY_TICKER[ticker]
    debt = _sum_instant_tags(cik, _DEBT_TAGS).rename(columns={"val": "debt"})
    time.sleep(_SLEEP)
    equity = pd.DataFrame()
    for tag in _EQUITY_TAGS:
        equity = _instant_facts(cik, tag)
        if not equity.empty:
            break
        time.sleep(_SLEEP)
    equity = equity.rename(columns={"val": "equity"})

    cols = ["quarter_end", "availability_date", "debt", "equity", "leverage"]
    if debt.empty or equity.empty:
        out = pd.DataFrame(columns=cols)
    else:
        out = debt[["quarter_end", "debt", "availability_date"]].merge(
            equity[["quarter_end", "equity", "availability_date"]],
            on="quarter_end", suffixes=("_d", "_e"))
        out["availability_date"] = out[["availability_date_d", "availability_date_e"]].max(axis=1)
        out["leverage"] = out["debt"] / out["equity"]
        out = out[cols].sort_values("quarter_end").reset_index(drop=True)
    if use_cache:
        out.to_parquet(cache)
    return out


def fetch_fcf(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """quarter_end, availability_date, cfo, capex, revenue, fcf, fcf_conversion (annual)."""
    cache = CACHE_DIR / f"fcf_{ticker}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    cik = CIK_BY_TICKER[ticker]
    cfo = _first_nonempty_annual(cik, _CFO_TAGS).rename(columns={"val": "cfo"})
    time.sleep(_SLEEP)
    capex = _first_nonempty_annual(cik, _CAPEX_TAGS).rename(columns={"val": "capex"})
    time.sleep(_SLEEP)
    rev = _union_annual_tags(cik, _REVENUE_TAGS).rename(columns={"val": "revenue"})

    cols = ["quarter_end", "availability_date", "cfo", "capex", "revenue", "fcf", "fcf_conversion"]
    if cfo.empty or capex.empty or rev.empty:
        out = pd.DataFrame(columns=cols)
    else:
        out = (cfo[["quarter_end", "cfo", "availability_date"]]
               .merge(capex[["quarter_end", "capex", "availability_date"]],
                      on="quarter_end", suffixes=("_cfo", "_capex"))
               .merge(rev[["quarter_end", "revenue", "availability_date"]], on="quarter_end"))
        out = out.rename(columns={"availability_date": "availability_date_rev"})
        out["availability_date"] = out[
            ["availability_date_cfo", "availability_date_capex", "availability_date_rev"]
        ].max(axis=1)
        out["fcf"] = out["cfo"] - out["capex"]
        out["fcf_conversion"] = out["fcf"] / out["revenue"]
        out = out[cols].sort_values("quarter_end").reset_index(drop=True)
    if use_cache:
        out.to_parquet(cache)
    return out


def fetch_margins(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """quarter_end, availability_date, revenue, gross_profit, gross_margin (quarterly).

    Calls grid_equipment_basket.margin_data's private helpers directly (not its
    fetch_fundamentals(), which is hardwired to the 9-name CIK_BY_TICKER) -- same
    tag lists, same union/fallback logic, just a different CIK source."""
    cache = CACHE_DIR / f"margins_{ticker}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    # margin_data's _union_tags/_flow_facts build the URL directly from `cik` with no
    # internal zero-padding (unlike this module's own _get_json) -- pad here.
    cik = str(CIK_BY_TICKER[ticker]).zfill(10)
    rev = _md._union_tags(cik, _md._REVENUE_TAGS).rename(columns={"val": "revenue"})
    time.sleep(_SLEEP)
    gross = _md._flow_facts(cik, _md._GROSS_TAGS[0]).rename(columns={"val": "gross_profit"})
    if gross.empty:
        time.sleep(_SLEEP)
        cost = _md._first_nonempty(cik, _md._COST_TAGS).rename(columns={"val": "cost"})
        if not cost.empty and not rev.empty:
            gross = rev.merge(cost[["quarter_end", "cost", "availability_date"]],
                               on="quarter_end", how="inner", suffixes=("_rev", "_cost"))
            gross["gross_profit"] = gross["revenue"] - gross["cost"]
            gross["availability_date"] = gross[
                ["availability_date_rev", "availability_date_cost"]].max(axis=1)
            gross = gross[["quarter_end", "availability_date", "gross_profit"]]

    cols = ["quarter_end", "availability_date", "revenue", "gross_profit", "gross_margin"]
    if rev.empty or gross.empty:
        out = pd.DataFrame(columns=cols)
    else:
        out = rev[["quarter_end", "availability_date", "revenue"]].merge(
            gross[["quarter_end", "availability_date", "gross_profit"]],
            on="quarter_end", suffixes=("_rev", "_gross"))
        out["availability_date"] = out[["availability_date_rev", "availability_date_gross"]].max(axis=1)
        out["gross_margin"] = out["gross_profit"] / out["revenue"]
        out = out[cols].sort_values("quarter_end").reset_index(drop=True)
    if use_cache:
        out.to_parquet(cache)
    return out


def fetch_all(tickers=SCREEN_27, use_cache: bool = True):
    """Fetch (or load from cache) leverage / fcf / margins for every ticker.
    Returns three dicts keyed by ticker."""
    leverage, fcf, margins = {}, {}, {}
    for i, t in enumerate(tickers):
        print(f"[{i + 1}/{len(tickers)}] {t} ...", flush=True)
        leverage[t] = fetch_leverage(t, use_cache)
        fcf[t] = fetch_fcf(t, use_cache)
        margins[t] = fetch_margins(t, use_cache)
        time.sleep(_SLEEP)
    return leverage, fcf, margins


if __name__ == "__main__":
    lev, fcf, marg = fetch_all()
    for label, d in (("leverage", lev), ("fcf", fcf), ("margins", marg)):
        n_ok = sum(1 for df in d.values() if not df.empty)
        print(f"{label}: {n_ok}/{len(d)} tickers with data")
