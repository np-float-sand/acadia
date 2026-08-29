from __future__ import annotations

"""Point-in-time quarterly fundamentals (revenue, gross profit) from SEC XBRL,
for the value-chain gross-margin signal. Mirrors backlog_data.py: structured
companyconcept fetch, filing-date dating, per-ticker parquet cache.

Flow concepts carry both quarterly (~90d) and annual (~365d) facts in one
response — only the quarterly facts are kept."""

import time

import pandas as pd
import requests

from grid_equipment_basket.backlog_data import CIK_BY_TICKER
from grid_equipment_basket.config import CACHE_DIR

_SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
_COLS = ["quarter_end", "availability_date", "revenue", "gross_profit"]
_REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")
_GROSS_TAGS = ("GrossProfit",)
_COST_TAGS = ("CostOfGoodsAndServicesSold", "CostOfRevenue")


def _flow_facts(cik: str, tag: str) -> pd.DataFrame:
    """Quarterly (~90-day) facts for a us-gaap flow concept, filing-dated."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    r = requests.get(url, headers=_SEC_HEADERS, timeout=30)
    if r.status_code == 404:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "val"])
    r.raise_for_status()
    rows = []
    for _unit, facts in r.json().get("units", {}).items():
        for f in facts:
            start, end, filed, val = f.get("start"), f.get("end"), f.get("filed"), f.get("val")
            if not (start and end and filed and val is not None):
                continue
            span = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if not (80 <= span <= 100):          # keep quarterly, drop annual/semi
                continue
            rows.append((pd.Timestamp(end), pd.Timestamp(filed), float(val)))
    if not rows:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "val"])
    df = pd.DataFrame(rows, columns=["quarter_end", "availability_date", "val"])
    return (df.sort_values("availability_date")
              .drop_duplicates("quarter_end", keep="first")
              .sort_values("quarter_end")
              .reset_index(drop=True))


def _first_nonempty(cik: str, tags) -> pd.DataFrame:
    for tag in tags:
        df = _flow_facts(cik, tag)
        if not df.empty:
            return df
        time.sleep(0.2)
    return pd.DataFrame(columns=["quarter_end", "availability_date", "val"])


def fetch_fundamentals(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE_DIR / f"fundamentals_{ticker}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    cik = CIK_BY_TICKER.get(ticker)
    if cik is None:
        raise KeyError(f"no CIK for {ticker}; add it to backlog_data.CIK_BY_TICKER")

    rev = _first_nonempty(cik, _REVENUE_TAGS).rename(columns={"val": "revenue"})
    gross = _flow_facts(cik, _GROSS_TAGS[0]).rename(columns={"val": "gross_profit"})
    if gross.empty:
        cost = _first_nonempty(cik, _COST_TAGS).rename(columns={"val": "cost"})
        if not cost.empty and not rev.empty:
            gross = rev.merge(cost[["quarter_end", "cost", "availability_date"]], on="quarter_end", how="inner", suffixes=("_rev", "_cost"))
            gross["gross_profit"] = gross["revenue"] - gross["cost"]
            gross["availability_date"] = gross[["availability_date_rev", "availability_date_cost"]].max(axis=1)
            gross = gross[["quarter_end", "availability_date", "gross_profit"]]

    if rev.empty or gross.empty:
        out = pd.DataFrame(columns=_COLS)
    else:
        out = rev[["quarter_end", "revenue"]].merge(gross[["quarter_end", "availability_date", "gross_profit"]], on="quarter_end", how="inner")
        out = out[_COLS].sort_values("quarter_end").reset_index(drop=True)

    if use_cache and not out.empty:
        out.to_parquet(cache)
    return out


def combine_fundamentals(by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for tkr, df in by_ticker.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["ticker"] = tkr
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=_COLS + ["ticker"])
    return pd.concat(parts, ignore_index=True).sort_values(["ticker", "quarter_end"]).reset_index(drop=True)


from grid_equipment_basket import config as _cfg


def _visible(fund_df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    asof = pd.Timestamp(asof)
    return fund_df[fund_df["availability_date"] <= asof]


def ttm_gross_margin_signal(fund_df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    out: dict[str, float] = {}
    for tkr, g in _visible(fund_df, asof).groupby("ticker"):
        g = g.sort_values("quarter_end").reset_index(drop=True)
        if len(g) < _cfg.VC_SIGNAL_MIN_QUARTERS:
            out[tkr] = float("nan")
            continue
        span = (g["quarter_end"].iloc[-1] - g["quarter_end"].iloc[-5]).days
        if not (_cfg.VC_SPAN_MIN_DAYS <= span <= _cfg.VC_SPAN_MAX_DAYS):
            out[tkr] = float("nan")
            continue
        if (asof - g["quarter_end"].iloc[-1]).days > _cfg.VC_STALENESS_MAX_DAYS:
            out[tkr] = float("nan")
            continue
        recent = g.iloc[-4:]
        prior = g.iloc[-8:-4]
        rev_recent = recent["revenue"].sum()
        rev_prior = prior["revenue"].sum()
        if not (rev_recent and rev_prior):
            out[tkr] = float("nan")
            continue
        m_recent = recent["gross_profit"].sum() / rev_recent
        m_prior = prior["gross_profit"].sum() / rev_prior
        out[tkr] = float(m_recent - m_prior)
    return pd.Series(out, dtype=float)


def ttm_revenue(fund_df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    out: dict[str, float] = {}
    for tkr, g in _visible(fund_df, asof).groupby("ticker"):
        g = g.sort_values("quarter_end").reset_index(drop=True)
        if len(g) < 4:
            out[tkr] = float("nan")
            continue
        if (asof - g["quarter_end"].iloc[-1]).days > _cfg.VC_STALENESS_MAX_DAYS:
            out[tkr] = float("nan")
            continue
        out[tkr] = float(g.iloc[-4:]["revenue"].sum())
    return pd.Series(out, dtype=float)
