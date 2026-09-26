"""
SEC EDGAR XBRL companyconcept fetcher — quarterly income-statement flows.

Reused by probe_a (merchant generators) and probe_b (power-intensive industrials).
Free, no key. Pattern mirrors backlog_factor/data/rpo.py but adds discrete-quarter
extraction for *flow* concepts (Revenues, CostOfRevenue, GrossProfit, …), which XBRL
reports as a mix of 3-/6-/9-/12-month durations under one tag.

Not production code. Caches per-CIK JSON under ./_cache/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
_CONCEPT = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
_CACHE = Path(__file__).parent / "_cache"
_CACHE.mkdir(exist_ok=True)


def _get(url: str) -> requests.Response:
    for i in range(5):
        r = requests.get(url, headers=SEC_HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(1.5 * (i + 1))
            continue
        return r
    return r


def _raw_concept(cik: str, tag: str) -> dict | None:
    cik10 = f"{int(cik):010d}"
    cache = _CACHE / f"{cik10}_{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = _get(_CONCEPT.format(cik=cik10, tag=tag))
    if r.status_code == 404:
        cache.write_text("null")
        return None
    r.raise_for_status()
    body = r.json()
    cache.write_text(json.dumps(body))
    time.sleep(0.2)
    return body


def quarterly_flow(cik: str, tags: list[str]) -> pd.DataFrame:
    """
    Return discrete-quarter values for the first tag that yields data.

    Columns: quarter_end (Timestamp, period end), filed (Timestamp), val (float),
             q (1..4).  Q4 is derived as FY - (Q1+Q2+Q3) when only an annual
             12-month duration is tagged for the year.
    """
    for tag in tags:
        body = _raw_concept(cik, tag)
        if not body:
            continue
        rows = []
        for _unit, facts in body.get("units", {}).items():
            for f in facts:
                s, e, filed, val = f.get("start"), f.get("end"), f.get("filed"), f.get("val")
                if not (s and e and filed and val is not None):
                    continue
                s, e = pd.Timestamp(s), pd.Timestamp(e)
                ndays = (e - s).days
                rows.append((s, e, pd.Timestamp(filed), float(val), ndays, f.get("fp")))
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["start", "end", "filed", "val", "ndays", "fp"])
        # discrete quarters: ~3-month durations
        q = df[(df.ndays >= 80) & (df.ndays <= 100)].copy()
        q["quarter_end"] = q["end"]
        q = (q.sort_values("filed")
               .drop_duplicates("quarter_end", keep="first")
               .set_index("quarter_end")["val"]
               .sort_index())
        # annual (12-month) durations, for Q4 derivation
        fy = df[(df.ndays >= 350) & (df.ndays <= 380)].copy()
        fy["year"] = fy["end"].dt.year
        fy = (fy.sort_values("filed").drop_duplicates("year", keep="first")
                .set_index("year")["val"])
        derived = []
        by_year: dict[int, list] = {}
        for ts in q.index:
            by_year.setdefault(ts.year, []).append(ts)
        for year, val in fy.items():
            qs = sorted(by_year.get(year, []))
            has_q4 = any(ts.month in (12, 1) and ts.month == 12 for ts in qs) or \
                     any(abs((ts - pd.Timestamp(year, 12, 31)).days) <= 20 for ts in qs)
            if len(qs) == 3 and not has_q4:
                q4v = val - sum(q[ts] for ts in qs)
                derived.append((pd.Timestamp(year, 12, 31), q4v))
        out = q.copy()
        for ts, v in derived:
            out.loc[ts] = v
        out = out.sort_index()
        filed_map = (df[(df.ndays >= 80) & (df.ndays <= 100)]
                     .sort_values("filed").drop_duplicates("end", keep="first")
                     .set_index("end")["filed"])
        res = pd.DataFrame({"val": out})
        res["filed"] = filed_map.reindex(res.index)
        res["filed"] = res["filed"].fillna(res.index.to_series() + pd.Timedelta(days=60))
        res["q"] = res.index.quarter
        res.index.name = "quarter_end"
        return res.reset_index()
    return pd.DataFrame(columns=["quarter_end", "val", "filed", "q"])


REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"]
COGS_TAGS = ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold",
             "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
             "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization"]


def quarterly_gross_margin(cik: str) -> pd.DataFrame:
    """
    Best-effort quarterly gross profit and gross margin.
    gross_profit = GrossProfit  OR  Revenues - COGS  (broad tag fallbacks — energy
    and metals filers each use a different COGS concept).
    Columns: quarter_end, revenue, cogs, gross_profit, gross_margin, filed
    """
    rev = quarterly_flow(cik, REV_TAGS)
    gp = quarterly_flow(cik, ["GrossProfit"])
    cogs = quarterly_flow(cik, COGS_TAGS)
    out = rev.rename(columns={"val": "revenue"})[["quarter_end", "revenue", "filed"]]
    if not gp.empty:
        out = out.merge(gp.rename(columns={"val": "gross_profit"})[["quarter_end", "gross_profit"]],
                        on="quarter_end", how="outer")
    else:
        out["gross_profit"] = pd.NA
    if not cogs.empty:
        out = out.merge(cogs.rename(columns={"val": "cogs"})[["quarter_end", "cogs"]],
                        on="quarter_end", how="left")
    else:
        out["cogs"] = pd.NA
    out = out.sort_values("quarter_end").reset_index(drop=True)
    need = out["gross_profit"].isna() & out["revenue"].notna() & out["cogs"].notna()
    out.loc[need, "gross_profit"] = out.loc[need, "revenue"] - out.loc[need, "cogs"]
    out["gross_margin"] = out["gross_profit"] / out["revenue"]
    return out
