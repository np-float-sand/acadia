"""
Item 9, Approach A spike: narrative-extremity short (crowding/overhype).

Composite = z(EDGAR "data center" mention frequency, per-company, quarterly)
          + z(GDELT news-volume, per-company, daily -> resampled)
Hypothesis: high composite (heavily narrated) predicts WORSE forward returns
(overhype short thesis) on the grid_resilience 25-name universe.

Throwaway spike. Re-run to regenerate. Findings -> docs/RESEARCH-LOG.md,
docs/untried-tasks-log.md item 9.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, "/Users/sandhyapersad/acadia")
from grid_resilience.data.universe import UNIVERSE

SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}

CIK_BY_TICKER = {
    "NRG": "0001013871", "VST": "0001692819", "CNP": "0001130310", "AEP": "0000004904",
    "EXC": "0001109357", "PPL": "0000922224", "FE": "0001031296", "CEG": "0001868275",
    "TLN": "0001622536", "ETR": "0000065984", "WEC": "0000783325", "DTE": "0000936340",
    "CMS": "0000811156", "PCG": "0001004980", "EIX": "0000827052", "XEL": "0000072903",
    "ES": "0000072741", "ED": "0001047862", "D": "0000715957", "PEG": "0000788784",
    "AEE": "0001002910", "EVRG": "0001711269", "DUK": "0001326160", "SO": "0000092122",
    "NEE": "0000753308",
}

# Company name to feed GDELT (short form, better hit precision than full legal name)
GDELT_NAME = {
    "NRG": "NRG Energy", "VST": "Vistra Corp", "CNP": "CenterPoint Energy",
    "AEP": "American Electric Power", "EXC": "Exelon", "PPL": "PPL Corp",
    "FE": "FirstEnergy", "CEG": "Constellation Energy", "TLN": "Talen Energy",
    "ETR": "Entergy", "WEC": "WEC Energy", "DTE": "DTE Energy", "CMS": "CMS Energy",
    "PCG": "PG&E", "EIX": "Edison International", "XEL": "Xcel Energy",
    "ES": "Eversource Energy", "ED": "Consolidated Edison", "D": "Dominion Energy",
    "PEG": "Public Service Enterprise Group", "AEE": "Ameren", "EVRG": "Evergy",
    "DUK": "Duke Energy", "SO": "Southern Company", "NEE": "NextEra Energy",
}

QUARTERS = pd.period_range("2018Q1", "2025Q3", freq="Q")


def edgar_quarterly_counts(cik: str) -> pd.Series:
    counts = {}
    for q in QUARTERS:
        start = q.start_time.strftime("%Y-%m-%d")
        end = q.end_time.strftime("%Y-%m-%d")
        try:
            r = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": '"data center"', "forms": "8-K,10-Q,10-K",
                        "dateRange": "custom", "startdt": start, "enddt": end,
                        "ciks": cik},
                headers=SEC_HEADERS, timeout=20,
            )
            r.raise_for_status()
            counts[q.end_time.normalize()] = r.json()["hits"]["total"]["value"]
        except Exception as e:
            print(f"  [edgar] {cik} {q} failed: {e}")
            counts[q.end_time.normalize()] = np.nan
        time.sleep(0.3)
    return pd.Series(counts)


def gdelt_volume(name: str, start: str, end: str) -> pd.Series:
    try:
        r = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": name, "mode": "timelinevol", "format": "json",
                    "startdatetime": start + "000000", "enddatetime": end + "000000"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        series = data["timeline"][0]["data"]
        s = pd.Series({pd.Timestamp(d["date"][:8]): d["value"] for d in series})
        return s
    except Exception as e:
        print(f"  [gdelt] {name} failed: {e}")
        return pd.Series(dtype=float)


def main():
    tickers = list(CIK_BY_TICKER.keys())

    print("=== EDGAR quarterly 'data center' mention counts ===")
    edgar = {}
    for t in tickers:
        edgar[t] = edgar_quarterly_counts(CIK_BY_TICKER[t])
        print(f"  {t}: total mentions {int(edgar[t].sum())}")
    edgar_df = pd.DataFrame(edgar)
    edgar_df.to_parquet("research/narrative_mispricing/edgar_quarterly_mentions.parquet")

    print("\n=== GDELT daily news-volume ===")
    gdelt = {}
    for t in tickers:
        gdelt[t] = gdelt_volume(GDELT_NAME[t], "20180101", "20250930")
        print(f"  {t}: {len(gdelt[t])} days, mean intensity {gdelt[t].mean():.5f}")
        time.sleep(5.2)
    gdelt_df = pd.DataFrame(gdelt)
    gdelt_df.to_parquet("research/narrative_mispricing/gdelt_daily_volume.parquet")


if __name__ == "__main__":
    main()
