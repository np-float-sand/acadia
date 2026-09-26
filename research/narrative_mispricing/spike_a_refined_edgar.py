"""
Approach A refinement, fix 1: broader EDGAR keyword set beyond "data center" alone --
"hyperscaler" and "large load" (the terms merchant generators actually use / PJM's own
established term per the thematic-catalyst-detector precedent), to fix VST's
under-narration problem (it says "data center" almost never but is a major
hyperscaler-PPA / large-load counterparty).

Reuses the existing edgar_quarterly_mentions.parquet ("data center" counts) and ADDS
these two keywords as new columns per ticker, summed later into a combined score.
"""
import sys
import time

import numpy as np
import pandas as pd
import requests

CIK_BY_TICKER = {
    "NRG": "0001013871", "VST": "0001692819", "CNP": "0001130310", "AEP": "0000004904",
    "EXC": "0001109357", "PPL": "0000922224", "FE": "0001031296", "CEG": "0001868275",
    "TLN": "0001622536", "ETR": "0000065984", "WEC": "0000783325", "DTE": "0000936340",
    "CMS": "0000811156", "PCG": "0001004980", "EIX": "0000827052", "XEL": "0000072903",
    "ES": "0000072741", "ED": "0001047862", "D": "0000715957", "PEG": "0000788784",
    "AEE": "0001002910", "EVRG": "0001711269", "DUK": "0001326160", "SO": "0000092122",
    "NEE": "0000753308",
}
SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
QUARTERS = pd.period_range("2018Q1", "2025Q3", freq="Q")
KEYWORDS = ["hyperscaler", "large load"]


def edgar_quarterly_counts(cik: str, keyword: str) -> pd.Series:
    counts = {}
    for q in QUARTERS:
        start = q.start_time.strftime("%Y-%m-%d")
        end = q.end_time.strftime("%Y-%m-%d")
        try:
            r = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": f'"{keyword}"', "forms": "8-K,10-Q,10-K",
                        "dateRange": "custom", "startdt": start, "enddt": end,
                        "ciks": cik},
                headers=SEC_HEADERS, timeout=20,
            )
            r.raise_for_status()
            counts[q.end_time.normalize()] = r.json()["hits"]["total"]["value"]
        except Exception as e:
            print(f"  [edgar] {cik} {keyword} {q} failed: {e}")
            counts[q.end_time.normalize()] = np.nan
        time.sleep(0.25)
    return pd.Series(counts)


def main():
    tickers = list(CIK_BY_TICKER.keys())
    for kw in KEYWORDS:
        print(f"=== keyword: {kw} ===")
        result = {}
        for t in tickers:
            result[t] = edgar_quarterly_counts(CIK_BY_TICKER[t], kw)
            print(f"  {t}: total {int(result[t].sum())}")
        df = pd.DataFrame(result)
        safe_kw = kw.replace(" ", "_")
        df.to_parquet(f"research/narrative_mispricing/edgar_quarterly_{safe_kw}.parquet")


if __name__ == "__main__":
    main()
